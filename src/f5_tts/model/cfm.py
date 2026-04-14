"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device

    def _masked_mse(self, student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        loss = F.mse_loss(student, teacher, reduction="none")
        if mask is None:
            return loss.mean()

        valid = mask.unsqueeze(-1).expand_as(loss)
        loss = loss.masked_select(valid)
        if loss.numel() == 0:
            return student.new_zeros(())
        return loss.mean()

    def _resolve_hidden_distill_layers(self, distill_config: dict | None = None) -> tuple[int, ...]:
        distill_config = default(distill_config, {})
        hidden_layers = distill_config.get("hidden_layers")
        if exists(hidden_layers):
            hidden_layers = tuple(sorted({int(layer) for layer in hidden_layers}))
        elif hasattr(self.transformer, "get_default_hidden_distill_layers"):
            hidden_layers = self.transformer.get_default_hidden_distill_layers()
        elif hasattr(self.transformer, "depth"):
            hidden_layers = (int(self.transformer.depth) - 1,)
        else:
            hidden_layers = tuple()
        return hidden_layers

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=65536,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow (cond)
            if cfg_strength < 1e-5:
                pred = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
                return pred

            # predict flow (cond and uncond), for classifier-free guidance
            pred_cfg = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                cfg_infer=True,
                cache=True,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        noise_scheduler: str | None = None,
        teacher_model: CFM | None = None,
        distill_config: dict | None = None,
        return_metadata: bool = False,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):  # if lens not acquired by trainer from collate_fn
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        distill_config = default(distill_config, {})
        distill_enabled = teacher_model is not None and distill_config.get("enabled", True)
        hidden_distill_layers = self._resolve_hidden_distill_layers(distill_config) if distill_enabled else tuple()
        capture_hidden_states = distill_enabled and len(hidden_distill_layers) > 0

        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        if capture_hidden_states:
            pred, student_hidden_states = self.transformer(
                x=φ,
                cond=cond,
                text=text,
                time=time,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                mask=mask,
                capture_hidden_layers=hidden_distill_layers,
                return_hidden_states=True,
            )
        else:
            pred = self.transformer(
                x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask
            )
            student_hidden_states = {}

        # flow matching loss
        flow_matching_loss = F.mse_loss(pred, flow, reduction="none")
        flow_matching_loss = flow_matching_loss[rand_span_mask].mean()

        hidden_distill_loss = pred.new_zeros(())
        output_distill_loss = pred.new_zeros(())
        hidden_distill_by_layer = {}
        if distill_enabled:
            with torch.no_grad():
                if capture_hidden_states:
                    teacher_pred, teacher_hidden_states = teacher_model.transformer(
                        x=φ,
                        cond=cond,
                        text=text,
                        time=time,
                        drop_audio_cond=drop_audio_cond,
                        drop_text=drop_text,
                        mask=mask,
                        capture_hidden_layers=hidden_distill_layers,
                        return_hidden_states=True,
                    )
                else:
                    teacher_pred = teacher_model.transformer(
                        x=φ,
                        cond=cond,
                        text=text,
                        time=time,
                        drop_audio_cond=drop_audio_cond,
                        drop_text=drop_text,
                        mask=mask,
                    )
                    teacher_hidden_states = {}

            if hidden_distill_layers:
                layer_losses = []
                for layer_id in hidden_distill_layers:
                    layer_loss = self._masked_mse(student_hidden_states[layer_id], teacher_hidden_states[layer_id], mask)
                    hidden_distill_by_layer[layer_id] = float(layer_loss.detach().item())
                    layer_losses.append(layer_loss)
                hidden_distill_loss = torch.stack(layer_losses).mean()

            output_distill_loss = self._masked_mse(pred, teacher_pred, mask)

        total_loss = (
            flow_matching_loss
            + float(distill_config.get("hidden_weight", 0.02)) * hidden_distill_loss
            + float(distill_config.get("output_weight", 0.05)) * output_distill_loss
        )

        if return_metadata:
            return total_loss, cond, pred, {
                "flow_matching_loss": float(flow_matching_loss.detach().item()),
                "hidden_distill_loss": float(hidden_distill_loss.detach().item()),
                "output_distill_loss": float(output_distill_loss.detach().item()),
                "distill_hidden_layers": list(hidden_distill_layers),
                "hidden_distill_by_layer": hidden_distill_by_layer,
            }

        return total_loss, cond, pred

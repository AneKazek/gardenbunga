from __future__ import annotations

import gc
import math
import os

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
from f5_tts.model.utils import default, exists, get_allowed_missing_state_dict_prefixes, load_state_dict_with_allowed_missing


# trainer


class Trainer:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        weight_decay=0.01,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        wandb_project="test_f5-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        last_per_updates=None,
        mixed_precision: str | None = "auto",
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
        is_local_vocoder: bool = False,  # use local path vocoder
        local_vocoder_path: str = "",  # local vocoder path
        teacher_model: CFM | None = None,
        teacher_checkpoint_path: str | None = None,
        student_init_checkpoint_path: str | None = None,
        teacher_use_ema: bool = True,
        student_init_use_ema: bool = True,
        distill_config: dict = dict(),
        model_cfg_dict: dict = dict(),  # training config
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerate_kwargs = dict(accelerate_kwargs)
        mixed_precision = self._resolve_mixed_precision(default(accelerate_kwargs.get("mixed_precision"), mixed_precision))
        accelerate_kwargs["mixed_precision"] = mixed_precision

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            if not model_cfg_dict:
                model_cfg_dict = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "noise_scheduler": noise_scheduler,
                    "bnb_optimizer": bnb_optimizer,
                    "mixed_precision": mixed_precision,
                }
            model_cfg_dict["gpus"] = self.accelerator.num_processes
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=model_cfg_dict,
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = None
            if self.accelerator.is_main_process:
                self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

            print(f"Using logger: {logger}")
            if grad_accumulation_steps > 1:
                print(
                    "Gradient accumulation checkpointing with per_updates now, old logic per_steps used with before f992c4e"
                )

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_f5-tts")

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor
        self.teacher_model = teacher_model
        self.teacher_checkpoint_path = teacher_checkpoint_path
        self.student_init_checkpoint_path = default(student_init_checkpoint_path, teacher_checkpoint_path)
        self.teacher_use_ema = teacher_use_ema
        self.student_init_use_ema = student_init_use_ema
        self.distill_config = dict(default(distill_config, {}))
        self.mixed_precision = mixed_precision

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        if self.teacher_model is not None:
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()
            self.teacher_model.to(self.accelerator.device)
            teacher_checkpoint_path = default(self.teacher_checkpoint_path, self.student_init_checkpoint_path)
            if not exists(teacher_checkpoint_path):
                raise ValueError("teacher_checkpoint_path or student_init_checkpoint_path must be set when distillation is enabled.")
            self._load_model_from_path(
                self.teacher_model,
                teacher_checkpoint_path,
                use_ema=self.teacher_use_ema,
                description="teacher",
            )

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def _resolve_mixed_precision(self, mixed_precision: str | None) -> str:
        if mixed_precision in [None, "auto"]:
            if torch.cuda.is_available():
                return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
            return "no"
        return mixed_precision

    def _resolve_checkpoint_reference(self, checkpoint_path: str) -> str:
        if "://" in checkpoint_path:
            from cached_path import cached_path

            return str(cached_path(checkpoint_path))
        return checkpoint_path

    def _load_checkpoint_file(self, checkpoint_path: str):
        checkpoint_path = self._resolve_checkpoint_reference(checkpoint_path)
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            checkpoint = load_file(checkpoint_path, device="cpu")
        else:
            checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        return checkpoint, checkpoint_path

    def _extract_model_state_dict(self, checkpoint: dict[str, torch.Tensor], use_ema: bool) -> dict[str, torch.Tensor]:
        if "ema_model_state_dict" in checkpoint and use_ema:
            state_dict = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in state_dict:
                del state_dict[key]

        return state_dict

    def _load_model_from_path(
        self,
        model: torch.nn.Module,
        checkpoint_path: str,
        *,
        use_ema: bool,
        description: str,
        extra_allowed_missing_prefixes: tuple[str, ...] = tuple(),
    ) -> None:
        checkpoint, resolved_path = self._load_checkpoint_file(checkpoint_path)
        state_dict = self._extract_model_state_dict(checkpoint, use_ema=use_ema)
        load_state_dict_with_allowed_missing(
            model,
            state_dict,
            extra_allowed_missing_prefixes=extra_allowed_missing_prefixes,
        )
        if self.is_main:
            print(f"Loaded {description} checkpoint from {resolved_path}")
        del checkpoint
        gc.collect()

    def _find_latest_checkpoint(self) -> str | None:
        if not exists(self.checkpoint_path) or not os.path.exists(self.checkpoint_path):
            return None

        checkpoint_files = [f for f in os.listdir(self.checkpoint_path) if f.endswith((".pt", ".safetensors"))]
        if not checkpoint_files:
            return None

        if "model_last.pt" in checkpoint_files:
            return os.path.join(self.checkpoint_path, "model_last.pt")

        all_checkpoints = [
            f for f in checkpoint_files if (f.startswith("model_") or f.startswith("pretrained_"))
        ]
        training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
        if training_checkpoints:
            latest_checkpoint = sorted(training_checkpoints, key=lambda x: int("".join(filter(str.isdigit, x))))[-1]
            return os.path.join(self.checkpoint_path, latest_checkpoint)

        pretrained_checkpoints = [f for f in all_checkpoints if f.startswith("pretrained_")]
        if pretrained_checkpoints:
            return os.path.join(self.checkpoint_path, sorted(pretrained_checkpoints)[-1])

        return None

    def _collect_mamba_metrics(self) -> dict[str, float]:
        metrics = {}
        transformer = self.accelerator.unwrap_model(self.model).transformer
        for block_id in getattr(transformer, "mamba_block_ids", tuple()):
            block = transformer.transformer_blocks[block_id]
            if block.last_mamba_metrics is None:
                continue

            prefix = f"mamba/block_{block_id}"
            metrics[f"{prefix}/alpha"] = block.last_mamba_metrics["alpha"]
            metrics[f"{prefix}/output_scale"] = block.last_mamba_metrics["output_scale"]
            metrics[f"{prefix}/norm_ratio"] = block.last_mamba_metrics["norm_ratio"]
            metrics[f"{prefix}/executed"] = block.last_mamba_metrics["executed"]

            grad_norm = 0.0
            for name, param in block.mamba_attn.named_parameters():
                if name == "output_scale":
                    continue
                if param.grad is not None:
                    grad_norm = float(param.grad.detach().float().norm().item())
                break
            metrics[f"{prefix}/grad_norm"] = grad_norm

        return metrics

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
                if self.keep_last_n_checkpoints > 0:
                    # Updated logic to exclude pretrained model from rotation
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_")
                        and not f.startswith("pretrained_")  # Exclude pretrained models
                        and f.endswith(".pt")
                        and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        print(f"Removed old checkpoint: {oldest_checkpoint}")

    def load_checkpoint(self):
        latest_checkpoint = self._find_latest_checkpoint()
        if latest_checkpoint is None:
            if exists(self.student_init_checkpoint_path):
                self._load_model_from_path(
                    self.accelerator.unwrap_model(self.model),
                    self.student_init_checkpoint_path,
                    use_ema=self.student_init_use_ema,
                    description="student init",
                )
                if self.is_main:
                    self.ema_model.copy_params_from_model_to_ema()
            return 0

        self.accelerator.wait_for_everyone()
        checkpoint, resolved_path = self._load_checkpoint_file(latest_checkpoint)
        if self.is_main:
            print(f"Loading training state from {resolved_path}")

        model_allowed_missing = get_allowed_missing_state_dict_prefixes(self.accelerator.unwrap_model(self.model))
        ema_allowed_missing = model_allowed_missing + tuple(f"ema_model.{prefix}" for prefix in model_allowed_missing)

        is_training_checkpoint = "update" in checkpoint or "step" in checkpoint

        if is_training_checkpoint:
            for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["ema_model_state_dict"]:
                    del checkpoint["ema_model_state_dict"][key]

            if self.is_main:
                load_state_dict_with_allowed_missing(
                    self.ema_model,
                    checkpoint["ema_model_state_dict"],
                    extra_allowed_missing_prefixes=ema_allowed_missing,
                )

            # patch for backward compatibility, with before f992c4e
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    print(
                        "F5-TTS WARNING: Loading checkpoint saved with per_steps logic (before f992c4e), will convert to per_updates according to grad_accumulation_steps setting, may have unexpected behaviour."
                    )
            # patch for backward compatibility, 305e3ea
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            load_state_dict_with_allowed_missing(
                self.accelerator.unwrap_model(self.model),
                checkpoint["model_state_dict"],
            )
            optimizer_state_loaded = True
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if self.scheduler:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except (ValueError, RuntimeError) as exc:
                optimizer_state_loaded = False
                if self.is_main:
                    print(
                        "F5-TTS WARNING: Optimizer/scheduler state is incompatible with the current model "
                        f"(likely due to architecture changes such as new Mamba blocks): {exc}"
                    )
                    print(
                        "F5-TTS WARNING: Model weights were loaded, but optimizer and scheduler will start fresh "
                        "from update 0."
                    )
            update = checkpoint["update"] if optimizer_state_loaded else 0
        else:
            load_state_dict_with_allowed_missing(
                self.accelerator.unwrap_model(self.model),
                self._extract_model_state_dict(checkpoint, use_ema=self.student_init_use_ema),
            )
            if self.is_main:
                self.ema_model.copy_params_from_model_to_ema()
            update = 0

        del checkpoint
        gc.collect()
        return update

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            )
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,  # This enables reproducible shuffling
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual multi_gpu updates = single_gpu updates / gpu nums
        start_update = self.load_checkpoint()
        global_update = start_update

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if self.teacher_model is not None:
                self.teacher_model.eval()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]
                    step_metrics = {}

                    # TODO. add duration predictor training
                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_update)

                    with self.accelerator.autocast():
                        loss, cond, pred, loss_breakdown = self.model(
                            mel_spec,
                            text=text_inputs,
                            lens=mel_lengths,
                            noise_scheduler=self.noise_scheduler,
                            teacher_model=self.teacher_model,
                            distill_config=self.distill_config,
                            return_metadata=True,
                        )
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        step_metrics.update(
                            {
                                "loss": float(loss.detach().item()),
                                "flow_matching_loss": loss_breakdown["flow_matching_loss"],
                                "hidden_distill_loss": loss_breakdown["hidden_distill_loss"],
                                "output_distill_loss": loss_breakdown["output_distill_loss"],
                            }
                        )
                        for layer_id, layer_loss in loss_breakdown["hidden_distill_by_layer"].items():
                            step_metrics[f"hidden_distill/layer_{layer_id}"] = layer_loss
                        step_metrics.update(self._collect_mamba_metrics())

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=loss.item())

                if self.accelerator.is_local_main_process and self.accelerator.sync_gradients:
                    step_metrics["lr"] = self.scheduler.get_last_lr()[0]
                    self.accelerator.log(step_metrics, step=global_update)
                if self.logger == "tensorboard" and self.accelerator.is_main_process and self.accelerator.sync_gradients:
                    for key, value in step_metrics.items():
                        self.writer.add_scalar(key, value, global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:
                        ref_audio_len = mel_lengths[0]
                        infer_text = [
                            text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
                        ]
                        with torch.inference_mode(), self.accelerator.autocast():
                            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                text=infer_text,
                                duration=ref_audio_len * 2,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                            )
                            generated = generated.to(torch.float32)
                            gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                            ref_mel_spec = batch["mel"][0, :, :ref_audio_len].unsqueeze(0)
                            if self.vocoder_name == "vocos":
                                gen_audio = vocoder.decode(gen_mel_spec).cpu()
                                ref_audio = vocoder.decode(ref_mel_spec).cpu()
                            elif self.vocoder_name == "bigvgan":
                                gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                                ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()

                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        )
                        self.model.train()

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()

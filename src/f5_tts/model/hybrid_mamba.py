from __future__ import annotations

import torch
from torch import nn


def _apply_sequence_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x
    return x.masked_fill(~mask.unsqueeze(-1), 0.0)


def reverse_by_length(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    for batch_idx, length in enumerate(lengths.tolist()):
        if length > 1:
            out[batch_idx, :length] = torch.flip(x[batch_idx, :length], dims=[0])
        if length < x.shape[1]:
            out[batch_idx, length:] = 0.0
    return out


class ConservativeMambaMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 1,
        bidirectional: bool = True,
        output_scale_init: float = 1e-3,
    ):
        super().__init__()

        try:
            from mamba_ssm import Mamba2
        except ImportError as exc:
            raise ImportError(
                "Hybrid Mamba blocks require `mamba-ssm==2.3.1` and `causal-conv1d>=1.4.0`."
            ) from exc

        self.bidirectional = bidirectional
        self.fwd = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand) if bidirectional else None

        self.output_scale = nn.Parameter(torch.full((1,), float(output_scale_init)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = _apply_sequence_mask(x, mask)
        y = self.fwd(x)

        if self.bwd is not None:
            if mask is None:
                x_reverse = torch.flip(x, dims=[1])
                y_reverse = torch.flip(self.bwd(x_reverse), dims=[1])
            else:
                lengths = mask.sum(dim=1).to(dtype=torch.long)
                x_reverse = reverse_by_length(x, lengths)
                y_reverse = reverse_by_length(self.bwd(x_reverse), lengths)
            y = (y + y_reverse) * 0.5

        y = self.output_scale * y
        return _apply_sequence_mask(y, mask)

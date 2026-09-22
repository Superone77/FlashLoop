"""Loop-boundary token and decode-key selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DecodeSelection:
    """Sorted selected key positions and their original global-softmax mass."""

    indices: torch.Tensor
    global_mass: torch.Tensor


def _validate_fraction(fraction: float) -> float:
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    return fraction


def select_prefill_tokens(
    scores: torch.Tensor,
    *,
    fraction: float,
    eligible: torch.Tensor | None = None,
    force_last_token: bool = True,
) -> torch.Tensor:
    """Select a global ceil budget, optionally nested in a prior-loop mask."""

    fraction = _validate_fraction(fraction)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("scores must have shape [batch, tokens]")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite")
    if eligible is None:
        eligible = torch.ones_like(scores, dtype=torch.bool)
    if eligible.shape != scores.shape or eligible.dtype != torch.bool:
        raise ValueError("eligible must be a bool tensor matching scores")

    _batch, tokens = scores.shape
    budget = max(1, math.ceil(tokens * fraction))
    result = torch.zeros_like(eligible)
    for batch_index in range(scores.shape[0]):
        candidates = torch.nonzero(eligible[batch_index], as_tuple=False).flatten()
        if candidates.numel() == 0:
            raise ValueError("eligible contains an empty batch row")
        forced = torch.empty(0, dtype=torch.long, device=scores.device)
        if force_last_token:
            last = tokens - 1
            if not bool(eligible[batch_index, last]):
                raise ValueError("the forced final token must be eligible")
            forced = torch.tensor([last], dtype=torch.long, device=scores.device)
            candidates = candidates[candidates != last]
        count = min(budget, int(candidates.numel()) + int(forced.numel()))
        remaining = count - int(forced.numel())
        if remaining > 0:
            chosen_local = torch.topk(
                scores[batch_index].index_select(0, candidates).float(),
                k=remaining,
            ).indices
            result[batch_index, candidates.index_select(0, chosen_local)] = True
        if forced.numel():
            result[batch_index, forced] = True
    return result


def select_decode_keys(weights: torch.Tensor, *, fraction: float) -> DecodeSelection:
    """Select top global probabilities independently for each decode head."""

    fraction = _validate_fraction(fraction)
    if weights.ndim != 4 or weights.shape[-2] != 1 or weights.shape[-1] == 0:
        raise ValueError("decode weights must have shape [batch, heads, 1, keys]")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("decode weights must be finite")
    keys = int(weights.shape[-1])
    budget = min(keys, max(1, math.ceil(keys * fraction)))
    top = torch.topk(weights.float(), k=budget, dim=-1)
    indices = torch.sort(top.indices, dim=-1).values.to(torch.int32)
    global_mass = torch.gather(weights.float(), -1, indices.to(torch.long)).sum(
        dim=-1,
        keepdim=True,
    )
    return DecodeSelection(indices=indices, global_mass=global_mass)

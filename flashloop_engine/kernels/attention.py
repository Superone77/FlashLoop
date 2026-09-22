"""Cached-global-mass sparse decode attention for one-token decoding."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only installations
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


@dataclass
class AttentionExecutionCounter:
    """Auditable distinction between the one source dense call and sparse targets."""

    dense_source_calls: int = 0
    dense_target_calls: int = 0
    sparse_target_calls: int = 0


@dataclass(frozen=True)
class DenseSourceAttention:
    """Loop-2 output and the cached state required by later sparse loops."""

    output: torch.Tensor
    indices: torch.Tensor
    global_mass: torch.Tensor
    selected_output: torch.Tensor


if _TRITON_AVAILABLE:

    @triton.jit
    def _cached_mass_sparse_delta_kernel(
        query,
        selected_keys,
        selected_values,
        global_mass,
        source_selected,
        output,
        log2_scale,
        selected_rows: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        head_batch = tl.program_id(0)
        dimensions = tl.arange(0, BLOCK_D)
        valid_dimension = dimensions < head_dim
        query_offset = head_batch * head_dim + dimensions
        query_row = tl.load(query + query_offset, mask=valid_dimension, other=0.0).to(tl.float32)
        maximum = -float("inf")
        denominator = 0.0
        accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start in range(0, selected_rows, BLOCK_N):
            rows = start + tl.arange(0, BLOCK_N)
            valid_row = rows < selected_rows
            key_offsets = (
                head_batch * selected_rows * head_dim
                + rows[:, None] * head_dim
                + dimensions[None, :]
            )
            valid = valid_row[:, None] & valid_dimension[None, :]
            keys = tl.load(selected_keys + key_offsets, mask=valid, other=0.0).to(tl.float32)
            logits = tl.sum(keys * query_row[None, :], axis=1) * log2_scale
            logits = tl.where(valid_row, logits, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(logits, axis=0))
            probabilities = tl.math.exp2(logits - next_maximum)
            alpha = tl.math.exp2(maximum - next_maximum)
            denominator = denominator * alpha + tl.sum(probabilities, axis=0)
            accumulator *= alpha
            value_offsets = key_offsets
            values = tl.load(selected_values + value_offsets, mask=valid, other=0.0).to(tl.float32)
            accumulator += tl.sum(probabilities[:, None] * values, axis=0)
            maximum = next_maximum

        target_selected = accumulator / denominator
        mass = tl.load(global_mass + head_batch)
        source = tl.load(source_selected + query_offset, mask=valid_dimension, other=0.0)
        delta = mass * (target_selected - source)
        tl.store(output + query_offset, delta, mask=valid_dimension)


def _validate_qkv(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> None:
    if query.ndim != 4 or query.shape[-2] != 1:
        raise ValueError("query must have shape [batch, heads, 1, head_dim]")
    if keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("keys and values must share [batch, heads, tokens, head_dim]")
    if keys.shape[:2] != query.shape[:2] or keys.shape[-1] != query.shape[-1]:
        raise ValueError("query, keys, and values disagree on batch/head dimensions")
    if keys.shape[-2] <= 0:
        raise ValueError("attention requires at least one key")


def dense_source_selector(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    fraction: float,
    counter: AttentionExecutionCounter | None = None,
) -> DenseSourceAttention:
    """Run the required dense Loop-2 attention and cache its selected mass."""

    _validate_qkv(query, keys, values)
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if counter is not None:
        counter.dense_source_calls += 1
    logits = torch.matmul(query.float(), keys.float().transpose(-1, -2)) / math.sqrt(
        int(query.shape[-1])
    )
    weights = torch.softmax(logits, dim=-1)
    selected_rows = min(
        int(keys.shape[-2]),
        max(1, math.ceil(int(keys.shape[-2]) * fraction)),
    )
    top = torch.topk(weights, k=selected_rows, dim=-1)
    indices = torch.sort(top.indices, dim=-1).values
    selected_weights = torch.gather(weights, -1, indices)
    global_mass = selected_weights.sum(dim=-1, keepdim=True)
    gather_indices = indices.transpose(-1, -2).expand(-1, -1, -1, int(values.shape[-1]))
    selected_values = torch.gather(values.float(), 2, gather_indices)
    selected_output = torch.matmul(selected_weights / global_mass, selected_values)
    output = torch.matmul(weights, values.float())
    return DenseSourceAttention(
        output=output.to(query.dtype),
        indices=indices.to(torch.int32),
        global_mass=global_mass,
        selected_output=selected_output.to(query.dtype),
    )


def cached_mass_sparse_delta(
    query: torch.Tensor,
    selected_keys: torch.Tensor,
    selected_values: torch.Tensor,
    *,
    global_mass: torch.Tensor,
    source_selected: torch.Tensor,
    counter: AttentionExecutionCounter | None = None,
    force_torch: bool = False,
) -> torch.Tensor:
    """Return ``mass * (target_subset - source_subset)`` without dense attention."""

    _validate_qkv(query, selected_keys, selected_values)
    expected_scalar_shape = (*query.shape[:2], 1, 1)
    if tuple(global_mass.shape) != expected_scalar_shape:
        raise ValueError("global_mass must have shape [batch, heads, 1, 1]")
    if source_selected.shape != query.shape:
        raise ValueError("source_selected must match query shape")
    if counter is not None:
        counter.sparse_target_calls += 1

    if force_torch or not query.is_cuda or not _TRITON_AVAILABLE:
        logits = torch.matmul(
            query.float(),
            selected_keys.float().transpose(-1, -2),
        ) / math.sqrt(int(query.shape[-1]))
        target_selected = torch.matmul(
            torch.softmax(logits, dim=-1),
            selected_values.float(),
        )
        return (global_mass.float() * (target_selected - source_selected.float())).to(
            query.dtype
        )

    if int(query.shape[-1]) != 128:
        raise ValueError("the first sparse attention kernel specializes head_dim=128")
    query = query.contiguous()
    selected_keys = selected_keys.contiguous()
    selected_values = selected_values.contiguous()
    global_mass = global_mass.contiguous()
    source_selected = source_selected.contiguous()
    output = torch.empty_like(query)
    selected_rows = int(selected_keys.shape[-2])
    _cached_mass_sparse_delta_kernel[(int(query.shape[0] * query.shape[1]),)](
        query,
        selected_keys,
        selected_values,
        global_mass,
        source_selected,
        output,
        (1.0 / math.sqrt(128)) * 1.4426950408889634,
        selected_rows=selected_rows,
        head_dim=128,
        BLOCK_N=64,
        BLOCK_D=128,
        num_warps=4,
    )
    return output

#!/usr/bin/env python3
"""Chipmunk-inspired sparse SwiGLU reference math for Ouro.

This module measures algorithm quality only.  The current dense MLP still runs
before a forward hook can replace its output, so none of its timings represent
a sparse-kernel speedup.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


EVALUATION_METHODS = (
    "dense",
    "sparse_attention",
    "sparse_attention_sparse_mlp",
    "cached_mass",
    "cached_mass_sparse_mlp",
    "cached_mass_token_sparse_prefill",
    "flashloop_quantized_kv",
    "flashloop_kivi2_channel_delta",
    "flashloop_kivi4_channel_delta",
    "flashloop_full_kivi4_per_loop",
    "cross_loop_kivi_channel_delta",
    "cross_loop_kivi_token_delta",
    "cross_loop_kivi4_channel_delta",
    "cross_loop_kivi4_token_delta",
)


def method_uses_sparse_attention(method: str) -> bool:
    if method not in EVALUATION_METHODS:
        raise ValueError(method)
    return method not in (
        "dense",
        "cross_loop_kivi_channel_delta",
        "cross_loop_kivi_token_delta",
        "cross_loop_kivi4_channel_delta",
        "cross_loop_kivi4_token_delta",
    )


def method_uses_cached_global_mass(method: str) -> bool:
    if method not in EVALUATION_METHODS:
        raise ValueError(method)
    return method in (
        "cached_mass",
        "cached_mass_sparse_mlp",
        "cached_mass_token_sparse_prefill",
        "flashloop_quantized_kv",
        "flashloop_kivi2_channel_delta",
        "flashloop_kivi4_channel_delta",
        "flashloop_full_kivi4_per_loop",
    )


def method_uses_cached_mass(method: str) -> bool:
    """Backward-compatible alias for cached global attention mass."""
    return method_uses_cached_global_mass(method)


def method_uses_sparse_mlp(method: str) -> bool:
    if method not in EVALUATION_METHODS:
        raise ValueError(method)
    return method in (
        "sparse_attention_sparse_mlp",
        "cached_mass_sparse_mlp",
    )


def method_uses_token_sparse_prefill(method: str) -> bool:
    if method not in EVALUATION_METHODS:
        raise ValueError(method)
    return method in (
        "cached_mass_token_sparse_prefill",
        "flashloop_quantized_kv",
        "flashloop_kivi2_channel_delta",
        "flashloop_kivi4_channel_delta",
        "flashloop_full_kivi4_per_loop",
    )


def method_uses_quantized_cross_loop_kv(method: str) -> bool:
    if method not in EVALUATION_METHODS:
        raise ValueError(method)
    return method in (
        "flashloop_quantized_kv",
        "flashloop_kivi2_channel_delta",
        "flashloop_kivi4_channel_delta",
        "cross_loop_kivi_channel_delta",
        "cross_loop_kivi_token_delta",
        "cross_loop_kivi4_channel_delta",
        "cross_loop_kivi4_token_delta",
    )


def method_uses_absolute_loop_kv(method: str) -> bool:
    if method not in EVALUATION_METHODS:
        raise ValueError(method)
    return method == "flashloop_full_kivi4_per_loop"


def attention_replacement_method(method: str) -> str | None:
    if method not in EVALUATION_METHODS:
        raise ValueError(method)
    if method in (
        "dense",
        "cross_loop_kivi_channel_delta",
        "cross_loop_kivi_token_delta",
        "cross_loop_kivi4_channel_delta",
        "cross_loop_kivi4_token_delta",
    ):
        return None
    if method in ("sparse_attention", "sparse_attention_sparse_mlp"):
        return "official"
    return "cached_source_mass"


def select_topk_column_mask(
    scores: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    """Select ceil(fraction * columns) independently for every token."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if scores.ndim < 1 or scores.shape[-1] == 0:
        raise ValueError("scores must have a non-empty column dimension")
    keep = min(scores.shape[-1], math.ceil(scores.shape[-1] * fraction))
    indices = torch.topk(scores, keep, dim=-1, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter_(-1, indices, True)


def cached_sparse_swiglu_output(
    cached_output: torch.Tensor,
    cached_intermediate: torch.Tensor,
    current_intermediate: torch.Tensor,
    mask: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Correct a cached MLP output with selected SwiGLU-column deltas."""
    selected_delta = (current_intermediate - cached_intermediate) * mask
    return cached_output + F.linear(selected_delta, down_weight)


def chipmunk_swiglu_scores(
    cached_gate: torch.Tensor,
    current_gate: torch.Tensor,
    cached_up: torch.Tensor,
    current_up: torch.Tensor,
    activation,
) -> torch.Tensor:
    """First-order magnitude proxy for a gated intermediate-column change."""
    gate_term = (current_gate - cached_gate).abs() * cached_up.abs()
    up_term = (current_up - cached_up).abs() * activation(cached_gate).abs()
    return gate_term + up_term


class SparseMLPIntervention:
    """Apply rolling-cache sparse SwiGLU corrections on decode Loops 3/4."""

    def __init__(
        self,
        model: torch.nn.Module,
        fraction: float = 0.30,
        record_metrics: bool = False,
    ) -> None:
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        self.fraction = fraction
        self.record_metrics = record_metrics
        self.current_loops: dict[int, int] = {}
        self.states: dict[int, dict[str, torch.Tensor]] = {}
        self.rows: list[dict[str, float | int]] = []
        self.prefill_mlp_calls = 0
        self.source_captures = 0
        self.target_replacements = 0
        self.selected_columns = 0
        self.handles = []
        for layer_index, layer in enumerate(model.model.layers):
            self.handles.append(
                layer.self_attn.register_forward_pre_hook(
                    self._attention_pre_hook(layer_index),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                layer.mlp.register_forward_hook(self._mlp_hook(layer_index))
            )

    def _attention_pre_hook(self, layer_index: int):
        def hook(_module, _inputs, kwargs):
            if "current_ut" not in kwargs:
                raise RuntimeError("Ouro attention call is missing current_ut")
            self.current_loops[layer_index] = int(kwargs["current_ut"])
            return None

        return hook

    def _mlp_hook(self, layer_index: int):
        def hook(module, inputs, output):
            current_loop = self.current_loops.get(layer_index)
            if current_loop is None:
                return None
            x = inputs[0]
            if x.shape[-2] != 1:
                self.prefill_mlp_calls += 1
                return None
            if current_loop == 0:
                return None

            with torch.no_grad():
                current_gate = module.gate_proj(x).detach()
                current_up = module.up_proj(x).detach()
                current_intermediate = (
                    module.act_fn(current_gate) * current_up
                ).detach()

                if current_loop == 1:
                    self.states[layer_index] = {
                        "gate": current_gate,
                        "up": current_up,
                        "intermediate": current_intermediate,
                        "output": output.detach(),
                    }
                    self.source_captures += 1
                    return None

                if current_loop not in (2, 3):
                    return None
                if layer_index not in self.states:
                    raise RuntimeError(
                        f"missing Loop-2 MLP cache for layer {layer_index}"
                    )

                state = self.states[layer_index]
                scores = chipmunk_swiglu_scores(
                    state["gate"],
                    current_gate,
                    state["up"],
                    current_up,
                    module.act_fn,
                )
                mask = select_topk_column_mask(scores, self.fraction)
                approximation = cached_sparse_swiglu_output(
                    state["output"],
                    state["intermediate"],
                    current_intermediate,
                    mask,
                    module.down_proj.weight,
                )

                if self.record_metrics:
                    error = output.float() - approximation.float()
                    self.rows.append(
                        {
                            "layer": layer_index + 1,
                            "from_loop": current_loop,
                            "to_loop": current_loop + 1,
                            "fraction": self.fraction,
                            "output_relative_l2_error": float(
                                (
                                    torch.linalg.vector_norm(error)
                                    / torch.linalg.vector_norm(output.float()).clamp_min(
                                        1e-12
                                    )
                                ).item()
                            ),
                        }
                    )

                state["gate"] = torch.where(mask, current_gate, state["gate"])
                state["up"] = torch.where(mask, current_up, state["up"])
                state["intermediate"] = torch.where(
                    mask,
                    current_intermediate,
                    state["intermediate"],
                )
                state["output"] = approximation.detach()
                self.target_replacements += 1
                self.selected_columns += (
                    mask.shape[0]
                    * mask.shape[1]
                    * math.ceil(mask.shape[-1] * self.fraction)
                )
                return approximation

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.current_loops.clear()
        self.states.clear()

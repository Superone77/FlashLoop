"""Memory-neutral packing of projection weights for fewer GEMM launches."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _require_biasless(module: nn.Linear, name: str) -> None:
    if module.bias is not None:
        raise ValueError(f"FlashLoop fused projection requires biasless {name}")
    if module.weight is None:
        raise ValueError(f"{name} has already released its weight")


@torch.no_grad()
def repack_projection_weights(layers: Sequence[nn.Module]) -> None:
    """Fuse Q/K/V and gate/up storage, then release the original Parameters."""

    for layer in layers:
        attention = layer.self_attn
        mlp = layer.mlp
        if hasattr(attention, "_flashloop_qkv_weight"):
            if not hasattr(mlp, "_flashloop_gate_up_weight"):
                raise RuntimeError("partially repacked FlashLoop layer")
            continue
        for name, module in (
            ("q_proj", attention.q_proj),
            ("k_proj", attention.k_proj),
            ("v_proj", attention.v_proj),
            ("gate_proj", mlp.gate_proj),
            ("up_proj", mlp.up_proj),
        ):
            _require_biasless(module, name)

        qkv_splits = (
            int(attention.q_proj.out_features),
            int(attention.k_proj.out_features),
            int(attention.v_proj.out_features),
        )
        qkv_weight = torch.cat(
            (
                attention.q_proj.weight,
                attention.k_proj.weight,
                attention.v_proj.weight,
            ),
            dim=0,
        ).detach()
        gate_up_splits = (
            int(mlp.gate_proj.out_features),
            int(mlp.up_proj.out_features),
        )
        gate_up_weight = torch.cat(
            (mlp.gate_proj.weight, mlp.up_proj.weight), dim=0
        ).detach()
        attention.register_buffer(
            "_flashloop_qkv_weight", qkv_weight, persistent=False
        )
        mlp.register_buffer(
            "_flashloop_gate_up_weight", gate_up_weight, persistent=False
        )
        attention._flashloop_qkv_splits = qkv_splits
        mlp._flashloop_gate_up_splits = gate_up_splits
        attention.q_proj.weight = None
        attention.k_proj.weight = None
        attention.v_proj.weight = None
        mlp.gate_proj.weight = None
        mlp.up_proj.weight = None


def project_qkv(
    attention: nn.Module,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project Q/K/V with one GEMM when the layer has packed weights."""

    weight = getattr(attention, "_flashloop_qkv_weight", None)
    if weight is None:
        return (
            attention.q_proj(values),
            attention.k_proj(values),
            attention.v_proj(values),
        )
    fused = F.linear(values, weight)
    return tuple(fused.split(attention._flashloop_qkv_splits, dim=-1))  # type: ignore[return-value]


def project_value(attention: nn.Module, values: torch.Tensor) -> torch.Tensor:
    """Project only V from a packed QKV weight without copying its storage."""

    weight = getattr(attention, "_flashloop_qkv_weight", None)
    if weight is None:
        return attention.v_proj(values)
    query_features, key_features, value_features = (
        int(size) for size in attention._flashloop_qkv_splits
    )
    value_weight = weight.narrow(
        0,
        query_features + key_features,
        value_features,
    )
    return F.linear(values, value_weight)


def fused_mlp(mlp: nn.Module, values: torch.Tensor) -> torch.Tensor:
    """Run gate/up with one GEMM when the layer has packed weights."""

    weight = getattr(mlp, "_flashloop_gate_up_weight", None)
    if weight is None:
        return mlp(values)
    gate, up = F.linear(values, weight).split(
        mlp._flashloop_gate_up_splits, dim=-1
    )
    return mlp.down_proj(mlp.act_fn(gate) * up)

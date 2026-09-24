"""Validated execution contracts for the first FlashLoop engine release."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


OFFICIAL_OURO_MODELING_SHA256 = (
    "c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413"
)


@dataclass(frozen=True)
class FlashLoopConfig:
    """Algorithm and physical-cache settings frozen by the paper method."""

    prefill_fraction_loop3: float = 0.25
    prefill_fraction_loop4: float = 0.10
    decode_key_fraction: float = 0.10
    kv_bits: int = 4
    group_size: int = 64
    residual_length: int = 64
    enable_token_sparse_prefill: bool = True
    enable_sparse_decode: bool = True
    quantize_cross_loop_kv: bool = True
    absolute_late_overrides: bool = False
    absolute_override_dtype: str = "int4"
    kv_reader_backend: str = "kivi_outer"
    reuse_selected_probabilities: bool = False
    fuse_projection_weights: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.prefill_fraction_loop4 <= self.prefill_fraction_loop3 <= 1.0:
            raise ValueError("prefill fractions require 0 < loop4 <= loop3 <= 1")
        if not 0.0 < self.decode_key_fraction <= 1.0:
            raise ValueError("decode_key_fraction must be in (0, 1]")
        if self.kv_bits != 4:
            raise ValueError("the production FlashLoop codec is fixed to 4 bits")
        if self.group_size != 64:
            raise ValueError("the production FlashLoop codec uses group_size=64")
        if self.residual_length != 64:
            raise ValueError("the production FlashLoop cache uses residual_length=64")
        if not isinstance(self.enable_token_sparse_prefill, bool):
            raise TypeError("enable_token_sparse_prefill must be bool")
        if not isinstance(self.enable_sparse_decode, bool):
            raise TypeError("enable_sparse_decode must be bool")
        if not isinstance(self.quantize_cross_loop_kv, bool):
            raise TypeError("quantize_cross_loop_kv must be bool")
        if not isinstance(self.absolute_late_overrides, bool):
            raise TypeError("absolute_late_overrides must be bool")
        if self.absolute_override_dtype not in {"int4", "bf16"}:
            raise ValueError("absolute_override_dtype must be 'int4' or 'bf16'")
        if self.absolute_override_dtype == "bf16" and not self.absolute_late_overrides:
            raise ValueError("bf16 absolute overrides require absolute_late_overrides=True")
        if self.kv_reader_backend not in {"kivi_outer", "triton_row"}:
            raise ValueError("kv_reader_backend must be 'kivi_outer' or 'triton_row'")
        if self.kv_reader_backend == "kivi_outer" and self.absolute_late_overrides:
            raise ValueError(
                "kivi_outer currently supports additive cross-loop deltas only"
            )
        if not isinstance(self.reuse_selected_probabilities, bool):
            raise TypeError("reuse_selected_probabilities must be bool")
        if not isinstance(self.fuse_projection_weights, bool):
            raise TypeError("fuse_projection_weights must be bool")

    @property
    def effective_prefill_fractions(self) -> tuple[float, float]:
        if not self.enable_token_sparse_prefill:
            return (1.0, 1.0)
        return (self.prefill_fraction_loop3, self.prefill_fraction_loop4)

    @property
    def effective_decode_key_fraction(self) -> float:
        if not self.enable_sparse_decode:
            return 1.0
        return self.decode_key_fraction


@dataclass(frozen=True)
class OuroModelSpec:
    """Shape information required by the specialized kernels."""

    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    head_dim: int
    total_loops: int


def validate_ouro_model(model: Any) -> OuroModelSpec:
    """Reject model variants outside the measured Ouro-1.4B/2.6B contract."""

    config = getattr(model, "config", None)
    backbone = getattr(model, "model", None)
    if config is None or backbone is None:
        raise ValueError("expected an OuroForCausalLM-style model")
    if getattr(config, "model_type", None) != "ouro":
        raise ValueError("FlashLoop requires model_type='ouro'")

    total_loops = int(getattr(config, "total_ut_steps", 0))
    num_layers = int(getattr(config, "num_hidden_layers", 0))
    num_heads = int(getattr(config, "num_attention_heads", 0))
    num_kv_heads = int(getattr(config, "num_key_value_heads", 0))
    head_dim = int(getattr(config, "head_dim", 0))
    if total_loops != 4:
        raise ValueError("FlashLoop currently specializes exactly four Ouro loops")
    if num_layers not in (24, 48):
        raise ValueError("supported Ouro checkpoints have 24 or 48 physical layers")
    if len(getattr(backbone, "layers", ())) != num_layers:
        raise ValueError("model layer count does not match num_hidden_layers")
    if num_heads != num_kv_heads:
        raise ValueError("the first engine release requires equal Q and KV head counts")
    if head_dim != 128:
        raise ValueError("the first engine release specializes head_dim=128")
    layer_types = tuple(getattr(config, "layer_types", ()))
    if len(layer_types) != num_layers or set(layer_types) != {"full_attention"}:
        raise ValueError("the first engine release requires full attention in every layer")
    if getattr(config, "sliding_window", None) is not None:
        raise ValueError("sliding-window Ouro variants are not supported")

    hidden_size = int(getattr(config, "hidden_size", 0))
    intermediate_size = int(getattr(config, "intermediate_size", 0))
    if hidden_size != num_heads * head_dim or intermediate_size <= 0:
        raise ValueError("invalid Ouro hidden or MLP dimensions")
    return OuroModelSpec(
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        head_dim=head_dim,
        total_loops=total_loops,
    )

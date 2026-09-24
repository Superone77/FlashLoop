"""CUDA kernels used by the FlashLoop inference engine."""

from .attention import (
    AttentionExecutionCounter,
    DenseSourceAttention,
    cached_mass_sparse_delta,
    dense_source_selector,
)
from .quantization import (
    triton_gather_dequantize,
    triton_pack_4bit,
    triton_packed_pv,
    triton_packed_qk,
)

__all__ = [
    "AttentionExecutionCounter",
    "DenseSourceAttention",
    "cached_mass_sparse_delta",
    "dense_source_selector",
    "triton_gather_dequantize",
    "triton_pack_4bit",
    "triton_packed_pv",
    "triton_packed_qk",
]

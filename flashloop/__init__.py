"""FlashLoop inference engine for recurrent Ouro language models."""

from .cache import CrossLoopKVCache, CrossLoopKVLayer
from .config import FlashLoopConfig, OuroModelSpec, validate_ouro_model
from .engine import (
    DecodeOutput,
    DecodeReceipt,
    EngineState,
    FlashLoopEngine,
    PrefillOutput,
)
from .packing import Packed4Bit, pack_k_per_channel, pack_v_per_token
from .prefill import (
    PrefillCompositeCache,
    PrefillLoopOutput,
    PrefillReceipt,
    dense_prefill_loop,
    sparse_prefill_loop,
)
from .selection import DecodeSelection, select_decode_keys, select_prefill_tokens

__all__ = [
    "DecodeSelection",
    "DecodeOutput",
    "DecodeReceipt",
    "CrossLoopKVCache",
    "CrossLoopKVLayer",
    "FlashLoopConfig",
    "FlashLoopEngine",
    "EngineState",
    "OuroModelSpec",
    "Packed4Bit",
    "PrefillCompositeCache",
    "PrefillLoopOutput",
    "PrefillReceipt",
    "PrefillOutput",
    "dense_prefill_loop",
    "pack_k_per_channel",
    "pack_v_per_token",
    "select_decode_keys",
    "select_prefill_tokens",
    "sparse_prefill_loop",
    "validate_ouro_model",
]

"""Fair preallocated dense-cache baseline for Ouro.

Ouro's reference ``UniversalTransformerCache`` grows every logical layer with
``torch.cat`` during decode.  This factory preserves its public behavior while
writing new K/V rows into fixed storage, so FlashLoop is compared against a
dense implementation without avoidable cache-growth copies.
"""

from __future__ import annotations

from typing import Any, TypeVar

import torch


CacheT = TypeVar("CacheT")


def make_static_causal_mask(
    cache_position: torch.Tensor,
    *,
    token_capacity: int,
    dtype: torch.dtype,
    batch_size: int,
) -> torch.Tensor:
    """Build the additive mask for a full-capacity static K/V allocation."""

    if not dtype.is_floating_point:
        raise TypeError("static causal mask requires a floating-point dtype")
    positions = cache_position.to(dtype=torch.long).flatten()
    key_positions = torch.arange(
        int(token_capacity), device=positions.device, dtype=torch.long
    )
    allowed = key_positions.unsqueeze(0) <= positions.unsqueeze(1)
    mask = torch.zeros(
        (positions.numel(), int(token_capacity)),
        device=positions.device,
        dtype=dtype,
    )
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0).expand(int(batch_size), 1, -1, -1)


def _positions_for_update(
    cache_position: torch.Tensor | None,
    *,
    current_length: int,
    update_rows: int,
    device: torch.device,
) -> torch.Tensor:
    if cache_position is None:
        positions = torch.arange(
            current_length,
            current_length + update_rows,
            device=device,
            dtype=torch.long,
        )
    else:
        positions = cache_position.to(device=device, dtype=torch.long).flatten()
    if positions.numel() != update_rows:
        raise ValueError("cache_position must identify every incoming K/V row")
    if positions.numel() > 1 and not bool(
        torch.all(positions[1:] == positions[:-1] + 1).item()
    ):
        raise ValueError("static dense cache positions must be contiguous")
    return positions


def make_preallocated_universal_cache(
    base_class: type[CacheT],
    *,
    logical_layers: int,
    token_capacity: int,
) -> CacheT:
    """Instantiate an Ouro-compatible cache with stable K/V storage pointers.

    ``base_class`` is resolved from the checkpoint's trusted remote-code module
    at runtime.  Returning a subclass keeps Ouro's ``isinstance`` gate true.
    """

    logical_layers = int(logical_layers)
    token_capacity = int(token_capacity)
    if logical_layers <= 0 or token_capacity <= 0:
        raise ValueError("logical_layers and token_capacity must be positive")

    class PreallocatedUniversalTransformerCache(base_class):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__(logical_layers)
            self.token_capacity = token_capacity
            self.key_cache = [None] * logical_layers
            self.value_cache = [None] * logical_layers
            self._layer_lengths = [0] * logical_layers

        def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: dict[str, Any] | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            layer_idx = int(layer_idx)
            if layer_idx not in range(logical_layers):
                raise IndexError("cache layer index exceeds configured logical layers")
            if key_states.ndim != 4 or value_states.shape != key_states.shape:
                raise ValueError("incoming K/V must share [batch, heads, rows, channels]")
            positions = _positions_for_update(
                None if cache_kwargs is None else cache_kwargs.get("cache_position"),
                current_length=self._layer_lengths[layer_idx],
                update_rows=int(key_states.shape[2]),
                device=key_states.device,
            )
            if positions.numel() == 0:
                return key_states, value_states
            first = int(positions[0].item())
            end = int(positions[-1].item()) + 1
            if first < 0 or end > token_capacity:
                raise IndexError(
                    f"cache update [{first}, {end}) exceeds token capacity {token_capacity}"
                )
            if self.key_cache[layer_idx] is None:
                shape = (
                    int(key_states.shape[0]),
                    int(key_states.shape[1]),
                    token_capacity,
                    int(key_states.shape[3]),
                )
                self.key_cache[layer_idx] = torch.empty(
                    shape, device=key_states.device, dtype=key_states.dtype
                )
                self.value_cache[layer_idx] = torch.empty_like(
                    self.key_cache[layer_idx]
                )
            storage_key = self.key_cache[layer_idx]
            storage_value = self.value_cache[layer_idx]
            assert storage_key is not None and storage_value is not None
            expected_prefix = (
                int(key_states.shape[0]),
                int(key_states.shape[1]),
                int(key_states.shape[3]),
            )
            if (
                int(storage_key.shape[0]),
                int(storage_key.shape[1]),
                int(storage_key.shape[3]),
            ) != expected_prefix:
                raise ValueError("incoming K/V batch, head, or channel shape changed")
            storage_key[:, :, first:end, :].copy_(key_states)
            storage_value[:, :, first:end, :].copy_(value_states)
            self._layer_lengths[layer_idx] = max(self._layer_lengths[layer_idx], end)
            self._seen_tokens = self._layer_lengths[0]
            # Match Hugging Face StaticLayer semantics: attention sees the
            # complete contiguous allocation and the causal mask excludes
            # unoccupied future positions.  Returning a prefix view would be
            # strided across heads and force SDPA to repack it every layer.
            return storage_key, storage_value

        def get_seq_length(self, layer_idx: int | None = 0) -> int:
            layer_idx = 0 if layer_idx is None else int(layer_idx)
            if layer_idx not in range(logical_layers):
                return 0
            return int(self._layer_lengths[layer_idx])

        def get_max_cache_shape(self, layer_idx: int = 0) -> int:
            return token_capacity

        def get_max_length(self) -> int:
            return token_capacity

        def get_mask_sizes(
            self,
            cache_position: torch.Tensor,
            layer_idx: int = 0,
        ) -> tuple[int, int]:
            # ``update`` returns the complete preallocated tensor, rather than
            # a prefix view.  The causal mask therefore has to cover that same
            # token dimension and exclude unwritten future rows.  Ouro's base
            # cache leaves HF's ``layers`` list empty, whose inherited fallback
            # reports only ``cache_position.shape[0]`` and would broadcast a
            # one-column decode mask across the whole allocation.
            return token_capacity, 0

        def clear(self) -> None:
            self.key_cache = [None] * logical_layers
            self.value_cache = [None] * logical_layers
            self._layer_lengths = [0] * logical_layers
            self._seen_tokens = 0

    return PreallocatedUniversalTransformerCache()  # type: ignore[return-value]

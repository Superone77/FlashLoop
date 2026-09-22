#!/usr/bin/env python3
"""Evaluate the actual Chipmunk attention selector on Ouro-1.4B.

This is an algorithm-quality experiment, not a sparse-kernel benchmark.  Dense
attention is still computed so that we can compare against exact targets and
inject a reference implementation of the sparse replacement into the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from .diagnostic_utils import PROMPT


DEFAULT_FRACTIONS = (0.10, 0.20, 0.30, 0.50, 1.00)


def scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    return float(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def repeat_kv(values: torch.Tensor, target_heads: int) -> torch.Tensor:
    groups = target_heads // values.shape[1]
    if groups == 1:
        return values
    return values.repeat_interleave(groups, dim=1)


def causal_allowed(query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
    past = key_length - query_length
    query_positions = torch.arange(query_length, device=device) + past
    key_positions = torch.arange(key_length, device=device)
    return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)


def attention_state(
    module: torch.nn.Module,
    kwargs: dict[str, Any],
    output: tuple[torch.Tensor, torch.Tensor],
    num_layers: int,
) -> dict[str, torch.Tensor | int]:
    attention_output, weights = output
    if weights is None:
        raise RuntimeError("eager attention weights are required")
    current_loop = int(kwargs["current_ut"])
    cache = kwargs["past_key_value"]
    cache_index = current_loop * num_layers + int(module.layer_idx)
    keys = cache.key_cache[cache_index]
    values = cache.value_cache[cache_index]
    if keys is None or values is None:
        raise RuntimeError(f"missing KV cache at index {cache_index}")

    hidden = kwargs["hidden_states"]
    hidden_shape = (*hidden.shape[:-1], -1, int(module.head_dim))
    query = module.q_proj(hidden).view(hidden_shape).transpose(1, 2)
    cos, sin = kwargs["position_embeddings"]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query = query * cos + rotate_half(query) * sin

    keys = repeat_kv(keys, query.shape[1])
    values = repeat_kv(values, query.shape[1])
    logits = torch.matmul(query.float(), keys.float().transpose(2, 3)) * float(
        module.scaling
    )
    allowed = causal_allowed(logits.shape[-2], logits.shape[-1], logits.device)
    allowed_full = allowed.view(1, 1, *allowed.shape)
    masked_logits = logits.masked_fill(~allowed_full, -torch.inf)
    softmax_max = masked_logits.max(dim=-1, keepdim=True).values
    softmax_sum = (
        torch.exp(masked_logits - softmax_max)
        .masked_fill(~allowed_full, 0)
        .sum(dim=-1, keepdim=True)
    )
    recomputed_weights = (
        torch.exp(masked_logits - softmax_max)
        .masked_fill(~allowed_full, 0)
        / softmax_sum
    )
    return {
        "loop": current_loop,
        "output": attention_output,
        "weights": weights,
        "values": values,
        "logits": logits,
        "allowed": allowed,
        "softmax_max": softmax_max,
        "softmax_sum": softmax_sum,
        "softmax_recompute_max_abs": (recomputed_weights - weights.float()).abs().max(),
    }


def make_topk_mask(
    scores: torch.Tensor, allowed: torch.Tensor, fraction: float
) -> torch.Tensor:
    batch, heads, queries, keys = scores.shape
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for query in range(queries):
        eligible = int(allowed[query].sum().item())
        active = min(eligible, max(1, math.ceil(eligible * fraction)))
        query_scores = scores[:, :, query, :].masked_fill(
            ~allowed[query].view(1, 1, keys), -torch.inf
        )
        indices = torch.topk(query_scores, active, dim=-1).indices
        mask[:, :, query, :].scatter_(-1, indices, True)
    return mask


def make_decode_topk_mask(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    """Top-k mask for one-token decoding, where every cached key is eligible."""
    if scores.shape[-2] != 1:
        raise ValueError("decode top-k mask requires exactly one query token")
    keys = scores.shape[-1]
    active = min(keys, max(1, math.ceil(keys * fraction)))
    indices = torch.topk(scores, active, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return mask


def masked_subset_softmax(
    logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Normalize attention probabilities over selected keys only."""
    if mask.dtype != torch.bool:
        raise TypeError("subset mask must be boolean")
    if mask.shape != logits.shape:
        raise ValueError(
            f"mask/logits shape mismatch: {tuple(mask.shape)} vs {tuple(logits.shape)}"
        )
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("every query/head row must select at least one key")
    probabilities = torch.softmax(
        logits.float().masked_fill(~mask, -torch.inf), dim=-1
    )
    return probabilities.masked_fill(~mask, 0)


def official_sparse_output(
    source_output: torch.Tensor,
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    source_values: torch.Tensor,
    target_values: torch.Tensor,
    mask: torch.Tensor,
    output_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply Chipmunk's cached-base plus subset-softmax replacement."""
    source_probabilities = masked_subset_softmax(source_logits, mask)
    target_probabilities = masked_subset_softmax(target_logits, mask)
    source_selected = torch.matmul(source_probabilities, source_values.float())
    target_selected = torch.matmul(target_probabilities, target_values.float())
    delta = target_selected - source_selected
    if output_weight is not None:
        delta = delta.transpose(1, 2).contiguous().reshape(
            *source_output.shape[:-1], -1
        )
        delta = F.linear(delta, output_weight.float())
    if delta.shape != source_output.shape:
        raise ValueError(
            f"delta/output shape mismatch: {tuple(delta.shape)} vs {tuple(source_output.shape)}"
        )
    return (source_output.float() + delta).to(source_output.dtype)


def selected_attention_mass(
    logits: torch.Tensor,
    mask: torch.Tensor,
    allowed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the selected keys' probability mass under the global softmax."""
    if mask.dtype != torch.bool or mask.shape != logits.shape:
        raise ValueError("mass mask must be boolean and match logits")
    if allowed is None:
        allowed = torch.ones_like(mask, dtype=torch.bool)
    else:
        allowed = allowed.expand_as(logits)
    if not bool((mask <= allowed).all()):
        raise ValueError("selected keys must be globally eligible")
    probabilities = torch.softmax(
        logits.float().masked_fill(~allowed, -torch.inf), dim=-1
    )
    return (probabilities * mask).sum(dim=-1, keepdim=True)


def mass_scaled_sparse_output(
    source_output: torch.Tensor,
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    source_values: torch.Tensor,
    target_values: torch.Tensor,
    mask: torch.Tensor,
    source_scale: torch.Tensor | float,
    target_scale: torch.Tensor | float,
    output_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply independently mass-scaled source and target subset attention."""
    source_subset = torch.matmul(
        masked_subset_softmax(source_logits, mask), source_values.float()
    )
    target_subset = torch.matmul(
        masked_subset_softmax(target_logits, mask), target_values.float()
    )
    delta = target_subset * torch.as_tensor(
        target_scale, device=target_subset.device, dtype=torch.float32
    ) - source_subset * torch.as_tensor(
        source_scale, device=source_subset.device, dtype=torch.float32
    )
    if output_weight is not None:
        delta = delta.transpose(1, 2).contiguous().reshape(
            *source_output.shape[:-1], -1
        )
        delta = F.linear(delta, output_weight.float())
    if delta.shape != source_output.shape:
        raise ValueError(
            f"delta/output shape mismatch: {tuple(delta.shape)} vs {tuple(source_output.shape)}"
        )
    return (source_output.float() + delta).to(source_output.dtype)


def atomic_delta_scores(source: dict[str, Any], target: dict[str, Any]) -> torch.Tensor:
    p0 = source["weights"].float()
    p1 = target["weights"].float()
    v0 = source["values"].float()
    v1 = target["values"].float()
    v0_norm = v0.square().sum(dim=-1).unsqueeze(-2)
    v1_norm = v1.square().sum(dim=-1).unsqueeze(-2)
    cross = (v0 * v1).sum(dim=-1).unsqueeze(-2)
    return (
        p1.square() * v1_norm
        + p0.square() * v0_norm
        - 2 * p0 * p1 * cross
    ).clamp_min(0)


def selected_target_probabilities(
    source: dict[str, Any], target: dict[str, Any], probability_mode: str
) -> torch.Tensor:
    if probability_mode == "exact":
        return target["weights"].float()
    if probability_mode == "cached_normalizer":
        allowed = target["allowed"].view(1, 1, *target["allowed"].shape)
        return (
            torch.exp(target["logits"] - source["softmax_max"])
            .masked_fill(~allowed, 0)
            / source["softmax_sum"]
        )
    raise ValueError(probability_mode)


def approximate_attention_output(
    module: torch.nn.Module,
    source: dict[str, Any],
    target: dict[str, Any],
    mask: torch.Tensor,
    probability_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_prob = source["weights"].float()
    target_prob = selected_target_probabilities(source, target, probability_mode)
    source_values = source["values"].float()
    target_values = target["values"].float()
    source_pre = torch.matmul(source_prob, source_values)
    old_selected = torch.matmul(source_prob * mask, source_values)
    new_selected = torch.matmul(target_prob * mask, target_values)
    approximation_pre = source_pre - old_selected + new_selected
    target_output = target["output"]
    flattened = approximation_pre.transpose(1, 2).contiguous().reshape(
        *target_output.shape[:-1], -1
    )
    approximation = F.linear(flattened, module.o_proj.weight.float()).to(
        target_output.dtype
    )
    return approximation, target_prob


def reconstruction_metrics(
    previous: torch.Tensor,
    current: torch.Tensor,
    approximation: torch.Tensor,
    norm_module: torch.nn.Module,
) -> dict[str, float]:
    previous_f = previous.float()
    current_f = current.float()
    approximation_f = approximation.float()
    true_delta = current_f - previous_f
    error = current_f - approximation_f
    current_norm = torch.linalg.vector_norm(current_f).clamp_min(1e-12)
    delta_norm = torch.linalg.vector_norm(true_delta).clamp_min(1e-12)

    previous_n = norm_module(previous).float()
    current_n = norm_module(current).float()
    approximation_n = norm_module(approximation).float()
    delta_n = current_n - previous_n
    error_n = current_n - approximation_n
    delta_n_norm = torch.linalg.vector_norm(delta_n).clamp_min(1e-12)
    relative_delta_error = torch.linalg.vector_norm(error) / delta_norm
    relative_delta_error_n = torch.linalg.vector_norm(error_n) / delta_n_norm
    return {
        "output_relative_l2_error": scalar(torch.linalg.vector_norm(error) / current_norm),
        "delta_relative_l2_error_raw": scalar(relative_delta_error),
        "delta_energy_recovered_raw": scalar(1 - relative_delta_error.square()),
        "delta_relative_l2_error_postnorm": scalar(relative_delta_error_n),
        "delta_energy_recovered_postnorm": scalar(1 - relative_delta_error_n.square()),
    }


class DenseAttentionCapture:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.num_layers = int(model.config.num_hidden_layers)
        self.states: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.handles = []
        for layer_index, layer in enumerate(model.model.layers):
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    self._hook(layer_index), with_kwargs=True
                )
            )

    def _hook(self, layer_index: int):
        def hook(module, _inputs, kwargs, output):
            state = attention_state(module, kwargs, output, self.num_layers)
            self.states[layer_index].append(
                {
                    key: value.detach() if isinstance(value, torch.Tensor) else value
                    for key, value in state.items()
                }
            )
        return hook

    def take(self, num_loops: int) -> list[list[dict[str, Any]]]:
        result = []
        for loop in range(num_loops):
            loop_states = []
            for layer in range(self.num_layers):
                observed = self.states[layer]
                if len(observed) != num_loops:
                    raise RuntimeError(
                        f"layer {layer} attention calls={len(observed)}, expected={num_loops}"
                    )
                if int(observed[loop]["loop"]) != loop:
                    raise RuntimeError(f"unexpected loop order in layer {layer}")
                loop_states.append(observed[loop])
            result.append(loop_states)
        self.states.clear()
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def analyze_snapshot(
    phase: str,
    decode_step: int,
    snapshot: list[list[dict[str, Any]]],
    model: torch.nn.Module,
    fractions: tuple[float, ...],
    rows: list[dict[str, Any]],
) -> None:
    for source_loop in range(len(snapshot) - 1):
        for layer_index, layer in enumerate(model.model.layers):
            source = snapshot[source_loop][layer_index]
            target = snapshot[source_loop + 1][layer_index]
            allowed = source["allowed"]
            allowed_full = allowed.view(1, 1, *allowed.shape)
            oracle_scores = atomic_delta_scores(source, target).masked_fill(
                ~allowed_full, 0
            )
            random_generator = torch.Generator(device=oracle_scores.device)
            random_generator.manual_seed(
                1729 + 1000 * decode_step + 100 * source_loop + layer_index
            )
            random_scores = torch.rand(
                oracle_scores.shape,
                generator=random_generator,
                device=oracle_scores.device,
            )
            for fraction in fractions:
                oracle_mask = make_topk_mask(oracle_scores, allowed, fraction)
                selectors = {
                    "chipmunk_prev_attention": make_topk_mask(
                        source["weights"].float(), allowed, fraction
                    ),
                    "oracle_contribution_delta": oracle_mask,
                    "random": make_topk_mask(random_scores, allowed, fraction),
                }
                total_oracle_energy = oracle_scores.sum().clamp_min(1e-30)
                for selector, mask in selectors.items():
                    modes = (
                        ("exact", "cached_normalizer")
                        if selector == "chipmunk_prev_attention"
                        else ("exact",)
                    )
                    intersection = (mask & oracle_mask).sum()
                    union = (mask | oracle_mask).sum().clamp_min(1)
                    for probability_mode in modes:
                        approximation, target_prob = approximate_attention_output(
                            layer.self_attn,
                            source,
                            target,
                            mask,
                            probability_mode,
                        )
                        metrics = reconstruction_metrics(
                            source["output"],
                            target["output"],
                            approximation,
                            layer.input_layernorm_2,
                        )
                        row = {
                            "phase": phase,
                            "decode_step": decode_step,
                            "layer": layer_index + 1,
                            "from_loop": source_loop + 1,
                            "to_loop": source_loop + 2,
                            "selector": selector,
                            "probability_mode": probability_mode,
                            "queries": source["weights"].shape[-2],
                            "keys": source["weights"].shape[-1],
                            "target_fraction": fraction,
                            "effective_fraction": scalar(
                                mask.sum() / allowed_full.expand_as(mask).sum()
                            ),
                            "oracle_topk_recall": scalar(intersection / oracle_mask.sum()),
                            "oracle_topk_jaccard": scalar(intersection / union),
                            "oracle_atomic_energy_selected": scalar(
                                oracle_scores.masked_select(mask).sum()
                                / total_oracle_energy
                            ),
                            "source_attention_mass_selected": scalar(
                                source["weights"].float().masked_select(mask).sum()
                                / source["weights"].float().sum().clamp_min(1e-30)
                            ),
                            "target_probability_row_sum_mean": scalar(
                                target_prob.sum(dim=-1).mean()
                            ),
                            "softmax_recompute_max_abs": scalar(
                                target["softmax_recompute_max_abs"]
                            ),
                        }
                        row.update(metrics)
                        rows.append(row)


class ChipmunkIntervention:
    """Inject one loop-to-loop sparse replacement across all attention layers."""

    def __init__(
        self,
        model: torch.nn.Module,
        source_loop: int,
        fraction: float,
        probability_mode: str,
    ):
        self.model = model
        self.num_layers = int(model.config.num_hidden_layers)
        self.source_loop = source_loop
        self.target_loop = source_loop + 1
        self.fraction = fraction
        self.probability_mode = probability_mode
        self.source_states: dict[int, dict[str, Any]] = {}
        self.rows: list[dict[str, Any]] = []
        self.handles = []
        for layer_index, layer in enumerate(model.model.layers):
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    self._hook(layer_index), with_kwargs=True
                )
            )

    def _hook(self, layer_index: int):
        def hook(module, _inputs, kwargs, output):
            current_loop = int(kwargs["current_ut"])
            if current_loop not in (self.source_loop, self.target_loop):
                return None
            state = attention_state(module, kwargs, output, self.num_layers)
            if current_loop == self.source_loop:
                self.source_states[layer_index] = {
                    key: value.detach() if isinstance(value, torch.Tensor) else value
                    for key, value in state.items()
                }
                return None

            source = self.source_states.get(layer_index)
            if source is None:
                raise RuntimeError(f"missing source state for layer {layer_index}")
            mask = make_topk_mask(
                source["weights"].float(), source["allowed"], self.fraction
            )
            approximation, target_prob = approximate_attention_output(
                module, source, state, mask, self.probability_mode
            )
            dense_output = output[0]
            dense_delta = dense_output.float() - source["output"].float()
            error = dense_output.float() - approximation.float()
            self.rows.append(
                {
                    "layer": layer_index + 1,
                    "from_loop": self.source_loop + 1,
                    "to_loop": self.target_loop + 1,
                    "fraction": self.fraction,
                    "probability_mode": self.probability_mode,
                    "queries": source["weights"].shape[-2],
                    "keys": source["weights"].shape[-1],
                    "local_output_relative_l2_error": scalar(
                        torch.linalg.vector_norm(error)
                        / torch.linalg.vector_norm(dense_output.float()).clamp_min(1e-12)
                    ),
                    "local_delta_relative_l2_error": scalar(
                        torch.linalg.vector_norm(error)
                        / torch.linalg.vector_norm(dense_delta).clamp_min(1e-12)
                    ),
                    "source_attention_mass_selected": scalar(
                        source["weights"].float().masked_select(mask).sum()
                        / source["weights"].float().sum().clamp_min(1e-30)
                    ),
                    "target_probability_row_sum_mean": scalar(
                        target_prob.sum(dim=-1).mean()
                    ),
                }
            )
            return approximation, output[1]

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.source_states.clear()


def token_comparison(reference: list[int], candidate: list[int]) -> dict[str, Any]:
    shared = min(len(reference), len(candidate))
    matches = [reference[index] == candidate[index] for index in range(shared)]
    prefix = 0
    for match in matches:
        if not match:
            break
        prefix += 1
    return {
        "exact_token_sequence_match": reference == candidate,
        "position_token_match_rate": sum(matches) / max(1, shared),
        "matching_prefix_tokens": prefix,
    }


def generate_tokens(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    decode_tokens: int,
    pad_token_id: int,
) -> list[int]:
    output = model.generate(
        **inputs,
        max_new_tokens=decode_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=pad_token_id,
    )
    prompt_tokens = inputs["input_ids"].shape[-1]
    return [int(value) for value in output[0, prompt_tokens:].tolist()]


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def summarize_offline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for source in (1, 2, 3):
        for fraction in DEFAULT_FRACTIONS:
            for selector, mode in (
                ("chipmunk_prev_attention", "exact"),
                ("chipmunk_prev_attention", "cached_normalizer"),
                ("oracle_contribution_delta", "exact"),
                ("random", "exact"),
            ):
                selected = [
                    row
                    for row in rows
                    if row["phase"] == "decode"
                    and row["from_loop"] == source
                    and row["target_fraction"] == fraction
                    and row["selector"] == selector
                    and row["probability_mode"] == mode
                ]
                if not selected:
                    continue
                key = f"decode/{source}->{source + 1}/f={fraction:.2f}/{selector}/{mode}"
                summary[key] = {
                    "row_count": len(selected),
                    "effective_fraction_mean": mean(selected, "effective_fraction"),
                    "oracle_topk_recall_mean": mean(selected, "oracle_topk_recall"),
                    "oracle_atomic_energy_selected_mean": mean(
                        selected, "oracle_atomic_energy_selected"
                    ),
                    "delta_energy_recovered_postnorm_mean": mean(
                        selected, "delta_energy_recovered_postnorm"
                    ),
                    "output_relative_l2_error_mean": mean(
                        selected, "output_relative_l2_error"
                    ),
                }
    return summary


def run_self_test() -> None:
    allowed = causal_allowed(3, 3, torch.device("cpu"))
    assert torch.equal(
        allowed,
        torch.tensor(
            [[True, False, False], [True, True, False], [True, True, True]]
        ),
    )
    scores = torch.tensor([[[[3.0, 2.0, 1.0], [1.0, 4.0, 2.0], [1.0, 2.0, 5.0]]]])
    mask = make_topk_mask(scores, allowed, 0.01)
    assert mask.sum().item() == 3
    assert mask[0, 0].diag().all()
    reference = [1, 2, 3, 4]
    comparison = token_comparison(reference, [1, 2, 7, 4])
    assert comparison["matching_prefix_tokens"] == 2
    assert comparison["position_token_match_rate"] == 0.75
    print("CHIPMUNK_SELECTOR_SELF_TEST_OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--fractions", type=float, nargs="+", default=DEFAULT_FRACTIONS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.model is None or args.output_dir is None:
        parser.error("--model and --output-dir are required unless --self-test is used")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    fractions = tuple(float(value) for value in args.fractions)
    if any(value <= 0 or value > 1 for value in fractions):
        raise ValueError(f"invalid fractions: {fractions}")

    started = time.perf_counter()
    set_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().to("cuda")
    num_loops = int(model.config.total_ut_steps)
    encoded = tokenizer(PROMPT, return_tensors="pt")
    inputs = {key: value.to("cuda") for key, value in encoded.items()}
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    torch.cuda.reset_peak_memory_stats()

    offline_rows: list[dict[str, Any]] = []
    capture = DenseAttentionCapture(model)
    with torch.inference_mode():
        prefill_output = model(
            **inputs, use_cache=True, logits_to_keep=1, return_dict=True
        )
        prefill_snapshot = capture.take(num_loops)
        analyze_snapshot(
            "prefill", 0, prefill_snapshot, model, fractions, offline_rows
        )
        cache = prefill_output.past_key_values
        attention_mask = inputs["attention_mask"]
        next_token = prefill_output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        dense_generated: list[int] = []
        for decode_step in range(1, args.decode_tokens + 1):
            dense_generated.append(int(next_token.item()))
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )
            decode_output = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            snapshot = capture.take(num_loops)
            analyze_snapshot(
                "decode", decode_step, snapshot, model, fractions, offline_rows
            )
            cache = decode_output.past_key_values
            next_token = decode_output.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
    capture.close()

    # End-to-end interventions are separate runs. Dense attention is still
    # evaluated internally; the hook replaces its output with the reference
    # Chipmunk approximation so errors propagate through later layers/loops.
    intervention_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        baseline_tokens = generate_tokens(
            model, inputs, args.decode_tokens, int(pad_token_id)
        )
        if baseline_tokens != dense_generated:
            raise RuntimeError("manual dense decode and generate baseline disagree")
        for source_loop in range(num_loops - 1):
            for probability_mode in ("exact", "cached_normalizer"):
                for fraction in fractions:
                    intervention = ChipmunkIntervention(
                        model, source_loop, fraction, probability_mode
                    )
                    candidate = generate_tokens(
                        model, inputs, args.decode_tokens, int(pad_token_id)
                    )
                    intervention.close()
                    comparison = token_comparison(baseline_tokens, candidate)
                    generation_rows.append(
                        {
                            "from_loop": source_loop + 1,
                            "to_loop": source_loop + 2,
                            "fraction": fraction,
                            "probability_mode": probability_mode,
                            "generated_tokens": len(candidate),
                            **comparison,
                            "completion": tokenizer.decode(
                                candidate, skip_special_tokens=True
                            ),
                        }
                    )
                    intervention_rows.extend(intervention.rows)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "selector_reconstruction.csv", offline_rows)
    write_csv(args.output_dir / "intervention_local_metrics.csv", intervention_rows)
    write_csv(args.output_dir / "generation_comparison.csv", generation_rows)
    payload = {
        "metadata": {
            "model_path": str(args.model),
            "prompt": PROMPT,
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
            "decode_tokens": args.decode_tokens,
            "dense_token_ids": baseline_tokens,
            "dense_completion": tokenizer.decode(
                baseline_tokens, skip_special_tokens=True
            ),
            "fractions": list(fractions),
            "selector": "top-k previous-loop attention probability per query/head",
            "exact_probability_mode": "selected target probabilities from dense target; isolates selector quality",
            "cached_normalizer_mode": "selected target logits normalized with source-loop softmax max and denominator",
            "experiment_scope": "algorithm quality only; dense PyTorch attention remains computed; no sparse kernel or speed claim",
            "git_commit": os.environ.get("OURO_GIT_COMMIT", "unknown"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": str(next(model.parameters()).dtype),
            "elapsed_seconds": round(elapsed, 3),
            "peak_gpu_memory_gib": round(
                torch.cuda.max_memory_allocated() / (1024**3), 3
            ),
        },
        "offline_summary": summarize_offline(offline_rows),
        "generation": generation_rows,
    }
    (args.output_dir / "chipmunk_selector_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(payload["metadata"], indent=2, ensure_ascii=False))
    print(json.dumps(generation_rows, indent=2, ensure_ascii=False))
    print("OURO_CHIPMUNK_SELECTOR_OK")
    return 0



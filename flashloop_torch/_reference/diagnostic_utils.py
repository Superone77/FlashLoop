#!/usr/bin/env python3
"""Measure per-layer activation and KV similarity across Ouro recurrent loops."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


QUESTION = (
    "A shop discounts a 120 euro jacket by 25%, then adds 20% VAT to the "
    "discounted price. What is the final price? Explain briefly."
)
PROMPT = f"Question: {QUESTION}\nAnswer:"


def scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    return float(value)


def similarity_metrics(previous: torch.Tensor, current: torch.Tensor) -> dict[str, float]:
    """Metrics for directly reusing `previous` as an approximation to `current`."""
    a = previous.float()
    b = current.float()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")

    cosine = F.cosine_similarity(a, b, dim=-1, eps=1e-12).reshape(-1)
    a_centered = a - a.mean(dim=-1, keepdim=True)
    b_centered = b - b.mean(dim=-1, keepdim=True)
    centered_cosine = F.cosine_similarity(
        a_centered, b_centered, dim=-1, eps=1e-12
    ).reshape(-1)

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    difference = b_flat - a_flat
    b_norm = torch.linalg.vector_norm(b_flat).clamp_min(1e-12)
    a_norm = torch.linalg.vector_norm(a_flat).clamp_min(1e-12)
    global_cosine = F.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0), dim=-1, eps=1e-12
    )[0]

    return {
        "cosine_mean": scalar(cosine.mean()),
        "cosine_median": scalar(cosine.median()),
        "cosine_p05": scalar(torch.quantile(cosine, 0.05)),
        "centered_cosine_mean": scalar(centered_cosine.mean()),
        "global_cosine": scalar(global_cosine),
        "relative_l2_reuse": scalar(torch.linalg.vector_norm(difference) / b_norm),
        "delta_to_previous_norm": scalar(
            torch.linalg.vector_norm(difference) / a_norm
        ),
        "current_to_previous_norm": scalar(b_norm / a_norm),
    }


def extrapolation_metrics(
    previous: torch.Tensor, current: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    """Compare reuse, fixed linear extrapolation, and an oracle scalar extrapolation."""
    a = previous.float().reshape(-1)
    b = current.float().reshape(-1)
    c = target.float().reshape(-1)
    if a.shape != b.shape or b.shape != c.shape:
        raise ValueError("extrapolation tensors must have the same shape")

    prior_delta = b - a
    next_delta = c - b
    denominator = torch.dot(prior_delta, prior_delta).clamp_min(1e-12)
    alpha = torch.dot(prior_delta, next_delta) / denominator
    fixed_prediction = b + prior_delta
    oracle_prediction = b + alpha * prior_delta
    target_norm = torch.linalg.vector_norm(c).clamp_min(1e-12)

    def relative_error(prediction: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(c - prediction) / target_norm

    def cosine(prediction: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(
            prediction.unsqueeze(0), c.unsqueeze(0), dim=-1, eps=1e-12
        )[0]

    reuse_error = relative_error(b)
    fixed_error = relative_error(fixed_prediction)
    oracle_error = relative_error(oracle_prediction)
    return {
        "oracle_alpha": scalar(alpha),
        "reuse_relative_l2": scalar(reuse_error),
        "fixed_extrap_relative_l2": scalar(fixed_error),
        "oracle_extrap_relative_l2": scalar(oracle_error),
        "fixed_extrap_cosine": scalar(cosine(fixed_prediction)),
        "oracle_extrap_cosine": scalar(cosine(oracle_prediction)),
        "fixed_error_reduction_vs_reuse": scalar(
            (reuse_error - fixed_error) / reuse_error.clamp_min(1e-12)
        ),
        "oracle_error_reduction_vs_reuse": scalar(
            (reuse_error - oracle_error) / reuse_error.clamp_min(1e-12)
        ),
        "delta_direction_cosine": scalar(
            F.cosine_similarity(
                prior_delta.unsqueeze(0),
                next_delta.unsqueeze(0),
                dim=-1,
                eps=1e-12,
            )[0]
        ),
        "next_to_prior_delta_norm": scalar(
            torch.linalg.vector_norm(next_delta)
            / torch.linalg.vector_norm(prior_delta).clamp_min(1e-12)
        ),
    }


class LayerCapture:
    def __init__(self, model: torch.nn.Module):
        self.layer_outputs: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.loop_outputs: list[torch.Tensor] = []
        self.handles = []
        for layer_index, layer in enumerate(model.model.layers):
            self.handles.append(
                layer.register_forward_hook(self._layer_hook(layer_index))
            )
        self.handles.append(model.model.norm.register_forward_hook(self._norm_hook))

    def _layer_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"unexpected layer output type: {type(output)}")
            self.layer_outputs[layer_index].append(output.detach())

        return hook

    def _norm_hook(self, _module, _inputs, output):
        self.loop_outputs.append(output.detach())

    def take(
        self, num_layers: int, num_loops: int
    ) -> tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
        for layer_index in range(num_layers):
            observed = len(self.layer_outputs[layer_index])
            if observed != num_loops:
                raise RuntimeError(
                    f"layer {layer_index} ran {observed} times, expected {num_loops}"
                )
        if len(self.loop_outputs) != num_loops:
            raise RuntimeError(
                f"final norm ran {len(self.loop_outputs)} times, expected {num_loops}"
            )

        by_loop = [
            [
                self.layer_outputs[layer_index][loop_index]
                .float()
                .cpu()
                .contiguous()
                for layer_index in range(num_layers)
            ]
            for loop_index in range(num_loops)
        ]
        loop_outputs = [
            output.float().cpu().contiguous() for output in self.loop_outputs
        ]
        self.clear()
        return by_loop, loop_outputs

    def clear(self) -> None:
        self.layer_outputs.clear()
        self.loop_outputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def cache_snapshot(
    cache: Any,
    num_layers: int,
    num_loops: int,
    token_slice: slice,
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    keys: list[list[torch.Tensor]] = []
    values: list[list[torch.Tensor]] = []
    for loop_index in range(num_loops):
        loop_keys = []
        loop_values = []
        for layer_index in range(num_layers):
            cache_index = loop_index * num_layers + layer_index
            key = cache.key_cache[cache_index]
            value = cache.value_cache[cache_index]
            if key is None or value is None:
                raise RuntimeError(f"empty KV cache at index {cache_index}")
            loop_keys.append(
                key[:, :, token_slice, :].detach().float().cpu().contiguous()
            )
            loop_values.append(
                value[:, :, token_slice, :].detach().float().cpu().contiguous()
            )
        keys.append(loop_keys)
        values.append(loop_values)
    return keys, values


def append_tensor_rows(
    phase: str,
    kind: str,
    tensors: list[list[torch.Tensor]],
    similarity_rows: list[dict[str, Any]],
    extrapolation_rows: list[dict[str, Any]],
) -> None:
    num_loops = len(tensors)
    num_layers = len(tensors[0])
    for layer_index in range(num_layers):
        layer = layer_index + 1
        for previous_loop in range(num_loops - 1):
            row: dict[str, Any] = {
                "phase": phase,
                "kind": kind,
                "layer": layer,
                "from_loop": previous_loop + 1,
                "to_loop": previous_loop + 2,
            }
            row.update(
                similarity_metrics(
                    tensors[previous_loop][layer_index],
                    tensors[previous_loop + 1][layer_index],
                )
            )
            similarity_rows.append(row)

        for start_loop in range(num_loops - 2):
            row = {
                "phase": phase,
                "kind": kind,
                "layer": layer,
                "source_loops": f"{start_loop + 1},{start_loop + 2}",
                "target_loop": start_loop + 3,
            }
            row.update(
                extrapolation_metrics(
                    tensors[start_loop][layer_index],
                    tensors[start_loop + 1][layer_index],
                    tensors[start_loop + 2][layer_index],
                )
            )
            extrapolation_rows.append(row)


def append_loop_output_rows(
    phase: str,
    tensors: list[torch.Tensor],
    similarity_rows: list[dict[str, Any]],
    extrapolation_rows: list[dict[str, Any]],
) -> None:
    nested = [[tensor] for tensor in tensors]
    before_similarity = len(similarity_rows)
    before_extrapolation = len(extrapolation_rows)
    append_tensor_rows(
        phase,
        "loop_output_after_final_norm",
        nested,
        similarity_rows,
        extrapolation_rows,
    )
    for row in similarity_rows[before_similarity:]:
        row["layer"] = "post_norm"
    for row in extrapolation_rows[before_extrapolation:]:
        row["layer"] = "post_norm"


def concatenate_decode(
    steps: list[tuple[list[list[torch.Tensor]], list[torch.Tensor]]],
    num_layers: int,
    num_loops: int,
) -> tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
    activations = [
        [
            torch.cat([step[0][loop][layer] for step in steps], dim=1)
            for layer in range(num_layers)
        ]
        for loop in range(num_loops)
    ]
    loop_outputs = [
        torch.cat([step[1][loop] for step in steps], dim=1)
        for loop in range(num_loops)
    ]
    return activations, loop_outputs


def concatenate_cache_steps(
    steps: list[tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]],
    num_layers: int,
    num_loops: int,
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    keys = [
        [
            torch.cat([step[0][loop][layer] for step in steps], dim=2)
            for layer in range(num_layers)
        ]
        for loop in range(num_loops)
    ]
    values = [
        [
            torch.cat([step[1][loop][layer] for step in steps], dim=2)
            for layer in range(num_layers)
        ]
        for loop in range(num_loops)
    ]
    return keys, values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    similarity_rows: list[dict[str, Any]],
    extrapolation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    groups = sorted({(row["phase"], row["kind"]) for row in similarity_rows})
    for phase, kind in groups:
        rows = [
            row
            for row in similarity_rows
            if row["phase"] == phase
            and row["kind"] == kind
            and isinstance(row["layer"], int)
        ]
        if not rows:
            continue
        key = f"{phase}/{kind}"
        by_transition = {}
        for from_loop in (1, 2, 3):
            transition = [row for row in rows if row["from_loop"] == from_loop]
            by_transition[f"{from_loop}->{from_loop + 1}"] = {
                "cosine_mean_across_layers": sum(r["cosine_mean"] for r in transition)
                / len(transition),
                "relative_l2_mean_across_layers": sum(
                    r["relative_l2_reuse"] for r in transition
                )
                / len(transition),
                "most_similar_layer_by_relative_l2": min(
                    transition, key=lambda row: row["relative_l2_reuse"]
                )["layer"],
                "least_similar_layer_by_relative_l2": max(
                    transition, key=lambda row: row["relative_l2_reuse"]
                )["layer"],
            }
        summary[key] = by_transition

    extrap_groups = sorted(
        {(row["phase"], row["kind"]) for row in extrapolation_rows}
    )
    for phase, kind in extrap_groups:
        rows = [
            row
            for row in extrapolation_rows
            if row["phase"] == phase
            and row["kind"] == kind
            and isinstance(row["layer"], int)
        ]
        if not rows:
            continue
        key = f"{phase}/{kind}/extrapolation"
        by_target = {}
        for target_loop in (3, 4):
            target_rows = [row for row in rows if row["target_loop"] == target_loop]
            by_target[str(target_loop)] = {
                "fixed_error_reduction_mean": sum(
                    r["fixed_error_reduction_vs_reuse"] for r in target_rows
                )
                / len(target_rows),
                "oracle_error_reduction_mean": sum(
                    r["oracle_error_reduction_vs_reuse"] for r in target_rows
                )
                / len(target_rows),
                "oracle_alpha_mean": sum(r["oracle_alpha"] for r in target_rows)
                / len(target_rows),
                "fixed_extrap_better_layer_count": sum(
                    r["fixed_error_reduction_vs_reuse"] > 0 for r in target_rows
                ),
                "oracle_extrap_better_layer_count": sum(
                    r["oracle_error_reduction_vs_reuse"] > 0 for r in target_rows
                ),
            }
        summary[key] = by_target
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decode-tokens", type=int, default=16)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    model.to("cuda")

    num_layers = int(model.config.num_hidden_layers)
    num_loops = int(model.config.total_ut_steps)
    if num_loops != 4:
        raise ValueError(f"expected 4 loops, got {num_loops}")

    encoded = tokenizer(PROMPT, return_tensors="pt")
    inputs = {key: value.to("cuda") for key, value in encoded.items()}
    attention_mask = inputs["attention_mask"]
    capture = LayerCapture(model)

    similarity_rows: list[dict[str, Any]] = []
    extrapolation_rows: list[dict[str, Any]] = []
    decode_capture_steps = []
    decode_cache_steps = []

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        prefill_output = model(
            **inputs,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        prefill_activations, prefill_loop_outputs = capture.take(
            num_layers, num_loops
        )
        cache = prefill_output.past_key_values
        prefill_keys, prefill_values = cache_snapshot(
            cache, num_layers, num_loops, slice(None)
        )
        append_tensor_rows(
            "prefill",
            "activation",
            prefill_activations,
            similarity_rows,
            extrapolation_rows,
        )
        append_loop_output_rows(
            "prefill",
            prefill_loop_outputs,
            similarity_rows,
            extrapolation_rows,
        )
        append_tensor_rows(
            "prefill", "key", prefill_keys, similarity_rows, extrapolation_rows
        )
        append_tensor_rows(
            "prefill",
            "value",
            prefill_values,
            similarity_rows,
            extrapolation_rows,
        )

        next_token = prefill_output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_token_ids = []
        for _step in range(args.decode_tokens):
            generated_token_ids.append(int(next_token.item()))
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
            decode_capture_steps.append(capture.take(num_layers, num_loops))
            cache = decode_output.past_key_values
            decode_cache_steps.append(
                cache_snapshot(cache, num_layers, num_loops, slice(-1, None))
            )
            next_token = decode_output.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )

        decode_activations, decode_loop_outputs = concatenate_decode(
            decode_capture_steps, num_layers, num_loops
        )
        decode_keys, decode_values = concatenate_cache_steps(
            decode_cache_steps, num_layers, num_loops
        )
        append_tensor_rows(
            "decode_16",
            "activation",
            decode_activations,
            similarity_rows,
            extrapolation_rows,
        )
        append_loop_output_rows(
            "decode_16",
            decode_loop_outputs,
            similarity_rows,
            extrapolation_rows,
        )
        append_tensor_rows(
            "decode_16", "key", decode_keys, similarity_rows, extrapolation_rows
        )
        append_tensor_rows(
            "decode_16",
            "value",
            decode_values,
            similarity_rows,
            extrapolation_rows,
        )

        full_keys, full_values = cache_snapshot(
            cache, num_layers, num_loops, slice(None)
        )
        append_tensor_rows(
            "full_cache_after_decode_16",
            "key",
            full_keys,
            similarity_rows,
            extrapolation_rows,
        )
        append_tensor_rows(
            "full_cache_after_decode_16",
            "value",
            full_values,
            similarity_rows,
            extrapolation_rows,
        )

    capture.close()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    similarity_path = args.output_dir / "similarity.csv"
    extrapolation_path = args.output_dir / "extrapolation.csv"
    write_csv(similarity_path, similarity_rows)
    write_csv(extrapolation_path, extrapolation_rows)

    payload = {
        "metadata": {
            "model_path": str(args.model),
            "prompt": PROMPT,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "decode_tokens": args.decode_tokens,
            "generated_token_ids": generated_token_ids,
            "generated_text": tokenizer.decode(
                generated_token_ids, skip_special_tokens=True
            ),
            "num_layers": num_layers,
            "num_loops": num_loops,
            "hidden_size": int(model.config.hidden_size),
            "num_attention_heads": int(model.config.num_attention_heads),
            "head_dim": int(model.config.head_dim),
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
        "summary": summarize(similarity_rows, extrapolation_rows),
        "similarity_rows": similarity_rows,
        "extrapolation_rows": extrapolation_rows,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(payload["metadata"], indent=2, ensure_ascii=False))
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print("OURO_LOOP_SIMILARITY_OK")
    return 0



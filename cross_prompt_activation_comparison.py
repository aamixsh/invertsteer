"""
Cross-prompt activation baseline vs steering.

For each test instruction, extract baseline and steered hidden states on the
prompt only (no generation), at inversion_layer = steering_layer + 1 — same
as experiment.py.

After locating the first user-content token (longest common prefix of
tokenizing \"\" vs a marker string) and trimming the fixed trailing chat
suffix (longest common suffix of those two tokenizations — e.g. eot + assistant
header), comparisons use activations on [start : end) with
end = seq_len - suffix_len. L = min_i(end_i - start) unless --compare-len caps.

For instruction i (only on the user-content span, not system / fixed chat prefix):
  - steering_delta: mean_t ||steered_i[t] - baseline_i[t]||_2
  - vs each j != i:
      - cross_baseline_ij: mean_t ||baseline_j[t] - baseline_i[t]||_2
      - cross_steered_ij: mean_t ||steered_j[t] - baseline_i[t]||_2

This gives a scale reference: changing the prompt (baseline_j vs baseline_i)
often moves activations more or less than the steering vector on the same
prompt (steered_i - baseline_i).

Additionally, on the fixed trailing assistant span (last ``suffix_len`` tokens,
same slice for every instruction), we report the same pairwise distances —
cross-prompt alignment on the shared template tail vs steering on that tail.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from config import Config, EVIL_TEST_INSTRUCTIONS, TEST_INSTRUCTIONS, TEST_INSTRUCTIONS_REPHRASED
from model_utils import get_input_device, get_num_layers, get_tokenize_fn, load_model, set_seed
from steering import (
    SteeringConfig,
    get_hidden_states_with_steering,
    load_steering_direction,
)
from inversion import extract_hidden_states


def _user_content_bounds_from_probe_tokenizations(
    ids_empty: List[int], ids_marker: List[int]
) -> Tuple[int, int]:
    """
    Returns (start, suffix_len) where user-only tokens are ids[start : len(ids)-suffix_len).

    start = longest common prefix length (probe: "" vs marker).
    suffix_len = longest common suffix length (fixed assistant / generation-prompt tail).
    """
    lcp = 0
    for a, b in zip(ids_empty, ids_marker):
        if a == b:
            lcp += 1
        else:
            break
    if lcp >= len(ids_empty) or lcp >= len(ids_marker):
        lcp = 0

    i, j = len(ids_empty) - 1, len(ids_marker) - 1
    suffix_len = 0
    while i >= lcp and j >= lcp and ids_empty[i] == ids_marker[j]:
        suffix_len += 1
        i -= 1
        j -= 1

    return lcp, suffix_len


def _user_content_start_and_suffix_len(
    tokenize_fn, min_seq_len: int
) -> Tuple[int, int, int]:
    """
    Returns (start, suffix_len_applied, suffix_len_probe).

    suffix_len_probe is from empty vs marker; applied is capped so the shortest
    prompt in the run still has at least one user-content token.
    """
    try:
        ids_empty = tokenize_fn(instructions=[""]).input_ids[0].tolist()
        ids_marker = tokenize_fn(instructions=["UNIQUE_MARKER_XYZ"]).input_ids[0].tolist()
    except Exception:
        return 0, 0, 0
    start, suffix_probe = _user_content_bounds_from_probe_tokenizations(ids_empty, ids_marker)
    max_suffix = max(0, min_seq_len - start - 1)
    suffix_len = min(suffix_probe, max_suffix)
    return start, suffix_len, suffix_probe


def _mean_token_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """a, b: [T, d] on same device, same T."""
    return torch.norm(a.float() - b.float(), dim=-1).mean().item()


def _mse_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.mse_loss(a.float(), b.float()).item()


def _collect_prompt_activations(
    model,
    tokenize_fn,
    instructions: List[str],
    steering_config: SteeringConfig,
    inversion_layer: int,
) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    """
    Returns (seq_lens, baseline_stack, steered_stack) where each stack is
    [N, Lmax, d] with variable lengths — caller must slice [:L] per row
    or use seq_lens for valid length before padding.

    Actually we have different lengths — easier to return list of tensors.
    """
    input_device = get_input_device(model)
    seq_lens: List[int] = []
    baselines: List[torch.Tensor] = []
    steereds: List[torch.Tensor] = []
    for instr in instructions:
        inputs = tokenize_fn(instructions=[instr])
        input_ids = inputs.input_ids.to(input_device)
        seq_lens.append(int(input_ids.size(1)))
        b = extract_hidden_states(input_ids, model, inversion_layer)
        s = get_hidden_states_with_steering(model, input_ids, steering_config, inversion_layer)
        baselines.append(b.cpu())
        steereds.append(s.cpu())
    return seq_lens, baselines, steereds


def _common_content_L(content_lengths: List[int], compare_len: Optional[int]) -> int:
    m = min(content_lengths)
    if compare_len is not None:
        m = min(m, compare_len)
    if m < 1:
        raise ValueError("common user-content compare length L must be >= 1")
    return m


def _median_sorted(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    l2_sorted = sorted(xs)
    mid = len(l2_sorted) // 2
    if len(l2_sorted) % 2 == 1:
        return l2_sorted[mid]
    return 0.5 * (l2_sorted[mid - 1] + l2_sorted[mid])


def _pairwise_region_metrics(
    n: int,
    baseline_slices: List[torch.Tensor],
    steered_slices: List[torch.Tensor],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    baseline_slices[i], steered_slices[i]: [T, d] same T for all i.
    Returns (per_instruction rows without index/instruction_preview, pair_records).
    """
    per_row: List[Dict[str, Any]] = []
    pair_records: List[Dict[str, Any]] = []

    for i in range(n):
        bi = baseline_slices[i]
        si = steered_slices[i]
        steer_l2 = _mean_token_l2(si, bi)
        steer_mse = _mse_mean(si, bi)

        cross_base: List[Dict[str, Any]] = []
        cross_steer: List[Dict[str, Any]] = []
        for j in range(n):
            if j == i:
                continue
            bj = baseline_slices[j]
            sj = steered_slices[j]
            lb = _mean_token_l2(bj, bi)
            mb = _mse_mean(bj, bi)
            ls = _mean_token_l2(sj, bi)
            ms = _mse_mean(sj, bi)
            cross_base.append({"j": j, "l2_mean": lb, "mse_mean": mb})
            cross_steer.append({"j": j, "l2_mean": ls, "mse_mean": ms})
            pair_records.append(
                {
                    "i": i,
                    "j": j,
                    "cross_baseline_l2_mean": lb,
                    "cross_baseline_mse_mean": mb,
                    "cross_steered_vs_bi_l2_mean": ls,
                    "cross_steered_vs_bi_mse_mean": ms,
                }
            )

        l2_base_others = [x["l2_mean"] for x in cross_base]
        median_cross = _median_sorted(l2_base_others)

        per_row.append(
            {
                "steering_l2_mean": steer_l2,
                "steering_mse_mean": steer_mse,
                "cross_prompt_baseline_l2_vs_others": cross_base,
                "cross_prompt_steered_vs_bi_l2": cross_steer,
                "summary_vs_other_prompts": {
                    "cross_baseline_l2_min": min(l2_base_others) if l2_base_others else float("nan"),
                    "cross_baseline_l2_max": max(l2_base_others) if l2_base_others else float("nan"),
                    "cross_baseline_l2_mean": sum(l2_base_others) / len(l2_base_others)
                    if l2_base_others
                    else float("nan"),
                    "cross_baseline_l2_median": median_cross,
                    "ratio_median_cross_baseline_to_steering": (median_cross / steer_l2)
                    if steer_l2 > 1e-12
                    else float("inf"),
                },
            }
        )

    return per_row, pair_records


def run_comparison(
    config: Config,
    instructions: List[str],
    compare_len: Optional[int],
) -> Dict[str, Any]:
    set_seed(config.seed)
    model, tokenizer = load_model(
        config.model_id, config.device, config.dtype, load_in_4bit=config.load_in_4bit
    )
    tokenize_fn = get_tokenize_fn(
        tokenizer,
        use_chat_template=config.use_chat_template,
        add_special_tokens=config.add_special_tokens,
    )

    direction_path = config.get_direction_path()
    if not os.path.exists(direction_path):
        raise FileNotFoundError(f"Direction file not found: {direction_path}")

    direction, layer, metadata = load_steering_direction(direction_path, config.device)
    direction = direction.to(get_input_device(model))
    steering_config = SteeringConfig(
        direction=direction,
        layer=layer,
        method=config.steering_method,
        coeff=config.steering_coeff,
        steering_type=config.steering_type,
    )
    inversion_layer = layer + 1
    n_layers = get_num_layers(model)
    if inversion_layer < 0 or inversion_layer > n_layers:
        raise ValueError(f"inversion_layer {inversion_layer} out of range for model depth")

    seq_lens, baselines, steereds = _collect_prompt_activations(
        model, tokenize_fn, instructions, steering_config, inversion_layer
    )
    min_seq = min(seq_lens) if seq_lens else 0
    start, suffix_len, suffix_probe = _user_content_start_and_suffix_len(
        tokenize_fn, min_seq_len=min_seq
    )
    content_lengths = [max(0, s - start - suffix_len) for s in seq_lens]
    L = _common_content_L(content_lengths, compare_len)
    n = len(instructions)

    user_baseline_slices = [baselines[i][start : start + L] for i in range(n)]
    user_steered_slices = [steereds[i][start : start + L] for i in range(n)]
    user_per, pair_records = _pairwise_region_metrics(n, user_baseline_slices, user_steered_slices)

    per_row: List[Dict[str, Any]] = []
    for i in range(n):
        row = {
            "index": i,
            "instruction_preview": instructions[i][:200]
            + ("..." if len(instructions[i]) > 200 else ""),
            "seq_len_full": seq_lens[i],
            "user_content_start_index": start,
            "chat_suffix_tokens_trimmed": suffix_len,
            "compare_len_used": L,
            **user_per[i],
        }
        per_row.append(row)

    assistant_tail_pair_records: List[Dict[str, Any]] = []
    if suffix_len > 0:
        tail_baseline_slices = [
            baselines[i][seq_lens[i] - suffix_len : seq_lens[i]] for i in range(n)
        ]
        tail_steered_slices = [
            steereds[i][seq_lens[i] - suffix_len : seq_lens[i]] for i in range(n)
        ]
        tail_per, assistant_tail_pair_records = _pairwise_region_metrics(
            n, tail_baseline_slices, tail_steered_slices
        )
        for i in range(n):
            per_row[i]["assistant_tail"] = {
                "assistant_tail_tokens": suffix_len,
                **tail_per[i],
            }
    else:
        for i in range(n):
            per_row[i]["assistant_tail"] = None

    return {
        "timestamp": datetime.now().isoformat(),
        "model_id": config.model_id,
        "steering_layer": layer,
        "inversion_layer": inversion_layer,
        "steering_method": config.steering_method,
        "steering_coeff": config.steering_coeff,
        "steering_type": config.steering_type,
        "user_content_start_index": start,
        "chat_suffix_tokens_trimmed": suffix_len,
        "chat_suffix_tokens_probe_before_cap": suffix_probe,
        "compare_len": L,
        "compare_len_cap": compare_len,
        "seq_lens": seq_lens,
        "user_content_token_lengths": content_lengths,
        "direction_metadata": metadata,
        "per_instruction": per_row,
        "all_ordered_pairs": pair_records,
        "assistant_tail_ordered_pairs": assistant_tail_pair_records
        if suffix_len > 0
        else [],
    }


def run_rephrased_within_group_assistant_tail(config: Config) -> Dict[str, Any]:
    """
    For each original prompt i, compare its assistant-tail activations to the assistant-tail
    activations of the 5 rephrased variants of the same i.

    Produces exactly 5 pairs per original prompt (total 50 when there are 10 originals).
    Distances are computed on:
      tail slice = [seq_len - suffix_len : seq_len], where suffix_len is the common fixed tail
    discovered via the same empty-vs-marker probe (and capped so we don't wipe out all user
    tokens for the shortest prompt).
    """
    set_seed(config.seed)
    model, tokenizer = load_model(
        config.model_id, config.device, config.dtype, load_in_4bit=config.load_in_4bit
    )
    tokenize_fn = get_tokenize_fn(
        tokenizer,
        use_chat_template=config.use_chat_template,
        add_special_tokens=config.add_special_tokens,
    )

    direction_path = config.get_direction_path()
    if not os.path.exists(direction_path):
        raise FileNotFoundError(f"Direction file not found: {direction_path}")

    direction, layer, metadata = load_steering_direction(direction_path, config.device)
    direction = direction.to(get_input_device(model))
    steering_config = SteeringConfig(
        direction=direction,
        layer=layer,
        method=config.steering_method,
        coeff=config.steering_coeff,
        steering_type=config.steering_type,
    )
    inversion_layer = layer + 1
    n_layers = get_num_layers(model)
    if inversion_layer < 0 or inversion_layer > n_layers:
        raise ValueError(f"inversion_layer {inversion_layer} out of range for model depth")

    originals = TEST_INSTRUCTIONS
    rephrased_groups = TEST_INSTRUCTIONS_REPHRASED

    # Flatten but keep mapping original i -> variant indices in the flattened list.
    variant_indices_by_i: List[List[int]] = []
    instructions: List[str] = []
    for i, orig in enumerate(originals):
        instructions.append(orig)

    instructions_offset = len(originals)
    offset = 0
    for i, group in enumerate(rephrased_groups):
        # Stop at the shortest list to keep mapping consistent.
        if i >= len(originals):
            break
        idxs = []
        for _k, variant_prompt in enumerate(group):
            idxs.append(instructions_offset + offset)
            instructions.append(variant_prompt)
            offset += 1
        variant_indices_by_i.append(idxs)

    seq_lens, baselines, steereds = _collect_prompt_activations(
        model, tokenize_fn, instructions, steering_config, inversion_layer
    )

    min_seq = min(seq_lens) if seq_lens else 0
    start, suffix_len, suffix_probe = _user_content_start_and_suffix_len(
        tokenize_fn, min_seq_len=min_seq
    )
    if suffix_len < 1:
        return {
            "timestamp": datetime.now().isoformat(),
            "model_id": config.model_id,
            "steering_layer": layer,
            "inversion_layer": inversion_layer,
            "steering_method": config.steering_method,
            "steering_coeff": config.steering_coeff,
            "steering_type": config.steering_type,
            "user_content_start_index": start,
            "chat_suffix_tokens_trimmed": suffix_len,
            "chat_suffix_tokens_probe_before_cap": suffix_probe,
            "rephrased_within_group_assistant_tail_pairs": [],
            "rephrased_within_group_per_original_summary": [],
            "note": "suffix_len computed to 0; assistant-tail slice would be empty",
            "direction_metadata": metadata,
        }

    def _tail_slice(acts: torch.Tensor, seq_len: int) -> torch.Tensor:
        return acts[seq_len - suffix_len : seq_len]

    pairs: List[Dict[str, Any]] = []
    per_original: List[Dict[str, Any]] = []

    n_originals = len(variant_indices_by_i)
    for i in range(n_originals):
        orig_tail_base = _tail_slice(baselines[i], seq_lens[i])
        orig_tail_steered = _tail_slice(steereds[i], seq_lens[i])

        variant_idxs = variant_indices_by_i[i]
        variant_l2s_baseline_to_orig: List[float] = []

        for k, vidx in enumerate(variant_idxs):
            variant_tail_base = _tail_slice(baselines[vidx], seq_lens[vidx])
            variant_tail_steered = _tail_slice(steereds[vidx], seq_lens[vidx])

            lb = _mean_token_l2(variant_tail_base, orig_tail_base)
            mb = _mse_mean(variant_tail_base, orig_tail_base)
            ls = _mean_token_l2(variant_tail_steered, orig_tail_base)
            ms = _mse_mean(variant_tail_steered, orig_tail_base)

            variant_l2s_baseline_to_orig.append(lb)
            pairs.append(
                {
                    "original_index": i,
                    "rephrase_variant_index": k,
                    "original_seq_len_full": seq_lens[i],
                    "variant_seq_len_full": seq_lens[vidx],
                    "assistant_tail_tokens": suffix_len,
                    "baseline_variant_tail_l2_mean_vs_original_baseline_tail": lb,
                    "baseline_variant_tail_mse_mean_vs_original_baseline_tail": mb,
                    "steered_variant_tail_l2_mean_vs_original_baseline_tail": ls,
                    "steered_variant_tail_mse_mean_vs_original_baseline_tail": ms,
                }
            )

        per_original.append(
            {
                "original_index": i,
                "assistant_tail_tokens": suffix_len,
                "baseline_l2_mean_to_variants": sum(variant_l2s_baseline_to_orig)
                / len(variant_l2s_baseline_to_orig),
                "baseline_l2_median_to_variants": _median_sorted(variant_l2s_baseline_to_orig),
                "original_steering_on_tail_l2_mean": _mean_token_l2(orig_tail_steered, orig_tail_base),
                "original_steering_on_tail_mse_mean": _mse_mean(orig_tail_steered, orig_tail_base),
            }
        )

    return {
        "timestamp": datetime.now().isoformat(),
        "model_id": config.model_id,
        "steering_layer": layer,
        "inversion_layer": inversion_layer,
        "steering_method": config.steering_method,
        "steering_coeff": config.steering_coeff,
        "steering_type": config.steering_type,
        "user_content_start_index": start,
        "chat_suffix_tokens_trimmed": suffix_len,
        "chat_suffix_tokens_probe_before_cap": suffix_probe,
        "rephrased_within_group_assistant_tail_pairs": pairs,
        "rephrased_within_group_per_original_summary": per_original,
        "direction_metadata": metadata,
    }


def _json_safe(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="Compare prompt steering deltas to cross-prompt activation distances"
    )
    parser.add_argument("--model", type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--direction", type=str, default=None)
    parser.add_argument("--method", type=str, default="actadd")
    parser.add_argument("--coeff", type=float, default=-1.0)
    parser.add_argument("--steering-type", type=str, default="refusal")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--add-special-tokens", action="store_true")
    parser.add_argument(
        "--use-rephrased-prompts",
        action="store_true",
        default=False,
        help="Use TEST_INSTRUCTIONS_REPHRASED from config (flattened) as the instruction set.",
    )
    parser.add_argument(
        "--rephrased-within-group-tail",
        action="store_true",
        default=False,
        help=(
            "Compute assistant-tail activation distances only within each original prompt's "
            "5 rephrased variants (50 pairs total) and write a dedicated JSON output."
        ),
    )
    parser.add_argument(
        "--compare-len",
        type=int,
        default=None,
        help="Max user-content tokens after skipping fixed chat prefix and assistant tail; "
        "default L=min per prompt of (seq_len - start - suffix_trim)",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Comma-separated instructions; default TEST_INSTRUCTIONS by steering type",
    )
    cli = parser.parse_args()

    cfg = Config()
    if cli.model:
        cfg.model_id = cli.model
    cfg.device = cli.device
    cfg.load_in_4bit = cli.load_in_4bit
    cfg.seed = cli.seed
    if cli.direction:
        cfg.direction_path = cli.direction
    cfg.steering_method = cli.method
    cfg.steering_coeff = cli.coeff
    cfg.steering_type = cli.steering_type
    cfg.use_chat_template = not cli.no_chat_template
    cfg.add_special_tokens = cli.add_special_tokens
    cfg.special_start_tokens = cfg._get_default_special_tokens()

    rephrased_meta: Optional[List[Dict[str, int]]] = None

    if cli.rephrased_within_group_tail:
        results = run_rephrased_within_group_assistant_tail(cfg)
        model_alias = cfg.model_id.split("/")[-1]
        out_dir = os.path.join(cfg.output_dir, model_alias, "cross_prompt_activation")
        os.makedirs(out_dir, exist_ok=True)
        tag = f"{cfg.steering_type}_{cfg.steering_method}_coeff_{cfg.steering_coeff}_assistant_tail_L{results['chat_suffix_tokens_trimmed']}"
        path = os.path.join(out_dir, f"cross_prompt_activation_rephrased_within_group_assistant_tail_{tag}.json")
        with open(path, "w") as f:
            json.dump(_json_safe(results), f, indent=2)
        print(f"Wrote {path}")
        return

    if cli.instructions:
        instructions = [s.strip() for s in cli.instructions.split(",") if s.strip()]
    elif cli.use_rephrased_prompts:
        instructions = []
        rephrased_meta = []
        for rephrased_from_idx, variant_list in enumerate(TEST_INSTRUCTIONS_REPHRASED):
            for variant_idx, instr in enumerate(variant_list):
                instructions.append(instr)
                rephrased_meta.append(
                    {
                        "rephrased_from_instruction_index": rephrased_from_idx,
                        "rephrase_variant_index": variant_idx,
                    }
                )
    else:
        instructions = (
            TEST_INSTRUCTIONS if cfg.steering_type == "refusal" else EVIL_TEST_INSTRUCTIONS
        )

    model_alias = cfg.model_id.split("/")[-1]
    out_dir = os.path.join(cfg.output_dir, model_alias, "cross_prompt_activation")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Instructions: {len(instructions)}, compare_len cap: {cli.compare_len}")
    results = run_comparison(cfg, instructions, cli.compare_len)

    tag = f"{cfg.steering_type}_{cfg.steering_method}_coeff_{cfg.steering_coeff}_L{results['compare_len']}"
    if rephrased_meta is not None:
        tag = f"{tag}_rephrased"
    path = os.path.join(out_dir, f"cross_prompt_activation_{tag}.json")
    with open(path, "w") as f:
        json.dump(_json_safe(results), f, indent=2)

    if rephrased_meta is not None:
        results["rephrased_meta"] = rephrased_meta
        for idx, row in enumerate(results["per_instruction"]):
            meta = rephrased_meta[idx]
            row.update(meta)

    print(f"Wrote {path}")
    for row in results["per_instruction"]:
        r = row["summary_vs_other_prompts"]["ratio_median_cross_baseline_to_steering"]
        line = (
            f"  [{row['index']}] user_span: steer_l2={row['steering_l2_mean']:.6f} "
            f"median_cross/steer={r:.4f}"
        )
        if "rephrase_variant_index" in row:
            line = (
                f"  [{row['index']}] (from={row['rephrased_from_instruction_index']} "
                f"var={row['rephrase_variant_index']}) user_span: steer_l2={row['steering_l2_mean']:.6f} "
                f"median_cross/steer={r:.4f}"
            )
        at = row.get("assistant_tail")
        if at:
            rt = at["summary_vs_other_prompts"]["ratio_median_cross_baseline_to_steering"]
            line += f" | assistant_tail: steer_l2={at['steering_l2_mean']:.6f} median_cross/steer={rt:.4f}"
        print(line)


if __name__ == "__main__":
    main()

"""
Copying baseline: can the model repeat a steered generation (tokens + activations)?

For each instruction:
1) Greedy steered generation yields reference token ids and reference hidden states
   at the steering layer (same index convention as experiment.py: inversion_layer = layer + 1)
   for the generated span (teacher-forced full sequence under steering).

2) Explicit instruction: user text asks the model to repeat that steered text verbatim,
   then we greedy-generate without steering and compare tokens and activations.

3) In-context repetition (N): the steered assistant text is embedded in the user message
   N times, then a cue to output it again; same metrics.

Metrics per condition:
- Token: exact match, prefix match length, per-position match up to min length.
- Activations (generated span, same layer as main experiment), comparing the copy run to **two**
  references on the original instruction + steered token sequence:
  - **steered_ref**: hidden states with steering applied (matches get_hidden_states_with_steering).
  - **baseline_ref**: hidden states without steering on the same ids (natural teacher-forcing
    of the steered-generated text after the original prompt).
  For each reference we report teacher-forcing on the copy prompt + reference ids, and free
  generation on copy prompt + model ids (overlap-aligned MSE/L2).

With `--save-activation-tensors`, a companion .pkl stores tensors only: per instruction,
`ref_generated_span_{steered,baseline}` on the original prompt + steered ids, and per
condition `teacher_force_generated_span` / `free_generation_generated_span` plus ids and
lengths (no duplicate JSON metrics).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from config import Config, EVIL_TEST_INSTRUCTIONS, TEST_INSTRUCTIONS
from model_utils import get_input_device, get_num_layers, get_tokenize_fn, load_model, set_seed
from steering import (
    SteeringConfig,
    get_hidden_states_with_steering,
    load_steering_direction,
    steering_context,
)
from inversion import extract_hidden_states


def _gen_steered_ids(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    steering_config: SteeringConfig,
    max_new_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (full_output_ids [1, L+G], generated_ids [1, G])."""
    with steering_context(model, steering_config):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    gen = outputs[:, input_ids.size(1) :]
    return outputs, gen


def _gen_baseline_ids(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    max_new_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = outputs[:, input_ids.size(1) :]
    return outputs, gen


def _token_metrics(ref: torch.Tensor, hyp: torch.Tensor) -> Dict[str, Any]:
    """ref, hyp are 1D CPU long tensors."""
    r = ref.tolist()
    h = hyp.tolist()
    n = min(len(r), len(h))
    prefix = 0
    for i in range(n):
        if r[i] != h[i]:
            break
        prefix += 1
    per_pos = [r[i] == h[i] for i in range(n)] if n else []
    return {
        "ref_len": len(r),
        "hyp_len": len(h),
        "exact_match": r == h,
        "prefix_match_len": prefix,
        "per_position_match": per_pos,
    }


def _activation_metrics(
    ref_slice: torch.Tensor,
    hyp_slice: torch.Tensor,
) -> Dict[str, float]:
    """Both [G, d], possibly different G — compare overlapping prefix."""
    m = min(ref_slice.size(0), hyp_slice.size(0))
    if m == 0:
        return {"overlap_len": 0, "mse_mean": float("nan"), "l2_mean": float("nan")}
    a = ref_slice[:m].float()
    b = hyp_slice[:m].float()
    mse = F.mse_loss(a, b).item()
    l2 = torch.norm(a - b, dim=-1).mean().item()
    return {"overlap_len": m, "mse_mean": mse, "l2_mean": l2}


def _build_explicit_user_text(_instruction: str, steered_text: str) -> str:
    return (
        "Your task is to repeat the following text exactly, verbatim, with no additions or omissions:\n\n"
        f"{steered_text}\n\n"
        "Begin your repetition now:"
    )


def _build_icl_user_text(_instruction: str, steered_text: str, n_repeats: int) -> str:
    blocks = [f"{steered_text}" for _ in range(n_repeats)]
    body = "\n\n".join(blocks)
    return (
        "Your task is to repeat the following text exactly, verbatim once again:\n\n"
        f"{body}\n\n"
        "Begin your repetition now:"
    )


def run_single_instruction(
    model,
    tokenizer,
    tokenize_fn,
    instruction: str,
    steering_config: SteeringConfig,
    inversion_layer: int,
    max_new_tokens: int,
    icl_ns: List[int],
    run_explicit: bool,
    return_activation_tensors: bool = False,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    input_device = get_input_device(model)
    inputs = tokenize_fn(instructions=[instruction])
    instr_ids = inputs.input_ids.to(input_device)
    attn = inputs.attention_mask.to(input_device)
    instr_len = instr_ids.size(1)

    _, steered_gen = _gen_steered_ids(
        model, tokenizer, instr_ids, attn, steering_config, max_new_tokens
    )
    steered_gen = steered_gen[0].cpu()
    if steered_gen.numel() == 0:
        err = {
            "instruction": instruction,
            "error": "empty_steered_generation",
            "timestamp": datetime.now().isoformat(),
        }
        return err, None

    steered_text = tokenizer.decode(steered_gen.tolist(), skip_special_tokens=True)
    G = steered_gen.numel()

    full_steered = torch.cat([instr_ids, steered_gen.unsqueeze(0).to(input_device)], dim=1)
    ref_hidden_steered = get_hidden_states_with_steering(
        model, full_steered, steering_config, inversion_layer
    )
    ref_slice_steered = ref_hidden_steered[instr_len : instr_len + G, :].cpu()

    ref_hidden_baseline = extract_hidden_states(full_steered, model, inversion_layer)
    ref_slice_baseline = ref_hidden_baseline[instr_len : instr_len + G, :].cpu()

    activation_record: Optional[Dict[str, Any]] = None
    if return_activation_tensors:
        activation_record = {
            "instruction": instruction,
            "instr_len": int(instr_len),
            "steered_gen_len": int(G),
            "steered_gen_ids": steered_gen.clone(),
            "ref_generated_span_steered": ref_slice_steered.clone(),
            "ref_generated_span_baseline": ref_slice_baseline.clone(),
            "conditions": {},
        }

    out: Dict[str, Any] = {
        "instruction": instruction,
        "timestamp": datetime.now().isoformat(),
        "steering_layer": steering_config.layer,
        "inversion_layer": inversion_layer,
        "instr_len": instr_len,
        "steered_gen_len": G,
        "steered_generation_text": steered_text,
        "steered_gen_ids": steered_gen.tolist(),
        "reference_activation_gap_steered_vs_baseline": _activation_metrics(
            ref_slice_steered, ref_slice_baseline
        ),
        "conditions": {},
    }

    if run_explicit:
        user_text = _build_explicit_user_text(instruction, steered_text)
        metrics, act_t = _eval_copy_condition(
            model,
            tokenizer,
            tokenize_fn,
            user_text,
            steered_gen,
            ref_slice_steered,
            ref_slice_baseline,
            input_device,
            inversion_layer,
            G,
        )
        out["conditions"]["explicit"] = metrics
        if activation_record is not None:
            activation_record["conditions"]["explicit"] = act_t

    for n in icl_ns:
        if n < 1:
            continue
        user_text = _build_icl_user_text(instruction, steered_text, n)
        metrics, act_t = _eval_copy_condition(
            model,
            tokenizer,
            tokenize_fn,
            user_text,
            steered_gen,
            ref_slice_steered,
            ref_slice_baseline,
            input_device,
            inversion_layer,
            G,
        )
        key = f"icl_n{n}"
        out["conditions"][key] = metrics
        if activation_record is not None:
            activation_record["conditions"][key] = act_t

    return out, activation_record


def _eval_copy_condition(
    model,
    tokenizer,
    tokenize_fn,
    user_text: str,
    steered_gen_cpu: torch.Tensor,
    ref_slice_steered_cpu: torch.Tensor,
    ref_slice_baseline_cpu: torch.Tensor,
    input_device: torch.device,
    inversion_layer: int,
    G: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    copy_in = tokenize_fn(instructions=[user_text])
    copy_ids = copy_in.input_ids.to(input_device)
    copy_attn = copy_in.attention_mask.to(input_device)
    copy_len = copy_ids.size(1)

    _, hyp_gen = _gen_baseline_ids(model, tokenizer, copy_ids, copy_attn, G)
    hyp_gen = hyp_gen[0].cpu()
    tok = _token_metrics(steered_gen_cpu, hyp_gen)

    full_tf = torch.cat([copy_ids, steered_gen_cpu.unsqueeze(0).to(input_device)], dim=1)
    h_tf = extract_hidden_states(full_tf, model, inversion_layer)
    hyp_tf_slice = h_tf[copy_len : copy_len + G, :].cpu()
    act_tf_vs_steered = _activation_metrics(ref_slice_steered_cpu, hyp_tf_slice)
    act_tf_vs_baseline = _activation_metrics(ref_slice_baseline_cpu, hyp_tf_slice)

    full_free = torch.cat([copy_ids, hyp_gen.unsqueeze(0).to(input_device)], dim=1)
    h_free = extract_hidden_states(full_free, model, inversion_layer)
    hyp_free_len = hyp_gen.numel()
    hyp_free_slice = h_free[copy_len : copy_len + hyp_free_len, :].cpu()
    act_free_vs_steered = _activation_metrics(ref_slice_steered_cpu, hyp_free_slice)
    act_free_vs_baseline = _activation_metrics(ref_slice_baseline_cpu, hyp_free_slice)

    hyp_text = tokenizer.decode(hyp_gen.tolist(), skip_special_tokens=True)
    preview = user_text[:500] + ("..." if len(user_text) > 500 else "")
    metrics = {
        "copy_user_text_preview": preview,
        "copy_prompt_len": copy_len,
        "generated_text": hyp_text,
        "generated_ids": hyp_gen.tolist(),
        "token_metrics": tok,
        "activation_teacher_force_vs_steered_ref": act_tf_vs_steered,
        "activation_teacher_force_vs_baseline_ref": act_tf_vs_baseline,
        "activation_free_generation_vs_steered_ref": act_free_vs_steered,
        "activation_free_generation_vs_baseline_ref": act_free_vs_baseline,
    }
    activation_tensors = {
        "copy_prompt_len": int(copy_len),
        "generated_len": int(hyp_free_len),
        "generated_ids": hyp_gen.clone(),
        "teacher_force_generated_span": hyp_tf_slice.clone(),
        "free_generation_generated_span": hyp_free_slice.clone(),
    }
    return metrics, activation_tensors


def _json_safe(result: Any) -> Any:
    return json.loads(json.dumps(result, default=str))


def run_experiment(config: Config, instructions: Optional[List[str]], args: argparse.Namespace):
    set_seed(config.seed)
    model_alias = config.model_id.split("/")[-1]
    model_output_dir = os.path.join(config.output_dir, model_alias, "copying_baseline")
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"Loading model: {config.model_id}")
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

    if instructions is None:
        instructions = (
            TEST_INSTRUCTIONS if config.steering_type == "refusal" else EVIL_TEST_INSTRUCTIONS
        )

    icl_ns = [int(x.strip()) for x in args.icl_ns.split(",") if x.strip()]
    results = []
    activation_records: List[Dict[str, Any]] = []
    for instr in instructions:
        print(f"\n{'='*60}\nInstruction: {instr[:100]}...")
        row, act_rec = run_single_instruction(
            model,
            tokenizer,
            tokenize_fn,
            instr,
            steering_config,
            inversion_layer,
            config.max_new_tokens,
            icl_ns=icl_ns,
            run_explicit=not args.skip_explicit,
            return_activation_tensors=args.save_activation_tensors,
        )
        results.append(row)
        if act_rec is not None:
            activation_records.append(act_rec)
        gap = row.get("reference_activation_gap_steered_vs_baseline", {})
        if gap.get("overlap_len"):
            print(
                f"  [ref gap on instr+steered text] steered vs baseline MSE={gap.get('mse_mean'):.6f}"
            )
        for name, cond in row.get("conditions", {}).items():
            tm = cond.get("token_metrics", {})
            tfs = cond.get("activation_teacher_force_vs_steered_ref", {})
            tfb = cond.get("activation_teacher_force_vs_baseline_ref", {})
            frs = cond.get("activation_free_generation_vs_steered_ref", {})
            frb = cond.get("activation_free_generation_vs_baseline_ref", {})
            print(
                f"  [{name}] exact_token_match={tm.get('exact_match')} "
                f"prefix={tm.get('prefix_match_len')}/{tm.get('ref_len')} "
                f"tf_mse_steered={tfs.get('mse_mean'):.6f} tf_mse_baseline={tfb.get('mse_mean'):.6f} "
                f"free_mse_steered={frs.get('mse_mean'):.6f} free_mse_baseline={frb.get('mse_mean'):.6f}"
            )

    tag = f"{config.steering_type}_{config.steering_method}_coeff_{config.steering_coeff}"
    results_path = os.path.join(model_output_dir, f"copying_baseline_{tag}.json")
    with open(results_path, "w") as f:
        json.dump(_json_safe(results), f, indent=2)

    optional_pkl = os.path.join(model_output_dir, f"copying_baseline_{tag}_activations.pkl")
    if args.save_activation_tensors:
        payload = {
            "model_id": config.model_id,
            "steering_layer": layer,
            "inversion_layer": inversion_layer,
            "metadata": metadata,
            "records": activation_records,
        }
        with open(optional_pkl, "wb") as f:
            pickle.dump(payload, f)
        print(
            f"Saved activation tensors ({len(activation_records)} instructions) to {optional_pkl}"
        )

    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copying baseline: repeat steered generation")
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
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--icl-ns",
        type=str,
        default="",
        help="Comma-separated N values for in-context repetition count",
    )
    parser.add_argument("--skip-explicit", action="store_true", help="Only run ICL conditions")
    parser.add_argument(
        "--save-activation-tensors",
        action="store_true",
        default=True,
        help="Save a .pkl with activation tensors only (aligned by instruction / condition), not JSON metrics",
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
    if cli.max_new_tokens is not None:
        cfg.max_new_tokens = cli.max_new_tokens
    cfg.special_start_tokens = cfg._get_default_special_tokens()

    run_experiment(cfg, None, cli)

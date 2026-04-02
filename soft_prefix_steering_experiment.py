"""
Soft prefix tuning to match steered hidden states at prompt positions.

For each prefix length in {1, ..., max_prefix_len}:
1. Target: hidden states at inversion_layer when running the tokenized prompt with steering
   (same convention as experiment.py: inversion_layer = steering_layer + 1).
2. Optimize a continuous soft prefix prepended after optional fixed chat prefix tokens:
   sequence is [before_ids | soft_prefix | after_ids], where before/after split uses the
   same insertion heuristic as prompt_activation_alignment_experiment.
3. Loss: match steered activations on original positions i >= insertion_index (pred at i+p).

Reports best activation-space error per length, nearest-vocab projection distances for the
soft prefix, and generations: baseline, steered, soft-embedding prefix, discrete projected prefix.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from config import Config, TEST_INSTRUCTIONS, EVIL_TEST_INSTRUCTIONS
from model_utils import load_model, get_tokenize_fn, set_seed, get_input_device
from steering import (
    SteeringConfig,
    load_steering_direction,
    get_hidden_states_with_steering,
    steering_context,
)


def get_prefix_insertion_index(tokenizer, tokenize_fn) -> int:
    try:
        ids_empty = tokenize_fn(instructions=[""]).input_ids[0].tolist()
        ids_marker = tokenize_fn(instructions=["UNIQUE_MARKER_XYZ"]).input_ids[0].tolist()
    except Exception:
        return 0
    lcp = 0
    for a, b in zip(ids_empty, ids_marker):
        if a == b:
            lcp += 1
        else:
            break
    if lcp >= len(ids_empty) or lcp >= len(ids_marker):
        return 0
    return lcp


def build_stitched_ids(
    before: torch.Tensor,
    middle_ids: torch.Tensor,
    after: torch.Tensor,
) -> torch.Tensor:
    """Concatenate [before | middle | along batch dim 1]."""
    device = before.device
    middle_ids = middle_ids.to(device)
    return torch.cat([before, middle_ids, after], dim=1)


def nearest_token_ids_and_dists(
    virtual_embeddings: torch.Tensor,
    embedding_matrix: torch.Tensor,
    chunk: int = 4096,
) -> Tuple[List[int], List[float]]:
    """For each row in virtual_embeddings, nearest row in embedding_matrix (L2)."""
    vemb = virtual_embeddings.float()
    emat = embedding_matrix.float()
    best_ids: List[int] = []
    best_dists: List[float] = []
    for v in vemb:
        best_id = 0
        best_dist = float("inf")
        for start in range(0, emat.size(0), chunk):
            end = min(start + chunk, emat.size(0))
            dists = torch.norm(emat[start:end] - v.unsqueeze(0), dim=1)
            local_best = int(dists.argmin().item())
            d = float(dists[local_best].item())
            if d < best_dist:
                best_dist = d
                best_id = start + local_best
        best_ids.append(best_id)
        best_dists.append(best_dist)
    return best_ids, best_dists


def activation_match_loss(
    model: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    inversion_layer: int,
    target_slice: torch.Tensor,
    pred_start: int,
) -> torch.Tensor:
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    h = outputs.hidden_states[inversion_layer][0]
    span = target_slice.size(0)
    pred = h[pred_start : pred_start + span]
    return F.mse_loss(pred.float(), target_slice.float())


@torch.no_grad()
def eval_activation_l2(
    model: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    inversion_layer: int,
    target_slice: torch.Tensor,
    pred_start: int,
) -> float:
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    h = outputs.hidden_states[inversion_layer][0]
    span = target_slice.size(0)
    pred = h[pred_start : pred_start + span]
    return torch.norm(pred.float() - target_slice.float()).item()


def optimize_soft_prefix(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    steering_config: SteeringConfig,
    inversion_layer: int,
    prefix_len: int,
    insertion_index: int,
    *,
    steps: int,
    lr: float,
    device: torch.device,
    model_dtype: torch.dtype,
) -> Dict[str, Any]:
    """
    Optimize soft embeddings of length prefix_len placed at insertion_index.
    Matches steered hidden states for original positions insertion_index..L-1.
    """
    input_ids = input_ids.to(device)
    L = input_ids.size(1)
    if insertion_index > L:
        raise ValueError("insertion_index exceeds sequence length")
    n_match = L - insertion_index
    if n_match <= 0:
        raise ValueError("No positions to match after insertion_index")

    with torch.no_grad():
        steered_full = get_hidden_states_with_steering(
            model, input_ids, steering_config, inversion_layer
        )
        target_slice = steered_full[insertion_index : L].to(device).to(model_dtype)

    emb = model.get_input_embeddings()
    with torch.no_grad():
        before = emb(input_ids[:, :insertion_index]).detach()
        after = emb(input_ids[:, insertion_index:]).detach()

    p = prefix_len
    scale = float(emb.weight.float().std().clamp_min(1e-6))
    prefix = torch.randn(1, p, emb.embedding_dim, device=device, dtype=model_dtype)
    prefix = prefix * (0.1 * scale)
    prefix = torch.nn.Parameter(prefix)
    opt = torch.optim.Adam([prefix], lr=lr)

    model.train(False)
    best_mse = float("inf")
    best_prefix: Optional[torch.Tensor] = None

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        inputs_embeds = torch.cat([before, prefix, after], dim=1)
        attn = torch.ones(
            1, inputs_embeds.size(1), device=device, dtype=torch.long
        )
        loss = activation_match_loss(
            model, inputs_embeds, attn, inversion_layer, target_slice, insertion_index + p
        )
        loss.backward()
        opt.step()
        v = float(loss.detach().item())
        if v < best_mse:
            best_mse = v
            best_prefix = prefix.detach().clone()

    if best_prefix is None:
        best_prefix = prefix.detach().clone()

    inputs_embeds_final = torch.cat([before, best_prefix, after], dim=1)
    attn = torch.ones(1, inputs_embeds_final.size(1), device=device, dtype=torch.long)
    l2_total = eval_activation_l2(
        model,
        inputs_embeds_final,
        attn,
        inversion_layer,
        target_slice,
        insertion_index + p,
    )
    per_token = l2_total / max(n_match, 1)

    emb_w = emb.weight.detach()
    nearest_ids, nearest_dists = nearest_token_ids_and_dists(
        best_prefix.squeeze(0), emb_w
    )

    return {
        "prefix_len": p,
        "insertion_index": insertion_index,
        "n_match_positions": n_match,
        "best_mse": best_mse,
        "final_l2_total": l2_total,
        "final_l2_mean_per_token": per_token,
        "soft_prefix": best_prefix.detach().cpu(),
        "nearest_token_ids": nearest_ids,
        "nearest_token_l2_dists": nearest_dists,
        "mean_nearest_emb_l2": float(sum(nearest_dists) / max(len(nearest_dists), 1)),
    }


@torch.no_grad()
def generate_from_embed_prefix(
    model,
    tokenizer,
    before_emb: torch.Tensor,
    soft_prefix: torch.Tensor,
    after_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
    model_dtype: torch.dtype,
) -> str:
    emb = model.get_input_embeddings()
    after_emb = emb(after_ids.to(device))
    inputs_embeds = torch.cat([before_emb, soft_prefix.to(device, dtype=model_dtype), after_emb], dim=1)
    attn = torch.ones(1, inputs_embeds.size(1), device=device, dtype=torch.long)
    out = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    # New tokens only; prompt length in "virtual tokens" = inputs_embeds.size(1)
    gen = out[0, inputs_embeds.size(1) :]
    return tokenizer.decode(gen, skip_special_tokens=True)


@torch.no_grad()
def generate_from_discrete_prefix(
    model,
    tokenizer,
    before_ids: torch.Tensor,
    middle_ids: torch.Tensor,
    after_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    full = build_stitched_ids(before_ids.to(device), middle_ids.to(device), after_ids.to(device))
    out = model.generate(
        input_ids=full,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[0, full.size(1) :]
    return tokenizer.decode(gen, skip_special_tokens=True)


def run_instruction(
    model,
    tokenizer,
    tokenize_fn,
    instruction: str,
    steering_config: SteeringConfig,
    *,
    max_prefix_len: int,
    opt_steps: int,
    opt_lr: float,
    max_new_tokens: int,
    insertion_index: Optional[int],
) -> Dict[str, Any]:
    input_device = get_input_device(model)
    model_dtype = next(model.parameters()).dtype

    inputs = tokenize_fn(instructions=[instruction])
    input_ids = inputs.input_ids.to(input_device)
    attention_mask = inputs.attention_mask.to(input_device)
    inversion_layer = steering_config.layer + 1

    if insertion_index is None:
        insertion_index = get_prefix_insertion_index(tokenizer, tokenize_fn)

    with torch.no_grad():
        base_out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    baseline_gen = tokenizer.decode(
        base_out[0, input_ids.size(1) :], skip_special_tokens=True
    )

    with steering_context(model, steering_config):
        steer_out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    steered_gen = tokenizer.decode(
        steer_out[0, input_ids.size(1) :], skip_special_tokens=True
    )

    per_length: List[Dict[str, Any]] = []
    soft_cache: Dict[int, torch.Tensor] = {}

    for p in range(1, max_prefix_len + 1):
        print(f"  Optimizing soft prefix length {p}/{max_prefix_len}...")
        row = optimize_soft_prefix(
            model,
            input_ids,
            steering_config,
            inversion_layer,
            p,
            insertion_index,
            steps=opt_steps,
            lr=opt_lr,
            device=input_device,
            model_dtype=model_dtype,
        )
        soft_cache[p] = row["soft_prefix"]
        per_length.append(
            {
                "prefix_len": row["prefix_len"],
                "insertion_index": row["insertion_index"],
                "n_match_positions": row["n_match_positions"],
                "best_mse": row["best_mse"],
                "final_l2_total": row["final_l2_total"],
                "final_l2_mean_per_token": row["final_l2_mean_per_token"],
                "nearest_token_ids": row["nearest_token_ids"],
                "nearest_token_l2_dists": row["nearest_token_l2_dists"],
                "mean_nearest_emb_l2": row["mean_nearest_emb_l2"],
            }
        )

    best_idx = min(
        range(len(per_length)),
        key=lambda i: per_length[i]["final_l2_mean_per_token"],
    )
    best_p = per_length[best_idx]["prefix_len"]
    best_soft = soft_cache[best_p].to(input_device, dtype=model_dtype)

    emb = model.get_input_embeddings()
    with torch.no_grad():
        before_emb = emb(input_ids[:, :insertion_index])
        after_ids = input_ids[:, insertion_index:]
        before_ids = input_ids[:, :insertion_index]

    nearest_ids = per_length[best_idx]["nearest_token_ids"]
    middle_tensor = torch.tensor([nearest_ids], device=input_device, dtype=torch.long)

    soft_prefix_gen = generate_from_embed_prefix(
        model,
        tokenizer,
        before_emb,
        best_soft,
        after_ids,
        max_new_tokens,
        input_device,
        model_dtype,
    )
    discrete_prefix_gen = generate_from_discrete_prefix(
        model,
        tokenizer,
        before_ids,
        middle_tensor,
        after_ids,
        max_new_tokens,
        input_device,
    )

    nearest_text = tokenizer.decode(nearest_ids, skip_special_tokens=False)

    return {
        "instruction": instruction,
        "timestamp": datetime.now().isoformat(),
        "steering_layer": steering_config.layer,
        "inversion_layer": inversion_layer,
        "steering_method": steering_config.method,
        "steering_coeff": steering_config.coeff,
        "insertion_index": insertion_index,
        "seq_len": int(input_ids.size(1)),
        "per_prefix_length": per_length,
        "best_prefix_len_by_mean_l2": best_p,
        "generations": {
            "baseline": baseline_gen,
            "steered": steered_gen,
            "soft_prefix": soft_prefix_gen,
            "discrete_projected_prefix": discrete_prefix_gen,
        },
        "best_discrete_prefix_text": nearest_text,
        "best_discrete_prefix_ids": nearest_ids,
    }


def _strip_for_json(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _strip_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_for_json(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Soft prefix tuning to match steered prompt activations"
    )
    parser.add_argument("--model", type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--steering-type", type=str, default="refusal")
    parser.add_argument("--method", type=str, default="actadd")
    parser.add_argument("--coeff", type=float, default=-1.0)
    parser.add_argument("--direction", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-prefix-len", type=int, default=25)
    parser.add_argument("--opt-steps", type=int, default=300)
    parser.add_argument("--opt-lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Comma-separated instructions; default from config by steering type.",
    )
    parser.add_argument(
        "--insertion-index",
        type=int,
        default=-1,
        help="Where to insert soft prefix; -1 = auto (chat-aware).",
    )
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--add-special-tokens", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    if args.model:
        cfg.model_id = args.model
    cfg.device = args.device
    cfg.dtype = args.dtype
    cfg.load_in_4bit = args.load_in_4bit
    cfg.steering_type = args.steering_type
    cfg.steering_method = args.method
    cfg.steering_coeff = args.coeff
    cfg.max_new_tokens = args.max_new_tokens
    cfg.use_chat_template = not args.no_chat_template
    cfg.add_special_tokens = args.add_special_tokens
    if args.direction:
        cfg.direction_path = args.direction

    set_seed(args.seed)

    if args.instructions:
        instructions = [s.strip() for s in args.instructions.split(",") if s.strip()]
    else:
        instructions = (
            TEST_INSTRUCTIONS
            if cfg.steering_type == "refusal"
            else EVIL_TEST_INSTRUCTIONS
        )

    insertion_index: Optional[int]
    if args.insertion_index < 0:
        insertion_index = None
    else:
        insertion_index = args.insertion_index

    model_alias = cfg.model_id.split("/")[-1]
    out_dir = os.path.join(cfg.output_dir, model_alias, "soft_prefix_steering")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model: {cfg.model_id}")
    model, tokenizer = load_model(
        cfg.model_id, cfg.device, cfg.dtype, load_in_4bit=cfg.load_in_4bit
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenize_fn = get_tokenize_fn(
        tokenizer,
        use_chat_template=cfg.use_chat_template,
        add_special_tokens=cfg.add_special_tokens,
    )

    direction_path = cfg.get_direction_path()
    if not os.path.exists(direction_path):
        alt = cfg.get_direction_path(model_alias.lower())
        if os.path.exists(alt):
            direction_path = alt
        else:
            raise FileNotFoundError(f"Direction file not found: {direction_path}")

    direction, layer, _metadata = load_steering_direction(direction_path, cfg.device)
    direction = direction.to(get_input_device(model))
    steering_config = SteeringConfig(
        direction=direction,
        layer=layer,
        method=cfg.steering_method,
        coeff=cfg.steering_coeff,
        steering_type=cfg.steering_type,
    )
    print(f"Direction layer={layer}, coeff={cfg.steering_coeff}, path={direction_path}")

    results: List[Dict[str, Any]] = []
    for instr in instructions:
        print(f"\n{'='*60}\nInstruction: {instr[:100]}...\n{'='*60}")
        r = run_instruction(
            model,
            tokenizer,
            tokenize_fn,
            instr,
            steering_config,
            max_prefix_len=args.max_prefix_len,
            opt_steps=args.opt_steps,
            opt_lr=args.opt_lr,
            max_new_tokens=cfg.max_new_tokens,
            insertion_index=insertion_index,
        )
        results.append(r)

        print("\n--- Activation alignment (mean L2 per matched token) ---")
        for row in r["per_prefix_length"]:
            print(
                f"  len={row['prefix_len']:2d}  mse={row['best_mse']:.6f}  "
                f"l2_mean/token={row['final_l2_mean_per_token']:.6f}  "
                f"mean_emb_dist={row['mean_nearest_emb_l2']:.4f}"
            )
        print(f"\nBest prefix length (by mean token L2): {r['best_prefix_len_by_mean_l2']}")
        print("\n--- Generations ---")
        g = r["generations"]
        print(f"Baseline:   {g['baseline'][:500]}")
        print(f"Steered:    {g['steered'][:500]}")
        print(f"Soft pref:  {g['soft_prefix'][:500]}")
        print(f"Discrete:   {g['discrete_projected_prefix'][:500]}")
        print(f"Projected prefix tokens text: {r['best_discrete_prefix_text'][:200]!r}")

    out_path = os.path.join(
        out_dir,
        f"soft_prefix_results_{cfg.steering_type}_{cfg.steering_method}_coeff_{cfg.steering_coeff}.json",
    )
    with open(out_path, "w") as f:
        json.dump(_strip_for_json(results), f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

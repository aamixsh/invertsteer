"""
Utility to collect reconstructed prompt activations and align them with
baseline and steered activations.

This script:
1. Finds all experiment result JSONs for a given model.
2. For each experiment, loads the steering direction and configuration.
3. Recomputes **baseline**, **steered**, and **reconstructed** activations
   at the inversion layer for every instruction with a reconstructed prompt.
4. Produces a single pickle containing all three activations, token-aligned,
   across all experiments for that model.

Usage (minimal):

python -m invertsteer.collect_reconstructed_activations \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --device cuda:0

This will:
- Look under `Config().output_dir/<model_alias>/` for
  `experiment_results_*.json`,
- Derive steering settings from filenames and config,
- Load directions automatically,
- And save a combined pickle next to the experiment outputs.
"""

import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Optional

import torch

from config import Config
from model_utils import load_model, get_tokenize_fn
from steering import SteeringConfig, load_steering_direction, get_hidden_states_with_steering
from inversion import extract_hidden_states


def _load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def _load_pickle(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _discover_experiment_jsons(model_id: str, output_dir: str) -> List[str]:
    """
    Find all experiment JSON files for a given model.
    Includes:
      - experiment_results_*.json (single-coeff experiments)
      - steering_invert_results_*.json (coeff_sweep experiments)
    """
    model_alias = model_id.split("/")[-1]
    model_output_dir = os.path.join(output_dir, model_alias)
    if not os.path.isdir(model_output_dir):
        raise FileNotFoundError(f"Model output dir not found: {model_output_dir}")

    files: List[str] = []
    for name in os.listdir(model_output_dir):
        if not name.endswith(".json"):
            continue
        if name.startswith("experiment_results_") or name.startswith(
            "steering_invert_results_"
        ):
            files.append(os.path.join(model_output_dir, name))
    files.sort()
    if not files:
        raise FileNotFoundError(
            f"No experiment_results_*.json files found in {model_output_dir}"
        )
    return files


def _parse_experiment_filename(path: str) -> Dict[str, Any]:
    """
    Parse steering_type, steering_method, coeff from an experiment filename.

    Supported patterns:
      - experiment_results_{steering_type}_{steering_method}_coeff_{coeff}.json
      - steering_invert_results_{steering_type}_{steering_method}_coeff_{coeff}.json
    """
    name = os.path.basename(path)
    if name.startswith("experiment_results_") and name.endswith(".json"):
        kind = "single"
        core = name[len("experiment_results_") : -len(".json")]
    elif name.startswith("steering_invert_results_") and name.endswith(".json"):
        kind = "coeff_sweep"
        core = name[len("steering_invert_results_") : -len(".json")]
    else:
        raise ValueError(f"Unexpected experiment filename: {name}")
    # Split "{steering_type}_{method}_coeff_{coeff}"
    if "_coeff_" not in core:
        raise ValueError(f"Unexpected experiment filename format: {name}")
    left, coeff_str = core.split("_coeff_", 1)
    if "_" not in left:
        raise ValueError(f"Unexpected experiment filename format: {name}")
    steering_type, steering_method = left.split("_", 1)
    try:
        coeff = float(coeff_str)
    except ValueError:
        raise ValueError(f"Could not parse coeff from filename: {name}")
    return {
        "kind": kind,
        "steering_type": steering_type,
        "steering_method": steering_method,
        "steering_coeff": coeff,
    }


def collect_reconstructed_prompt_activations_for_experiment(
    results_path: str,
    model,
    tokenizer,
    tokenize_fn,
    steering_config: SteeringConfig,
    experiment_kind: str,
    inversion_layer: int,
) -> List[Dict[str, Any]]:
    """
    For a single experiment JSON, recompute baseline, steered, and reconstructed
    activations and return aligned triples per instruction.

    Args:
        results_path: Path to experiment_results_*.json
        model: Loaded model
        tokenizer: Tokenizer
        tokenize_fn: Instruction tokenizer (matching original experiment)
        steering_config: Steering configuration for this experiment
        experiment_kind: "single" or "coeff_sweep"
        inversion_layer: Layer index at which to extract activations

    Returns:
        List of dicts with aligned activations for each instruction that has a
        reconstructed prompt.
    """
    results = _load_json(results_path)

    device = next(model.parameters()).device

    out_results: List[Dict[str, Any]] = []

    for r in results:
        if "error" in r:
            continue

        instruction = r.get("instruction")
        if instruction is None:
            continue

        # Recompute baseline and steered activations from the model.
        inputs = tokenize_fn(instructions=[instruction])
        input_ids = inputs.input_ids.to(device)

        baseline_acts = extract_hidden_states(
            input_ids, model, inversion_layer
        ).detach().cpu()

        steered_acts = get_hidden_states_with_steering(
            model, input_ids, steering_config, inversion_layer
        ).detach().cpu()

        reconstructed_text = r.get("steered_inversion", {}).get("reconstructed_text")
        if reconstructed_text is None:
            continue

        recon_ids = torch.tensor(tokenizer.encode(reconstructed_text, add_special_tokens=False), dtype=torch.long).unsqueeze(0).to(device)
        
        recon_acts = extract_hidden_states(
            recon_ids, model, inversion_layer
        ).detach().cpu()

        seq_len = baseline_acts.size(0)
        recon_seq_len = recon_acts.size(0)
        steered_seq_len = steered_acts.size(0)

        # Align all activation sequences by truncating to the minimum length so
        # that token i corresponds across baseline / steered / reconstructed.
        min_len = min(seq_len, recon_seq_len, steered_seq_len)
        baseline_aligned = baseline_acts[:min_len].float().cpu().numpy()
        steered_aligned = steered_acts[:min_len].float().cpu().numpy()
        recon_aligned = recon_acts[:min_len].float().cpu().numpy()

        out_results.append(
            {
                "instruction": instruction,
                "baseline_activations": baseline_aligned,
                "steered_activations": steered_aligned,
                "reconstructed_activations": recon_aligned,
                "original_ids": r.get("original_ids"),
                "reconstructed_ids": recon_ids,
                "seq_len": seq_len,
                "steered_seq_len": steered_seq_len,
                "reconstructed_seq_len": recon_seq_len,
                "aligned_seq_len": min_len,
            }
        )

    return out_results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect reconstructed prompt activations and align them with "
            "baseline and steered activations."
        )
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output pickle path. If omitted, a default path is chosen under "
            "Config().output_dir/<model_alias>/."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model ID to load (defaults to Config().model_id).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:2",
        help="Device to use (defaults to Config().device).",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="NF4 BitsAndBytes load (GPU + bitsandbytes).",
    )
    args = parser.parse_args()

    config = Config()
    if args.model:
        config.model_id = args.model
    if args.device:
        config.device = args.device
    config.load_in_4bit = args.load_in_4bit

    print(f"Loading model: {config.model_id} on device {config.device}")
    model, tokenizer = load_model(
        config.model_id, config.device, config.dtype, load_in_4bit=config.load_in_4bit
    )
    tokenize_fn = get_tokenize_fn(
        tokenizer,
        use_chat_template=config.use_chat_template,
        add_special_tokens=config.add_special_tokens,
    )

    # Discover experiments for this model.
    experiment_jsons = _discover_experiment_jsons(
        model_id=config.model_id, output_dir=config.output_dir
    )
    print(f"Found {len(experiment_jsons)} experiment result files.")


    for results_path in experiment_jsons:
        all_results: List[Dict[str, Any]] = []
        meta = _parse_experiment_filename(results_path)
        experiment_kind = meta["kind"]
        steering_type = meta["steering_type"]
        steering_method = meta["steering_method"]
        steering_coeff = meta["steering_coeff"]

        # Construct a temporary config to locate the steering direction file.
        exp_config = Config()
        exp_config.model_id = config.model_id
        exp_config.device = config.device
        exp_config.steering_type = steering_type
        exp_config.steering_method = steering_method
        exp_config.steering_coeff = steering_coeff

        direction_path = exp_config.get_direction_path()
        if not os.path.exists(direction_path):
            raise FileNotFoundError(
                f"Direction file not found for experiment {results_path}:\n"
                f"  expected at: {direction_path}"
            )

        direction, layer, _ = load_steering_direction(direction_path, config.device)
        inversion_layer = layer + 1
        steering_config = SteeringConfig(
            direction=direction,
            layer=layer,
            method=steering_method,
            coeff=steering_coeff,
            steering_type=steering_type,
        )

        print(
            f"Processing {os.path.basename(results_path)} "
            f"(steering_type={steering_type}, method={steering_method}, coeff={steering_coeff}) "
            f"at layer={layer}, inversion_layer={inversion_layer}"
        )

        exp_results = collect_reconstructed_prompt_activations_for_experiment(
            results_path=results_path,
            model=model,
            tokenizer=tokenizer,
            tokenize_fn=tokenize_fn,
            steering_config=steering_config,
            experiment_kind=experiment_kind,
            inversion_layer=inversion_layer,
        )

        for item in exp_results:
            item["experiment_kind"] = experiment_kind
            item["steering_type"] = steering_type
            item["steering_method"] = steering_method
            item["steering_coeff"] = steering_coeff
            item["steering_layer"] = layer
            item["inversion_layer"] = inversion_layer
            all_results.append(item)

        model_alias = config.model_id.split("/")[-1]
        model_output_dir = os.path.join(config.output_dir, model_alias)
        os.makedirs(model_output_dir, exist_ok=True)

        if experiment_kind == "single":
            prefix = "experiment_activations"
        elif experiment_kind == "coeff_sweep":
            prefix = "steering_activations"

        if args.output is not None:
            output_path = args.output
        else:
            output_path = os.path.join(
                model_output_dir, f"{prefix}_{exp_config.steering_type}_{exp_config.steering_method}_coeff_{exp_config.steering_coeff}_with_recon.pkl"
            )

        out_data = {
            "model_id": config.model_id,
            "results": all_results,
        }

        print(
            f"Collected aligned activations for {len(all_results)} instructions "
            f"across {len(experiment_jsons)} experiments."
        )
        print(
            f"Saving combined activations (baseline, steered, reconstructed) to: {output_path}"
        )

        with open(output_path, "wb") as f:
            pickle.dump(out_data, f)


if __name__ == "__main__":
    main()


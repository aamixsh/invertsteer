"""
Coefficient sweep experiment for inverting steered activations.

This script runs the steered activation inversion experiment across multiple
steering coefficients to analyze how coefficient strength affects inversion success.
"""

import os
import json
import torch
from typing import List, Dict, Any, Optional
from datetime import datetime
import pickle

from config import Config, TEST_INSTRUCTIONS, EVIL_TEST_INSTRUCTIONS
from model_utils import load_model, get_tokenize_fn, set_seed, get_num_layers
from steering import (
    SteeringConfig,
    load_steering_direction,
    compare_generations,
    get_hidden_states_iterative_with_steering,
)
from inversion import (
    extract_hidden_states_iterative,
    inversion_attack,
    inversion_attack_with_target,
    compute_activation_mse,
    print_top_k_tokens,
)


def run_single_experiment_with_coeff(
    model,
    tokenizer,
    tokenize_fn,
    instruction: str,
    steering_config: SteeringConfig,
    coeff: float,
    lr: float = 1.0,
    seed: int = 42,
    max_new_tokens: int = 64,
    linear: bool = False,
    cmaes: bool = False,
    special_start_tokens: Optional[List[int]] = None,
    continue_on_failure: bool = False,
    top_k: int = 10,
    batch_size: int = 256,
    n_gpus: int = 1,
    shuffle_candidates: bool = True,
    cmaes_max_iterations: int = 1000,
    cmaes_stdev_init: float = 0.1,
) -> Dict[str, Any]:
    """
    Run experiment for a single instruction and coefficient.

    Args:
        model: The model (with original layer norm)
        tokenizer: The tokenizer
        tokenize_fn: Function to tokenize instructions
        instruction: The instruction to test
        steering_config: Base configuration for the steering (coeff will be overridden)
        coeff: The steering coefficient to use
        lr: Learning rate for inversion
        seed: Random seed
        max_new_tokens: Max tokens for generation
        linear: Whether to invert using baseline linear search
        special_start_tokens: Special tokens to prepend
        continue_on_failure: Continue with ground truth when inversion fails
        top_k: Number of top candidate tokens to track
        batch_size: Candidate batch size for baseline search
        n_gpus: Number of GPUs to use for batched evaluation
        shuffle_candidates: Whether to shuffle candidate order in baseline

    Returns a dict with experiment results for this coefficient.
    """
    # Override the coefficient in steering config
    steering_config = SteeringConfig(
        direction=steering_config.direction,
        layer=steering_config.layer,
        method=steering_config.method,
        coeff=coeff,
        steering_type=steering_config.steering_type,
    )

    result = {
        "instruction": instruction,
        "timestamp": datetime.now().isoformat(),
        "steering_layer": steering_config.layer,
        "steering_method": steering_config.method,
        "steering_coeff": coeff,
    }

    # Tokenize the instruction
    inputs = tokenize_fn(instructions=[instruction])
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    seq_len = input_ids.size(1)
    result["seq_len"] = seq_len
    result["original_ids"] = input_ids[0].tolist()

    print(f"\n{'='*60}")
    print(f"Instruction: {instruction[:80]}...")
    print(f"Tokenized length: {seq_len}")
    print(f"Steering: layer={steering_config.layer}, method={steering_config.method}, coeff={coeff}")
    print(f"{'='*60}")

    # Inversion layer = steering layer (the layer where actadd is applied)
    inversion_layer = steering_config.layer + 1

    # Step 1: Compare baseline vs steered generations
    print("\n[Step 1] Comparing baseline vs steered generations...")
    baseline_gen, steered_gen = compare_generations(
        model, tokenizer, [instruction], steering_config,
        tokenize_fn, max_new_tokens
    )
    result["baseline_generation"] = baseline_gen[0]
    result["steered_generation"] = steered_gen[0]

    print(f"Baseline: {baseline_gen[0][:100]}...")
    print(f"Steered:  {steered_gen[0][:100]}...")

    # Step 2: Extract activations at the steering layer
    print(f"\n[Step 2] Extracting activations at layer {inversion_layer}...")

    baseline_acts = extract_hidden_states_iterative(
        input_ids, model, inversion_layer
    )

    steered_acts = get_hidden_states_iterative_with_steering(
        model, input_ids, steering_config, inversion_layer
    )

    act_diff = torch.norm(baseline_acts - steered_acts).item()
    act_diff_per_token = torch.norm(baseline_acts - steered_acts, dim=1).mean().item()
    result["activation_diff_total"] = act_diff
    result["activation_diff_per_token"] = act_diff_per_token

    # Store activations for analysis
    result["baseline_activations"] = baseline_acts.cpu()  # Store as tensor for later analysis
    result["steered_activations"] = steered_acts.cpu()   # Store as tensor for later analysis

    print(f"Activation difference (L2): {act_diff:.4f}")
    print(f"Per-token difference: {act_diff_per_token:.4f}")

    # Step 4: Steered inversion
    print("\n[Step 4] Inverting steered activations...")

    steered_match, steered_time, steered_recon_ids, steered_times, steered_inv_result = inversion_attack_with_target(
        steered_acts, model, tokenizer, inversion_layer, lr, seed,
        verbose=True, original_ids=input_ids, linear=True, cmaes=False,
        special_start_tokens=special_start_tokens,
        continue_on_failure=continue_on_failure,
        top_k=top_k, batch_size=batch_size, n_gpus=n_gpus, shuffle_candidates=shuffle_candidates,
        cmaes_max_iterations=cmaes_max_iterations,
        cmaes_stdev_init=cmaes_stdev_init,
    )

    result["steered_inversion"] = {
        "match_original": steered_match,
        "time": steered_time,
        "reconstructed_ids": steered_recon_ids,
        "failed_positions": steered_inv_result.failed_positions if steered_inv_result else [],
    }

    # Store top-k tokens for analysis
    if steered_inv_result and steered_inv_result.top_k_per_position:
        result["steered_inversion"]["top_k_per_position"] = [
            [(tid, dist) for tid, dist in pos_top_k]
            for pos_top_k in steered_inv_result.top_k_per_position
        ]

    if steered_recon_ids:
        steered_recon_text = tokenizer.decode(steered_recon_ids)
        result["steered_inversion"]["reconstructed_text"] = steered_recon_text

        recon_ids_tensor = torch.tensor(steered_recon_ids).unsqueeze(0).to(model.device)
        steered_mse = compute_activation_mse(
            model, recon_ids_tensor, steered_acts, inversion_layer
        )
        result["steered_inversion"]["mse_to_target"] = steered_mse

        baseline_mse_from_steered = compute_activation_mse(
            model, recon_ids_tensor, baseline_acts, inversion_layer
        )
        result["steered_inversion"]["mse_to_baseline"] = baseline_mse_from_steered

        # Step 5: Test if reconstructed prompt has same behavior
        print("\n[Step 5] Testing reconstructed prompt...")

        recon_inputs = tokenize_fn(instructions=[steered_recon_text])
        recon_input_ids = recon_inputs.input_ids.to(model.device)
        recon_attention_mask = recon_inputs.attention_mask.to(model.device)

        with torch.no_grad():
            recon_outputs = model.generate(
                input_ids=recon_input_ids,
                attention_mask=recon_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        recon_gen_tokens = recon_outputs[0, recon_input_ids.size(1):]
        recon_generation = tokenizer.decode(recon_gen_tokens)

        result["reconstructed_prompt_generation"] = recon_generation
        print(f"Reconstructed prompt generation: {recon_generation[:100]}...")

    # Print top-k
    if steered_inv_result:
        print_top_k_tokens(steered_inv_result, tokenizer, result["original_ids"])

    return result


def run_coeff_sweep_experiment(
    config: Config,
    coefficients: List[float],
    instructions: Optional[List[str]] = None
):
    """Run the coefficient sweep experiment."""
    print("="*60)
    print("COEFFICIENT SWEEP: INVERTING STEERED ACTIVATIONS")
    print(f"Model: {config.model_id}")
    print(f"Coefficients: {coefficients}")
    print("="*60)

    set_seed(config.seed)

    # Create output directory structure
    model_alias = config.model_id.split('/')[-1]
    model_output_dir = os.path.join(config.output_dir, model_alias)
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"Results will be saved to: {model_output_dir}")

    # Load model
    print(f"\nLoading model: {config.model_id}")
    model, tokenizer = load_model(
        config.model_id, config.device, config.dtype, load_in_4bit=config.load_in_4bit
    )
    tokenize_fn = get_tokenize_fn(tokenizer, use_chat_template=config.use_chat_template, add_special_tokens=config.add_special_tokens)

    n_layers = get_num_layers(model)
    print(f"Model has {n_layers} layers")
    print(f"Use chat template: {config.use_chat_template}")

    # Step 1: Load steering direction
    print("\n" + "="*60)
    print("STEP 1: LOADING STEERING DIRECTION")
    print("="*60)

    direction_path = config.get_direction_path()
    if not os.path.exists(direction_path):
        raise FileNotFoundError(
            f"Direction file not found: {direction_path}\n"
            f"Please run the refusal_direction pipeline first:\n"
            f"  cd {config.refusal_dir_path}\n"
            f"  python -m pipeline.run_pipeline --model_path {config.model_id}"
        )

    direction, layer, metadata = load_steering_direction(direction_path, config.device)
    print(f"Loaded direction from: {direction_path}")
    print(f"Steering layer: {layer}")
    print(f"Direction norm: {direction.norm().item():.4f}")
    print(f"Metadata: {metadata}")

    base_steering_config = SteeringConfig(
        direction=direction,
        layer=layer,
        method=config.steering_method,
        coeff=1.0,  # Will be overridden for each coefficient
        steering_type=config.steering_type,
    )

    # Step 2: Run experiments for each coefficient
    print("\n" + "="*60)
    print("STEP 2: RUNNING COEFFICIENT SWEEP")
    print("="*60)

    if instructions is None:
        if config.steering_type == "refusal":
            instructions = TEST_INSTRUCTIONS
        elif config.steering_type == "persona":
            instructions = EVIL_TEST_INSTRUCTIONS
        else:
            raise ValueError(f"Unknown steering type: {config.steering_type}")

    all_results = {}
    model_alias = config.model_id.split('/')[-1]

    for coeff in coefficients:
        print(f"\n{'*'*60}")
        print(f"RUNNING EXPERIMENTS FOR COEFFICIENT: {coeff}")
        print(f"{'*'*60}")

        coeff_results_path = os.path.join(
            model_output_dir,
            f"steering_invert_results_{config.steering_type}_{config.steering_method}_coeff_{coeff}.json"
        )
        activations_path = os.path.join(
            model_output_dir,
            f"steering_activations_{config.steering_type}_{config.steering_method}_coeff_{coeff}.pkl"
        )

        # Load existing results for this coefficient (resume)
        existing_by_instr = {}
        act_by_instr = {}
        if os.path.exists(coeff_results_path):
            with open(coeff_results_path, 'r') as f:
                existing_results = json.load(f)
            existing_by_instr = {r["instruction"]: r for r in existing_results}
        if os.path.exists(activations_path):
            with open(activations_path, 'rb') as f:
                activations_data_loaded = pickle.load(f)
            act_by_instr = {item["instruction"]: item for item in activations_data_loaded.get("results", [])}
        processed = {instr for instr in existing_by_instr if instr in act_by_instr}
        if processed:
            print(f"Resuming coeff {coeff}: {len(processed)} instructions already done, {len(instructions) - len(processed)} to run.")

        def save_coeff_results(results_list):
            results_for_json = []
            activations_data = {
                "coefficient": coeff,
                "steering_layer": layer,
                "inversion_layer": layer + 1,
                "results": []
            }
            for result in results_list:
                result_copy = result.copy()
                baseline_acts = result_copy.pop("baseline_activations", None)
                steered_acts = result_copy.pop("steered_activations", None)
                if baseline_acts is not None and steered_acts is not None:
                    activations_data["results"].append({
                        "instruction": result["instruction"],
                        "baseline_activations": baseline_acts,
                        "steered_activations": steered_acts,
                        "original_ids": result.get("original_ids"),
                        "seq_len": result.get("seq_len")
                    })
                results_for_json.append(result_copy)
            with open(coeff_results_path, 'w') as f:
                json.dump(results_for_json, f, indent=2, default=str)
            with open(activations_path, 'wb') as f:
                pickle.dump(activations_data, f)

        results = []
        for instruction in instructions:
            if instruction in processed:
                r = existing_by_instr[instruction].copy()
                r["baseline_activations"] = act_by_instr[instruction]["baseline_activations"]
                r["steered_activations"] = act_by_instr[instruction]["steered_activations"]
                results.append(r)
                continue
            try:
                result = run_single_experiment_with_coeff(
                    model, tokenizer, tokenize_fn,
                    instruction, base_steering_config, coeff,
                    lr=config.learning_rate, seed=config.seed,
                    max_new_tokens=config.max_new_tokens,
                    linear=config.linear,
                    cmaes=config.cmaes,
                    special_start_tokens=config.special_start_tokens,
                    continue_on_failure=config.continue_on_failure,
                    top_k=config.top_k,
                    batch_size=config.batch_size,
                    n_gpus=config.n_gpus,
                    shuffle_candidates=config.shuffle_candidates,
                    cmaes_max_iterations=config.cmaes_max_iterations,
                    cmaes_stdev_init=config.cmaes_stdev_init,
                )
                results.append(result)
            except Exception as e:
                import traceback
                print(f"Error processing instruction with coeff {coeff}: {e}")
                traceback.print_exc()
                results.append({
                    "instruction": instruction,
                    "steering_coeff": coeff,
                    "error": str(e),
                })
            save_coeff_results(results)

        all_results[str(coeff)] = results

        # Summary for this coefficient
        print(f"\n{'='*40} SUMMARY FOR COEFF {coeff} {'='*40}")

        n_steered_match = sum(1 for r in results if r.get("steered_inversion", {}).get("match_original", False))
        n_steered_failed = sum(1 for r in results if r.get("steered_inversion", {}).get("failed_positions", []))

        print(f"Total experiments: {len(results)}")
        print(f"Steered matches original: {n_steered_match}/{len(results)}")
        print(f"Steered with failures: {n_steered_failed}/{len(results)}")
        print(f"Steering invert results saved to: {coeff_results_path}")
        print(f"Activations saved to: {activations_path}")

    # Step 3: Save comprehensive results
    print("\n" + "="*60)
    print("STEP 3: SAVING COMPREHENSIVE RESULTS")
    print("="*60)

    # Create baseline activations file (same for all coefficients)
    baseline_activations_data = {
        "model_id": config.model_id,
        "steering_type": config.steering_type,
        "steering_layer": layer,
        "inversion_layer": layer + 1,
        "instructions": instructions,
        "results": []
    }

    # Extract baseline activations from the first coefficient's results
    if all_results:
        first_coeff = list(all_results.keys())[0]
        first_results = all_results[first_coeff]
        for result in first_results:
            baseline_activations_data["results"].append({
                "instruction": result["instruction"],
                "baseline_activations": result["baseline_activations"],
                "original_ids": result["original_ids"],
                "seq_len": result["seq_len"]
            })

    baseline_path = os.path.join(model_output_dir, f"baseline_activations_{config.steering_type}.pkl")
    with open(baseline_path, 'wb') as f:
        pickle.dump(baseline_activations_data, f)

    # Strip activations for JSON (keep comprehensive results serializable)
    all_results_for_json = {}
    for k, result_list in all_results.items():
        all_results_for_json[k] = []
        for r in result_list:
            r_copy = r.copy()
            r_copy.pop("baseline_activations", None)
            r_copy.pop("steered_activations", None)
            all_results_for_json[k].append(r_copy)

    comprehensive_results = {
        "experiment_metadata": {
            "model_id": config.model_id,
            "steering_type": config.steering_type,
            "steering_method": config.steering_method,
            "steering_layer": layer,
            "coefficients_tested": coefficients,
            "instructions_tested": instructions,
            "timestamp": datetime.now().isoformat(),
        },
        "results": all_results_for_json
    }

    comprehensive_path = os.path.join(
        model_output_dir,
        f"comprehensive_results_{config.steering_type}_{config.steering_method}.json"
    )
    with open(comprehensive_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)

    print(f"Baseline activations saved to: {baseline_path}")
    print(f"Comprehensive results saved to: {comprehensive_path}")

    # Step 4: Generate summary statistics
    print("\n" + "="*60)
    print("STEP 4: GENERATING SUMMARY STATISTICS")
    print("="*60)

    summary_stats = {}
    for coeff in coefficients:
        coeff_str = str(coeff)
        results = all_results[coeff_str]

        stats = {
            "total_experiments": len(results),
            "steered_matches_original": sum(1 for r in results if r.get("steered_inversion", {}).get("match_original", False)),
            "steered_with_failures": sum(1 for r in results if r.get("steered_inversion", {}).get("failed_positions", [])),
            "avg_activation_diff_total": sum(r.get("activation_diff_total", 0) for r in results) / len(results),
            "avg_activation_diff_per_token": sum(r.get("activation_diff_per_token", 0) for r in results) / len(results),
        }

        # Add MSE stats if available
        mse_to_target = [r.get("steered_inversion", {}).get("mse_to_target") for r in results if r.get("steered_inversion", {}).get("mse_to_target") is not None]
        mse_to_baseline = [r.get("steered_inversion", {}).get("mse_to_baseline") for r in results if r.get("steered_inversion", {}).get("mse_to_baseline") is not None]

        if mse_to_target:
            stats["avg_mse_to_target"] = sum(mse_to_target) / len(mse_to_target)
        if mse_to_baseline:
            stats["avg_mse_to_baseline"] = sum(mse_to_baseline) / len(mse_to_baseline)

        summary_stats[coeff_str] = stats

        print(f"Coefficient {coeff}:")
        print(f"  Matches: {stats['steered_matches_original']}/{stats['total_experiments']}")
        print(f"  Failures: {stats['steered_with_failures']}/{stats['total_experiments']}")
        if "avg_mse_to_target" in stats:
            print(f" Avg MSE to target: {stats['avg_mse_to_target']:.4f}")
        if "avg_mse_to_baseline" in stats:
            print(f" Avg MSE to baseline: {stats['avg_mse_to_baseline']:.4f}")
        print()

    summary_path = os.path.join(
        model_output_dir,
        f"summary_statistics_{config.steering_type}_{config.steering_method}.json"
    )
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)

    print(f"Summary statistics saved to: {summary_path}")
    print(f"\nAll results saved in: {model_output_dir}")

    return all_results, summary_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run coefficient sweep experiment for steered activation inversion")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model to use (default: meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--device", type=str, default="cuda:3", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate for inversion")
    parser.add_argument("--direction", type=str, default=None, help="Path to direction.pt")
    parser.add_argument("--method", type=str, default="actadd", help="Steering method: actadd or ablation")
    parser.add_argument("--coefficients", type=str, default="1.0",
                        help="Comma-separated list of coefficients to test")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Do not use chat template")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="Continue with ground truth when inversion fails")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top candidates to track")
    parser.add_argument("--add-special-tokens", action="store_true",
                        help="Add special tokens (like BOS token for Llama)")
    parser.add_argument("--steering-type", type=str, default="persona", help="Steering type: refusal or persona")
    parser.add_argument("--batch-size", type=int, default=1024, help="Candidate batch size for baseline search")
    parser.add_argument("--n-gpus", type=int, default=1, help="Number of GPUs to use for batched evaluation")
    parser.add_argument("--shuffle-candidates", action="store_true",
                        help="Shuffle candidate order in baseline")
    parser.add_argument("--cmaes", action="store_true",
                        help="Use CMA-ES evolutionary search instead of gradient-based method")
    parser.add_argument("--linear", action="store_true", help="Invert using baseline linear search")
    parser.add_argument("--cmaes-max-iterations", type=int, default=1000,
                        help="Maximum iterations for CMA-ES")
    parser.add_argument("--cmaes-stdev-init", type=float, default=1.0,
                        help="Initial standard deviation for CMA-ES")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="NF4 BitsAndBytes load (GPU + bitsandbytes).",
    )

    args = parser.parse_args()

    # Parse coefficients
    coefficients = [float(x.strip()) for x in args.coefficients.split(",")]

    config = Config()
    config.linear = args.linear
    config.steering_type = args.steering_type
    config.model_id = args.model
    config.device = args.device
    config.seed = args.seed
    config.learning_rate = args.lr
    if args.direction:
        config.direction_path = args.direction
    config.steering_method = args.method
    config.use_chat_template = not args.no_chat_template
    config.add_special_tokens = args.add_special_tokens
    config.continue_on_failure = args.continue_on_failure
    config.top_k = args.top_k
    config.batch_size = args.batch_size
    config.n_gpus = args.n_gpus
    config.shuffle_candidates = args.shuffle_candidates
    config.cmaes = args.cmaes
    config.cmaes_max_iterations = args.cmaes_max_iterations
    config.cmaes_stdev_init = args.cmaes_stdev_init
    config.load_in_4bit = args.load_in_4bit

    # Re-compute special tokens after setting use_chat_template
    config.special_start_tokens = config._get_default_special_tokens()

    run_coeff_sweep_experiment(config, coefficients)
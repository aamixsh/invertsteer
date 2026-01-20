"""
Main experiment script for inverting steered activations.

This script:
1. Loads a steering direction from refusal_direction pipeline output
2. For a set of test prompts:
   a. Compares baseline vs steered generations (using original model with layer norm)
   b. Extracts baseline and steered activations at the steering layer
   c. Attempts to invert both back to prompts
3. Evaluates and compares the results
"""

import os
import json
import torch
from typing import List, Dict, Any, Optional
from datetime import datetime

from config import Config, TEST_INSTRUCTIONS
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


def run_single_experiment(
    model,
    tokenizer,
    tokenize_fn,
    instruction: str,
    steering_config: SteeringConfig,
    lr: float = 1.0,
    seed: int = 42,
    max_new_tokens: int = 64,
    special_start_tokens: Optional[List[int]] = None,
    continue_on_failure: bool = False,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Run experiment for a single instruction.
    
    Args:
        model: The model (with original layer norm)
        tokenizer: The tokenizer
        tokenize_fn: Function to tokenize instructions
        instruction: The instruction to test
        steering_config: Configuration for the steering
        lr: Learning rate for inversion
        seed: Random seed
        max_new_tokens: Max tokens for generation
        special_start_tokens: Special tokens to prepend
        continue_on_failure: Continue with ground truth when inversion fails
        top_k: Number of top candidate tokens to track
    
    Returns a dict with:
    - instruction: The original instruction
    - original_ids: Original token IDs
    - baseline_generation: Model output without steering
    - steered_generation: Model output with steering
    - baseline_inversion: Result of inverting baseline activations
    - steered_inversion: Result of inverting steered activations
    - metrics: Various metrics (MSE, match, etc.)
    """
    result = {
        "instruction": instruction,
        "timestamp": datetime.now().isoformat(),
        "steering_layer": steering_config.layer,
        "steering_method": steering_config.method,
        "steering_coeff": steering_config.coeff,
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
    print(f"Steering: layer={steering_config.layer}, method={steering_config.method}, coeff={steering_config.coeff}")
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
    print(f"Activation difference (L2): {act_diff:.4f}")
    print(f"Per-token difference: {act_diff_per_token:.4f}")
    
    # # Step 3: Baseline inversion
    # print("\n[Step 3] Inverting baseline activations...")
    
    # baseline_match, baseline_time, baseline_recon_ids, baseline_times, baseline_inv_result = inversion_attack(
    #     input_ids, model, tokenizer, inversion_layer, lr, seed,
    #     special_start_tokens=special_start_tokens,
    #     continue_on_failure=continue_on_failure,
    #     top_k=top_k,
    # )
    
    # result["baseline_inversion"] = {
    #     "match": baseline_match,
    #     "time": baseline_time,
    #     "reconstructed_ids": baseline_recon_ids,
    #     "failed_positions": baseline_inv_result.failed_positions if baseline_inv_result else [],
    # }
    
    # # Store top-k tokens for analysis
    # if baseline_inv_result and baseline_inv_result.top_k_per_position:
    #     result["baseline_inversion"]["top_k_per_position"] = [
    #         [(tid, dist) for tid, dist in pos_top_k]
    #         for pos_top_k in baseline_inv_result.top_k_per_position
    #     ]
    
    # if baseline_recon_ids:
    #     baseline_recon_text = tokenizer.decode(baseline_recon_ids)
    #     result["baseline_inversion"]["reconstructed_text"] = baseline_recon_text
        
    #     recon_ids_tensor = torch.tensor(baseline_recon_ids).unsqueeze(0).to(model.device)
    #     baseline_mse = compute_activation_mse(
    #         model, recon_ids_tensor, baseline_acts, inversion_layer
    #     )
    #     result["baseline_inversion"]["mse"] = baseline_mse

    # # Print top-k
    # if baseline_inv_result:
    #     print_top_k_tokens(baseline_inv_result, tokenizer, result["original_ids"])
    
    # Step 4: Steered inversion
    print("\n[Step 4] Inverting steered activations...")
    
    steered_match, steered_time, steered_recon_ids, steered_times, steered_inv_result = inversion_attack_with_target(
        steered_acts, model, tokenizer, inversion_layer, lr, seed, 
        verbose=True, original_ids=input_ids,
        special_start_tokens=special_start_tokens,
        continue_on_failure=continue_on_failure,
        top_k=top_k,
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
        
        from transformers import GenerationConfig
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        with torch.no_grad():
            recon_outputs = model.generate(
                input_ids=recon_input_ids,
                attention_mask=recon_attention_mask,
                generation_config=gen_config,
            )
        recon_gen_tokens = recon_outputs[0, recon_input_ids.size(1):]
        recon_generation = tokenizer.decode(recon_gen_tokens)
        
        result["reconstructed_prompt_generation"] = recon_generation
        print(f"Reconstructed prompt generation: {recon_generation[:100]}...")
    
    # Print top-k
    if steered_inv_result:
        print_top_k_tokens(steered_inv_result, tokenizer, result["original_ids"])
    
    return result


def run_experiment(config: Config, instructions: Optional[List[str]] = None):
    """Run the full experiment."""
    print("="*60)
    print("INVERTING STEERED ACTIVATIONS EXPERIMENT")
    print("="*60)
    
    set_seed(config.seed)
    
    # Load model
    print(f"\nLoading model: {config.model_id}")
    model, tokenizer = load_model(config.model_id, config.device, config.dtype)
    tokenize_fn = get_tokenize_fn(tokenizer, use_chat_template=config.use_chat_template, add_special_tokens=config.add_special_tokens)
    
    n_layers = get_num_layers(model)
    print(f"Model has {n_layers} layers")
    print(f"Use chat template: {config.use_chat_template}")
    # print(f"Special start tokens: {config.special_start_tokens}")
    
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
    
    steering_config = SteeringConfig(
        direction=direction,
        layer=layer,
        method=config.steering_method,
        coeff=config.steering_coeff,
    )
    
    # Step 2: Run experiments
    print("\n" + "="*60)
    print("STEP 2: RUNNING INVERSION EXPERIMENTS")
    print("="*60)
    
    if instructions is None:
        instructions = TEST_INSTRUCTIONS
    
    results = []
    for instruction in instructions:
        try:
            result = run_single_experiment(
                model, tokenizer, tokenize_fn,
                instruction, steering_config,
                lr=config.learning_rate, seed=config.seed,
                max_new_tokens=config.max_new_tokens,
                special_start_tokens=config.special_start_tokens,
                continue_on_failure=config.continue_on_failure,
                top_k=config.top_k,
            )
            results.append(result)
        except Exception as e:
            import traceback
            print(f"Error processing instruction: {e}")
            traceback.print_exc()
            results.append({
                "instruction": instruction,
                "error": str(e),
            })
        
        # Save intermediate results
        results_path = os.path.join(config.output_dir, "experiment_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    n_baseline_match = sum(1 for r in results if r.get("baseline_inversion", {}).get("match", False))
    n_steered_match = sum(1 for r in results if r.get("steered_inversion", {}).get("match_original", False))
    n_baseline_failed = sum(1 for r in results if r.get("baseline_inversion", {}).get("failed_positions", []))
    n_steered_failed = sum(1 for r in results if r.get("steered_inversion", {}).get("failed_positions", []))
    
    print(f"Total experiments: {len(results)}")
    print(f"Baseline exact matches: {n_baseline_match}/{len(results)}")
    print(f"Baseline with failures: {n_baseline_failed}/{len(results)}")
    print(f"Steered matches original: {n_steered_match}/{len(results)}")
    print(f"Steered with failures: {n_steered_failed}/{len(results)}")
    
    print(f"\nResults saved to: {results_path}")
    
    return results


def demo(config: Config):
    """Run a quick demo with a single instruction."""
    print("="*60)
    print("DEMO: INVERTING STEERED ACTIVATIONS")
    print("="*60)
    
    set_seed(config.seed)
    
    print(f"\nLoading model: {config.model_id}")
    model, tokenizer = load_model(config.model_id, config.device, config.dtype)
    tokenize_fn = get_tokenize_fn(tokenizer, use_chat_template=config.use_chat_template, add_special_tokens=config.add_special_tokens)
    
    direction_path = config.get_direction_path()
    if not os.path.exists(direction_path):
        raise FileNotFoundError(
            f"Direction file not found: {direction_path}\n"
            f"Please run the refusal_direction pipeline first."
        )
    
    direction, layer, metadata = load_steering_direction(direction_path, config.device)
    print(f"Loaded direction from layer {layer}")
    
    steering_config = SteeringConfig(
        direction=direction,
        layer=layer,
        method=config.steering_method,
        coeff=config.steering_coeff,
    )
    
    test_instructions = TEST_INSTRUCTIONS
    
    print("\n--- Comparing generations ---")
    baseline_gens, steered_gens = compare_generations(
        model, tokenizer, test_instructions, steering_config,
        tokenize_fn, max_new_tokens=config.max_new_tokens
    )
    
    for i, (instr, base, steered) in enumerate(zip(test_instructions, baseline_gens, steered_gens)):
        print(f"\n[{i+1}] Instruction: {instr[:60]}...")
        print(f"    Baseline: {base[:80]}...")
        print(f"    Steered:  {steered[:80]}...")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run steered activation inversion experiment")
    parser.add_argument("--demo", action="store_true", help="Run quick demo")
    parser.add_argument("--model", type=str, default=None, help="Model to use")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate for inversion")
    parser.add_argument("--direction", type=str, default=None, help="Path to direction.pt")
    parser.add_argument("--method", type=str, default="actadd", help="Steering method: actadd or ablation")
    parser.add_argument("--coeff", type=float, default=-1.0, help="Steering coefficient")
    parser.add_argument("--no-chat-template", action="store_true", 
                        help="Do not use chat template")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="Continue with ground truth when inversion fails")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top candidates to track")
    parser.add_argument("--add-special-tokens", action="store_true",
                        help="Add special tokens (like BOS token for Llama)")
    args = parser.parse_args()
    
    config = Config()
    if args.model:
        config.model_id = args.model
    config.device = args.device
    config.seed = args.seed
    config.learning_rate = args.lr
    if args.direction:
        config.direction_path = args.direction
    config.steering_method = args.method
    config.steering_coeff = args.coeff
    config.use_chat_template = not args.no_chat_template
    config.add_special_tokens = args.add_special_tokens
    config.continue_on_failure = args.continue_on_failure
    config.top_k = args.top_k
    
    # Re-compute special tokens after setting use_chat_template
    config.special_start_tokens = config._get_default_special_tokens()
    
    if args.demo:
        demo(config)
    else:
        run_experiment(config)

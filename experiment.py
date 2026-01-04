"""
Main experiment script for inverting intervened activations.

This script:
1. Extracts the refusal direction for Llama-3.2-1B-Instruct
2. For a set of harmful prompts:
   a. Gets baseline activations
   b. Gets intervened (refusal-ablated) activations
   c. Attempts to invert both back to prompts
3. Compares the results and evaluates behavioral equivalence
"""

import os
import json
import torch
from typing import List, Dict, Any
from tqdm import tqdm
from datetime import datetime

from config import Config, HARMFUL_INSTRUCTIONS, HARMLESS_INSTRUCTIONS
from model_utils import (
    load_model, 
    get_tokenize_fn, 
    replace_final_norm_with_identity,
    restore_final_norm,
    set_seed,
    get_num_layers,
)
from refusal_direction import (
    compute_refusal_direction,
    select_best_direction,
    extract_and_save_refusal_direction,
)
from intervention import (
    get_hidden_states_iterative_with_ablation,
    compare_generations,
)
from inversion import (
    extract_hidden_states_iterative,
    inversion_attack,
    inversion_attack_with_target,
    compute_activation_mse,
)


def run_single_experiment(
    model,
    tokenizer,
    tokenize_fn,
    instruction: str,
    refusal_direction: torch.Tensor,
    inversion_layer: int,
    lr: float = 1.0,
    seed: int = 42,
    max_new_tokens: int = 64,
) -> Dict[str, Any]:
    """
    Run experiment for a single instruction.
    
    Returns a dict with:
    - instruction: The original instruction
    - original_ids: Original token IDs
    - baseline_generation: Model output without intervention
    - ablated_generation: Model output with refusal ablation
    - baseline_inversion: Result of inverting baseline activations
    - ablated_inversion: Result of inverting ablated activations
    - metrics: Various metrics (MSE, match, etc.)
    """
    result = {
        "instruction": instruction,
        "timestamp": datetime.now().isoformat(),
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
    print(f"{'='*60}")
    
    # Step 1: Get baseline and ablated generations
    print("\n[Step 1] Comparing baseline vs ablated generations...")
    baseline_gen, ablated_gen = compare_generations(
        model, tokenizer, instruction, refusal_direction,
        tokenize_fn, max_new_tokens
    )
    result["baseline_generation"] = baseline_gen
    result["ablated_generation"] = ablated_gen
    
    print(f"Baseline: {baseline_gen[:100]}...")
    print(f"Ablated:  {ablated_gen[:100]}...")
    
    # Step 2: Extract activations
    print("\n[Step 2] Extracting activations...")
    
    # Baseline activations (standard forward pass)
    baseline_acts = extract_hidden_states_iterative(
        input_ids, model, inversion_layer
    )
    
    # Ablated activations (with refusal direction removed)
    ablated_acts = get_hidden_states_iterative_with_ablation(
        model, input_ids, refusal_direction, inversion_layer
    )
    
    # Compute activation difference
    act_diff = torch.norm(baseline_acts - ablated_acts).item()
    act_diff_per_token = torch.norm(baseline_acts - ablated_acts, dim=1).mean().item()
    result["activation_diff_total"] = act_diff
    result["activation_diff_per_token"] = act_diff_per_token
    print(f"Activation difference (L2): {act_diff:.4f}")
    print(f"Per-token difference: {act_diff_per_token:.4f}")
    
    # Step 3: Baseline inversion
    print("\n[Step 3] Inverting baseline activations...")
    
    baseline_match, baseline_time, baseline_recon_ids, baseline_times = inversion_attack(
        input_ids, model, tokenizer, inversion_layer, lr, seed, verbose=True
    )
    
    result["baseline_inversion"] = {
        "match": baseline_match,
        "time": baseline_time,
        "reconstructed_ids": baseline_recon_ids,
    }
    
    if baseline_recon_ids:
        baseline_recon_text = tokenizer.decode(baseline_recon_ids, skip_special_tokens=True)
        result["baseline_inversion"]["reconstructed_text"] = baseline_recon_text
        
        # Compute MSE of reconstructed vs target
        recon_ids_tensor = torch.tensor(baseline_recon_ids).unsqueeze(0).to(model.device)
        baseline_mse = compute_activation_mse(
            model, recon_ids_tensor, baseline_acts, inversion_layer
        )
        result["baseline_inversion"]["mse"] = baseline_mse
    
    # Step 4: Ablated inversion
    print("\n[Step 4] Inverting ablated activations...")
    
    ablated_match, ablated_time, ablated_recon_ids, ablated_times = inversion_attack_with_target(
        ablated_acts, model, tokenizer, inversion_layer, lr, seed, 
        verbose=True, original_ids=input_ids
    )
    
    result["ablated_inversion"] = {
        "match_original": ablated_match,
        "time": ablated_time,
        "reconstructed_ids": ablated_recon_ids,
    }
    
    if ablated_recon_ids:
        ablated_recon_text = tokenizer.decode(ablated_recon_ids, skip_special_tokens=True)
        result["ablated_inversion"]["reconstructed_text"] = ablated_recon_text
        
        # Compute MSE of reconstructed vs target (ablated activations)
        recon_ids_tensor = torch.tensor(ablated_recon_ids).unsqueeze(0).to(model.device)
        ablated_mse = compute_activation_mse(
            model, recon_ids_tensor, ablated_acts, inversion_layer
        )
        result["ablated_inversion"]["mse_to_target"] = ablated_mse
        
        # Also compute MSE to baseline activations
        baseline_mse_from_ablated = compute_activation_mse(
            model, recon_ids_tensor, baseline_acts, inversion_layer
        )
        result["ablated_inversion"]["mse_to_baseline"] = baseline_mse_from_ablated
        
        # Step 5: Test if reconstructed prompt bypasses refusal
        print("\n[Step 5] Testing reconstructed prompt...")
        recon_instruction = ablated_recon_text
        
        # Generate with the reconstructed prompt
        recon_inputs = tokenize_fn(instructions=[recon_instruction])
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
        recon_generation = tokenizer.decode(recon_gen_tokens, skip_special_tokens=True)
        
        result["reconstructed_prompt_generation"] = recon_generation
        print(f"Reconstructed prompt generation: {recon_generation[:100]}...")
    
    return result


def run_experiment(config: Config):
    """Run the full experiment."""
    print("="*60)
    print("INVERTING INTERVENED ACTIVATIONS EXPERIMENT")
    print("="*60)
    
    set_seed(config.seed)
    
    # Load model
    print(f"\nLoading model: {config.model_id}")
    model, tokenizer = load_model(config.model_id, config.device, config.dtype)
    tokenize_fn = get_tokenize_fn(tokenizer)
    
    # Replace final norm for SIP-It
    replace_final_norm_with_identity(model)
    
    # Get number of layers and set inversion layer
    n_layers = get_num_layers(model)
    if config.inversion_layer < 0:
        inversion_layer = n_layers + config.inversion_layer + 1
    else:
        inversion_layer = config.inversion_layer
    print(f"Using layer {inversion_layer} for inversion (out of {n_layers})")
    
    # Step 1: Extract refusal direction
    print("\n" + "="*60)
    print("STEP 1: EXTRACTING REFUSAL DIRECTION")
    print("="*60)
    
    refusal_path = os.path.join(config.output_dir, "refusal_direction.pt")
    if os.path.exists(refusal_path):
        print(f"Loading existing refusal direction from {refusal_path}")
        saved = torch.load(refusal_path)
        refusal_direction = saved['direction'].to(model.device)
        refusal_layer = saved['layer']
    else:
        print("Computing refusal direction...")
        harmful = HARMFUL_INSTRUCTIONS[:config.n_harmful_train]
        harmless = HARMLESS_INSTRUCTIONS[:config.n_harmless_train]
        
        refusal_directions = compute_refusal_direction(
            model, tokenizer, harmful, harmless,
            batch_size=8, positions=[-1]
        )
        
        refusal_direction, refusal_layer = select_best_direction(
            refusal_directions, position=0, layer=config.refusal_layer
        )
        refusal_direction = refusal_direction.to(model.device)
        
        # Save
        torch.save({
            'direction': refusal_direction.cpu(),
            'layer': refusal_layer,
            'all_directions': refusal_directions.cpu(),
            'config': {
                'model_id': config.model_id,
                'n_harmful': len(harmful),
                'n_harmless': len(harmless),
            }
        }, refusal_path)
    
    print(f"Refusal direction extracted at layer {refusal_layer}")
    print(f"Direction norm: {refusal_direction.norm().item():.4f}")
    
    # Step 2: Run experiments on harmful instructions
    print("\n" + "="*60)
    print("STEP 2: RUNNING INVERSION EXPERIMENTS")
    print("="*60)
    
    # Use a subset of harmful instructions for the experiment
    test_instructions = HARMFUL_INSTRUCTIONS[:5]  # Start with 5 for testing
    
    results = []
    for instruction in test_instructions:
        try:
            result = run_single_experiment(
                model, tokenizer, tokenize_fn,
                instruction, refusal_direction, inversion_layer,
                lr=config.learning_rate, seed=config.seed,
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing instruction: {e}")
            results.append({
                "instruction": instruction,
                "error": str(e),
            })
        
        # Save intermediate results
        results_path = os.path.join(config.output_dir, "experiment_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # Restore final norm
    restore_final_norm(model)
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    n_baseline_match = sum(1 for r in results if r.get("baseline_inversion", {}).get("match", False))
    n_ablated_match = sum(1 for r in results if r.get("ablated_inversion", {}).get("match_original", False))
    
    print(f"Total experiments: {len(results)}")
    print(f"Baseline exact matches: {n_baseline_match}/{len(results)}")
    print(f"Ablated matches original: {n_ablated_match}/{len(results)}")
    
    print(f"\nResults saved to: {results_path}")
    
    return results


def quick_demo(config: Config):
    """Run a quick demo with a single instruction."""
    print("="*60)
    print("QUICK DEMO: INVERTING INTERVENED ACTIVATIONS")
    print("="*60)
    
    set_seed(config.seed)
    
    # Load model
    print(f"\nLoading model: {config.model_id}")
    model, tokenizer = load_model(config.model_id, config.device, config.dtype)
    tokenize_fn = get_tokenize_fn(tokenizer)
    
    # Replace final norm for SIP-It
    replace_final_norm_with_identity(model)
    
    # Get number of layers
    n_layers = get_num_layers(model)
    inversion_layer = n_layers  # Last layer
    print(f"Using layer {inversion_layer} for inversion")
    
    harmful = HARMFUL_INSTRUCTIONS[:config.n_harmful_train]
    harmless = HARMLESS_INSTRUCTIONS[:config.n_harmless_train]
    
    # Extract simple refusal direction (just from this one sample)
    print("\nComputing quick refusal direction...")
    from refusal_direction import compute_refusal_direction, select_best_direction
    
    refusal_directions = compute_refusal_direction(
        model, tokenizer, harmful, harmless,
        batch_size=1, positions=[-1]
    )
    refusal_direction, refusal_layer = select_best_direction(
        refusal_directions, position=0, layer=config.refusal_layer  # Middle layer
    )
    refusal_direction = refusal_direction.to(model.device)
    
    # Compare generations
    print("\nComparing generations...")
    baseline_gen, ablated_gen = compare_generations(
        model, tokenizer, harmful[0], refusal_direction,
        tokenize_fn, max_new_tokens=32
    )
    print(f"Baseline: {baseline_gen[:100]}")
    print(f"Ablated:  {ablated_gen[:100]}")
    
    # Extract activations
    print("\nExtracting activations...")
    inputs = tokenize_fn(instructions=[harmful[0]])
    input_ids = inputs.input_ids.to(model.device)
    baseline_acts = extract_hidden_states_iterative(input_ids, model, inversion_layer)
    ablated_acts = get_hidden_states_iterative_with_ablation(
        model, inputs.input_ids.to(model.device), refusal_direction, inversion_layer
    )
    
    diff = torch.norm(baseline_acts - ablated_acts).item()
    print(f"Activation difference: {diff:.4f}")
    
    # Try to invert ablated activations
    print("\nInverting ablated activations...")
    match, time_taken, recon_ids, _ = inversion_attack_with_target(
        ablated_acts, model, tokenizer, inversion_layer,
        lr=1.0, seed=42, verbose=True, original_ids=input_ids
    )
    
    if recon_ids:
        recon_text = tokenizer.decode(recon_ids, skip_special_tokens=True)
        print(f"\nReconstructed: {recon_text}")
        
        # Test the reconstructed prompt
        print("\nGenerating with reconstructed prompt...")
        recon_inputs = tokenize_fn(instructions=[recon_text])
        
        from transformers import GenerationConfig
        gen_config = GenerationConfig(
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=recon_inputs.input_ids.to(model.device),
                attention_mask=recon_inputs.attention_mask.to(model.device),
                generation_config=gen_config,
            )
        gen_text = tokenizer.decode(outputs[0, recon_inputs.input_ids.size(1):], skip_special_tokens=True)
        print(f"Generation: {gen_text[:100]}")
    
    # Restore
    restore_final_norm(model)
    print("\nDemo complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inversion experiment")
    parser.add_argument("--demo", action="store_true", help="Run quick demo")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                       help="Model to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate")
    args = parser.parse_args()
    
    config = Config()
    config.model_id = args.model
    config.device = args.device
    config.seed = args.seed
    config.learning_rate = args.lr
    config.n_harmful_train = 32
    config.n_harmless_train = 32
    
    if args.demo:
        quick_demo(config)
    else:
        run_experiment(config)


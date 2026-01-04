"""
Extract the refusal direction from Llama-3.2-1B-Instruct.
"""

import torch
import os
from typing import List, Tuple
from tqdm import tqdm
from torch import Tensor

from config import Config, HARMFUL_INSTRUCTIONS, HARMLESS_INSTRUCTIONS
from model_utils import load_model, get_tokenize_fn, get_model_layers


def get_activation_hook(layer: int, cache: Tensor, n_samples: int, positions: List[int]):
    """
    Create a forward pre-hook that accumulates mean activations.
    
    Args:
        layer: Layer index
        cache: Tensor to accumulate activations into [n_positions, n_layers, d_model]
        n_samples: Total number of samples (for averaging)
        positions: Token positions to capture (e.g., [-1] for last token)
    """
    def hook_fn(module, input):
        activation = input[0].clone().to(cache)  # [batch, seq_len, d_model]
        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)
    return hook_fn


def get_mean_activations(
    model, 
    tokenizer, 
    instructions: List[str], 
    tokenize_fn,
    batch_size: int = 8,
    positions: List[int] = [-1]
) -> Tensor:
    """
    Compute mean activations for a set of instructions.
    
    Returns:
        Tensor of shape [n_positions, n_layers, d_model]
    """
    torch.cuda.empty_cache()
    
    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size
    
    # Store mean activations in high precision
    mean_activations = torch.zeros(
        (n_positions, n_layers, d_model), 
        dtype=torch.float64, 
        device=model.device
    )
    
    # Register hooks on all layers
    layers = get_model_layers(model)
    hooks = []
    for layer_idx in range(n_layers):
        hook = layers[layer_idx].register_forward_pre_hook(
            get_activation_hook(
                layer=layer_idx, 
                cache=mean_activations, 
                n_samples=n_samples, 
                positions=positions
            )
        )
        hooks.append(hook)
    
    try:
        for i in tqdm(range(0, len(instructions), batch_size), desc="Computing activations"):
            batch_instructions = instructions[i:i+batch_size]
            inputs = tokenize_fn(instructions=batch_instructions)
            
            with torch.no_grad():
                model(
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                )
    finally:
        # Always remove hooks
        for hook in hooks:
            hook.remove()
    
    return mean_activations


def compute_refusal_direction(
    model,
    tokenizer,
    harmful_instructions: List[str],
    harmless_instructions: List[str],
    batch_size: int = 8,
    positions: List[int] = [-1]
) -> Tensor:
    """
    Compute the refusal direction as the difference of mean activations.
    
    Returns:
        Tensor of shape [n_positions, n_layers, d_model]
    """
    tokenize_fn = get_tokenize_fn(tokenizer)
    
    print("Computing harmful activations...")
    harmful_acts = get_mean_activations(
        model, tokenizer, harmful_instructions, tokenize_fn, batch_size, positions
    )
    print (f"Harmful activations: {harmful_acts.shape}")
    
    print("Computing harmless activations...")
    harmless_acts = get_mean_activations(
        model, tokenizer, harmless_instructions, tokenize_fn, batch_size, positions
    )
    print (f"Harmless activations: {harmless_acts.shape}")

    input("Press Enter to continue...")
    
    refusal_direction = harmful_acts - harmless_acts
    
    return refusal_direction


def select_best_direction(
    refusal_directions: Tensor,
    position: int = 0,
    layer: int = -1
) -> Tuple[Tensor, int]:
    """
    Select the best refusal direction from the computed directions.
    
    Args:
        refusal_directions: [n_positions, n_layers, d_model]
        position: Which token position to use
        layer: Which layer to use (-1 for auto-select based on norm)
    
    Returns:
        Tuple of (direction tensor [d_model], selected layer index)
    """
    if layer >= 0:
        direction = refusal_directions[position, layer]
        direction = direction / direction.norm()
        return direction, layer
    
    # Auto-select layer with largest norm (strongest refusal signal)
    norms = refusal_directions[position].norm(dim=-1)  # [n_layers]
    best_layer = norms.argmax().item()
    
    direction = refusal_directions[position, best_layer]
    direction = direction / direction.norm()
    
    return direction, best_layer


def extract_and_save_refusal_direction(config: Config):
    """Main function to extract and save the refusal direction."""
    print(f"Loading model: {config.model_id}")
    model, tokenizer = load_model(config.model_id, config.device, config.dtype)
    
    # Use subset of instructions
    harmful = HARMFUL_INSTRUCTIONS[:config.n_harmful_train]
    harmless = HARMLESS_INSTRUCTIONS[:config.n_harmless_train]

    print(f"Using {len(harmful)} harmful and {len(harmless)} harmless instructions")
    
    # Compute refusal directions
    refusal_directions = compute_refusal_direction(
        model, tokenizer, harmful, harmless, 
        batch_size=8, positions=[-1]
    )
    
    # Select best direction
    direction, selected_layer = select_best_direction(
        refusal_directions, 
        position=0, 
        layer=config.refusal_layer
    )
    
    print(f"Selected layer {selected_layer} for refusal direction")
    print(f"Direction norm: {direction.norm().item():.4f}")
    
    # Save
    output_path = os.path.join(config.output_dir, "refusal_direction.pt")
    torch.save({
        'direction': direction.cpu(),
        'layer': selected_layer,
        'all_directions': refusal_directions.cpu(),
        'config': {
            'model_id': config.model_id,
            'n_harmful': len(harmful),
            'n_harmless': len(harmless),
        }
    }, output_path)
    
    print(f"Saved refusal direction to {output_path}")
    
    return direction, selected_layer


if __name__ == "__main__":
    config = Config()
    extract_and_save_refusal_direction(config)


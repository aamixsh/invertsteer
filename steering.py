"""
Modular steering abstraction for invertsteer.

This module provides a unified interface for different steering methods (actadd, ablation, etc.)
that can be applied to model activations. It reuses hooks from the refusal_direction repo.
"""

import sys
import os
import torch
import contextlib
from typing import List, Tuple, Callable, Optional, Dict, Any
from dataclasses import dataclass
from torch import Tensor

# Add refusal_direction to path for importing hooks
REFUSAL_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'refusal_direction'))
if REFUSAL_DIR_PATH not in sys.path:
    sys.path.insert(0, REFUSAL_DIR_PATH)

from pipeline.utils.hook_utils import (
    add_hooks,
    get_activation_addition_input_pre_hook,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
    get_all_direction_ablation_hooks,
)


@dataclass
class SteeringConfig:
    """Configuration for a steering intervention."""
    direction: Tensor  # The steering direction vector [d_model]
    layer: int  # Layer to apply the steering
    method: str = "actadd"  # "actadd" or "ablation"
    coeff: float = 1.0  # Coefficient for actadd (positive = add, negative = subtract)
    steering_type: str = "refusal"  # "refusal" or "persona"

    def __post_init__(self):
        if self.method not in ["actadd", "ablation"]:
            raise ValueError(f"Unknown steering method: {self.method}. Use 'actadd' or 'ablation'.")
        if self.steering_type not in ["refusal", "persona"]:
            raise ValueError(f"Unknown steering type: {self.steering_type}. Use 'refusal' or 'persona'.")


def load_steering_direction(direction_path: str, device: str = "cuda") -> Tuple[Tensor, int, Dict[str, Any]]:
    """
    Load a steering direction from a saved file.
    
    Supports loading from refusal_direction pipeline outputs:
    - direction.pt: Contains just the direction tensor
    - Also loads direction_metadata.json if present
    
    Args:
        direction_path: Path to direction.pt file
        device: Device to load the tensor to
    
    Returns:
        Tuple of (direction tensor, layer, metadata dict)
    """
    import json
    
    direction = torch.load(direction_path, map_location=device)
    
    # Handle different save formats
    if isinstance(direction, dict):
        # Old format from our implementation
        dir_tensor = direction['direction'].to(device)
        layer = direction.get('layer', -1)
        metadata = direction.get('config', {})
    else:
        # New format from refusal_direction pipeline (just the tensor)
        dir_tensor = direction.to(device)
        layer = -1
        metadata = {}

    if len(direction.shape) == 2:
        layer = (direction.shape[0] - 1) // 2
        return dir_tensor[layer], layer, metadata

    # Try to load metadata from companion file
    metadata_path = os.path.join(os.path.dirname(direction_path), 'direction_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
            layer = meta.get('layer', layer)
            metadata.update(meta)
    
    return dir_tensor, layer, metadata


def get_steering_hooks(
    model,
    steering_config: SteeringConfig,
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Get hooks for applying steering to a model.
    
    Args:
        model: The model to get hooks for
        steering_config: Configuration for the steering
    
    Returns:
        Tuple of (fwd_pre_hooks, fwd_hooks) lists
    """
    direction = steering_config.direction
    layer = steering_config.layer
    method = steering_config.method
    coeff = steering_config.coeff
    
    # Get model layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        model_layers = model.model.layers
    else:
        raise ValueError("Model structure not supported. Expected model.model.layers")
    
    if method == "actadd":
        # Activation addition at a single layer
        fwd_pre_hooks = [
            (model_layers[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=coeff))
        ]
        fwd_hooks = []
    elif method == "ablation":
        # Direction ablation across all layers
        n_layers = len(model_layers)
        fwd_pre_hooks = [
            (model_layers[l], get_direction_ablation_input_pre_hook(direction=direction))
            for l in range(n_layers)
        ]
        # Also ablate from attention and MLP outputs
        fwd_hooks = []
        for l in range(n_layers):
            fwd_hooks.append((model_layers[l].self_attn, get_direction_ablation_output_hook(direction=direction)))
            fwd_hooks.append((model_layers[l].mlp, get_direction_ablation_output_hook(direction=direction)))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return fwd_pre_hooks, fwd_hooks


@contextlib.contextmanager
def steering_context(
    model,
    steering_config: SteeringConfig,
):
    """
    Context manager that temporarily applies steering hooks to a model.
    
    Usage:
        with steering_context(model, steering_config):
            outputs = model(input_ids)
    """
    fwd_pre_hooks, fwd_hooks = get_steering_hooks(model, steering_config)
    
    with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
        yield


def generate_with_steering(
    model,
    tokenizer,
    input_ids: Tensor,
    steering_config: SteeringConfig,
    max_new_tokens: int = 64,
    attention_mask: Optional[Tensor] = None,
) -> List[str]:
    """
    Generate text with steering applied.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        input_ids: Input token IDs [batch, seq_len]
        steering_config: Configuration for the steering
        max_new_tokens: Maximum tokens to generate
        attention_mask: Optional attention mask
    
    Returns:
        List of generated text strings (excluding the prompt)
    """
    with steering_context(model, steering_config):
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Remove prompt tokens
    generated_tokens = outputs[:, input_ids.size(1):]
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def get_hidden_states_with_steering(
    model,
    input_ids: Tensor,
    steering_config: SteeringConfig,
    layer_idx: int,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Get hidden states at a specific layer with steering applied.
    
    Args:
        model: The model
        input_ids: Input token IDs [batch, seq_len]
        steering_config: Configuration for the steering
        layer_idx: Layer to extract hidden states from
        attention_mask: Optional attention mask
    
    Returns:
        Hidden states tensor [batch, seq_len, d_model]
    """
    with steering_context(model, steering_config):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
    
    # hidden_states is a tuple of (embedding, layer1, layer2, ..., layerN)
    hidden_states = outputs.hidden_states[layer_idx][0]
    return hidden_states.detach()


def get_hidden_states_iterative_with_steering(
    model,
    input_ids: Tensor,
    steering_config: SteeringConfig,
    layer_idx: int,
) -> Tensor:
    """
    Get hidden states iteratively (one token at a time) with steering.
    This matches the SIP-It approach where hidden states are computed
    for each prefix length.
    
    Args:
        model: The model
        input_ids: Input token IDs [1, seq_len]
        steering_config: Configuration for the steering
        layer_idx: Layer to extract hidden states from
    
    Returns:
        Hidden states tensor [seq_len, d_model] where each row is the
        hidden state of the last token when the input is the first i tokens.
    """
    seq_len = input_ids.size(1)
    
    # Get embeddings first
    embeddings = model.get_input_embeddings()(input_ids)  # [1, seq_len, d_model]
    
    target_hidden_states = []
    
    with steering_context(model, steering_config):
        with torch.no_grad():
            for i in range(1, seq_len + 1):
                outputs = model(
                    inputs_embeds=embeddings[:, :i, :],
                    output_hidden_states=True,
                    use_cache=False,
                )
                # Get hidden state of last token at target layer
                h = outputs.hidden_states[layer_idx][0, -1, :]
                target_hidden_states.append(h.detach())
    
    return torch.stack(target_hidden_states)


def compare_generations(
    model,
    tokenizer,
    instructions: List[str],
    steering_config: SteeringConfig,
    tokenize_fn: Callable,
    max_new_tokens: int = 64,
) -> Tuple[List[str], List[str]]:
    """
    Compare baseline and steered generations for instructions.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        instructions: List of instruction strings
        steering_config: Configuration for the steering
        tokenize_fn: Function to tokenize instructions
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Tuple of (baseline_generations, steered_generations)
    """
    # Tokenize
    inputs = tokenize_fn(instructions=instructions)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # Baseline generation (no steering)
    with torch.no_grad():
        baseline_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    baseline_text = tokenizer.batch_decode(
        baseline_outputs[:, input_ids.size(1):], 
        skip_special_tokens=True
    )
    
    # Steered generation
    steered_text = generate_with_steering(
        model, tokenizer, input_ids, steering_config,
        max_new_tokens, attention_mask
    )
    
    return baseline_text, steered_text

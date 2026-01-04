"""
Activation intervention utilities for ablating the refusal direction.
"""

import torch
import contextlib
import functools
from typing import List, Tuple, Callable, Optional
from torch import Tensor

from model_utils import get_model_layers


def get_ablation_hook(direction: Tensor):
    """
    Create a hook that ablates a direction from activations.
    
    The ablation is: activation' = activation - (activation · r̂) r̂
    where r̂ is the normalized direction.
    """
    # Normalize direction
    direction = direction / (direction.norm() + 1e-8)
    
    def hook_fn(module, input, output=None):
        # Handle both pre-hooks (input only) and post-hooks (input + output)
        if output is not None:
            # Post-hook
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
        else:
            # Pre-hook
            if isinstance(input, tuple):
                activation = input[0]
            else:
                activation = input
        
        # Move direction to activation's device and dtype
        dir_tensor = direction.to(activation)
        
        # Project activation onto direction and subtract
        # activation: [batch, seq_len, d_model]
        # direction: [d_model]
        projection = (activation @ dir_tensor).unsqueeze(-1) * dir_tensor
        ablated = activation - projection
        
        if output is not None:
            if isinstance(output, tuple):
                return (ablated,) + output[1:]
            return ablated
        else:
            if isinstance(input, tuple):
                return (ablated,) + input[1:]
            return ablated
    
    return hook_fn


@contextlib.contextmanager
def ablation_context(model, direction: Tensor, layers: Optional[List[int]] = None):
    """
    Context manager that temporarily applies refusal ablation hooks.
    
    Args:
        model: The model to apply hooks to
        direction: The refusal direction to ablate [d_model]
        layers: Optional list of layer indices. If None, applies to all layers.
    """
    model_layers = get_model_layers(model)
    n_layers = len(model_layers)
    
    if layers is None:
        layers = list(range(n_layers))
    
    handles = []
    hook_fn = get_ablation_hook(direction)
    
    try:
        for layer_idx in layers:
            # Apply to layer input (residual stream before layer)
            handle = model_layers[layer_idx].register_forward_pre_hook(hook_fn)
            handles.append(handle)
            
            # Also apply to attention and MLP outputs
            attn_handle = model_layers[layer_idx].self_attn.register_forward_hook(hook_fn)
            mlp_handle = model_layers[layer_idx].mlp.register_forward_hook(hook_fn)
            handles.append(attn_handle)
            handles.append(mlp_handle)
        
        yield
    finally:
        for handle in handles:
            handle.remove()


def get_hidden_states_with_ablation(
    model,
    input_ids: Tensor,
    direction: Tensor,
    layer_idx: int,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Get hidden states at a specific layer with refusal ablation applied.
    
    Args:
        model: The model
        input_ids: Input token IDs [batch, seq_len]
        direction: Refusal direction to ablate [d_model]
        layer_idx: Layer to extract hidden states from
        attention_mask: Optional attention mask
    
    Returns:
        Hidden states tensor [batch, seq_len, d_model]
    """
    with ablation_context(model, direction):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
    
    # hidden_states is a tuple of (embedding, layer1, layer2, ..., layerN)
    hidden_states = outputs.hidden_states[layer_idx]
    return hidden_states


def get_hidden_states_iterative_with_ablation(
    model,
    input_ids: Tensor,
    direction: Tensor,
    layer_idx: int,
) -> Tensor:
    """
    Get hidden states iteratively (one token at a time) with ablation.
    This matches the SIP-It approach where hidden states are computed
    for each prefix length.
    
    Args:
        model: The model
        input_ids: Input token IDs [1, seq_len]
        direction: Refusal direction to ablate [d_model]
        layer_idx: Layer to extract hidden states from
    
    Returns:
        Hidden states tensor [seq_len, d_model] where each row is the
        hidden state of the last token when the input is the first i tokens.
    """
    device = input_ids.device
    seq_len = input_ids.size(1)
    
    # Get embeddings first
    embeddings = model.get_input_embeddings()(input_ids)  # [1, seq_len, d_model]
    
    target_embeddings = []
    
    with ablation_context(model, direction):
        with torch.no_grad():
            for i in range(1, seq_len + 1):
                outputs = model(
                    inputs_embeds=embeddings[:, :i, :],
                    output_hidden_states=True,
                    use_cache=False,
                )
                # Get hidden state of last token at target layer
                h = outputs.hidden_states[layer_idx][0, -1, :]
                target_embeddings.append(h.detach())
    
    return torch.stack(target_embeddings)


def generate_with_ablation(
    model,
    tokenizer,
    input_ids: Tensor,
    direction: Tensor,
    max_new_tokens: int = 64,
    attention_mask: Optional[Tensor] = None,
) -> str:
    """
    Generate text with refusal ablation applied.
    
    Returns:
        Generated text (excluding the prompt)
    """
    from transformers import GenerationConfig
    
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    with ablation_context(model, direction):
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
    
    # Remove prompt tokens
    generated_tokens = outputs[0, input_ids.size(1):]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def compare_generations(
    model,
    tokenizer,
    instruction: str,
    direction: Tensor,
    tokenize_fn,
    max_new_tokens: int = 64,
) -> Tuple[str, str]:
    """
    Compare baseline and ablated generations for an instruction.
    
    Returns:
        Tuple of (baseline_generation, ablated_generation)
    """
    from transformers import GenerationConfig
    
    # Tokenize
    inputs = tokenize_fn(instructions=[instruction])
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Baseline generation
    with torch.no_grad():
        baseline_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
    baseline_tokens = baseline_outputs[0, input_ids.size(1):]
    baseline_text = tokenizer.decode(baseline_tokens, skip_special_tokens=True)
    
    # Ablated generation
    ablated_text = generate_with_ablation(
        model, tokenizer, input_ids, direction, 
        max_new_tokens, attention_mask
    )
    
    return baseline_text, ablated_text


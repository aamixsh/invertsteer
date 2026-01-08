"""
Inversion module adapted from SIP-It for inverting steered activations.

This module implements the core SIP-It algorithm for inverting hidden states
back to prompts. It supports inverting both baseline and steered activations.

Key difference from original SIP-It:
- We invert at the steering layer (not the last layer)
- No need to replace layer norm with identity
"""

import gc
import torch
from time import time
from typing import List, Tuple, Optional
from torch import Tensor

from model_utils import set_seed


def format_token(token: str, length: int = 15) -> str:
    """Format token for display, escaping non-printable characters."""
    result = []
    for ch in token:
        code = ord(ch)
        if 32 <= code <= 126:
            result.append(ch)
        elif ch == '\n':
            result.append('\\n')
        elif ch == '\r':
            result.append('\\r')
        elif ch == '\t':
            result.append('\\t')
        elif code < 128:
            result.append(f'\\x{code:02x}')
        else:
            result.append(''.join(f'\\x{b:02x}' for b in ch.encode('utf-8')))
    
    token_str = ''.join(result)
    if len(token_str) > length:
        token_str = token_str[:length - 3] + '...'
    return token_str


def extract_hidden_states_iterative(
    input_ids: Tensor,
    model,
    layer_idx: int,
) -> Tensor:
    """
    Extract hidden states iteratively for each prefix of the input.
    
    For each position i, computes the hidden state of token i when
    the input is tokens 0..i.
    
    Args:
        input_ids: Input token IDs [1, seq_len]
        model: The model
        layer_idx: Layer to extract from (1-indexed, where 0 is embedding)
    
    Returns:
        Hidden states [seq_len, d_model]
    """
    embeddings = model.get_input_embeddings().weight[input_ids.squeeze(0)]
    embeddings = embeddings.detach().unsqueeze(0)  # [1, seq_len, d_model]
    
    target_embeddings = []
    
    with torch.no_grad():
        for i in range(1, embeddings.size(1) + 1):
            outputs = model(
                inputs_embeds=embeddings[:, :i, :],
                output_hidden_states=True,
                use_cache=False,
            )
            h = outputs.hidden_states[layer_idx][0, -1, :]
            target_embeddings.append(h.detach())
    
    return torch.stack(target_embeddings)


def compute_gradient_and_loss(
    cont_embeddings: Tensor,
    disc_tokens: Tensor,
    model,
    layer_idx: int,
    h_target: Tensor,
) -> Tuple[Tensor, float]:
    """
    Compute gradient for the last token embedding and discrete token loss.
    
    Args:
        cont_embeddings: Continuous embeddings [1, seq_len, d_model]
        disc_tokens: Discrete token IDs [1, seq_len]
        model: The model
        layer_idx: Target layer
        h_target: Target hidden state [d_model]
    
    Returns:
        Tuple of (gradient [d_model], loss value)
    """
    device = next(model.parameters()).device
    
    cont_embeddings = cont_embeddings.to(device)
    disc_tokens = disc_tokens.to(device)
    h_target = h_target.to(device)
    
    # Prepare embeddings with grad only on last token
    fixed_embs = cont_embeddings.clone().detach()
    last_emb = fixed_embs[:, -1:, :].clone().requires_grad_(True)
    
    inputs_embeds = torch.cat([fixed_embs[:, :-1, :], last_emb], dim=1)
    
    # Forward with continuous embeddings
    outputs = model(
        inputs_embeds=inputs_embeds,
        output_hidden_states=True,
    )
    h_last_cont = outputs.hidden_states[layer_idx][0, -1, :]
    
    # Forward with discrete tokens (no grad)
    with torch.no_grad():
        outputs_disc = model(
            input_ids=disc_tokens,
            output_hidden_states=True,
        )
    h_last_disc = outputs_disc.hidden_states[layer_idx][0, -1, :].detach()
    
    # Compute losses
    loss_cont = torch.nn.functional.mse_loss(h_last_cont, h_target, reduction='mean')
    loss_disc = torch.nn.functional.mse_loss(h_last_disc, h_target, reduction='sum')
    
    loss_cont.backward()
    
    return last_emb.grad.squeeze(0, 1), loss_disc.item()


def find_token(
    token_idx: int,
    embedding_matrix: Tensor,
    discovered_embeddings: List[Tensor],
    discovered_ids: List[int],
    model,
    tokenizer,
    layer_idx: int,
    h_target: Tensor,
    lr: float,
    verbose: bool = True,
) -> Tuple[Optional[int], Optional[Tensor], int]:
    """
    Find the next token that minimizes the distance to target hidden state.
    
    Uses gradient-guided search over the embedding matrix.
    
    Args:
        token_idx: Current token position
        embedding_matrix: Token embedding matrix [vocab_size, d_model]
        discovered_embeddings: List of discovered token embeddings so far
        discovered_ids: List of discovered token IDs so far
        model: The model
        tokenizer: The tokenizer
        layer_idx: Target layer
        h_target: Target hidden states [seq_len, d_model]
        lr: Learning rate for gradient step
        verbose: Whether to print progress
    
    Returns:
        Tuple of (token_id, token_embedding, final_timestep)
    """
    device = next(model.parameters()).device
    
    copy_embedding_matrix = embedding_matrix.clone().detach().requires_grad_(False)
    
    # Random initialization
    token_id = torch.randint(0, embedding_matrix.size(0), (1,)).item()
    embedding = copy_embedding_matrix[token_id].clone().requires_grad_(True)
    temp_embedding = copy_embedding_matrix[token_id].clone().detach()
    
    optimizer = torch.optim.SGD([embedding], lr=lr)
    
    initial_desc = f'Token [{token_idx + 1:2d}/{h_target.size(0):2d}]'
    final_timestep = embedding_matrix.size(0)
    start_time = time()
    
    for i in range(embedding_matrix.size(0)):
        # Build input embeddings
        input_embeddings = torch.stack(
            discovered_embeddings + [embedding]
        ).unsqueeze(0)
        input_ids = torch.tensor(
            discovered_ids + [token_id]
        ).unsqueeze(0)
        
        # Compute gradient and loss
        grad_oracle, loss = compute_gradient_and_loss(
            input_embeddings,
            input_ids,
            model,
            layer_idx,
            h_target[token_idx]
        )
        
        if torch.isnan(torch.tensor(loss)) or torch.isnan(grad_oracle).any():
            return None, None, 0
        
        grad_norm = grad_oracle.norm().item()
        curr_token = tokenizer.decode([token_id], skip_special_tokens=True)
        emb_norm = embedding.norm().item()
        
        if verbose and (i + 1) % 100 == 0:
            print(f"\r{initial_desc}[{i + 1:5d}/{embedding_matrix.size(0):5d}]: "
                  f"Loss: {loss:.2e} - Grad: {grad_norm:.2e} - "
                  f"Token: {format_token(curr_token):15s} - "
                  f"Time: {time() - start_time:.1f}s", end="")
        
        # Check convergence
        if loss < 1e-5 or grad_norm < 1e-12:
            final_timestep = i + 1
            break
        
        # Normalize gradient if too large
        if grad_norm > 1:
            grad_oracle = grad_oracle / grad_norm
        
        embedding.grad = grad_oracle
        optimizer.step()
        
        # Mark current token as visited
        copy_embedding_matrix[token_id] = float('inf')
        
        # Find nearest token
        distances = torch.norm(copy_embedding_matrix - embedding, dim=1)
        token_id = int(torch.argmin(distances))
        temp_embedding = copy_embedding_matrix[token_id].clone()
        
        # Periodically reset embedding to discrete value
        if (i + 1) % 50 == 0:
            embedding.data = temp_embedding.data
    
    if verbose:
        print()  # Newline after progress
    
    return token_id, copy_embedding_matrix[token_id], final_timestep


def find_prompt(
    model,
    tokenizer,
    layer_idx: int,
    h_target: Tensor,
    lr: float,
    verbose: bool = True,
) -> Tuple[Optional[float], Optional[List[int]], Optional[List[int]], Optional[List[float]]]:
    """
    Find a prompt that produces the target hidden states.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        layer_idx: Target layer
        h_target: Target hidden states [seq_len, d_model]
        lr: Learning rate
        verbose: Whether to print progress
    
    Returns:
        Tuple of (total_time, token_ids, timesteps_per_token, time_per_token)
    """
    embedding_matrix = model.get_input_embeddings().weight
    
    if h_target.dim() == 1:
        h_target = h_target.unsqueeze(0)
    
    discovered_embeddings = []
    discovered_ids = []
    timesteps = []
    times = []
    
    start_time = time()
    
    for i in range(h_target.size(0)):
        token_start_time = time()
        
        next_token_id, next_token_embedding, final_timestep = find_token(
            i, embedding_matrix,
            discovered_embeddings, discovered_ids,
            model, tokenizer, layer_idx, h_target,
            lr, verbose
        )
        
        token_end_time = time()
        
        if next_token_embedding is None:
            return None, None, None, None
        
        discovered_embeddings.append(next_token_embedding)
        discovered_ids.append(next_token_id)
        timesteps.append(final_timestep)
        times.append(token_end_time - token_start_time)
        
        gc.collect()
        torch.cuda.empty_cache()
    
    end_time = time()
    
    return end_time - start_time, discovered_ids, timesteps, times


def inversion_attack(
    input_ids: Tensor,
    model,
    tokenizer,
    layer_idx: int,
    lr: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[bool, Optional[float], Optional[List[int]], Optional[List[float]]]:
    """
    Perform standard inversion attack (no steering).
    
    Args:
        input_ids: Target token IDs [1, seq_len]
        model: The model
        tokenizer: The tokenizer
        layer_idx: Layer to target for inversion
        lr: Learning rate
        seed: Random seed
        verbose: Whether to print progress
    
    Returns:
        Tuple of (success, time, reconstructed_ids, times)
    """
    set_seed(seed)
    
    # Extract target hidden states
    h_target = extract_hidden_states_iterative(input_ids, model, layer_idx)
    
    # Find prompt
    inversion_time, discovered_ids, timesteps, times = find_prompt(
        model, tokenizer, layer_idx, h_target, lr, verbose
    )
    
    if discovered_ids is None:
        if verbose:
            print("Inversion failed or diverged.")
        return False, None, None, None
    
    # Check if we found exact match
    match = all([x == y for x, y in zip(input_ids[0].tolist(), discovered_ids)])
    
    if verbose:
        original = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        reconstructed = tokenizer.decode(discovered_ids, skip_special_tokens=True)
        print(f"Original:      {original[:100]}...")
        print(f"Reconstructed: {reconstructed[:100]}...")
        print(f"Match: {match}")
        print(f"Time: {inversion_time:.2f}s")
    
    return match, inversion_time, discovered_ids, times


def inversion_attack_with_target(
    h_target: Tensor,
    model,
    tokenizer,
    layer_idx: int,
    lr: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
    original_ids: Optional[Tensor] = None,
) -> Tuple[bool, Optional[float], Optional[List[int]], Optional[List[float]]]:
    """
    Perform inversion attack with custom target hidden states.
    
    This is the key function for inverting steered activations.
    
    Args:
        h_target: Target hidden states [seq_len, d_model]
        model: The model
        tokenizer: The tokenizer
        layer_idx: Layer to target for inversion
        lr: Learning rate
        seed: Random seed
        verbose: Whether to print progress
        original_ids: Optional original token IDs for comparison
    
    Returns:
        Tuple of (match_original, time, discovered_ids, times)
    """
    set_seed(seed)
    
    # Find prompt
    inversion_time, discovered_ids, timesteps, times = find_prompt(
        model, tokenizer, layer_idx, h_target, lr, verbose
    )
    
    if discovered_ids is None:
        if verbose:
            print("Inversion failed or diverged.")
        return False, None, None, None
    
    # Check if we match original (if provided)
    match = False
    if original_ids is not None:
        match = all([x == y for x, y in zip(original_ids[0].tolist(), discovered_ids)])
    
    if verbose:
        reconstructed = tokenizer.decode(discovered_ids, skip_special_tokens=True)
        print(f"Reconstructed: {reconstructed[:100]}...")
        if original_ids is not None:
            original = tokenizer.decode(original_ids[0], skip_special_tokens=True)
            print(f"Original:      {original[:100]}...")
            print(f"Match: {match}")
        print(f"Time: {inversion_time:.2f}s")
    
    return match, inversion_time, discovered_ids, times


def compute_activation_mse(
    model,
    input_ids: Tensor,
    target_activations: Tensor,
    layer_idx: int,
) -> float:
    """
    Compute MSE between activations produced by input_ids and target activations.
    
    Args:
        model: The model
        input_ids: Token IDs [1, seq_len]
        target_activations: Target hidden states [seq_len, d_model]
        layer_idx: Layer to compare at
    
    Returns:
        MSE value
    """
    # Get actual activations
    actual_acts = extract_hidden_states_iterative(input_ids, model, layer_idx)
    
    # Compute MSE
    mse = torch.nn.functional.mse_loss(actual_acts, target_activations).item()
    
    return mse

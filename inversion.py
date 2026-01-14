"""
Inversion module for inverting hidden states back to prompts.

This module implements the SIP-It algorithm for inverting hidden states
back to prompts. It supports inverting both baseline and steered activations.
"""

import gc
import torch
from time import time
from typing import List, Tuple, Optional, NamedTuple
from torch import Tensor
from time import strftime, gmtime
from dataclasses import dataclass
import numpy as np


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_time_minutes(seconds: float) -> str:
    return strftime("%M:%S", gmtime(seconds))


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


class ExhaustiveOptimizer:
    """Dummy optimizer for baseline exhaustive search."""
    def __init__(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass




def compute_grad_and_elim(
    embeddings: tuple[torch.Tensor, torch.Tensor],
    model: torch.nn.Module,
    layer_idx: int,
    h_target: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    # Move to device
    device = next(model.parameters()).device


    cont_embeddings = embeddings[0].to(device)
    disc_tokens     = embeddings[1].to(device)
    h_target = h_target.to(device)

    fixed_embs = cont_embeddings.clone().detach()
    last_emb = fixed_embs[:, -1:, :].clone().requires_grad_(True)

    inputs_embeds_cont = torch.cat([fixed_embs[:, :-1, :], last_emb], dim=1)
    outputs = model(
        inputs_embeds=inputs_embeds_cont,
        output_hidden_states=True
    )
    hidden_states = outputs.hidden_states
    h_last_cont = hidden_states[layer_idx][0, -1, :]

    with torch.no_grad():
        outputs = model(
            input_ids=disc_tokens,
            output_hidden_states=True,
            use_cache=False
        )
    hidden_states = outputs.hidden_states
    h_last_disc = hidden_states[layer_idx][0, -1, :].detach()


    # Compute MSE loss for last token
    # loss_cont = torch.nn.functional.mse_loss(h_last_cont, h_target, reduction='sum')
    loss_cont = torch.nn.functional.mse_loss(h_last_cont, h_target, reduction='mean')
    loss_disc = torch.nn.functional.mse_loss(h_last_disc, h_target, reduction='sum')
    
    loss_cont.backward()
    return last_emb.grad.squeeze(0, 1), loss_disc

@dataclass
class TokenSearchResult:
    """Result of searching for a single token."""
    token_id: Optional[int]
    embedding: Optional[Tensor]
    timesteps: int
    top_k_tokens: List[Tuple[int, float]]  # List of (token_id, distance) pairs
    success: bool


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
    scheduler: bool = False,
    verbose: bool = True,
    baseline: bool = False,
    top_k: int = 10,
) -> TokenSearchResult:
    """
    Find a single token that best matches the target hidden state.
    
    Args:
        token_idx: Index of the token being searched for
        embedding_matrix: The model's embedding matrix
        discovered_embeddings: Embeddings of already discovered tokens
        discovered_ids: IDs of already discovered tokens
        model: The model
        tokenizer: The tokenizer
        layer_idx: Layer to target
        h_target: Target hidden states [seq_len, d_model]
        lr: Learning rate
        scheduler: Whether to use learning rate scheduler
        verbose: Whether to print progress
        baseline: Whether to use exhaustive baseline search
        top_k: Number of top candidates to track
    
    Returns:
        TokenSearchResult containing the found token info and top-k candidates
    """
    device = next(model.parameters()).device
    copy_embedding_matrix = embedding_matrix.clone().detach().requires_grad_(False)

    reset_extra = embedding_matrix.size(0) // 25_000

    if baseline:
        perm = torch.randperm(copy_embedding_matrix.size(0))
        copy_embedding_matrix = copy_embedding_matrix[perm]

    token_id = torch.randint(0, embedding_matrix.size(0), (1,)).item()
    
    embedding = copy_embedding_matrix[token_id].clone().requires_grad_(True)
    temp_embedding = copy_embedding_matrix[token_id].clone().detach()

    optimizer = torch.optim.SGD([embedding], lr=lr) if not baseline else ExhaustiveOptimizer()
    if scheduler:
        scheduler_obj = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.99, threshold=lr / 100, patience=200
        )

    initial_desc = f'Token [{token_idx + 1:2d}/{h_target.size(0):2d}]'
    final_timestep = embedding_matrix.size(0)
    start_time = time()
    
    # Track top-k candidates: (token_id, loss)
    top_k_candidates: List[Tuple[int, float]] = []
    
    for i in range(embedding_matrix.size(0)):
        input_embeddings = torch.stack(
            discovered_embeddings + [embedding]
        ).unsqueeze(0) 
        input_embeddings_disc = torch.tensor(
            discovered_ids + [token_id]
        ).unsqueeze(0) 

        grad_oracle = loss = torch.zeros_like(h_target[token_idx])

        if baseline:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_embeddings_disc.to(device),
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states
            
            h_pred = hidden_states[layer_idx][0, -1, :].detach()
            loss = torch.nn.functional.mse_loss(h_pred, h_target[token_idx], reduction='sum')
        else:
            grad_oracle, loss = compute_grad_and_elim(
                (input_embeddings, input_embeddings_disc),
                model=model,
                layer_idx=layer_idx, 
                h_target=h_target[token_idx]
            )

        # Check for NaN
        if torch.isnan(loss) or (not baseline and torch.isnan(grad_oracle).any()):
            return TokenSearchResult(
                token_id=None,
                embedding=None,
                timesteps=i,
                top_k_tokens=top_k_candidates,
                success=False
            )

        loss_val = loss if isinstance(loss, float) else loss.item()
        
        # Update top-k candidates
        top_k_candidates.append((token_id, loss_val))
        top_k_candidates.sort(key=lambda x: x[1])
        top_k_candidates = top_k_candidates[:top_k]

        grad_norm = grad_oracle.norm().item() if not baseline else 0.0
        curr_token = tokenizer.decode([token_id])
        
        emb_norm = embedding.norm().item()
        if verbose:
            print(
                '\r{}[{:5d}/{:5d}]: Loss: {:.2e} - Gradient norm: {:.2e} - Token: {:15s} - Emb Norm: {:.2e} - Time: {}'.format(
                    initial_desc, 
                    i + 1, 
                    embedding_matrix.size(0),
                    loss_val, 
                    grad_norm, 
                    format_token(curr_token, length=15), 
                    emb_norm,
                    format_time_minutes(time() - start_time)       
                ),
                end=''
            )

        if loss_val < 1e-5 or (not baseline and grad_norm < 1e-12):
            final_timestep = i + 1
            break

        if not baseline and grad_norm > 1:
            grad_oracle = grad_oracle / grad_norm

        embedding.grad = grad_oracle

        optimizer.step(lambda: loss)
        if scheduler:
            scheduler_obj.step(loss)

        copy_embedding_matrix[token_id] = float('inf')
        distances = torch.norm(copy_embedding_matrix - embedding, dim=1)
        token_id = int(torch.argmin(distances))
        temp_embedding = copy_embedding_matrix[token_id].clone()

        # if not baseline and (i + 1) % (50 * (1 + reset_extra * int(token_idx == 0))) == 0:
        if not baseline and (i + 1) % 50 == 0:
            embedding.data = temp_embedding.data

        if (i + 1) % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if verbose:
        print()

    correct_embedding = copy_embedding_matrix[token_id].clone().detach()
    del copy_embedding_matrix, embedding, temp_embedding
    
    return TokenSearchResult(
        token_id=token_id,
        embedding=correct_embedding,
        timesteps=final_timestep,
        top_k_tokens=top_k_candidates,
        success=True
    )


@dataclass
class InversionResult:
    """Result of a prompt inversion."""
    success: bool
    total_time: Optional[float]
    discovered_ids: Optional[List[int]]
    timesteps: Optional[List[int]]
    times: Optional[List[float]]
    top_k_per_position: List[List[Tuple[int, float]]]  # Top-k tokens per position
    failed_positions: List[int]  # Positions where inversion failed


def find_prompt(
    model,
    tokenizer,
    layer_idx: int,
    h_target: Tensor,
    lr: float,
    scheduler: bool = False,
    verbose: bool = True,
    baseline: bool = False,
    special_start_tokens: Optional[List[int]] = None,
    continue_on_failure: bool = False,
    ground_truth_ids: Optional[List[int]] = None,
    top_k: int = 10,
) -> InversionResult:
    """
    Find a prompt that produces the target hidden states.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        layer_idx: Layer to target
        h_target: Target hidden states [seq_len, d_model]
        lr: Learning rate
        scheduler: Whether to use LR scheduler
        verbose: Whether to print progress
        baseline: Whether to use exhaustive baseline search
        special_start_tokens: Special tokens to prepend (e.g., [128000])
        continue_on_failure: If True, continue with ground truth token when inversion fails
        ground_truth_ids: Ground truth token IDs (required if continue_on_failure=True)
        top_k: Number of top candidate tokens to track per position
    
    Returns:
        InversionResult containing discovered tokens and metadata
    """
    embedding_matrix = model.get_input_embeddings().weight

    if h_target.dim() == 1:
        h_target = h_target.unsqueeze(0)

    discovered_embeddings = []
    discovered_ids = []
    timesteps = []
    times = []
    top_k_per_position = []
    failed_positions = []

    if special_start_tokens is not None:
        for token_id in special_start_tokens:
            discovered_ids.append(token_id)
            discovered_embeddings.append(
                embedding_matrix[token_id]
                .clone()
                .detach()
                .requires_grad_(False)
            )

    start_time = time()
    for i in range(h_target.size(0)):
        token_start_time = time()

        result = find_token(
            i, embedding_matrix, 
            discovered_embeddings, discovered_ids,
            model, tokenizer, layer_idx, h_target,
            lr, scheduler, verbose, baseline,
            top_k=top_k,
        )

        token_end_time = time()
        
        # Store top-k for this position regardless of success
        top_k_per_position.append(result.top_k_tokens)

        if not result.success or result.embedding is None:
            failed_positions.append(i)
            
            if continue_on_failure and ground_truth_ids is not None:
                # Use ground truth token for this position
                gt_token_id = ground_truth_ids[i]
                gt_embedding = embedding_matrix[gt_token_id].clone().detach().requires_grad_(False)
                
                discovered_embeddings.append(gt_embedding)
                discovered_ids.append(gt_token_id)
                timesteps.append(result.timesteps)
                times.append(token_end_time - token_start_time)
                
                if verbose:
                    print(f"  [Using ground truth token {gt_token_id} for position {i}]")
                continue
            else:
                # Return with partial results
                return InversionResult(
                    success=False,
                    total_time=time() - start_time,
                    discovered_ids=discovered_ids,
                    timesteps=timesteps,
                    times=times,
                    top_k_per_position=top_k_per_position,
                    failed_positions=failed_positions,
                )

        discovered_embeddings.append(result.embedding)
        discovered_ids.append(result.token_id)
        timesteps.append(result.timesteps)
        times.append(token_end_time - token_start_time)

        gc.collect()
        torch.cuda.empty_cache()

    end_time = time()

    return InversionResult(
        success=len(failed_positions) == 0,
        total_time=end_time - start_time,
        discovered_ids=discovered_ids,
        timesteps=timesteps,
        times=times,
        top_k_per_position=top_k_per_position,
        failed_positions=failed_positions,
    )


def inversion_attack(
    input_ids: Tensor, 
    model, 
    tokenizer, 
    layer_idx: int,
    lr: float,
    seed: int = 8, 
    scheduler: bool = False,
    verbose: bool = True,
    baseline: bool = False,
    special_start_tokens: Optional[List[int]] = None,
    continue_on_failure: bool = False,
    top_k: int = 10,
) -> Tuple[bool, Optional[float], Optional[List[int]], Optional[List[float]], InversionResult]:
    """
    Perform inversion attack on input IDs.
    
    Args:
        input_ids: Input token IDs [1, seq_len]
        model: The model
        tokenizer: The tokenizer
        layer_idx: Layer to target
        lr: Learning rate
        seed: Random seed
        scheduler: Whether to use LR scheduler
        verbose: Whether to print progress
        baseline: Whether to use exhaustive baseline
        special_start_tokens: Special tokens to prepend
        continue_on_failure: If True, continue with ground truth when inversion fails
        top_k: Number of top candidate tokens to track
    
    Returns:
        Tuple of (match, time, discovered_ids, times, full_result)
    """
    set_seed(seed)
    
    # new_input_ids = (
    #     torch.cat(
    #         [
    #             torch.tensor(
    #                 special_start_tokens, 
    #                 dtype=torch.long, 
    #                 device=input_ids.device
    #             ).unsqueeze(0), 
    #             input_ids
    #         ],
    #         dim=1
    #     ) if special_start_tokens is not None else
    #     input_ids
    # )

    # h_target = extract_hidden_states_iterative(new_input_ids, model, layer_idx)
    # start_from = new_input_ids.size(1) - input_ids.size(1)
    h_target = extract_hidden_states_iterative(input_ids, model, layer_idx)
    start_from = 0

    # Extract ground truth IDs for continue_on_failure mode
    ground_truth_ids = input_ids[0].tolist() if continue_on_failure else None

    result = find_prompt(
        model, tokenizer, layer_idx, h_target[start_from:], 
        lr, scheduler, verbose, baseline, special_start_tokens,
        continue_on_failure=continue_on_failure,
        ground_truth_ids=ground_truth_ids,
        top_k=top_k,
    )
    
    if result.discovered_ids is None:
        if verbose:
            print('Inversion failed or diverged with the given parameters.')
        return False, None, None, None, result

    match = all([x == y for x, y in zip(input_ids[0].tolist(), result.discovered_ids[start_from:])])
    
    if verbose:
        print(f'Original {"==" if match else "!="} Reconstructed')
        print(f'Inversion time: {result.total_time:.2f} seconds')
        if result.timesteps:
            print(f'Average Timesteps: {np.mean(result.timesteps):.2f}')
        if result.failed_positions:
            print(f'Failed positions: {result.failed_positions}')

    return match, result.total_time, result.discovered_ids, result.times, result


def inversion_attack_with_target(
    h_target: Tensor,
    model,
    tokenizer,
    layer_idx: int,
    lr: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
    original_ids: Optional[Tensor] = None,
    special_start_tokens: Optional[List[int]] = None,
    continue_on_failure: bool = False,
    top_k: int = 10,
) -> Tuple[bool, Optional[float], Optional[List[int]], Optional[List[float]], InversionResult]:
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
        special_start_tokens: Special tokens to prepend
        continue_on_failure: If True, continue with ground truth when inversion fails
        top_k: Number of top candidate tokens to track
    
    Returns:
        Tuple of (match_original, time, discovered_ids, times, full_result)
    """
    set_seed(seed)
    
    # Extract ground truth IDs for continue_on_failure mode
    ground_truth_ids = original_ids[0].tolist() if continue_on_failure and original_ids is not None else None
    
    # Find prompt
    result = find_prompt(
        model, tokenizer, layer_idx, h_target, lr,
        verbose=verbose,
        special_start_tokens=special_start_tokens,
        continue_on_failure=continue_on_failure,
        ground_truth_ids=ground_truth_ids,
        top_k=top_k,
    )
    
    if result.discovered_ids is None:
        if verbose:
            print("Inversion failed or diverged.")
        return False, None, None, None, result
    
    # Check if we match original (if provided)
    match = False
    if original_ids is not None:
        match = all([x == y for x, y in zip(original_ids[0].tolist(), result.discovered_ids)])
    
    if verbose:
        reconstructed = tokenizer.decode(result.discovered_ids, skip_special_tokens=True)
        print(f"Reconstructed: {reconstructed[:100]}...")
        if original_ids is not None:
            original = tokenizer.decode(original_ids[0].tolist(), skip_special_tokens=True)
            print(f"Original:      {original[:100]}...")
            print(f"Match: {match}")
        print(f"Time: {result.total_time:.2f}s")
        if result.failed_positions:
            print(f"Failed positions: {result.failed_positions}")
    
    return match, result.total_time, result.discovered_ids, result.times, result


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


def print_top_k_tokens(
    result: InversionResult,
    tokenizer,
    original_ids: Optional[List[int]] = None,
) -> None:
    """
    Print the top-k tokens found at each position during inversion.
    
    Args:
        result: The inversion result
        tokenizer: The tokenizer
        original_ids: Optional original token IDs for comparison
    """
    print("\n" + "="*60)
    print("TOP-K TOKENS PER POSITION")
    print("="*60)
    
    for pos, top_k in enumerate(result.top_k_per_position):
        gt_marker = ""
        if original_ids is not None and pos < len(original_ids):
            gt_id = original_ids[pos]
            gt_token = tokenizer.decode([gt_id])
            gt_marker = f" (GT: {format_token(gt_token)})"
        
        failed_marker = " [FAILED]" if pos in result.failed_positions else ""
        print(f"\nPosition {pos}{gt_marker}{failed_marker}:")
        
        for rank, (token_id, loss) in enumerate(top_k):
            token_str = tokenizer.decode([token_id])
            is_gt = original_ids is not None and pos < len(original_ids) and token_id == original_ids[pos]
            gt_flag = " <-- GT" if is_gt else ""
            print(f"  {rank+1:2d}. [{token_id:6d}] {format_token(token_str, 20):20s} loss={loss:.4e}{gt_flag}")

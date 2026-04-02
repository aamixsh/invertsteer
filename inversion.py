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

try:
    from evotorch import Problem
    from evotorch.algorithms import CMAES
    EVOTORCH_AVAILABLE = True
except ImportError:
    EVOTORCH_AVAILABLE = False


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


def extract_hidden_states(
    input_ids: Tensor,
    model,
    layer_idx: int,
) -> Tensor:
    """
    Extract hidden states for all tokens in a sequence of input IDs.
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False
        )
        hidden_states = outputs.hidden_states
        h = hidden_states[layer_idx][0]
        return h.detach()


class ExhaustiveOptimizer:
    """Dummy optimizer for baseline exhaustive search."""
    def __init__(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

@dataclass
class TokenSearchResult:
    """Result of searching for a single token."""
    token_id: Optional[int]
    embedding: Optional[Tensor]
    timesteps: int
    top_k_tokens: List[Tuple[int, float]]  # List of (token_id, distance) pairs
    success: bool

def _maybe_wrap_data_parallel(model, n_gpus: int):
    if n_gpus is None or n_gpus <= 1:
        return model
    if not torch.cuda.is_available():
        return model
    if isinstance(model, torch.nn.DataParallel):
        return model
    # Models loaded with accelerate device_map="auto" are already sharded across GPUs.
    # Wrapping them in DataParallel replicates the full module on GPU0 and usually OOMs.
    if getattr(model, "hf_device_map", None):
        return model

    device = next(model.parameters()).device
    if device.type != "cuda":
        return model

    available_gpus = torch.cuda.device_count()
    use_gpus = min(n_gpus, available_gpus)
    if use_gpus <= 1:
        return model

    device_ids = list(range(use_gpus))
    return torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])

def _get_embedding_matrix(model) -> Tensor:
    if isinstance(model, torch.nn.DataParallel):
        return model.module.get_input_embeddings().weight
    return model.get_input_embeddings().weight

def _find_token_cmaes(
    token_idx: int,
    embedding_matrix: Tensor,
    discovered_embeddings: List[Tensor],
    discovered_ids: List[int],
    model,
    tokenizer,
    layer_idx: int,
    h_target: Tensor,
    verbose: bool,
    top_k: int,
    batch_size: int,
    max_iterations: int = 1000,
    stdev_init: float = 0.1,
) -> TokenSearchResult:
    """
    Find token using CMA-ES evolutionary search over discrete token space.
    
    The search is over the discrete set of all tokens. Continuous embeddings are
    projected to nearest token embeddings, and MSE is calculated using token embeddings.
    
    Args:
        token_idx: Index of the token being searched for
        embedding_matrix: The model's embedding matrix
        discovered_embeddings: Embeddings of already discovered tokens
        discovered_ids: IDs of already discovered tokens
        model: The model
        tokenizer: The tokenizer
        layer_idx: Layer to target
        h_target: Target hidden states [seq_len, d_model]
        verbose: Whether to print progress
        top_k: Number of top candidates to track
        batch_size: Population size for CMA-ES (larger = more parallel evaluation)
        max_iterations: Maximum number of CMA-ES iterations
        stdev_init: Initial standard deviation for CMA-ES
    
    Returns:
        TokenSearchResult containing the found token info and top-k candidates
    """
    if not EVOTORCH_AVAILABLE:
        raise ImportError("EvoTorch is not installed. Please install it with: pip install evotorch")
    
    device = next(model.parameters()).device
    embedding_dim = embedding_matrix.size(1)
    num_tokens = embedding_matrix.size(0)
    h_target_token = h_target[token_idx].to(device)
    
    # Track visited tokens to avoid re-evaluating them
    visited_tokens: set = set()
    
    # Track all evaluated candidates: (token_id, distance)
    all_candidates: set = set()
    best_token_id = None
    best_distance = float("inf")
    
    initial_desc = f'Token [{token_idx + 1:2d}/{h_target.size(0):2d}]'
    start_time = time()
    
    # Initialize population with random token embeddings (not visited)
    available_tokens = [i for i in range(num_tokens) if i not in visited_tokens]
    if len(available_tokens) < batch_size:
        # If we don't have enough tokens, use all available
        initial_token_ids = available_tokens[:batch_size]
    else:
        # Randomly sample batch_size tokens
        initial_token_ids = torch.randperm(len(available_tokens), device=device)[:batch_size].tolist()
        initial_token_ids = [available_tokens[i] for i in initial_token_ids]
    
    # Evaluate initial population tokens
    if initial_token_ids:
        initial_token_embeddings = embedding_matrix[initial_token_ids]
        input_embeddings_list = []
        for token_emb in initial_token_embeddings:
            if discovered_embeddings:
                full_emb = torch.stack(discovered_embeddings + [token_emb]).unsqueeze(0)
            else:
                full_emb = token_emb.unsqueeze(0).unsqueeze(0)
            input_embeddings_list.append(full_emb)
        
        all_input_embeds = torch.cat(input_embeddings_list, dim=0)
        
        with torch.no_grad():
            outputs = model(
                inputs_embeds=all_input_embeds,
                output_hidden_states=True,
                use_cache=False,
            )
        
        h_pred = outputs.hidden_states[layer_idx][:, -1, :]
        target_expanded = h_target_token.unsqueeze(0).expand(len(initial_token_ids), -1)
        mse_losses = torch.nn.functional.mse_loss(
            h_pred, target_expanded, reduction='none'
        ).mean(dim=1)
        
        # Track initial candidates
        for token_id, mse_loss in zip(initial_token_ids, mse_losses):
            distance_val = float(mse_loss.item())
            all_candidates.add((token_id, distance_val))
            visited_tokens.add(token_id)
            
            if distance_val < best_distance:
                best_distance = distance_val
                best_token_id = token_id
        
        evaluated = len(initial_token_ids)
    else:
        evaluated = 0
    
    # Check if we already found a good match in initial population
    if best_distance < 1e-5:
        if verbose:
            print(f"\nFound good match in initial population.")
        final_top_k_candidates = sorted(all_candidates, key=lambda x: x[1])[:top_k]
        return TokenSearchResult(
            token_id=best_token_id,
            embedding=embedding_matrix[best_token_id].clone().detach() if best_token_id is not None else None,
            timesteps=evaluated,
            top_k_tokens=final_top_k_candidates,
            success=True,
        )
    
    # Check if all tokens have been visited
    if len(visited_tokens) >= num_tokens:
        if verbose:
            print(f"\nAll {num_tokens} tokens have been searched.")
        final_top_k_candidates = sorted(all_candidates, key=lambda x: x[1])[:top_k]
        return TokenSearchResult(
            token_id=best_token_id,
            embedding=embedding_matrix[best_token_id].clone().detach() if best_token_id is not None else None,
            timesteps=evaluated,
            top_k_tokens=final_top_k_candidates,
            success=best_distance < 1e-5 if best_distance != float("inf") else False,
        )
    
    # Initialize center from mean of initial token embeddings
    if initial_token_ids:
        initial_embeddings = embedding_matrix[initial_token_ids].float().cpu().numpy()
        center_init = initial_embeddings.mean(axis=0)
    else:
        # Fallback: use random token embedding
        initial_token_id = torch.randint(0, num_tokens, (1,), device=device).item()
        center_init = embedding_matrix[initial_token_id].clone().detach().float().cpu().numpy()
    
    def fitness_fn(solutions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate fitness (MSE loss) for a batch of embedding vectors.
        Projects continuous embeddings to nearest token embeddings before evaluation.
        
        Args:
            solutions: Batch of solution vectors [batch_size, embedding_dim] from EvoTorch
        
        Returns:
            MSE losses [batch_size] as a tensor
        """
        nonlocal all_candidates, best_token_id, best_distance, visited_tokens
        
        batch_size_eval = solutions.shape[0]
        device = next(model.parameters()).device
        
        # Convert to torch tensor and move to device
        embeddings_torch = solutions.float().to(device)
        
        # Project each continuous embedding to nearest unvisited token embedding
        token_embeddings_list = []
        token_ids_list = []
        valid_indices = []  # Indices of solutions that map to unvisited tokens
        
        for i in range(batch_size_eval):
            candidate_emb = embeddings_torch[i]
            # Find nearest token
            distances_to_tokens = torch.norm(embedding_matrix - candidate_emb, dim=1)
            
            # Find nearest unvisited token
            unvisited_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
            for visited_id in visited_tokens:
                unvisited_mask[visited_id] = False
            
            if unvisited_mask.any():
                # Get distances only for unvisited tokens
                unvisited_distances = distances_to_tokens.clone()
                unvisited_distances[~unvisited_mask] = float("inf")
                nearest_token_id = int(torch.argmin(unvisited_distances).item())
                
                # Mark as visited and add to evaluation list
                token_embeddings_list.append(embedding_matrix[nearest_token_id])
                token_ids_list.append(nearest_token_id)
                valid_indices.append(i)
                visited_tokens.add(nearest_token_id)
        
        if len(token_embeddings_list) == 0:
            # All tokens already visited, return high loss
            return torch.full((batch_size_eval,), float("inf"), device=device)
        
        # Prepare input embeddings for each valid candidate using token embeddings
        input_embeddings_list = []
        for token_emb in token_embeddings_list:
            # Stack discovered embeddings + token embedding
            if discovered_embeddings:
                full_emb = torch.stack(discovered_embeddings + [token_emb]).unsqueeze(0)
            else:
                full_emb = token_emb.unsqueeze(0).unsqueeze(0)
            input_embeddings_list.append(full_emb)
        
        # Batch process all valid candidates
        all_input_embeds = torch.cat(input_embeddings_list, dim=0)  # [valid_count, seq_len+1, emb_dim]
        
        with torch.no_grad():
            outputs = model(
                inputs_embeds=all_input_embeds,
                output_hidden_states=True,
                use_cache=False,
            )
        
        h_pred = outputs.hidden_states[layer_idx][:, -1, :]  # [valid_count, d_model]
        
        # Compute MSE loss for each valid candidate
        target_expanded = h_target_token.unsqueeze(0).expand(len(token_embeddings_list), -1)
        mse_losses_valid = torch.nn.functional.mse_loss(
            h_pred, target_expanded, reduction='none'
        ).mean(dim=1)  # [valid_count]
        
        # Track candidates and update best
        for idx, (token_id, mse_loss) in enumerate(zip(token_ids_list, mse_losses_valid)):
            distance_val = float(mse_loss.item())
            all_candidates.add((token_id, distance_val))
            
            # Update best
            if distance_val < best_distance:
                best_distance = distance_val
                best_token_id = token_id
        
        # Create full loss tensor, using inf for invalid/visited tokens
        full_losses = torch.full((batch_size_eval,), float("inf"), device=device)
        for valid_idx, orig_idx in enumerate(valid_indices):
            full_losses[orig_idx] = mse_losses_valid[valid_idx]
        
        return full_losses
    
    # Create EvoTorch problem
    problem = Problem(
        objective_sense="min",
        objective_func=fitness_fn,
        solution_length=embedding_dim,
        initial_bounds=(-2.0, 2.0),  # Reasonable bounds for embeddings
        device=device,
    )
    
    # Initialize CMA-ES with center from initial token embeddings
    searcher = CMAES(
        problem,
        stdev_init=stdev_init,
        popsize=batch_size,  # Population size
        center_init=center_init,
    )
    
    # Run CMA-ES optimization
    for iteration in range(max_iterations):
        # Check if all tokens have been visited
        if len(visited_tokens) >= num_tokens:
            if verbose:
                print(f"\nAll {num_tokens} tokens have been searched.")
            break
        
        searcher.step()
        evaluated += batch_size
        
        if verbose and (iteration % 10 == 0 or iteration == max_iterations - 1):
            curr_token_str = tokenizer.decode([best_token_id]) if best_token_id is not None else ""
            print(
                '\r{}[Iter {:4d}]: Best MSE: {:.2e} - Token: {:15s} - Visited: {:5d}/{:5d} - Eval: {:5d} - Time: {}'.format(
                    initial_desc,
                    iteration + 1,
                    best_distance,
                    format_token(curr_token_str, length=15),
                    len(visited_tokens),
                    num_tokens,
                    evaluated,
                    format_time_minutes(time() - start_time),
                ),
                end=''
            )
        
        # Early stopping if we found a very good match
        if best_distance < 1e-5:
            break
        
        # Periodic cleanup
        if (iteration + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    if verbose:
        print()
    
    # Find the final best token
    if best_token_id is not None:
        final_token_id = best_token_id
        final_embedding = embedding_matrix[final_token_id].clone().detach()
    else:
        # Fallback: use the token with minimum distance from all candidates
        if all_candidates:
            final_token_id, _ = min(all_candidates, key=lambda x: x[1])
            final_embedding = embedding_matrix[final_token_id].clone().detach()
        else:
            final_token_id = None
            final_embedding = None
    
    # Select top-k tokens from all candidates
    final_top_k_candidates = sorted(all_candidates, key=lambda x: x[1])[:top_k]
    
    success = best_distance < 1e-5 if best_distance != float("inf") else False
    
    return TokenSearchResult(
        token_id=final_token_id,
        embedding=final_embedding,
        timesteps=evaluated,
        top_k_tokens=final_top_k_candidates,
        success=success,
    )

def _score_discrete_candidates(
    discovered_ids: List[int],
    candidate_ids: Tensor,
    model,
    layer_idx: int,
    h_target_token: Tensor,
) -> Tensor:
    # IMPORTANT: this function is called repeatedly during token search.
    # The previous implementation requested `output_hidden_states=True`, which
    # materializes hidden states for *all* layers and OOMs quickly for large
    # candidate batches (e.g. 1024 candidates × ~30 tokens × 40 layers).
    #
    # Instead, we capture only the output of the targeted decoder layer using a
    # forward hook, and run the model without `output_hidden_states`.
    input_device = model.get_input_embeddings().weight.device
    candidate_ids = candidate_ids.to(input_device)

    if discovered_ids:
        prefix = torch.tensor(discovered_ids, device=input_device).unsqueeze(0)
        prefix = prefix.expand(candidate_ids.size(0), -1)
        input_ids = torch.cat([prefix, candidate_ids.unsqueeze(1)], dim=1)
    else:
        input_ids = candidate_ids.unsqueeze(1)

    # hidden_states[layer_idx] corresponds to output after decoder layer (layer_idx - 1)
    # because hidden_states[0] is the embedding output.
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Model structure not supported. Expected model.model.layers")
    decoder_layer = model.model.layers[layer_idx - 1]

    h_pred = None

    def _hook_fn(module, inp, out):
        nonlocal h_pred
        if isinstance(out, tuple):
            out = out[0]
        # out: [batch, seq_len, hidden]
        h_pred = out[:, -1, :]

    handle = decoder_layer.register_forward_hook(_hook_fn)
    try:
        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                output_hidden_states=False,
                use_cache=False,
            )
    finally:
        handle.remove()

    if h_pred is None:
        raise RuntimeError("Failed to capture target layer output via hook.")

    target = h_target_token.to(h_pred.device)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    distances = torch.norm(h_pred - target, dim=1)
    return distances


def _find_token_linear(
    token_idx: int,
    embedding_matrix: Tensor,
    discovered_ids: List[int],
    model,
    tokenizer,
    layer_idx: int,
    h_target: Tensor,
    verbose: bool,
    top_k: int,
    batch_size: int,
    shuffle_candidates: bool,
) -> TokenSearchResult:
    # For sharded models, ensure candidate batches are created on the input
    # embedding device (not an arbitrary parameter shard).
    device = model.get_input_embeddings().weight.device
    num_tokens = embedding_matrix.size(0)
    candidate_ids = torch.arange(num_tokens, device=device)

    if shuffle_candidates:
        candidate_ids = candidate_ids[torch.randperm(num_tokens, device=device)]

    initial_desc = f'Token [{token_idx + 1:2d}/{h_target.size(0):2d}]'
    start_time = time()
    all_candidates: set = set()

    best_token_id = None
    best_distance = float("inf")
    evaluated = 0

    batch_size = max(1, batch_size)
    for start in range(0, num_tokens, batch_size):
        batch_ids = candidate_ids[start:start + batch_size]
        distances = _score_discrete_candidates(
            discovered_ids=discovered_ids,
            candidate_ids=batch_ids,
            model=model,
            layer_idx=layer_idx,
            h_target_token=h_target[token_idx],
        )

        if torch.isnan(distances).any():
            final_top_k_candidates = sorted(all_candidates, key=lambda x: x[1])[:top_k]
            return TokenSearchResult(
                token_id=None,
                embedding=None,
                timesteps=evaluated,
                top_k_tokens=final_top_k_candidates,
                success=False,
            )

        distances_cpu = distances.detach().cpu().tolist()
        for idx, distance_val in enumerate(distances_cpu):
            token_id = int(batch_ids[idx].item())
            all_candidates.add((token_id, float(distance_val)))

        batch_min_distance, batch_min_idx = torch.min(distances, dim=0)
        batch_min_distance_val = float(batch_min_distance.item())
        batch_mean_distance_val = float(distances.mean().item())
        if batch_min_distance_val < best_distance:
            best_distance = batch_min_distance_val
            best_token_id = int(batch_ids[int(batch_min_idx)].item())

        evaluated += batch_ids.size(0)

        if verbose:
            token_str = tokenizer.decode([best_token_id]) if best_token_id is not None else ""
            print(
                '\r{}[{:5d}/{:5d}]: Best Distance: {:.2e} - Token: {:15s} - Time: {}'.format(
                    initial_desc,
                    evaluated,
                    num_tokens,
                    best_distance,
                    format_token(token_str, length=15),
                    format_time_minutes(time() - start_time),
                ),
                end=''
            )

        if best_distance < 1e-5:
            break

        if (start // batch_size + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if verbose:
        print()

    final_top_k_candidates = sorted(all_candidates, key=lambda x: x[1])[:top_k]
    if best_token_id is None:
        return TokenSearchResult(
            token_id=None,
            embedding=None,
            timesteps=evaluated,
            top_k_tokens=final_top_k_candidates,
            success=False,
        )

    return TokenSearchResult(
        token_id=best_token_id,
        embedding=embedding_matrix[best_token_id].clone().detach(),
        timesteps=evaluated,
        top_k_tokens=final_top_k_candidates,
        success=best_distance < 1e-5,
    )

def compute_grad_and_elim(
    input_embeddings: torch.Tensor,
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    layer_idx: int,
    h_target: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    # Move to device
    device = next(model.parameters()).device


    cont_embeddings = input_embeddings.to(device)
    disc_tokens     = input_ids.to(device)
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
    distance = torch.norm(h_last_disc - h_target).item()
    
    loss_cont.backward()
    return last_emb.grad.squeeze(0, 1), loss_disc, distance


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
    linear: bool = False,
    cmaes: bool = False,
    top_k: int = 10,
    batch_size: int = 256,
    shuffle_candidates: bool = True,
    cmaes_max_iterations: int = 1000,
    cmaes_stdev_init: float = 0.1,
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
        linear: Whether to use baseline linear search
        cmaes: Whether to use CMA-ES evolutionary search
        top_k: Number of top candidates to track
        batch_size: Candidate batch size for baseline/CMA-ES search
        shuffle_candidates: Whether to shuffle candidate order in baseline
        cmaes_max_iterations: Maximum iterations for CMA-ES
        cmaes_stdev_init: Initial standard deviation for CMA-ES
    
    Returns:
        TokenSearchResult containing the found token info and top-k candidates
    """
    if linear:
        return _find_token_linear(
            token_idx=token_idx,
            embedding_matrix=embedding_matrix,
            discovered_ids=discovered_ids,
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            h_target=h_target,
            verbose=verbose,
            top_k=top_k,
            batch_size=batch_size,
            shuffle_candidates=shuffle_candidates,
        )
    
    if cmaes:
        return _find_token_cmaes(
            token_idx=token_idx,
            embedding_matrix=embedding_matrix,
            discovered_embeddings=discovered_embeddings,
            discovered_ids=discovered_ids,
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            h_target=h_target,
            verbose=verbose,
            top_k=top_k,
            batch_size=batch_size,
            max_iterations=cmaes_max_iterations,
            stdev_init=cmaes_stdev_init,
        )

    # Otherwise use previous gradient based search.

    device = next(model.parameters()).device
    copy_embedding_matrix = embedding_matrix.clone().detach().requires_grad_(False)

    token_id = torch.randint(0, embedding_matrix.size(0), (1,)).item()
    
    embedding = copy_embedding_matrix[token_id].clone().requires_grad_(True)
    temp_embedding = copy_embedding_matrix[token_id].clone().detach()

    optimizer = torch.optim.SGD([embedding], lr=lr) if not linear else ExhaustiveOptimizer()
    if scheduler:
        scheduler_obj = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.99, threshold=lr / 100, patience=200
        )

    initial_desc = f'Token [{token_idx + 1:2d}/{h_target.size(0):2d}]'
    final_timestep = embedding_matrix.size(0)
    start_time = time()
    
    # Track all candidates: (token_id, distance)
    all_candidates: set = set()

    success = False

    for i in range(embedding_matrix.size(0)):
        input_embeddings = torch.stack(
            discovered_embeddings + [embedding]
        ).unsqueeze(0) 
        input_embeddings_disc = torch.tensor(
            discovered_ids + [token_id]
        ).unsqueeze(0) 

        grad_oracle = loss = torch.zeros_like(h_target[token_idx])
        distance = float("inf")

        if linear:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_embeddings_disc.to(device),
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states
            
            h_pred = hidden_states[layer_idx][0, -1, :].detach()
            loss = torch.nn.functional.mse_loss(h_pred, h_target[token_idx], reduction='sum')
            distance = float(torch.norm(h_pred - h_target[token_idx]).item())
        else:
            grad_oracle, loss, distance = compute_grad_and_elim(
                input_embeddings, input_embeddings_disc,
                model=model,
                layer_idx=layer_idx, 
                h_target=h_target[token_idx]
            )

        # Check for NaN
        if torch.isnan(loss) or (not linear and torch.isnan(grad_oracle).any()):
            # Select top-k tokens at failure point
            final_top_k_candidates = sorted(all_candidates, key=lambda x: x[1])[:top_k]
            return TokenSearchResult(
                token_id=None,
                embedding=None,
                timesteps=i,
                top_k_tokens=final_top_k_candidates,
                success=False
            )

        # Add all tokens as candidate
        all_candidates.add((token_id, float(distance)))

        loss_val = loss if isinstance(loss, float) else loss.item()

        grad_norm = grad_oracle.norm().item() if not linear else 0.0
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

        if loss_val < 1e-5 or (not linear and grad_norm < 1e-12):
            final_timestep = i + 1
            success = True
            break

        if not linear and grad_norm > 1:
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
        if not linear and (i + 1) % 50 == 0:
            embedding.data = temp_embedding.data

        if (i + 1) % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if verbose:
        print()

    correct_embedding = copy_embedding_matrix[token_id].clone().detach()
    del copy_embedding_matrix, embedding, temp_embedding

    # Select top-k tokens from all candidates by smallest distance
    final_top_k_candidates = sorted(all_candidates, key=lambda x: x[1])[:top_k]
    
    return TokenSearchResult(
        token_id=token_id,
        embedding=correct_embedding,
        timesteps=final_timestep,
        top_k_tokens=final_top_k_candidates,
        success=success
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
    linear: bool = False,
    cmaes: bool = False,
    special_start_tokens: Optional[List[int]] = None,
    continue_on_failure: bool = False,
    ground_truth_ids: Optional[List[int]] = None,
    top_k: int = 10,
    batch_size: int = 256,
    n_gpus: int = 1,
    shuffle_candidates: bool = True,
    cmaes_max_iterations: int = 1000,
    cmaes_stdev_init: float = 0.1,
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
        linear: Whether to use baseline linear search
        cmaes: Whether to use CMA-ES evolutionary search
        special_start_tokens: Special tokens to prepend (e.g., [128000])
        continue_on_failure: If True, continue with ground truth token when inversion fails
        ground_truth_ids: Ground truth token IDs (required if continue_on_failure=True)
        top_k: Number of top candidate tokens to track per position
        batch_size: Candidate batch size for baseline search
        n_gpus: Number of GPUs to use for batched evaluation
        shuffle_candidates: Whether to shuffle candidate order in baseline
    
    Returns:
        InversionResult containing discovered tokens and metadata
    """
    model = _maybe_wrap_data_parallel(model, n_gpus)
    embedding_matrix = _get_embedding_matrix(model)

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
            lr, scheduler, verbose, linear, cmaes,
            top_k=top_k,
            batch_size=batch_size,
            shuffle_candidates=shuffle_candidates,
            cmaes_max_iterations=cmaes_max_iterations,
            cmaes_stdev_init=cmaes_stdev_init,
        )

        token_end_time = time()
        
        # Store top-k for this position regardless of success
        top_k_per_position.append(result.top_k_tokens)

        if not result.success or result.embedding is None:
            failed_positions.append(i)
            
            if continue_on_failure:
                # Use the top closest token found for this position (if any), else fallback to failure
                if result.top_k_tokens and len(result.top_k_tokens) > 0:
                    top_token_id = result.top_k_tokens[0][0]
                else:
                    # Fallback: cannot find any candidate, return with partial results
                    return InversionResult(
                        success=False,
                        total_time=time() - start_time,
                        discovered_ids=discovered_ids,
                        timesteps=timesteps,
                        times=times,
                        top_k_per_position=top_k_per_position,
                        failed_positions=failed_positions,
                    )
                top_embedding = embedding_matrix[top_token_id].clone().detach().requires_grad_(False)
                
                discovered_embeddings.append(top_embedding)
                discovered_ids.append(top_token_id)
                timesteps.append(result.timesteps)
                times.append(token_end_time - token_start_time)

                gc.collect()
                torch.cuda.empty_cache()

                if verbose:
                    print(f"  [Using top closest token {top_token_id}: {tokenizer.decode([top_token_id])} for position {i}]")
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
        success=True,
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
        linear: Whether to use baseline linear search
        cmaes: Whether to use CMA-ES evolutionary search
        special_start_tokens: Special tokens to prepend
        continue_on_failure: If True, continue with ground truth when inversion fails
        top_k: Number of top candidate tokens to track
        batch_size: Candidate batch size for baseline search
        n_gpus: Number of GPUs to use for batched evaluation
        shuffle_candidates: Whether to shuffle candidate order in baseline
    
    Returns:
        Tuple of (match, time, discovered_ids, times, full_result)
    """
    set_seed(seed)

    h_target = extract_hidden_states(input_ids, model, layer_idx)

    # Extract ground truth IDs for continue_on_failure mode
    ground_truth_ids = input_ids[0].tolist() if continue_on_failure else None

    result = find_prompt(
        model, tokenizer, layer_idx, h_target, 
        lr, scheduler, verbose, linear, cmaes, special_start_tokens,
        continue_on_failure=continue_on_failure,
        ground_truth_ids=ground_truth_ids,
        top_k=top_k,
        batch_size=batch_size,
        n_gpus=n_gpus,
        shuffle_candidates=shuffle_candidates,
        cmaes_max_iterations=cmaes_max_iterations,
        cmaes_stdev_init=cmaes_stdev_init,
    )
    
    if not result.success or result.discovered_ids is None:
        if verbose:
            print('Inversion failed or diverged with the given parameters.')
        return False, None, None, None, result

    match = all([x == y for x, y in zip(input_ids[0].tolist(), result.discovered_ids)])
    
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
    linear: bool = False,
    cmaes: bool = False,
    verbose: bool = True,
    original_ids: Optional[Tensor] = None,
    special_start_tokens: Optional[List[int]] = None,
    continue_on_failure: bool = False,
    top_k: int = 10,
    batch_size: int = 256,
    n_gpus: int = 1,
    shuffle_candidates: bool = True,
    cmaes_max_iterations: int = 1000,
    cmaes_stdev_init: float = 0.1,
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
        linear: Whether to use baseline linear search
        cmaes: Whether to use CMA-ES evolutionary search
        verbose: Whether to print progress
        original_ids: Optional original token IDs for comparison
        special_start_tokens: Special tokens to prepend
        continue_on_failure: If True, continue with ground truth when inversion fails
        top_k: Number of top candidate tokens to track
        batch_size: Candidate batch size for baseline search
        n_gpus: Number of GPUs to use for batched evaluation
        shuffle_candidates: Whether to shuffle candidate order in baseline
    
    Returns:
        Tuple of (match_original, time, discovered_ids, times, full_result)
    """
    set_seed(seed)
    
    # Extract ground truth IDs
    ground_truth_ids = original_ids[0].tolist() if continue_on_failure and original_ids is not None else None
    
    # Find prompt
    result = find_prompt(
        model, tokenizer, layer_idx, h_target, lr,
        linear=linear,
        cmaes=cmaes,
        verbose=verbose,
        special_start_tokens=special_start_tokens,
        continue_on_failure=continue_on_failure,
        ground_truth_ids=ground_truth_ids,
        top_k=top_k,
        batch_size=batch_size,
        n_gpus=n_gpus,
        shuffle_candidates=shuffle_candidates,
        cmaes_max_iterations=cmaes_max_iterations,
        cmaes_stdev_init=cmaes_stdev_init,
    )
    
    if not result.success or result.discovered_ids is None:
        if verbose:
            print("Inversion failed or diverged.")
        return False, None, None, None, result
    
    # Check if we match original (if provided)
    match = False
    if original_ids is not None:
        match = all([x == y for x, y in zip(original_ids[0].tolist(), result.discovered_ids)])
    
    if verbose:
        reconstructed = tokenizer.decode(result.discovered_ids)
        print(f"Reconstructed: {reconstructed[:100]}...")
        if original_ids is not None:
            original = tokenizer.decode(original_ids[0].tolist())
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

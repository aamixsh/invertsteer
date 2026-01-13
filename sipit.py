#!/usr/bin/env python3
"""
Standalone script for running prompt inversion attacks.

This script provides a simple interface to run the SIP-It inversion algorithm
on prompts. It can be run directly without the full experiment pipeline.

Usage:
    python sipit.py --prompt "Your prompt here"
    python sipit.py --prompt "Your prompt" --no-chat-template
    python sipit.py --prompt "Your prompt" --continue-on-failure
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import torch
from typing import Any, Optional, List

import transformers
transformers.logging.set_verbosity_error()

from transformers import AutoModelForCausalLM, AutoTokenizer

from inversion import (
    set_seed,
    inversion_attack,
    print_top_k_tokens,
    InversionResult,
)


# Model layer counts for supported models
MODEL_LAYERS = {
    'roneneldan/TinyStories-1M': 8,
    'roneneldan/TinyStories-8M': 8,
    'roneneldan/TinyStories-33M': 4,
    
    'openai-community/gpt2': 12,
    'openai-community/gpt2-medium': 24,
    'openai-community/gpt2-large': 36,

    'google/gemma-3-1b-pt': 26,
    'google/gemma-3-4b-pt': 34,
    'google/gemma-3-12b-pt': 48,
    
    'microsoft/Phi-4-mini-instruct': 32,
    'mistralai/Mistral-7B-v0.1': 32,
    'meta-llama/Llama-3.1-8B': 32,
    'meta-llama/Llama-3.1-8B-Instruct': 32,

    'Qwen/Qwen2.5-0.5B': 24,
    'meta-llama/Llama-3.2-1B-Instruct': 16,
}


# Llama 3 special tokens
LLAMA3_BOS_TOKEN = 128000  # <|begin_of_text|>
LLAMA3_CHAT_TOKENS = [128006, 882, 128007, 271]  # <|start_header_id|>user<|end_header_id|>\n\n


def get_num_layers(model_id: str) -> int:
    """Get the number of layers for a model."""
    if model_id not in MODEL_LAYERS:
        # Try to get from model config
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id)
            return config.num_hidden_layers
        except Exception:
            raise NotImplementedError(f'Model ID `{model_id}` is not supported!')
    return MODEL_LAYERS[model_id]


def load_model(
    model_id: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer."""
    print(f"Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    model.to(device)
    
    # Print model size
    size_in_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_in_gb = size_in_bytes / ((2 ** 10) ** 3)
    print(f'Model memory: {size_in_gb:.2f} GB')
    
    # Setup for inference
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.set_grad_enabled(True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def get_special_start_tokens(
    model_id: str,
    use_chat_template: bool = True,
) -> Optional[List[int]]:
    """
    Get special start tokens for a model.
    
    Args:
        model_id: The model ID
        use_chat_template: If True, include full chat template tokens.
                          If False, only include BOS token.
    
    Returns:
        List of special token IDs, or None for models without special tokens
    """
    # Llama 3 models
    if 'llama' in model_id.lower() or 'Llama' in model_id:
        if use_chat_template:
            return [LLAMA3_BOS_TOKEN] + LLAMA3_CHAT_TOKENS
        else:
            return [LLAMA3_BOS_TOKEN]
    
    # GPT-2 and similar models don't need special start tokens
    if 'gpt2' in model_id.lower():
        return None
    
    # Default: no special tokens
    return None


def run_inversion(
    prompt: str,
    model_id: str = 'meta-llama/Llama-3.1-8B-Instruct',
    device: str = 'cuda',
    layer: int = -1,
    learning_rate: float = 1.0,
    seed: int = 8,
    scheduler: bool = False,
    baseline: bool = False,
    use_chat_template: bool = True,
    continue_on_failure: bool = False,
    top_k: int = 10,
    verbose: bool = True,
) -> tuple[bool, Optional[float], Optional[List[int]], InversionResult]:
    """
    Run prompt inversion attack.
    
    Args:
        prompt: The prompt to invert
        model_id: HuggingFace model ID
        device: Device to use
        layer: Layer to target (-1 for last layer)
        learning_rate: Learning rate for optimization
        seed: Random seed
        scheduler: Whether to use LR scheduler
        baseline: Whether to use exhaustive baseline
        use_chat_template: If True, use full chat template tokens.
                          If False, only use BOS token.
        continue_on_failure: If True, continue with ground truth when inversion fails
        top_k: Number of top candidate tokens to track
        verbose: Whether to print progress
    
    Returns:
        Tuple of (match, time, discovered_ids, full_result)
    """
    set_seed(seed)
    
    # Load model
    dtype = torch.float32
    model, tokenizer = load_model(model_id, device, dtype)
    
    # Get layer count and adjust layer index
    total_layers = get_num_layers(model_id)
    if layer < 0:
        layer = total_layers + layer + 1
    
    # Get special start tokens
    special_start_tokens = get_special_start_tokens(model_id, use_chat_template)
    
    if verbose:
        print(f'Prompt: {prompt}')
        print(f'Layer: {layer}/{total_layers}')
        print(f'Use chat template: {use_chat_template}')
        if special_start_tokens:
            print(f'Special start tokens: {special_start_tokens}')
    
    # Tokenize prompt (without special tokens, we add them separately)
    enc = tokenizer(
        prompt,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    input_ids_list: List[int] = enc['input_ids']
    
    if verbose:
        print(f'Input IDs ({len(input_ids_list)} tokens): {input_ids_list}')
        for i, token_id in enumerate(input_ids_list):
            print(f'  {i}: [{token_id:6d}] "{tokenizer.decode([token_id])}"')
    
    # Run inversion
    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device).unsqueeze(0)
    
    match, time_taken, discovered_ids, times, result = inversion_attack(
        input_ids, 
        model, 
        tokenizer, 
        layer,
        learning_rate, 
        seed, 
        scheduler, 
        verbose,
        baseline,
        special_start_tokens,
        continue_on_failure=continue_on_failure,
        top_k=top_k,
    )
    
    # Print top-k tokens if requested
    if verbose and result is not None:
        print_top_k_tokens(result, tokenizer, input_ids_list)
    
    return match, time_taken, discovered_ids, result


def main():
    parser = argparse.ArgumentParser(
        description='Run SIP-It prompt inversion attack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inversion
  python sipit.py --prompt "Hello, how are you?"
  
  # Without chat template (only BOS token for Llama)
  python sipit.py --prompt "Hello" --no-chat-template
  
  # Continue even if some tokens fail
  python sipit.py --prompt "Test prompt" --continue-on-failure
  
  # Use different model
  python sipit.py --prompt "Hello" --model openai-community/gpt2
"""
    )
    
    parser.add_argument('--prompt', type=str, default="Hey Aayush, I am Giorgos and I am sending you the hidden states of this super secret message: `jw!@@L901~~!==`! It should be encrypted enough :-)", 
                        help='Prompt to invert')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Model ID')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device (cuda, cuda:0, cpu)')
    parser.add_argument('--layer', type=int, default=-1,
                        help='Layer to target (-1 for last)')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=8,
                        help='Random seed')
    parser.add_argument('--scheduler', action='store_true',
                        help='Use LR scheduler')
    parser.add_argument('--baseline', action='store_true',
                        help='Use exhaustive baseline')
    parser.add_argument('--no-chat-template', action='store_true',
                        help='Do not use chat template (only BOS token for Llama)')
    parser.add_argument('--continue-on-failure', action='store_true',
                        help='Continue with ground truth when inversion fails')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of top candidate tokens to track')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    match, time_taken, discovered_ids, result = run_inversion(
        prompt=args.prompt,
        model_id=args.model,
        device=args.device,
        layer=args.layer,
        learning_rate=args.lr,
        seed=args.seed,
        scheduler=args.scheduler,
        baseline=args.baseline,
        use_chat_template=not args.no_chat_template,
        continue_on_failure=args.continue_on_failure,
        top_k=args.top_k,
        verbose=not args.quiet,
    )
    
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(f"Match: {match}")
    if time_taken is not None:
        print(f"Time: {time_taken:.2f}s")
    if result is not None and result.failed_positions:
        print(f"Failed positions: {result.failed_positions}")


if __name__ == '__main__':
    main()

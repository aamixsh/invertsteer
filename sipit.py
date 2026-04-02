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
import torch.nn as nn

import transformers
transformers.logging.set_verbosity_error()

from transformers import AutoModelForCausalLM, AutoTokenizer

from inversion import (
    set_seed,
    inversion_attack,
    print_top_k_tokens,
    InversionResult,
)

from model_utils import get_tokenize_fn, load_model as load_hf_causal_lm, get_input_device

def replace_last_norm(model_id: str, model: AutoModelForCausalLM):
    attr_name = {
        'roneneldan/TinyStories-1M': 'transformer.ln_f',
        'roneneldan/TinyStories-8M': 'transformer.ln_f',
        'roneneldan/TinyStories-33M': 'transformer.ln_f',
        
        'openai-community/gpt2': 'transformer.ln_f',
        'openai-community/gpt2-medium': 'transformer.ln_f',
        'openai-community/gpt2-large': 'transformer.ln_f',

        'google/gemma-3-1b-pt': 'model.norm',
        'google/gemma-3-4b-pt': 'model.norm',
        'google/gemma-3-12b-pt': 'model.norm',
        
        'microsoft/Phi-4-mini-instruct': 'model.norm',
        'mistralai/Mistral-7B-v0.1': 'model.norm',

        'Qwen/Qwen2.5-0.5B': 'model.norm',
        'meta-llama/Llama-3.2-1B': 'model.norm',
        'meta-llama/Llama-3.2-1B-Instruct': 'model.norm',
        'meta-llama/Llama-3.2-3B': 'model.norm',
        'meta-llama/Llama-3.2-3B-Instruct': 'model.norm',
        'meta-llama/Llama-3.1-8B': 'model.norm',
        'meta-llama/Llama-3.1-8B-Instruct': 'model.norm',
    }

    if model_id not in attr_name:
        raise NotImplementedError(f'Model ID `{model_id}` is not supported!')
    
    parts = attr_name[model_id].split('.')
    parent_path, leaf = '.'.join(parts[:-1]), parts[-1]

    parent = _get_attr_by_path(model, parent_path) if parent_path else model
    model.norm_bk = getattr(parent, leaf)
    setattr(parent, leaf, nn.Identity())


def _get_attr_by_path(root, path: str):
    """Get attribute by a dotted path, or raise AttributeError."""
    obj = root
    for part in path.split('.'):
        obj = getattr(obj, part)
    return obj

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
    *,
    load_in_4bit: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer (shared path with invertsteer ``model_utils.load_model``)."""
    dtype_str = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }.get(dtype, "float32")
    print(f"Loading model: {model_id}")
    model, tokenizer = load_hf_causal_lm(
        model_id, device, dtype_str, load_in_4bit=load_in_4bit
    )
    size_in_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_in_gb = size_in_bytes / ((2**10) ** 3)
    print(f"Model memory (parameter storage): {size_in_gb:.2f} GB")
    torch.set_grad_enabled(True)
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
    add_special_tokens: bool = False,
    continue_on_failure: bool = False,
    top_k: int = 10,
    verbose: bool = True,
    load_in_4bit: bool = False,
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
        add_special_tokens: If True, add special tokens (like BOS token for Llama)
        continue_on_failure: If True, continue with ground truth when inversion fails
        top_k: Number of top candidate tokens to track
        verbose: Whether to print progress
    
    Returns:
        Tuple of (match, time, discovered_ids, full_result)
    """
    set_seed(seed)
    
    # Load model
    dtype = torch.float32
    model, tokenizer = load_model(model_id, device, dtype, load_in_4bit=load_in_4bit)

    replace_last_norm(model_id, model)
    
    # Get layer count and adjust layer index
    total_layers = get_num_layers(model_id)
    if layer < 0:
        layer = total_layers + layer + 1
    
    # # Get special start tokens
    # special_start_tokens = get_special_start_tokens(model_id, use_chat_template)
    special_start_tokens = None
    
    if verbose:
        print(f'Prompt: {prompt}')
        print(f'Layer: {layer}/{total_layers}')
        print(f'Use chat template: {use_chat_template}')
        if special_start_tokens:
            print(f'Special start tokens: {special_start_tokens}')
    
    # Tokenize prompt (without special tokens, we add them separately)
    tokenize_fn = get_tokenize_fn(tokenizer, use_chat_template=use_chat_template, add_special_tokens=add_special_tokens)
    inputs = tokenize_fn(instructions=[prompt])
    input_ids = inputs.input_ids.to(get_input_device(model))
    
    if verbose:
        print(f'Input IDs ({len(input_ids[0])} tokens): {input_ids}')
        for i, token_id in enumerate(input_ids[0]):
            print(f'  {i}: [{token_id:6d}] "{tokenizer.decode([token_id])}"')
    
    # Run inversion
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
        print_top_k_tokens(result, tokenizer, input_ids[0].tolist())
    
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
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B-Instruct',
                        help='Model ID')
    parser.add_argument('--device', type=str, default='cuda:0',
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
    parser.add_argument('--add-special-tokens', action='store_true',
                        help='Add special tokens (like BOS token for Llama)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of top candidate tokens to track')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument(
        '--load-in-4bit',
        action='store_true',
        help='NF4 BitsAndBytes load (GPU + bitsandbytes).',
    )

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
        add_special_tokens=args.add_special_tokens,
        continue_on_failure=args.continue_on_failure,
        top_k=args.top_k,
        verbose=not args.quiet,
        load_in_4bit=args.load_in_4bit,
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

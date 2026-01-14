"""
Model loading and utility functions.
"""

import torch
import functools
import random
import numpy as np
from typing import List, Optional
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


# Llama 3 special tokens
LLAMA3_CHAT_TOKENS = [128000, 128006, 882, 128007, 271]  # <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n

# Llama 3 chat template
LLAMA3_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def load_model(model_id: str, device: str = "cuda", dtype: str = "float32"):
    """Load the model and tokenizer."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype, torch.float32)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()
    model.requires_grad_(False)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def format_instruction(
    instruction: str,
    output: Optional[str] = None,
    system: Optional[str] = None,
    include_trailing_whitespace: bool = True
) -> str:
    """Format an instruction using the Llama 3 chat template."""
    if system is not None:
        formatted = LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM.format(
            instruction=instruction, 
            system_prompt=system
        )
    else:
        formatted = LLAMA3_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted = formatted.rstrip()

    if output is not None:
        formatted += output

    return formatted


def tokenize_instructions(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: Optional[List[str]] = None,
    system: Optional[str] = None,
    include_trailing_whitespace: bool = True,
    use_chat_template: bool = True,
):
    """
    Tokenize a list of instructions.
    
    Args:
        tokenizer: The tokenizer
        instructions: List of instruction strings
        outputs: Optional list of output strings
        system: Optional system prompt
        include_trailing_whitespace: Whether to include trailing whitespace
        use_chat_template: If True, format with chat template.
                          If False, just add BOS token and raw text.
    """
    if use_chat_template:
        if outputs is not None:
            prompts = [
                format_instruction(
                    instruction=instruction, 
                    output=output, 
                    system=system, 
                    include_trailing_whitespace=include_trailing_whitespace
                )
                for instruction, output in zip(instructions, outputs)
            ]
        else:
            prompts = [
                format_instruction(
                    instruction=instruction, 
                    system=system, 
                    include_trailing_whitespace=include_trailing_whitespace
                )
                for instruction in instructions
            ]
    else:
        # No chat template - just use raw text
        prompts = instructions

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    )

    return result


def get_tokenize_fn(
    tokenizer: AutoTokenizer, 
    system: Optional[str] = None,
    use_chat_template: bool = True,
):
    """Get a partial function for tokenizing instructions."""
    return functools.partial(
        tokenize_instructions, 
        tokenizer=tokenizer, 
        system=system, 
        include_trailing_whitespace=True,
        use_chat_template=use_chat_template,
    )


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
    # # Llama 3 models
    # if 'llama' in model_id.lower():
    #     if use_chat_template:
    #         return LLAMA3_CHAT_TOKENS
    #     else:
    #         return None
    
    # # GPT-2 and similar models don't need special start tokens
    # if 'gpt2' in model_id.lower():
    #     return None
    
    # Default: no special tokens for now.
    return None


def get_model_layers(model: AutoModelForCausalLM) -> torch.nn.ModuleList:
    """Get the transformer layers from the model."""
    return model.model.layers


def get_num_layers(model: AutoModelForCausalLM) -> int:
    """Get the number of layers in the model."""
    return model.config.num_hidden_layers


def get_hidden_size(model: AutoModelForCausalLM) -> int:
    """Get the hidden size of the model."""
    return model.config.hidden_size


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

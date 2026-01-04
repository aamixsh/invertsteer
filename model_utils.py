"""
Model loading and utility functions for Llama-3.2-1B-Instruct.
"""

import torch
import functools
from typing import List, Optional
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import LLAMA32_CHAT_TEMPLATE, LLAMA32_CHAT_TEMPLATE_WITH_SYSTEM


def load_model(model_id: str, device: str = "cuda", dtype: str = "float32"):
    """Load the model and tokenizer."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
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
    """Format an instruction using the Llama 3.2 chat template."""
    if system is not None:
        formatted = LLAMA32_CHAT_TEMPLATE_WITH_SYSTEM.format(
            instruction=instruction, 
            system_prompt=system
        )
    else:
        formatted = LLAMA32_CHAT_TEMPLATE.format(instruction=instruction)

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
):
    """Tokenize a list of instructions using the chat template."""
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

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result


def get_tokenize_fn(tokenizer: AutoTokenizer, system: Optional[str] = None):
    """Get a partial function for tokenizing instructions."""
    return functools.partial(
        tokenize_instructions, 
        tokenizer=tokenizer, 
        system=system, 
        include_trailing_whitespace=True
    )


def get_model_layers(model: AutoModelForCausalLM) -> torch.nn.ModuleList:
    """Get the transformer layers from the model."""
    return model.model.layers


def get_num_layers(model: AutoModelForCausalLM) -> int:
    """Get the number of layers in the model."""
    return model.config.num_hidden_layers


def get_hidden_size(model: AutoModelForCausalLM) -> int:
    """Get the hidden size of the model."""
    return model.config.hidden_size


def replace_final_norm_with_identity(model: AutoModelForCausalLM):
    """
    Replace the final layer norm with identity for SIP-It.
    This is needed because SIP-It targets hidden states before the final norm.
    Stores the original norm for restoration.
    """
    model.norm_backup = model.model.norm
    model.model.norm = torch.nn.Identity()
    return model


def restore_final_norm(model: AutoModelForCausalLM):
    """Restore the original final layer norm."""
    if hasattr(model, 'norm_backup'):
        model.model.norm = model.norm_backup
        del model.norm_backup
    return model


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


"""
Model loading and utility functions.
"""

import torch
import functools
from typing import List, Optional
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


# Llama 3 chat template
LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM = """<|start_header_id|>system<|end_header_id|>

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
    torch_dtype = dtype_map.get(dtype, torch.float32)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
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

"""
Model loading and utility functions.
"""

import torch
import functools
import random
import numpy as np
import os
from typing import List, Optional
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaTokenizerFast, GemmaTokenizer, GemmaTokenizerFast, Qwen2Tokenizer, Qwen2TokenizerFast, PreTrainedTokenizerFast


def ensure_module_set_submodule_polyfill() -> None:
    """Transformers bitsandbytes uses ``nn.Module.set_submodule`` (PyTorch 2.1+)."""
    import torch.nn as nn

    if hasattr(nn.Module, "set_submodule"):
        return

    def set_submodule(self, target: str, module: nn.Module) -> None:
        if not target:
            raise ValueError("Cannot set the root module")
        parent_path, _, child_name = target.rpartition(".")
        parent = self.get_submodule(parent_path) if parent_path else self
        setattr(parent, child_name, module)

    nn.Module.set_submodule = set_submodule  # type: ignore[method-assign]


def bitsandbytes_4bit_config(compute_dtype: torch.dtype):
    import bitsandbytes  # noqa: F401
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


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

GEMMA_CHAT_TEMPLATE = """<bos><start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""


def load_model(
    model_id: str,
    device: str = "cuda",
    dtype: str = "float32",
    *,
    load_in_4bit: bool = False,
):
    """Load the model and tokenizer.

    When ``load_in_4bit`` is True, loads NF4-quantized weights (BitsAndBytes) with
    ``device_map`` placement; ``dtype`` is used as ``bnb_4bit_compute_dtype``.
    Requires GPU, ``bitsandbytes``, and ``accelerate``.
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    load_kwargs: dict = dict(trust_remote_code=True)
    if load_in_4bit:
        ensure_module_set_submodule_polyfill()
        load_kwargs["quantization_config"] = bitsandbytes_4bit_config(torch_dtype)
        load_kwargs["low_cpu_mem_usage"] = True
    else:
        load_kwargs["torch_dtype"] = torch_dtype

    # Helps reduce CUDA memory fragmentation during large model loads.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # Multi-GPU sharding (for large models like Qwen3.5-35B-A3B):
    # Set INVERTSTEER_N_GPUS=2 (or higher). We use accelerate device_map="auto"
    # with per-GPU max_memory. GPU keys must be integers: 0, 1, ...
    n_gpus_env = os.getenv("INVERTSTEER_N_GPUS", "1")
    try:
        n_gpus = max(1, int(n_gpus_env))
    except ValueError:
        n_gpus = 1

    print(f"device: {device}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    use_device_map = torch.cuda.is_available() and (
        device == "auto"
        or (device == "cuda" and n_gpus > 1)
        or load_in_4bit
    )
    if use_device_map:
        n_use = min(n_gpus, torch.cuda.device_count())
        if n_use > 1:
            # Leave more headroom than we think we need; HF loading may do transient
            # allocations during materialization/conversion.
            reserve_bytes = 6 * 1024**3
            max_memory = {}
            for i in range(n_use):
                total = torch.cuda.get_device_properties(i).total_memory
                budget = max(1, total - reserve_bytes)
                max_memory[i] = f"{budget // (1024**2)}MiB"
            # Allow a small CPU spillover to prevent hard OOM during load.
            max_memory["cpu"] = os.getenv("INVERTSTEER_CPU_MAX_MEMORY", "32GiB")
            load_kwargs.update(
                dict(
                    device_map="auto",
                    max_memory=max_memory,
                    low_cpu_mem_usage=True,
                )
            )
        elif device == "auto" or load_in_4bit:
            dm: object = "auto"
            if load_in_4bit and isinstance(device, str):
                if device.startswith("cuda:") and device != "cuda":
                    try:
                        di = int(device.split(":", 1)[1])
                        if 0 <= di < torch.cuda.device_count():
                            dm = {"": di}
                    except (ValueError, IndexError):
                        pass
                elif device == "cuda":
                    dm = {"": 0}
            load_kwargs.update(dict(device_map=dm, low_cpu_mem_usage=True))

    def _disable_transformers_allocator_warmup_if_needed(kwargs) -> None:
        """Avoid huge transient allocations during HF load.

        Transformers 5.x may call `caching_allocator_warmup()` during sharded loads,
        which can attempt a large allocation on GPU0 and OOM even when the final
        sharded model would fit. We monkeypatch it to a no-op for the duration of
        this load when using device_map/max_memory.
        """
        if not (isinstance(kwargs, dict) and ("device_map" in kwargs or "max_memory" in kwargs)):
            return
        try:
            import transformers.modeling_utils as modeling_utils  # type: ignore
            if hasattr(modeling_utils, "caching_allocator_warmup"):
                modeling_utils.caching_allocator_warmup = lambda *args, **kw: None  # type: ignore
        except Exception:
            pass

    def _load_with(kwargs):
        _disable_transformers_allocator_warmup_if_needed(kwargs)
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    model = _load_with(load_kwargs)

    # Only do an explicit .to(...) when not sharded via device_map.
    if device != "auto" and not hasattr(model, "hf_device_map"):
        target_device = device
        if isinstance(target_device, str) and target_device.startswith("cuda"):
            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                target_device = "cpu"
            elif target_device == "cuda":
                target_device = "cuda:0"
            elif target_device.startswith("cuda:"):
                try:
                    idx = int(target_device.split("cuda:")[1])
                except Exception:
                    idx = 0
                if idx < 0 or idx >= torch.cuda.device_count():
                    target_device = "cuda:0"
        try:
            model.to(target_device)
        except torch.OutOfMemoryError:
            # For very large models (e.g. Qwen3.5-35B-A3B), a single GPU may not fit.
            # Fall back to multi-GPU sharding if possible.
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                try:
                    del model
                except Exception:
                    pass
                torch.cuda.empty_cache()

                n_gpus_env2 = os.getenv("INVERTSTEER_N_GPUS", str(torch.cuda.device_count()))
                try:
                    n_gpus2 = max(1, int(n_gpus_env2))
                except ValueError:
                    n_gpus2 = torch.cuda.device_count()
                n_use2 = min(n_gpus2, torch.cuda.device_count())

                reserve_bytes2 = 2 * 1024**3
                max_memory2 = {}
                for i in range(n_use2):
                    total = torch.cuda.get_device_properties(i).total_memory
                    budget = max(1, total - (6 * 1024**3))
                    max_memory2[i] = f"{budget // (1024**2)}MiB"
                max_memory2["cpu"] = os.getenv("INVERTSTEER_CPU_MAX_MEMORY", "32GiB")

                shard_kwargs = dict(load_kwargs)
                shard_kwargs.update(
                    dict(
                        device_map="auto",
                        max_memory=max_memory2,
                        low_cpu_mem_usage=True,
                    )
                )
                model = _load_with(shard_kwargs)
            else:
                raise
    model.eval()
    model.requires_grad_(False)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"model.device: {model.device}")
    
    return model, tokenizer


def get_input_device(model: AutoModelForCausalLM) -> torch.device:
    """Return the device that input_ids / attention_mask should be placed on.

    For sharded models (device_map='auto'), there is no single model.device; the
    safest choice is the input-embedding device.
    """
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    return next(model.parameters()).device


def format_instruction(
    tokenizer: AutoTokenizer,
    instruction: str,
    output: Optional[str] = None,
    system: Optional[str] = None,
    include_trailing_whitespace: bool = True
) -> str:
    """Format an instruction using a model-appropriate chat template.

    For Qwen3.5, we enforce instruct/non-thinking mode by closing the template's
    trailing <think> tag (the HF template currently emits '<think>\\n' even when
    enable_thinking=False).
    """

    name = getattr(tokenizer, "name_or_path", "") or ""
    name_l = name.lower()
    if hasattr(tokenizer, "apply_chat_template") and ("qwen3.5" in name_l or "qwen3_5" in name_l):
        messages = [{"role": "user", "content": instruction}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        if formatted.endswith("<think>\n") and "</think>" not in formatted:
            formatted += "</think>\n\n"
        if output is not None:
            formatted += output
        if not include_trailing_whitespace:
            formatted = formatted.rstrip()
        return formatted

    if system is not None:
        if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
            formatted = LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM.format(
            instruction=instruction, 
            system_prompt=system
        )
        elif isinstance(tokenizer, GemmaTokenizer) or isinstance(tokenizer, GemmaTokenizerFast):
            formatted = GEMMA_CHAT_TEMPLATE.format(instruction=instruction)
        elif isinstance(tokenizer, Qwen2Tokenizer) or isinstance(tokenizer, Qwen2TokenizerFast):
            formatted = QWEN_CHAT_TEMPLATE.format(instruction=instruction)
        else:
            # Fall back to tokenizer chat template if available (best-effort).
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": instruction}]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                raise ValueError(f"Unsupported tokenizer: {type(tokenizer)}")
    else:
        if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
            formatted = LLAMA3_CHAT_TEMPLATE.format(instruction=instruction)
        elif isinstance(tokenizer, GemmaTokenizer) or isinstance(tokenizer, GemmaTokenizerFast):
            formatted = GEMMA_CHAT_TEMPLATE.format(instruction=instruction)
        elif isinstance(tokenizer, Qwen2Tokenizer) or isinstance(tokenizer, Qwen2TokenizerFast):
            formatted = QWEN_CHAT_TEMPLATE.format(instruction=instruction)
        else:
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": instruction}]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                raise ValueError(f"Unsupported tokenizer: {type(tokenizer)}")

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
    add_special_tokens: bool = False,
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
                    tokenizer=tokenizer,
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
                    tokenizer=tokenizer,
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
        add_special_tokens=add_special_tokens,
    )

    return result


def get_tokenize_fn(
    tokenizer: AutoTokenizer, 
    system: Optional[str] = None,
    use_chat_template: bool = True,
    add_special_tokens: bool = False,
):
    """Get a partial function for tokenizing instructions."""
    return functools.partial(
        tokenize_instructions, 
        tokenizer=tokenizer, 
        system=system, 
        include_trailing_whitespace=True,
        use_chat_template=use_chat_template,
        add_special_tokens=add_special_tokens,
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

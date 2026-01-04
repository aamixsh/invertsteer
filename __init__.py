"""
InvertSteer: Inverting Intervened Activations

This package combines refusal direction ablation with prompt inversion
to explore whether natural prompts can produce the same activations as
intervened (refusal-ablated) activations.
"""

from .config import Config
from .model_utils import load_model, get_tokenize_fn, set_seed
from .refusal_direction import compute_refusal_direction, extract_and_save_refusal_direction
from .intervention import ablation_context, get_hidden_states_with_ablation
from .inversion import inversion_attack, inversion_attack_with_target

__all__ = [
    "Config",
    "load_model",
    "get_tokenize_fn", 
    "set_seed",
    "compute_refusal_direction",
    "extract_and_save_refusal_direction",
    "ablation_context",
    "get_hidden_states_with_ablation",
    "inversion_attack",
    "inversion_attack_with_target",
]


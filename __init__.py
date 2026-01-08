"""
InvertSteer: Inverting Steered Activations

This package inverts steered activations to find prompts that naturally
produce the same effect as activation-space interventions.

The core idea:
1. Apply steering (e.g., refusal direction) to model activations
2. Use SIP-It inversion to find a prompt that produces those steered activations
3. Evaluate if the inverted prompt has the same behavioral effect

Usage:
    # Run refusal_direction pipeline first to extract direction
    cd ../refusal_direction
    python -m pipeline.run_pipeline --model_path meta-llama/Llama-3.2-1B-Instruct
    
    # Then run inversion experiment
    cd ../invertsteer
    python experiment.py --demo
"""

from .config import Config
from .model_utils import load_model, get_tokenize_fn, set_seed
from .steering import (
    SteeringConfig,
    load_steering_direction,
    steering_context,
    compare_generations,
    get_hidden_states_with_steering,
    get_hidden_states_iterative_with_steering,
)
from .inversion import (
    inversion_attack,
    inversion_attack_with_target,
    extract_hidden_states_iterative,
    compute_activation_mse,
)

__all__ = [
    "Config",
    "load_model",
    "get_tokenize_fn",
    "set_seed",
    "SteeringConfig",
    "load_steering_direction",
    "steering_context",
    "compare_generations",
    "get_hidden_states_with_steering",
    "get_hidden_states_iterative_with_steering",
    "inversion_attack",
    "inversion_attack_with_target",
    "extract_hidden_states_iterative",
    "compute_activation_mse",
]

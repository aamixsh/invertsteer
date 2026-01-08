# Inverting Steered Activations: Technical Plan

## Goal

Investigate whether natural prompts exist that produce the same activations as steered (intervened) activations. Specifically: can we invert an activation that has been modified by steering (e.g., refusal direction) back into the prompt space?

## Background

### SIP-It (Soft Inverse Prompt Inversion)
SIP-It reconstructs prompts from hidden state activations by:
1. Computing target hidden states at a specific layer for the original input
2. For each token position iteratively:
   - Start with a random token embedding
   - Compute gradient of MSE loss between current hidden state and target
   - Use gradient direction to find the nearest token in vocabulary
   - Repeat until convergence or finding an exact match
3. Chain token finding across all positions for full prompt reconstruction

### Refusal Direction
The refusal direction method identifies a 1-dimensional subspace in activation space that mediates refusal behavior:
1. Collect activations from harmful and harmless prompts
2. Compute refusal direction = mean(harmful_activations) - mean(harmless_activations)
3. Apply via activation addition: `activation' = activation + coeff * direction`
4. This can bypass refusal (coeff < 0) or induce refusal (coeff > 0)

## Architecture

### Modular Steering Design

The project is designed to be **steering-agnostic**. The `steering.py` module provides a unified interface for any steering method:

```python
@dataclass
class SteeringConfig:
    direction: Tensor  # The steering direction vector [d_model]
    layer: int  # Layer to apply the steering
    method: str = "actadd"  # "actadd" or "ablation"
    coeff: float = 1.0  # Coefficient for actadd
```

This allows testing different steering approaches without changing the inversion code.

### Key Design Decisions

1. **Direction Extraction**: Handled by the `refusal_direction` repo, not reimplemented here. This ensures we use the well-tested, validated approach.

2. **Layer Norm**: Original model kept intact (no layer norm replacement). We invert at the steering layer, not the last layer.

3. **Inversion Layer**: Set equal to the steering layer. When steering is applied at layer L, we extract hidden_states[L+1] (the output after that layer).

4. **Hook Reuse**: Import and use hooks from `refusal_direction/pipeline/utils/hook_utils.py` directly.

## Technical Approach

### Phase 1: Extract Steering Direction (External)

Run the refusal_direction pipeline:
```bash
cd refusal_direction
python -m pipeline.run_pipeline --model_path meta-llama/Llama-3.2-1B-Instruct
```

This produces:
- `pipeline/runs/Llama-3.2-1B-Instruct/direction.pt`: The direction tensor
- `pipeline/runs/Llama-3.2-1B-Instruct/direction_metadata.json`: Layer info

### Phase 2: Apply Steering and Compare Generations

For each test prompt:
1. Generate baseline output (no steering)
2. Generate steered output (with actadd applied)
3. Compare to validate steering effect

### Phase 3: Extract Activations

1. Extract baseline hidden states at steering layer (iteratively per token)
2. Extract steered hidden states at steering layer (with hooks applied)
3. Measure activation difference

### Phase 4: Inversion

1. Invert baseline activations → should recover original prompt
2. Invert steered activations → key experiment
3. Compare reconstruction accuracy

### Phase 5: Behavioral Evaluation

If steered inversion succeeds:
1. Generate with the inverted prompt (no steering)
2. Compare behavior to steered generation
3. Measure if inverted prompt naturally bypasses refusal

## Key Questions

1. **Feasibility**: Do prompts exist that naturally produce steered activations?
2. **Uniqueness**: How similar are inverted prompts to originals?
3. **Behavioral Equivalence**: Do inverted prompts exhibit the same behavior as steering?
4. **Semantic Preservation**: Is meaning preserved in inversion?

## Implementation Structure

```
invertsteer/
├── config.py          # Configuration (loads direction from refusal_direction)
├── model_utils.py     # Model loading (no layer norm modification)
├── steering.py        # Modular steering interface (uses refusal_direction hooks)
├── inversion.py       # SIP-It algorithm (inverts at steering layer)
├── experiment.py      # Main experiment script
├── evaluate.py        # Metrics and analysis
├── environment.yml    # Conda environment
├── requirements.txt   # Pip dependencies
└── README.md          # Setup and usage
```

## Expected Outcomes

### If inversion succeeds:
- Natural prompts exist that bypass refusal without intervention
- Implications for jailbreak discovery
- Understanding of refusal geometry in activation space

### If inversion fails:
- Steered activations are "unnatural" (not producible by any prompt)
- Activation-space interventions are fundamentally different from prompt engineering
- Steering creates out-of-distribution activations

## Experiments

### Experiment 1: Baseline Inversion Validation
- Invert normal (non-steered) activations
- Validate SIP-It works at intermediate layers (not just last layer)

### Experiment 2: Steered Inversion
- Apply steering and invert resulting activations
- Compare reconstruction accuracy with baseline

### Experiment 3: Behavioral Comparison
- Test if inverted prompts actually bypass refusal
- Compare: original prompt, steered activation, inverted prompt

### Experiment 4: Layer Sweep
- Try inverting at different layers
- Find optimal layer for reconstruction

## Considerations

### Layer Selection
- Inversion at steering layer is theoretically cleaner
- May need to experiment with ±1 layer

### Activation Manifold
- Steered activations may lie outside natural activation manifold
- High MSE after inversion indicates "unreachability"

### Computational Cost
- O(vocab_size × prompt_length) forward passes per inversion
- Focus on shorter prompts initially

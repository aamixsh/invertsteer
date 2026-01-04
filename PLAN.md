# Inverting Intervened Activations: Bridging Refusal Direction Ablation and Prompt Inversion

## Goal

Find whether there exists a natural prompt that produces the same activations as a refusal-direction-intervened activation. In other words, can we invert an activation that has been modified to remove the refusal direction back into the prompt space?

## Background

### SIP-It (Soft Inverse Prompt Inversion)
SIP-It is an algorithm that reconstructs prompts from hidden state activations by:
1. Computing target hidden states at a specific layer for the original input
2. For each token position iteratively:
   - Start with a random token embedding
   - Compute gradient of MSE loss between current hidden state and target
   - Use gradient direction to find the nearest token in vocabulary
   - Repeat until convergence or finding an exact match
3. Chain token finding across all positions for full prompt reconstruction

Key insight: The algorithm uses the gradient to navigate the discrete token space efficiently, eliminating tokens that increase the loss.

### Refusal Direction
The refusal direction method identifies a 1-dimensional subspace in activation space that mediates refusal behavior:
1. Collect activations from harmful and harmless prompts
2. Compute refusal direction = mean(harmful_activations) - mean(harmless_activations)
3. Ablate this direction via: `activation' = activation - (activation · r̂) r̂`
4. This removes the refusal signal, causing the model to comply with harmful requests

## Technical Approach

### Phase 1: Extract Refusal Direction for Llama-3.2-1B-Instruct

1. Load model and tokenizer for `meta-llama/Llama-3.2-1B-Instruct`
2. Prepare harmful instructions (from AdvBench) and harmless instructions (from Alpaca)
3. Compute mean activations for both sets at the last token position
4. Compute refusal direction as the normalized difference of means
5. Select optimal layer for intervention (typically around layer 14 for smaller models, or sweep)

### Phase 2: Apply Intervention and Extract Activations

For a given harmful prompt:
1. Run forward pass with refusal direction ablation hooks on all layers
2. Extract the intervened hidden states at the inversion layer
3. These are the "target" activations we want to invert

### Phase 3: Invert Intervened Activations

Use the SIP-It algorithm to find a prompt that produces the intervened activations:
1. Target the intervened hidden states (not the original ones)
2. Run the gradient-guided token search
3. Evaluate if the reconstructed prompt:
   - Produces activations close to the intervened ones
   - Has the same effect on model behavior (bypasses refusal)

## Key Questions to Answer

1. **Feasibility**: Do prompts exist in the vocabulary space that naturally produce refusal-ablated activations?
2. **Uniqueness**: How similar are the inverted prompts to the original harmful prompts?
3. **Behavioral Equivalence**: Do inverted prompts actually bypass refusal like the activation intervention does?
4. **Semantic Preservation**: Do inverted prompts preserve the semantic meaning of the original query?

## Implementation Structure

```
invertsteer/
├── PLAN.md                    # This document
├── requirements.txt           # Dependencies
├── config.py                  # Configuration and constants
├── model_utils.py             # Model loading and utilities
├── refusal_direction.py       # Extract refusal direction
├── intervention.py            # Apply activation interventions
├── inversion.py               # Adapt SIP-It for intervened activations
├── experiment.py              # Main experiment script
└── evaluate.py                # Evaluation utilities
```

## Challenges and Considerations

### 1. Layer Selection
- SIP-It works at a specific layer, while refusal ablation applies to all layers
- Need to determine the optimal layer for inversion (likely the one where refusal direction is strongest)

### 2. Activation Mismatch
- Intervened activations may lie outside the natural manifold of activations producible by any prompt
- This could cause SIP-It to fail to find an exact match
- May need to measure the "reachability gap"

### 3. Iterative vs. Holistic Hidden States
- SIP-It uses iterative hidden states (computing hidden state after each token)
- Intervention applies to the full-sequence forward pass
- Need to reconcile these two approaches

### 4. Computational Cost
- SIP-It is expensive (O(vocab_size × prompt_length) forward passes)
- May need to limit experiments to short prompts or use approximations

## Proposed Experiments

### Experiment 1: Baseline Inversion
- Invert normal (non-intervened) activations from harmful prompts
- Establish baseline accuracy of SIP-It on Llama-3.2-1B-Instruct

### Experiment 2: Intervened Inversion
- Apply refusal ablation and invert the resulting activations
- Compare reconstruction accuracy with baseline

### Experiment 3: Behavioral Evaluation
- Test if inverted prompts actually bypass refusal
- Compare model outputs for: original prompt, intervened activation, inverted prompt

### Experiment 4: Semantic Analysis
- Analyze semantic similarity between original and inverted prompts
- Look for patterns in how the inversion modifies harmful queries

## Expected Outcomes

1. **If inversion succeeds**: We've found natural prompts that bypass refusal, which has implications for:
   - Jailbreak discovery
   - Understanding the geometry of refusal in activation space
   - Potential defense mechanisms

2. **If inversion fails**: This suggests that:
   - Refusal-ablated activations are "unnatural" (not producible by any prompt)
   - The refusal direction truly separates harmful from safe prompt activations
   - Activation-space interventions are fundamentally different from prompt engineering

## Implementation Plan

### Step 1: Setup (model_utils.py, config.py)
- Load Llama-3.2-1B-Instruct with HuggingFace
- Define chat template and tokenization utilities
- Configure device and data types

### Step 2: Refusal Direction (refusal_direction.py)
- Load harmful/harmless datasets
- Compute mean activations using hooks
- Extract and save refusal direction

### Step 3: Intervention (intervention.py)
- Implement ablation hooks for HuggingFace models
- Extract intervened activations

### Step 4: Inversion (inversion.py)
- Adapt SIP-It code to work with Llama-3.2-1B-Instruct
- Support inverting arbitrary target activations
- Handle the case where target is intervened activation

### Step 5: Experiment (experiment.py)
- Run end-to-end pipeline
- Collect metrics and results

### Step 6: Evaluation (evaluate.py)
- Measure reconstruction accuracy
- Test behavioral equivalence
- Analyze results


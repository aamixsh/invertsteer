# InvertSteer: Inverting Steered Activations

This project investigates whether natural prompts exist that produce the same activations as steered (intervened) activations in language models.

## Core Question

When we apply activation steering (e.g., adding/removing a refusal direction), we modify the model's internal representations. This project asks:

**Can we find a prompt that naturally produces these steered activations without any intervention?**

## How It Works

1. **Extract Steering Direction**: Use the `refusal_direction` pipeline to extract a steering direction (e.g., the refusal direction that mediates model refusal behavior)

2. **Apply Steering**: Use activation addition (actadd) to modify model activations at a specific layer

3. **Invert Activations**: Use the SIP-It algorithm to find a prompt that produces the steered activations

4. **Evaluate**: Compare the inverted prompt's behavior to the original steered behavior

## Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate invertsteer
```

Or if you already have a compatible environment:

```bash
pip install -r requirements.txt
```

### 2. Set Up Refusal Direction Pipeline

The steering direction extraction is handled by the `refusal_direction` repo. First, set it up:

```bash
cd ../refusal_direction
source setup.sh  # Will prompt for HuggingFace and Together AI tokens
```

### 3. Extract Steering Direction

Run the refusal_direction pipeline to extract the direction:

```bash
cd ../refusal_direction
python -m pipeline.run_pipeline --model_path meta-llama/Llama-3.2-1B-Instruct
```

This will save artifacts to `pipeline/runs/Llama-3.2-1B-Instruct/`, including:
- `direction.pt`: The extracted steering direction
- `direction_metadata.json`: Layer and position information

## Usage

### Standalone Inversion Script

The simplest way to test inversion is using `sipit.py`:

```bash
# Basic inversion
python sipit.py --prompt "Hello, how are you?"

# Without chat template (only BOS token for Llama)
python sipit.py --prompt "Hello" --no-chat-template

# Continue even if some tokens fail (use ground truth for failed positions)
python sipit.py --prompt "Test prompt" --continue-on-failure

# Track top-k candidates per position
python sipit.py --prompt "Hello" --top-k 10

# Use a different model
python sipit.py --prompt "Hello" --model openai-community/gpt2
```

### Quick Demo

Test the full pipeline:

```bash
python experiment.py --demo
```

### Run Full Experiment

```bash
python experiment.py
```

### Options

```bash
python experiment.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --device cuda \
    --direction ../refusal_direction/pipeline/runs/Llama-3.2-1B-Instruct/direction.pt \
    --method actadd \
    --coeff 1.0 \
    --lr 1.0 \
    --seed 42 \
    --no-chat-template \
    --continue-on-failure \
    --top-k 10
```

| Option | Description |
|--------|-------------|
| `--model` | HuggingFace model path |
| `--device` | Device to use (cuda, cuda:0, cpu) |
| `--direction` | Path to direction.pt file |
| `--method` | Steering method (`actadd` or `ablation`) |
| `--coeff` | Steering coefficient (use -1 to remove direction, +1 to add) |
| `--lr` | Learning rate for SIP-It inversion |
| `--seed` | Random seed |
| `--no-chat-template` | Don't use chat template, only BOS token (e.g., `<\|begin_of_text\|>` for Llama) |
| `--continue-on-failure` | Continue inversion with ground truth when a token fails |
| `--top-k` | Number of top candidate tokens to track per position (default: 10) |

## Key Features

### Top-K Token Tracking

During inversion, the algorithm tracks the top-k closest tokens at each position. This is useful for:
- Understanding which tokens are "close" to the target
- Debugging failed inversions
- Analyzing the embedding space structure

Even if inversion fails, you can see what candidates were considered:

```python
from inversion import print_top_k_tokens

result = inversion_attack(...)
print_top_k_tokens(result, tokenizer, original_ids)
```

### Continue on Failure

When `--continue-on-failure` is enabled:
- If inversion fails at a token position, the ground truth token is used
- Inversion continues for subsequent tokens
- Failed positions are tracked in the result

This is useful for understanding partial inversion success.

### No Chat Template Mode

For Llama models, there are two modes:

1. **With chat template** (default): Uses the full chat template tokens
   - `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}...`
   - Special tokens: `[128000, 128006, 882, 128007, 271]`

2. **Without chat template** (`--no-chat-template`): Only uses BOS token
   - `<|begin_of_text|>{prompt}`
   - Special tokens: `[128000]`

This is useful for testing raw prompt inversion without chat formatting.

## Project Structure

```
invertsteer/
├── config.py          # Configuration and constants
├── model_utils.py     # Model loading and tokenization utilities
├── steering.py        # Modular steering abstraction (uses refusal_direction hooks)
├── inversion.py       # SIP-It inversion algorithm (core implementation)
├── sipit.py           # Standalone inversion script
├── experiment.py      # Main experiment script
├── evaluate.py        # Evaluation utilities
├── environment.yml    # Conda environment
├── requirements.txt   # Pip requirements
└── outputs/           # Experiment outputs
```

## Key Concepts

### Steering Methods

1. **ActAdd (Activation Addition)**: Add a direction to activations at a specific layer
   ```
   activation' = activation + coeff * direction
   ```

2. **Ablation**: Remove a direction from activations across all layers
   ```
   activation' = activation - (activation · r̂) r̂
   ```

### Inversion

The SIP-It algorithm finds a prompt by iteratively:
1. Starting with a random token embedding
2. Computing the gradient of MSE loss between current and target hidden states
3. Using the gradient to find the nearest token in vocabulary
4. Repeating until convergence

### Layer Selection

- Steering is applied at the layer identified by the refusal_direction pipeline
- Inversion targets the same layer (hidden_states[layer+1] after that layer processes)
- No layer norm replacement needed (unlike original SIP-It which inverts from last layer)

## Return Types

### InversionResult

```python
@dataclass
class InversionResult:
    success: bool                           # Whether all tokens were found
    total_time: Optional[float]             # Total inversion time
    discovered_ids: Optional[List[int]]     # Found token IDs
    timesteps: Optional[List[int]]          # Steps per token
    times: Optional[List[float]]            # Time per token
    top_k_per_position: List[List[Tuple[int, float]]]  # Top-k tokens per position
    failed_positions: List[int]             # Positions where inversion failed
```

### TokenSearchResult

```python
@dataclass
class TokenSearchResult:
    token_id: Optional[int]                 # Found token ID
    embedding: Optional[Tensor]             # Token embedding
    timesteps: int                          # Steps taken
    top_k_tokens: List[Tuple[int, float]]   # Top-k candidates (token_id, loss)
    success: bool                           # Whether token was found
```

## Expected Outcomes

### If inversion succeeds:
- We've found natural prompts that bypass refusal
- Implications for jailbreak discovery and defense mechanisms

### If inversion fails:
- Steered activations are "unnatural" (not producible by any prompt)
- Activation-space interventions are fundamentally different from prompt engineering
- Use `--continue-on-failure` to see partial results

## Dependencies

This project depends on:
- **refusal_direction**: For steering direction extraction and hook utilities
- **sipit**: Algorithm inspiration for activation inversion

## Citation

If you use this code, please cite the relevant papers:

```bibtex
@article{arditi2024refusal,
  title={Refusal in Language Models Is Mediated by a Single Direction},
  author={Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Paleka, Daniel and Panickssery, Nina and Gurnee, Wes and Nanda, Neel},
  journal={arXiv preprint arXiv:2406.11717},
  year={2024}
}
@article{nikolaou2025language,
  title={Language models are injective and hence invertible},
  author={Nikolaou, Giorgos and Mencattini, Tommaso and Crisostomi, Donato and Santilli, Andrea and Panagakis, Yannis and Rodol{\`a}, Emanuele},
  journal={arXiv preprint arXiv:2510.15511},
  year={2025}
}
```

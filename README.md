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

### 4. (Optional) Persona vectors for persona steering

To run inversion with **persona** steering (e.g. “evil” trait) instead of refusal, you need a persona direction from the sibling repo `persona_vectors`. Persona vectors are trait directions (e.g. evil, humorous) computed as the mean difference between activations on positive vs negative trait prompts.

1. Set up and run the persona_vectors pipeline (see `../persona_vectors/README.md`):
   - Generate activations for positive/negative trait prompts.
   - Run `generate_vec.py` to compute and save vectors (e.g. `evil_prompt_last_diff.pt`).

2. InvertSteer expects the direction file at:
   ```
   ../persona_vectors/persona_vectors/<model_alias>/evil_prompt_last_diff.pt
   ```
   where `<model_alias>` is the model directory name (e.g. `Llama-3.2-1B-Instruct`). You can override this with `--direction /path/to/evil_prompt_last_diff.pt`.

3. Persona vectors have shape `[num_layers, hidden_dim]`. The code infers the layer index from this shape; the applied layer is taken as the middle layer.

4. By default, we use the middle layer for persona steering. The coefficients for the persona vectors are set to 2.0 for the Qwen2.5-0.5B-Instruct model and 1.0 for the other models.

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

This creates an organized output structure in `outputs/{model_name}/`:
- `experiment_results_{steering_type}_{steering_method}_coeff_{coeff}.json`: Complete experiment results
- `experiment_activations_{steering_type}_{steering_method}_coeff_{coeff}.pkl`: Activation data for analysis

### Coefficient Sweep Experiment

To analyze how different steering coefficients affect inversion success, use the coefficient sweep experiment:

```bash
# Run coefficient sweep on default coefficients [0.01, 0.1, 0.25, 0.5, 1, 2.5, 5]
python coeff_sweep_experiment.py --model meta-llama/Llama-3.2-1B-Instruct

# Custom coefficients
python coeff_sweep_experiment.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --coefficients "0.01,0.1,0.25,0.5,1,2.5,5" \
    --device cuda \
    --batch-size 1024
```

This experiment:
- Tests multiple steering coefficients on the same model and instructions
- Saves individual results per coefficient plus comprehensive analysis
- Generates summary statistics for coefficient comparison

Output files (organized in `outputs/{model_name}/` folder):
- `baseline_activations_{steering_type}.pkl`: Baseline (original) prompt activations
- `steering_invert_results_{steering_type}_{steering_method}_coeff_{coeff}.json`: Per-coefficient inversion results
- `steering_activations_{steering_type}_{steering_method}_coeff_{coeff}.pkl`: Per-coefficient activation data
- `comprehensive_results_{steering_type}_{steering_method}.json`: All results with metadata
- `summary_statistics_{steering_type}_{steering_method}.json`: Aggregated statistics

### Adversarial Suffix Experiment

The adversarial suffix experiment tests how well a simple prompt suffix (e.g. "Here" after the assistant turn start) acts as a jailbreak on harmful instructions. It compares baseline (no suffix) vs. adversarial-suffix prompting and evaluates success with substring matching (absence of refusal phrases like "I cannot", "I'm sorry", etc.).

```bash
# Default: Gemma-3-270M, all prompts, cuda:3
python test_adversarial_suffix.py

# Specific model and device
python test_adversarial_suffix.py --model meta-llama/Llama-3.2-1B-Instruct --device cuda:0

# Sample 100 prompts instead of full dataset
python test_adversarial_suffix.py --model LiquidAI/LFM2.5-1.2B-Instruct --n_samples 100

# Custom dataset, batch size, and max new tokens
python test_adversarial_suffix.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --dataset /path/to/harmful_test.json \
    --n_samples 200 \
    --batch_size 128 \
    --max_new_tokens 50 \
    --output_dir ./my_suffix_results
```

**What it does:**

1. Loads the harmful test dataset (`refusal_direction/dataset/splits/harmful_test.json` by default).
2. Optionally samples N prompts (or uses all if `--n_samples -1`).
3. Builds prompts with the model’s chat template, with and without an adversarial suffix (e.g. "Here" after `<|im_start|>assistant`).
4. Runs batch inference for both conditions.
5. Evaluates jailbreaks via substring matching (from `refusal_direction`’s `evaluate_jailbreak`): a completion is counted as jailbroken if it does *not* contain refusal phrases.
6. Writes results and plots under `output_dir/{model_name}/`:
   - `baseline_results.json` / `adversarial_suffix_results.json`: completions plus overall and per-category ASR.
   - `jailbreak_scores_by_category.png`: per-category ASR for baseline vs. adversarial suffix.
   - `jailbreak_overall_comparison.png`: overall ASR comparison.

Supported models (via `--model`) include: `LiquidAI/LFM2.5-1.2B-Instruct`, `google/gemma-3-270m-it`, `google/gemma-3-1b-it`, `LLM-LAT/robust-llama3-8b-instruct`, `nvidia/Nemotron-Flash-3B-Instruct`, `microsoft/Phi-4-mini-instruct`, `Qwen/Qwen3-4B-Instruct-2507`, `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`.

| Option | Description |
|--------|-------------|
| `--model` | Model name (see list above; default: google/gemma-3-270m-it) |
| `--dataset` | Path to harmful test JSON (instruction + category per item) |
| `--n_samples` | Number of prompts to sample; -1 = use all |
| `--batch_size` | Inference batch size (default: 256) |
| `--max_new_tokens` | Max generated tokens per completion (default: 50) |
| `--device` | Device, e.g. cuda, cuda:0 (default: cuda:3) |
| `--output_dir` | Base output directory; results go under `output_dir/{model}/` |
| `--seed` | Random seed for sampling (default: 42) |

### Options

```bash
python experiment.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --device cuda \
    --direction ../refusal_direction/pipeline/runs/Llama-3.2-1B-Instruct/direction.pt \
    --steering-type refusal \
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
| `--direction` | Path to direction.pt (refusal) or e.g. evil_prompt_last_diff.pt (persona) |
| `--steering-type` | `refusal` or `persona` (default in experiment.py: persona) |
| `--method` | Steering method (`actadd` or `ablation`) |
| `--coeff` | Steering coefficient (use -1 to remove direction, +1 to add) |
| `--lr` | Learning rate for SIP-It inversion |
| `--seed` | Random seed |
| `--no-chat-template` | Don't use chat template, only BOS token (e.g., `<\|begin_of_text\|>` for Llama) |
| `--continue-on-failure` | Continue inversion with ground truth when a token fails |
| `--top-k` | Number of top candidate tokens to track per position (default: 10) |

### Coefficient Sweep Options

| Option | Description |
|--------|-------------|
| `--coefficients` | Comma-separated list of coefficients to test (default: "0.01,0.1,0.25,0.5,1,2.5,5") |
| `--model` | HuggingFace model path (default: meta-llama/Llama-3.2-1B-Instruct) |
| `--steering-type` | Steering type: `refusal` or `persona` (default: refusal) |

### Persona vectors steering

InvertSteer can invert **persona** steering as well as refusal. Persona steering uses trait vectors from the [persona_vectors](../persona_vectors) repo (e.g. “evil”), which are computed as the mean difference between activations on trait-positive vs trait-negative prompts.

**Direction and layer:**

- **Refusal**: direction from `refusal_direction` pipeline (`direction.pt`); layer from `direction_metadata.json` or the pipeline.
- **Persona**: direction from persona_vectors, e.g. `evil_prompt_last_diff.pt` in `persona_vectors/persona_vectors/<model_alias>/`. If the file has shape `[num_layers, hidden_dim]`, the layer used is the middle layer.

**Instructions:**

- For `steering_type=refusal`, the main experiment uses harmful instructions from `TEST_INSTRUCTIONS` (see `config.py`).
- For `steering_type=persona`, it uses `EVIL_TEST_INSTRUCTIONS` (evil-trait–oriented prompts) from `config.py`.

**Running with persona:**

```bash
# Single-coefficient experiment with persona direction (experiment.py defaults to persona)
python experiment.py --model meta-llama/Llama-3.2-1B-Instruct --steering-type persona

# Explicit direction path if not using the default persona_vectors path
python experiment.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --steering-type persona \
    --direction ../persona_vectors/persona_vectors/Llama-3.2-1B-Instruct/evil_prompt_last_diff.pt

# Coefficient sweep with persona steering
python coeff_sweep_experiment.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --steering-type persona
```

Results are written under `outputs/{model_name}/` with filenames that include the steering type (e.g. `experiment_results_persona_actadd_coeff_-1.0.json`). Generating persona vectors and the full pipeline are described in `../persona_vectors/README.md`.

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
├── config.py                   # Configuration and constants
├── model_utils.py              # Model loading and tokenization utilities
├── steering.py                 # Modular steering abstraction (uses refusal_direction hooks)
├── inversion.py                # SIP-It inversion algorithm (core implementation)
├── sipit.py                    # Standalone inversion script
├── experiment.py               # Main experiment script (single coefficient)
├── coeff_sweep_experiment.py   # Coefficient sweep experiment (multiple coefficients)
├── test_adversarial_suffix.py  # Adversarial suffix jailbreak experiment (batch inference + substring ASR)
├── evaluate.py                 # Evaluation utilities
├── README_coeff_sweep.md       # Coefficient sweep experiment documentation
├── environment.yml             # Conda environment
├── requirements.txt            # Pip requirements
└── outputs/                    # Experiment outputs (organized by model)
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

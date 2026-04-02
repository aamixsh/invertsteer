# InvertSteer: Inverting Steered Activations

This project investigates whether natural prompts exist that produce the same activations as steered (intervened) activations in language models.

## Core Question

When we apply activation steering (e.g., adding/removing a refusal direction), we modify the model's internal representations. This project asks:

**Can we find a prompt that naturally produces these steered activations without any intervention?**

## How It Works

1. **Extract steering direction**: Use the `refusal_direction` pipeline (or persona vectors) to obtain a direction.
2. **Apply steering**: Use activation addition (actadd) or ablation at a chosen layer.
3. **Invert activations**: Use SIP-It (gradient-based), optional CMA-ES or linear search, or softer methods (soft prefix / prompt optimizers) to approximate steered activations with natural or continuous prefixes.
4. **Evaluate**: Compare inverted prompts, copying baselines, activation alignment, and cross-prompt geometry.

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

The steering direction extraction is handled by the `refusal_direction` repo:

```bash
cd ../refusal_direction
source setup.sh  # Will prompt for HuggingFace and Together AI tokens
```

### 3. Extract Steering Direction

```bash
cd ../refusal_direction
python -m pipeline.run_pipeline --model_path meta-llama/Llama-3.2-1B-Instruct
```

Artifacts go to `pipeline/runs/Llama-3.2-1B-Instruct/`, including:

- `direction.pt`: The extracted steering direction
- `direction_metadata.json`: Layer and position information

### 4. (Optional) Persona vectors for persona steering

For **persona** steering (e.g. an “evil” trait) instead of refusal, use vectors from the sibling repo `persona_vectors`. They are trait directions computed as the mean difference between activations on positive vs negative trait prompts.

1. Set up and run the persona_vectors pipeline (see `../persona_vectors/README.md`).
2. InvertSteer expects a file such as:
   ```
   ../persona_vectors/persona_vectors/<model_alias>/evil_prompt_last_diff.pt
   ```
   Override with `--direction /path/to/evil_prompt_last_diff.pt` when needed.
3. Persona tensors with shape `[num_layers, hidden_dim]` use the **middle layer** unless you change direction handling in code.
4. Coefficients for persona steering are tuned per model family in the steering utilities (e.g. higher for small Qwen instruct models).

## Configuration (`config.py`)

`Config` holds defaults used across scripts:

| Field | Role |
|--------|------|
| `model_id`, `device`, `dtype` | Model and compute |
| `load_in_4bit` | NF4 via BitsAndBytes (GPU + `bitsandbytes`) |
| `steering_type` | `refusal` or `persona` |
| `steering_method`, `steering_coeff` | `actadd` / `ablation` and scale (often `-1.0` to subtract refusal) |
| `batch_size`, `n_gpus`, `shuffle_candidates` | Baseline / inversion search |
| `cmaes`, `cmaes_max_iterations`, `cmaes_stdev_init` | Optional evolutionary inversion |
| `max_prompt_length`, `continue_on_failure`, `top_k` | SIP-It behavior |

Instruction sets:

- `TEST_INSTRUCTIONS`: harmful probes (refusal experiments).
- `TEST_INSTRUCTIONS_REPHRASED`: five paraphrases per harmful probe (cross-prompt tools).
- `EVIL_TEST_INSTRUCTIONS`: persona-oriented prompts.
- `HARMLESS_INSTRUCTIONS`: benign comparison prompts.

## Usage

### Standalone inversion (`sipit.py`)

```bash
python sipit.py --prompt "Hello, how are you?"
python sipit.py --prompt "Hello" --no-chat-template
python sipit.py --prompt "Test prompt" --continue-on-failure
python sipit.py --prompt "Hello" --top-k 10
python sipit.py --prompt "Hello" --model openai-community/gpt2
```

### Main experiment (`experiment.py`)

Runs baseline vs steered generation and SIP-It inversion per instruction.

```bash
python experiment.py --demo          # quick demo
python experiment.py                 # full run with Config / CLI defaults
```

Outputs under `outputs/{model_alias}/`:

- `experiment_results_{steering_type}_{steering_method}_coeff_{coeff}.json`
- `experiment_activations_{steering_type}_{steering_method}_coeff_{coeff}.pkl`

**Notable flags:**

| Option | Description |
|--------|-------------|
| `--model` | HuggingFace model id |
| `--device` | e.g. `cuda`, `cuda:0`, `auto` |
| `--load-in-4bit` | NF4 load (needs GPU + bitsandbytes) |
| `--direction` | Path to `direction.pt` / persona `.pt` |
| `--steering-type` | `refusal` (default) or `persona` |
| `--method` | `actadd` or `ablation` |
| `--coeff` | Steering scale (default `-1.0`) |
| `--invert-original` / `--invert-steered` | Which run to invert |
| `--lr` | SIP-It learning rate |
| `--no-chat-template` / `--add-special-tokens` | Tokenization controls |
| `--continue-on-failure` | Fill failed positions with ground-truth tokens |
| `--top-k` | Track top-k candidate tokens per position |
| `--batch-size`, `--n-gpus`, `--shuffle-candidates` | Search / sharding |
| `--cmaes`, `--linear` | Alternative inversion modes |
| `--cmaes-max-iterations`, `--cmaes-stdev-init` | CMA-ES hyperparameters |

### Coefficient sweep (`coeff_sweep_experiment.py`)

Runs inversion across multiple coefficients.

```bash
python coeff_sweep_experiment.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --coefficients "0.01,0.1,0.25,0.5,1,2.5,5"
```

**Defaults (check `--help` if you rely on them):** `--coefficients` is `1.0`, `--steering-type` is `persona`, `--device` is `cuda:3`. Override explicitly for refusal sweeps or a different GPU.

Outputs in `outputs/{model_alias}/`:

- `steering_invert_results_{steering_type}_{steering_method}_coeff_{coeff}.json`
- `steering_activations_{steering_type}_{steering_method}_coeff_{coeff}.pkl`
- `comprehensive_results_{steering_type}_{steering_method}.json`
- `summary_statistics_{steering_type}_{steering_method}.json`

### Reconstructed activations pickle (`collect_reconstructed_activations.py`)

After `experiment.py` or coeff sweep, aggregates **baseline / steered / reconstructed** activations aligned per instruction:

```bash
python collect_reconstructed_activations.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --device cuda:0
```

Discovers `experiment_results_*.json` and `steering_invert_results_*.json` under `outputs/{model_alias}/` and writes e.g.  
`experiment_activations_{steering_type}_{steering_method}_coeff_{coeff}_with_recon.pkl` (or `steering_activations_*` for sweeps). Use these with `evaluate.py` plots.

### Soft prefix steering (`soft_prefix_steering_experiment.py`)

Optimizes a **continuous** prefix in embedding space to match steered hidden states at prompt positions; reports best length, discrete projection, and generations (baseline / steered / soft / projected discrete).

```bash
python soft_prefix_steering_experiment.py \
  --model meta-llama/Llama-3.2-1B-Instruct
```

Outputs: `outputs/{model_alias}/soft_prefix_steering/soft_prefix_results_{steering_type}_{steering_method}_coeff_{coeff}.json`

Key knobs: `--max-prefix-len`, `--opt-steps`, `--opt-lr`, `--max-new-tokens`, `--load-in-4bit`.

### Prompt optimizers — DSPy GEPA + virtual tokens (`prompt_activation_alignment_experiment.py`)

Aligns natural prompts to steered completions using **GEPA** (DSPy) and/or **prefix tuning** (virtual tokens), with teacher-forced vs free-generation activation metrics.

```bash
python prompt_activation_alignment_experiment.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

Outputs: `outputs/{model_alias}/prompt_opt/prompt_opt_results_*.json` and `prompt_opt_activations_*.pkl`.

### Many-shot activation alignment (`manyshot_alignment_experiment.py`)

Builds ICL prefixes from steered completions and measures whether natural many-shot contexts move activations toward steered targets (variable `N` demos).

```bash
python manyshot_alignment_experiment.py \
  --n-demos "1,2,4,8,16,32,64"
```

Outputs: `outputs/{model_alias}/manyshot/manyshot_results_*.json` and `manyshot_activations_*.pkl`. Optional `--icl-path` for precomputed demonstrations; `--analyze` for post-hoc figure/analysis on a saved JSON.

### Cross-prompt activation comparison (`cross_prompt_activation_comparison.py`)

On **prompt-only** forward passes, compares the L2 “steering delta” on each instruction to pairwise distances when swapping instructions—on the user span and on the shared assistant tail. Supports `TEST_INSTRUCTIONS_REPHRASED` and within-group tail metrics.

```bash
python cross_prompt_activation_comparison.py
python cross_prompt_activation_comparison.py --use-rephrased-prompts
python cross_prompt_activation_comparison.py --rephrased-within-group-tail
```

Outputs: `outputs/{model_alias}/cross_prompt_activation/cross_prompt_activation_*.json`

### Copying baseline (`copying_baseline_experiment.py`)

Tests whether the model can **repeat** a steered generation verbatim (explicit instruction and in-context repetition), with token and activation metrics vs steered and baseline teacher-forcing references.

```bash
python copying_baseline_experiment.py --icl-ns "1,2,4"
```

Outputs: `outputs/{model_alias}/copying_baseline/copying_baseline_*.json` and optional `*_activations.pkl`.

### Adversarial suffix benchmark (`test_adversarial_suffix.py`)

Batch harmful completions with/without a short adversarial suffix; substring-based ASR vs refusal phrases.

```bash
python test_adversarial_suffix.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset ../refusal_direction/dataset/splits/harmful_test.json
```

Defaults: model `meta-llama/Meta-Llama-3-8B-Instruct`, device `cuda:0`, output under `./adversarial_suffix_results/{model}/`. Adjust `--dataset` to your local `harmful_test.json` path.

### Evaluation and plotting (`evaluate.py`)

Python helpers to load experiment JSONs / pickles, aggregate metrics, and plot top-k distances, activation distances (including files from `collect_reconstructed_activations.py`). Intended for notebooks or small drivers after experiments have run.

## Persona steering (quick reference)

- **Refusal**: direction from `refusal_direction` (`direction.pt`); layer from metadata / pipeline.
- **Persona**: e.g. `evil_prompt_last_diff.pt` under `persona_vectors/persona_vectors/<model_alias>/`.
- **Instructions**: refusal runs use `TEST_INSTRUCTIONS`; persona runs use `EVIL_TEST_INSTRUCTIONS`.

```bash
python experiment.py --steering-type persona \
  --model meta-llama/Llama-3.2-1B-Instruct

python coeff_sweep_experiment.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --steering-type persona \
  --coefficients "0.5,1,2"
```

## Key Features

### Top-K token tracking

During inversion, top-k closest tokens per position are stored for debugging and analysis. Use `inversion.print_top_k_tokens` in code on an `InversionResult`.

### Continue on failure

With `--continue-on-failure`, failed positions get the ground-truth token and inversion continues; `failed_positions` records where this happened.

### Chat template vs raw prompt

- **With chat template** (default): full chat formatting.
- **`--no-chat-template`**: minimal special-token prefix (e.g. Llama BOS only).

Use the flag consistent with how you extracted directions and what you want to invert.

## Project Structure

```
invertsteer/
├── config.py                          # Config, instruction lists
├── model_utils.py                     # Load model, tokenize, specials
├── steering.py                        # Steering abstraction + generation hooks
├── inversion.py                       # SIP-It and related inversion
├── sipit.py                           # CLI inversion
├── experiment.py                      # Main inversion experiment
├── coeff_sweep_experiment.py         # Multi-coefficient inversion sweep
├── collect_reconstructed_activations.py
├── soft_prefix_steering_experiment.py
├── prompt_activation_alignment_experiment.py
├── manyshot_alignment_experiment.py
├── cross_prompt_activation_comparison.py
├── copying_baseline_experiment.py
├── test_adversarial_suffix.py
├── evaluate.py                        # Metrics and plots
├── environment.yml
├── requirements.txt
└── outputs/                           # Per-model subfolders (see above)
```

## Key Concepts

### Steering methods

1. **ActAdd**: `activation' = activation + coeff * direction`
2. **Ablation**: remove component along a unit direction.

### Inversion layer

Steering applies at `layer` from the direction pipeline; inversion targets hidden states at `layer + 1` after that block, consistent across experiments.

### Return types (inversion)

- **`InversionResult`**: `success`, `discovered_ids`, `top_k_per_position`, `failed_positions`, timings, etc.
- **`TokenSearchResult`**: per-token search outcome and top-k candidates.

## Expected Outcomes

- If inversion succeeds, steered geometry may be reachable with natural discrete prompts (security and interpretability implications).
- If it fails, steered states may lie off the prompt manifold; partial results via `--continue-on-failure` or soft-prefix / optimizer baselines still inform the gap.

## Dependencies

- **`refusal_direction`**: directions, hooks, optional jailbreak eval helpers.
- **Persona / DSPy / bitsandbytes**: as required by the scripts you run (`prompt_activation_alignment_experiment.py` needs DSPy stack).

## Citation

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

"""
Prompt-optimizer activation alignment experiment.

For each instruction:
1) Generate a steered response (using activation steering).
2) Extract steering-layer activations for the target sequence [instruction + steered_response].
3) Run prompt optimizers to create natural prompts that match the steered outputs.
   - Free replacement prompt (replaces the instruction)
   - Discrete prefix prompt (prefix uses real tokens; length is fixed)
4) Evaluate activation alignment under:
   - Teacher forcing (natural forward on target output tokens)
   - Generated output (natural forward on model-generated output tokens)
"""

import os
import json
import pickle
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm

import torch
import threading

from config import Config, TEST_INSTRUCTIONS
from model_utils import load_model, get_tokenize_fn, set_seed, format_instruction
from steering import (
    SteeringConfig,
    load_steering_direction,
    generate_with_steering,
    get_hidden_states_with_steering,
)
from inversion import extract_hidden_states
from model_utils import get_input_device

def compute_token_alignment(
    acts_steered: torch.Tensor,
    acts_natural: torch.Tensor,
    metric: str = "l2",
) -> torch.Tensor:
    if metric == "cosine":
        a = acts_steered / (acts_steered.norm(dim=-1, keepdim=True) + 1e-8)
        b = acts_natural / (acts_natural.norm(dim=-1, keepdim=True) + 1e-8)
        return (a * b).sum(dim=-1)
    if metric == "l2":
        return torch.norm(acts_steered - acts_natural, dim=-1)
    raise ValueError(f"Unknown metric: {metric}")


def get_output_ids(full_input_ids: torch.Tensor, instruction_ids: torch.Tensor) -> torch.Tensor:
    instr_len = instruction_ids.size(1)
    return full_input_ids[:, instr_len:]


def get_prefix_insertion_index(tokenizer, tokenize_fn) -> int:
    try:
        ids_empty = tokenize_fn(instructions=[""]).input_ids[0].tolist()
        ids_marker = tokenize_fn(instructions=["UNIQUE_MARKER_XYZ"]).input_ids[0].tolist()
    except Exception:
        return 0
    lcp = 0
    for a, b in zip(ids_empty, ids_marker):
        if a == b:
            lcp += 1
        else:
            break
    if lcp >= len(ids_empty) or lcp >= len(ids_marker):
        return 0
    return lcp


def build_prefix_input_ids(
    instruction_ids: torch.Tensor,
    prefix_ids: torch.Tensor,
    insertion_index: int,
) -> torch.Tensor:
    device = instruction_ids.device
    prefix_ids = prefix_ids.to(device)
    before = instruction_ids[:, :insertion_index]
    after = instruction_ids[:, insertion_index:]
    return torch.cat([before, prefix_ids, after], dim=1)


def compute_generated_ids(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = outputs[:, input_ids.size(1):]
    # Validate generated ids to avoid downstream decode crashes.
    vocab_size = int(model.get_input_embeddings().weight.size(0))
    if gen.numel():
        mn = int(gen.min().item())
        mx = int(gen.max().item())
        if mn < 0 or mx >= vocab_size:
            raise ValueError(
                f"Generated token id out of range: min={mn}, max={mx}, vocab_size={vocab_size}"
            )
    return gen


def compute_alignment_with_prefix_stitch(
    steered_acts_full: torch.Tensor,
    natural_acts_candidate: torch.Tensor,
    insertion_index: int,
    prefix_len: int,
    full_seq_len: int,
    metric: str = "l2",
) -> torch.Tensor:
    stitched = torch.cat(
        [
            natural_acts_candidate[:insertion_index],
            natural_acts_candidate[insertion_index + prefix_len:],
        ],
        dim=0,
    )
    length = min(full_seq_len, stitched.size(0))
    return compute_token_alignment(
        steered_acts_full[:length],
        stitched[:length],
        metric=metric,
    )


def evaluate_candidate_prompt(
    model,
    tokenizer,
    candidate_input_ids: torch.Tensor,   # [1, prompt_len]
    output_ids_target: torch.Tensor,     # [1, resp_len]
    steered_acts_full: torch.Tensor,     # [full_seq_len, d] on CPU
    inversion_layer: int,
    max_new_tokens: int,
    mode: str = "replacement",           # "replacement" | "prefix"
    insertion_index: int = 0,
    prefix_len: int = 0,
    full_seq_len: int = 0,
    metric: str = "l2",
) -> Dict[str, Any]:
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    candidate_input_ids = candidate_input_ids.to(device)
    output_ids_target_dev = output_ids_target.to(device)

    # Teacher forcing activations
    tf_input_ids = torch.cat([candidate_input_ids, output_ids_target_dev], dim=1)
    natural_acts_tf = extract_hidden_states(tf_input_ids, model, inversion_layer).cpu()

    if mode == "replacement":
        length = min(full_seq_len, natural_acts_tf.size(0))
        align_tf = compute_token_alignment(
            steered_acts_full[:length],
            natural_acts_tf[:length],
            metric=metric,
        )
    else:
        align_tf = compute_alignment_with_prefix_stitch(
            steered_acts_full,
            natural_acts_tf,
            insertion_index,
            prefix_len,
            full_seq_len,
            metric=metric,
        )

    # Generated output
    gen_ids = compute_generated_ids(model, tokenizer, candidate_input_ids, max_new_tokens)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    target_list = output_ids_target[0].tolist()
    gen_list = gen_ids[0].tolist()
    min_len = min(len(target_list), len(gen_list))
    exact_match = (gen_list[:min_len] == target_list[:min_len]) and (len(gen_list) == len(target_list))
    matches = sum(1 for a, b in zip(gen_list[:min_len], target_list[:min_len]) if a == b)
    token_overlap = matches / max(len(target_list), 1)

    gen_input_ids = torch.cat([candidate_input_ids, gen_ids.to(device)], dim=1)
    natural_acts_gen = extract_hidden_states(gen_input_ids, model, inversion_layer).cpu()

    if mode == "replacement":
        length = min(full_seq_len, natural_acts_gen.size(0))
        align_gen = compute_token_alignment(
            steered_acts_full[:length],
            natural_acts_gen[:length],
            metric=metric,
        )
    else:
        align_gen = compute_alignment_with_prefix_stitch(
            steered_acts_full,
            natural_acts_gen,
            insertion_index,
            prefix_len,
            full_seq_len,
            metric=metric,
        )

    return {
        "teacher_forcing": {
            "alignment_per_token": align_tf.tolist(),
            "alignment_mean": align_tf.mean().item(),
        },
        "generated_text": gen_text,
        "exact_match": exact_match,
        "token_overlap": token_overlap,
        "generated_output": {
            "alignment_per_token": align_gen.tolist(),
            "alignment_mean": align_gen.mean().item(),
        },
    }


def _nearest_token_ids(
    virtual_embeddings: torch.Tensor,  # [num_virtual, d_model]
    embedding_matrix: torch.Tensor,    # [vocab_size, d_model]
) -> List[int]:
    chunk = 4096
    vemb = virtual_embeddings.float()
    emat = embedding_matrix.float()
    best_ids: List[int] = []
    for v in vemb:
        best_id = 0
        best_dist = float("inf")
        for start in range(0, emat.size(0), chunk):
            end = min(start + chunk, emat.size(0))
            dists = torch.norm(emat[start:end] - v.unsqueeze(0), dim=1)
            local_best = dists.argmin().item()
            if dists[local_best].item() < best_dist:
                best_dist = dists[local_best].item()
                best_id = start + local_best
        best_ids.append(best_id)
    return best_ids


def run_gepa_experiment(
    model,
    tokenizer,
    tokenize_fn,
    instruction: str,
    output_ids_target: torch.Tensor,
    steered_acts_full: torch.Tensor,
    inversion_layer: int,
    insertion_index: int,
    prefix_len: int,
    full_seq_len: int,
    max_new_tokens: int,
    gepa_budget: str = "medium",
    metric: str = "l2",
    gepa_max_steps: Optional[int] = None,
    gepa_reflection_lm_provider: str = "inproc",
    gepa_reflection_model: str = "gpt-4o-mini",
    gepa_reflection_max_tokens: int = 512,
    gepa_log_candidates: bool = False,
) -> Dict[str, Any]:
    import dspy
    import inspect

    target_token_list = output_ids_target[0].tolist()
    target_text = tokenizer.decode(target_token_list, skip_special_tokens=True)
    device = get_input_device(model)

    class InProcHFLM(dspy.BaseLM):
        def __init__(
            self,
            hf_model,
            hf_tokenizer,
            *,
            max_new_tokens: int = 256,
            wrap_json_output: bool = False,
        ):
            super().__init__(model="inproc-hf", model_type="chat")
            self.hf_model = hf_model
            self.hf_tokenizer = hf_tokenizer
            self._max_new_tokens = max_new_tokens
            self._wrap_json_output = wrap_json_output
            self.history = []
            self._gen_lock = threading.Lock()

        def __call__(self, prompt=None, messages=None, **kwargs):
            import json as _json
            if messages is not None:
                text = self.hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            elif prompt is not None:
                text = prompt
            else:
                return [""]

            # Tokenize on CPU first, validate id range, then move to model input device.
            inputs = self.hf_tokenizer(text, return_tensors="pt")
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise RuntimeError("Tokenizer did not return input_ids")

            # Validate token-id range to avoid CUDA device-side asserts in embedding lookup.
            # NOTE: For some HF tokenizers, `.vocab_size` excludes added/special tokens.
            # Use the largest of: len(tokenizer) (includes added tokens) and the model's
            # embedding matrix size (the authoritative bound for embedding lookup).
            vocab_size = 0
            try:
                vocab_size = max(vocab_size, int(len(self.hf_tokenizer)))
            except Exception:
                pass
            try:
                vocab_size = max(vocab_size, int(getattr(self.hf_tokenizer, "vocab_size", 0) or 0))
            except Exception:
                pass
            try:
                vocab_size = max(vocab_size, int(self.hf_model.get_input_embeddings().weight.size(0)))
            except Exception:
                pass
            if vocab_size > 0:
                mn = int(input_ids.min().item())
                mx = int(input_ids.max().item())
                if mn < 0 or mx >= vocab_size:
                    raise ValueError(f"Token id out of range: min={mn}, max={mx}, vocab_size={vocab_size}")

            inputs = inputs.to(device)
            # IMPORTANT: HF generation is not thread-safe; DSPy may call this LM
            # concurrently (parallelizer). Serialize generate() calls to avoid
            # corrupted outputs / invalid token ids.
            with self._gen_lock, torch.no_grad():
                out = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.hf_tokenizer.pad_token_id,
                )
            # Surface any CUDA error at the real callsite.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            gen_ids = out[0, inputs.input_ids.size(1):].detach().to("cpu")
            # Validate generated ids before decode (prevents tokenizer overflow errors).
            # Defensive checks: ensure output tensor looks like token ids.
            if not isinstance(out, torch.Tensor):
                raise TypeError(f"model.generate returned non-tensor: {type(out)}")
            if out.dtype != torch.long:
                raise TypeError(f"model.generate returned dtype {out.dtype}, expected torch.long")
            if out.dim() != 2:
                raise ValueError(f"model.generate returned shape {tuple(out.shape)}, expected [batch, seq]")

            emb_vocab = int(self.hf_model.get_input_embeddings().weight.size(0))
            if gen_ids.numel():
                mn2 = int(gen_ids.min().item())
                mx2 = int(gen_ids.max().item())
                if mn2 < 0 or mx2 >= emb_vocab:
                    raise ValueError(
                        f"Generated token id out of range: min={mn2}, max={mx2}, vocab_size={emb_vocab}"
                    )
            gen = self.hf_tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
            self.history.append({"prompt": text, "response": gen, "messages": messages})
            if self._wrap_json_output:
                # DSPy Predict('prompt -> output') expects a parseable 'output' field.
                # Returning JSON keeps the adapter happy.
                return [_json.dumps({"output": gen})]
            return [gen]

    # Use a JSON-wrapping LM for the task program (so outputs parse),
    # and a plain-text LM for reflection (so GEPA reflection isn't constrained to JSON).
    inproc_task_lm = InProcHFLM(model, tokenizer, wrap_json_output=True)
    if gepa_reflection_lm_provider == "openai":
        # Requires OPENAI_API_KEY in environment.
        # Keep task LM local HF and only offload reflection LM.
        try:
            reflection_lm = dspy.LM(
                model=f"openai/{gepa_reflection_model}",
                temperature=1.0,
                max_tokens=gepa_reflection_max_tokens,
            )
        except Exception:
            # Fallback for DSPy versions that expect model without provider prefix.
            reflection_lm = dspy.LM(
                model=gepa_reflection_model,
                temperature=1.0,
                max_tokens=gepa_reflection_max_tokens,
            )
    else:
        reflection_lm = InProcHFLM(model, tokenizer, wrap_json_output=False)

    # Force JSON adapter so the 'output' field is parsed from JSON.
    dspy.configure(lm=inproc_task_lm, adapter=dspy.JSONAdapter())

    eval_counter = {"count": 0}

    class PromptOnlyModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.Predict("prompt -> output")

        def forward(self, prompt: str):
            if gepa_log_candidates:
                preview = (prompt or "")[:240].replace("\n", "\\n")
                print(f"[GEPA][candidate_input] prompt='{preview}'")
            return self.generate(prompt=prompt)

    def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        eval_counter["count"] += 1
        got_text = getattr(pred, "output", "") or ""
        got_ids = tokenizer.encode(got_text, add_special_tokens=False)
        target = target_token_list
        min_len = min(len(got_ids), len(target))
        if min_len == 0:
            score = 0.0
        else:
            matches = sum(1 for a, b in zip(got_ids[:min_len], target[:min_len]) if a == b)
            score = matches / max(len(target), 1)

        if score == 1.0 and len(got_ids) == len(target):
            feedback = "Perfect match! All target tokens produced correctly."
        else:
            first_mismatch = next(
                (i for i, (a, b) in enumerate(zip(got_ids[:min_len], target[:min_len])) if a != b),
                min_len,
            )
            got_str = tokenizer.decode(got_ids[:5], skip_special_tokens=True)
            tgt_str = tokenizer.decode(target[:5], skip_special_tokens=True)
            feedback = (
                f"Token overlap score: {score:.3f}. "
                f"First mismatch at position {first_mismatch} "
                f"(got '{got_str}', expected '{tgt_str}'). "
                f"Target output: '{target_text[:120]}'. Got: '{got_text[:120]}'."
            )
        if gepa_log_candidates:
            candidate_instruction = None
            # pred_trace often contains call records where modules expose signature.instructions.
            if pred_trace is not None:
                try:
                    for item in reversed(pred_trace):
                        module_obj = item[0] if isinstance(item, tuple) and len(item) >= 1 else None
                        sig = getattr(module_obj, "signature", None)
                        instr = getattr(sig, "instructions", None) if sig is not None else None
                        if isinstance(instr, str) and instr.strip():
                            candidate_instruction = instr
                            break
                except Exception:
                    candidate_instruction = None
            candidate_preview = (
                (candidate_instruction[:240].replace("\n", "\\n"))
                if candidate_instruction
                else "<not available from trace>"
            )
            output_preview = got_text[:240].replace("\n", "\\n")
            print(
                f"[GEPA][eval={eval_counter['count']}] "
                f"candidate_instruction='{candidate_preview}' "
                f"score={score:.4f} output='{output_preview}'"
            )

        # IMPORTANT: Must return a DSPy score object (not a raw dict), otherwise
        # the evaluator will treat the score as a dict and crash when summing.
        return dspy.Prediction(score=score, feedback=feedback)

    results: Dict[str, Any] = {}
    gepa_init_params = set(inspect.signature(dspy.GEPA.__init__).parameters.keys())

    def _build_gepa_kwargs() -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "metric": gepa_metric,
            "reflection_lm": reflection_lm,
            "num_threads": 1,
        }
        if gepa_max_steps is None:
            kwargs["auto"] = gepa_budget
            return kwargs

        # If explicit steps are provided, avoid auto mode and route to whichever
        # step/budget knob this DSPy GEPA version supports.
        if "auto" in gepa_init_params:
            kwargs["auto"] = None
        if "max_steps" in gepa_init_params:
            kwargs["max_steps"] = int(gepa_max_steps)
        elif "num_steps" in gepa_init_params:
            kwargs["num_steps"] = int(gepa_max_steps)
        elif "num_trials" in gepa_init_params:
            kwargs["num_trials"] = int(gepa_max_steps)
        elif "budget" in gepa_init_params:
            kwargs["budget"] = int(gepa_max_steps)
        elif "max_full_evals" in gepa_init_params:
            kwargs["max_full_evals"] = int(gepa_max_steps)
        else:
            raise ValueError(
                "This DSPy GEPA version does not expose a recognized step/budget parameter "
                "for --gepa-max-steps."
            )
        return kwargs

    # Replacement
    optimizer_repl = dspy.GEPA(**_build_gepa_kwargs())
    prog_repl = PromptOnlyModule()
    trainset = [dspy.Example(prompt=instruction, output=target_text).with_inputs("prompt")]
    best_prog_repl = optimizer_repl.compile(prog_repl, trainset=trainset)
    best_instruction_repl = best_prog_repl.generate.signature.instructions
    # Keep on CPU; evaluation helper will move to the correct device.
    repl_ids = tokenize_fn(instructions=[best_instruction_repl]).input_ids
    results["replacement"] = evaluate_candidate_prompt(
        model=model,
        tokenizer=tokenizer,
        candidate_input_ids=repl_ids,
        output_ids_target=output_ids_target,
        steered_acts_full=steered_acts_full,
        inversion_layer=inversion_layer,
        max_new_tokens=max_new_tokens,
        mode="replacement",
        full_seq_len=full_seq_len,
        metric=metric,
    )
    results["replacement"]["optimized_instruction"] = best_instruction_repl

    # Prefix
    optimizer_prefix = dspy.GEPA(**_build_gepa_kwargs())

    class PrefixModule(dspy.Module):
        def __init__(self, base_instruction: str):
            super().__init__()
            self.base_instruction = base_instruction
            self.generate = dspy.Predict("prefix -> output")

        def forward(self, prefix: str):
            combined = prefix + " " + self.base_instruction
            return self.generate(prefix=combined)

    prog_prefix = PrefixModule(base_instruction=instruction)
    trainset = [dspy.Example(prefix="", output=target_text).with_inputs("prefix")]
    best_prog_prefix = optimizer_prefix.compile(prog_prefix, trainset=trainset)
    best_prefix_text = best_prog_prefix.generate.signature.instructions

    prefix_ids_raw = tokenizer.encode(best_prefix_text, add_special_tokens=False)
    if len(prefix_ids_raw) >= prefix_len:
        prefix_ids_raw = prefix_ids_raw[:prefix_len]
    else:
        prefix_ids_raw = prefix_ids_raw + [tokenizer.eos_token_id] * (prefix_len - len(prefix_ids_raw))
    prefix_ids_tensor = torch.tensor([prefix_ids_raw], dtype=torch.long)

    instr_ids = tokenize_fn(instructions=[instruction]).input_ids
    cand_ids = build_prefix_input_ids(instr_ids, prefix_ids_tensor, insertion_index)

    results["prefix"] = evaluate_candidate_prompt(
        model=model,
        tokenizer=tokenizer,
        candidate_input_ids=cand_ids,
        output_ids_target=output_ids_target,
        steered_acts_full=steered_acts_full,
        inversion_layer=inversion_layer,
        max_new_tokens=max_new_tokens,
        mode="prefix",
        insertion_index=insertion_index,
        prefix_len=prefix_len,
        full_seq_len=full_seq_len,
        metric=metric,
    )
    results["prefix"]["optimized_prefix_text"] = best_prefix_text
    results["prefix"]["optimized_prefix_ids"] = prefix_ids_raw

    return results


def run_prompt_tuning_experiment(
    model,
    tokenizer,
    tokenize_fn,
    instruction: str,
    output_ids_target: torch.Tensor,
    steered_acts_full: torch.Tensor,
    inversion_layer: int,
    insertion_index: int,
    prefix_len: int,
    full_seq_len: int,
    max_new_tokens: int,
    pt_steps: int = 200,
    pt_lr: float = 3e-2,
    pt_virtual_tokens: int = 10,
    metric: str = "l2",
) -> Dict[str, Any]:
    from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model

    import copy

    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    output_ids_target_dev = output_ids_target.to(device)

    def _run_one(mode_name: str, peft_config: PromptTuningConfig, prompt_input_ids: torch.Tensor) -> Dict[str, Any]:
        peft_model = get_peft_model(copy.deepcopy(model), peft_config)
        peft_model.train()

        trainable = [p for p in peft_model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(trainable, lr=pt_lr)

        prompt_input_ids = prompt_input_ids.to(device)
        full_target = torch.cat([prompt_input_ids, output_ids_target_dev], dim=1)
        labels = torch.full_like(full_target, -100)
        labels[:, prompt_input_ids.size(1):] = output_ids_target_dev

        best_loss = float("inf")
        best_virtual = None

        print(f"Running {mode_name} prompt tuning for {pt_steps} steps...")

        for _ in tqdm(range(pt_steps)):
            optim.zero_grad()
            out = peft_model(input_ids=full_target, labels=labels)
            loss = out.loss
            loss.backward()
            optim.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_virtual = peft_model.prompt_encoder.default.embedding.weight.detach().clone().cpu()

        peft_model.eval()
        if best_virtual is None:
            best_virtual = peft_model.prompt_encoder.default.embedding.weight.detach().clone().cpu()

        emb_matrix = model.get_input_embeddings().weight.detach().cpu()
        virtual_token_ids = _nearest_token_ids(best_virtual, emb_matrix)
        virtual_token_text = tokenizer.decode(virtual_token_ids, skip_special_tokens=False)

        virtual_ids_tensor = torch.tensor([virtual_token_ids], dtype=torch.long)
        if mode_name == "replacement":
            candidate_ids = virtual_ids_tensor
        else:
            candidate_ids = build_prefix_input_ids(prompt_input_ids.cpu(), virtual_ids_tensor, insertion_index)

        eval_result = evaluate_candidate_prompt(
            model=model,
            tokenizer=tokenizer,
            candidate_input_ids=candidate_ids.to(device),
            output_ids_target=output_ids_target,
            steered_acts_full=steered_acts_full,
            inversion_layer=inversion_layer,
            max_new_tokens=max_new_tokens,
            mode=mode_name,
            insertion_index=insertion_index,
            prefix_len=pt_virtual_tokens,
            full_seq_len=full_seq_len,
            metric=metric,
        )
        eval_result["virtual_token_ids"] = virtual_token_ids
        eval_result["virtual_token_text"] = virtual_token_text
        eval_result["best_training_loss"] = best_loss
        print(eval_result)
        return eval_result

    results: Dict[str, Any] = {}
    instruction_only_ids = tokenize_fn(instructions=[instruction]).input_ids

    cfg_repl = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=pt_virtual_tokens,
        prompt_tuning_init_text=instruction,
        tokenizer_name_or_path=tokenizer.name_or_path,
    )
    bos_id = tokenizer.bos_token_id
    repl_base_ids = (
        torch.tensor([[bos_id]], dtype=torch.long)
        if bos_id is not None
        else torch.zeros((1, 1), dtype=torch.long)
    )
    results["replacement"] = _run_one("replacement", cfg_repl, repl_base_ids)

    cfg_prefix = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=pt_virtual_tokens,
        tokenizer_name_or_path=tokenizer.name_or_path,
    )
    results["prefix"] = _run_one("prefix", cfg_prefix, instruction_only_ids)

    return results


def _strip_tensors_from_prompt_opt(data: Any) -> Any:
    if isinstance(data, torch.Tensor):
        return data.tolist()
    if isinstance(data, dict):
        return {k: _strip_tensors_from_prompt_opt(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_strip_tensors_from_prompt_opt(v) for v in data]
    if isinstance(data, (int, float, bool, str)) or data is None:
        return data
    try:
        import json as _json
        _json.dumps(data)
        return data
    except Exception:
        return str(data)


def load_instructions_from_arg(instructions_arg: Optional[str]) -> List[str]:
    if not instructions_arg:
        return TEST_INSTRUCTIONS
    # Comma-separated list
    items = [x.strip() for x in instructions_arg.split(",") if x.strip()]
    return items if items else TEST_INSTRUCTIONS


def run_single_instruction(
    model,
    tokenizer,
    tokenize_fn,
    instruction: str,
    steering_config: SteeringConfig,
    max_new_tokens: int,
    run_gepa: bool,
    run_prompt_tuning: bool,
    prefix_len: int,
    gepa_budget: str,
    gepa_max_steps: Optional[int],
    gepa_reflection_lm_provider: str,
    gepa_reflection_model: str,
    gepa_reflection_max_tokens: int,
    gepa_log_candidates: bool,
    pt_steps: int,
    pt_lr: float,
    pt_virtual_tokens: int,
    alignment_metric: str,
) -> Dict[str, Any]:
    inversion_layer = steering_config.layer + 1

    result: Dict[str, Any] = {
        "instruction": instruction,
        "timestamp": datetime.now().isoformat(),
        "steering_layer": steering_config.layer,
        "steering_method": steering_config.method,
        "steering_coeff": steering_config.coeff,
    }

    # Step 1: tokenize instruction-only prompt
    inputs = tokenize_fn(instructions=[instruction])
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    instruction_seq_len = input_ids.size(1)
    result["instruction_seq_len"] = instruction_seq_len
    result["instruction_ids"] = input_ids[0].tolist()

    # Step 2: generate steered response
    steered_response = generate_with_steering(
        model,
        tokenizer,
        input_ids,
        steering_config,
        max_new_tokens,
        attention_mask,
    )[0]
    result["steered_response"] = steered_response

    # Baseline response (no steering)
    with torch.no_grad():
        baseline_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    baseline_response = tokenizer.decode(
        baseline_outputs[0, input_ids.size(1) :],
        skip_special_tokens=True,
    )
    result["baseline_response"] = baseline_response

    # Step 3: build target full prompt [instruction + steered_response]
    full_prompt = format_instruction(
        tokenizer=tokenizer,
        instruction=instruction,
        output=steered_response,
        include_trailing_whitespace=True,
    )
    full_inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    full_input_ids = full_inputs.input_ids.to(model.device)
    full_seq_len = full_input_ids.size(1)
    result["full_seq_len"] = full_seq_len
    result["full_input_ids"] = full_input_ids[0].tolist()

    # Step 4: extract steering-layer activations
    steered_acts_full = get_hidden_states_with_steering(
        model,
        full_input_ids,
        steering_config,
        inversion_layer,
    ).cpu()
    natural_acts_full = extract_hidden_states(
        full_input_ids,
        model,
        inversion_layer,
    ).cpu()
    result["steered_acts_full_mean"] = compute_token_alignment(
        steered_acts_full, natural_acts_full, metric=alignment_metric
    ).mean().item()

    alignment_per_token = compute_token_alignment(
        steered_acts_full,
        natural_acts_full,
        metric=alignment_metric,
    ).cpu()

    result["alignment_per_token_no_prefix"] = alignment_per_token.tolist()
    result["alignment_instruction_mean_no_prefix"] = (
        alignment_per_token[:instruction_seq_len].mean().item()
    )
    alignment_response = alignment_per_token[instruction_seq_len:]
    result["alignment_response_mean_no_prefix"] = (
        alignment_response.mean().item() if alignment_response.numel() else None
    )

    # Step 5 (optional): prompt optimizer experiments
    if run_gepa or run_prompt_tuning:
        insertion_index = get_prefix_insertion_index(tokenizer, tokenize_fn)
        output_ids_target = get_output_ids(full_input_ids.cpu(), input_ids.cpu())

        prompt_opt_results: Dict[str, Any] = {
            "insertion_index": insertion_index,
            "prefix_len": prefix_len,
            "output_len": int(output_ids_target.size(1)),
        }

        if run_gepa:
            print("\n[PromptOpt] Running GEPA...")
            prompt_opt_results["gepa"] = run_gepa_experiment(
                model=model,
                tokenizer=tokenizer,
                tokenize_fn=tokenize_fn,
                instruction=instruction,
                output_ids_target=output_ids_target,
                steered_acts_full=steered_acts_full,
                inversion_layer=inversion_layer,
                insertion_index=insertion_index,
                prefix_len=prefix_len,
                full_seq_len=full_seq_len,
                max_new_tokens=max_new_tokens,
                gepa_budget=gepa_budget,
                gepa_max_steps=gepa_max_steps,
                metric=alignment_metric,
                gepa_reflection_lm_provider=gepa_reflection_lm_provider,
                gepa_reflection_model=gepa_reflection_model,
                gepa_reflection_max_tokens=gepa_reflection_max_tokens,
                gepa_log_candidates=gepa_log_candidates,
            )

        if run_prompt_tuning:
            print("\n[PromptOpt] Running PEFT prompt-tuning...")
            prompt_opt_results["prompt_tuning"] = run_prompt_tuning_experiment(
                model=model,
                tokenizer=tokenizer,
                tokenize_fn=tokenize_fn,
                instruction=instruction,
                output_ids_target=output_ids_target,
                steered_acts_full=steered_acts_full,
                inversion_layer=inversion_layer,
                insertion_index=insertion_index,
                prefix_len=prefix_len,
                full_seq_len=full_seq_len,
                max_new_tokens=max_new_tokens,
                pt_steps=pt_steps,
                pt_lr=pt_lr,
                pt_virtual_tokens=pt_virtual_tokens,
                metric=alignment_metric,
            )

        # Ensure JSON serialization doesn't choke on tensors
        result["prompt_opt_results"] = _strip_tensors_from_prompt_opt(
            prompt_opt_results
        )

    # Store activations for later analysis (kept out of json)
    result["_activations_store"] = {
        "steered_acts_full": steered_acts_full,
        "natural_acts_full": natural_acts_full,
        "alignment_per_token": alignment_per_token,
    }

    return result


def save_results(
    results: List[Dict[str, Any]],
    activations_data: Dict[str, Any],
    output_dir: str,
    config: Config,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    results_for_json: List[Dict[str, Any]] = []
    for r in results:
        r_copy = r.copy()
        r_copy.pop("_activations_store", None)
        results_for_json.append(r_copy)

    results_path = os.path.join(
        output_dir,
        f"prompt_opt_results_{config.steering_type}_{config.steering_method}_coeff_{config.steering_coeff}.json",
    )
    with open(results_path, "w") as f:
        json.dump(results_for_json, f, indent=2, default=str)

    activations_path = os.path.join(
        output_dir,
        f"prompt_opt_activations_{config.steering_type}_{config.steering_method}_coeff_{config.steering_coeff}.pkl",
    )
    with open(activations_path, "wb") as f:
        pickle.dump(activations_data, f)

    print(f"Results saved to: {results_path}")
    print(f"Activations saved to: {activations_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompt optimizer activation alignment experiment"
    )
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="NF4 BitsAndBytes load (GPU + bitsandbytes).",
    )

    parser.add_argument("--steering-type", type=str, default="refusal")
    parser.add_argument("--method", type=str, default="actadd")
    parser.add_argument("--coeff", type=float, default=-1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)

    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Comma-separated list of instructions. Default: TEST_INSTRUCTIONS.",
    )

    parser.add_argument("--prompt-opt", action="store_true", default=True)
    parser.add_argument("--no-gepa", action="store_true", default=False)
    parser.add_argument("--no-pt", action="store_true", default=False)

    parser.add_argument("--prefix-len", type=int, default=10)
    parser.add_argument("--gepa-budget", type=str, default="medium", choices=["light", "medium", "heavy"])
    parser.add_argument(
        "--gepa-max-steps",
        type=int,
        default=20,
        help="Explicit GEPA step budget. When set, overrides --gepa-budget auto mode.",
    )
    parser.add_argument(
        "--gepa-reflection-lm-provider",
        type=str,
        default="openai",
        choices=["inproc", "openai"],
        help="Provider for GEPA reflection LM. Uses OPENAI_API_KEY when provider=openai.",
    )
    parser.add_argument(
        "--gepa-reflection-model",
        type=str,
        default="gpt-5.4-mini",
        help="Reflection model name when using provider=openai.",
    )
    parser.add_argument(
        "--gepa-reflection-max-tokens",
        type=int,
        default=8192,
        help="Max completion tokens for reflection LM.",
    )
    parser.add_argument(
        "--gepa-log-candidates",
        action="store_true",
        default=True,
        help="Print each GEPA candidate prompt/instruction and model output during optimization.",
    )

    parser.add_argument("--pt-steps", type=int, default=200)
    parser.add_argument("--pt-lr", type=float, default=3e-2)
    parser.add_argument("--pt-virtual-tokens", type=int, default=10)

    parser.add_argument(
        "--alignment-metric",
        type=str,
        default="l2",
        choices=["l2", "cosine"],
    )

    args = parser.parse_args()

    cfg = Config()
    cfg.model_id = args.model
    cfg.device = args.device
    cfg.dtype = args.dtype
    cfg.load_in_4bit = args.load_in_4bit
    cfg.steering_type = args.steering_type
    cfg.steering_method = args.method
    cfg.steering_coeff = args.coeff
    cfg.max_new_tokens = args.max_new_tokens

    run_gepa = args.prompt_opt and not args.no_gepa
    run_prompt_tuning = args.prompt_opt and not args.no_pt

    instructions = load_instructions_from_arg(args.instructions)
    set_seed(cfg.seed)

    model_alias = cfg.model_id.split("/")[-1]
    output_dir = os.path.join(cfg.output_dir, model_alias, "prompt_opt")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Writing output under: {output_dir}")

    print(f"Loading model: {cfg.model_id}")
    model, tokenizer = load_model(
        cfg.model_id, cfg.device, cfg.dtype, load_in_4bit=cfg.load_in_4bit
    )
    tokenizer.pad_token = tokenizer.eos_token

    tokenize_fn = get_tokenize_fn(
        tokenizer,
        use_chat_template=cfg.use_chat_template,
        add_special_tokens=cfg.add_special_tokens,
    )

    print("\nLoading steering direction...")
    direction_path = cfg.get_direction_path()
    if not os.path.exists(direction_path):
        try:
            direction_path = cfg.get_direction_path(model_alias.lower())
        except FileNotFoundError:
            raise FileNotFoundError(f"Direction file not found: {direction_path}")
    direction, layer, metadata = load_steering_direction(direction_path, cfg.device)
    print(f"Loaded direction from layer {layer}")

    steering_config = SteeringConfig(
        direction=direction,
        layer=layer,
        method=cfg.steering_method,
        coeff=cfg.steering_coeff,
        steering_type=cfg.steering_type,
    )

    results: List[Dict[str, Any]] = []
    activations_data: Dict[str, Any] = {
        "steering_coeff": cfg.steering_coeff,
        "instructions": [],
        "results": [],
    }

    for instruction in instructions:
        print(f"\n====================\nInstruction: {instruction[:80]}...\n====================")
        r = run_single_instruction(
            model=model,
            tokenizer=tokenizer,
            tokenize_fn=tokenize_fn,
            instruction=instruction,
            steering_config=steering_config,
            max_new_tokens=cfg.max_new_tokens,
            run_gepa=run_gepa,
            run_prompt_tuning=run_prompt_tuning,
            prefix_len=args.prefix_len,
            gepa_budget=args.gepa_budget,
            gepa_max_steps=args.gepa_max_steps,
            gepa_reflection_lm_provider=args.gepa_reflection_lm_provider,
            gepa_reflection_model=args.gepa_reflection_model,
            gepa_reflection_max_tokens=args.gepa_reflection_max_tokens,
            gepa_log_candidates=args.gepa_log_candidates,
            pt_steps=args.pt_steps,
            pt_lr=args.pt_lr,
            pt_virtual_tokens=args.pt_virtual_tokens,
            alignment_metric=args.alignment_metric,
        )
        results.append(r)

        acts_store = r.get("_activations_store", {})
        activations_data["instructions"].append(instruction)
        activations_data["results"].append(
            {
                "steered_acts_full": acts_store.get("steered_acts_full"),
                "natural_acts_full": acts_store.get("natural_acts_full"),
                "alignment_per_token": acts_store.get("alignment_per_token"),
            }
        )

    # Clean in case tensors accidentally remain in json results
    save_results(results, activations_data, output_dir, cfg)


if __name__ == "__main__":
    main()


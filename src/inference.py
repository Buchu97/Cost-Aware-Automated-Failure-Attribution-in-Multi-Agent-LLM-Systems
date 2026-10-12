"""Local LLM inference engine with logprob extraction.

Runs a HuggingFace model on a single prompt N times, collecting:
  - Generated text
  - Token counts (prompt + completion)
  - Per-token log-probabilities
  - Wall-clock time
"""

from __future__ import annotations

import math
import time
import json
from pathlib import Path
from typing import Optional

from src.response_parser import parse_attribution_response
from src.run_records import save_json
from src.h100_attention import h100_attention


def extract_logprobs(
    scores: tuple[torch.Tensor, ...],
    generated_ids: torch.Tensor,
) -> list[float]:
    """Extract the log-probability of each generated token from model scores.

    Args:
        scores: Tuple of (vocab_size,) tensors, one per generated step.
        generated_ids: 1-D tensor of token IDs that were actually generated.

    Returns:
        List of float log-probabilities, one per generated token.
    """
    logprobs: list[float] = []
    import torch
    for step_idx, score in enumerate(scores):
        # score shape: (1, vocab_size) or (vocab_size,)
        s = score[0] if score.dim() == 2 else score
        log_softmax = torch.log_softmax(s.float(), dim=-1)
        token_id = generated_ids[step_idx].item()
        logprobs.append(log_softmax[token_id].item())
    return logprobs


def run_single_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    seed: int | None = None,
    context_window: int = 131072,
) -> dict:
    """Run a single local inference call and return structured results.

    Returns:
        Dict with keys: generated_text, prompt_tokens, completion_tokens,
        total_tokens, logprobs, mean_logprob, logprob_confidence,
        elapsed_seconds, predicted_agent, predicted_step,
        self_reported_confidence, reason, parse_success.
    """
    import torch
    from transformers import set_seed
    import config
    if seed is not None:
        set_seed(seed)
    # Tokenise using the chat template
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
    prompt_tokens = inputs["input_ids"].shape[1]
    if prompt_tokens + max_new_tokens > context_window:
        raise ValueError(f"Full trace requires {prompt_tokens} input tokens plus {max_new_tokens} output tokens; context is {context_window}. No truncation performed.")
    inputs = inputs.to(model.device)

    torch.cuda.synchronize(0)
    start = time.perf_counter()
    prefill_chunk_size = 1024 if prompt_tokens > 32768 else None
    if prefill_chunk_size:
        print(f'[prefill] {prompt_tokens} input tokens in {prefill_chunk_size}-token chunks; '
              'the complete trace is retained.', flush=True)
    with torch.inference_mode(), h100_attention():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=config.TOP_P,
            top_k=config.TOP_K,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            prefill_chunk_size=prefill_chunk_size,
        )
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - start

    # Decode only the newly generated tokens
    generated_ids = outputs.sequences[0][prompt_tokens:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    completion_tokens = len(generated_ids)

    # Log-probabilities
    logprobs = extract_logprobs(outputs.scores, generated_ids)
    mean_lp = sum(logprobs) / len(logprobs) if logprobs else 0.0

    # Parse structured fields
    parsed = parse_attribution_response(generated_text)

    return {
        "generated_text": generated_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "logprobs": logprobs,
        "mean_logprob": mean_lp,
        "logprob_confidence": math.exp(mean_lp),
        "elapsed_seconds": round(elapsed, 3),
        "seed": seed,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "prefill_chunk_size": prefill_chunk_size,
        "attention_policy": "fused_prompt_math_single_token_decode",
        # Parsed fields
        "predicted_agent": parsed["agent_name"],
        "predicted_step": parsed["step_number"],
        "self_reported_confidence": parsed["confidence"],
        "reason": parsed["reason"],
        "parse_success": parsed["agent_name"] is not None
                         and parsed["step_number"] is not None,
    }


def run_n_times(
    model,
    tokenizer,
    prompt: str,
    n_runs: int = 3,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> list[dict]:
    """Run inference N times for a single trace and return list of results."""
    results: list[dict] = []
    for run_idx in range(n_runs):
        result = run_single_inference(
            model, tokenizer, prompt, max_new_tokens, temperature
        )
        result["run_index"] = run_idx
        results.append(result)
    return results


def save_run_result(
    result: dict,
    output_dir: Path,
    model_alias: str,
    trace_id: str,
    run_index: int,
) -> Path:
    """Save a single run result to a JSON file.

    Path: output_dir / model_alias / {trace_id}_run{run_index}.json
    """
    model_dir = output_dir / model_alias
    model_dir.mkdir(parents=True, exist_ok=True)

    # Strip logprobs list to save space (keep summary stats only)
    save_data = {k: v for k, v in result.items() if k != "logprobs"}
    save_data["logprob_count"] = len(result.get("logprobs", []))

    stem = trace_id.replace(".json", "")
    out_path = model_dir / f"{stem}_run{run_index}.json"
    save_json(out_path, save_data)
    return out_path

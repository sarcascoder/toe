"""Speculative decoding helper for the *text* generation path (MLX-LM).

Context (frontier-report systems section): on Apple Silicon, token generation
is memory-bandwidth-bound, not compute-bound. A small draft model proposes k
tokens; the target verifies all k in ONE forward pass. Because verifying a
batch costs barely more bandwidth than a single token, you get a real 1.5-2.5x
speedup with *identical* output distribution (it's exact, not lossy) -- as long
as the draft and target share a tokenizer/vocab.

SCOPE NOTE: this helps pure-text LLM decoding (e.g. post-processing extracted
text, summarization, RAG over extracted docs). It does NOT directly accelerate
the *vision* forward pass of a VLM -- mlx-vlm doesn't expose draft-model
verification for the multimodal path yet. Use this for the text stages of your
pipeline.

Requires: pip install mlx-lm . Untested on non-Apple hardware (mlx is arm64
macOS only); the logic mirrors mlx_lm's documented draft_model API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


# Good draft/target pairs must share a tokenizer family.
SUGGESTED_PAIRS = {
    "qwen2.5-7b": {
        "target": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "draft": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    },
    "llama3.1-8b": {
        "target": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "draft": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    },
}


@dataclass
class SpecResult:
    text: str
    seconds: float
    gen_tokens: int | None
    tok_per_s: float | None
    used_spec: bool


def generate_with_spec(
    target_repo: str,
    prompt: str,
    draft_repo: str | None = None,
    num_draft_tokens: int = 4,
    max_tokens: int = 512,
):
    """Generate text from `target_repo`, optionally accelerated by `draft_repo`.

    If draft_repo is None, runs a normal (non-speculative) generation so you can
    A/B the speedup with the same code path.
    """
    try:
        from mlx_lm import load, generate  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "mlx-lm not installed (Apple Silicon only):  pip install -U mlx-lm\n"
            f"{e}"
        )

    model, tokenizer = load(target_repo)
    draft_model = None
    if draft_repo:
        draft_model, _ = load(draft_repo)  # tokenizer must match target's vocab

    kwargs = dict(max_tokens=max_tokens, verbose=False)
    if draft_model is not None:
        # mlx-lm exposes speculative decoding via draft_model + num_draft_tokens
        kwargs["draft_model"] = draft_model
        kwargs["num_draft_tokens"] = num_draft_tokens

    t0 = time.perf_counter()
    out = generate(model, tokenizer, prompt=prompt, **kwargs)
    dt = time.perf_counter() - t0

    text = out if isinstance(out, str) else getattr(out, "text", str(out))
    gen_tokens = None
    try:
        gen_tokens = len(tokenizer.encode(text))
    except Exception:
        pass
    tps = gen_tokens / dt if (gen_tokens and dt) else None
    return SpecResult(text=text, seconds=round(dt, 3), gen_tokens=gen_tokens,
                      tok_per_s=round(tps, 1) if tps else None,
                      used_spec=draft_model is not None)


def ab_compare(pair_key: str, prompt: str, num_draft_tokens: int = 4,
               max_tokens: int = 512):
    """Run the same prompt with and without the draft model; print the speedup."""
    if pair_key not in SUGGESTED_PAIRS:
        raise SystemExit(f"Unknown pair '{pair_key}'. Known: "
                         f"{', '.join(SUGGESTED_PAIRS)}")
    pair = SUGGESTED_PAIRS[pair_key]
    baseline = generate_with_spec(pair["target"], prompt, draft_repo=None,
                                  max_tokens=max_tokens)
    sped = generate_with_spec(pair["target"], prompt, draft_repo=pair["draft"],
                              num_draft_tokens=num_draft_tokens,
                              max_tokens=max_tokens)
    speedup = (sped.tok_per_s / baseline.tok_per_s
               if (sped.tok_per_s and baseline.tok_per_s) else None)
    print(f"baseline : {baseline.tok_per_s} tok/s  ({baseline.seconds}s)")
    print(f"spec(k={num_draft_tokens}): {sped.tok_per_s} tok/s  ({sped.seconds}s)")
    if speedup:
        print(f"speedup  : {speedup:.2f}x")
    return baseline, sped

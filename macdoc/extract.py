"""Run a VLM over rendered pages and return text/markdown.

Wraps mlx_vlm.generate. The mlx-vlm generate() signature has shifted across
releases; we normalize the return value (some versions return a string, newer
ones return a GenerationResult with .text and timing fields) and surface tokens
+ timing so the benchmark harness can reuse it.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PageResult:
    page_index: int
    text: str
    prompt_tokens: int | None = None
    generation_tokens: int | None = None
    seconds: float | None = None

    @property
    def tok_per_s(self) -> float | None:
        if self.generation_tokens and self.seconds:
            return self.generation_tokens / self.seconds
        return None


def _apply_template(processor, model, prompt: str, n_images: int = 1) -> str:
    """Build a chat-formatted prompt with the right number of image slots."""
    try:
        from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore
    except ImportError:  # pragma: no cover
        return prompt
    # config carries the chat template / image-token wiring
    config = getattr(model, "config", None)
    try:
        return apply_chat_template(processor, config, prompt, num_images=n_images)
    except TypeError:
        # older signature without num_images
        return apply_chat_template(processor, config, prompt)


def _normalize_generate_return(ret):
    """Return (text, prompt_tokens, gen_tokens, seconds) across mlx-vlm versions."""
    if isinstance(ret, str):
        return ret, None, None, None
    if isinstance(ret, tuple):  # very old: (text, stats)
        text = ret[0]
        return text, None, None, None
    # newer: GenerationResult-like object
    text = getattr(ret, "text", None) or str(ret)
    pt = getattr(ret, "prompt_tokens", None)
    gt = getattr(ret, "generation_tokens", None)
    # timing may be tps or total time depending on version
    secs = None
    gtps = getattr(ret, "generation_tps", None)
    if gt and gtps:
        try:
            secs = gt / gtps
        except ZeroDivisionError:
            secs = None
    return text, pt, gt, secs


def extract_pages(
    model,
    processor,
    page_image_paths,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    verbose: bool = False,
) -> list[PageResult]:
    """Run the model on each page image independently."""
    import time

    from mlx_vlm import generate  # type: ignore

    results: list[PageResult] = []
    for idx, img_path in enumerate(page_image_paths):
        formatted = _apply_template(processor, model, prompt, n_images=1)
        t0 = time.perf_counter()
        ret = generate(
            model,
            processor,
            formatted,
            image=[str(img_path)],
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=verbose,
        )
        dt = time.perf_counter() - t0
        text, pt, gt, secs = _normalize_generate_return(ret)
        results.append(
            PageResult(
                page_index=idx,
                text=text.strip(),
                prompt_tokens=pt,
                generation_tokens=gt,
                seconds=secs if secs is not None else dt,
            )
        )
    return results


def extract_document(
    input_path,
    model_arg: str,
    dpi: int = 200,
    max_pages: int | None = None,
    prompt: str | None = None,
    max_tokens: int = 4096,
    verbose: bool = False,
) -> tuple[list[PageResult], str]:
    """High-level: render -> load model -> extract. Returns (results, joined_md)."""
    from . import models, render

    repo, spec = models.resolve_repo(model_arg)
    use_prompt = prompt or (spec.default_prompt if spec else
                            "Transcribe this document image to clean Markdown.")

    pages = render.load_pages(input_path, dpi=dpi, max_pages=max_pages)
    with tempfile.TemporaryDirectory() as tmp:
        page_paths = render.save_pages_as_pngs(pages, tmp)
        model, processor = models.load_model(repo)
        results = extract_pages(
            model, processor, page_paths, use_prompt,
            max_tokens=max_tokens, verbose=verbose,
        )

    joined = "\n\n---\n\n".join(
        f"<!-- page {r.page_index + 1} -->\n{r.text}" for r in results
    )
    return results, joined

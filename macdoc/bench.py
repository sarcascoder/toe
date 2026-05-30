"""Benchmark harness: tok/s + peak RAM for one or more models on the same doc.

This is the experiment from the frontier report: feel the
optical-compression-vs-specialist tradeoff on YOUR hardware. It reports, per
model: generation tok/s, wall time, peak process RSS, and MLX peak GPU memory
(if exposed by your mlx build).
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, asdict


@dataclass
class BenchRow:
    model: str
    repo: str
    pages: int
    total_gen_tokens: int | None
    mean_tok_per_s: float | None
    wall_seconds: float
    peak_rss_gb: float | None
    mlx_peak_gb: float | None


def _peak_rss_gb() -> float | None:
    try:
        import resource
        # ru_maxrss is bytes on macOS, kilobytes on Linux
        import sys
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return maxrss / (1024 ** 3)
        return maxrss / (1024 ** 2)
    except Exception:
        return None


def _mlx_peak_gb() -> float | None:
    try:
        import mlx.core as mx  # type: ignore
        # API has moved between mx.metal.get_peak_memory and mx.get_peak_memory
        for getter in (
            getattr(getattr(mx, "metal", None), "get_peak_memory", None),
            getattr(mx, "get_peak_memory", None),
        ):
            if callable(getter):
                return getter() / (1024 ** 3)
    except Exception:
        return None
    return None


def _reset_mlx_peak():
    try:
        import mlx.core as mx  # type: ignore
        for resetter in (
            getattr(getattr(mx, "metal", None), "reset_peak_memory", None),
            getattr(mx, "reset_peak_memory", None),
        ):
            if callable(resetter):
                resetter()
                return
    except Exception:
        pass


def bench_models(input_path, model_keys, dpi=200, max_pages=2, max_tokens=2048):
    """Run each model over the same pages; return list[BenchRow]."""
    from . import models, render, extract

    pages = render.load_pages(input_path, dpi=dpi, max_pages=max_pages)
    import tempfile
    rows: list[BenchRow] = []

    with tempfile.TemporaryDirectory() as tmp:
        page_paths = render.save_pages_as_pngs(pages, tmp)
        for key in model_keys:
            repo, spec = models.resolve_repo(key)
            prompt = spec.default_prompt if spec else \
                "Transcribe this document image to clean Markdown."

            _reset_mlx_peak()
            gc.collect()
            t0 = time.perf_counter()
            model, processor = models.load_model(repo)
            results = extract.extract_pages(
                model, processor, page_paths, prompt, max_tokens=max_tokens
            )
            wall = time.perf_counter() - t0

            gen_tokens = [r.generation_tokens for r in results
                          if r.generation_tokens]
            tps = [r.tok_per_s for r in results if r.tok_per_s]
            rows.append(BenchRow(
                model=key,
                repo=repo,
                pages=len(results),
                total_gen_tokens=sum(gen_tokens) if gen_tokens else None,
                mean_tok_per_s=(sum(tps) / len(tps)) if tps else None,
                wall_seconds=round(wall, 2),
                peak_rss_gb=round(_peak_rss_gb() or 0, 2) or None,
                mlx_peak_gb=round(_mlx_peak_gb() or 0, 2) or None,
            ))

            # free before next model so peak-RAM numbers are per-model-ish
            del model, processor
            gc.collect()

    return rows


def format_table(rows) -> str:
    headers = ["model", "tok/s", "gen_tok", "wall_s", "peak_rss_gb", "mlx_peak_gb"]
    lines = [" | ".join(headers), "-|-".join("-" * len(h) for h in headers)]
    for r in rows:
        d = asdict(r)
        lines.append(" | ".join(str(x) for x in [
            d["model"],
            f'{d["mean_tok_per_s"]:.1f}' if d["mean_tok_per_s"] else "n/a",
            d["total_gen_tokens"] or "n/a",
            d["wall_seconds"],
            d["peak_rss_gb"] or "n/a",
            d["mlx_peak_gb"] or "n/a",
        ]))
    return "\n".join(lines)

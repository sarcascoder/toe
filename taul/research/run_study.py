"""Run the PORE study: generate the suite, evaluate models, produce figures.

Two prediction sources:
  * Mock models (no GPU/model needed) -- simulate canonical failure modes so the
    benchmark + metric can be validated end-to-end and the paper's figures
    reproduced deterministically:
        - column_aware : perfect transcription, valid reading order.
        - raster       : perfect transcription, NAIVE top-to-bottom/left-to-right
                         order (ignores columns) -> the classic VLM failure.
        - noisy_reader : valid order, but per-block character corruption.
  * Real VLM via mlx-vlm (`--model <key>`): runs on your Mac, parses the
    model's Markdown output into blocks (in output order), matches to GT.

Outputs: results.csv, results.md, fig_complexity.png, fig_orthogonality.png.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from . import synth
from .metric import evaluate, PoreReport


# ---------------- mock models ----------------

def _reading_order_ids(doc) -> list:
    """A *valid* reading order = blocks sorted by region then vertical position,
    consistent with the partial order (used by column_aware / noisy_reader)."""
    # region priority: left/story0/body first, then right, then sidebar
    region_rank = {}
    for i, b in enumerate(doc.blocks):
        r = b.region
        rank = (0 if r in ("left", "body", "story0") else
                1 if r in ("right", "story1") else
                2 if r in ("sidebar", "story2") else 1)
        region_rank[b.id] = (rank, b.bbox[1], b.bbox[0])
    return sorted((b.id for b in doc.blocks), key=lambda i: region_rank[i])


def _raster_order_ids(doc) -> list:
    """Naive raster scan: pure top-to-bottom, then left-to-right. Ignores
    column structure -> interleaves columns (the failure we want to expose)."""
    return sorted((b.id for b in doc.blocks),
                  key=lambda i: (doc.blocks[i].bbox[1] // 60, doc.blocks[i].bbox[0]))


def _corrupt(text: str, rng: random.Random, rate=0.08) -> str:
    chars = list(text)
    for k in range(len(chars)):
        if rng.random() < rate and chars[k].isalpha():
            chars[k] = rng.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)


def mock_predict(doc, model: str, seed: int = 0):
    """Return predicted block texts IN PREDICTED ORDER for a mock model."""
    rng = random.Random(seed)
    by_id = {b.id: b.text for b in doc.blocks}
    if model == "column_aware":
        order = _reading_order_ids(doc)
        return [by_id[i] for i in order]
    if model == "raster":
        order = _raster_order_ids(doc)
        return [by_id[i] for i in order]
    if model == "noisy_reader":
        order = _reading_order_ids(doc)
        return [_corrupt(by_id[i], rng) for i in order]
    raise KeyError(f"unknown mock model '{model}'")


MOCK_MODELS = ["column_aware", "raster", "noisy_reader"]


# ---------------- real VLM path ----------------

def vlm_predict(doc, png_path, model_key: str, dpi_note=None):
    """Run a real mlx-vlm model on the rendered page, return predicted block
    texts in output order. Imported lazily; Apple Silicon only."""
    from ..models import resolve_repo, load_model
    from ..extract import extract_pages
    repo, spec = resolve_repo(model_key)
    model, processor = load_model(repo)
    prompt = (spec.default_prompt if spec else
              "Transcribe this document to Markdown, preserving reading order.")
    results = extract_pages(model, processor, [str(png_path)], prompt,
                            max_tokens=2048)
    text = results[0].text
    # split into blocks the same way the metric expects
    import re
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    parts = re.split(r"\n\s*\n", text) if "\n\n" in text else text.split("\n")
    return [p.strip() for p in parts if p.strip()]


# ---------------- study ----------------

@dataclass
class Row:
    model: str
    layout: str
    complexity: int
    doc_id: str
    transcription_error: float
    ordering_violation: float
    coverage: float
    spurious_rate: float


def run(out_dir, per_layout=3, models=None, real_model=None, render=True):
    out_dir = Path(out_dir)
    items = synth.build_suite(out_dir / "suite", per_layout=per_layout,
                              render=render)
    models = models or MOCK_MODELS
    rows: list[Row] = []

    for doc, png in items:
        gt = doc.gt_texts()
        # mock models
        for m in models:
            pred = mock_predict(doc, m, seed=hash(doc.doc_id) % 10000)
            rep: PoreReport = evaluate(gt, doc.partial_order, pred)
            rows.append(Row(m, doc.layout, doc.complexity, doc.doc_id,
                            rep.transcription_error, rep.ordering_violation,
                            rep.coverage, rep.spurious_rate))
        # optional real VLM
        if real_model and png is not None:
            pred = vlm_predict(doc, png, real_model)
            rep = evaluate(gt, doc.partial_order, pred)
            rows.append(Row(real_model, doc.layout, doc.complexity, doc.doc_id,
                            rep.transcription_error, rep.ordering_violation,
                            rep.coverage, rep.spurious_rate))

    _write_csv(out_dir / "results.csv", rows)
    summary = _aggregate_md(rows)
    (out_dir / "results.md").write_text(summary)
    try:
        _plots(out_dir, rows)
    except Exception as e:  # plotting optional
        print(f"[plots skipped: {e}]")
    return rows, summary


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "layout", "complexity", "doc_id",
                    "transcription_error", "ordering_violation",
                    "coverage", "spurious_rate"])
        for r in rows:
            w.writerow([r.model, r.layout, r.complexity, r.doc_id,
                        r.transcription_error, r.ordering_violation,
                        r.coverage, r.spurious_rate])


def _aggregate_md(rows) -> str:
    # mean by (model, layout)
    from collections import defaultdict
    agg = defaultdict(lambda: [0.0, 0.0, 0])
    for r in rows:
        a = agg[(r.model, r.layout, r.complexity)]
        a[0] += r.transcription_error
        a[1] += r.ordering_violation
        a[2] += 1
    lines = ["# PORE study results (mean per model x layout)\n",
             "| model | layout | cx | transcription_err | ordering_violation |",
             "|---|---|---|---|---|"]
    for (model, layout, cx), (te, ov, n) in sorted(agg.items()):
        lines.append(f"| {model} | {layout} | {cx} | {te / n:.3f} | {ov / n:.3f} |")
    # headline: per-model mean ordering violation by complexity
    lines.append("\n## Headline: ordering violation rises with layout complexity")
    by_mc = defaultdict(lambda: [0.0, 0])
    for r in rows:
        a = by_mc[(r.model, r.complexity)]
        a[0] += r.ordering_violation; a[1] += 1
    lines.append("\n| model | complexity | mean ordering_violation |")
    lines.append("|---|---|---|")
    for (model, cx), (ov, n) in sorted(by_mc.items()):
        lines.append(f"| {model} | {cx} | {ov / n:.3f} |")
    return "\n".join(lines) + "\n"


def _plots(out_dir, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    models = sorted({r.model for r in rows})

    # Fig 1: ordering violation vs complexity
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in models:
        by_cx = defaultdict(list)
        for r in rows:
            if r.model == m:
                by_cx[r.complexity].append(r.ordering_violation)
        xs = sorted(by_cx)
        ys = [sum(by_cx[x]) / len(by_cx[x]) for x in xs]
        ax.plot(xs, ys, marker="o", label=m)
    ax.set_xlabel("layout complexity")
    ax.set_ylabel("mean ordering violation rate")
    ax.set_title("Reading-order failure rises with layout complexity")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "fig_complexity.png", dpi=130)
    plt.close(fig)

    # Fig 2: orthogonality scatter (transcription vs ordering)
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {m: c for m, c in zip(models, ["#2563eb", "#dc2626", "#16a34a",
                                            "#9333ea", "#ea580c"])}
    for m in models:
        xs = [r.transcription_error for r in rows if r.model == m]
        ys = [r.ordering_violation for r in rows if r.model == m]
        ax.scatter(xs, ys, label=m, alpha=0.7, c=colors.get(m))
    ax.set_xlabel("transcription error (order-invariant)")
    ax.set_ylabel("ordering violation (transcription-invariant)")
    ax.set_title("Two orthogonal error axes current metrics conflate")
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(-0.02, 1.0)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "fig_orthogonality.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run the PORE study")
    ap.add_argument("--out", default="pore_study")
    ap.add_argument("--per-layout", type=int, default=3)
    ap.add_argument("--no-render", action="store_true",
                    help="skip PNG rendering (metric-only, faster)")
    ap.add_argument("--real-model", default=None,
                    help="also eval a real mlx-vlm model (registry key)")
    args = ap.parse_args()
    rows, summary = run(args.out, per_layout=args.per_layout,
                        real_model=args.real_model, render=not args.no_render)
    print(summary)
    print(f"\nwrote results + figures to {args.out}/")

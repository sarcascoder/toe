"""Matcher-robustness stress study (paper Section 6.1).

THREAT TO VALIDITY: PORE's orthogonal decomposition assumes blocks are matched.
If transcription degrades enough that a predicted block no longer matches its
ground-truth block, two things happen: (a) coverage falls, and (b) the ordering
score, computed over *matched* blocks, can become unreliable because the wrong
blocks get aligned. We must quantify the operating regime where the
decomposition is trustworthy.

Two experiments:
  E1 corruption sweep: take a KNOWN-VALID reading order, corrupt every block's
     text at rate r in [0, r_max]. Track coverage(r), ordering_violation(r),
     transcription_error(r). Within the safe regime, ordering_violation must
     stay ~0 (order was never changed); its rise marks where matching breaks.
  E2 confound check: take a KNOWN-WRONG order (raster), add the same corruption.
     The *true* ordering violation is fixed by the layout; check the measured
     value stays stable across corruption until matching collapses (i.e.
     transcription noise does not mask a real ordering error in the safe regime).

Reports the corruption level r* (and corresponding transcription_error) at which
coverage first drops below `coverage_floor` -- the recommended validity bound.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import synth
from .metric import evaluate
from .run_study import _reading_order_ids, _raster_order_ids, _corrupt


@dataclass
class SweepPoint:
    rate: float
    coverage: float
    ordering_violation: float
    transcription_error: float
    n: int


def _corrupt_at(texts, rate, rng):
    return [_corrupt(t, rng, rate=rate) for t in texts]


def corruption_sweep(layouts=None, per_layout=8, rates=None,
                     order="valid", seed0=0):
    """E1/E2. order in {'valid','raster'}. Returns list[SweepPoint] (averaged)."""
    layouts = layouts or list(synth.LAYOUTS)
    rates = rates or [round(0.05 * i, 2) for i in range(0, 13)]  # 0.0 .. 0.60
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # rate -> [cov, ov, te, n]

    s = seed0
    for layout in layouts:
        for _ in range(per_layout):
            doc = synth.make_doc(layout, s); s += 1
            gt = doc.gt_texts()
            by_id = {b.id: b.text for b in doc.blocks}
            ids = (_reading_order_ids(doc) if order == "valid"
                   else _raster_order_ids(doc))
            base_pred = [by_id[i] for i in ids]
            for r in rates:
                rng = random.Random(hash((doc.doc_id, r)) % (2**31))
                pred = _corrupt_at(base_pred, r, rng)
                rep = evaluate(gt, doc.partial_order, pred)
                a = acc[r]
                a[0] += rep.coverage
                a[1] += rep.ordering_violation
                a[2] += rep.transcription_error
                a[3] += 1

    points = []
    for r in sorted(acc):
        cov, ov, te, n = acc[r]
        points.append(SweepPoint(r, cov / n, ov / n, te / n, n))
    return points


def find_validity_bound(points, coverage_floor=0.9):
    """First corruption rate where mean coverage drops below the floor."""
    for p in points:
        if p.coverage < coverage_floor:
            return p
    return None


def threshold_sweep(layouts=None, per_layout=8, thresholds=None, rate=0.20,
                    seed0=100):
    """How match threshold tau trades coverage vs spurious-matching at a fixed
    corruption rate. Returns list[(tau, coverage, ordering_violation)]."""
    layouts = layouts or list(synth.LAYOUTS)
    thresholds = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    out = []
    for tau in thresholds:
        cov = ov = 0.0; n = 0
        s = seed0
        for layout in layouts:
            for _ in range(per_layout):
                doc = synth.make_doc(layout, s); s += 1
                gt = doc.gt_texts()
                by_id = {b.id: b.text for b in doc.blocks}
                ids = _reading_order_ids(doc)
                rng = random.Random(s)
                pred = _corrupt_at([by_id[i] for i in ids], rate, rng)
                rep = evaluate(gt, doc.partial_order, pred, threshold=tau)
                cov += rep.coverage; ov += rep.ordering_violation; n += 1
        out.append((tau, cov / n, ov / n))
    return out


def run(out_dir="pore_robustness", per_layout=8):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    valid = corruption_sweep(per_layout=per_layout, order="valid")
    raster = corruption_sweep(per_layout=per_layout, order="raster")
    taus = threshold_sweep(per_layout=per_layout)
    bound = find_validity_bound(valid, coverage_floor=0.9)

    # text report
    lines = ["# PORE matcher-robustness (Section 6.1)\n",
             "## E1: valid order under corruption",
             "| corruption | coverage | ordering_violation | transcription_err |",
             "|---|---|---|---|"]
    for p in valid:
        lines.append(f"| {p.rate:.2f} | {p.coverage:.3f} | "
                     f"{p.ordering_violation:.3f} | {p.transcription_error:.3f} |")
    if bound:
        lines.append(f"\n**Validity bound (coverage>=0.90):** corruption < "
                     f"{bound.rate:.2f}, i.e. transcription_error up to "
                     f"~{bound.transcription_error:.2f}. Within this regime, the "
                     f"valid-order ordering_violation stays "
                     f"{valid[0].ordering_violation:.3f}->"
                     f"{[p for p in valid if p.rate<bound.rate][-1].ordering_violation:.3f}.")
    else:
        lines.append("\n**Validity bound:** coverage stayed >=0.90 across the "
                     "entire sweep.")
    lines.append("\n## E2: raster (wrong) order under corruption "
                 "(true violation should persist)")
    lines.append("| corruption | coverage | ordering_violation |")
    lines.append("|---|---|---|")
    for p in raster:
        lines.append(f"| {p.rate:.2f} | {p.coverage:.3f} | {p.ordering_violation:.3f} |")
    lines.append("\n## Threshold sweep (corruption=0.20)")
    lines.append("| tau | coverage | ordering_violation |")
    lines.append("|---|---|---|")
    for tau, cov, ov in taus:
        lines.append(f"| {tau:.1f} | {cov:.3f} | {ov:.3f} |")
    (out_dir / "robustness.md").write_text("\n".join(lines) + "\n")

    try:
        _plot(out_dir, valid, raster, bound)
    except Exception as e:
        print(f"[plot skipped: {e}]")
    return valid, raster, taus, bound


def _plot(out_dir, valid, raster, bound):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    rs = [p.rate for p in valid]
    ax.plot(rs, [p.coverage for p in valid], "o-", color="#2563eb",
            label="coverage (valid order)")
    ax.plot(rs, [p.ordering_violation for p in valid], "s--", color="#16a34a",
            label="ordering_violation (valid order, true=0)")
    ax.plot(rs, [p.ordering_violation for p in raster], "^--", color="#dc2626",
            label="ordering_violation (raster, true>0)")
    ax.plot(rs, [p.transcription_error for p in valid], "d:", color="#9333ea",
            label="transcription_error")
    if bound:
        ax.axvline(bound.rate, color="grey", ls=":")
        ax.text(bound.rate + 0.005, 0.5, f"validity bound\ncov<0.9 @ r={bound.rate:.2f}",
                fontsize=9, color="grey")
    ax.set_xlabel("per-character corruption rate")
    ax.set_ylabel("metric value")
    ax.set_title("PORE stays faithful until matching collapses")
    ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "fig_robustness.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="PORE matcher-robustness study")
    ap.add_argument("--out", default="pore_robustness")
    ap.add_argument("--per-layout", type=int, default=8)
    args = ap.parse_args()
    valid, raster, taus, bound = run(args.out, per_layout=args.per_layout)
    print(Path(args.out, "robustness.md").read_text())
    print(f"wrote results + fig to {args.out}/")

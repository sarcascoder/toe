"""PORE metric: orthogonal decomposition of document-parsing error.

Given a ground-truth document = (blocks with text, partial order P) and a model
prediction (a linear sequence of predicted text blocks), we compute three
independent components:

  transcription_error  -- order-INVARIANT. Mean normalized edit distance over
                          matched (gt_block, pred_block) pairs. Measures content
                          fidelity only.
  ordering_violation    -- transcription-INVARIANT. violation_rate of the
                          predicted block order against P (Section partial_order).
                          Measures reading order only, robust to ambiguity.
  detection: coverage   -- fraction of gt blocks matched.
             spurious   -- fraction of predicted blocks unmatched (hallucinated).

The headline contribution is ORTHOGONALITY: transcription_error is invariant to
reordering blocks, and ordering_violation is invariant to corrupting block text
(as long as matching still succeeds). Proven empirically in tests/.

Matching: greedy max-similarity assignment (difflib ratio) above a threshold.
This is the same matching family OmniDocBench uses; the novelty is the
ambiguity-robust ordering score and the clean decomposition, not the matcher.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher

from .partial_order import PartialOrder, violation_rate


# ---- text utils ----

def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _ned(a: str, b: str) -> float:
    """Normalized edit distance in [0,1]."""
    na, nb = _norm(a), _norm(b)
    denom = max(len(na), len(nb))
    if denom == 0:
        return 0.0
    return _lev(na, nb) / denom


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


# ---- matching ----

def match(gt_texts, pred_texts, threshold: float = 0.5):
    """Greedy best-match gt block -> pred block. Returns list of
    (gt_idx, pred_idx, sim) and the set of used pred indices."""
    used = set()
    pairs = []
    # match in descending best-similarity to reduce greedy mistakes
    candidates = []
    for gi, g in enumerate(gt_texts):
        for pi, p in enumerate(pred_texts):
            candidates.append((_sim(g, p), gi, pi))
    candidates.sort(reverse=True)
    matched_gt = set()
    for sim, gi, pi in candidates:
        if sim < threshold:
            break
        if gi in matched_gt or pi in used:
            continue
        matched_gt.add(gi)
        used.add(pi)
        pairs.append((gi, pi, round(sim, 3)))
    pairs.sort(key=lambda t: t[0])
    return pairs, used


@dataclass
class PoreReport:
    transcription_error: float   # 0 good .. 1 bad  (order-invariant)
    ordering_violation: float    # 0 good .. 1 bad  (transcription-invariant)
    order_consistency: float     # 1 - ordering_violation
    coverage: float
    spurious_rate: float
    n_gt: int
    n_pred: int
    n_matched: int
    n_required_pairs: int

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"transcription_error : {self.transcription_error:.3f}  (order-invariant)\n"
            f"ordering_violation  : {self.ordering_violation:.3f}  "
            f"(transcription-invariant; 0 = valid reading)\n"
            f"order_consistency   : {self.order_consistency:.3f}\n"
            f"coverage            : {self.coverage:.3f}  "
            f"({self.n_matched}/{self.n_gt} blocks)\n"
            f"spurious_rate       : {self.spurious_rate:.3f}\n"
            f"required precedences: {self.n_required_pairs}"
        )


def evaluate(gt_texts, P: PartialOrder, pred_texts, threshold: float = 0.5) -> PoreReport:
    """Score a prediction (list of predicted block texts, IN PREDICTED ORDER)
    against a ground-truth doc (gt block texts + partial order P over gt ids)."""
    pairs, used = match(gt_texts, pred_texts, threshold)

    # transcription: mean NED over matched pairs (order-invariant)
    if pairs:
        trans = sum(_ned(gt_texts[gi], pred_texts[pi]) for gi, pi, _ in pairs) / len(pairs)
    else:
        trans = 1.0

    # ordering: predicted reading order of matched GT block ids.
    # Sort matched gt ids by the position of their matched pred block.
    matched_by_pred_pos = sorted(pairs, key=lambda t: t[1])
    predicted_gt_order = [gi for (gi, _, _) in matched_by_pred_pos]
    ov = violation_rate(predicted_gt_order, P)

    coverage = len(pairs) / len(gt_texts) if gt_texts else 1.0
    spurious = (len(pred_texts) - len(used)) / len(pred_texts) if pred_texts else 0.0

    return PoreReport(
        transcription_error=round(trans, 4),
        ordering_violation=round(ov, 4),
        order_consistency=round(1.0 - ov, 4),
        coverage=round(coverage, 4),
        spurious_rate=round(spurious, 4),
        n_gt=len(gt_texts),
        n_pred=len(pred_texts),
        n_matched=len(pairs),
        n_required_pairs=P.num_required_pairs(),
    )

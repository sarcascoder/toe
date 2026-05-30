"""Reading-order + transcription evaluation for document extraction.

Why this exists (frontier-report opportunity #3): public OCR benchmarks report
character/word error rate, which *conflates* two distinct failures:

    (A) transcription error  -- got the characters wrong.
    (B) reading-order error  -- transcribed the right text in the wrong order
                                (multi-column linearized as one column, table
                                cells serialized wrong, sidebars interleaved).

(B) is the dominant silent failure on real documents and is essentially
un-measured in isolation. This module separates them:

    * CER  -- character error rate (Levenshtein / len(ref)). Sensitive to BOTH.
    * block_order_score -- match predicted blocks to reference blocks by text
      similarity, then measure how well the *order* of matched blocks agrees
      with the reference order via Kendall's tau-b. This isolates (B): a model
      that transcribes every block perfectly but in scrambled order scores ~1.0
      CER-wise-fine yet low on order.
    * coverage / spurious -- fraction of reference blocks found, and predicted
      blocks with no good reference match (hallucinated structure).

Pure-Python, no heavy deps (difflib for fuzzy match). Deterministic + tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher


# ---------- text normalization ----------

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)  # drop punctuation for order/coverage matching
    return s.strip()


def split_blocks(s: str) -> list[str]:
    """Split into comparable blocks. Blank-line-separated paragraphs if present,
    else non-empty lines. Markdown table rows and headings become their own
    blocks, which is what we want for order checking."""
    s = s.replace("\r\n", "\n")
    # strip our page markers / html comments
    s = re.sub(r"<!--.*?-->", "", s, flags=re.DOTALL)
    if "\n\n" in s:
        parts = re.split(r"\n\s*\n", s)
    else:
        parts = s.split("\n")
    blocks = [p.strip() for p in parts if p.strip()]
    return blocks


# ---------- edit distance / CER ----------

def levenshtein(a: str, b: str) -> int:
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
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def cer(reference: str, prediction: str) -> float:
    ref = normalize_text(reference)
    pred = normalize_text(prediction)
    if not ref:
        return 0.0 if not pred else 1.0
    return levenshtein(ref, pred) / len(ref)


# ---------- block matching + reading order ----------

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def match_blocks(ref_blocks: list[str], pred_blocks: list[str],
                 threshold: float = 0.5):
    """Greedy best-match each reference block to a predicted block.

    Returns (matches, coverage, spurious) where matches is a list of
    (ref_idx, pred_idx, sim) for matched pairs in reference order.
    """
    nref = [normalize_text(b) for b in ref_blocks]
    npred = [normalize_text(b) for b in pred_blocks]
    used_pred: set[int] = set()
    matches = []
    for ri, rb in enumerate(nref):
        best_pi, best_sim = -1, 0.0
        for pi, pb in enumerate(npred):
            if pi in used_pred:
                continue
            sim = _similarity(rb, pb)
            if sim > best_sim:
                best_sim, best_pi = sim, pi
        if best_pi >= 0 and best_sim >= threshold:
            used_pred.add(best_pi)
            matches.append((ri, best_pi, round(best_sim, 3)))

    coverage = len(matches) / len(ref_blocks) if ref_blocks else 1.0
    spurious = (len(pred_blocks) - len(used_pred)) / len(pred_blocks) \
        if pred_blocks else 0.0
    return matches, coverage, spurious


def kendall_tau_b(order: list[int]) -> float:
    """Kendall's tau-b of `order` (predicted positions, in reference order)
    against the identity 0..n-1. +1 = perfect order, -1 = fully reversed,
    0 = no correlation. Handles ties (shouldn't occur with unique indices,
    but kept correct)."""
    n = len(order)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = order[j] - order[i]
            if d > 0:
                concordant += 1
            elif d < 0:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return 1.0
    return (concordant - discordant) / denom


@dataclass
class ReadingOrderReport:
    cer: float
    block_order_score: float   # kendall tau-b on matched blocks, rescaled 0..1
    kendall_tau: float         # raw tau-b, -1..1
    coverage: float            # fraction of reference blocks matched
    spurious_rate: float       # fraction of predicted blocks unmatched
    n_ref_blocks: int
    n_pred_blocks: int
    n_matched: int

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"CER:                 {self.cer:.3f}  (lower better)\n"
            f"reading-order score: {self.block_order_score:.3f}  "
            f"(1=perfect order, 0.5=random, tau={self.kendall_tau:+.3f})\n"
            f"coverage:            {self.coverage:.3f}  "
            f"({self.n_matched}/{self.n_ref_blocks} ref blocks found)\n"
            f"spurious rate:       {self.spurious_rate:.3f}  "
            f"(unmatched predicted blocks; proxy for hallucinated structure)\n"
            f"blocks: ref={self.n_ref_blocks} pred={self.n_pred_blocks}"
        )


def evaluate(reference: str, prediction: str,
             match_threshold: float = 0.5) -> ReadingOrderReport:
    ref_blocks = split_blocks(reference)
    pred_blocks = split_blocks(prediction)
    matches, coverage, spurious = match_blocks(
        ref_blocks, pred_blocks, threshold=match_threshold)
    pred_order = [pi for (_, pi, _) in matches]  # predicted positions in ref order
    tau = kendall_tau_b(pred_order)
    order_score = (tau + 1) / 2  # rescale -1..1 -> 0..1
    return ReadingOrderReport(
        cer=round(cer(reference, prediction), 4),
        block_order_score=round(order_score, 4),
        kendall_tau=round(tau, 4),
        coverage=round(coverage, 4),
        spurious_rate=round(spurious, 4),
        n_ref_blocks=len(ref_blocks),
        n_pred_blocks=len(pred_blocks),
        n_matched=len(matches),
    )

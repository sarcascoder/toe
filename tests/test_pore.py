"""Property tests for the PORE metric. These ARE the paper's correctness claims.

Run: PYTHONPATH=. python3 tests/test_pore.py
"""

from itertools import permutations

from taul.research.partial_order import (
    PartialOrder, violation_rate, kendall_tau_distance_normalized,
)
from taul.research import synth, metric


def test_P1_ambiguity_robustness():
    """violation_rate == 0 for EVERY valid linear extension of P."""
    # body(0,1) chain ; sidebar(2) independent
    P = PartialOrder(3).chain([0, 1])
    exts = P.linear_extensions()
    assert len(exts) == 3, exts  # 2 can go before/between/after 0,1
    for ext in exts:
        assert violation_rate(list(ext), P) == 0.0, ext
    # an order that violates 0<1 must be penalized
    assert violation_rate([1, 0, 2], P) > 0.0
    print(f"P1 ambiguity-robustness OK ({len(exts)} valid extensions, all 0 violation)")


def test_P2_total_order_reduction():
    """For a total order, violation_rate == normalized Kendall-tau distance."""
    P = PartialOrder(4).chain([0, 1, 2, 3])
    for perm in permutations(range(4)):
        order = list(perm)
        vr = violation_rate(order, P)
        kt = kendall_tau_distance_normalized(order, [0, 1, 2, 3])
        assert abs(vr - kt) < 1e-9, (order, vr, kt)
    print("P2 total-order reduction OK (violation_rate == normalized Kendall tau)")


def test_P3_transcription_invariance():
    """Ordering score is invariant to per-block character corruption."""
    doc = synth.make_doc("two_column", seed=1)
    gt = doc.gt_texts()
    # correct order, clean
    clean = synth.SynthDoc  # noqa (silence linter)
    from taul.research.run_study import mock_predict, _corrupt
    import random
    pred_clean = mock_predict(doc, "column_aware")
    pred_noisy = [_corrupt(t, random.Random(7), rate=0.15) for t in pred_clean]
    r_clean = metric.evaluate(gt, doc.partial_order, pred_clean)
    r_noisy = metric.evaluate(gt, doc.partial_order, pred_noisy)
    assert r_noisy.transcription_error > r_clean.transcription_error
    assert abs(r_noisy.ordering_violation - r_clean.ordering_violation) < 1e-9
    print(f"P3 transcription-invariance OK (ordering identical {r_clean.ordering_violation} "
          f"despite CER {r_clean.transcription_error}->{r_noisy.transcription_error})")


def test_P4_order_invariance_of_transcription():
    """Transcription error is invariant to reordering blocks (matching holds)."""
    doc = synth.make_doc("single_column", seed=2)
    gt = doc.gt_texts()
    from taul.research.run_study import mock_predict
    pred = mock_predict(doc, "column_aware")
    pred_rev = list(reversed(pred))
    r1 = metric.evaluate(gt, doc.partial_order, pred)
    r2 = metric.evaluate(gt, doc.partial_order, pred_rev)
    assert abs(r1.transcription_error - r2.transcription_error) < 1e-9
    assert r2.ordering_violation > r1.ordering_violation  # reversal hurts order
    print(f"P4 order-invariance of transcription OK "
          f"(trans {r1.transcription_error}=={r2.transcription_error}; "
          f"order {r1.ordering_violation}->{r2.ordering_violation})")


def test_discrimination_raster_vs_aware():
    """The benchmark must DISCRIMINATE: a raster reader should fail multi-column
    ordering while transcribing perfectly; a column-aware reader should not."""
    from taul.research.run_study import mock_predict
    doc = synth.make_doc("two_column", seed=3)
    gt = doc.gt_texts()
    aware = metric.evaluate(gt, doc.partial_order, mock_predict(doc, "column_aware"))
    raster = metric.evaluate(gt, doc.partial_order, mock_predict(doc, "raster"))
    assert aware.ordering_violation == 0.0
    assert raster.ordering_violation > 0.0
    # both have ~perfect transcription -> CER alone can't tell them apart
    assert aware.transcription_error < 0.05 and raster.transcription_error < 0.05
    print(f"DISCRIMINATION OK: same transcription (~0), "
          f"ordering aware={aware.ordering_violation} vs raster={raster.ordering_violation}")


def test_ambiguity_not_penalized_on_newsletter():
    """A model reading independent stories in a different (but internally
    correct) order must NOT be penalized -- the whole point vs. gold-sequence."""
    from taul.research.run_study import _reading_order_ids
    doc = synth.make_doc("newsletter", seed=4)
    gt = doc.gt_texts()
    by_id = {b.id: b.text for b in doc.blocks}
    # read stories in a permuted order but each story internally in order
    # find the chains from constraints
    order = _reading_order_ids(doc)
    # build an alternative valid extension: reverse the story blocks order
    alt = list(reversed(order))
    # repair internal chains to remain valid by re-sorting within story groups
    # (simplest: take any linear extension)
    exts = doc.partial_order.linear_extensions(limit=50)
    assert len(exts) > 1, "newsletter should have many valid orders"
    pred_alt = [by_id[i] for i in exts[-1]]
    r = metric.evaluate(gt, doc.partial_order, pred_alt)
    assert r.ordering_violation == 0.0, r.ordering_violation
    print(f"AMBIGUITY OK: {len(exts)}+ valid orders on newsletter, "
          f"alt extension scores violation={r.ordering_violation}")


if __name__ == "__main__":
    test_P1_ambiguity_robustness()
    test_P2_total_order_reduction()
    test_P3_transcription_invariance()
    test_P4_order_invariance_of_transcription()
    test_discrimination_raster_vs_aware()
    test_ambiguity_not_penalized_on_newsletter()
    print("\nALL PORE PROPERTY TESTS PASSED")

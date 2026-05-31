# Disentangling Reading Order from Transcription: A Partial-Order Metric and Synthetic Benchmark for Document Parsing

*Working draft / paper scaffold. Status: method + metric implemented and
property-verified; synthetic benchmark implemented; real-VLM study is the
remaining empirical work (runs on Apple Silicon via the provided runner).*

---

## Abstract

Modern vision-language models (VLMs) report document-parsing quality with a
single edit-distance-style score (CER/WER, normalized edit distance, or
OmniDocBench's reading-order component). These scores **conflate two
fundamentally different failures**: getting characters wrong (transcription) and
emitting correct text in the wrong order (reading order). They also penalize
models against a single, arbitrarily-chosen gold linearization, even though the
"true" reading order of multi-column, sidebar, and figure-heavy pages is
*intrinsically a partial order* — many traversals are equally valid.

We introduce **PORE (Partial-Order Reading-order Evaluation)**, which (i) models
the acceptable reading orders of a page as the linear extensions of a
ground-truth partial order over content blocks, scoring a prediction by its
*violation rate of required precedences* rather than its distance to one gold
sequence; and (ii) **orthogonally decomposes** parsing error into an
order-invariant transcription component and a transcription-invariant ordering
component. We prove and empirically verify the metric's invariance properties.
We release a parametric **synthetic benchmark** whose layouts span a
constraint-density gradient and carry exact ground-truth partial orders by
construction, sidestepping the ambiguity and cost of human reading-order
annotation. In controlled experiments, a naive raster reader achieves
**near-zero transcription error yet up to ~0.27 ordering-violation rate** on
multi-column and sidebar layouts — a failure invisible to current scalar
metrics. We argue ordering quality should be reported as a separate axis, and
that "OCR accuracy" leaderboards systematically over-credit models that read
complex layouts incorrectly.

---

## 1. Introduction & motivation

- Document parsing / "PDF-to-Markdown" with VLMs is a fast-growing capability;
  leaderboards (OmniDocBench, PureDocBench, MPDocBench-Parse) drive model
  selection.
- OmniDocBench is reported **saturated**; the community is explicitly asking
  "what's next for OCR benchmarks?" — a timing argument for a new *axis* rather
  than a harder dataset.
- Central claim: the field measures *content fidelity* well and *reading order*
  poorly, and worse, the two are entangled in one number. A model that
  transcribes perfectly but linearizes a two-column page as one column can score
  similarly to one with scattered character errors — despite being a completely
  different (and often more damaging, for RAG/extraction) failure.

**Contributions.**
1. **PORE metric**: partial-order, ambiguity-robust ordering score +
   orthogonal transcription/ordering/detection decomposition (Section 3).
2. **Provable + tested properties**: ambiguity-robustness (P1), total-order
   reduction to Kendall-τ (P2), transcription-invariance of ordering (P3),
   order-invariance of transcription (P4) (Section 3.4, `tests/test_pore.py`).
3. **Synthetic benchmark** with exact ground-truth partial orders across a
   constraint-density gradient (Section 4).
4. **Controlled study** showing reading-order failure is real, large, and
   invisible to scalar metrics; and the finding that **ordering difficulty
   tracks cross-region constraint density, not visual complexity** (Section 5).

---

## 2. Related work & honest positioning

*(This section deliberately states what already exists so the contribution is
not overclaimed.)*

- **OmniDocBench (CVPR 2025)** evaluates four modules including a reading-order
  component computed via block matching + edit distance, with v1.6 adding
  Multi-Granularity Adaptive Matching. → Reading order *is* measured, but as an
  entangled edit-distance figure against a single gold order. PORE differs by
  (a) decomposing it out as an orthogonal axis and (b) scoring against a partial
  order rather than one sequence.
- **LayoutReader / ReadingBank (2021)** target reading-order *detection given
  layout boxes*, scored with Average Relative Distance (ARD) and page-BLEU. This
  is a different task (ordering known boxes) from end-to-end VLM parsing, and ARD
  still assumes a single gold permutation.
- **MPDocBench-Parse (2026)** evaluates reading order and heading hierarchy for
  multi-page parsing. Complementary; again single-gold-order based.
- **Known open problem (cited in the literature):** for multi-column,
  marginalia, and figure-heavy layouts, "the ground-truth reading order is
  intrinsically ambiguous; acceptable traversals can differ while preserving
  content." **PORE's partial-order formulation is a direct, operational answer
  to exactly this stated problem.**

**Novelty, precisely:** not "we measure reading order" (others do), but
(1) ambiguity-robust scoring via partial-order violation rate, (2) a clean
orthogonal decomposition with proven invariances, and (3) a synthetic benchmark
that makes the partial order *exact and free* rather than hand-annotated.

---

## 3. Method: the PORE metric

### 3.1 Setup
A ground-truth page is a set of content blocks `B = {b_1..b_n}`, each with text
`t_i`, plus a **partial order** `P ⊆ B×B` of required precedences (`(a,b)∈P`
means "a must be read before b"). The set of correct reading orders is exactly
the linear extensions of `P`. A model prediction is a *sequence* of predicted
text blocks (in output/reading order).

### 3.2 Matching
Greedy maximum-similarity assignment (difflib ratio over normalized text, ≥τ)
maps GT blocks to predicted blocks — the same matching family OmniDocBench uses.
Coverage = matched/|B|; spurious = unmatched-predicted/|pred| (a proxy for
hallucinated structure).

### 3.3 Three orthogonal components
- **Transcription error** (order-invariant): mean normalized edit distance over
  matched (GT, pred) text pairs.
- **Ordering violation** (transcription-invariant): take the matched GT block
  ids in the order their matched predictions appear; report the fraction of
  `closure(P)` precedences this order *contradicts*. `0` iff it is a valid
  linear extension of `P`.
- **Detection**: coverage, spurious rate.

### 3.4 Properties (proved + unit-tested)
- **P1 Ambiguity-robustness.** `violation_rate = 0` for *every* linear extension
  of `P`. Independent regions (sidebars, separate stories) incur no penalty.
- **P2 Total-order reduction.** If `P` is total, `violation_rate` equals the
  normalized Kendall-τ distance to that order — i.e. PORE generalizes the
  familiar metric.
- **P3 Transcription-invariance.** Per-block character corruption leaves the
  ordering score unchanged (matching held).
- **P4 Order-invariance of transcription.** Reordering blocks leaves the
  transcription score unchanged (matching held).

All four pass in `tests/test_pore.py` (deterministic).

---

## 4. Synthetic benchmark

Hand-annotating reading order is costly and, for complex layouts, ill-posed. We
*generate* documents so the partial order is exact by construction. Layouts span
a gradient:

| layout | regions | partial order | valid orders |
|---|---|---|---|
| single_column | 1 chain | total | 1 |
| table | grid | row-major total | 1 |
| two_column | L,R | L-chain, R-chain, all-L≺all-R (total) | 1 |
| body_with_sidebar | L,R,sidebar | as two_column; sidebar **unordered** | many |
| newsletter | 3 stories | 3 independent chains, no cross-constraints | many |

Each sample renders to a PNG (for running real VLMs) with a sidecar JSON of
blocks + bboxes + constraints. Generator: `taul/research/synth.py`.

---

## 5. Experiments

### 5.1 Controlled mock-model study (provides ground-truth failure modes)
Three reference readers isolate each failure axis:
- `column_aware` — perfect transcription, valid order.
- `raster` — perfect transcription, naive top-to-bottom/left-to-right order.
- `noisy_reader` — valid order, per-block character noise.

**Result (5 docs/layout; `pore_study/results.md`).**

| model | metric | single | table | two_col | sidebar | newsletter |
|---|---|---|---|---|---|---|
| column_aware | ordering_violation | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| raster | transcription_error | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| raster | **ordering_violation** | 0.00 | 0.00 | **0.21** | **0.27** | 0.00 |
| noisy_reader | transcription_error | 0.06 | 0.06 | 0.06 | 0.06 | 0.07 |
| noisy_reader | ordering_violation | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Two findings:
1. **Conflation is real (Fig. orthogonality).** `raster` and `noisy_reader`
   occupy orthogonal corners of the (transcription, ordering) plane yet a single
   scalar metric collapses them onto one axis and cannot tell them apart.
2. **Difficulty = constraint density, not visual complexity (Fig. complexity).**
   `raster` fails two_column/sidebar (cross-region precedence required) but
   scores 0 on the *more visually complex* newsletter, because independent
   stories impose no cross-constraints — so interleaving is genuinely valid.
   A single-gold-order metric would wrongly penalize this. PORE does not.

### 5.2 Real-VLM study (remaining work — runs on Apple Silicon)
`python -m taul.research.run_study --real-model deepseek-ocr2` evaluates a
real model on the suite and plots it in the same (transcription, ordering)
plane. **Hypothesis H1:** strong document VLMs show low transcription error but
non-trivial ordering violation that grows with cross-region constraint density —
i.e. a chunk of reported "OCR error" is actually ordering error. Confirming H1
on 3–5 open VLMs (DeepSeek-OCR-2, PaddleOCR-VL-1.5, Qwen3-VL) is the empirical
core of the submission.

---

## 6. Limitations & threats to validity (state plainly)

### 6.1 Matcher robustness (empirical; `taul/research/robustness.py`)
The decomposition assumes blocks are matched, so the obvious attack is: does
transcription noise break matching and contaminate the ordering axis? We sweep
per-character corruption `r ∈ [0, 0.6]` on (i) a known-*valid* order and (ii) a
known-*wrong* (raster) order, averaging over all layouts (10 docs each).

Finding (Fig. robustness): the matcher loses **recall** under corruption but
keeps **precision** — surviving matches preserve correct relative order — so the
ordering axis stays faithful well past where transcription degrades:
- valid-order `ordering_violation` stays **exactly 0.000** across the *entire*
  sweep, even as `transcription_error` climbs to 0.69;
- wrong-order `ordering_violation` holds steady at **~0.095** until coverage
  falls below ~0.30 (corruption ≳ 0.55), where it finally decays and becomes
  unreliable.

**Recommended validity bound is stated in coverage, not corruption:** the
ordering axis is trustworthy while **coverage ≳ 0.4**. Match threshold τ is
stable in [0.3, 0.7] (τ=0.8 over-prunes). *Honest caveat:* our coverage-vs-
corruption curve is **pessimistic** — synthetic blocks share a ~40-word
vocabulary, so corrupted blocks collide with each other and drop out fast
(coverage 0.82 already at r=0.05). Real document blocks are far more lexically
distinct, so matching (and thus the usable corruption range) should be
considerably more robust on real pages. This is a synthetic-text artifact, not a
property of the metric.

### 6.2 Other limitations
- **Synthetic ≠ real.** Generated pages lack the noise, fonts, and visual
  richness of scanned documents; results must be confirmed on real pages, where
  the partial order must be human-annotated (we propose annotating *constraints*,
  which is easier and less ambiguous than annotating a full sequence).
- **Matching dependence.** Quantified in 6.1: safe while coverage ≳ 0.4; we
  always report coverage so out-of-regime evaluations are visible.
- **Partial-order specification is a modeling choice.** Reasonable annotators may
  disagree on whether a sidebar is truly unordered. PORE makes this assumption
  *explicit and inspectable* (the DAG), which is itself an improvement over an
  implicit single gold order.
- **Block granularity.** Like OmniDocBench, results depend on segmentation
  granularity; we fix it via generation, but real-doc evaluation inherits this.

---

## 7. Why this could matter (and how it could spread)

- It reframes a saturated leaderboard around a *new axis* rather than a harder
  dataset — historically a high-citation move (cf. metrics papers that expose a
  blind spot).
- The claim "your 95%-accurate OCR model is silently misreading multi-column
  pages" is concrete, checkable, and consequential for RAG/extraction pipelines.
- Everything is reproducible on a laptop (no GPU); the benchmark is free to
  regenerate; the metric is ~200 lines and property-tested.

**Honest caveat:** virality is not engineerable. The defensible target is a
solid workshop/benchmark-track paper; broader attention depends on the real-VLM
results in 5.2 actually showing a large, surprising gap. The infrastructure to
find out is built and verified here.

---

## 8. Reproducibility

```
pip install -r requirements.txt
PYTHONPATH=. python3 tests/test_pore.py                 # property proofs
PYTHONPATH=. python3 -m taul.research.run_study \
    --out pore_study --per-layout 5                     # mock study + figures
# real VLM (Apple Silicon):
PYTHONPATH=. python3 -m taul.research.run_study \
    --out pore_real --real-model deepseek-ocr2
```

Artifacts: `pore_study/results.csv`, `results.md`, `fig_complexity.png`,
`fig_orthogonality.png`, and per-doc `suite/*.png` + `*.json`.
```
```

# toe

**Measure how well a document extractor preserved reading order — on top of whatever tool you already use.**

When you convert a PDF to Markdown (with Marker, Docling, MinerU, olmOCR, an LLM,
anything), the text can come out **correct word-for-word but in the wrong order** —
a two-column page read straight across, a sidebar spliced into the body, table
cells serialized wrong. Plain accuracy scores (CER/edit distance) hide this, and
it quietly breaks RAG, search, and downstream extraction.

`toe` scores an extraction's **reading order separately from its character
accuracy**, so that silent failure becomes visible. It's extractor-agnostic: bring
any tool's output. It also includes *optional* lightweight on-device extraction on
Apple Silicon if you want to generate predictions locally — but the point of toe
is the scoring, not being another extractor.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Pure-Python core](https://img.shields.io/badge/core-pure--python-blue)

---

## What toe is (and isn't)

- ✅ **A quality/eval layer.** Score reading order vs. transcription for the output
  of *any* extractor. Compare tools, regression-test your pipeline, build a labeled
  eval set.
- ✅ **Extractor-agnostic & local.** Pure-Python core; nothing leaves your machine.
- ➕ **Optional on-device extraction** (Apple Silicon / MLX) for generating outputs.
- ❌ **Not** a replacement for Marker/Docling/MinerU/olmOCR. Those are better, more
  capable extractors — toe *evaluates* their output, it doesn't try to beat them.

> **Honest scope:** scoring is **reference-based** — you provide a ground-truth file
> and toe scores a prediction against it. So today toe is a benchmarking /
> regression-testing tool (great for "compare N extractors on a labeled set" or "did
> my pipeline regress?"), not a no-reference "is this extraction good?" detector.

## Install

```bash
pip install toe            # core scorer (pure-python: demo, eval, list-models)
pip install "toe[full]"    # + JSON-schema validation, plots, PDF rasterization
pip install "toe[mlx]"     # + OPTIONAL on-device extraction (Apple Silicon only)
```

Installing toe does **not** download any OCR/VLM models. If you use the optional
`extract`, the model is pulled from Hugging Face on first run and cached.

## Quickstart — score an extraction

```bash
# 0) verify the install instantly (no model, no network)
toe demo

# 1) you have a ground-truth markdown and an extractor's output -> score it
toe eval --ref truth.md --pred marker_output.md
```

```
CER:                 0.012   (character error — lower is better)
reading-order score: 0.640   (1 = correct order, lower = scrambled)
coverage:            1.000   (fraction of reference blocks found)
spurious rate:       0.000   (hallucinated/extra blocks)
```

The two numbers are **independent**: a model can score ~0 CER (perfect characters)
yet a low reading-order score (it read the layout wrong). That gap is the whole
point.

## Use it with any extractor

toe doesn't care how `pred.md` was produced. For example:

```bash
# Marker
marker_single mydoc.pdf --output_dir out/ && \
  toe eval --ref truth.md --pred out/mydoc.md

# Docling
docling mydoc.pdf --to md --output out/ && \
  toe eval --ref truth.md --pred out/mydoc.md

# MinerU, olmOCR, an LLM, your own pipeline ... same pattern
```

Compare several extractors on the same labeled doc and see which preserves reading
order best — not just which has the lowest character error.

## Optional: generate predictions locally (Apple Silicon)

If you'd rather produce outputs on-device instead of running a separate tool:

```bash
pip install "toe[mlx]"
toe extract mydoc.pdf -o pred.md            # local VLM via MLX
toe structured receipt.jpg --example-invoice -o out.json
toe list-models                              # local model registry + RAM
```

This is a convenience, not the headline — small specialist models (DeepSeek-OCR-2,
PaddleOCR-VL, Qwen3-VL) that fit on a laptop.

## Commands

| command | what it does | needs a model? |
|---|---|---|
| `eval` | **score an extraction**: character error + reading-order, separately | no |
| `demo` | self-check; shows what a reading-order failure looks like | no |
| `bench` | compare local models on the same doc (tok/s + peak RAM) | yes |
| `list-models` | the optional local-extraction registry | no |
| `extract` | *(optional)* PDF/image → Markdown via on-device VLM | yes |
| `structured` | *(optional)* PDF/image → schema-validated JSON | yes |

## How the scoring works

`toe eval` segments reference and prediction into blocks, matches them by text
similarity, then reports:

- **CER** — normalized edit distance (sensitive to wrong characters).
- **reading-order score** — based on how well the predicted block order agrees with
  the reference order; *invariant to character errors*, so it isolates ordering.
- **coverage / spurious** — how many reference blocks were found / how many extra
  predicted blocks appeared.

A research-grade evaluator lives in `toe/research/` (`PORE`): it models valid
reading orders as a **partial order** (so independent regions like sidebars aren't
unfairly penalized) and decomposes error into transcription vs. ordering with tested
invariances. See [`PAPER.md`](PAPER.md).

```bash
pip install -e ".[full,dev]"
python tests/test_pore.py                       # property tests
python -m toe.research.run_study --out pore_study --per-layout 5
```

## Honest prior art

Reading-order *detection* is well covered — Docling, MinerU, and Éclair produce
reading order; HURIDOCS ships a dedicated model; ParseBench and OmniDocBench score
it inside their benchmarks. toe's contribution is **packaging**: a small,
extractor-agnostic, local CLI that gives you the order-vs-transcription split on
your own outputs in one command. It's convenience and transparency, not new model
tech.

## Limitations

- **Reference-based.** You need ground-truth text to score against (it's a
  benchmark/regression tool, not a no-reference quality detector — yet).
- Optional `extract` needs the `[mlx]` extra and Apple Silicon; it won't run on
  Intel/Windows/Linux. The scorer (`eval`, `demo`) runs anywhere.
- Block matching can degrade if a prediction is *extremely* corrupted; toe
  reports `coverage` so out-of-regime scores are visible.

## License

MIT — see [LICENSE](LICENSE). Free to use, modify, and redistribute.

## Contributing

Issues and PRs welcome — especially **real-document reading-order test cases**
(a reference + an extractor's output), adapters for popular extractors, and model
registry entries. See [CONTRIBUTING.md](CONTRIBUTING.md).

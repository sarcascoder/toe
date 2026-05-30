# macdoc

**On-device document extraction for Apple Silicon — no GPU, no cloud, no data leaving your Mac.**

`macdoc` turns PDFs and images into clean Markdown or structured JSON using small
vision-language models that run locally via [MLX](https://github.com/ml-explore/mlx).
It also ships a built-in **reading-order checker** — a feature most OCR tools
lack — so you can catch the silent failure where a model transcribes every word
correctly but in the wrong order (a classic problem on multi-column pages,
tables, and forms).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1--M4-black)

---

## Why macdoc

- **Private & offline.** Everything runs on your Mac. Documents never leave the device.
- **No GPU needed.** Designed for Apple Silicon unified memory (M1–M4). Small,
  efficient models — a 0.9B specialist beats a 70B generalist on real-world scans.
- **More than OCR.** Transcribe to Markdown, extract schema-validated JSON, **and**
  measure reading-order quality — separately from character accuracy.
- **Try it in 5 seconds.** `macdoc demo` runs with no model download and no network.

## Install

```bash
pip install macdoc            # core (pure-python: demo, eval, list-models)
pip install "macdoc[mlx]"     # + on-device inference (Apple Silicon only)
pip install "macdoc[full]"    # + PDF rasterization, JSON-schema validation, plots
```

From source:

```bash
git clone https://github.com/your-org/macdoc && cd macdoc
pip install -e ".[mlx,full]"
```

## Quickstart

```bash
# 1) verify the install instantly — no model, no network
macdoc demo

# 2) transcribe a PDF to Markdown (downloads a small model on first run)
macdoc extract invoice.pdf -o invoice.md

# 3) pull structured fields as schema-validated JSON
macdoc structured receipt.jpg --example-invoice -o receipt.json
macdoc structured form.png --schema my_schema.json -o form.json

# 4) check reading-order quality against a ground-truth file
macdoc eval --ref truth.md --pred invoice.md

# 5) see available local models and their RAM footprint
macdoc list-models
```

## Commands

| command | what it does | needs a model? |
|---|---|---|
| `demo` | self-check + shows what reading-order error looks like | no |
| `extract` | PDF/image → Markdown, layout & reading order preserved | yes |
| `structured` | PDF/image → JSON validated against a schema (parse→validate→repair) | yes |
| `eval` | score an extraction: character error **and** reading-order, separately | no |
| `bench` | tok/s + peak RAM across local models on the same doc | yes |
| `list-models` | the local model registry with RAM estimates | no |

## Local models (registry)

Pick by RAM budget; pass any Hugging Face repo id to override.

| key | ~RAM @4-bit | best for |
|---|---|---|
| `deepseek-ocr2` (default) | ~4 GB | optical-compression OCR, layout + reading order |
| `paddleocr-vl` | ~2 GB | tiny 0.9B specialist, robust on messy/photographed scans |
| `qwen3-vl` | ~3 GB | reasoning over docs (VQA, charts), not just transcription |
| `deepseek-vl2` | ~8 GB | 27B MoE / 4.5B active — OCR + tables + charts + grounding |

> Repo ids are best-effort; if one 404s, search the HF Hub for an
> `mlx-community/<name>` build or convert with `python -m mlx_vlm.convert`.

## The reading-order checker

Standard OCR scores conflate two different failures: **wrong characters** and
**right text in the wrong order**. `macdoc eval` reports them as separate numbers:

```
$ macdoc eval --ref truth.md --pred out.md
CER:                 0.012   (character error — lower is better)
reading-order score: 0.640   (1 = correct order, lower = scrambled)
coverage:            1.000   (fraction of reference blocks found)
spurious rate:       0.000   (hallucinated/extra blocks)
```

A high CER-accuracy model can still score poorly on reading order — which quietly
breaks RAG, search, and data extraction downstream. macdoc makes that visible.

## Advanced / research

The `macdoc/research/` package contains a partial-order reading-order evaluator
(`PORE`), a synthetic benchmark generator with ground-truth reading orders, and
study/robustness runners. See [`PAPER.md`](PAPER.md). Run the property tests:

```bash
pip install -e ".[full,dev]"
python -m pytest tests/ -q          # or: python tests/test_pore.py
python -m macdoc.research.run_study --out pore_study --per-layout 5
```

## Requirements

- macOS on Apple Silicon (M1–M4) for **inference**. The `demo`, `eval`,
  `list-models`, and research tools are pure-Python and run anywhere.
- Python 3.9+.

## Limitations (honest notes)

- Inference needs the `[mlx]` extra and Apple Silicon; it won't run on Intel/Windows/Linux.
- Structured extraction uses prompt→validate→repair, not hard grammar-constrained
  decoding (mlx-vlm doesn't expose that yet); always validate against your schema.
- Model repo ids drift; `list-models` shows the current set.

## License

MIT — see [LICENSE](LICENSE). Free to use, modify, and redistribute.

## Contributing

Issues and PRs welcome — model registry additions, new schema templates, and
real-document reading-order test sets are especially useful.

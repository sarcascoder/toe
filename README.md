# taul

**A pure-Python CLI that scores document reading order — separately from character accuracy.**

> *taul* — the silent failure mode that kills your RAG pipeline.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/taul.svg)](https://pypi.org/project/taul/)

---

## The 30-second pitch

Your OCR is "98% accurate" on character level. Your RAG still returns garbage. Why?

Because the model OCR'd a two-column page top-to-bottom-column-1, then top-to-bottom-column-2 — but your document has interleaved columns, footnotes referenced across pages, and a sidebar legend. The characters are right. The **order** is wrong. RAG retrieves chunk 47 from what is actually the bottom of column 2 of page 5, which mentions "the above table" — and you have no idea why your answer is hallucinated.

Standard OCR metrics (CER, WER, exact match) don't catch this. **taul does.**

---

## Install

```bash
pip install taul
```

## Use

```bash
taul score --pred my_ocr_output.json --gold ground_truth.json
```

Output:

```
Document: contract_47.pdf
  Character accuracy: 0.984
  Reading-order accuracy: 0.612  ← this is why your RAG is broken
  Worst spans:
    [page 3] col-2 block 4 -> read before col-1 block 7 (Kendall τ = -0.4)
    [page 5] footnote orphaned (1.2 KB before parent reference)
  Recommended layout strategy: 2-column with explicit footnote linking
```

## Use in Python

```python
from taul import score, ReadingOrderError

result = score(pred="ocr.json", gold="gold.json")
if result.reading_order < 0.85:
    raise ReadingOrderError(result.worst_spans)
```

---

## Why this is a separate tool

Character accuracy and reading-order accuracy are *orthogonal failure modes*. Combining them into one metric hides which one is broken. taul scores reading order alone, surfacing layout-pipeline issues that standard OCR evals silently mask.

Pairs natively with [parakh](https://github.com/sarcascoder/parakh) for full extraction-quality evaluation, and with [parakh Cloud](https://parakh.cloud) for hosted dashboards and history.

## License

MIT. Built because I needed it and you probably do too.

📧 **tanupam760@gmail.com**

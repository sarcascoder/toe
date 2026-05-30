# Contributing to macdoc

Thanks for your interest! macdoc is a small, friendly project and contributions
of all sizes are welcome.

## Quick start

```bash
git clone https://github.com/sarcascoder/macdoc && cd macdoc
python -m venv .venv && source .venv/bin/activate
pip install -e ".[full,dev]"     # add ,mlx on Apple Silicon for inference
macdoc demo                      # 5-second self-check, no model needed
python -m pytest tests/ -q       # run the property tests
```

## Good first contributions

- **Model registry entries** (`macdoc/models.py`): add a small VLM with a working
  `mlx-community/*` repo id, RAM estimate, and a one-line note on what it's good at.
- **Schema templates** for `structured` extraction (invoices, receipts, forms, IDs).
- **Real-document reading-order test cases**: a page image + its block reading
  order, to grow the eval set beyond synthetic.
- **Docs**: clearer examples, troubleshooting, screenshots/GIFs.
- **Bug fixes** for the mlx-vlm `generate()` API drift across versions.

## Guidelines

- Keep the **core pure-Python** so `demo`, `eval`, and `list-models` work without
  mlx. Put inference-only code behind lazy imports (see `models.load_model`).
- Add or update a test when you change metric behavior — the PORE properties in
  `tests/test_pore.py` are the correctness contract.
- Be honest in docs about limitations; this project values signal over hype.
- Run `macdoc demo` and `pytest` before opening a PR. CI runs both on Python
  3.9–3.12.

## Filing issues

Include your macOS + chip (e.g. M4 Pro), Python version, the command you ran, and
the full error. For model issues, paste the repo id and `macdoc list-models`.

By contributing you agree your contributions are licensed under the project's MIT
license.

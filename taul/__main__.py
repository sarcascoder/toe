"""CLI entrypoint.

Examples:
  # list the model registry (works without mlx installed)
  python -m taul list-models

  # transcribe a PDF to Markdown with the default model
  python -m taul extract invoice.pdf -o out.md

  # structured extraction against the built-in invoice schema
  python -m taul structured receipt.jpg --example-invoice -o out.json

  # structured extraction against your own JSON Schema
  python -m taul structured form.png --schema my_schema.json -o out.json

  # benchmark two models on the same doc (the report's experiment)
  python -m taul bench sample.pdf -m deepseek-ocr2 -m paddleocr-vl

  # score an extraction against ground truth (CER + reading-order)
  python -m taul eval --ref truth.md --pred out.md

  # zero-setup demo: no model or network needed, verifies the install
  python -m taul demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_list_models(args):
    from .models import REGISTRY, DEFAULT_MODEL
    print(f"{'key':<16} {'~RAM':>6}  repo")
    print("-" * 72)
    for key, spec in REGISTRY.items():
        star = " *" if key == DEFAULT_MODEL else "  "
        print(f"{key:<16}{star}{spec.approx_ram_gb:>5.1f}G  {spec.repo}")
        print(f"{'':<18}{spec.note}")
    print("\n* = default. Pass any raw HF repo id to override the registry.")


def cmd_extract(args):
    from .extract import extract_document
    results, joined = extract_document(
        args.input, args.model, dpi=args.dpi, max_pages=args.max_pages,
        prompt=args.prompt, max_tokens=args.max_tokens, verbose=args.verbose,
    )
    for r in results:
        if r.tok_per_s:
            print(f"[page {r.page_index + 1}] {r.tok_per_s:.1f} tok/s "
                  f"({r.generation_tokens} tok)", file=sys.stderr)
    if args.output:
        Path(args.output).write_text(joined, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(joined)


def cmd_structured(args):
    from . import models, render, schema
    import tempfile

    if args.example_invoice:
        json_schema = schema.EXAMPLE_INVOICE_SCHEMA
    elif args.schema:
        json_schema = json.loads(Path(args.schema).read_text())
    else:
        raise SystemExit("Pass --schema <file.json> or --example-invoice")

    pages = render.load_pages(args.input, dpi=args.dpi, max_pages=args.max_pages)
    repo, _ = models.resolve_repo(args.model)
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        page_paths = render.save_pages_as_pngs(pages, tmp)
        model, processor = models.load_model(repo)
        for i, p in enumerate(page_paths):
            outcome = schema.extract_structured(
                model, processor, p, json_schema,
                instruction=args.prompt, max_tokens=args.max_tokens,
            )
            status = "ok" if outcome.ok else "FAILED"
            tag = " (repaired)" if outcome.repaired else ""
            print(f"[page {i + 1}] {status}{tag}"
                  + (f" -- {outcome.error}" if outcome.error else ""),
                  file=sys.stderr)
            out.append({"page": i + 1, "ok": outcome.ok,
                        "data": outcome.data, "error": outcome.error})

    payload = json.dumps(out if len(out) > 1 else out[0], indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(payload)


def cmd_bench(args):
    from .bench import bench_models, format_table
    rows = bench_models(args.input, args.model, dpi=args.dpi,
                        max_pages=args.max_pages, max_tokens=args.max_tokens)
    print(format_table(rows))


def cmd_eval(args):
    from .eval_reading_order import evaluate
    ref = Path(args.ref).read_text(encoding="utf-8")
    pred = Path(args.pred).read_text(encoding="utf-8")
    report = evaluate(ref, pred, match_threshold=args.threshold)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(report.summary())


def cmd_demo(args):
    """Zero-setup self-check: generate one synthetic page per layout, run the
    built-in reference readers, and print the PORE decomposition. Needs no model
    and no network -- proves the install works and shows what the tool measures."""
    from .research import synth, metric
    from .research.run_study import mock_predict
    print("taul demo -- no model/network needed\n")
    print(f"{'layout':<20}{'reader':<14}{'transcription':>14}{'ordering':>11}")
    print("-" * 59)
    for layout in synth.LAYOUTS:
        doc = synth.make_doc(layout, seed=42)
        gt = doc.gt_texts()
        for reader in ("column_aware", "raster", "noisy_reader"):
            pred = mock_predict(doc, reader)
            r = metric.evaluate(gt, doc.partial_order, pred)
            print(f"{layout:<20}{reader:<14}"
                  f"{r.transcription_error:>14.3f}{r.ordering_violation:>11.3f}")
        print()
    print("Note how 'raster' (a reader that ignores columns) shows ~0 "
          "transcription\nerror yet high ordering violation on multi-column "
          "layouts -- the failure\nplain text-accuracy scores hide. "
          "Install OK if you see this table.")


def build_parser():
    p = argparse.ArgumentParser(prog="taul",
                                description="Score how well a document extractor "
                                            "preserved reading order (works with any "
                                            "extractor). Optional local extraction.")
    p.add_argument("--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="cmd", required=False)

    dm = sub.add_parser("demo", help="zero-setup self-check (no model needed)")
    dm.set_defaults(func=cmd_demo)

    lm = sub.add_parser("list-models", help="show model registry")
    lm.set_defaults(func=cmd_list_models)

    def common(sp):
        sp.add_argument("input", help="PDF or image path")
        sp.add_argument("-m", "--model", default="deepseek-ocr2",
                        help="registry key or raw HF repo id")
        sp.add_argument("--dpi", type=int, default=200)
        sp.add_argument("--max-pages", type=int, default=None)
        sp.add_argument("--max-tokens", type=int, default=4096)
        sp.add_argument("-o", "--output", default=None)
        sp.add_argument("--prompt", default=None, help="override task prompt")

    ex = sub.add_parser("extract", help="transcribe to Markdown/text")
    common(ex)
    ex.add_argument("--verbose", action="store_true")
    ex.set_defaults(func=cmd_extract)

    st = sub.add_parser("structured", help="schema-validated JSON extraction")
    common(st)
    st.add_argument("--schema", help="path to a JSON Schema file")
    st.add_argument("--example-invoice", action="store_true",
                    help="use the built-in invoice schema")
    st.set_defaults(func=cmd_structured)

    bn = sub.add_parser("bench", help="tok/s + peak RAM across models")
    bn.add_argument("input")
    bn.add_argument("-m", "--model", action="append", required=True,
                    help="repeatable; e.g. -m deepseek-ocr2 -m paddleocr-vl")
    bn.add_argument("--dpi", type=int, default=200)
    bn.add_argument("--max-pages", type=int, default=2)
    bn.add_argument("--max-tokens", type=int, default=2048)
    bn.set_defaults(func=cmd_bench)

    ev = sub.add_parser("eval", help="CER + reading-order vs. ground truth")
    ev.add_argument("--ref", required=True, help="ground-truth text/markdown file")
    ev.add_argument("--pred", required=True, help="model output file")
    ev.add_argument("--threshold", type=float, default=0.5,
                    help="block-match similarity threshold (0..1)")
    ev.add_argument("--json", action="store_true", help="emit JSON instead of summary")
    ev.set_defaults(func=cmd_eval)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        from . import __version__
        print(f"taul {__version__}")
        return
    if not getattr(args, "func", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()

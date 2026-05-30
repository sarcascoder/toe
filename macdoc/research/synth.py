"""Synthetic document benchmark with ground-truth partial reading orders.

Why synthetic: hand-annotating reading order is expensive AND ambiguous (the
core problem we solve). By *generating* the layout we know, by construction:
  - each block's text and bounding box,
  - the partial order P of required precedences (within-region mandatory,
    cross-region only where layout dictates), and therefore
  - the full equivalence class of valid reading orders (linear extensions of P).

Layouts form a complexity gradient, each with a known P:

  single_column      total order; no ambiguity.                  [complexity 1]
  two_column         left col fully precedes right col;          [complexity 2]
                     within-column sequential. (still total)
  body_with_sidebar  two-column body + an independent sidebar     [complexity 3]
                     box that is UNORDERED w.r.t. the body
                     -> genuine ambiguity (many linear extensions).
  newsletter         3 independent stories, each an internal      [complexity 4]
                     chain, no cross-story constraints
                     -> large equivalence class.
  table              row-major reading order over a grid; strict. [complexity 2]

The generator renders a PNG (for running real VLMs on your Mac) and emits a
sidecar JSON with blocks + constraints. Rendering uses Pillow only.
"""

from __future__ import annotations

import json
import random
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from .partial_order import PartialOrder

# small bank of sentence fragments to compose unique-ish block text
_WORDS = (
    "system model token attention kernel latency cache gradient routing sparse "
    "decode encoder context quantize inference memory bandwidth throughput "
    "policy reward rollout entropy distill embedding retrieval pipeline schema "
    "vision document layout column sidebar footnote heading paragraph table"
).split()


def _sentence(rng: random.Random, k_words=12) -> str:
    ws = [rng.choice(_WORDS) for _ in range(k_words)]
    ws[0] = ws[0].capitalize()
    return " ".join(ws) + "."


def _para(rng: random.Random, n_sent=3) -> str:
    return " ".join(_sentence(rng, rng.randint(8, 16)) for _ in range(n_sent))


@dataclass
class Block:
    id: int
    text: str
    bbox: tuple   # (x0, y0, x1, y1)
    region: str   # logical region label


@dataclass
class SynthDoc:
    doc_id: str
    layout: str
    complexity: int
    blocks: list           # list[Block]
    partial_order: PartialOrder
    page_size: tuple = (1200, 1600)

    # ----- serialization -----
    def gt_texts(self):
        return [b.text for b in self.blocks]

    def to_json(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "layout": self.layout,
            "complexity": self.complexity,
            "page_size": list(self.page_size),
            "blocks": [
                {"id": b.id, "text": b.text, "bbox": list(b.bbox),
                 "region": b.region} for b in self.blocks
            ],
            "constraints": sorted([list(c) for c in self.partial_order.constraints]),
            "n_required_pairs": self.partial_order.num_required_pairs(),
        }


# ---------- layout builders ----------

def _build_single_column(rng, W, H):
    blocks, ids = [], []
    y = 80
    for i in range(6):
        txt = _para(rng, rng.randint(2, 4))
        blocks.append(Block(i, txt, (80, y, W - 80, y + 180), "body"))
        ids.append(i)
        y += 200
    P = PartialOrder(len(blocks)).chain(ids)
    return blocks, P


def _build_two_column(rng, W, H):
    mid = W // 2
    blocks = []
    left_ids, right_ids = [], []
    y = 80
    for i in range(4):
        blocks.append(Block(i, _para(rng, 2), (80, y, mid - 30, y + 220), "left"))
        left_ids.append(i)
        y += 240
    y = 80
    for k in range(4):
        i = 4 + k
        blocks.append(Block(i, _para(rng, 2), (mid + 30, y, W - 80, y + 220), "right"))
        right_ids.append(i)
        y += 240
    P = PartialOrder(len(blocks))
    P.chain(left_ids).chain(right_ids).precede_all(left_ids, right_ids)
    return blocks, P


def _build_body_with_sidebar(rng, W, H):
    """Two-column body + an independent sidebar box (genuinely unordered)."""
    mid = int(W * 0.62)
    blocks = []
    left_ids, right_ids, side_ids = [], [], []
    y = 80
    for i in range(3):
        blocks.append(Block(i, _para(rng, 2), (80, y, mid - 30, y + 240), "left"))
        left_ids.append(i); y += 260
    y = 80
    for k in range(3):
        i = 3 + k
        blocks.append(Block(i, _para(rng, 2), (mid + 30, y, W - 80, y + 220), "right"))
        right_ids.append(i); y += 240
    # sidebar pinned bottom-right, independent of body reading order
    si = 6
    blocks.append(Block(si, _para(rng, 2), (mid + 30, H - 360, W - 80, H - 80),
                        "sidebar"))
    side_ids.append(si)
    P = PartialOrder(len(blocks))
    P.chain(left_ids).chain(right_ids).precede_all(left_ids, right_ids)
    # NOTE: sidebar has NO cross constraints -> ambiguous placement (the point).
    return blocks, P


def _build_newsletter(rng, W, H):
    """Three independent stories; each an internal chain, no cross constraints."""
    blocks = []
    P_chains = []
    bid = 0
    col_x = [80, W // 3 + 20, 2 * W // 3 + 20]
    for c in range(3):
        chain_ids = []
        y = 80
        for _ in range(3):
            x0 = col_x[c]
            x1 = (col_x[c + 1] - 20) if c < 2 else (W - 80)
            blocks.append(Block(bid, _para(rng, 2), (x0, y, x1, y + 300),
                               f"story{c}"))
            chain_ids.append(bid); bid += 1; y += 320
        P_chains.append(chain_ids)
    P = PartialOrder(len(blocks))
    for ch in P_chains:
        P.chain(ch)   # no cross-story constraints -> large equivalence class
    return blocks, P


def _build_table(rng, W, H):
    """Row-major reading order over a 3x3 grid (strict total order)."""
    blocks, ids = [], []
    rows, cols = 3, 3
    cw = (W - 160) // cols
    ch = 180
    bid = 0
    for r in range(rows):
        for c in range(cols):
            x0 = 80 + c * cw
            y0 = 120 + r * (ch + 30)
            cell = " ".join(rng.choice(_WORDS) for _ in range(4))
            blocks.append(Block(bid, cell, (x0, y0, x0 + cw - 20, y0 + ch),
                               f"r{r}c{c}"))
            ids.append(bid); bid += 1
    P = PartialOrder(len(blocks)).chain(ids)  # row-major
    return blocks, P


LAYOUTS = {
    "single_column": (_build_single_column, 1),
    "table": (_build_table, 2),
    "two_column": (_build_two_column, 2),
    "body_with_sidebar": (_build_body_with_sidebar, 3),
    "newsletter": (_build_newsletter, 4),
}


def make_doc(layout: str, seed: int, page_size=(1200, 1600)) -> SynthDoc:
    if layout not in LAYOUTS:
        raise KeyError(f"unknown layout '{layout}'. choices: {list(LAYOUTS)}")
    builder, complexity = LAYOUTS[layout]
    rng = random.Random(seed)
    W, H = page_size
    blocks, P = builder(rng, W, H)
    if not P.is_consistent():
        raise RuntimeError("generated an inconsistent (cyclic) partial order")
    return SynthDoc(f"{layout}_{seed:04d}", layout, complexity, blocks, P, page_size)


# ---------- rendering ----------

def render_png(doc: SynthDoc, out_path) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    W, H = doc.page_size
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for b in doc.blocks:
        x0, y0, x1, y1 = b.bbox
        d.rectangle([x0, y0, x1, y1], outline=(210, 210, 210))
        # wrap text to box width (~ chars)
        approx_chars = max(8, int((x1 - x0) / 11))
        wrapped = textwrap.fill(b.text, width=approx_chars)
        d.multiline_text((x0 + 8, y0 + 8), wrapped, fill=(20, 20, 20),
                         font=font, spacing=4)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def build_suite(out_dir, per_layout: int = 3, render: bool = True,
                seed0: int = 0) -> list:
    """Generate the full benchmark suite. Returns list of (SynthDoc, png_path)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    s = seed0
    manifest = []
    for layout in LAYOUTS:
        for _ in range(per_layout):
            doc = make_doc(layout, s); s += 1
            png = None
            if render:
                png = render_png(doc, out_dir / f"{doc.doc_id}.png")
            (out_dir / f"{doc.doc_id}.json").write_text(
                json.dumps(doc.to_json(), indent=2))
            manifest.append(doc.doc_id)
            items.append((doc, png))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return items

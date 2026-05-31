"""Rasterize inputs (PDF pages or image files) into RGB PIL images.

PDFs are rendered with PyMuPDF (fitz) if available -- it's fast and has no
system deps -- otherwise we fall back to pdf2image (needs poppler).

DPI guidance for OCR on Apple Silicon:
  - 150 dpi: cheap, fine for clean digital PDFs.
  - 200-220 dpi: good default for scans / small fonts.
  - >300 dpi: diminishing returns; blows up vision-token count and RAM.
Optical-compression models (DeepSeek-OCR) tolerate higher dpi because they
compress vision tokens after perception -- push dpi up there, keep it modest
for dense-attention generalists.
"""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _render_pdf(path: Path, dpi: int):
    from PIL import Image  # noqa: F401  (ensures PIL present)

    # Preferred: PyMuPDF
    try:
        import fitz  # type: ignore  (PyMuPDF)
    except ImportError:
        fitz = None

    if fitz is not None:
        from PIL import Image

        pages = []
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        with fitz.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                pages.append(img)
        return pages

    # Fallback: pdf2image (requires poppler installed via brew)
    try:
        from pdf2image import convert_from_path  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "To read PDFs install one of:\n"
            "  pip install PyMuPDF        # recommended, no system deps\n"
            "  pip install pdf2image && brew install poppler\n"
            f"Import error: {e}"
        )
    return convert_from_path(str(path), dpi=dpi)


def load_pages(input_path: str | Path, dpi: int = 200, max_pages: int | None = None):
    """Return a list of RGB PIL.Image, one per page/image."""
    from PIL import Image

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".pdf":
        pages = _render_pdf(path, dpi)
    elif path.suffix.lower() in IMAGE_EXTS:
        pages = [Image.open(path).convert("RGB")]
    else:
        raise ValueError(
            f"Unsupported input '{path.suffix}'. Give a PDF or an image "
            f"({', '.join(sorted(IMAGE_EXTS))})."
        )

    if max_pages is not None:
        pages = pages[:max_pages]
    return pages


def save_pages_as_pngs(pages, out_dir: str | Path) -> list[Path]:
    """mlx-vlm's generate() takes image *paths*. Dump pages to temp PNGs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, page in enumerate(pages):
        p = out_dir / f"page_{i:03d}.png"
        page.save(p)
        paths.append(p)
    return paths

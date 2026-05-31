"""Model registry + loader for small specialist OCR / VLM models on MLX.

Why these models (see the frontier report): for *extraction* on real-world,
messy documents, compact domain-specialist VLMs beat huge generalists. We keep
a registry of small, low-active-param models that run comfortably on an M-series
Mac with unified memory.

The exact HF repo ids below are the MLX-converted community builds that are the
common path on Apple Silicon. They drift over time -- run `python -m toe
list-models` to print them, and if a repo 404s, search the HF Hub for an
`mlx-community/<name>` conversion or convert it yourself with `mlx_vlm.convert`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    key: str                      # short CLI alias
    repo: str                     # HF repo id (MLX-converted build)
    note: str                     # what it's good at
    default_prompt: str           # task prompt baked in
    # rough resident RAM at 4-bit, GiB -- guide for picking on your Mac
    approx_ram_gb: float = 4.0
    tags: tuple[str, ...] = field(default_factory=tuple)


# NOTE: repo ids are best-effort as of the report date. Verify with
# `list-models` and substitute an mlx-community conversion if one has 404'd.
REGISTRY: dict[str, ModelSpec] = {
    "deepseek-ocr2": ModelSpec(
        key="deepseek-ocr2",
        repo="mlx-community/DeepSeek-OCR-2",
        note="3B-MoE / ~570M active. Optical-compression OCR, strong layout "
             "preservation + Causal Visual Flow reading order. Best all-rounder.",
        default_prompt="Convert this document image to clean Markdown. "
                       "Preserve tables, headings and reading order. "
                       "Do not summarize; transcribe faithfully.",
        approx_ram_gb=4.0,
        tags=("ocr", "layout", "moe"),
    ),
    "paddleocr-vl": ModelSpec(
        key="paddleocr-vl",
        repo="mlx-community/PaddleOCR-VL-1.5",
        note="0.9B specialist. SOTA on OmniDocBench/Real5 for in-the-wild "
             "(skewed, blurred, photographed) docs. Tiny + robust.",
        default_prompt="OCR this document. Output Markdown with tables as "
                       "Markdown tables. Transcribe exactly, no summary.",
        approx_ram_gb=2.0,
        tags=("ocr", "specialist", "robust"),
    ),
    "qwen3-vl": ModelSpec(
        key="qwen3-vl",
        repo="mlx-community/Qwen3-VL-4B-Instruct-4bit",
        note="General multimodal + OCR. Use when you need reasoning over the "
             "doc (VQA, chart understanding), not just transcription.",
        default_prompt="Read the document image and answer the user's request. "
                       "When transcribing, output Markdown.",
        approx_ram_gb=3.0,
        tags=("vlm", "reasoning", "general"),
    ),
    "deepseek-vl2": ModelSpec(
        key="deepseek-vl2",
        repo="mlx-community/DeepSeek-VL2",
        note="27B MoE / 4.5B active. OCR + tables + charts + grounding combo. "
             "Heavier but still edge-feasible thanks to low active params.",
        default_prompt="Analyze the document image. Transcribe to Markdown and "
                       "preserve tables and structure.",
        approx_ram_gb=8.0,
        tags=("vlm", "moe", "grounding"),
    ),
}

DEFAULT_MODEL = "deepseek-ocr2"


def get_spec(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown model '{key}'. Known: {', '.join(REGISTRY)}. "
            f"Pass a raw HF repo id to override."
        )
    return REGISTRY[key]


def resolve_repo(model_arg: str) -> tuple[str, ModelSpec | None]:
    """Accept either a registry key or a raw HF repo id.

    Returns (repo_id, spec_or_None). If a raw repo id is given we still return a
    None spec so the caller can fall back to a generic prompt.
    """
    if model_arg in REGISTRY:
        spec = REGISTRY[model_arg]
        return spec.repo, spec
    # treat as a raw repo id ("org/name")
    return model_arg, None


def load_model(repo_id: str):
    """Load an MLX-VLM model + processor. Imported lazily so the rest of the
    package (registry listing, schema utils) works without mlx installed --
    handy for inspecting on non-Apple machines."""
    try:
        from mlx_vlm import load  # type: ignore
    except ImportError as e:  # pragma: no cover - environment dependent
        raise SystemExit(
            "mlx-vlm is not installed (Apple Silicon only).\n"
            "  pip install -U mlx-vlm\n"
            f"Original import error: {e}"
        )
    model, processor = load(repo_id)
    return model, processor

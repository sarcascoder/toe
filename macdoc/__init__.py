"""On-device document extraction toolkit for Apple Silicon (MLX-VLM).

Modules:
    models   - registry of small specialist OCR/VLM models + loader
    render   - rasterize PDFs/images into RGB page images
    extract  - run a VLM over pages, return markdown/text
    schema   - JSON-schema / pydantic constrained extraction with repair retry
    bench    - tok/s + peak-RAM benchmark harness
"""

__version__ = "0.1.0"

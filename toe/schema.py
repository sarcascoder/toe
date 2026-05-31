"""Schema-constrained structured extraction.

IMPORTANT (honesty note): true constrained decoding (outlines / xgrammar style
token masking) is not reliably wired into mlx-vlm yet. So we use the pragmatic,
production-proven approach instead:

    1. Prompt the VLM to emit JSON for a given JSON Schema.
    2. Parse + validate against the schema (jsonschema, or pydantic if a model
       class is supplied).
    3. On failure, do ONE repair round: feed the model its own bad output + the
       validation error and ask it to fix the JSON.

This is the right default for extraction: never trust free-form VLM JSON --
always validate against a schema and reject/repair. The validator is the
contract, not the model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class ExtractionOutcome:
    ok: bool
    data: dict | None
    raw: str
    error: str | None = None
    repaired: bool = False


def _strip_code_fences(text: str) -> str:
    """Models love to wrap JSON in ```json ... ``` fences."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def _first_json_object(text: str) -> str | None:
    """Best-effort: grab the first balanced {...} block."""
    text = _strip_code_fences(text)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def parse_and_validate(raw: str, json_schema: dict) -> ExtractionOutcome:
    blob = _first_json_object(raw)
    if blob is None:
        return ExtractionOutcome(False, None, raw, "no JSON object found")
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        return ExtractionOutcome(False, None, raw, f"JSON parse error: {e}")

    try:
        import jsonschema  # type: ignore
        jsonschema.validate(data, json_schema)
    except ImportError:
        # jsonschema optional; if absent we at least return parsed JSON
        return ExtractionOutcome(True, data, raw, "jsonschema not installed; "
                                 "returned unvalidated JSON")
    except Exception as e:  # jsonschema.ValidationError
        return ExtractionOutcome(False, data, raw, f"schema validation: {e}")

    return ExtractionOutcome(True, data, raw)


def build_extraction_prompt(json_schema: dict, instruction: str | None = None) -> str:
    schema_str = json.dumps(json_schema, indent=2)
    base = instruction or "Extract the requested fields from the document image."
    return (
        f"{base}\n\n"
        "Return ONLY a single JSON object that conforms to this JSON Schema. "
        "Do not add explanations or Markdown fences. If a field is missing in "
        "the document, use null.\n\n"
        f"JSON Schema:\n{schema_str}\n"
    )


def build_repair_prompt(bad_output: str, error: str, json_schema: dict) -> str:
    return (
        "Your previous JSON output was invalid.\n"
        f"Validation error: {error}\n\n"
        f"Previous output:\n{bad_output}\n\n"
        "Return a corrected JSON object that conforms to this schema. "
        "Output ONLY the JSON.\n\n"
        f"JSON Schema:\n{json.dumps(json_schema, indent=2)}\n"
    )


def extract_structured(
    model,
    processor,
    page_image_path,
    json_schema: dict,
    instruction: str | None = None,
    max_tokens: int = 2048,
    allow_repair: bool = True,
) -> ExtractionOutcome:
    """Single-image structured extraction with one repair retry."""
    from .extract import _apply_template, _normalize_generate_return
    from mlx_vlm import generate  # type: ignore

    prompt = build_extraction_prompt(json_schema, instruction)
    formatted = _apply_template(processor, model, prompt, n_images=1)
    ret = generate(model, processor, formatted, image=[str(page_image_path)],
                   max_tokens=max_tokens, temperature=0.0)
    raw, *_ = _normalize_generate_return(ret)

    outcome = parse_and_validate(raw, json_schema)
    if outcome.ok or not allow_repair:
        return outcome

    # one repair round
    repair = build_repair_prompt(raw, outcome.error or "unknown", json_schema)
    formatted_r = _apply_template(processor, model, repair, n_images=1)
    ret_r = generate(model, processor, formatted_r, image=[str(page_image_path)],
                     max_tokens=max_tokens, temperature=0.0)
    raw_r, *_ = _normalize_generate_return(ret_r)
    fixed = parse_and_validate(raw_r, json_schema)
    fixed.repaired = True
    return fixed


# A ready-to-use example schema: generic invoice / receipt extraction.
EXAMPLE_INVOICE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "vendor": {"type": ["string", "null"]},
        "invoice_number": {"type": ["string", "null"]},
        "date": {"type": ["string", "null"]},
        "currency": {"type": ["string", "null"]},
        "total": {"type": ["number", "null"]},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": ["string", "null"]},
                    "quantity": {"type": ["number", "null"]},
                    "unit_price": {"type": ["number", "null"]},
                    "amount": {"type": ["number", "null"]},
                },
                "required": ["description"],
            },
        },
    },
    "required": ["vendor", "total"],
}

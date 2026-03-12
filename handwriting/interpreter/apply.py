"""Apply shorthand dictionary and regex rules to produce interpreted text.

CLI usage::

    python -m handwriting.interpreter.apply --text "pt -> f/u ↑ dose"
    # → "patient to follow-up increase dose"

The interpretation pipeline:
1. Normalize whitespace (NBSP → space, multiple spaces → single space).
2. Apply all token replacements from ``shorthand.yml`` (longer keys first to
   prevent partial-match issues).
3. Tidy up punctuation spacing.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict

import yaml


_DEFAULT_DICT = Path(__file__).with_name("shorthand.yml")


def load_dict(path: Path) -> Dict[str, str]:
    """Load ``shorthand.yml`` and return a ``{token: expansion}`` mapping."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"shorthand.yml must be a YAML mapping, got {type(raw)}")
    return {str(k): str(v) for k, v in raw.items()}


def interpret(text: str, mapping: Dict[str, str]) -> str:
    """
    Expand *text* using *mapping* and regex cleanup rules.

    Parameters
    ----------
    text:
        Raw (literal) OCR output string.
    mapping:
        Token → expansion dictionary (e.g. from :func:`load_dict`).

    Returns
    -------
    Interpreted plain-text string.
    """
    # 1. Normalise whitespace
    t = text.replace("\u00a0", " ")            # non-breaking space
    t = re.sub(r"[ \t]+", " ", t).strip()

    # 2. Token replacements — longer keys first to avoid partial matches.
    # For purely alphabetic tokens, use word-boundary matching so that e.g.
    # "pt" does not corrupt "option".  Non-alphabetic tokens (arrows, symbols)
    # are replaced as plain substrings.
    for key in sorted(mapping, key=len, reverse=True):
        if re.match(r"^[A-Za-z/]+$", key):
            t = re.sub(r"(?<!\w)" + re.escape(key) + r"(?!\w)", mapping[key], t)
        else:
            t = t.replace(key, mapping[key])

    # 3. Tidy punctuation spacing
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)    # no space before punctuation
    t = re.sub(r"\s{2,}", " ", t).strip()

    return t


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Expand shorthand tokens in TEXT using shorthand.yml."
    )
    ap.add_argument("--text", required=True, help="Literal text to interpret.")
    ap.add_argument(
        "--dict",
        default=str(_DEFAULT_DICT),
        help="Path to shorthand YAML dictionary (default: built-in shorthand.yml).",
    )
    args = ap.parse_args()

    mapping = load_dict(Path(args.dict))
    print(interpret(args.text, mapping))


if __name__ == "__main__":
    main()

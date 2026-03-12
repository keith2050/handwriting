"""One-shot pipeline: segment page → OCR → interpret → write outputs.

CLI usage::

    python -m handwriting.export_text \\
        --page  data/pages/page1.jpg \\
        --checkpoint checkpoints/best.pt \\
        --out   outputs/page1/

Outputs
-------
``<out>/<page_id>_literal.md``
    Raw OCR transcription, one line per paragraph.
``<out>/<page_id>_interpreted.md``
    Expanded / interpreted version using the shorthand dictionary.

When no checkpoint is supplied the literal column is left blank with a
placeholder, and you are asked to use the Streamlit app to fill it in.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .segmentation import segment_lines
from .interpreter.apply import interpret, load_dict

_DEFAULT_DICT = Path(__file__).with_name("interpreter") / "shorthand.yml"


def export_page(
    page_path: Path,
    out_dir: Path,
    page_id: str,
    checkpoint: Path | None = None,
) -> None:
    """Run the full pipeline for a single page and write markdown outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines_dir = out_dir / "lines"

    img = cv2.imread(str(page_path))
    if img is None:
        raise ValueError(f"Cannot read image: {page_path}")

    line_boxes = segment_lines(img, lines_dir, page_id)

    if checkpoint is not None and checkpoint.exists():
        from .ocr.infer_ocr import predict_lines

        predictions = predict_lines(
            [lb.image_path for lb in line_boxes],
            str(checkpoint),
        )
    else:
        predictions = ["" for _ in line_boxes]

    mapping = load_dict(_DEFAULT_DICT)

    literal_lines: list[str] = []
    interpreted_lines: list[str] = []
    for lb, pred in zip(line_boxes, predictions):
        note = " *(margin)*" if lb.is_margin else ""
        literal_lines.append(f"{pred}{note}")
        interpreted_lines.append(f"{interpret(pred, mapping)}{note}")

    lit_text = "\n\n".join(literal_lines)
    int_text = "\n\n".join(interpreted_lines)

    (out_dir / f"{page_id}_literal.md").write_text(
        f"# {page_id} — literal\n\n{lit_text}\n", encoding="utf-8"
    )
    (out_dir / f"{page_id}_interpreted.md").write_text(
        f"# {page_id} — interpreted\n\n{int_text}\n", encoding="utf-8"
    )
    print(f"Wrote outputs to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end: segment page → OCR → interpret → markdown."
    )
    ap.add_argument("--page", required=True, help="Path to page image")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Path to OCR model checkpoint (.pt).  Omit to skip OCR.",
    )
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument(
        "--page_id",
        default=None,
        help="Page ID (defaults to image stem)",
    )
    args = ap.parse_args()

    page_path = Path(args.page)
    page_id = args.page_id or page_path.stem
    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    export_page(page_path, Path(args.out), page_id, checkpoint)


if __name__ == "__main__":
    main()

"""Page-to-line segmentation.

Segments a full handwriting page image into ordered line crops and saves them
together with a ``meta.json`` containing bounding boxes and reading order.

CLI usage::

    python -m handwriting.segmentation \\
        --input  data/pages/page1.jpg \\
        --out_dir data/lines/page1 \\
        --page_id page1

Margin notes (text blobs that lie outside the main column) are detected but
emitted *after* the main-column lines in reading order.  Their ``line_id``
values continue the same counter so the numbering remains unique.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from . import normalize_path as _normalize_path


@dataclass
class LineBox:
    line_id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h (in deskewed page coords)
    image_path: str
    is_margin: bool = False


def _main_column_bounds(bw: np.ndarray) -> Tuple[int, int]:
    """
    Estimate left/right bounds of the main text column via x-projection.

    Returns ``(x_left, x_right)`` pixel columns.  Falls back to full width
    when the page appears to be a single wide column.
    """
    h, w = bw.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))
    dil = cv2.dilate(bw, kernel, iterations=2)

    xsum = dil.sum(axis=0).astype(np.float32)
    max_val = xsum.max()
    if max_val == 0:
        return 0, w

    xsum_norm = xsum / max_val
    mask = xsum_norm > 0.10

    if mask.sum() < w * 0.15:
        return 0, w

    idxs = np.where(mask)[0]
    return int(idxs[0]), int(idxs[-1])


def segment_lines(
    page_bgr: np.ndarray,
    out_dir: Path,
    page_id: str,
) -> List[LineBox]:
    """
    Segment *page_bgr* into horizontal line crops.

    Crops are saved as PNG files under *out_dir*.  A ``meta.json`` file is
    also written there with the bounding boxes and reading order.

    Parameters
    ----------
    page_bgr:
        Full-page BGR image as loaded by ``cv2.imread``.
    out_dir:
        Directory where line crops and ``meta.json`` are written.
    page_id:
        Identifier string used in ``meta.json`` and file names.

    Returns
    -------
    List of :class:`LineBox` sorted in reading order (top→bottom for the main
    column, then margin notes if any).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prep = preprocess_page(page_bgr)
    img_bgr = prep["img_bgr"]
    bw = prep["bw"]  # ink=255

    x_left, x_right = _main_column_bounds(bw)

    # --- dilate to merge characters into line blobs ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 5))
    dil = cv2.dilate(bw, kernel, iterations=2)

    # split into main column and margin regions
    main_mask = np.zeros_like(dil)
    main_mask[:, x_left:x_right] = 255
    margin_mask = cv2.bitwise_not(main_mask)

    dil_main = cv2.bitwise_and(dil, main_mask)
    dil_margin = cv2.bitwise_and(dil, margin_mask)

    def _extract_boxes(blob_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(
            blob_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < 80 or h < 15:
                continue
            boxes.append((x, y, w, h))
        return sorted(boxes, key=lambda b: b[1])

    main_boxes = _extract_boxes(dil_main)
    margin_boxes = _extract_boxes(dil_margin)

    page_h, page_w = img_bgr.shape[:2]
    line_boxes: List[LineBox] = []

    def _save_crop(
        boxes: List[Tuple[int, int, int, int]],
        is_margin: bool,
        start_id: int,
    ) -> int:
        line_id = start_id
        for x, y, w, h in boxes:
            pad_y = max(4, int(0.15 * h))
            pad_x = max(2, int(0.02 * w))
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(page_w, x + w + pad_x)
            y1 = min(page_h, y + h + pad_y)

            crop = img_bgr[y0:y1, x0:x1]
            fname = f"line_{line_id:04d}.png"
            fpath = out_dir / fname
            cv2.imwrite(str(fpath), crop)

            line_boxes.append(
                LineBox(
                    line_id=line_id,
                    bbox=(x0, y0, x1 - x0, y1 - y0),
                    image_path=str(fpath).replace("\\", "/"),
                    is_margin=is_margin,
                )
            )
            line_id += 1
        return line_id

    next_id = _save_crop(main_boxes, is_margin=False, start_id=1)
    _save_crop(margin_boxes, is_margin=True, start_id=next_id)

    meta = {
        "page_id": page_id,
        "num_lines": len(line_boxes),
        "lines": [asdict(lb) for lb in line_boxes],
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return line_boxes


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Segment a handwriting page image into line crops."
    )
    ap.add_argument("--input", required=True, help="Path to page image (jpg/png)")
    ap.add_argument(
        "--out_dir", required=True, help="Output directory for line crops"
    )
    ap.add_argument(
        "--page_id", required=True, help="Page ID (folder name / label key)"
    )
    args = ap.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.input}")

    lines = segment_lines(img, Path(args.out_dir), args.page_id)
    print(f"Segmented {len(lines)} line(s) → {args.out_dir}")


if __name__ == "__main__":
    main()

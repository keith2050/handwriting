"""Run OCR inference on a directory of line-crop images.

CLI usage::

    python -m handwriting.ocr.infer_ocr \\
        --checkpoint checkpoints/best.pt \\
        --input_dir  data/lines/page1 \\
        --out        outputs/page1_literal.txt

The output file has one line per input image (sorted by filename).
Each line is ``<filename>\\t<predicted_text>\\t<confidence>``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from .model import CRNN
from .train_ocr import decode


def _load_checkpoint(checkpoint_path: str) -> Tuple[CRNN, dict, str]:
    """Load a checkpoint produced by ``train_ocr.py``.

    Returns ``(model, checkpoint_dict, device_str)``.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device)

    alphabet = ckpt["alphabet"]
    img_height = ckpt.get("img_height", 64)
    cnn_channels = tuple(ckpt.get("cnn_channels", [32, 64, 128]))
    lstm_hidden = ckpt.get("lstm_hidden", 256)

    model = CRNN(
        alphabet=alphabet,
        img_height=img_height,
        cnn_channels=cnn_channels,
        lstm_hidden=lstm_hidden,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt, device


def _preprocess_image(
    img_path: str,
    img_height: int = 64,
    max_width: int = 1024,
) -> torch.Tensor:
    """Load and resize a single line image to a (1, 1, H, W) tensor."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    h, w = img.shape
    new_w = min(int(w * img_height / h), max_width)
    img = cv2.resize(img, (new_w, img_height))
    pad = np.full((img_height, max_width), 255, dtype=np.uint8)
    pad[:, :new_w] = img
    t = torch.from_numpy(pad).float() / 255.0
    return t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def predict_lines(
    image_paths: List[str],
    checkpoint_path: str,
) -> List[str]:
    """Return predicted literal text for each image path.

    Parameters
    ----------
    image_paths:
        Ordered list of line-crop PNG paths.
    checkpoint_path:
        Path to a ``best.pt`` / ``last.pt`` checkpoint.

    Returns
    -------
    List of predicted strings in the same order as *image_paths*.
    """
    if not image_paths:
        return []

    model, ckpt, device = _load_checkpoint(checkpoint_path)
    alphabet = ckpt["alphabet"]
    idx_to_char = {i + 1: c for i, c in enumerate(alphabet)}
    img_height = ckpt.get("img_height", 64)
    max_width = ckpt.get("max_width", 1024)

    results = []
    with torch.no_grad():
        for path in image_paths:
            try:
                img_t = _preprocess_image(path, img_height, max_width).to(device)
                log_probs = model(img_t)          # (T, 1, C)
                pred_idx = log_probs[:, 0, :].argmax(dim=1).tolist()
                text = decode(pred_idx, idx_to_char, blank_idx=0)
            except Exception as exc:
                text = f"[ERROR: {exc}]"
            results.append(text)
    return results


def predict_with_confidence(
    image_paths: List[str],
    checkpoint_path: str,
) -> List[Tuple[str, float]]:
    """Like :func:`predict_lines` but also returns a mean log-probability as
    a proxy confidence score (higher is better, range roughly [-inf, 0]).
    """
    if not image_paths:
        return []

    model, ckpt, device = _load_checkpoint(checkpoint_path)
    alphabet = ckpt["alphabet"]
    idx_to_char = {i + 1: c for i, c in enumerate(alphabet)}
    img_height = ckpt.get("img_height", 64)
    max_width = ckpt.get("max_width", 1024)

    results = []
    with torch.no_grad():
        for path in image_paths:
            try:
                img_t = _preprocess_image(path, img_height, max_width).to(device)
                log_probs = model(img_t)  # (T, 1, C)
                lp = log_probs[:, 0, :]   # (T, C)
                max_lp = lp.max(dim=1).values
                confidence = float(max_lp.mean().exp())  # convert to probability
                pred_idx = lp.argmax(dim=1).tolist()
                text = decode(pred_idx, idx_to_char, blank_idx=0)
            except Exception as exc:
                text = f"[ERROR: {exc}]"
                confidence = 0.0
            results.append((text, confidence))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="OCR inference on line-crop images.")
    ap.add_argument(
        "--checkpoint", required=True, help="Path to best.pt checkpoint."
    )
    ap.add_argument(
        "--input_dir", required=True, help="Directory containing line PNG crops."
    )
    ap.add_argument(
        "--out", required=True, help="Output text file (TSV: filename, text, conf)."
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    image_paths = sorted(input_dir.glob("line_*.png"))
    if not image_paths:
        raise SystemExit(f"No line_*.png files found in {input_dir}")

    pairs = predict_with_confidence(
        [str(p) for p in image_paths], args.checkpoint
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("filename\tprediction\tconfidence\n")
        for img_path, (text, conf) in zip(image_paths, pairs):
            fh.write(f"{img_path.name}\t{text}\t{conf:.4f}\n")

    print(f"Wrote {len(pairs)} prediction(s) to {args.out}")


if __name__ == "__main__":
    main()

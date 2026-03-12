"""Train (or fine-tune) the CRNN OCR model on labeled line crops.

CLI usage::

    python -m handwriting.ocr.train_ocr \\
        --labels data/labels.csv \\
        --out    checkpoints/

Training notes
--------------
* Reads ``literal`` text from ``labels.csv``; ``image_path`` column points to
  the line-crop PNGs.
* Resizes every crop to ``--img_height`` (default 64 px) while preserving
  aspect ratio; then pads/truncates width to ``--max_width`` (default 1024).
* CTC loss with blank index 0.
* Reports Character Error Rate (CER) on a held-out validation split.
* Saves ``best.pt`` (best validation CER) and ``last.pt`` each epoch.
* Supports ``--accumulate`` for gradient accumulation (useful on low-VRAM
  hardware like the GT 1030).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .model import build_model, CRNN


# ---------------------------------------------------------------------------
# Character set helpers
# ---------------------------------------------------------------------------

def build_alphabet(labels: List[str]) -> List[str]:
    """Return a sorted list of unique characters appearing in *labels*."""
    chars = set()
    for t in labels:
        chars.update(t)
    return sorted(chars)


def encode(text: str, char_to_idx: dict) -> List[int]:
    return [char_to_idx[c] for c in text if c in char_to_idx]


def decode(indices: List[int], idx_to_char: dict, blank_idx: int = 0) -> str:
    """CTC greedy decode: collapse repeats then remove blanks."""
    prev = None
    out = []
    for idx in indices:
        if idx != prev:
            if idx != blank_idx:
                out.append(idx_to_char.get(idx, ""))
            prev = idx
    return "".join(out)


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate (edit distance / len(reference))."""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    # Simple Levenshtein
    r, h = list(reference), list(hypothesis)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / len(r)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LineDataset(Dataset):
    def __init__(
        self,
        records: List[dict],
        char_to_idx: dict,
        img_height: int = 64,
        max_width: int = 1024,
    ) -> None:
        self.records = records
        self.char_to_idx = char_to_idx
        self.img_height = img_height
        self.max_width = max_width

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        h, w = img.shape
        new_w = min(int(w * self.img_height / h), self.max_width)
        img = cv2.resize(img, (new_w, self.img_height))
        # Pad to max_width
        pad = np.full((self.img_height, self.max_width), 255, dtype=np.uint8)
        pad[:, :new_w] = img
        return pad

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        rec = self.records[idx]
        img = self._load_image(rec["image_path"])
        img_t = torch.from_numpy(img).float() / 255.0
        img_t = img_t.unsqueeze(0)  # (1, H, W)

        label = encode(rec["literal"], self.char_to_idx)
        label_t = torch.tensor(label, dtype=torch.long)
        return img_t, label_t, len(label)


def collate_fn(batch):
    imgs, labels, label_lens = zip(*batch)
    imgs = torch.stack(imgs)
    label_lens = torch.tensor(label_lens, dtype=torch.long)
    labels_concat = torch.cat(labels)
    return imgs, labels_concat, label_lens


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    labels_csv: Path,
    out_dir: Path,
    img_height: int = 64,
    max_width: int = 1024,
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-3,
    val_split: float = 0.1,
    accumulate: int = 1,
    device: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_csv, dtype=str).fillna("")
    # Filter rows with non-empty literal and valid image_path
    df = df[df["literal"].str.strip() != ""]
    if df.empty:
        raise SystemExit(
            "No labeled rows found in labels.csv.  "
            "Add literal transcriptions via the Streamlit app first."
        )

    records = df[["image_path", "literal"]].to_dict("records")
    alphabet = build_alphabet([r["literal"] for r in records])
    char_to_idx = {c: i + 1 for i, c in enumerate(alphabet)}  # 0 = blank
    idx_to_char = {v: k for k, v in char_to_idx.items()}

    # Save alphabet for inference
    meta = {"alphabet": alphabet, "img_height": img_height, "max_width": max_width}
    (out_dir / "alphabet.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    random.shuffle(records)
    split = max(1, int(len(records) * val_split))
    val_records, train_records = records[:split], records[split:]

    train_ds = LineDataset(train_records, char_to_idx, img_height, max_width)
    val_ds = LineDataset(val_records, char_to_idx, img_height, max_width)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(alphabet, img_height=img_height, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    best_cer = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, (imgs, labels, label_lens) in enumerate(train_loader, 1):
            imgs = imgs.to(device)
            labels = labels.to(device)

            log_probs = model(imgs)  # (T, B, C)
            T, B, _ = log_probs.shape
            input_lens = torch.full((B,), T, dtype=torch.long)

            loss = ctc_loss(log_probs.cpu(), labels.cpu(), input_lens, label_lens)
            (loss / accumulate).backward()
            total_loss += loss.item()

            if step % accumulate == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()

        # Final step flush
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()

        # Validation CER
        model.eval()
        total_cer = 0.0
        with torch.no_grad():
            for imgs, labels_flat, label_lens in val_loader:
                imgs = imgs.to(device)
                log_probs = model(imgs)  # (T, 1, C)
                pred_idx = log_probs[:, 0, :].argmax(dim=1).tolist()
                pred_str = decode(pred_idx, idx_to_char, blank_idx=0)

                # Reconstruct reference from flat tensor
                ref_idx = labels_flat.tolist()
                ref_str = "".join(idx_to_char.get(i, "") for i in ref_idx)
                total_cer += cer(ref_str, pred_str)

        avg_cer = total_cer / max(len(val_ds), 1)
        avg_loss = total_loss / max(len(train_loader), 1)
        print(
            f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_CER={avg_cer:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "alphabet": alphabet,
            "img_height": img_height,
            "max_width": max_width,
            "val_cer": avg_cer,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if avg_cer < best_cer:
            best_cer = avg_cer
            torch.save(checkpoint, out_dir / "best.pt")
            print(f"  ✓ Saved best checkpoint (CER={best_cer:.4f})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CRNN OCR on labeled line crops.")
    ap.add_argument("--labels", required=True, help="Path to labels.csv")
    ap.add_argument("--out", required=True, help="Output directory for checkpoints")
    ap.add_argument("--img_height", type=int, default=64)
    ap.add_argument("--max_width", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument(
        "--accumulate",
        type=int,
        default=1,
        help="Gradient accumulation steps (use 4–8 for low-VRAM GPU).",
    )
    ap.add_argument("--device", default=None, help="'cuda' or 'cpu' (auto-detect)")
    args = ap.parse_args()

    train(
        labels_csv=Path(args.labels),
        out_dir=Path(args.out),
        img_height=args.img_height,
        max_width=args.max_width,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        accumulate=args.accumulate,
        device=args.device,
    )


if __name__ == "__main__":
    main()

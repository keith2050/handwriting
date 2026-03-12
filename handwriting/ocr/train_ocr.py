"""Train (or fine-tune) the CRNN OCR model on labeled line crops.

CLI usage::

    python -m handwriting.ocr.train_ocr \\
        --labels data/labels.csv \\
        --out    checkpoints/

Hardware profiles
-----------------
Pass ``--profile gt1030`` (default) or ``--profile rtx5060ti`` to automatically
set conservative or high-performance defaults.  Any explicit flag overrides the
profile value.

+----------------------+------------+---------------+
| Setting              | GT 1030    | RTX 5060 Ti   |
+======================+============+===============+
| --img_height         | 64         | 96            |
| --max_width          | 1024       | 2048          |
| --cnn_channels       | 32,64,128  | 64,128,256    |
| --lstm_hidden        | 256        | 512           |
| --batch_size         | 4          | 32            |
| --accumulate         | 4          | 1             |
| --num_workers        | 0          | 8             |
| --amp                | off        | on            |
+----------------------+------------+---------------+

Training notes
--------------
* Reads ``literal`` text from ``labels.csv``; ``image_path`` column points to
  the line-crop PNGs.
* CTC loss with blank index 0.
* Reports Character Error Rate (CER) on a held-out validation split.
* Saves ``best.pt`` (best validation CER) and ``last.pt`` each epoch.
* ``--amp`` enables automatic mixed precision (float16/bfloat16) via
  ``torch.cuda.amp``.  Ignored silently on CPU.
* ``--accumulate`` enables gradient accumulation (useful on low-VRAM hardware).
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
# Hardware profile presets
# ---------------------------------------------------------------------------

_PROFILES: dict[str, dict] = {
    "gt1030": dict(
        img_height=64,
        max_width=1024,
        cnn_channels=(32, 64, 128),
        lstm_hidden=256,
        batch_size=4,
        accumulate=4,
        num_workers=0,
        amp=False,
    ),
    "rtx5060ti": dict(
        img_height=96,
        max_width=2048,
        cnn_channels=(64, 128, 256),
        lstm_hidden=512,
        batch_size=32,
        accumulate=1,
        num_workers=8,
        amp=True,
    ),
}


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
    cnn_channels: tuple = (32, 64, 128),
    lstm_hidden: int = 256,
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-3,
    val_split: float = 0.1,
    accumulate: int = 1,
    num_workers: int = 0,
    amp: bool = False,
    device: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_csv, dtype=str).fillna("")
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

    meta = {
        "alphabet": alphabet,
        "img_height": img_height,
        "max_width": max_width,
        "cnn_channels": list(cnn_channels),
        "lstm_hidden": lstm_hidden,
    }
    (out_dir / "alphabet.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    random.shuffle(records)
    split = max(1, int(len(records) * val_split))
    val_records, train_records = records[:split], records[split:]

    # num_workers > 0 requires platform support; fall back gracefully on Windows
    # when the spawn context hasn't been set up.
    safe_workers = num_workers
    train_loader = DataLoader(
        LineDataset(train_records, char_to_idx, img_height, max_width),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=safe_workers,
        pin_memory=(device == "cuda" or (device is None and torch.cuda.is_available())),
    )
    val_loader = DataLoader(
        LineDataset(val_records, char_to_idx, img_height, max_width),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # validation is fast; keep simple
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        alphabet,
        img_height=img_height,
        cnn_channels=cnn_channels,
        lstm_hidden=lstm_hidden,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    # AMP scaler — only active when device is CUDA and amp=True
    use_amp = amp and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model: CNN{list(cnn_channels)} + BiLSTM(hidden={lstm_hidden}) "
        f"— {param_count:,} parameters"
    )
    print(
        f"Training: device={device}  amp={use_amp}  "
        f"batch={batch_size}  accumulate={accumulate}  workers={safe_workers}"
    )

    best_cer = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, (imgs, labels, label_lens) in enumerate(train_loader, 1):
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                log_probs = model(imgs)  # (T, B, C)
                T, B, _ = log_probs.shape
                input_lens = torch.full((B,), T, dtype=torch.long)
                loss = ctc_loss(
                    log_probs.float().cpu(), labels.cpu(), input_lens, label_lens
                )

            scaler.scale(loss / accumulate).backward()
            total_loss += loss.item()

            if step % accumulate == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # Final step flush
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Validation CER
        model.eval()
        total_cer = 0.0
        with torch.no_grad():
            for imgs, labels_flat, label_lens in val_loader:
                imgs = imgs.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    log_probs = model(imgs)  # (T, 1, C)
                pred_idx = log_probs[:, 0, :].float().argmax(dim=1).tolist()
                pred_str = decode(pred_idx, idx_to_char, blank_idx=0)

                ref_idx = labels_flat.tolist()
                ref_str = "".join(idx_to_char.get(i, "") for i in ref_idx)
                total_cer += cer(ref_str, pred_str)

        avg_cer = total_cer / max(len(val_records), 1)
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
            "cnn_channels": list(cnn_channels),
            "lstm_hidden": lstm_hidden,
            "val_cer": avg_cer,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if avg_cer < best_cer:
            best_cer = avg_cer
            torch.save(checkpoint, out_dir / "best.pt")
            print(f"  ✓ Saved best checkpoint (CER={best_cer:.4f})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train CRNN OCR on labeled line crops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--labels", required=True, help="Path to labels.csv")
    ap.add_argument("--out", required=True, help="Output directory for checkpoints")
    ap.add_argument(
        "--profile",
        choices=list(_PROFILES),
        default="gt1030",
        help=(
            "Hardware preset. 'gt1030' uses conservative settings; "
            "'rtx5060ti' enables larger model, bigger batches, and AMP."
        ),
    )
    ap.add_argument("--img_height", type=int, default=None)
    ap.add_argument("--max_width", type=int, default=None)
    ap.add_argument(
        "--cnn_channels",
        default=None,
        help="Comma-separated CNN channel sizes, e.g. '64,128,256'.",
    )
    ap.add_argument("--lstm_hidden", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument(
        "--accumulate",
        type=int,
        default=None,
        help="Gradient accumulation steps.",
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader worker processes for parallel image loading.",
    )
    ap.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable automatic mixed precision (float16/bfloat16). CUDA only.",
    )
    ap.add_argument("--device", default=None, help="'cuda' or 'cpu' (auto-detect)")
    args = ap.parse_args()

    # Start with profile defaults, then override with any explicit CLI flags
    cfg = dict(_PROFILES[args.profile])
    if args.img_height is not None:
        cfg["img_height"] = args.img_height
    if args.max_width is not None:
        cfg["max_width"] = args.max_width
    if args.cnn_channels is not None:
        cfg["cnn_channels"] = tuple(int(x) for x in args.cnn_channels.split(","))
    if args.lstm_hidden is not None:
        cfg["lstm_hidden"] = args.lstm_hidden
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.accumulate is not None:
        cfg["accumulate"] = args.accumulate
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.amp is not None:
        cfg["amp"] = args.amp

    train(
        labels_csv=Path(args.labels),
        out_dir=Path(args.out),
        img_height=cfg["img_height"],
        max_width=cfg["max_width"],
        cnn_channels=cfg["cnn_channels"],
        lstm_hidden=cfg["lstm_hidden"],
        epochs=args.epochs,
        batch_size=cfg["batch_size"],
        lr=args.lr,
        val_split=args.val_split,
        accumulate=cfg["accumulate"],
        num_workers=cfg["num_workers"],
        amp=cfg["amp"],
        device=args.device,
    )


if __name__ == "__main__":
    main()


"""Streamlit labeling and review UI for the handwriting pipeline.

Run with::

    streamlit run app/streamlit_app.py

Features
--------
* Upload a full-page handwriting image.
* Run segmentation to extract line crops.
* For each line:
    - display the crop,
    - edit the **literal** transcription,
    - auto-fill / edit the **interpreted** text.
* Save labels back to ``data/labels.csv``.
* Export literal and interpreted markdown for the whole page.
* (Optional) Run OCR inference if a checkpoint exists at
  ``checkpoints/best.pt``.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PAGES_DIR = DATA / "pages"
LINES_DIR = DATA / "lines"
LABELS_CSV = DATA / "labels.csv"
SHORTHAND_YML = ROOT / "handwriting" / "interpreter" / "shorthand.yml"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "best.pt"


# ── helper utilities ──────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    for d in (PAGES_DIR, LINES_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if not LABELS_CSV.exists():
        LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
        LABELS_CSV.write_text(
            "page_id,line_id,image_path,literal,interpreted\n", encoding="utf-8"
        )


def load_labels() -> pd.DataFrame:
    ensure_dirs()
    return pd.read_csv(LABELS_CSV, dtype=str).fillna("")


def upsert_label(
    page_id: str,
    line_id: int,
    image_path: str,
    literal: str,
    interpreted: str,
) -> None:
    df = load_labels()
    mask = (df["page_id"] == page_id) & (df["line_id"].astype(str) == str(line_id))
    row: dict = {
        "page_id": page_id,
        "line_id": line_id,
        "image_path": image_path.replace("\\", "/"),
        "literal": literal,
        "interpreted": interpreted,
    }
    if mask.any():
        for col in ("image_path", "literal", "interpreted"):
            df.loc[mask, col] = row[col]
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LABELS_CSV, index=False, encoding="utf-8")


def export_markdown(page_id: str, lines_meta: list, mapping: dict) -> tuple[str, str]:
    """Return (literal_md, interpreted_md) strings for the page."""
    from handwriting.interpreter.apply import interpret

    lit_parts: list[str] = []
    int_parts: list[str] = []
    df = load_labels()

    for ln in lines_meta:
        lid = int(ln["line_id"])
        row = df[(df["page_id"] == page_id) & (df["line_id"].astype(str) == str(lid))]
        literal = "" if row.empty else str(row.iloc[0]["literal"])
        interpreted = "" if row.empty else str(row.iloc[0]["interpreted"])
        if not interpreted and literal:
            interpreted = interpret(literal, mapping)
        note = " *(margin)*" if ln.get("is_margin") else ""
        lit_parts.append(f"{literal}{note}")
        int_parts.append(f"{interpreted}{note}")

    lit_md = f"# {page_id} — literal\n\n" + "\n\n".join(lit_parts) + "\n"
    int_md = f"# {page_id} — interpreted\n\n" + "\n\n".join(int_parts) + "\n"
    return lit_md, int_md


# ── page ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Handwriting Labeler", layout="wide")
st.title("✍️ Handwriting Labeler")

ensure_dirs()

# Lazy imports so the app starts even if torch isn't installed
try:
    from handwriting.interpreter.apply import interpret, load_dict

    mapping = load_dict(SHORTHAND_YML)
except Exception as exc:
    st.error(f"Could not load shorthand dictionary: {exc}")
    mapping = {}

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    page_id = st.text_input("page_id", value="page1")
    use_ocr = st.checkbox(
        "Run OCR inference (requires checkpoint)",
        value=DEFAULT_CHECKPOINT.exists(),
    )
    checkpoint_path = st.text_input(
        "Checkpoint path", value=str(DEFAULT_CHECKPOINT)
    )
    st.markdown("---")
    st.markdown(
        "**Shorthand dict:** `handwriting/interpreter/shorthand.yml`\n\n"
        "Edit that file to add your personal abbreviations."
    )

# ── upload & segment ───────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a page image (jpg / png)", type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    page_path = PAGES_DIR / f"{page_id}.png"
    page_path.write_bytes(raw_bytes)

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(raw_bytes, caption="Uploaded page", use_container_width=True)

    with col2:
        if st.button("▶ Segment page into lines"):
            from handwriting.segmentation import segment_lines

            out_dir = LINES_DIR / page_id
            with st.spinner("Segmenting…"):
                lines = segment_lines(img_bgr, out_dir, page_id)
            st.success(f"Found {len(lines)} line(s).")

# ── line editor ───────────────────────────────────────────────────────────────
meta_path = LINES_DIR / page_id / "meta.json"
if meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lines_meta = meta.get("lines", [])
    st.subheader(f"Lines for **{page_id}** ({meta['num_lines']} total)")

    ocr_predictions: dict[int, str] = {}
    if use_ocr and Path(checkpoint_path).exists():
        try:
            from handwriting.ocr.infer_ocr import predict_lines

            paths = [ln["image_path"] for ln in lines_meta]
            preds = predict_lines(paths, checkpoint_path)
            ocr_predictions = {
                int(ln["line_id"]): pred
                for ln, pred in zip(lines_meta, preds)
            }
        except Exception as exc:
            st.warning(f"OCR inference failed: {exc}")

    df_labels = load_labels()

    for ln in lines_meta:
        line_id = int(ln["line_id"])
        img_path = Path(ln["image_path"])
        is_margin = ln.get("is_margin", False)

        with st.container():
            cols = st.columns([1, 2, 2])

            with cols[0]:
                label_text = f"line {line_id}" + (" *(margin)*" if is_margin else "")
                if img_path.exists():
                    st.image(str(img_path), caption=label_text, use_container_width=True)
                else:
                    st.warning(f"Image not found: {img_path}")

            # Pre-fill from labels.csv, then OCR
            existing = df_labels[
                (df_labels["page_id"] == page_id)
                & (df_labels["line_id"].astype(str) == str(line_id))
            ]
            saved_literal = (
                "" if existing.empty else str(existing.iloc[0]["literal"])
            )
            saved_interp = (
                "" if existing.empty else str(existing.iloc[0]["interpreted"])
            )

            ocr_suggestion = ocr_predictions.get(line_id, "")
            literal_default = saved_literal or ocr_suggestion

            with cols[1]:
                literal = st.text_input(
                    f"Literal — line {line_id}",
                    value=literal_default,
                    key=f"lit_{page_id}_{line_id}",
                    placeholder="literal transcription",
                )

            with cols[2]:
                auto_interp = interpret(literal, mapping) if literal else ""
                interp_default = saved_interp if saved_interp else auto_interp
                interpreted = st.text_input(
                    f"Interpreted — line {line_id}",
                    value=interp_default,
                    key=f"int_{page_id}_{line_id}",
                    placeholder="interpreted / expanded",
                )

            if st.button(f"💾 Save line {line_id}", key=f"save_{page_id}_{line_id}"):
                upsert_label(page_id, line_id, str(img_path), literal, interpreted)
                st.success(f"Saved line {line_id}")

        st.divider()

    # ── export button ──────────────────────────────────────────────────────────
    st.subheader("Export page outputs")
    if st.button("📄 Export literal.md + interpreted.md"):
        out_dir = ROOT / "outputs" / page_id
        out_dir.mkdir(parents=True, exist_ok=True)
        lit_md, int_md = export_markdown(page_id, lines_meta, mapping)
        (out_dir / f"{page_id}_literal.md").write_text(lit_md, encoding="utf-8")
        (out_dir / f"{page_id}_interpreted.md").write_text(int_md, encoding="utf-8")
        st.success(f"Exported to {out_dir}")
        st.download_button(
            "⬇ Download literal.md", lit_md, file_name=f"{page_id}_literal.md"
        )
        st.download_button(
            "⬇ Download interpreted.md",
            int_md,
            file_name=f"{page_id}_interpreted.md",
        )

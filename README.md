# handwriting — local handwriting OCR agent

A Windows-friendly end-to-end pipeline to:

1. Ingest full-page handwriting images.
2. Segment into line crops.
3. Run personalized OCR (lightweight CNN+BiLSTM+CTC — works on GT 1030 / CPU).
4. Produce **literal** (raw) and **interpreted** (expanded shorthand) output.
5. Review and correct labels with a local **Streamlit** UI.

---

## Repository layout

```
handwriting/
  preprocess.py       grayscale, threshold, deskew
  segmentation.py     page → ordered line crops + meta.json
  export_text.py      one-shot pipeline producing literal.md + interpreted.md
  ocr/
    model.py          lightweight CNN+BiLSTM+CTC model
    train_ocr.py      train/fine-tune on labeled line crops
    infer_ocr.py      predict literal text + confidences
  interpreter/
    shorthand.yml     token → expansion dictionary (edit to add yours)
    apply.py          rules + dictionary: literal → interpreted
app/
  streamlit_app.py    local labeling/review UI
tests/
  test_interpreter.py
  test_segmentation_smoke.py
data/
  labels.csv          training labels
  pages/              full-page images
  lines/              line crops
requirements.txt
README.md
```

---

## Setup (Windows 11)

### 1. Install Git for Windows
Download from https://git-scm.com/download/win

### 2. Clone and create a virtual environment

```powershell
git clone https://github.com/keith2050/handwriting.git
cd handwriting
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

> **GPU note:** PyTorch will automatically use CUDA if available. Otherwise everything runs on CPU.

---

## Data format

| Column | Description |
|--------|-------------|
| `page_id` | Unique identifier for the source page |
| `line_id` | 1-based line number within the page |
| `image_path` | Relative path to the line-crop PNG |
| `literal` | Raw OCR transcription |
| `interpreted` | Expanded / interpreted version |

Line crops: `data/lines/<page_id>/line_NNNN.png`
Segmentation metadata: `data/lines/<page_id>/meta.json`

---

## CLI usage

### Segment a page into lines

```powershell
python -m handwriting.segmentation `
    --input   data/pages/page1.jpg `
    --out_dir data/lines/page1 `
    --page_id page1
```

### Interpret a literal string

```powershell
python -m handwriting.interpreter.apply --text "pt -> f/u ↑ dose"
# patient to follow-up increase dose
```

### Train OCR

```powershell
python -m handwriting.ocr.train_ocr `
    --labels     data/labels.csv `
    --out        checkpoints/ `
    --epochs     30 `
    --batch_size 4 `
    --accumulate 4
```

### Run OCR inference

```powershell
python -m handwriting.ocr.infer_ocr `
    --checkpoint checkpoints/best.pt `
    --input_dir  data/lines/page1 `
    --out        outputs/page1_literal.txt
```

### Export literal + interpreted markdown for a page

```powershell
python -m handwriting.export_text `
    --page       data/pages/page1.jpg `
    --checkpoint checkpoints/best.pt `
    --out        outputs/page1/
```

---

## Streamlit labeling UI

```powershell
streamlit run app/streamlit_app.py
```

Steps:
1. Enter a `page_id` in the sidebar.
2. Upload a page image.
3. Click **Segment page into lines**.
4. Fill in `literal` for each line (OCR suggestion auto-filled if checkpoint exists).
5. Click **Save line N** to write to `data/labels.csv`.
6. Click **Export** when done.

---

## Shorthand dictionary

Edit `handwriting/interpreter/shorthand.yml` to add your abbreviations:

```yaml
"pt":  "patient"
"->":  "to"
"↑":   "increase"
```

---

## Margin notes

Lines outside the main text column are marked `is_margin: true` in `meta.json`.
In exported markdown they appear after main-column lines with a `*(margin)*` suffix.

---

## Tests

```powershell
pytest tests/ -v
```

---

## Model details

- **CNN:** 3 conv-pool blocks (32 → 64 → 128 channels)
- **RNN:** 2-layer BiLSTM (hidden 256)
- **Loss:** CTC
- **Image height:** 64 px, width padded to 1024
- **Size:** ~5 MB — fits GT 1030 VRAM

Conservative defaults for GT 1030: `--batch_size 4 --accumulate 4`

---

## Hardware profiles

The training script supports two built-in hardware profiles selected with
`--profile`.  Any flag you pass explicitly overrides the profile value.

### Comparison table

| Setting | `--profile gt1030` (default) | `--profile rtx5060ti` |
|---|---|---|
| GPU | GT 1030 (~2 GB VRAM) | RTX 5060 Ti (~16 GB VRAM) |
| CPU | i5-6500 (4 cores) | Ultra 9 285 (24 cores) |
| `--img_height` | 64 | 96 |
| `--max_width` | 1024 | 2048 |
| `--cnn_channels` | 32,64,128 | 64,128,256 |
| `--lstm_hidden` | 256 | 512 |
| `--batch_size` | 4 | 32 |
| `--accumulate` | 4 (eff. batch 16) | 1 |
| `--num_workers` | 0 | 8 |
| `--amp` | off | on (float16) |
| Model parameters | ~2.8 M | ~21 M |
| VRAM (fp32 / fp16) | ~0.4 GB / — | ~1.2 GB / ~0.6 GB |

### What changes on the new hardware

**RTX 5060 Ti (Blackwell, ~16 GB VRAM)**

* **Larger model** — the bigger CNN (64→128→256 channels) and LSTM (hidden=512)
  learns finer stroke details and longer context, which directly lowers CER on
  your personal handwriting.
* **Bigger batches** — `batch_size=32` with no gradient accumulation means
  cleaner gradients and faster wall-clock convergence.
* **Automatic mixed precision (AMP)** — `torch.cuda.amp` stores activations in
  float16, roughly halving VRAM usage and doubling throughput on Tensor Cores.
  Gradients are automatically scaled to avoid underflow.
* **Larger line images** — `img_height=96` preserves more stroke detail for
  fine-grained character recognition; `max_width=2048` handles long lines
  without truncation.
* No need for gradient accumulation (`--accumulate 1`).

**Intel Core Ultra 9 285 (24 P+E cores)**

* **More DataLoader workers** (`--num_workers 8`) — image loading, grayscale
  conversion, and resizing run in 8 parallel processes, keeping the GPU
  saturated between batches.  On the old i5-6500 (4 cores) `num_workers=0`
  avoids overhead.
* **Faster segmentation** — OpenCV's deskew, adaptive threshold, and contour
  detection use SIMD/AVX instructions that are wider and faster on Arrow Lake.
* **Faster Streamlit UI** — the Streamlit server process can schedule Python
  threads across more cores.

### RTX 5060 Ti training command

```powershell
python -m handwriting.ocr.train_ocr `
    --labels  data/labels.csv `
    --out     checkpoints/ `
    --profile rtx5060ti `
    --epochs  50
```

### GT 1030 training command (original)

```powershell
python -m handwriting.ocr.train_ocr `
    --labels  data/labels.csv `
    --out     checkpoints/ `
    --profile gt1030 `
    --epochs  30
```

### Mixing profiles and custom flags

```powershell
# RTX 5060 Ti but with a custom learning rate and 100 epochs
python -m handwriting.ocr.train_ocr `
    --labels  data/labels.csv `
    --out     checkpoints/ `
    --profile rtx5060ti `
    --lr      3e-4 `
    --epochs  100
```

### PyTorch with CUDA for RTX 5060 Ti (Blackwell)

The RTX 5060 Ti requires CUDA 12.8+.  Install the matching PyTorch build:

```powershell
# Check CUDA version first
nvidia-smi

# Install PyTorch with CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

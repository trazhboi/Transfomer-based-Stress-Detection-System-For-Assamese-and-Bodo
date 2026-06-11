# 🧠 Transformer-Based Psychological Stress Detection from Speech
### Low-Resource Languages: Assamese & Bodo

> Fine-tuning **Wav2Vec2** for binary stress classification in two severely underrepresented Indian languages — with a custom-built dataset, automated VAD pipeline, and classical ML baselines for comparison.

---

## Overview

Most speech stress detection research is locked to English and Mandarin. This project pushes into territory that has essentially zero prior work: **Assamese** and **Bodo** — languages spoken by millions in Northeast India's Brahmaputra valley, with no existing annotated stress speech corpora.

We built everything from scratch:
- Scraped and curated raw audio from movies, podcasts, news, and interviews
- Preprocessed it through a full VAD-based segmentation pipeline
- Auto-labeled stress using pitch (F0) + short-time energy analysis
- Fine-tuned `facebook/wav2vec2-base` for binary stress classification
- Benchmarked against SVM, Random Forest, XGBoost, Logistic Regression, and KNN using Wav2Vec2 embeddings

**Result: Wav2Vec2 hits 87.3% accuracy / 87.0% F1 — ~12 points ahead of the best classical baseline (XGBoost at 75.8%).**

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Wav2Vec2 (fine-tuned)** | **87.3%** | **86.9%** | **87.1%** | **87.0%** |
| XGBoost | 75.8% | 75.0% | 75.4% | 75.2% |
| SVM | 74.6% | 73.8% | 74.2% | 74.0% |
| Random Forest | 72.1% | 71.5% | 71.9% | 71.7% |
| Logistic Regression | 68.4% | 67.9% | 68.1% | 68.0% |
| KNN | 65.2% | 64.7% | 65.0% | 64.8% |

*Classical ML models use Wav2Vec2 mean-pooled embeddings as features — same representation, only the classifier changes.*

**Language-wise breakdown (Wav2Vec2):**

| Language | Accuracy | F1-Score | Test Segments |
|---|---|---|---|
| Assamese | 88.7% | 88.4% | 840 |
| Bodo | 85.6% | 85.2% | 620 |

---

## Dataset

No public annotated stress speech corpus exists for Assamese or Bodo. The dataset was constructed from:

- Assamese movies and short films
- Assamese podcasts and news recordings
- Bodo news channel content
- Bodo interviews and public speech recordings

| Language | Raw Duration | Segments (post-VAD) |
|---|---|---|
| Assamese | ~18 hrs | ~4,200 |
| Bodo | ~14 hrs | ~3,100 |

**Automatic labeling:** Segments are labeled `stressed=1` if both normalized mean F0 and mean short-time energy exceed the 60th percentile for that language. Otherwise `stressed=0`. Ambiguous segments (only one criterion met) are excluded.

Final labels saved to `dataset/labels/labels.csv`.

> ⚠️ The raw audio files are **not included** in this repo due to copyright. The preprocessing and labeling scripts are fully reproducible — see [Setup](#setup).

---

## Architecture

```
Raw Audio (Assamese / Bodo)
        │
        ▼
  Noise Reduction (noisereduce)
        │
        ▼
  Silence Removal + Resampling → 16kHz mono WAV
        │
        ▼
  VAD Segmentation (webrtcvad) → 1–10s speech segments
        │
        ▼
  Auto-Labeling (Pitch + Energy analysis)
        │
        ├─────────────────────────────┐
        ▼                             ▼
  Wav2Vec2 Fine-tuning         Wav2Vec2 Embeddings
  (end-to-end)                 (mean-pooled, 768-dim)
        │                             │
        ▼                             ▼
  Stress Classifier           Classical ML Classifiers
  (Linear head)               (SVM / RF / XGB / LR / KNN)
        │                             │
        └─────────────┬───────────────┘
                      ▼
              Comparative Evaluation
```

---

## Setup

**Requirements:** Python 3.10+, Google Colab (recommended for GPU), Google Drive mounted.

```bash
pip install transformers datasets evaluate accelerate \
            noisereduce webrtcvad librosa xgboost \
            scikit-learn pandas numpy matplotlib seaborn
```

Mount Drive in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Usage

### 1. Preprocess Audio

```bash
python src/preprocess.py \
  --input_dir /content/drive/MyDrive/dataset/raw \
  --output_dir /content/drive/MyDrive/dataset/processed
```

### 2. Run VAD Segmentation

```bash
python src/vad_segment.py \
  --input_dir dataset/processed \
  --output_dir dataset/segments
```

### 3. Auto-label

```bash
python src/autolabel.py \
  --segments_dir dataset/segments \
  --output_csv dataset/labels/labels.csv
```

### 4. Train Wav2Vec2

```bash
python src/train_transformer.py \
  --data_csv dataset/labels/labels.csv \
  --model_output models/wav2vec2_stress
```

### 5. Train Classical ML Baselines

```bash
python src/train_classical.py \
  --data_csv dataset/labels/labels.csv \
  --embeddings_output features/embeddings.npy
```

### 6. Evaluate All Models

```bash
python src/evaluate.py \
  --model_dir models/wav2vec2_stress \
  --embeddings features/embeddings.npy \
  --labels dataset/labels/labels.csv
```

---

## Project Structure

```
.
├── src/
│   ├── preprocess.py       # Noise reduction, silence removal, resampling
│   ├── vad_segment.py      # WebRTC VAD-based segmentation
│   ├── autolabel.py        # Pitch + energy auto-labeling
│   ├── train_transformer.py # Wav2Vec2 fine-tuning
│   ├── train_classical.py  # SVM / RF / XGB / LR / KNN training
│   └── evaluate.py         # Metrics, confusion matrix, comparison plots
├── dataset/
│   ├── segments/           # Post-VAD audio segments (not tracked)
│   └── labels/
│       └── labels.csv      # file_path, language, label
├── models/
│   └── wav2vec2_stress/    # Saved model checkpoint (not tracked)
├── notebooks/
│   └── experiments.ipynb   # Full Colab notebook
├── results/
│   └── figures/            # Confusion matrices, bar charts
└── README.md
```

---

## Model Config

| Parameter | Value |
|---|---|
| Base model | `facebook/wav2vec2-base` |
| Optimizer | AdamW |
| Learning rate | 3e-5 |
| Warmup steps | 500 |
| Epochs | 30 (early stop, patience=5) |
| Batch size | 8 |
| Gradient clip | 1.0 |
| Train/test split | 80:20 stratified |

---

## Tech Stack

`Python` `PyTorch` `Hugging Face Transformers` `librosa` `webrtcvad` `noisereduce` `scikit-learn` `XGBoost` `Google Colab`

---

## Limitations

- Labels are automatically generated — not manually annotated. Estimated ~12% label noise on a spot-checked subset.
- `wav2vec2-base` is pretrained on English (LibriSpeech). Neither Assamese nor Bodo is in the pretraining data — there's an inherent cross-lingual transfer gap.
- Binary classification only. Multi-level stress intensity is future work.

---

## Future Work

- [ ] Manual annotation with Assamese/Bodo-speaking linguists
- [ ] Fine-tune on `wav2vec2-large-xlsr-53` (multilingual pretraining)
- [ ] Multi-class stress intensity grading
- [ ] Real-time mobile app deployment for BTR/Assam region
- [ ] Extend to other NE Indian languages (Meitei, Khasi, Mizo)

---

## Citation

If you use this work, please cite:

```
@project{stress_detection_assamese_bodo_2026,
  title     = {Transformer-Based Psychological Stress Detection from Speech Signals
               in Low-Resource Languages (Assamese and Bodo)},
  author    = {[Your Name]},
  year      = {2026},
  institution = {Central Institute of Technology Kokrajhar},
}
```

---

## License

This project is for academic research purposes. The model code is MIT licensed. Audio data sources retain their original copyrights and are not redistributed.

# Biomedical NLP: Classical vs. Transformer Models for Adverse Drug Effect Detection

Binary sentence classification on the **ADE Corpus V2**: This project benchmarks a classical TF-IDF + Logistic Regression baseline against a fine-tuned **BioBERT** model on imbalanced biomedical text data.

## Results (Test Set, N=3,528)

| Model | Weighted F1 | ADE Class F1 | Accuracy | Total Errors |
|---|---|---|---|---|
| MaxEnt + BoW (Gurulingappa et al. 2012) | 0.70 | — | — | — |
| TF-IDF + Logistic Regression | 0.90 | 0.84 | 90% | 349 |
| **BioBERT (fine-tuned)** | **0.96** | **0.93** | **96%** | **145** |

**BioBERT reduced misclassifications by 58% (349 → 145 errors)** compared to the classical baseline.

### BioBERT Training Curve

| Epoch | Train Loss | Val F1 | Val Accuracy |
|---|---|---|---|
| 1 | 0.175 | 0.943 | 0.943 |
| 2 | 0.114 | 0.952 | 0.952 |
| 3 | 0.043 | 0.952 | 0.952 |

**Best checkpoint:** Epoch 3. Final val F1 **(0.952)** exceeds the reported literature range **(0.88–0.91)** for this task.

## Approach

- **Data:** [ADE Corpus V2](https://huggingface.co/datasets/ade-benchmark-corpus/ade_corpus_v2) (classification subset), stratified 70/15/15 train/val/test split (seed=42)
- **Baseline:** `TfidfVectorizer` (1-2 grams) + `LogisticRegression` (class-weighted), tuned via 3-fold GridSearchCV over `max_features` (20k/50k/100k) and `C` (0.1/1/10)
- **Transformer:** `dmis-lab/biobert-base-cased-v1.2`, fine-tuned 3 epochs (lr=2e-5, batch size=16, fp16), max sequence length 128
- **Evaluation:** Weighted F1 (primary metric due to class imbalance), confusion matrices, per-sentence error analysis

## Project Structure

```
biomed-nlp-comparison/
├── Dockerfile                 # GPU container definition
├── docker-compose.yml         # Container orchestration
├── requirements.txt           # Python dependencies
│
├── src/models/
│   ├── baseline_model.py      # TF-IDF + LR pipeline builder & grid search
│   └── biobert_model.py        # BioBERT tokenizer, model loader, Trainer setup
│
├── scripts/
│   ├── check_gpu.py            # CUDA diagnostics (run before training)
│   ├── prepare_data.py         # Download + split data → data/splits/
│   ├── train_baseline.py        # Grid search + evaluate baseline
│   ├── train_biobert.py         # Fine-tune BioBERT + evaluate
│   └── evaluate.py              # Shared metrics, confusion matrix, error analysis
│
├── notebooks/
│   ├── 01_exploration.ipynb     # EDA: class balance, sentence lengths, word clouds
│   └── 02_analysis.ipynb        # Post-training analysis & error analysis
│
├── data/splits/                 # train/val/test CSVs (included, pre-split)
├── models/                       # Saved baseline pipeline + fine-tuned BioBERT
├── reports/
│   ├── figures/                  # Confusion matrices, training curves, EDA plots
│   ├── results/                  # Per-model error CSVs (text, true, predicted)
│   ├── html/                      # Rendered notebook HTMLs
│   └── presentation/              # Slide-format summary
└── references/                   # Source papers (ADE Corpus, BioBERT)
```

## Reproducing Results

This project runs in a GPU-enabled Docker container for full reproducibility. The container is based on `nvcr.io/nvidia/pytorch:25.03-py3` (Python 3.12, CUDA 12.x, PyTorch pre-installed), with Quarto and pinned dependencies installed on top. The HuggingFace cache is mounted as a volume, so BioBERT (~440MB) downloads once and is reused across runs.

```bash
docker compose build
docker compose up               # JupyterLab at localhost:8888

# Inside the container:
python scripts/check_gpu.py     # Verify GPU meets minimum requirements
python scripts/prepare_data.py  # Optional: pre-split data already included in repo
python scripts/train_baseline.py
python scripts/train_biobert.py
```

**Outputs:**
- `models/baseline/tfidf_pipeline.joblib`: best sklearn pipeline from grid search
- `models/biobert_finetuned/`: fine-tuned BioBERT model + tokenizer
- `reports/figures/`: confusion matrices (`baseline_cm.png`, `biobert_cm.png`), training curve, EDA plots
- `reports/results/`: `baseline_errors.csv`, `biobert_errors.csv` (misclassified sentences for error analysis)

Explore `notebooks/01_exploration.ipynb` for EDA and `notebooks/02_analysis.ipynb` for cross-model error analysis.

> **Note:** Trained BioBERT weights (~440MB) are not included in this repo due to size. Run `train_biobert.py` to regenerate them locally on a GPU with ≥8GB VRAM.

## References

- Gurulingappa et al. (2012), *Development of a benchmark corpus to support the automatic extraction of drug-related adverse effects from medical case reports*
- Lee et al. (2020), *BioBERT: a pre-trained biomedical language representation model for biomedical text mining*

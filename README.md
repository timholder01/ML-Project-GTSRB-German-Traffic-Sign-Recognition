# GTSRB — German Traffic Sign Recognition Benchmark

A small CNN for **43 classes** of German traffic signs on the
[GTSRB Kaggle dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

Training uses a balanced subset of **19,995** images (465 per class) to stay within a ~10–20k sample budget.
On the official test set the model reaches about **97.9%** accuracy and **0.978** macro-F1.

## Pipeline

1. Download data with `kagglehub` and inspect class imbalance
2. Hold out whole **tracks** for validation (avoid near-duplicate frame leakage)
3. Build a balanced 19,995-image training subset
4. Preprocess: ROI crop → autocontrast → resize 32×32 → normalise
5. Train a small CNN (3 conv blocks + BatchNorm) and evaluate on the official test set

## Requirements

Python 3 with the packages in `requirements.txt` (`torch`, `torchvision`, `pandas`, `scikit-learn`, `matplotlib`, `Pillow`, `kagglehub`, …).

```bash
pip install -r requirements.txt
```

Kaggle credentials are required once (`kagglehub` / API token). On Google Colab the notebook prompts for login when needed.

## Usage

Open `notebook.ipynb` and run the cells in order. The dataset is downloaded automatically from Kaggle.

Generated figures used in the write-up live under `figures/`.

## Notes

- Augmentation is **sign-safe**: no horizontal flip, rotation at most ±10°.
- The checkpoint is selected by validation **macro-F1**, not only accuracy.
- Model weights under `models/` are not committed; retrain by running the notebook.

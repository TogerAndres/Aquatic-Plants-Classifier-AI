# 🌿 Aquatic Plants Classifier

An explainable deep learning classifier for aquatic plant species, built with **transfer learning (EfficientNetB0)**, **stratified k-fold cross-validation** for reliable accuracy estimation, and **Grad-CAM explainability** — every prediction comes with a heatmap showing exactly which part of the image the model focused on to make its decision.

Includes a REST API (Flask) and a modern frontend (React + Vite) to test the model by uploading photos directly from the browser.

![status](https://img.shields.io/badge/status-trained-7C5CFC)
![python](https://img.shields.io/badge/python-3.10%2B-2DD4BF)
![tensorflow](https://img.shields.io/badge/tensorflow-2.16-FF6F00)
![license](https://img.shields.io/badge/license-MIT-blue)

---

## 📸 Screenshots

| Upload screen | Prediction + Grad-CAM explainability |
|---|---|
| ![Upload screen](docs/screenshots/app-home.png) | ![Grad-CAM demo](docs/screenshots/gradcam-demo.png) |

The center slider in the second screenshot is the **"explainability lens"**: drag it across the original photo to reveal the Grad-CAM heatmap overlay, showing which leaves or plant structures the model attended to when making its decision.

---

## 🧠 Species Recognized

| Common Name | Scientific Name |
|---|---|
| Common Duckweed | *Lemna minor* |
| Common Water Hyacinth | *Eichornia crassipes* |
| Heartleaf False Pickerelweed | *Monochoria korsakowii* |
| Water Lettuce | *Pistia stratiotes* |

---

## ✨ Key Features

- **Swappable backbone**: EfficientNetB0 by default, switchable to `EfficientNetB1`, `ResNet50V2`, or `MobileNetV3Large` with a single CLI flag — the data pipeline and classifier head stay identical, making architecture comparisons fair (see [`src/training/config.py`](src/training/config.py)).
- **Stratified K-Fold cross-validation**: the model is trained and evaluated K times on class-balanced partitions. Results are reported as mean ± standard deviation, not a single optimistic number from a fixed split.
- **Automatic class weights**: compensates for class imbalance so the model doesn't favor the most common species during training.
- **Real explainability**: every API prediction includes a Grad-CAM heatmap (base64 PNG, ready to render) and a plain-text explanation of which image region the model focused on and with what confidence level.
- **Explainability lens UI**: the before/after slider shown above — not a side image, but a live reveal inside the original photo.

---

## 📊 Training Results

Training ran for 4 epochs per phase (2 with the backbone frozen, 2 of fine-tuning) over 5 stratified folds, followed by a final retraining run on 100% of the dataset.

![Training curves](docs/screenshots/training-curves.png)

### Cross-Validation Summary (5 folds)

| Metric | Mean ± Std Dev |
|---|---|
| Accuracy | 0.9993 ± 0.0010 |
| AUC | 1.0000 ± 0.0000 |

### Final Model (trained on 100% of the dataset)

| Metric | Value |
|---|---|
| Train Loss | 0.8432 |
| Train Accuracy | 98.09% |

### Reading the curves

- **Transfer learning kicks in immediately**: by the very first head-training epoch (`Head E1`), validation accuracy already exceeds 98% across all 5 folds — confirming that EfficientNetB0's ImageNet features transfer strongly to these 4 aquatic species.
- **Stable fine-tuning**: unlocking the full backbone (`FT E1`–`FT E2`) reduces loss monotonically from ~1.7 to ~0.84 with no spikes or divergence, thanks to the low learning rate (`1e-5`) used in that phase.
- **Consistent folds**: the thin background lines (one per fold) cluster tightly, meaning the result doesn't depend on which images happened to fall in each partition.

### ⚠️ Honest note on these numbers

A validation accuracy of ~99.9% and AUC of 1.0 across 5 folds, in just 4 epochs per phase, is unusually high — even for transfer learning on 4 classes. Before presenting this model as production-ready, the most common cause of this pattern should be ruled out:

> **Data leakage between augmented copies.** If the dataset used is the pre-augmented folder (`Augmented Images`) and the K-Fold split was done at the file level rather than at the original photo level, it is very likely that multiple augmented copies of the **same source photo** ended up split across the train and validation sets of the same fold. The model would then be recognizing variations of images it effectively saw during training — not generalizing to truly new plants.

This does not invalidate the pipeline or the code — the K-Fold, class weights, and Grad-CAM logic are all correct — but it does mean the exact number (99.9%) likely won't survive truly new photos. To verify:

1. Group images by original photo before splitting, using `StratifiedGroupKFold` from scikit-learn instead of `StratifiedKFold`.
2. Evaluate against an **external holdout**: photos that never went through the augmentation process at all (see [`docs/DATASET.md`](docs/DATASET.md)).

If accuracy stays high after those adjustments, the result is real and the model is genuinely robust.

---

## 🗂️ Repository Structure

```
.
├── src/
│   ├── training/        # config, dataset pipeline, model factory, K-Fold training
│   ├── inference/       # high-level predictor + Grad-CAM engine
│   ├── api/             # Flask REST API (/api/predict, /api/classes, /api/health)
│   └── utils/           # global seed, metrics, plotting utilities
├── frontend/            # React + Vite — upload UI, results panel, explainability lens
├── docs/
│   ├── screenshots/     # images used in this README
│   ├── ARCHITECTURE.md
│   ├── DATASET.md
│   ├── TRAINING.md
│   └── API.md
├── requirements.txt
└── README.md
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a detailed breakdown of each module and the key design decisions behind them.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare the dataset

Download the [Aquatic Plants Image Dataset](https://data.mendeley.com/datasets/vz6z64nwby/1)
from Mendeley Data and place the augmented images folder at `./data/Augmented Images`,
or set the path via environment variable:

```bash
export AQUATIC_DATASET_DIR="/path/to/Augmented Images"
```

### 3. Train with cross-validation

```bash
python -m src.training.train --backbone efficientnetb0 --folds 5
```

This writes to `artifacts/`:
- `folds/fold_N/best.keras` — best checkpoint per fold
- `reports/cross_validation_summary.json` — mean ± std metrics across folds
- `final_model/model.keras` — production model retrained on the full dataset

See the full guide in [`docs/TRAINING.md`](docs/TRAINING.md).

> Trained weights (`*.keras`) are excluded from this repository via `.gitignore` due to file size. Retrain locally using the dataset above, or publish weights separately (GitHub Releases, Hugging Face Hub, etc.) if you want to distribute a pre-trained model.

### 4. Run the API

```bash
export AQUATIC_MODEL_DIR=./artifacts/final_model
python -m src.api.app
```

API runs at `http://localhost:5000`. Full endpoint reference in [`docs/API.md`](docs/API.md).

### 5. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. In development mode, Vite proxies `/api/*` to the Flask backend on port 5000 (see `frontend/vite.config.js`) — no CORS issues.

---

## 💡 Design Decisions

### Why cross-validation instead of a single train/val split?

With a moderately sized dataset and only 4 classes, a fixed split can produce an accuracy that reflects luck (which images landed where) more than model quality. Stratified K-Fold trains K models on different, class-balanced partitions and averages the result. The standard deviation across folds is in practice more informative than the mean: a high std signals instability or insufficient data for some class.

### Why Grad-CAM for explainability?

A species classifier that only outputs a confidence score is a black box — there is no way to know whether it learned to recognize the plant or the water color in the background of the training photos. Grad-CAM computes the gradient of the predicted class with respect to the last convolutional layer of the backbone, producing a map of which image regions most influenced the decision. This is exposed in both the API (`gradcam_overlay_base64`) and the frontend (the before/after slider visible in the screenshot above).

### Why freeze BatchNorm during fine-tuning?

Keeping `BatchNormalization` layers frozen during the fine-tuning phase prevents their running statistics from drifting with the small batch sizes typical of fine-tuning — a common cause of a model that was performing well suddenly degrading after backbone unfreezing.

---

## 🛣️ Roadmap

- [ ] Switch to `StratifiedGroupKFold` (group by original photo) to eliminate data leakage from augmented copies.
- [ ] External holdout evaluation with photos not part of the augmentation pipeline.
- [ ] Fold ensemble: combine the K fold models at inference time instead of retraining from scratch.
- [ ] Calibration metrics (reliability diagram) in addition to accuracy and AUC.
- [ ] Docker Compose setup to deploy API + frontend together.

---

## 📄 License

MIT — see [`LICENSE`](LICENSE).

## 🙏 Dataset

[Aquatic Plants Image Dataset](https://data.mendeley.com/datasets/vz6z64nwby/1) — Mendeley Data.

## 👨‍💻 Author

**Roger Andrés Álvarez Díaz** · [@TogerAndres](https://github.com/TogerAndres) · [LinkedIn](https://linkedin.com/in/roger-andrés-alvarez-diaz-52b395333)

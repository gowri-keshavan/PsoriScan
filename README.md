# PsoriScan

An end-to-end machine learning system that estimates, from a photograph of skin, how likely the image is to show
psoriasis. It covers data preparation, model training and comparison across two deep learning frameworks,
rigorous evaluation, and a web application that returns a prediction with a visual explanation.



<!-- Live demo: add the deployed frontend URL here once it exists -->

## Highlights

- **Full pipeline:** dataset curation, training, evaluation, REST API and React interface, all reproducible
  from the command line.
- **Two frameworks, one comparison:** a custom CNN in TensorFlow/Keras and a transfer-learning ResNet18 in
  PyTorch, trained on identical data splits and evaluated with the same code.
- **Rigorous evaluation:** held-out test set, bootstrap 95% confidence intervals, confusion matrices, ROC
  curves, and a leakage check that stops the split if any duplicate or group crosses a boundary.
- **Dataset confound identified and quantified:** a baseline that sees only file properties (no pixels)
  reaches 95.6% test accuracy, which shows that high scores on this dataset must be interpreted with caution.
- **Explainability:** Grad-CAM heatmaps, implemented for both frameworks and displayed in the interface.
- **Engineering practice:** privacy-preserving inference (uploads are processed in memory and never stored),
  input validation, automated tests, continuous integration, and a Dockerfile.

## Technology

| Area | Tools |
|---|---|
| Machine learning | TensorFlow/Keras, PyTorch, torchvision, scikit-learn, OpenCV, NumPy, Pillow |
| Backend | Python, Flask, Gunicorn |
| Frontend | React, JavaScript (Create React App) |
| Quality and delivery | pytest, Jest and Testing Library, GitHub Actions, Docker |

## Results

All figures are measured on a held-out test split of 252 images that was not used for training or model
selection. Intervals are bootstrap 95% confidence intervals.

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Metadata-only decision tree (no image content) | 0.956 (0.933-0.980) | 0.983 (0.967-0.996) |
| Custom CNN (TensorFlow/Keras) | 0.960 (0.937-0.980) | 0.995 (0.988-0.999) |
| ResNet18, transfer learning (PyTorch) | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) |

| Model | Precision | Recall | Specificity | F1 |
|---|---|---|---|---|
| Custom CNN (TensorFlow/Keras) | 0.928 | 0.981 | 0.946 | 0.954 |
| ResNet18, transfer learning (PyTorch) | 1.000 | 1.000 | 1.000 | 1.000 |

**Interpretation.** These results should not be read as evidence of clinical performance. In the source data,
every positive image is 720 pixels on its longest side, compared with 2% of negative images, and the typical
negative image is roughly 305 x 310 pixels. The two classes were therefore collected from different sources,
and a model can achieve a high score by recognising the source rather than the condition. The custom CNN is
statistically indistinguishable from the baseline that uses file properties alone, and the perfect ResNet18
score on 252 images is consistent with the same effect. Generalisation to photographs from other sources has
not been established (see [Limitations](#limitations)).

## System overview

```
dataset/ ──prepare_dataset.py──> data/{train,val,test}/  ──train_tf.py────> models/psoriasis_cnn.keras
 (raw)     filter, dedupe, group,   (stratified, checked        train_torch.py─> models/psoriasis_resnet18.pt
           split                     for leakage)                      │
                                                  evaluate.py ◄────────┘   held-out metrics and plots
                                                  shortcut_baseline.py     metadata-only comparison
                                                            │
React app ──POST /predict──> Flask API ──> predictor (TensorFlow or PyTorch) ──> label, probability, heatmap
```

| Component | Script | Description |
|---|---|---|
| Data preparation | `prepare_dataset.py` | Removes unreadable and undersized images and exact duplicates; groups near-duplicate photographs using a 256-bit perceptual hash; optionally keeps numbered sibling files together; splits whole groups 70/15/15, stratified by class. Aborts if any duplicate or group appears in more than one split. Writes `data/manifest.csv` and `reports/dataset_summary.json`. |
| Training (TensorFlow) | `train_tf.py` | Compact CNN with three convolutional blocks and global average pooling; augmentation inside the model; early stopping on validation loss. |
| Training (PyTorch) | `train_torch.py` | ImageNet-pretrained ResNet18; the new classification head is trained first, then the full network is fine-tuned with early stopping. |
| Evaluation | `evaluate.py` | Accuracy, precision, recall, specificity, F1 and ROC-AUC with confidence intervals, plus confusion matrix, ROC curve and per-image predictions. Supports an external image folder through `--data-dir`. |
| Confound check | `shortcut_baseline.py` | Trains a decision tree on file properties only (width, height, size, type) and scores it on the same test split. |
| Serving | `app.py`, `psoriscan/` | REST API returning the label, psoriasis probability, confidence and a transparent heatmap. Images are processed in memory and never written to disk. |

Training, evaluation and serving share a single preprocessing module (`psoriscan/imaging.py`) and one predictor
class per framework (`psoriscan/predictors.py`), so the reported metrics describe exactly what the API serves.


## Getting started

Requires Python 3.10 or newer and Node.js 20 or newer.

```bash
# 1. Install (choose the framework or frameworks required)
pip install -r requirements.txt -r requirements-train.txt
pip install -r requirements-torch.txt        # and/or: pip install -r requirements-tf.txt

# 2. Prepare the data: place raw images in dataset/<any>/{no_psoriasis,psoriasis}/, then
python prepare_dataset.py --positive-keyword psoria --group-regex "^(\d+)[.,]\d+$"

# 3. Train, then evaluate once at the end
python train_torch.py
python train_tf.py
python evaluate.py --backend both
python shortcut_baseline.py

# 4. Start the API (serves MODEL_BACKEND, default torch) and the frontend
python app.py                                # http://localhost:5000
cd frontend && cp .env.example .env && npm install && npm start    # http://localhost:3000
```

Notes:
- In PowerShell, select the backend with `$env:MODEL_BACKEND="tf"`; in bash use `export MODEL_BACKEND=tf`.
- In bash, wrap the regular expression in single quotes: `--group-regex '^(\d+)[.,]\d+$'`.
- Both models train on a laptop CPU; the TensorFlow model needs roughly 75 seconds per epoch.
- `python train_torch.py --no-pretrained` trains without downloading ImageNet weights (lower accuracy).
- To evaluate external photographs, place them in `external/test/{psoriasis,no_psoriasis}/` and run
  `python evaluate.py --backend both --data-dir external --tag external`.

## API reference

| Endpoint | Description |
|---|---|
| `GET /health` | Returns `{"status": "ok", "backend": ..., "model": ...}`. |
| `POST /predict` | Multipart field `image` (JPEG, PNG, WEBP or BMP, up to 8 MB). Returns `label`, `is_psoriasis`, `probability_psoriasis`, `confidence`, `threshold`, `heatmap` (PNG data URL), `model` and `disclaimer`. Errors are returned as JSON, `{"error": {"code", "message"}}`, with status 400, 413 or 415. |

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_BACKEND` | `torch` | `torch` or `tf` |
| `MODEL_DIR` | `./models` | Location of model files |
| `MODEL_URL` | _(empty)_ | HTTPS link to the model file, downloaded on start if absent |
| `MODEL_SHA256` | _(empty)_ | Optional checksum for the download |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated list of allowed frontend origins |
| `DECISION_THRESHOLD` | `0.5` | Probability above which the label is "Psoriasis" |
| `MAX_UPLOAD_MB` | `8` | Upload size limit |
| `PORT`, `HOST`, `FLASK_DEBUG` | `5000`, `127.0.0.1`, off | Local development server only |

## Testing and continuous integration

```bash
pip install -r requirements-dev.txt
pytest -m "not slow"        # 26 tests: API, preprocessing, metrics and data-split logic (no ML framework needed)
pytest -m slow              # 2 tests: tiny end-to-end train, evaluate and serve runs per installed framework
cd frontend && npm test -- --watchAll=false    # 5 interface tests
```

GitHub Actions (`.github/workflows/ci.yml`) runs the backend tests, the end-to-end smoke tests for both
frameworks, and the frontend tests and production build on every push.


## Repository structure

```
app.py                 API entry point (python app.py | gunicorn 'app:create_app()')
psoriscan/             configuration, imaging, data access, models, predictors, API, reporting
prepare_dataset.py     filtering, de-duplication, grouping and splitting
train_tf.py            TensorFlow/Keras CNN training
train_torch.py         PyTorch ResNet18 training
evaluate.py            held-out evaluation, metrics and plots
shortcut_baseline.py   metadata-only baseline
frontend/              React application
tests/                 pytest suite
reports/               metrics, plots and dataset summary
Dockerfile, .github/   container image and CI workflow
```


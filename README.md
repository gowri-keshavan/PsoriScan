# PsoriScan

Psoriasis screening from skin photos. Two deep learning models (a custom CNN in **TensorFlow/Keras** and a
transfer-learning **ResNet18 in PyTorch**) are trained on the same leakage-free data splits, compared on a
held-out test set, and the better one is served through a **Flask** API to a **React** web app that also shows
a Grad-CAM heatmap of the regions that drove each prediction.

> **Not a medical device.** PsoriScan is a machine learning project. It cannot diagnose psoriasis or any
> other condition. See a dermatologist for any skin concern.

Live demo: _add your deployed frontend URL here_

## How it works

```
dataset/ ──prepare_dataset.py──> data/{train,val,test}/  ──train_tf.py────> models/psoriasis_cnn.keras
 (raw)     dedupe, group near-       (stratified, no leakage)  train_torch.py─> models/psoriasis_resnet18.pt
           duplicates, split                                         │
                                                      evaluate.py ◄──┘  (held-out test split, metrics + plots)
                                                                │
React app ──POST /predict──> Flask API ──> predictor (TF or PyTorch) ──> label, probability, Grad-CAM heatmap
```

| Step | Script | What it does |
|---|---|---|
| Data preparation | `prepare_dataset.py` | Drops unreadable/tiny files and exact duplicates, groups near-duplicate photos (256-bit perceptual hash), then splits whole groups 70/15/15 stratified by class, so a photo and its near-copy can never land in different splits. Writes `data/manifest.csv` and `reports/dataset_summary.json`. |
| Training (TensorFlow) | `train_tf.py` | Small CNN (3 conv blocks, global average pooling), augmentation inside the model, early stopping on validation loss. |
| Training (PyTorch) | `train_torch.py` | ImageNet-pretrained ResNet18: head-only phase, then full fine-tuning. |
| Evaluation | `evaluate.py` | Accuracy, precision, recall, specificity, F1, ROC-AUC with bootstrap 95% CIs, confusion matrix, ROC curve and per-image predictions for each model, on the **test** split only. |
| Serving | `app.py`, `psoriscan/` | `POST /predict` returns the label, psoriasis probability, confidence and a transparent heatmap PNG. Images are processed in memory and never written to disk. |

Training, evaluation and the API share one preprocessing path (`psoriscan/imaging.py`) and one predictor
class per framework (`psoriscan/predictors.py`), so evaluation numbers describe exactly what the API serves.

### Dataset

Source: _add the dataset name, link and licence here_ (the raw images are not included in this repo).

`prepare_dataset.py` on the original folders (2,002 images): 7 images dropped as too small (under 32 px on a
side), 48 exact duplicates removed, leaving 1,947 images split into 1,363 train / 292 validation / 292 test
(balanced between classes). In the original train/validation folders, 2 groups of identical images appeared in
both; the new split removes that leakage. Limitation: patient identity is unknown, so photos of the same person
could still appear in different splits.

## Results

Run `python evaluate.py --backend both` after training, then paste `reports/comparison_test.md` here.

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Custom CNN (TensorFlow/Keras) | | | | | | |
| ResNet18 transfer learning (PyTorch) | | | | | | |

Serve whichever model wins on the test split (`MODEL_BACKEND=tf` or `torch`).

## Quick start

```bash
# 1. Install (pick the framework(s) you need)
pip install -r requirements.txt -r requirements-train.txt
pip install -r requirements-torch.txt        # and/or: pip install -r requirements-tf.txt

# 2. Data: put the raw images in dataset/<anything>/{no_psoriasis,psoriasis}/ then
python prepare_dataset.py

# 3. Train and evaluate
python train_torch.py
python train_tf.py
python evaluate.py --backend both

# 4. Run the API (serves MODEL_BACKEND, default torch), then the frontend
python app.py                                # http://localhost:5000
cd frontend && cp .env.example .env && npm install && npm start    # http://localhost:3000
```

Use `python train_torch.py --no-pretrained` to train offline (random weights, lower accuracy).

### API

| Endpoint | Description |
|---|---|
| `GET /health` | `{"status": "ok", "backend": ..., "model": ...}` |
| `POST /predict` | Multipart field `image` (JPEG, PNG, WEBP or BMP, up to 8 MB). Returns `label`, `is_psoriasis`, `probability_psoriasis`, `confidence`, `threshold`, `heatmap` (PNG data URL), `model`, `disclaimer`. Errors are JSON: `{"error": {"code", "message"}}` with status 400, 413 or 415. |

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_BACKEND` | `torch` | `torch` or `tf` |
| `MODEL_DIR` | `./models` | Where model files live |
| `MODEL_URL` | _(empty)_ | https link to the model file; downloaded on start if missing |
| `MODEL_SHA256` | _(empty)_ | Optional checksum of the download |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed frontend origins |
| `DECISION_THRESHOLD` | `0.5` | Probability above which the label is "Psoriasis" |
| `MAX_UPLOAD_MB` | `8` | Upload size limit |
| `PORT`, `HOST`, `FLASK_DEBUG` | `5000`, `127.0.0.1`, off | Local dev server only |

## Deployment

1. **Host the model file** (not committed to git): create a GitHub Release or a Hugging Face Hub repo and upload
   `models/psoriasis_resnet18.pt`. Copy its direct https download link.
2. **API** (Render, Fly.io, Cloud Run or Hugging Face Spaces with the Docker SDK): deploy the `Dockerfile`.
   Set `MODEL_URL` to the link above, `CORS_ORIGINS` to your frontend URL, and `PORT` if the platform requires
   one. For the TensorFlow model build with `--build-arg BACKEND=tf` and `MODEL_BACKEND=tf`.
   Free tiers sleep when idle, so the first request after a pause can take a minute.
3. **Frontend** (Vercel or Netlify): root directory `frontend`, build command `npm run build`, output `build`,
   environment variable `REACT_APP_API_URL=https://<your-api-host>`.
4. Put both links at the top of this README and on your resume.

```bash
docker build -t psoriscan-api .
docker run -p 5000:5000 -e MODEL_URL=https://.../psoriasis_resnet18.pt -e CORS_ORIGINS=http://localhost:3000 psoriscan-api
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -m "not slow"        # API, preprocessing, metrics, data-split tests (no ML framework needed)
pytest -m slow              # tiny end-to-end train/evaluate/serve runs for each installed framework
cd frontend && npm test -- --watchAll=false
```

GitHub Actions (`.github/workflows/ci.yml`) runs all of the above on every push.

## Limitations

- The training photos come from one source, so the models may not generalize to other cameras, lighting or
  skin tones. Check the test set composition before trusting any number.
- The task is binary (psoriasis vs not). Look-alike conditions such as eczema or fungal infections are not
  modeled separately.
- Grad-CAM shows which regions influenced the model's score, which is not the same as the location of disease.
- Metrics come from a single random, group-aware split. Treat small differences between models as noise (see
  the confidence intervals).

## Project layout

```
app.py                 API entry point (python app.py | gunicorn 'app:create_app()')
psoriscan/             config, imaging, data access, models, predictors, API, reporting
prepare_dataset.py     dedupe + leakage-free split          train_tf.py / train_torch.py   training
evaluate.py            held-out evaluation and plots         tests/                         pytest suite
frontend/              React app                             Dockerfile, .github/workflows  deploy and CI
```

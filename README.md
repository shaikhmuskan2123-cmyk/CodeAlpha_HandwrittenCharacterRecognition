# GlyphSense AI: Handwritten Character Intelligence

CodeAlpha Machine Learning Internship · Task 3

GlyphSense AI classifies one handwritten uppercase English letter (A–Z) per image using a convolutional neural network trained on the real NIST EMNIST Letters dataset. It does not contain preset predictions or generated evaluation results. Until the model is trained and evaluated, the app reports that those artifacts are unavailable.

## Features

- Mouse drawing canvas and PNG/JPG/JPEG upload for single letters
- Batch inference with per-file error handling and CSV export
- Saved official-test evaluation metrics, class report, confusion matrix, curves, and sample mistakes
- Session-only prediction history and CSV export
- Separate quick smoke-training mode and full training mode

## Technology

Python 3.10, TensorFlow/Keras, NumPy, Pandas, scikit-learn, OpenCV, Pillow, Matplotlib, Seaborn, Plotly, Streamlit, pytest.

## Dataset and preprocessing

The project downloads the official EMNIST gzip archive from NIST: <https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip>. The Letters split has 26 classes. IDX image files are checked for their magic number, record count, and 28×28 dimensions. EMNIST's stored image axes are transposed into the conventional upright layout. Official labels 1–26 map to indices 0–25, corresponding to A–Z.

The official training split is divided reproducibly and stratified into training and validation subsets (seed 42); the official test split is kept separate for final evaluation. Training-only Keras augmentation applies small random rotations and translations. Uploaded/drawn images are converted to grayscale, polarity-normalized, cropped to foreground, resized with aspect ratio preserved, centered on a 28×28 black canvas, and normalized to [0, 1].

EMNIST Letters includes balanced lowercase/uppercase letter forms in one class per letter. Real-world drawings, different writing styles, image noise, and distribution differences can reduce accuracy. This system recognizes one character, not words or lines. Softmax scores are not automatically calibrated probabilities of correctness. Grad-CAM is omitted; no heatmap is presented as an explanation.

## Setup (Windows PowerShell)

Use Python 3.10 x64. TensorFlow is pinned to 2.15.1, which provides CPU wheels for Windows with Python 3.10. A virtual environment keeps dependencies isolated:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the environment's interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Download and train

Download and validate the official NIST archive (approximately 500 MB; requires internet access):

```powershell
python -m scripts.download_dataset
```

If automatic download is blocked, download the official gzip archive from the dataset link above, extract the four `emnist-letters-train/test-{images/labels}-idx*-ubyte.gz` files into `data/raw/`, and run `python -m scripts.download_dataset` to verify them.

For a pipeline check, train the clearly separate smoke model on 1,040 training images for two epochs:

```powershell
python -m scripts.train_model --smoke
```

Full training uses all official training examples and up to 30 epochs with early stopping and best-checkpoint saving:

```powershell
python -m scripts.train_model
```

CPU training can take tens of minutes or longer depending on the computer. TensorFlow uses CPU by default on standard Windows installs; keep the computer powered and allow the command to finish. The smoke model is saved separately and is never accepted by the app as the production model.

Evaluate only on the official test split:

```powershell
python -m scripts.evaluate_model
```

The full model (`models/glyphsense_emnist_letters.keras`), training history, class mapping, configuration, JSON metrics, and a small error sample are generated locally. Data, models, and misclassification arrays are excluded from Git. No metrics are claimed until evaluation completes.

## Run the application

```powershell
streamlit run app.py
```

The application can launch before training and will show model readiness clearly. Prediction pages require the full trained model. Use one handwritten letter per image; supported uploads are PNG/JPG/JPEG up to 10 MB.

## Tests

```powershell
python -m pytest -q
```

The lightweight suite checks preprocessing, empty input, IDX dimensions and label mapping, prediction tensor/output contracts, missing-model behavior, and CSV export. `python -m scripts.test_prediction` performs an inference smoke test after the real model and dataset are available.

## Evaluation results

The trained checkpoint was evaluated on all 20,800 samples in the official EMNIST Letters test split. Measured test accuracy is **93.5865%**, macro precision **93.6793%**, macro recall **93.5865%**, and macro F1 **93.5586%**. The full per-class classification report and confusion matrix are in `reports/evaluation_metrics.json`; rounded training and validation curves are in `reports/training_history.json`.

## Screenshots

The Streamlit dashboard is available locally after setup with `streamlit run app.py`. Screenshots can be captured from the Overview, Draw & Predict, and Model Analytics views after launching the app.

## Project layout

```text
app.py                    Streamlit entry point
config.py                 Paths, classes, and training defaults
src/dataset.py            EMNIST IDX parsing and stratified split
src/preprocessing.py      Shared 28x28 image normalization
src/model.py              CNN architecture
src/training.py           Reproducible full/smoke training
src/evaluation.py         Official held-out evaluation
src/inference.py          Model loading and prediction contracts
pages/                    Streamlit workflows
scripts/                  Download, train, evaluate, smoke inference
tests/                    Automated tests
assets/style.css          Application styling
data/ models/ reports/    Local generated artifacts
```

## 📸 Project Screenshots

### 1. Home Dashboard
![Overview]
<img width="1917" height="911" alt="overview 1" src="https://github.com/user-attachments/assets/6654d631-bad1-4a0f-bfe3-81dc693c90fc" />
<img width="1908" height="911" alt="overview 2" src="https://github.com/user-attachments/assets/6d7a70f4-9f84-4633-b857-6ac09c5ff606" />

### 2.Draw & Predict
![Prediction]
<img width="1915" height="908" alt="Draw 1" src="https://github.com/user-attachments/assets/4aa44b8d-2d23-4134-ba99-092e7190f457" />
<img width="1915" height="908" alt="Draw 1" src="https://github.com/user-attachments/assets/b56a642d-13d6-44f3-9d88-f1efb51f4137" />




## GitHub

From this project directory, create the repository on GitHub as `CodeAlpha_HandwrittenCharacterRecognition`, then:

```powershell
git init
git add .
git commit -m "Build GlyphSense handwritten character recognition"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/CodeAlpha_HandwrittenCharacterRecognition.git
git push -u origin main
```

The ignore rules exclude environments, downloaded data, and trained model files. Review `git status` before pushing.

## Future improvements

Calibrate confidence on a held-out calibration set, improve rejection of multi-character images, add more diverse handwriting data, and assess optional explainability methods against this architecture.

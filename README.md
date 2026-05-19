# 🔍 Deep Fake Detection using CSWin Transformer

> **Capstone Project — VIT Vellore, April 2024**  
> Siva Senthil Manikkam R · Atulya Prabhanjan M  
> Guide: Prof. R. Arumuga Arun

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9-orange.svg)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12-red.svg)](https://pytorch.org/)
[![Published](https://img.shields.io/badge/Published-Discover%20Computing%202025-green.svg)](https://doi.org/10.1007/s10791-025-09586-2)

---

## 📌 Overview

This project builds a deepfake image detection system comparing four deep learning architectures:

| Model | Accuracy | F1-Score |
|---|---|---|
| **CSWin Transformer** *(ours)* | **98.79%** | **98.72%** |
| MTCNN | 96.26% | 96.24% |
| InceptionV3 | 94.85% | 94.89% |
| Xception | 93.58% | 93.26% |

The **CSWin Transformer** uses a cross-shaped window self-attention mechanism, allowing it to capture both local and global facial inconsistencies — outperforming traditional CNNs while using 59% fewer parameters than a standard ViT (35M vs 85.8M).

This work was published in **Discover Computing (Springer, 2025)**: [https://doi.org/10.1007/s10791-025-09586-2](https://doi.org/10.1007/s10791-025-09586-2)

---

## 🗂️ Repository Structure

```
deepfake-detection/
│
├── notebooks/
│   ├── MTCNN.ipynb               # MTCNN-based detection
│   ├── InceptionV3.ipynb         # InceptionV3-based detection
│   ├── VGG16.ipynb               # VGG16 baseline
│   └── Xception.ipynb            # Xception-based detection
│
├── api/
│   └── main.py                   # FastAPI inference server for CSWin
│
├── results/
│   └── model_comparison.md       # Performance table & notes
│
├── requirements.txt
└── README.md
```

---

## 🧠 Models

### CSWin Transformer (Best Performer)
The backbone is the **Cross-Shaped Window Transformer**, which divides attention into horizontal and vertical stripes simultaneously. This reduces computational complexity from O(N²) to O(N√N) compared to standard ViTs, while preserving the ability to detect subtle deepfake artifacts across the full image.

The inference API (`api/main.py`) is a FastAPI server built around this model.

### CNN Baselines
Each notebook is standalone and trains the respective model from scratch on the [Deep Fake Face Detection dataset](https://www.kaggle.com/datasets/vasubhut/deep-fake-face-detection) (190K images, 256×256px).

---

## 📦 Dataset

**Primary:** [Deep Fake Face Detection](https://www.kaggle.com/datasets/vasubhut/deep-fake-face-detection) (Kaggle)

| Split | Real | Fake | Total |
|---|---|---|---|
| Train | 70,001 | 70,001 | 140,002 |
| Validation | 19,787 | 19,641 | 39,428 |
| Test | 5,413 | 5,492 | 10,905 |

**Secondary (robustness eval):** [Deepfake Detection Challenge — Face Images](https://www.kaggle.com/datasets/vijaydevane/deepfake-detection-challenge-dataset-face-images)

After downloading, update the dataset paths in each notebook:
```python
TRAINING_DIR   = "path/to/dataset/Train"
VALIDATION_DIR = "path/to/dataset/Validation"
TEST_DIR       = "path/to/dataset/Test"
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/deepfake-detection.git
cd deepfake-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run a notebook
Open any notebook in `notebooks/` in Jupyter or Google Colab and update the dataset paths.

### 4. Run the inference API (CSWin)
> **Note:** Model weights (`cswinmodel.pkl`) are not included due to file size (~92 MB). To use the API, train the model using the CSWin architecture defined in `api/main.py` and save the weights.

```bash
cd api
uvicorn main:app --reload
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Upload an image, returns `["Real"/"Fake", confidence%]` |

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr = 0.001) |
| Loss | Binary Cross-Entropy |
| Dropout | 0.5 → 0.1 |
| Epochs | 100 |
| Batch size | 32 |
| Hardware | NVIDIA Tesla T4 (Google Colab) |

---

## 📄 Publication

This work was accepted and published in **Discover Computing (Springer), 2025**:

> Magesh, A.P., Ramakrishnan, S.S.M., Arun, R.A., Priyanka, N., Kartheek, M.N. *Building an efficient Deep Fake detection system using the recognition capabilities of convolutional neural networks and transformers.* Discover Computing 28, 99 (2025). https://doi.org/10.1007/s10791-025-09586-2
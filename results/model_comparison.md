# Model Performance Results

All models trained from scratch on the [Deep Fake Face Detection dataset](https://www.kaggle.com/datasets/vasubhut/deep-fake-face-detection) for 100 epochs.

## Validation & Test Set Metrics

| Model | Val Acc | Val F1 | Test Acc | Test Sn | Test Sp | Test F1 |
|---|---|---|---|---|---|---|
| CSWin Transformer | 99.09% | 99.06% | **98.79%** | 99.23% | 98.28% | **98.72%** |
| MTCNN | 97.56% | 97.74% | 96.26% | 95.94% | 96.59% | 96.24% |
| InceptionV3 | 95.85% | 95.89% | 94.85% | 94.87% | 94.83% | 94.89% |
| Xception | 95.08% | 94.76% | 93.58% | 92.60% | 93.94% | 93.26% |

## Computational Complexity vs Performance

| Model | Parameters | Model Size | F1 |
|---|---|---|---|
| VGG16 (baseline) | 138.8M | 528 MB | 94.0% |
| ResNet-50 (baseline) | 25.6M | 98 MB | 78.0% |
| MTCNN | **1.5M** | **14 MB** | 96.24% |
| InceptionV3 | 22.85M | 87 MB | 94.89% |
| Xception | 23.42M | 89 MB | 93.26% |
| **CSWin Transformer** | **35M** | **92 MB** | **98.72%** |
| ViT (reference) | 85.8M | 344 MB | 99.0% |

> CSWin achieves near-ViT accuracy with **59% fewer parameters** than standard ViT.

## Robustness Evaluation (CSWin only)

| Test Condition | Accuracy | F1 |
|---|---|---|
| Standard test set | 98.79% | 98.72% |
| Pose variation | 96.70% | 96.23% |
| Varying lighting | 95.50% | 95.53% |
| Occluded faces | 62.25% | 64.52% |
| Combined | 97.86% | 97.53% |

> Occlusion is a known weakness — the training data contained no occluded faces.

## Cross-Dataset Generalisation

| Configuration | Acc (original test) | Acc (DFDC dataset) |
|---|---|---|
| Without fine-tuning | 98.79% | 75.51% |
| With fine-tuning (10 epochs) | 97.22% | **93.35%** |

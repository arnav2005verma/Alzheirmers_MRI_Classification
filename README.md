> **"A Deep-Ensemble Framework for Multi-Stage Alzheimer's Disease Classification from Structural MRI Using EfficientNetV2-S and XGBoost"**  
> Arnav Verma, Vivek Verma, Kuljeet Singh — Christ University, Delhi-NCR

A hybrid AI pipeline that classifies brain MRI scans into four Alzheimer's stages:
- Non-Demented (ND)
- Very Mild Demented (VMD)
- Mild Demented (MD)
- Moderate Demented (MOD)

## Results

### Overall Performance
| Metric | Value |
|--------|-------|
| Training Accuracy | **95%** |
| Validation Accuracy | **92%** |
| Test Accuracy | **93%** |
| Macro Precision | **0.93** |
| Macro Recall | **0.93** |
| Macro F1-Score | **0.93** |
| Total Test Samples | **1,396** |

### Class-wise Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Non-Demented | 0.89 | 0.87 | 0.88 | 392 |
| Very Mild Demented | 0.89 | 0.87 | 0.88 | 392 |
| Mild Demented | 0.93 | 0.97 | 0.95 | 392 |
| Moderate Demented | **1.00** | **1.00** | **1.00** | 292 |

### ROC-AUC Scores
| Class | AUC |
|-------|-----|
| Mild Demented (Class 0) | **1.00** |
| Moderate Demented (Class 1) | **1.00** |
| Non-Demented (Class 2) | **0.97** |
| Very Mild Demented (Class 3) | **0.98** |

### Key Highlights
- Moderate Demented cases achieved **perfect classification** (Precision = Recall = F1 = 1.00), reflecting highly distinct structural patterns in late-stage AD
- Minor overlap observed between Non-Demented and Very Mild Demented stages, which is **clinically expected** due to subtle structural similarities in early disease progression
- t-SNE projections confirmed **strong feature separability** across all four classes
- KDE probability plots showed **well-calibrated, high-confidence predictions** with probability mass concentrated at 0 and 1
- Model was evaluated exclusively on **non-augmented, original MRI scans** to ensure real-world applicability

## Method
1. **EfficientNetV2-S** (pretrained on ImageNet) → deep feature extraction (1280-dim)
2. **TruncatedSVD** → dimensionality reduction to 100 components
3. **XGBoost** → final multi-class classification

## Dataset
Dataset: [Alzheimer's Disease MRI Dataset — Mendeley Data](https://data.mendeley.com/datasets/ch87yswbz4/1)  
40,000+ structural MRI images across 4 classes. Not included in this repo due to size.

## Setup
```bash
git clone https://github.com/arnav2005verma/alzheimers-mri-classification
cd alzheimers-mri-classification
pip install -r requirements.txt
```

Download the dataset and place it in a `data/` folder with subfolders:
`data/MildDemented/`, `data/ModerateDemented/`, `data/NonDemented/`, `data/VeryMildDemented/`

## Requirements
- Python 3.10
- PyTorch 2.0
- XGBoost
- scikit-learn
- Matplotlib / Seaborn

## Citation
If you use this work, please cite:
> Verma, A., Verma, V., & Singh, K. (2026). A Deep-Ensemble Framework for Multi-Stage Alzheimer's Disease Classification from Structural MRI Using EfficientNetV2-S and XGBoost. Christ University, Delhi-NCR.

## License
MIT License

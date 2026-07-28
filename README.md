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
| Non-Demented | 0.89 | 0.87 | 0.88 | 320 |
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
- Moderate Demented cases achieved **perfect classification** (Precision = Recall = F1 = 1.00) — see Limitations below regarding how this figure should be interpreted
- Minor overlap observed between Non-Demented and Very Mild Demented stages, which is **clinically expected** due to subtle structural similarities in early disease progression
- t-SNE projections confirmed **strong feature separability** across all four classes
- KDE probability plots showed **well-calibrated, high-confidence predictions** with probability mass concentrated at 0 and 1

## Limitations

- **Splits are not grouped by subject.** `load_data()` draws train/validation/test from the augmented dataset directory (`AGD`) using `train_test_split(..., stratify=labels)`, which stratifies by class only. Because the augmented set contains multiple images derived from each original scan, derivatives of the same source scan can appear in more than one split. The metrics above should therefore be read as within-dataset performance, **not** as evidence of subject-level generalisation.
- **The reported metrics were not computed on original, non-augmented scans.** The `test_single_images()` routine does read from the original directory (`OD`), but it loads a single image per class for visualisation and contributes to no reported metric.
- The gap between final training accuracy (0.989) and validation accuracy (0.911), and the perfect scores on the smallest class, are both consistent with the leakage described above.

## Method
1. **EfficientNetV2-S** (pretrained on ImageNet) → deep feature extraction (1280-dim)
2. **TruncatedSVD** → dimensionality reduction to 100 components
3. **XGBoost** → final multi-class classification

## Dataset
Dataset: [Alzheimer's Disease MRI Dataset — Mendeley Data](https://data.mendeley.com/datasets/ch87yswbz4/1)  
40,000+ structural MRI images across 4 classes. Not included in this repo due to size.

## Setup

```bash
git clone https://github.com/arnav2005verma/Alzheirmers_MRI_Classification
cd Alzheirmers_MRI_Classification
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

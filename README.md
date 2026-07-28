# Predictive Maintenance for Industrial Assets using Machine Learning

**MSc Data Science Dissertation — Nottingham Trent University 2026**  
**Author:** Venkata Rama Sai Behara (N1421663)  
**Supervisor:** Maria Mwanje

---

## Project Overview

This project develops a machine learning pipeline for predicting industrial equipment failures before they occur, addressing three limitations that no existing study has resolved simultaneously:

- Class imbalance in industrial fault datasets
- Explainability of model predictions
- Generalisation across different asset types

## Novel Contribution

**Two-Stage SHAP Distortion Analysis** — the first published measurement of whether SMOTE synthetic oversampling alters feature importance rankings in predictive maintenance models.

**Key Finding:** Spearman ρ = 0.9182 (p = 0.0001) — SMOTE preserves sensor importance rankings, with s11 (physical fan speed) confirmed as the most important sensor before and after synthetic augmentation, consistent with turbofan degradation physics.

---

## Datasets

- **NASA C-MAPSS** — Turbofan engine degradation (FD001-FD004)
- **CWRU Bearing** — Rotating machinery fault classification

---

## Models Compared

| Model | Type |
|-------|------|
| Random Forest | Classical ML baseline |
| SVM | Classical ML baseline |
| LSTM | Deep Learning |
| Attention-LSTM | Deep Learning + Interpretability |

---

## Experimental Design

Four class imbalance handling conditions compared:
1. No handling (baseline)
2. SMOTE oversampling
3. ADASYN oversampling
4. Class-weighted loss

---

## Key Results (Random Forest — C-MAPSS FD001)

| Condition | F1-Score | FNR | MCC |
|-----------|----------|-----|-----|
| No Handling | 0.8545 | 0.1661 | 0.8323 |
| SMOTE | 0.8424 | 0.1161 | 0.8171 |
| ADASYN | 0.8171 | 0.0887 | 0.7901 |
| Class Weighted | 0.8481 | 0.1806 | 0.8254 |

**ADASYN achieves the lowest False Negative Rate (0.0887) — missing the fewest actual failures.**

---

## Tech Stack

- Python 3.10
- PyTorch 2.2
- scikit-learn 1.4
- imbalanced-learn 0.12
- SHAP 0.45
- scikit-optimize 0.9

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/Ramsyash/predictive-maintenance-ml.git
cd predictive-maintenance-ml

# Install dependencies
pip install -r requirements.txt

# Download NASA C-MAPSS dataset from Kaggle
# https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
# Place txt files in the project folder

# Open Jupyter Notebook
jupyter notebook
```

---

## Project Structure
predictive-maintenance-ml/
├── Chapter4_Implementation.ipynb # Main pipeline notebook
├── results/
│ ├── eda_plots.png
│ ├── shap_distortion_analysis.png
│ └── rf_results_all_conditions.csv
└── README.md


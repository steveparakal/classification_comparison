# Lazy Classification across 3 Datasets (Sklearn vs FCA/FCALC)

This repository contains a comparative study of classic supervised classifiers and lazy FCA-based classifiers across three real-world tabular datasets. The project compares standard scikit-learn models with classifiers based on Formal Concept Analysis (FCA), implemented using the FCALC library. All experiments (data download, preprocessing, training, tuning, and evaluation) are implemented in a single Jupyter notebook.

## Files in this Repository

- `Classification_Comparison.ipynb` — Main notebook with all experiments
- `Classification_Comparison_Report.pdf` — Written project report
- `README.md` — Project description
- `LICENSE` — Apache License 2.0

## Project Objective

The goal of this project is to compare:

1. Traditional supervised machine learning classifiers
2. Lazy FCA-based classifiers

Models are evaluated using:
- Accuracy
- Weighted F1-score
- Cross-validation on training data
- Final evaluation on a held-out 80/20 test split

## Datasets

All datasets are downloaded automatically from Kaggle using the `opendatasets` package.

1. **Heart Attack Analysis & Prediction** — `rashikrahmanpritom/heart-attack-analysis-prediction-dataset`
2. **Stroke Prediction Dataset** — `fedesoriano/stroke-prediction-dataset`
3. **Water Potability Dataset** — `adityakadiwal/water-potability`

A Kaggle account and `kaggle.json` API key are required.

## Models Compared

### Traditional Machine Learning Models

Each model follows the same pipeline:
1. Train/test split (80/20)
2. Hyperparameter tuning with GridSearchCV (5-fold CV)
3. Evaluation using Accuracy and Weighted F1-score

Models used: Decision Tree, Random Forest, Logistic Regression (with StandardScaler), k-Nearest Neighbors (with StandardScaler)

### FCA-Based Lazy Classifiers (FCALC)

Implemented using the [FCALC library](https://github.com/AndrewDiv/FCALC).

1. **BinarizedBinaryClassifier** — Requires binary (0/1) features with custom binarization rules. Decision methods: `standard`, `standard-support`, `ratio-support`
2. **PatternBinaryClassifier** — Works with non-binarized data, requires categorical feature indices (`cat_list`)

FCA experiments may use dataset subsets to reduce runtime.

## Results

### Standard Classifiers

| Dataset | Metric | Decision Tree | Random Forest | Logistic Regression | k-NN |
|---|---|---|---|---|---|
| **Heart Attack** | CV Accuracy | 0.756 | 0.805 | 0.822 | 0.839 |
| | CV F1 Score | 0.758 | 0.805 | 0.821 | 0.836 |
| | Test Accuracy | 0.852 | 0.852 | 0.885 | 0.885 |
| | Test F1 Score | 0.852 | 0.853 | 0.885 | 0.885 |
| **Stroke** | CV Accuracy | 0.942 | 0.954 | 0.955 | 0.955 |
| | CV F1 Score | 0.929 | 0.932 | 0.933 | 0.933 |
| | Test Accuracy | 0.931 | 0.939 | 0.939 | 0.940 |
| | Test F1 Score | 0.912 | 0.910 | 0.910 | 0.914 |
| **Water Potability** | CV Accuracy | 0.626 | 0.669 | 0.606 | 0.655 |
| | CV F1 Score | 0.595 | 0.630 | 0.459 | 0.621 |
| | Test Accuracy | 0.610 | 0.689 | 0.628 | 0.646 |
| | Test F1 Score | 0.594 | 0.661 | 0.485 | 0.615 |

### FCA Binarized Binary Classifier

| Dataset | Metric | Score |
|---|---|---|
| **Heart Attack** | CV Accuracy | 0.558 |
| | CV F1 Score | 0.527 |
| | Test Accuracy | 0.525 |
| | Test F1 Score | 0.407 |
| **Stroke** | CV Accuracy | 0.873 |
| | CV F1 Score | 0.815 |
| | Test Accuracy | 0.882 |
| | Test F1 Score | 0.827 |
| **Water Potability** | CV Accuracy | 0.583 |
| | CV F1 Score | 0.445 |
| | Test Accuracy | 0.601 |
| | Test F1 Score | 0.463 |

### FCA Pattern Classifier

| Dataset | Metric | Score |
|---|---|---|
| **Heart Attack** | CV Accuracy | 0.773 |
| | CV F1 Score | 0.768 |
| | Test Accuracy | 0.852 |
| | Test F1 Score | 0.853 |
| **Stroke** | CV Accuracy | 0.747 |
| | CV F1 Score | 0.763 |
| | Test Accuracy | 0.685 |
| | Test F1 Score | 0.711 |
| **Water Potability** | CV Accuracy | 0.624 |
| | CV F1 Score | 0.537 |
| | Test Accuracy | 0.580 |
| | Test F1 Score | 0.461 |

For full analysis and discussion, see [Classification_Comparison_Report.pdf](./Classification_Comparison_Report.pdf).

## How to Run

### Option 1 — Google Colab

1. Open `Classification_Comparison.ipynb` in Colab
2. Install dependencies:
   ```bash
   pip install opendatasets
   ```
3. Clone FCALC:
   ```bash
   git clone https://github.com/AndrewDiv/FCALC
   ```
4. Upload `kaggle.json` to `/root/.kaggle/kaggle.json` and set permissions:
   ```bash
   chmod 600 /root/.kaggle/kaggle.json
   ```
5. Run all cells

### Option 2 — Run Locally

Requirements: Python 3.9+, Jupyter Notebook, Kaggle credentials

```bash
git clone https://github.com/steveparakal/Classification_Comparison.git
cd Classification_Comparison
pip install numpy pandas scikit-learn opendatasets
git clone https://github.com/AndrewDiv/FCALC
jupyter notebook Classification_Comparison.ipynb
```

> **Note:** FCALC is not available on PyPI and must be installed via `git clone` as shown above.

## Reproducibility

- `random_state=42` used where applicable
- FCA experiments may use dataset subsets to reduce runtime
- Results may vary slightly on full datasets

## License

Apache License 2.0

## Acknowledgements

- [FCALC library](https://github.com/AndrewDiv/FCALC) by AndrewDiv
- Datasets provided by Kaggle

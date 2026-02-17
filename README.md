# Lazy Classification across 3 Datasets (Sklearn vs FCA/FCALC)

This repository contains a comparative study of classic supervised classifiers and lazy FCA-based classifiers across three real-world tabular datasets. The project compares standard scikit-learn models with classifiers based on Formal Concept Analysis (FCA), implemented using the FCALC library. All experiments (data download, preprocessing, training, tuning, and evaluation) are implemented in a single Jupyter notebook.

## Files in this Repository

- OSDA_BIG_HW.ipynb        Main notebook with all experiments
- Big_HW_Report_OSDS.pdf  Written project report
- README.md               Project description
- LICENSE                 Apache License 2.0

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

All datasets are downloaded automatically from Kaggle using the opendatasets package.

1. Heart Attack Analysis & Prediction
   Target column: output
   Kaggle dataset: rashikrahmanpritom/heart-attack-analysis-prediction-dataset

2. Stroke Prediction Dataset
   Target column: stroke
   Kaggle dataset: fedesoriano/stroke-prediction-dataset

3. Water Potability Dataset
   Target column: Potability
   Kaggle dataset: adityakadiwal/water-potability

A Kaggle account and kaggle.json API key are required.

## Models Compared

### Traditional Machine Learning Models

Each model follows the same pipeline:
1. Train/test split (80/20)
2. Hyperparameter tuning with GridSearchCV (5-fold CV)
3. Evaluation using Accuracy and Weighted F1-score

Models used:
- Decision Tree
- Random Forest
- Logistic Regression (with StandardScaler)
- k-Nearest Neighbors (with StandardScaler)

### FCA-Based Lazy Classifiers (FCALC)

The FCA-based classifiers are implemented using the FCALC library:
https://github.com/AndrewDiv/FCALC

Classifiers evaluated:

1. BinarizedBinaryClassifier
   - Requires binary (0/1) features
   - Uses custom binarization rules
   - Decision methods:
     standard
     standard-support
     ratio-support

2. PatternBinaryClassifier
   - Works with non-binarized data
   - Requires categorical feature indices (cat_list)

FCA experiments may use dataset subsets to reduce runtime.

## How to Run

Option 1 — Google Colab

1. Open OSDA_BIG_HW.ipynb in Colab
2. Install dependencies:
   pip install opendatasets
3. Clone FCALC:
   git clone https://github.com/AndrewDiv/FCALC
4. Upload kaggle.json to:
   /root/.kaggle/kaggle.json
5. Set permissions:
   chmod 600 /root/.kaggle/kaggle.json
6. Run all cells

Option 2 — Run Locally

Requirements:
- Python 3.9+
- Jupyter Notebook
- Kaggle credentials

Installation:
git clone https://github.com/steveparakal/OSDS-Big-HW.git
cd OSDS-Big-HW
pip install numpy pandas scikit-learn opendatasets
git clone https://github.com/AndrewDiv/FCALC

Run:
jupyter notebook OSDA_BIG_HW.ipynb

## Reproducibility

- random_state=42 used where applicable
- FCA experiments may use dataset subsets
- Results may vary slightly on full datasets

## Output

The notebook prints:
- Best hyperparameters
- Cross-validation metrics
- Test Accuracy and Weighted F1-score

Final discussion is provided in Big_HW_Report_OSDS.pdf.

## License

Apache License 2.0

## Acknowledgements

- FCALC library by AndrewDiv
- Datasets provided by Kaggle

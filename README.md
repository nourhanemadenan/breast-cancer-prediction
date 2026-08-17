<div align="center">

# Breast Cancer Prediction

**Six classifiers benchmarked head-to-head on the Wisconsin diagnostic dataset**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat-square)](https://xgboost.readthedocs.io)
[![Best accuracy](https://img.shields.io/badge/best%20accuracy-98.2%25-2ea043?style=flat-square)](#results)

</div>

---

## Overview

Classify a breast tumour as **malignant or benign** from 30 cell-nucleus measurements (radius, texture,
concavity, symmetry…) computed from a digitized fine-needle aspirate.

The point of the project isn't one model — it's the **comparison**. Six algorithms, same split, same
preprocessing, cross-validated, then ranked. Which family actually wins on a small, clean, well-separated
tabular dataset, and does gradient boosting earn its complexity here?

## Pipeline

1. Load the Wisconsin diagnostic dataset (`sklearn.datasets`)
2. EDA — class balance, feature correlation heatmap, distribution plots
3. Standardize features (`StandardScaler`) — mandatory for KNN and SVC, harmless for the trees
4. Stratified train/test split
5. Train six classifiers, tune with `GridSearchCV` / K-fold CV
6. Score on the held-out test set and rank

## Results

Test-set accuracy, 114 held-out samples:

| Rank | Model | Accuracy |
|:--:|---|--:|
| 1 | Support Vector Classifier | **98.25%** |
| 2 | Logistic Regression | 97.37% |
| 3 | Decision Tree | 96.49% |
| 3 | Random Forest | 96.49% |
| 5 | K-Nearest Neighbors *(k=8, CV-selected)* | 95.61% |
| 5 | XGBoost | 95.61% |

> **Note on the notebook's summary table.** The `models` DataFrame near the end of the notebook pairs its
> `Model` list with a `Score` list whose 2nd and 3rd entries are in the opposite order (`knn_acc` sits in
> the `SVC` row and `svc_acc` in the `KNN` row). The table above is the corrected mapping, verified
> against the per-model `classification_report` output printed higher up in the same notebook — SVM 0.98,
> KNN 0.96. Fixing the list order in the notebook is a one-line change and is worth doing.

**Takeaway:** on 569 samples of well-separated, standardized numeric features, the margin-based and linear
models win. SVC and plain logistic regression beat both the ensemble and the boosted tree — extra capacity
buys nothing when the classes are nearly linearly separable, and boosting has more room to overfit. 5-fold
CV scores for the top models sit in the 0.92–0.98 band, so the gap between the top four is within noise;
the honest reading is *"everything works here, so ship the simplest thing."*

Recall on the malignant class matters more than raw accuracy in this domain — the per-class
precision/recall breakdowns are printed in the notebook for each model.

## Repo layout

```
notebooks/
  breast_cancer_models.ipynb   EDA → preprocessing → 6 models → comparison
requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/breast_cancer_models.ipynb
```

No data download needed — the dataset ships with scikit-learn. The trained model is pickled at the end
of the notebook; the `.pkl` is gitignored, rerun to regenerate.

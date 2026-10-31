# Credit Card Fraud Detection

Supervised learning pipeline on a highly imbalanced fraud dataset.
SMOTE is used on the training set only. Models are compared with
Precision, Recall, and ROC-AUC (accuracy is not used as the main metric).

## Dataset

[Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Download `creditcard.csv` and put it in this folder (it is gitignored).

## How to run

```bash
pip install -r requirements.txt
jupyter notebook
```

Open `code.ipynb` and run all cells.

## Approach

1. EDA (class imbalance, no missing values, Amount by class)
2. Train/test split with stratification
3. Scale features (fit on train only)
4. SMOTE on train only
5. Logistic Regression and Random Forest
6. GridSearchCV on Logistic Regression (`C`)

## Results (test set)

| Model | Precision | Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.058 | 0.918 | 0.971 |
| Logistic Regression (tuned, C=0.01) | 0.059 | 0.918 | 0.971 |
| Random Forest | 0.871 | 0.827 | 0.968 |

Tuning selected `C=0.01`. The tuned Logistic Regression model slightly improves ROC-AUC, but Random Forest has much higher precision.
Random Forest is the better deployed model: similar ROC-AUC, much higher precision.

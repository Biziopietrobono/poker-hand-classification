# Poker Hand Classification

A comparative study of four supervised classifiers on the UCI Poker Hand dataset
(1,025,010 hands, 10 features, 10 classes). Individual project for the Machine
Learning course at Sapienza, November 2025.

The full written report is in [report/Poker_Hand_Classification_Report.pdf](report/Poker_Hand_Classification_Report.pdf).

## Results (test set)

| Model | Accuracy | Precision | Recall | F1 (weighted) | ROC-AUC (micro) |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.757 | 0.745 | 0.757 | 0.732 | 0.977 |
| Decision Tree | 0.665 | 0.665 | 0.665 | 0.665 | 0.814 |
| Softmax Regression | 0.095 | 0.462 | 0.095 | 0.141 | 0.370 |
| Gaussian Naive Bayes | 0.013 | 0.447 | 0.013 | 0.022 | 0.588 |

Random Forest is the best model on every metric. The problem is strongly
non-linear: a hand is defined by how the five cards combine (pairs, runs,
matching suits), not by any single card. Tree ensembles can capture those
interactions; a linear model and a model that assumes independent features cannot.
Feature importance confirms that card ranks (C1–C5) matter more than suits (S1–S5).

## Method

- Stratified split: 70% train / 15% validation / 15% test
- Features standardised with `StandardScaler`
- Hyperparameters tuned with `GridSearchCV` and stratified k-fold cross-validation,
  with `class_weight='balanced'` among the options to counter the class imbalance
  (class 0 is over 50% of the data, class 9 under 0.001%)
- Evaluation: per-class precision, recall and F1, confusion matrices,
  multiclass ROC-AUC, feature importance, learning curves

| Model | Best parameters |
| --- | --- |
| Gaussian Naive Bayes | `var_smoothing=1e-9` |
| Softmax Regression | `C=0.01`, `class_weight='balanced'`, `solver='saga'` |
| Decision Tree | `class_weight='balanced'`, `max_depth=None` |
| Random Forest | `n_estimators=200`, `max_depth=20`, `class_weight='balanced'` |

## Repository

```
notebooks/01_data_exploration.ipynb   dataset loading and class distribution
notebooks/02_model_comparison.ipynb   preprocessing, tuning, training, evaluation
figures/                              confusion matrices, feature importance, learning and ROC curves
report/                               full written report (PDF)
```

## Running it

```bash
pip install ucimlrepo scikit-learn pandas matplotlib seaborn
jupyter notebook notebooks/01_data_exploration.ipynb
```

The dataset is downloaded at runtime from the UCI repository (`fetch_ucirepo(id=158)`).
Training the Random Forest on the full dataset takes a few hours on a laptop.

## Author

Fabrizio Pietrobono — MSc Computer Science & AI, Sapienza Università di Roma.
[LinkedIn](https://www.linkedin.com/in/fabriziopietrobono)

# Poker Hand Classification

A comparative study of supervised classifiers on the UCI **Poker Hand** dataset
(1,025,010 instances, 10 features, 10 classes), done for the Machine Learning
course at Sapienza.

**The interesting result is not the best accuracy — it is the trade-off.**
A regularised Random Forest scores *lower* raw accuracy than an unconstrained one
(68.9% vs 75.9%), but doubles balanced accuracy and macro F1. On a dataset where
classes 0 and 1 are over 90% of the data, raw accuracy mostly measures how well a
model predicts "nothing in hand".

## Results

| Model | Accuracy | Balanced accuracy | Macro F1 |
| --- | --- | --- | --- |
| Logistic Regression (softmax) | 7.7% | — | 0.04 |
| Decision Tree | 66.0% | — | 0.25 |
| Random Forest | 75.9% | — | — |
| Random Forest, regularised (`max_depth=20`, `min_samples_leaf=10`) | 68.9% | 0.40 | 0.37 |

The linear baseline collapses: card suit and rank carry no linear signal about the
hand they form. Tree models pick up the non-linear structure but overfit towards the
majority classes. Constraining depth and leaf size is what buys back the rare hands.

<p align="center">
  <img src="figures/learning_curves_all.png" width="620" alt="Learning curves, regularised Random Forest">
</p>

Training F1 falls from 0.91 to 0.85 as data grows while validation F1 rises from 0.21
to 0.33 — the gap narrows, which is the shape you want to see.

## Method

- Stratified split: 70% train / 15% validation / 15% test
- `StandardScaler` on the features, `class_weight='balanced'` for the imbalance
- Metrics: precision, recall, F1 per class, plus balanced accuracy and macro F1,
  since plain accuracy is misleading on this distribution

## Repository

```
notebooks/01_data_exploration.ipynb   dataset loading and class distribution
notebooks/02_model_comparison.ipynb   preprocessing, training, evaluation
figures/                              confusion matrices, feature importance, learning and ROC curves
```

## Running it

```bash
pip install ucimlrepo scikit-learn pandas matplotlib seaborn
jupyter notebook notebooks/01_data_exploration.ipynb
```

The dataset is downloaded at runtime from the UCI repository (`fetch_ucirepo(id=158)`),
so there is nothing to download by hand.

## Author

Fabrizio Pietrobono — MSc Computer Science & AI, Sapienza Università di Roma.
Individual project.

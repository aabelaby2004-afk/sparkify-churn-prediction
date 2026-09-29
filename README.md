# Sparkify User Churn Prediction

End-to-end machine learning project that predicts **future churn** for a music-streaming service (Sparkify) using only user behavior observed **before** a fixed prediction cutoff, which avoids data leakage.

## Problem
Given event logs (songs played, thumbs up/down, upgrades, downgrades, etc.), estimate which users will cancel their subscription after the cutoff date.

- **Churn event:** `Cancellation Confirmation` (a `Downgrade` is a subscription change, not churn)
- **Prediction cutoff:** 2018-11-01. Users who churned before it are excluded; only pre-cutoff behavior is used as input.

## Approach
1. Data understanding and churn definition
2. Leakage-safe target construction
3. EDA: engagement, session gaps, feedback, social, subscription, content diversity
4. Feature engineering per user (songs played, session duration, thumbs-down rate, friend/playlist adds, upgrades/downgrades, unique artists, current subscription level)
5. Redundancy check via correlation, then feature pruning
6. Models: Logistic Regression (baseline) vs Random Forest, both with `class_weight="balanced"`
7. Stratified 5-fold cross-validation (accuracy, precision, recall, F1, ROC-AUC)
8. Threshold analysis, ROC and precision-recall curves, confusion matrix
9. Coefficient interpretation

## Results
| Metric (5-fold CV) | Logistic Regression | Random Forest |
|---|---|---|
| Recall | 59.0% | 13.0% |
| F1 | 35.7% | 16.4% |
| ROC-AUC | 0.761 | 0.711 |

Logistic Regression identified far more churners (59% vs 13% recall) and had higher F1 and ROC-AUC than the Random Forest configuration used here. Absolute F1 is modest, which is expected with very few churners and behavior-only features.

## Key findings
- Churn is not simply a low-activity problem: churned users can show substantial engagement.
- Thumbs-down rate is positively associated with churn.
- Subscription activity (upgrades/downgrades) carries signal.
- These are associations, not causal effects.

## Limitations
- Few churners, so estimates can be unstable.
- The 0.60 threshold was chosen on the same cross-validated predictions used for evaluation, so treat it as an analytical choice, not an unbiased production threshold.

## Run it
```bash
pip install -r requirements.txt
# Put mini_sparkify_event_data.json in the data/ folder
jupyter notebook Sparkify_Churn_Model.ipynb
```

## Tech
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

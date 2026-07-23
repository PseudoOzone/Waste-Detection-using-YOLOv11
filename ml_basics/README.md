# ML Fundamentals Lab

These four small, original projects cover the main machine-learning task types likely to appear in a fresher interview. Every example uses a built-in or synthetic dataset, a fixed random seed, a train/test or validation split where appropriate, and machine-readable metric output.

## 1. Binary classification

Run:

```bash
python ml_basics/01_classification.py
```

The model predicts one of two classes using logistic regression on the scikit-learn breast-cancer dataset.

What it teaches:

- supervised learning
- features `X` and target `y`
- stratified train/test splitting
- feature scaling
- logistic regression
- class weighting
- accuracy, precision, recall, F1, ROC-AUC
- confusion matrices

Interview explanation:

> Classification predicts a discrete category. I split the data before fitting any preprocessing, then placed scaling and logistic regression inside one pipeline. That prevents information from the test set leaking into the scaler. I used stratification so the class proportion remains similar in both splits.

## 2. Regression

Run:

```bash
python ml_basics/02_regression.py
```

The model predicts a continuous disease-progression score using ridge regression.

What it teaches:

- continuous targets
- linear regression with L2 regularisation
- MAE, RMSE and R²
- why scaling matters for regularised linear models

Interview explanation:

> Regression predicts a number rather than a category. Ridge adds an L2 penalty that discourages excessively large coefficients and can improve generalisation when features are correlated.

Metric interpretation:

- **MAE:** average absolute prediction error, in target units
- **RMSE:** penalises large errors more strongly
- **R²:** fraction of target variance explained relative to predicting the mean

## 3. Clustering

Run:

```bash
python ml_basics/03_clustering.py
```

K-Means groups wine samples without using their class labels during training.

What it teaches:

- unsupervised learning
- distance-based grouping
- why scaling is essential for K-Means
- centroid assignment
- silhouette score
- the difference between learning with and without labels

Interview explanation:

> K-Means alternates between assigning each point to its closest centroid and recalculating centroids. It requires the number of clusters in advance and works best for roughly compact, similarly sized groups. Feature scale affects Euclidean distance, so I standardised the data first.

The adjusted Rand index is included only to understand how discovered clusters compare with known dataset classes. Those labels are not supplied to K-Means.

## 4. Anomaly detection

Run:

```bash
python ml_basics/04_anomaly_detection.py
```

Isolation Forest learns a baseline from synthetic normal employee-access activity and flags unusual activity.

Features include:

- files accessed
- downloaded data volume
- hour of activity
- sensitive resources accessed
- number of countries seen in 24 hours

What it teaches:

- rare-event detection
- semi-supervised anomaly detection
- Isolation Forest
- alert thresholds and contamination assumptions
- false positives and alert fatigue
- precision versus recall in security

Interview explanation:

> Isolation Forest isolates unusual points through random feature splits. Anomalies tend to require fewer splits because they lie in sparse regions. I trained the baseline only on known-normal activity, then evaluated on mixed normal and anomalous traffic.

Important limitation:

The synthetic anomalies are intentionally easier than real insider threats. A production system would require user and peer-group baselines, temporal validation, drift monitoring, contextual features, threshold calibration and analyst feedback.

## Run all examples

```bash
python ml_basics/01_classification.py
python ml_basics/02_regression.py
python ml_basics/03_clustering.py
python ml_basics/04_anomaly_detection.py
```

Outputs are written to `artifacts/*.json`.

## What to know before claiming these projects

Be able to explain:

1. Why data is split before training.
2. What data leakage means.
3. Why scaling is inside a pipeline.
4. The difference between classification and regression.
5. Why accuracy can fail on imbalanced data.
6. Precision versus recall.
7. Supervised versus unsupervised learning.
8. What regularisation does.
9. Why K-Means needs scaling and a chosen `k`.
10. Why anomaly-detection thresholds create a precision–recall trade-off.

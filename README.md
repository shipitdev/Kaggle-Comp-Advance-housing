# Advanced House Price Prediction Pipeline

I built this to move past the "single messy notebook" stage of data science. This project takes the Kaggle House Prices dataset and cleans it up using a modular pipeline, focusing on data safety and custom mathematical optimizations to handle the weird quirks of real estate pricing.

---

## Performance Summary

- **Mean Training RMSE:** 0.09855
- **Out-Of-Fold Validation RMSE:** 0.11006
- **Final Kaggle Leaderboard Score:** **0.12741** (Global Rank: **1651**)

---

## Repository Structure

```text
advanced-housing-pipeline/
├── datasets/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── __init__.py
│   ├── pipeline.py          # Custom scikit-learn transformer class
│   └── evaluate.py          # Data alignment, cross-validation, and modeling loop
├── requirements.txt         # Package dependencies
└── README.md

```

---

## Strategy

### 1. EDA

Before modeling, I analyzed the feature distributions to find "high-leverage" outliers skewing the regression.

- **The "Anomaly" Purge:** I identified luxury homes with massive square footage (`GrLivArea > 4000`) that sold at suspiciously low prices. These were statistical flukes that dragged the regression line off-course. Dropping these enabled the model to map the "normal" market more accurately.
- **Log-Normalization:** Prices and areas are naturally right-skewed. Applying a $\log(1+x)$ transform compressed these high-variance clusters, creating a more symmetric distribution that prevents extreme values from dominating the gradient updates.

### 2. Matrix & Mathematical Optimization

To get a competitive score, I had to optimize the math behind the feature matrix.

- **Dimensionality & Collinearity:** Standard regression (OLS) struggles when features are linearly dependent because the covariance matrix becomes ill-conditioned (near-singular). I optimized this by merging redundant features (like various porch types) into a single `TotalOutdoorSF` vector, preserving mathematical degrees of freedom and reducing matrix noise.
- **Non-Linear Feature Weighting:** I engineered `OverallGrade` as $(Qual^2 + Cond)$. Squaring the quality creates a non-linear weight, which helps the model treat premium properties with the exponential value they possess in the real market.
- **Hurdle Flags:** For sparse amenities, I decoupled them into binary "Has_Asset" flags. This allows the linear model to calculate a distinct intercept shift (a "step-jump" in value) before accounting for the continuous area footprint.

### 3. Linear Algebra & $L2$ Regularization

The core of the ensemble is a **Ridge Regression** model.

- **Tikhonov Regularization ($L2$):** By adding the penalty term $\alpha ||w||^2$ to the loss function, I ensured the weight matrix $w$ remained stable against multicollinearity.
- **Alpha Tuning:** I used `GridSearchCV` to find the $\alpha$ that achieves the optimal bias-variance tradeoff, effectively shrinking coefficients of noisy, redundant features to near-zero.

### 4. Model Architecture & Weighted Ensemble

I didn't rely on one "magic" model. My final prediction is a weighted linear combination of three distinct estimators:

- **Ridge Regression (60% weight):** My anchor. It handles the bulk of the work once the matrix is stabilized.
- **Gradient Boosting (XGBoost 25% + LightGBM 15%):** I integrated these to capture the non-linear, "bracket-style" splits that a straight-line regression cannot map.

---

## Quick Start

Ensure you have a Python environment ready, then install the dependencies:

```bash
pip install -r requirements.txt

```

To run the complete pipeline, process data, train the ensemble, and export the submission arrays:

```bash
python src/evaluate.py

```

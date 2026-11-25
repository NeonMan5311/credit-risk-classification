# Credit Risk Classification (Decision-Support Prototype)

> **Important Disclaimer:** This project is an educational and **decision-support demonstration only**. It is **not** validated for production lending, automated credit underwriting, or regulatory compliance, and should never be used as the sole basis for financial decisions affecting real applicants.

---

## 1. Problem Statement

- **Real-World Problem:** Financial institutions and micro-lenders must assess whether a loan applicant is likely to repay a loan (`good` / Low Risk) or default (`bad` / High Risk) based on financial standing, credit history, and loan structure. Manual underwriting is time-consuming and inconsistent, while uncalibrated automated rules can either expose lenders to high default losses or unfairly deny creditworthy applicants.
- **Intended Users:** Credit analysts, risk-modeling students, and underwriting teams seeking a transparent decision-support triage tool to prioritize manual review.
- **Expected Benefit:** Consistent risk scoring, explicit probability estimates for threshold tuning, and interpretable risk drivers (e.g., checking account liquidity, loan duration, credit history) to support faster, more consistent human-in-the-loop review.

---

## 2. Dataset Documentation

- **Source:** UCI Machine Learning Repository — *Statlog (German Credit Data)* (loaded via `scikit-learn` `fetch_openml('credit-g', version=1)` and cached locally at `data/german_credit.csv`).
- **Records & Features:** **1,000 records**, **20 raw attributes** (7 numeric, 13 categorical), and **1 binary target variable** (`class`). After removing 5 redundant, obsolete, or protected attributes (`own_telephone`, `num_dependents`, `residence_since`, `personal_status`, `foreign_worker`), **15 core financial, employment, collateral, and repayment features** (5 numeric, 10 categorical) are used for modeling and in the prototype UI.
- **Target Variable & Class Imbalance:**
  - `good` (`0`, Low Risk): **700 records (70.0%)**
  - `bad` (`1`, High Risk / Positive Class): **300 records (30.0%)**
  - Framing `bad` as the positive class (`1`) ensures Precision, Recall, F1, ROC-AUC, and PR-AUC directly measure the model's ability to detect high-risk applications.
- **Selected Model Features (15):** `checking_status`, `duration` (months), `credit_history`, `purpose`, `credit_amount` (DM), `savings_status`, `employment`, `installment_commitment`, `other_parties`, `property_magnitude`, `age`, `other_payment_plans`, `housing`, `existing_credits`, `job`.
- **Excluded Attributes (5):** `own_telephone` (obsolete 1994 landline indicator), `num_dependents` (near-constant: 85% equal `1`), `residence_since` (weak non-financial proxy), `personal_status` (gender/marital status), and `foreign_worker` (protected attribute with 96.3% single-class imbalance).
- **Data Limitations:**
  1. **Historical & Currency Context:** Collected in 1994 in Germany with loan amounts in Deutsche Marks (DM); monetary thresholds do not reflect modern inflation or other geographies.
  2. **Small Sample Size:** 1,000 rows limits statistical power, especially for rare subgroups (e.g., `foreign_worker = no` has only 8 test samples).
  3. **No Macroeconomic / Post-Loan Variables:** Lacks debt-to-income ratio, employment sector stability, or macroeconomic shock indicators.
  4. **Sensitive Demographic Attributes:** Contains `age`, `personal_status` (encoding gender & marital status), and `foreign_worker`, which pose fairness and regulatory risks if used directly in real-world lending.

---

## 3. Algorithm Implementation & Comparison

All four recommended algorithms are implemented inside leak-free `scikit-learn` `Pipeline`s (`SimpleImputer` + `StandardScaler` for numeric features; `SimpleImputer` + `OneHotEncoder` for categorical features, fit **only** on the training split):

1. **Logistic Regression** (`class_weight='balanced'`, tuned `C=0.05`) — linear, highly interpretable log-odds baseline.
2. **Support Vector Machine (SVM)** (`SVC(probability=True, class_weight='balanced')`, tuned `C=0.5, kernel='rbf'`) — non-linear margin-based classifier.
3. **Decision Tree** (`class_weight='balanced'`, tuned `max_depth=3, min_samples_leaf=10`) — rule-based, human-readable splits.
4. **K-Nearest Neighbors (KNN)** (tuned `n_neighbors=21, weights='distance'`) — instance-based distance classifier.

---

## 4. Evaluation Strategy & Results

### Train / Validation / Test Split
- **Stratified 60% / 20% / 20% split** (`random_state=42`):
  - **Train:** 600 records (420 good / 180 bad)
  - **Validation:** 200 records (140 good / 60 bad) — used exclusively for hyperparameter selection via **Validation ROC-AUC**.
  - **Test:** 200 records (140 good / 60 bad) — held out for final reporting.

### Held-Out Test Set Performance (Imbalanced Classification Metrics)

Because the dataset is imbalanced (70% good / 30% bad), **accuracy alone is misleading**. Notice that KNN achieves the highest accuracy (`0.745`), yet KNN's Recall is only `0.283` (catching less than a third of defaults), whereas SVM and Logistic Regression achieve higher F1 (`0.581` / `0.580`) and Recall (`0.717` / `0.667`).

| Model | Best Val Params | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `C=0.05` | 0.710 | 0.513 | 0.667 | 0.580 | 0.784 | **0.639** |
| **SVM (RBF)** *(Best Val ROC-AUC)* | `C=0.5, kernel='rbf'` | 0.690 | 0.489 | 0.717 | **0.581** | 0.774 | 0.557 |
| **Decision Tree** | `max_depth=3, min_samples_leaf=10` | 0.605 | 0.423 | **0.867** | 0.568 | 0.693 | 0.430 |
| **KNN** | `n_neighbors=21, weights='distance'` | **0.745** | **0.680** | 0.283 | 0.400 | **0.792** | 0.600 |

Generated evaluation plots are saved in `artifacts/plots/`:
- `model_comparison.png`
- `roc_pr_curves.png`
- `confusion_matrices.png`
- `feature_importance.png`

### Error Analysis (Test Set — SVM)
- **False Negatives (17 Missed High-Risk Cases):** Averaged smaller loans (`2,809 DM`) and shorter durations (`15.5 months`), often with `no checking` account status and `existing paid` credit history—making them look superficially similar to low-risk borrowers.
- **False Positives (48 Good Applicants Flagged as High-Risk):** Averaged larger loans (`3,942 DM`), longer durations (`22.7 months`), younger age (`31.8` vs `38.6` for True Negatives), and `<0` checking status. In lending, False Negatives carry direct default loss, while False Positives unfairly deny credit to creditworthy applicants—highlighting why human review of borderline cases is essential.

---

## 5. Ethics, Bias & Responsible Use

- **Privacy & Data Minimization:** Credit applications contain sensitive financial and demographic data. Production systems must minimize PII, encrypt data at rest/in transit, and comply with applicable data protection laws.
- **Algorithmic Bias & Protected Attributes:** The dataset includes `age`, `personal_status` (gender/marital status), and `foreign_worker`. Our empirical **Subgroup Fairness Audit** on the held-out test set (SVM) shows substantial disparities in False Positive Rates (FPR):
  - **Age `19-29`:** `FPR = 0.549` (54.9% of good young applicants flagged as high-risk) vs `0.222` for ages `30-50` and `0.235` for `51+`.
  - **Personal Status:** `female div/dep/mar` has `FPR = 0.409` vs `0.275` for `male single`.
  - **Proxy Risk:** Even if protected columns are removed, correlated variables (`residence_since`, `employment`, `housing`) can act as proxies.
- **Potential Harm & Responsible Use:**
  - Automated rejection without explanation harms financial inclusion and can violate fair-lending laws (e.g., ECOA / EU AI Act high-risk AI obligations).
  - **Mitigation:** Use predictions strictly as **decision-support**, pair with human-in-the-loop review, provide adverse-action explanations (e.g., via Logistic Regression coefficients), and regularly audit subgroup error rates.

---

## 6. Reproducibility & Run Instructions

### Project Structure
```text
credit-risk-classification/
├── app.py                              # Interactive Streamlit prototype
├── data/german_credit.csv              # Cached UCI Statlog German Credit dataset
├── notebooks/credit_risk_classification.ipynb  # Reproducible Jupyter notebook
├── src/
│   ├── data_loader.py                  # Dataset loading & target encoding
│   ├── preprocessing.py                # Leak-free ColumnTransformer & 60/20/20 split
│   └── train_evaluate.py               # Training, tuning, evaluation, plots, error & fairness audit
├── artifacts/                          # Saved models, metrics_summary.json, and plots
├── tests/test_pipeline.py              # Automated verification tests
└── requirements.txt                    # Package dependencies
```

### Setup & Run Commands
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train & evaluate all 4 models (generates artifacts/metrics_summary.json & plots)
python -m src.train_evaluate

# 3. Run automated tests
pytest -v

# 4. Launch the interactive Streamlit prototype
streamlit run app.py
```

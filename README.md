# Credit Risk Classification

## 1. Problem Statement

- **What is the problem?** Banks and lenders need to decide whether a loan applicant is likely to pay back a loan on time (**Low Risk / Good**) or fail to pay (**High Risk / Bad**). Checking every application by hand takes a lot of time and can be inconsistent.
- **Who is it for?** Loan officers, credit analysts, and students learning how machine learning can support financial decisions.
- **What is the benefit?** A simple risk score helps reviewers spot high-risk applications faster, apply consistent criteria, and focus their manual review on borderline cases.

---

## 2. Dataset

- **Source:** UCI Machine Learning Repository — *Statlog (German Credit Data)*, saved locally at `data/german_credit.csv`.
- **Size:** **1,000 loan records** with **20 original columns** and **1 target column** (`class`).
- **Target Variable (`class`):**
  - `good` (`0`, Low Risk): **700 records (70%)**
  - `bad` (`1`, High Risk): **300 records (30%)**
- **Feature Selection (15 Core Attributes):**
  To keep the model focused on real financial and repayment factors, we removed **5 unnecessary or sensitive columns**:
  - `own_telephone` (outdated 1994 landline phone question)
  - `num_dependents` (almost always `1`, adds little information)
  - `residence_since` (weak non-financial detail)
  - `personal_status` (gender and marital status)
  - `foreign_worker` (sensitive personal status, and 96.3% of rows are `yes`)
  - **Remaining 15 attributes:** `checking_status`, `duration`, `credit_history`, `purpose`, `credit_amount`, `savings_status`, `employment`, `installment_commitment`, `other_parties`, `property_magnitude`, `age`, `other_payment_plans`, `housing`, `existing_credits`, and `job`.
- **Data Limitations:**
  1. **Old & Local Data:** Collected in 1994 in Germany using Deutsche Marks (DM), so the money amounts do not match today's prices or other countries.
  2. **Small Size:** 1,000 rows is small for machine learning, especially when checking smaller groups.
  3. **Missing Context:** The data does not include overall economic conditions (like inflation or unemployment rates).

---

## 3. Algorithms Compared

We trained and compared four standard classification algorithms using `scikit-learn`:

1. **Logistic Regression:** A simple, transparent linear model that shows which features increase or decrease risk.
2. **Support Vector Machine (SVM):** Finds a curved boundary to separate low-risk and high-risk applicants.
3. **Decision Tree:** Learns easy-to-read if/then rules (limited to depth 3 so it does not overfit).
4. **K-Nearest Neighbors (KNN):** Predicts risk by looking at the most similar past applicants.

All missing values are filled in and numbers are scaled using **only** the training data so there is no data leakage.

---

## 4. Evaluation & Error Analysis

### Data Split
We split the 1,000 records into three parts while keeping the 70% / 30% class balance in each:
- **Training set:** 600 records (used to train the models)
- **Validation set:** 200 records (used to pick the best settings for each model)
- **Test set:** 200 records (kept completely separate for final grading)

### Why Accuracy Alone Is Not Enough
Because 70% of applicants are `good`, a model can get ~70% accuracy just by guessing `good` most of the time. For example, **KNN** gets the highest accuracy (**0.745**), but it only catches **28.3%** of actual defaults (`Recall = 0.283`). That is why we also check **Precision**, **Recall**, **F1-Score**, **ROC-AUC**, and **PR-AUC**.

### Held-Out Test Set Results

| Model | Best Settings | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `C=0.05` | 0.710 | 0.513 | 0.667 | 0.580 | 0.784 | **0.639** |
| **SVM (RBF)** *(Best Validation ROC-AUC)* | `C=0.5, kernel='rbf'` | 0.690 | 0.489 | 0.717 | **0.581** | 0.774 | 0.557 |
| **Decision Tree** | `max_depth=3, min_samples_leaf=10` | 0.605 | 0.423 | **0.867** | 0.568 | 0.693 | 0.430 |
| **KNN** | `n_neighbors=21, weights='distance'` | **0.745** | **0.680** | 0.283 | 0.400 | **0.792** | 0.600 |

Charts are saved in `artifacts/plots/`:
- `model_comparison.png`
- `roc_pr_curves.png`
- `confusion_matrices.png`
- `feature_importance.png`

### Error Analysis (What Mistakes Does the Model Make?)
Looking at the **SVM** results on the 200 test cases:
- **False Negatives (Missed High-Risk Applicants):** These were often smaller loans with shorter durations and `no checking` account on file, so they looked similar to safe borrowers.
- **False Positives (Good Applicants Flagged as High-Risk):** These were often larger loans, longer durations, or applicants with `<0` (overdrawn) checking accounts. In real life, a False Negative costs the bank money, while a False Positive unfairly turns away a good customer.

---

## 5. Ethics & Responsible Use

- **Privacy:** Loan applications contain private financial details. Real systems must protect personal data and collect only what is needed.
- **Fairness & Bias:** Even after removing `personal_status` and `foreign_worker` from the model, we checked how the model performed across different groups on the test set. Younger applicants (`19–29`) and certain demographic groups still had higher false-alarm rates than older applicants (`30–50`).
- **Responsible Use:** Because of these biases and the small 1994 dataset, this model should **only** be used as a helper tool alongside a human reviewer, never as an automatic rejection system.

---

## 6. How to Run

### Project Files
```text
credit-risk-classification/
├── app.py                                      # Interactive Streamlit web app
├── data/german_credit.csv                      # Dataset
├── notebooks/credit_risk_classification.ipynb  # Jupyter notebook version
├── src/
│   ├── data_loader.py                          # Loads data and selects 15 features
│   ├── preprocessing.py                        # Splits data and scales/encodes features
│   └── train_evaluate.py                       # Trains models, saves plots & metrics
├── artifacts/                                  # Saved models, metrics, and plots
├── tests/test_pipeline.py                      # Automated tests
└── requirements.txt                            # Python packages needed
```

### Commands
```bash
# 1. Install required packages
pip install -r requirements.txt

# 2. Train and evaluate all 4 models
python -m src.train_evaluate

# 3. Run automated tests
python -m pytest -v

# 4. Start the interactive web app
streamlit run app.py
```

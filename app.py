"""Interactive Streamlit prototype for Credit Risk Classification (Decision-Support Demo)."""

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import streamlit as st

from src.data_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_german_credit

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

PRESET_PROFILES: Dict[str, Dict[str, Any]] = {
    "Low-Risk Profile (Established Saver)": {
        "checking_status": ">=200",
        "duration": 4,
        "credit_history": "critical/other existing credit",
        "purpose": "radio/tv",
        "credit_amount": 1500,
        "savings_status": ">=1000",
        "employment": ">=7",
        "installment_commitment": 2,
        "other_parties": "none",
        "property_magnitude": "real estate",
        "age": 42,
        "other_payment_plans": "none",
        "housing": "own",
        "existing_credits": 1,
        "job": "skilled",
    },
    "Borderline Profile (Moderate Loan)": {
        "checking_status": "0<=X<200",
        "duration": 24,
        "credit_history": "existing paid",
        "purpose": "used car",
        "credit_amount": 4200,
        "savings_status": "100<=X<500",
        "employment": "1<=X<4",
        "installment_commitment": 3,
        "other_parties": "none",
        "property_magnitude": "car",
        "age": 31,
        "other_payment_plans": "none",
        "housing": "rent",
        "existing_credits": 1,
        "job": "skilled",
    },
    "High-Risk Profile (Large Long-Term Loan, Low Liquidity)": {
        "checking_status": "<0",
        "duration": 48,
        "credit_history": "no credits/all paid",
        "purpose": "new car",
        "credit_amount": 12000,
        "savings_status": "<100",
        "employment": "<1",
        "installment_commitment": 4,
        "other_parties": "none",
        "property_magnitude": "no known property",
        "age": 24,
        "other_payment_plans": "bank",
        "housing": "rent",
        "existing_credits": 2,
        "job": "unskilled resident",
    },
}


@st.cache_resource
def load_artifacts():
    """Load trained models and evaluation summary."""
    models = joblib.load(ARTIFACTS_DIR / "all_models.joblib")
    with open(ARTIFACTS_DIR / "metrics_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    df = load_german_credit()
    return models, summary, df


def main() -> None:
    st.set_page_config(page_title="Credit Risk Classification Prototype", layout="wide")
    st.title("Credit Risk Classification — Decision-Support Prototype")
    

    if not (ARTIFACTS_DIR / "all_models.joblib").exists():
        st.error("Trained artifacts not found. Run `python -m src.train_evaluate` first.")
        return

    models, summary, df = load_artifacts()
    tab_predict, tab_eval, tab_ethics = st.tabs(
        ["1. Applicant Risk Predictor", "2. Model Comparison & Error Analysis", "3. Dataset & Ethics"]
    )

    with tab_predict:
        st.subheader("Simulate Loan Applicant Profile")
        col_top1, col_top2, col_top3 = st.columns([1.2, 1.2, 1.0])
        with col_top1:
            preset_name = st.selectbox("Load Preset Applicant Profile", list(PRESET_PROFILES.keys()))
        with col_top2:
            model_names = list(models.keys())
            default_idx = model_names.index("Logistic Regression") if "Logistic Regression" in model_names else 0
            selected_model_name = st.selectbox("Select Classification Model", model_names, index=default_idx)
        with col_top3:
            threshold = st.slider("High-Risk Decision Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

        preset = PRESET_PROFILES[preset_name]
        st.markdown("#### Applicant Attributes (15 Core Financial & Repayment Features)")
        user_input: Dict[str, Any] = {}

        c1, c2, c3 = st.columns(3)
        with c1:
            user_input["checking_status"] = st.selectbox(
                "Checking Status", sorted(df["checking_status"].unique()),
                index=sorted(df["checking_status"].unique()).index(preset["checking_status"]),
            )
            user_input["credit_history"] = st.selectbox(
                "Credit History", sorted(df["credit_history"].unique()),
                index=sorted(df["credit_history"].unique()).index(preset["credit_history"]),
            )
            user_input["purpose"] = st.selectbox(
                "Loan Purpose", sorted(df["purpose"].unique()),
                index=sorted(df["purpose"].unique()).index(preset["purpose"]),
            )
            user_input["savings_status"] = st.selectbox(
                "Savings Status", sorted(df["savings_status"].unique()),
                index=sorted(df["savings_status"].unique()).index(preset["savings_status"]),
            )
            user_input["employment"] = st.selectbox(
                "Employment Duration", sorted(df["employment"].unique()),
                index=sorted(df["employment"].unique()).index(preset["employment"]),
            )

        with c2:
            user_input["duration"] = st.number_input("Loan Duration (months)", min_value=4, max_value=72, value=int(preset["duration"]))
            user_input["credit_amount"] = st.number_input("Credit Amount (DM)", min_value=250, max_value=25000, value=int(preset["credit_amount"]), step=100)
            user_input["installment_commitment"] = st.slider("Installment Rate (% of income tier)", 1, 4, int(preset["installment_commitment"]))
            user_input["age"] = st.number_input("Age (years)", min_value=19, max_value=80, value=int(preset["age"]))
            user_input["existing_credits"] = st.slider("Existing Credits at Bank", 1, 4, int(preset["existing_credits"]))

        with c3:
            user_input["housing"] = st.selectbox(
                "Housing", sorted(df["housing"].unique()),
                index=sorted(df["housing"].unique()).index(preset["housing"]),
            )
            user_input["property_magnitude"] = st.selectbox(
                "Property Magnitude", sorted(df["property_magnitude"].unique()),
                index=sorted(df["property_magnitude"].unique()).index(preset["property_magnitude"]),
            )
            user_input["other_payment_plans"] = st.selectbox(
                "Other Payment Plans", sorted(df["other_payment_plans"].unique()),
                index=sorted(df["other_payment_plans"].unique()).index(preset["other_payment_plans"]),
            )
            user_input["other_parties"] = st.selectbox(
                "Other Parties / Guarantors", sorted(df["other_parties"].unique()),
                index=sorted(df["other_parties"].unique()).index(preset["other_parties"]),
            )
            user_input["job"] = st.selectbox(
                "Job Category", sorted(df["job"].unique()),
                index=sorted(df["job"].unique()).index(preset["job"]),
            )

        input_df = pd.DataFrame([user_input])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        model = models[selected_model_name]
        prob_bad = float(model.predict_proba(input_df)[0, 1])
        is_high_risk = prob_bad >= threshold

        st.divider()
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Predicted Risk Category", "HIGH RISK (Bad)" if is_high_risk else "LOW RISK (Good)")
        res_col2.metric("Estimated Default Probability", f"{prob_bad:.1%}")
        res_col3.metric("Decision Threshold", f"{threshold:.0%}")
        st.progress(min(max(prob_bad, 0.0), 1.0))

    with tab_eval:
        st.subheader("Held-Out Test Set Metrics (n = 200; 70% Good / 30% Bad)")
        rows = []
        for name, m_data in summary["models"].items():
            t = m_data["test"]
            rows.append(
                {
                    "Model": name,
                    "Best Params": str(m_data["best_params"]),
                    "Accuracy": t["accuracy"],
                    "Precision": t["precision"],
                    "Recall": t["recall"],
                    "F1-Score": t["f1"],
                    "ROC-AUC": t["roc_auc"],
                    "PR-AUC": t["pr_auc"],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        p1, p2 = st.columns(2)
        with p1:
            st.image(str(PLOTS_DIR / "model_comparison.png"), caption="Test Set Metric Comparison")
            st.image(str(PLOTS_DIR / "confusion_matrices.png"), caption="Test Set Confusion Matrices")
        with p2:
            st.image(str(PLOTS_DIR / "roc_pr_curves.png"), caption="ROC & Precision-Recall Curves")
            st.image(str(PLOTS_DIR / "feature_importance.png"), caption="Top Logistic Regression Risk Drivers")

        st.subheader(f"Error Analysis ({summary['best_model_name']} on Test Set)")
        err_rows = []
        for outcome, stats in summary["error_analysis"]["summary_by_outcome"].items():
            err_rows.append({"Outcome": outcome, **stats})
        st.dataframe(pd.DataFrame(err_rows), use_container_width=True, hide_index=True)

    with tab_ethics:
        st.subheader("Dataset Summary & Feature Selection")
        st.write(
            "- **Source:** UCI Statlog (German Credit Data), 1,000 records, 20 raw attributes, binary target (`good`=700, `bad`=300).\n"
            "- **Removed Redundant / Protected Attributes (5):** `own_telephone` (obsolete 1994 landline), `num_dependents` (near-constant), "
            "`residence_since` (weak proxy), `personal_status` (gender/marital status), and `foreign_worker` (protected & 96.3% single-class).\n"
            "- **Limitations:** Historical 1994 German dataset (amounts in Deutsche Marks), small sample size (1,000 rows), "
            "and no macroeconomic indicators."
        )
        st.subheader("Subgroup Fairness Audit (Test Set)")
        for attr, records in summary["fairness_audit"].items():
            st.markdown(f"**Group: `{attr}`**")
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

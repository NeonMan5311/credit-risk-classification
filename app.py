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
        "duration": 12,
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
    st.set_page_config(
        page_title="Credit Risk Classifier",
        page_icon="💳",
        layout="wide",
    )

    st.title("Credit Risk Classification Prototype")
    st.caption("A decision-support demonstration using the UCI Statlog (German Credit) dataset.")
    st.info(
        "**Note:** This tool is for educational and decision-support demonstration only. "
        "It is not meant to replace human review or real-world bank underwriting."
    )

    if not (ARTIFACTS_DIR / "all_models.joblib").exists():
        st.error("Trained models not found. Please run `python -m src.train_evaluate` first.")
        return

    models, summary, df = load_artifacts()

    # Sidebar controls
    with st.sidebar:
        st.header("Prediction Settings")
        preset_name = st.selectbox("Sample Applicant Profile", list(PRESET_PROFILES.keys()))
        model_names = list(models.keys())
        default_idx = model_names.index("Logistic Regression") if "Logistic Regression" in model_names else 0
        selected_model_name = st.selectbox("Classification Model", model_names, index=default_idx)
        threshold = st.slider(
            "Risk Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Applicants with a predicted default probability at or above this threshold are flagged as High Risk.",
        )
        st.divider()
        st.markdown(
            "**Quick Guide**\n"
            "- **Low Risk (Good):** Likely to repay.\n"
            "- **High Risk (Bad):** Higher chance of default; flag for manual review."
        )

    tab_predict, tab_eval, tab_ethics = st.tabs(
        ["Applicant Risk Predictor", "Model Comparison & Errors", "Dataset & Ethics"]
    )

    with tab_predict:
        preset = PRESET_PROFILES[preset_name]
        user_input: Dict[str, Any] = {}

        st.subheader("Applicant Details (15 Core Attributes)")
        with st.expander("Financial & Loan Details", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                user_input["checking_status"] = st.selectbox(
                    "Checking Account Balance (DM)",
                    sorted(df["checking_status"].unique()),
                    index=sorted(df["checking_status"].unique()).index(preset["checking_status"]),
                    help="<0 = overdrawn, 0<=X<200 = low, >=200 = healthy, no checking = none on file",
                )
                user_input["savings_status"] = st.selectbox(
                    "Savings Account Balance (DM)",
                    sorted(df["savings_status"].unique()),
                    index=sorted(df["savings_status"].unique()).index(preset["savings_status"]),
                )
            with c2:
                user_input["credit_amount"] = st.number_input(
                    "Loan Amount (DM)", min_value=250, max_value=25000, value=int(preset["credit_amount"]), step=100
                )
                user_input["duration"] = st.number_input(
                    "Loan Duration (Months)", min_value=4, max_value=72, value=int(preset["duration"])
                )
            with c3:
                user_input["purpose"] = st.selectbox(
                    "Loan Purpose",
                    sorted(df["purpose"].unique()),
                    index=sorted(df["purpose"].unique()).index(preset["purpose"]),
                )
                user_input["installment_commitment"] = st.slider(
                    "Installment Rate (% of Income Tier)", 1, 4, int(preset["installment_commitment"])
                )

        with st.expander("Credit History & Existing Obligations", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                user_input["credit_history"] = st.selectbox(
                    "Credit History",
                    sorted(df["credit_history"].unique()),
                    index=sorted(df["credit_history"].unique()).index(preset["credit_history"]),
                )
            with c2:
                user_input["existing_credits"] = st.slider(
                    "Existing Credits at Bank", 1, 4, int(preset["existing_credits"])
                )
            with c3:
                user_input["other_payment_plans"] = st.selectbox(
                    "Other Payment Plans",
                    sorted(df["other_payment_plans"].unique()),
                    index=sorted(df["other_payment_plans"].unique()).index(preset["other_payment_plans"]),
                )

        with st.expander("Employment, Housing & Collateral", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                user_input["employment"] = st.selectbox(
                    "Employment Length (Years)",
                    sorted(df["employment"].unique()),
                    index=sorted(df["employment"].unique()).index(preset["employment"]),
                )
                user_input["job"] = st.selectbox(
                    "Job Type",
                    sorted(df["job"].unique()),
                    index=sorted(df["job"].unique()).index(preset["job"]),
                )
            with c2:
                user_input["housing"] = st.selectbox(
                    "Housing Status",
                    sorted(df["housing"].unique()),
                    index=sorted(df["housing"].unique()).index(preset["housing"]),
                )
                user_input["property_magnitude"] = st.selectbox(
                    "Main Asset / Collateral",
                    sorted(df["property_magnitude"].unique()),
                    index=sorted(df["property_magnitude"].unique()).index(preset["property_magnitude"]),
                )
            with c3:
                user_input["other_parties"] = st.selectbox(
                    "Co-Applicant / Guarantor",
                    sorted(df["other_parties"].unique()),
                    index=sorted(df["other_parties"].unique()).index(preset["other_parties"]),
                )
                user_input["age"] = st.number_input(
                    "Applicant Age (Years)", min_value=19, max_value=80, value=int(preset["age"])
                )

        input_df = pd.DataFrame([user_input])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        model = models[selected_model_name]
        prob_bad = float(model.predict_proba(input_df)[0, 1])
        is_high_risk = prob_bad >= threshold

        st.divider()
        st.subheader("Prediction Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Risk Category", "HIGH RISK (Bad)" if is_high_risk else "LOW RISK (Good)")
        r2.metric("Estimated Default Probability", f"{prob_bad:.1%}")
        r3.metric("Model Used", selected_model_name)

        st.progress(min(max(prob_bad, 0.0), 1.0))
        if is_high_risk:
            st.error(
                f"Flagged as **High Risk** ({prob_bad:.1%} >= {threshold:.0%} threshold). "
                "Recommended action: route to a human underwriter for manual review."
            )
        else:
            st.success(
                f"Classified as **Low Risk** ({prob_bad:.1%} < {threshold:.0%} threshold). "
                "Applicant profile aligns with historical on-time repayments."
            )

    with tab_eval:
        st.subheader("Test Set Performance (200 Held-Out Applicants)")
        rows = []
        for name, m_data in summary["models"].items():
            t = m_data["test"]
            rows.append(
                {
                    "Model": name,
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
            st.image(str(PLOTS_DIR / "model_comparison.png"), caption="Model Metric Comparison")
            st.image(str(PLOTS_DIR / "confusion_matrices.png"), caption="Confusion Matrices")
        with p2:
            st.image(str(PLOTS_DIR / "roc_pr_curves.png"), caption="ROC & Precision-Recall Curves")
            st.image(str(PLOTS_DIR / "feature_importance.png"), caption="Top Risk Drivers (Logistic Regression)")

        st.subheader(f"Error Analysis ({summary['best_model_name']} on Test Set)")
        err_rows = []
        for outcome, stats in summary["error_analysis"]["summary_by_outcome"].items():
            err_rows.append({"Outcome": outcome, **stats})
        st.dataframe(pd.DataFrame(err_rows), use_container_width=True, hide_index=True)

    with tab_ethics:
        st.subheader("Dataset Summary & Removed Columns")
        st.write(
            "- **Source:** UCI Statlog (German Credit Data) — 1,000 records, 700 `good` (70%) and 300 `bad` (30%).\n"
            "- **Removed Attributes (5):** `own_telephone` (outdated 1994 landline), `num_dependents` (almost always 1), "
            "`residence_since` (weak proxy), `personal_status` (gender/marital status), and `foreign_worker` (protected & 96.3% one value).\n"
            "- **Limitations:** Data is from 1994 Germany (in Deutsche Marks), has a small sample size (1,000 rows), "
            "and does not include wider economic conditions."
        )
        st.subheader("Fairness Check Across Groups (Test Set)")
        for attr, records in summary["fairness_audit"].items():
            st.markdown(f"**Group: `{attr}`**")
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

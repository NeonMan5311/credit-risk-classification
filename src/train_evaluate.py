"""Train, tune, evaluate, and analyze Logistic Regression, SVM, Decision Tree, and KNN."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import PROJECT_ROOT, load_german_credit, split_features_target
from src.preprocessing import DatasetSplits, build_preprocessor, make_train_val_test_split

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"


def get_candidate_models() -> Dict[str, Tuple[Any, List[Dict[str, Any]]]]:
    """Return the four candidate classifiers and their validation hyperparameter grids."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            [{"C": 0.05}, {"C": 0.1}, {"C": 1.0}, {"C": 5.0}],
        ),
        "SVM": (
            SVC(probability=True, class_weight="balanced", random_state=42),
            [
                {"C": 0.5, "kernel": "rbf"},
                {"C": 1.0, "kernel": "rbf"},
                {"C": 2.0, "kernel": "rbf"},
                {"C": 1.0, "kernel": "linear"},
            ],
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=42),
            [
                {"max_depth": 3, "min_samples_leaf": 10},
                {"max_depth": 4, "min_samples_leaf": 8},
                {"max_depth": 5, "min_samples_leaf": 5},
                {"max_depth": 6, "min_samples_leaf": 5},
            ],
        ),
        "KNN": (
            KNeighborsClassifier(),
            [
                {"n_neighbors": 5, "weights": "uniform"},
                {"n_neighbors": 9, "weights": "distance"},
                {"n_neighbors": 15, "weights": "distance"},
                {"n_neighbors": 21, "weights": "distance"},
            ],
        ),
    }


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Compute imbalanced-classification metrics and confusion matrix counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def tune_and_train_models(splits: DatasetSplits) -> Tuple[Dict[str, Pipeline], Dict[str, Any]]:
    """Tune each algorithm on the validation split and evaluate on validation & test."""
    candidates = get_candidate_models()
    trained_models: Dict[str, Pipeline] = {}
    results: Dict[str, Any] = {}

    for name, (estimator, param_grid) in candidates.items():
        best_val_auc = -1.0
        best_pipeline = None
        best_params = None
        best_val_metrics = None

        for params in param_grid:
            clf = clone(estimator).set_params(**params)
            pipe = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    ("classifier", clf),
                ]
            )
            pipe.fit(splits.X_train, splits.y_train)
            val_prob = pipe.predict_proba(splits.X_val)[:, 1]
            val_pred = pipe.predict(splits.X_val)
            val_metrics = compute_metrics(splits.y_val, val_pred, val_prob)

            if val_metrics["roc_auc"] > best_val_auc:
                best_val_auc = val_metrics["roc_auc"]
                best_pipeline = pipe
                best_params = params
                best_val_metrics = val_metrics

        test_prob = best_pipeline.predict_proba(splits.X_test)[:, 1]
        test_pred = best_pipeline.predict(splits.X_test)
        test_metrics = compute_metrics(splits.y_test, test_pred, test_prob)

        trained_models[name] = best_pipeline
        results[name] = {
            "best_params": best_params,
            "validation": best_val_metrics,
            "test": test_metrics,
        }

    return trained_models, results


def run_error_analysis(
    best_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Any]:
    """Analyze False Positives and False Negatives on the held-out test set."""
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    analysis_df = X_test.copy()
    analysis_df["y_true"] = y_test.values
    analysis_df["y_pred"] = y_pred
    analysis_df["y_prob"] = np.round(y_prob, 4)

    def outcome_label(row: pd.Series) -> str:
        if row["y_true"] == 1 and row["y_pred"] == 1:
            return "TP (Caught High Risk)"
        if row["y_true"] == 0 and row["y_pred"] == 0:
            return "TN (Cleared Low Risk)"
        if row["y_true"] == 0 and row["y_pred"] == 1:
            return "FP (Good Flagged as High Risk)"
        return "FN (Missed High Risk)"

    analysis_df["outcome"] = analysis_df.apply(outcome_label, axis=1)

    summary_by_outcome = {}
    for outcome, group in analysis_df.groupby("outcome"):
        summary_by_outcome[outcome] = {
            "count": int(len(group)),
            "avg_credit_amount": round(float(group["credit_amount"].mean()), 2),
            "avg_duration_months": round(float(group["duration"].mean()), 2),
            "avg_age": round(float(group["age"].mean()), 2),
            "top_checking_status": str(group["checking_status"].mode().iloc[0]),
            "top_credit_history": str(group["credit_history"].mode().iloc[0]),
        }

    fn_examples = (
        analysis_df[analysis_df["outcome"] == "FN (Missed High Risk)"]
        .sort_values("y_prob", ascending=True)
        .head(5)[["checking_status", "duration", "credit_amount", "credit_history", "age", "y_prob"]]
        .to_dict(orient="records")
    )
    fp_examples = (
        analysis_df[analysis_df["outcome"] == "FP (Good Flagged as High Risk)"]
        .sort_values("y_prob", ascending=False)
        .head(5)[["checking_status", "duration", "credit_amount", "credit_history", "age", "y_prob"]]
        .to_dict(orient="records")
    )

    return {
        "summary_by_outcome": summary_by_outcome,
        "top_false_negatives": fn_examples,
        "top_false_positives": fp_examples,
    }


def run_fairness_audit(
    best_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    raw_test_df: pd.DataFrame,
) -> Dict[str, List[Dict[str, Any]]]:
    """Compute subgroup predictions and error rates on held-out test data for ethics auditing."""
    df = raw_test_df.copy().reset_index(drop=True)
    df["y_true"] = y_test.values
    df["y_pred"] = best_model.predict(X_test)
    df["age_group"] = pd.cut(df["age"], bins=[18, 29, 50, 100], labels=["19-29", "30-50", "51+"])

    audit: Dict[str, List[Dict[str, Any]]] = {}
    for col in ["age_group", "foreign_worker", "personal_status"]:
        rows = []
        for group_val, grp in df.groupby(col, observed=False):
            if len(grp) == 0:
                continue
            neg_grp = grp[grp["y_true"] == 0]
            pos_grp = grp[grp["y_true"] == 1]
            fpr = float((neg_grp["y_pred"] == 1).mean()) if len(neg_grp) > 0 else 0.0
            tpr = float((pos_grp["y_pred"] == 1).mean()) if len(pos_grp) > 0 else 0.0
            rows.append(
                {
                    "group": str(group_val),
                    "count": int(len(grp)),
                    "actual_high_risk_rate": round(float(grp["y_true"].mean()), 4),
                    "predicted_high_risk_rate": round(float(grp["y_pred"].mean()), 4),
                    "recall_tpr": round(tpr, 4),
                    "false_positive_rate": round(fpr, 4),
                }
            )
        audit[col] = rows
    return audit


def generate_plots(
    trained_models: Dict[str, Pipeline],
    results: Dict[str, Any],
    splits: DatasetSplits,
    best_model_name: str,
) -> None:
    """Generate evaluation and interpretability plots into artifacts/plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Model comparison bar chart (Test metrics)
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    rows = []
    for name, res in results.items():
        for k, lbl in zip(metric_keys, metric_labels):
            rows.append({"Model": name, "Metric": lbl, "Score": res["test"][k]})
    comp_df = pd.DataFrame(rows)

    plt.figure(figsize=(11, 5.5))
    sns.barplot(data=comp_df, x="Metric", y="Score", hue="Model", palette="viridis")
    plt.ylim(0, 1.05)
    plt.title("Test Set Performance Comparison Across Algorithms")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150)
    plt.close()

    # 2. ROC and Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, pipe in trained_models.items():
        y_prob = pipe.predict_proba(splits.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(splits.y_test, y_prob)
        prec, rec, _ = precision_recall_curve(splits.y_test, y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={results[name]['test']['roc_auc']:.3f})")
        axes[1].plot(rec, prec, label=f"{name} (AP={results[name]['test']['pr_auc']:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_title("ROC Curve (Test Set)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right", fontsize=8)

    baseline_ap = float(splits.y_test.mean())
    axes[1].axhline(baseline_ap, color="k", linestyle="--", alpha=0.5, label=f"Baseline ({baseline_ap:.2f})")
    axes[1].set_title("Precision-Recall Curve (Test Set)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_pr_curves.png", dpi=150)
    plt.close()

    # 3. Confusion Matrices (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, pipe) in zip(axes.ravel(), trained_models.items()):
        cm = confusion_matrix(splits.y_test, pipe.predict(splits.X_test), labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred Good (0)", "Pred Bad (1)"],
            yticklabels=["Actual Good (0)", "Actual Bad (1)"],
        )
        ax.set_title(name)
    plt.suptitle("Test Set Confusion Matrices", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close()

    # 4. Feature Importance / Coefficients from Logistic Regression (interpretable linear baseline)
    lr_pipe = trained_models["Logistic Regression"]
    feature_names = lr_pipe.named_steps["preprocessor"].get_feature_names_out()
    coefs = lr_pipe.named_steps["classifier"].coef_[0]
    imp_df = (
        pd.DataFrame({"feature": feature_names, "coef": coefs, "abs_coef": np.abs(coefs)})
        .sort_values("abs_coef", ascending=False)
        .head(15)
        .sort_values("coef", ascending=True)
    )
    plt.figure(figsize=(10, 6))
    colors = ["#d9534f" if c > 0 else "#2e8b57" for c in imp_df["coef"]]
    plt.barh(imp_df["feature"], imp_df["coef"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"Top 15 Risk Drivers (Logistic Regression Coefficients; Best Model: {best_model_name})")
    plt.xlabel("Log-Odds Coefficient (+ increases High-Risk probability, - decreases)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
    plt.close()


def main() -> None:
    """Run the end-to-end training, evaluation, error analysis, and artifact saving pipeline."""
    df = load_german_credit()
    X, y = split_features_target(df)
    splits = make_train_val_test_split(X, y, random_state=42)

    trained_models, results = tune_and_train_models(splits)

    # Select best model by Test ROC-AUC (all tuned on Validation ROC-AUC)
    best_model_name = max(results.keys(), key=lambda m: results[m]["validation"]["roc_auc"])
    best_model = trained_models[best_model_name]

    error_analysis = run_error_analysis(best_model, splits.X_test, splits.y_test)
    raw_splits = make_train_val_test_split(df, y, random_state=42)
    fairness_audit = run_fairness_audit(best_model, splits.X_test, splits.y_test, raw_splits.X_test)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, ARTIFACTS_DIR / "best_model.joblib")
    joblib.dump(trained_models, ARTIFACTS_DIR / "all_models.joblib")

    summary = {
        "dataset": {
            "total_records": int(len(df)),
            "num_features": int(X.shape[1]),
            "positive_class": "bad (High Risk)",
            "negative_class": "good (Low Risk)",
            "class_distribution": {
                "good_0": int((y == 0).sum()),
                "bad_1": int((y == 1).sum()),
            },
            "splits": {
                "train": int(len(splits.X_train)),
                "validation": int(len(splits.X_val)),
                "test": int(len(splits.X_test)),
            },
        },
        "best_model_name": best_model_name,
        "models": results,
        "error_analysis": error_analysis,
        "fairness_audit": fairness_audit,
    }
    with open(ARTIFACTS_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    generate_plots(trained_models, results, splits, best_model_name)

    print(f"Best model (selected on Validation ROC-AUC): {best_model_name}")
    for name, res in results.items():
        t = res["test"]
        print(
            f"  {name:20s} | Acc={t['accuracy']:.3f} | Prec={t['precision']:.3f} "
            f"| Rec={t['recall']:.3f} | F1={t['f1']:.3f} | ROC-AUC={t['roc_auc']:.3f} | PR-AUC={t['pr_auc']:.3f}"
        )


if __name__ == "__main__":
    main()

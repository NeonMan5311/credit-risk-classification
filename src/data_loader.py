"""Dataset loading utilities for the UCI Statlog (German Credit) dataset."""

from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.datasets import fetch_openml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "german_credit.csv"

# Excluded from model features:
# - own_telephone: obsolete 1994 landline indicator
# - num_dependents: near-constant (85% == 1) with negligible signal
# - residence_since: weak non-financial proxy
# - personal_status, foreign_worker: protected demographic attributes / near-zero variance (96.3% yes)
EXCLUDED_FEATURES = [
    "own_telephone",
    "foreign_worker",
    "personal_status",
    "num_dependents",
    "residence_since",
]

NUMERIC_FEATURES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "age",
    "existing_credits",
]

CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
]

TARGET_COLUMN = "class"
POSITIVE_LABEL = "bad"  # 1 = High Risk (bad), 0 = Low Risk (good)


def load_german_credit(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the German Credit dataset from disk, or fetch from OpenML if missing."""
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
        df = dataset.frame.copy()
        df.to_csv(data_path, index=False)
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate selected feature matrix X (15 attributes) and binary target y (1 = bad, 0 = good)."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = (df[TARGET_COLUMN].astype(str).str.strip().str.lower() == POSITIVE_LABEL).astype(int)
    return X, y

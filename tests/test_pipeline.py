"""Automated tests for data loading, preprocessing, training, and inference."""

import json
from pathlib import Path
import joblib

from src.data_loader import load_german_credit, split_features_target
from src.preprocessing import build_preprocessor, make_train_val_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_dataset_shape_and_balance():
    df = load_german_credit()
    assert df.shape == (1000, 21)
    X, y = split_features_target(df)
    assert X.shape == (1000, 15)
    assert int((y == 0).sum()) == 700
    assert int((y == 1).sum()) == 300


def test_stratified_split_proportions():
    df = load_german_credit()
    X, y = split_features_target(df)
    splits = make_train_val_test_split(X, y, random_state=42)
    assert len(splits.X_train) == 600
    assert len(splits.X_val) == 200
    assert len(splits.X_test) == 200
    assert abs(splits.y_train.mean() - 0.30) < 0.01
    assert abs(splits.y_val.mean() - 0.30) < 0.01
    assert abs(splits.y_test.mean() - 0.30) < 0.01


def test_preprocessor_no_nans():
    df = load_german_credit()
    X, y = split_features_target(df)
    splits = make_train_val_test_split(X, y, random_state=42)
    pre = build_preprocessor()
    X_train_t = pre.fit_transform(splits.X_train)
    X_test_t = pre.transform(splits.X_test)
    assert X_train_t.shape[0] == 600
    assert X_test_t.shape[0] == 200
    assert X_train_t.shape[1] == X_test_t.shape[1]


def test_saved_artifacts_and_inference():
    artifacts = PROJECT_ROOT / "artifacts"
    assert (artifacts / "best_model.joblib").exists()
    assert (artifacts / "all_models.joblib").exists()
    assert (artifacts / "metrics_summary.json").exists()

    with open(artifacts / "metrics_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert set(summary["models"].keys()) == {"Logistic Regression", "SVM", "Decision Tree", "KNN"}

    models = joblib.load(artifacts / "all_models.joblib")
    df = load_german_credit()
    X, _ = split_features_target(df)
    sample = X.iloc[[0]]
    for name, pipe in models.items():
        prob = pipe.predict_proba(sample)[0, 1]
        assert 0.0 <= prob <= 1.0, f"Invalid probability from {name}: {prob}"

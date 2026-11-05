"""Shared repeat-purchase feature engineering and evaluation for Olist.

The target is whether a customer with at least one delivered order before the
cutoff makes another delivered purchase during the following 180-day window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "order_count",
    "total_spend",
    "avg_order_value",
    "recency_days",
    "active_days",
    "active_months",
]
CATEGORICAL_FEATURES = ["customer_state"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "repeat_next_180d"


def _most_common_or_unknown(values: pd.Series) -> str:
    modes = values.dropna().mode()
    return str(modes.iloc[0]) if not modes.empty else "UNKNOWN"


def build_repeat_purchase_features(orders: pd.DataFrame, horizon_days: int = 180):
    """Build leakage-aware customer features and a future repeat-purchase label."""
    required = {
        "customer_unique_id",
        "order_status",
        "order_value",
        "customer_state",
    }
    time_col = "purchase_ts" if "purchase_ts" in orders.columns else "order_purchase_timestamp"
    missing = required.difference(orders.columns)
    if missing or time_col not in orders.columns:
        raise ValueError(
            f"Repeat-model input is missing columns: {sorted(missing | ({time_col} if time_col not in orders.columns else set()))}"
        )

    data = orders.copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    data["order_value"] = pd.to_numeric(data["order_value"], errors="coerce").fillna(0.0)
    data = data[
        (data["order_status"].astype(str).str.lower() == "delivered")
        & data[time_col].notna()
        & data["customer_unique_id"].notna()
    ].copy()
    if data.empty:
        raise ValueError("No delivered orders with valid timestamps and customer IDs were found.")

    max_date = data[time_col].max()
    cutoff = max_date - pd.Timedelta(days=horizon_days)
    past = data[data[time_col] < cutoff].copy()
    future_ids = set(
        data.loc[data[time_col] >= cutoff, "customer_unique_id"].dropna().astype(str)
    )
    if past.empty:
        raise ValueError("There are no delivered orders before the prediction cutoff.")

    past["customer_unique_id"] = past["customer_unique_id"].astype(str)
    past["_purchase_month"] = past[time_col].dt.to_period("M").astype(str)
    features = (
        past.groupby("customer_unique_id")
        .agg(
            order_count=(time_col, "count"),
            total_spend=("order_value", "sum"),
            avg_order_value=("order_value", "mean"),
            first_purchase=(time_col, "min"),
            last_purchase=(time_col, "max"),
            active_months=("_purchase_month", "nunique"),
            customer_state=("customer_state", _most_common_or_unknown),
        )
        .reset_index()
    )
    features["recency_days"] = (cutoff - features["last_purchase"]).dt.days.clip(lower=0)
    features["active_days"] = (
        features["last_purchase"] - features["first_purchase"]
    ).dt.days.clip(lower=0)
    features[TARGET] = features["customer_unique_id"].isin(future_ids).astype(int)
    features["customer_state"] = features["customer_state"].fillna("UNKNOWN").astype(str)
    features["total_spend"] = features["total_spend"].fillna(0.0)
    features["avg_order_value"] = features["avg_order_value"].fillna(0.0)
    features.drop(columns=["first_purchase", "last_purchase"], inplace=True)

    return features, cutoff, max_date


def _distribution_table(y: pd.Series):
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    total = int(counts.sum())
    return [
        {
            "class": int(label),
            "meaning": "No delivered purchase in next 180 days" if label == 0 else "Repeat purchase in next 180 days",
            "customers": int(counts.loc[label]),
            "share_pct": round((100.0 * int(counts.loc[label]) / total) if total else 0.0, 3),
        }
        for label in (0, 1)
    ]


def _make_estimator():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    # Do not use class_weight='balanced': it distorts raw probability estimates
    # for this naturally imbalanced target. We tune the decision threshold using
    # a separate validation split instead.
    base = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=1200, class_weight=None, solver="lbfgs")),
        ]
    )
    return base


def _best_f1_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5, 0.0
    denom = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    best_idx = int(np.nanargmax(f1_values))
    return float(thresholds[best_idx]), float(f1_values[best_idx])


def prepare_repeat_purchase_data(orders: pd.DataFrame, horizon_days: int = 180):
    """Create customer features and inspect target balance before any model fitting."""
    features, cutoff, max_date = build_repeat_purchase_features(orders, horizon_days)
    y = features[TARGET].astype(int)
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    positive_count, negative_count = int(counts.loc[1]), int(counts.loc[0])
    positive_rate = float(positive_count / len(y)) if len(y) else 0.0
    target_info = {
        "cutoff": cutoff.date().isoformat(),
        "last_observed_date": max_date.date().isoformat(),
        "horizon_days": int(horizon_days),
        "customers": int(len(features)),
        "positive_repeat_customers": positive_count,
        "negative_customers": negative_count,
        "positive_rate_pct": round(positive_rate * 100, 3),
        "class_distribution": _distribution_table(y),
        "threshold_method": "F1-maximizing threshold selected on validation data",
        "class_imbalance_handling": (
            "Unweighted logistic regression; tune threshold on validation set; "
            "sigmoid calibration when supported"
        ),
    }
    return features, target_info


def fit_repeat_purchase_model(features: pd.DataFrame, target_info: dict):
    """Fit and evaluate after target distribution has been inspected by the caller."""
    y = features[TARGET].astype(int)
    X = features[MODEL_FEATURES].copy()
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    positive_count, negative_count = int(counts.loc[1]), int(counts.loc[0])
    metrics = dict(target_info)

    # Need enough examples from each class for stratified train/validation/test
    # splits and cross-validated probability calibration.
    if min(positive_count, negative_count) < 15:
        metrics["model_status"] = "not_trained_insufficient_class_support"
        metrics["warning"] = (
            "At least 15 customers in each target class are required for the "
            "stratified train/validation/test and calibration workflow."
        )
        return None, metrics

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    metrics["split_distribution"] = {
        name: _distribution_table(split_y)
        for name, split_y in [
            ("train", y_train),
            ("validation", y_val),
            ("test", y_test),
        ]
    }

    base_estimator = _make_estimator()
    min_train_class = int(y_train.value_counts().min())
    use_calibration = min_train_class >= 3
    if use_calibration:
        estimator = CalibratedClassifierCV(estimator=base_estimator, method="sigmoid", cv=3)
    else:
        estimator = base_estimator
    estimator.fit(X_train, y_train)

    validation_probabilities = estimator.predict_proba(X_val)[:, 1]
    threshold, validation_best_f1 = _best_f1_threshold(y_val, validation_probabilities)

    # This test partition is not used to select the threshold or fit the model.
    test_probabilities = estimator.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_predictions, labels=[0, 1]).ravel()
    test_positive_rate = float(y_test.mean())
    metrics.update(
        {
            "model_status": "trained",
            "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
            "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
            "f1": float(f1_score(y_test, test_predictions, zero_division=0)),
            "accuracy": float(accuracy_score(y_test, test_predictions)),
            "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
            "average_precision": float(average_precision_score(y_test, test_probabilities)),
            "average_precision_baseline": float(test_positive_rate),
            "average_precision_lift_over_prevalence": float(
                average_precision_score(y_test, test_probabilities) / test_positive_rate
            ) if test_positive_rate > 0 else 0.0,
            "test_positive_rate_pct": round(test_positive_rate * 100, 3),
            "always_negative_baseline_accuracy": float(1.0 - test_positive_rate),
            "validation_best_f1": float(validation_best_f1),
            "decision_threshold": float(threshold),
            "probability_calibrated": bool(use_calibration),
            "test_confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        }
    )

    # Refit after test evaluation; final demo model uses train + validation only.
    if use_calibration:
        final_model = CalibratedClassifierCV(estimator=_make_estimator(), method="sigmoid", cv=3)
    else:
        final_model = _make_estimator()
    final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    return final_model, metrics


def train_repeat_purchase_model(orders: pd.DataFrame, horizon_days: int = 180):
    """Convenience wrapper; returns (features, final_model_or_None, metrics)."""
    features, target_info = prepare_repeat_purchase_data(orders, horizon_days)
    model, metrics = fit_repeat_purchase_model(features, target_info)
    return features, model, metrics

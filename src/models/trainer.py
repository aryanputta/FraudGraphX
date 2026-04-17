"""
Model training pipeline: load data, train all three models, evaluate, benchmark.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.common.config import get_settings
from src.common.logging import get_logger
from src.common.schemas import BenchmarkResult
from src.models.base import FEATURE_COLS, LABEL_COL, FraudClassifier
from src.models.gnn_model import GNNFraudClassifier
from src.models.logistic_regression import LogRegFraudClassifier
from src.models.xgboost_model import XGBoostFraudClassifier

logger = get_logger(__name__)
settings = get_settings()


def load_training_data(csv_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare the training dataset."""
    if csv_path is None:
        csv_path = str(Path(__file__).parent.parent.parent / "data" / "sample" / "transactions.csv")

    df = pd.read_csv(csv_path, low_memory=False)

    # Feature engineering
    df["amount_log"] = np.log1p(df["amount"].fillna(0))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour_of_day"] = df["timestamp"].dt.hour.fillna(0).astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
    df["velocity_1h"] = 0
    df["velocity_24h"] = 0
    df["amount_zscore"] = 0.0
    df["merchant_risk_score"] = 0.2
    df["device_seen_before"] = True
    df["ip_seen_before"] = True
    df["geo_distance_km"] = 0.0
    df["is_cross_border"] = False
    df["is_online"] = df.get("is_online", pd.Series([True] * len(df)))

    X = df.reindex(columns=FEATURE_COLS, fill_value=0.0)
    y = df[LABEL_COL].fillna(0).astype(int)
    logger.info("Training data loaded", n=len(df), fraud_rate=f"{y.mean():.3%}")
    return X, y


def evaluate_model(
    model: FraudClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    n_latency_samples: int = 200,
) -> BenchmarkResult:
    """Evaluate a model and measure latency."""
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    # Latency benchmark (single-row inference)
    sample = X_test.head(1)
    latencies = []
    for _ in range(n_latency_samples):
        t0 = time.perf_counter()
        model.predict_proba(sample)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_lat = float(np.mean(latencies))
    p99_lat = float(np.percentile(latencies, 99))
    fraud_dollars = float((y_test * X_test.get("amount_log", pd.Series([0] * len(y_test))).values * y_pred).sum())

    return BenchmarkResult(
        model_name=model.name,
        precision=round(precision_score(y_test, y_pred, zero_division=0), 4),
        recall=round(recall_score(y_test, y_pred, zero_division=0), 4),
        f1=round(f1_score(y_test, y_pred, zero_division=0), 4),
        roc_auc=round(roc_auc_score(y_test, proba), 4),
        avg_latency_ms=round(avg_lat, 2),
        p99_latency_ms=round(p99_lat, 2),
        false_positive_rate=round((y_pred[y_test == 0] == 1).mean(), 4),
        fraud_dollars_prevented=round(fraud_dollars, 2),
        threshold=threshold,
    )


def train_all(data_path: Optional[str] = None, tune: bool = False) -> Dict[str, BenchmarkResult]:
    """Train all three models and return benchmark comparison."""
    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_training_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )

    results: Dict[str, BenchmarkResult] = {}

    # ── Logistic Regression ────────────────────────────────────────────────────
    logger.info("Training Logistic Regression…")
    logreg = LogRegFraudClassifier()
    logreg.fit(X_train, y_train)
    logreg.save(Path(settings.logreg_model_path))
    results["logistic_regression"] = evaluate_model(logreg, X_test, y_test)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    logger.info("Training XGBoost…")
    xgb_clf = XGBoostFraudClassifier()
    if tune:
        xgb_clf.tune(X_train, y_train, n_trials=20)
    xgb_clf.fit(X_train, y_train, eval_set=(X_val, y_val))
    xgb_clf.save(Path(settings.xgboost_model_path))
    results["xgboost"] = evaluate_model(xgb_clf, X_test, y_test)

    # ── GNN ───────────────────────────────────────────────────────────────────
    logger.info("Training GNN…")
    gnn = GNNFraudClassifier(epochs=30)
    gnn.fit(X_train, y_train)
    gnn.save(Path(settings.gnn_model_path))
    results["gnn"] = evaluate_model(gnn, X_test, y_test)

    # Print comparison table
    _print_benchmark_table(results)
    return results


def _print_benchmark_table(results: Dict[str, BenchmarkResult]) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title="FraudGraphX Model Benchmark")
    cols = ["Model", "Precision", "Recall", "F1", "ROC-AUC", "Avg Lat (ms)", "P99 Lat (ms)", "FP Rate"]
    for col in cols:
        table.add_column(col, justify="right")

    for name, r in results.items():
        table.add_row(
            name, str(r.precision), str(r.recall), str(r.f1),
            str(r.roc_auc), str(r.avg_latency_ms), str(r.p99_latency_ms),
            str(r.false_positive_rate),
        )

    Console().print(table)


if __name__ == "__main__":
    train_all(tune=False)

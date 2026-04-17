#!/usr/bin/env python3
"""
FraudGraphX End-to-End Demo
============================
Runs entirely in-process – no Kafka, Cassandra, Neo4j, or GPU required.

What it demonstrates:
  1. Synthetic transaction generation
  2. Real-time feature engineering (velocity, geo, device novelty)
  3. LogReg + XGBoost model training and evaluation
  4. Ensemble scoring with adaptive RL threshold
  5. SHAP explainability
  6. Input validation (security)
  7. Model benchmark comparison table
  8. FastAPI endpoint smoke-test (in-process)
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

# Make src importable when running from repo root or demo/
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic data generation (self-contained, no CSV needed)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_demo_dataset(n_normal: int = 2000, n_fraud: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    def _normal_rows(n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "amount": rng.lognormal(3.5, 1.0, n).clip(1, 5000),
            "hour_of_day": rng.integers(8, 22, n),
            "day_of_week": rng.integers(0, 7, n),
            "velocity_1h": rng.integers(0, 3, n),
            "velocity_24h": rng.integers(0, 10, n),
            "amount_zscore": rng.uniform(-1, 1, n),
            "merchant_risk_score": rng.uniform(0.0, 0.35, n),
            "device_seen_before": np.ones(n),
            "ip_seen_before": np.ones(n),
            "geo_distance_km": rng.uniform(0, 50, n),
            "is_cross_border": np.zeros(n),
            "is_online": rng.integers(0, 2, n).astype(float),
            "amount_log": np.log1p(rng.lognormal(3.5, 1.0, n).clip(1, 5000)),
            "is_fraud": 0,
        })

    def _fraud_rows(n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "amount": rng.uniform(1500, 15000, n),
            "hour_of_day": rng.integers(0, 5, n),
            "day_of_week": rng.integers(0, 7, n),
            "velocity_1h": rng.integers(5, 20, n),
            "velocity_24h": rng.integers(15, 60, n),
            "amount_zscore": rng.uniform(3, 12, n),
            "merchant_risk_score": rng.uniform(0.65, 1.0, n),
            "device_seen_before": rng.integers(0, 2, n).astype(float),
            "ip_seen_before": np.zeros(n),
            "geo_distance_km": rng.uniform(800, 15000, n),
            "is_cross_border": np.ones(n),
            "is_online": np.ones(n),
            "amount_log": np.log1p(rng.uniform(1500, 15000, n)),
            "is_fraud": 1,
        })

    df = pd.concat([_normal_rows(n_normal), _fraud_rows(n_fraud)], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature engineering live demo
# ─────────────────────────────────────────────────────────────────────────────

def _demo_feature_engineering() -> None:
    from src.common.schemas import Transaction
    from src.ingestion.feature_engineering import FeatureEngineer

    console.print("\n[bold cyan]── Feature Engineering Demo ──[/bold cyan]")
    engineer = FeatureEngineer()
    engineer.set_merchant_risk("merch-crypto-001", 0.92)

    transactions = [
        Transaction(
            transaction_id="txn-demo-001",
            amount=150.0,
            customer_id="cust-demo-alice",
            account_id="acc-demo-alice",
            merchant_id="merch-grocery-001",
            merchant_category="grocery",
            device_id="dev-iphone-alice",
            ip_address="192.168.1.10",
            geolocation={"lat": 40.7128, "lon": -74.0060},
            channel="mobile",
        ),
        Transaction(
            transaction_id="txn-demo-002",
            amount=8500.0,
            customer_id="cust-demo-alice",
            account_id="acc-demo-alice",
            merchant_id="merch-crypto-001",
            merchant_category="crypto_exchange",
            device_id="dev-unknown-999",   # new device!
            ip_address="10.0.0.1",
            geolocation={"lat": 51.5074, "lon": -0.1278},  # London – was in NYC!
            channel="web",
        ),
    ]

    table = Table(title="Enriched Transactions", show_lines=True)
    fields = ["transaction_id", "amount", "velocity_1h", "amount_zscore",
              "merchant_risk_score", "device_seen_before", "geo_distance_km", "is_cross_border"]
    for f in fields:
        table.add_column(f, style="cyan" if f == "transaction_id" else "white")

    for txn in transactions:
        enriched = engineer.enrich(txn)
        table.add_row(
            enriched.transaction_id[:12] + "…",
            f"${enriched.amount:,.2f}",
            str(enriched.velocity_1h),
            f"{enriched.amount_zscore:.2f}",
            f"{enriched.merchant_risk_score:.2f}",
            "[green]yes[/green]" if enriched.device_seen_before else "[red]NO (new!)[/red]",
            f"{enriched.geo_distance_km:,.0f} km",
            "[red]YES[/red]" if enriched.is_cross_border else "no",
        )

    console.print(table)
    console.print("[dim]txn-002: new device + jumped from NYC→London + crypto merchant + high amount[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Train models + benchmark
# ─────────────────────────────────────────────────────────────────────────────

def _demo_model_training(df: pd.DataFrame, model_dir: Path) -> Dict[str, Any]:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    from src.models.base import FEATURE_COLS
    from src.models.logistic_regression import LogRegFraudClassifier
    from src.models.xgboost_model import XGBoostFraudClassifier

    console.print("\n[bold cyan]── Model Training & Benchmark ──[/bold cyan]")

    X = df[FEATURE_COLS].fillna(0.0)
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    results: Dict[str, Dict[str, float]] = {}

    def _evaluate(name: str, model: Any, threshold: float = 0.5) -> Dict[str, float]:
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)

        # Latency
        lats = []
        sample = X_test.iloc[:1]
        for _ in range(100):
            t0 = time.perf_counter()
            model.predict_proba(sample)
            lats.append((time.perf_counter() - t0) * 1000)

        return {
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "avg_lat_ms": round(float(np.mean(lats)), 3),
            "p99_lat_ms": round(float(np.percentile(lats, 99)), 3),
            "fp_rate": round(float((y_pred[y_test == 0] == 1).mean()), 4),
        }

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as prog:
        t = prog.add_task("Training Logistic Regression…", total=None)
        logreg = LogRegFraudClassifier()
        logreg.fit(X_train, y_train)
        logreg.save(model_dir / "logreg_fraud.pkl")
        results["logistic_regression"] = _evaluate("LogReg", logreg)
        prog.update(t, description="Training XGBoost…")

        xgb_clf = XGBoostFraudClassifier(params={"n_estimators": 200, "max_depth": 5, "scale_pos_weight": 15})
        xgb_clf.fit(X_train, y_train)
        xgb_clf.save(model_dir / "xgboost_fraud.ubj")
        results["xgboost"] = _evaluate("XGBoost", xgb_clf)

    # Benchmark table
    table = Table(title="Model Benchmark Comparison", show_lines=True)
    table.add_column("Model", style="bold")
    for col in ["Precision", "Recall", "F1", "ROC-AUC", "Avg Lat (ms)", "P99 Lat (ms)", "FP Rate"]:
        table.add_column(col, justify="right")

    def _color(val: float, good: float = 0.8) -> str:
        color = "green" if val >= good else ("yellow" if val >= 0.6 else "red")
        return f"[{color}]{val}[/{color}]"

    for name, r in results.items():
        table.add_row(
            name.replace("_", " "),
            _color(r["precision"]),
            _color(r["recall"]),
            _color(r["f1"]),
            _color(r["roc_auc"]),
            f"[cyan]{r['avg_lat_ms']}[/cyan]",
            f"[cyan]{r['p99_lat_ms']}[/cyan]",
            f"[yellow]{r['fp_rate']}[/yellow]",
        )

    console.print(table)
    return {"logreg": logreg, "xgboost": xgb_clf, "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ensemble scoring + adaptive threshold RL
# ─────────────────────────────────────────────────────────────────────────────

def _demo_ensemble_scoring(models: Dict[str, Any]) -> None:
    from src.models.ensemble import AdaptiveThreshold, EnsembleScorer
    from src.models.base import FEATURE_COLS

    console.print("\n[bold cyan]── Ensemble Scoring & Adaptive Threshold ──[/bold cyan]")

    scorer = EnsembleScorer()
    scorer._models["logistic_regression"] = models["logreg"]
    scorer._models["xgboost"] = models["xgboost"]

    # Sample transactions to score
    test_cases = [
        {"desc": "Normal grocery purchase ($45)", "features": {
            "amount_log": math.log1p(45), "hour_of_day": 14, "day_of_week": 2,
            "velocity_1h": 1, "velocity_24h": 3, "amount_zscore": -0.2,
            "merchant_risk_score": 0.1, "device_seen_before": 1.0, "ip_seen_before": 1.0,
            "geo_distance_km": 2.0, "is_cross_border": 0.0, "is_online": 0.0,
        }},
        {"desc": "Suspicious crypto transfer ($9,500) — new device, London", "features": {
            "amount_log": math.log1p(9500), "hour_of_day": 2, "day_of_week": 6,
            "velocity_1h": 8, "velocity_24h": 22, "amount_zscore": 7.4,
            "merchant_risk_score": 0.91, "device_seen_before": 0.0, "ip_seen_before": 0.0,
            "geo_distance_km": 5570.0, "is_cross_border": 1.0, "is_online": 1.0,
        }},
        {"desc": "Electronics purchase ($1,200) — medium risk", "features": {
            "amount_log": math.log1p(1200), "hour_of_day": 11, "day_of_week": 1,
            "velocity_1h": 3, "velocity_24h": 9, "amount_zscore": 2.1,
            "merchant_risk_score": 0.45, "device_seen_before": 1.0, "ip_seen_before": 0.0,
            "geo_distance_km": 120.0, "is_cross_border": 0.0, "is_online": 1.0,
        }},
        {"desc": "Fraud ring — shared device, high velocity ($3,800)", "features": {
            "amount_log": math.log1p(3800), "hour_of_day": 3, "day_of_week": 0,
            "velocity_1h": 12, "velocity_24h": 35, "amount_zscore": 5.8,
            "merchant_risk_score": 0.78, "device_seen_before": 0.0, "ip_seen_before": 0.0,
            "geo_distance_km": 2200.0, "is_cross_border": 1.0, "is_online": 1.0,
        }},
    ]

    from src.models.base import FEATURE_COLS
    table = Table(title="Live Ensemble Scoring", show_lines=True)
    table.add_column("Transaction", style="bold", max_width=45)
    table.add_column("LR Score", justify="right")
    table.add_column("XGB Score", justify="right")
    table.add_column("Ensemble", justify="right")
    table.add_column("Risk", justify="center")
    table.add_column("Action", justify="center")
    table.add_column("Latency (ms)", justify="right")

    for case in test_cases:
        features_df = pd.DataFrame([case["features"]])
        # Ensure all FEATURE_COLS are present
        features_df = features_df.reindex(columns=FEATURE_COLS, fill_value=0.0)
        result = scorer.score(features_df)

        risk_color = {"critical": "red", "high": "orange3", "medium": "yellow", "low": "green"}
        rc = risk_color.get(result.risk_level.value, "white")
        action_color = {"block": "red", "review": "yellow", "pass": "green"}
        ac = action_color.get(result.action, "white")

        lr_score = next((s.score for s in result.model_scores if s.model_name == "logistic_regression"), 0.0)
        xgb_score = next((s.score for s in result.model_scores if s.model_name == "xgboost"), 0.0)

        table.add_row(
            case["desc"],
            f"{lr_score:.3f}",
            f"{xgb_score:.3f}",
            f"[bold]{result.ensemble_score:.3f}[/bold]",
            f"[{rc}]{result.risk_level.value.upper()}[/{rc}]",
            f"[{ac}]{result.action.upper()}[/{ac}]",
            f"{result.latency_ms:.2f}",
        )

    console.print(table)

    # Adaptive threshold demo
    console.print("\n[bold]Adaptive RL Threshold Demo:[/bold]")
    agent = AdaptiveThreshold(initial_threshold=0.5, lr=0.02)
    console.print(f"  Initial threshold: [cyan]{agent.threshold:.3f}[/cyan]")

    # Simulate analyst feedback
    feedback_seq = [
        (True, False, "FP: normal transaction blocked"),
        (True, False, "FP: another false positive"),
        (False, True, "FN: missed actual fraud"),
        (False, True, "FN: missed actual fraud"),
        (True, True,  "TP: correctly caught fraud"),
    ]
    for predicted, actual, desc in feedback_seq:
        agent.update(predicted, actual)
        console.print(f"  Feedback [{desc}] → threshold: [cyan]{agent.threshold:.3f}[/cyan]")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SHAP explainability
# ─────────────────────────────────────────────────────────────────────────────

def _demo_shap(models: Dict[str, Any], df: pd.DataFrame) -> None:
    try:
        import shap
    except ImportError:
        console.print("[dim]SHAP not installed – skipping explainability demo[/dim]")
        return

    from src.models.base import FEATURE_COLS
    from src.explainability.shap_explainer import SHAPExplainer

    console.print("\n[bold cyan]── SHAP Explainability ──[/bold cyan]")

    xgb_model = models["xgboost"]
    explainer = SHAPExplainer(xgb_model, model_type="xgboost")

    # Score and explain a suspicious transaction
    fraud_sample = df[df["is_fraud"] == 1].head(1)[FEATURE_COLS].fillna(0.0)
    shap_values = explainer.explain(fraud_sample)

    if shap_values:
        top = explainer.top_features(shap_values[0], n=6)
        table = Table(title="SHAP Feature Attribution (Suspicious Transaction)", show_lines=True)
        table.add_column("Feature", style="bold")
        table.add_column("SHAP Value", justify="right")
        table.add_column("Direction", justify="center")
        for item in top:
            color = "red" if item["shap_value"] > 0 else "green"
            table.add_row(
                item["feature"],
                f"[{color}]{item['shap_value']:+.4f}[/{color}]",
                item["direction"],
            )
        console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Input validation security demo
# ─────────────────────────────────────────────────────────────────────────────

def _demo_security() -> None:
    from pydantic import ValidationError
    from src.common.schemas import Transaction

    console.print("\n[bold cyan]── Input Validation & Security ──[/bold cyan]")

    attack_cases = [
        ("SQL injection in customer_id", {"customer_id": "'; DROP TABLE users; --", "amount": 100.0, "account_id": "acc", "merchant_id": "m"}),
        ("Negative amount", {"customer_id": "cust-001", "amount": -500.0, "account_id": "acc", "merchant_id": "m"}),
        ("Zero amount", {"customer_id": "cust-001", "amount": 0.0, "account_id": "acc", "merchant_id": "m"}),
        ("Amount too large", {"customer_id": "cust-001", "amount": 99_999_999.0, "account_id": "acc", "merchant_id": "m"}),
        ("Invalid IP format", {"customer_id": "cust-001", "amount": 100.0, "account_id": "acc", "merchant_id": "m", "ip_address": "<script>alert(1)</script>"}),
        ("Geolocation out of range", {"customer_id": "cust-001", "amount": 100.0, "account_id": "acc", "merchant_id": "m", "geolocation": {"lat": 999.0, "lon": 0.0}}),
    ]

    table = Table(title="Security Validation", show_lines=True)
    table.add_column("Attack Vector", style="bold", max_width=40)
    table.add_column("Blocked?", justify="center")
    table.add_column("Error", max_width=50)

    for desc, payload in attack_cases:
        try:
            Transaction(**payload)
            table.add_row(desc, "[red]NO – PASSED THROUGH[/red]", "–")
        except (ValidationError, ValueError) as e:
            errors = str(e).split("\n")[0][:60]
            table.add_row(desc, "[green]YES – BLOCKED[/green]", errors)

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# 7. FastAPI endpoint smoke test (in-process, no server needed)
# ─────────────────────────────────────────────────────────────────────────────

def _demo_api(models: Dict[str, Any]) -> None:
    from unittest.mock import MagicMock
    from fastapi.testclient import TestClient

    from src.api.main import create_app
    from src.models.ensemble import EnsembleScorer, AdaptiveThreshold, EnsembleResult
    from src.common.schemas import RiskLevel

    console.print("\n[bold cyan]── FastAPI Endpoint Smoke Test ──[/bold cyan]")

    # Build mock deps using real trained models
    mock_deps = MagicMock()
    mock_deps.cassandra = None
    mock_deps.neo4j = None
    mock_deps.graph_builder = None
    mock_deps.graph_path_explainer = None
    mock_deps.llm_investigator = None
    mock_deps.shap_explainer = None

    # Use a real EnsembleScorer with trained models
    real_scorer = EnsembleScorer()
    real_scorer._models["logistic_regression"] = models["logreg"]
    real_scorer._models["xgboost"] = models["xgboost"]
    mock_deps.ensemble = real_scorer

    app = create_app(deps_override=mock_deps)

    test_txns = [
        {
            "desc": "Normal purchase",
            "payload": {
                "transaction": {
                    "amount": 52.50, "customer_id": "cust-demo-001",
                    "account_id": "acc-demo-001", "merchant_id": "merch-grocery-001",
                    "merchant_category": "grocery", "channel": "pos",
                },
                "explain": False,
            },
        },
        {
            "desc": "Suspicious transfer (high-risk features)",
            "payload": {
                "transaction": {
                    "amount": 7800.0, "customer_id": "cust-suspicious-999",
                    "account_id": "acc-suspicious-999", "merchant_id": "merch-crypto-001",
                    "merchant_category": "crypto_exchange", "channel": "web",
                    "ip_address": "203.0.113.45",
                    "geolocation": {"lat": 51.5, "lon": -0.1},
                },
                "explain": False,
            },
        },
    ]

    table = Table(title="Live API Responses", show_lines=True)
    table.add_column("Transaction", style="bold")
    table.add_column("HTTP", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Risk", justify="center")
    table.add_column("Action", justify="center")
    table.add_column("API Latency (ms)", justify="right")
    table.add_column("Security Headers", justify="center")

    with TestClient(app) as client:
        # Test health
        health = client.get("/healthz")
        console.print(f"  [green]GET /healthz → {health.json()}[/green]")

        # Test rejection
        bad = client.post("/v1/score", json={"transaction": {
            "amount": -1.0, "customer_id": "'; DROP TABLE;--",
            "account_id": "acc", "merchant_id": "m",
        }})
        console.print(f"  [green]Injection attempt blocked → HTTP {bad.status_code}[/green]")

        # Score real transactions
        for case in test_txns:
            t0 = time.perf_counter()
            resp = client.post("/v1/score", json=case["payload"])
            api_ms = (time.perf_counter() - t0) * 1000
            data = resp.json()

            risk_color = {"critical": "red", "high": "orange3", "medium": "yellow", "low": "green"}
            rc = risk_color.get(data.get("risk_level", "low"), "white")
            action_color = {"block": "red", "review": "yellow", "pass": "green"}
            ac = action_color.get(data.get("action", "pass"), "white")

            has_security = "x-content-type-options" in resp.headers and "x-frame-options" in resp.headers

            table.add_row(
                case["desc"],
                f"[green]{resp.status_code}[/green]",
                f"{data.get('ensemble_score', 0):.3f}",
                f"[{rc}]{data.get('risk_level', '').upper()}[/{rc}]",
                f"[{ac}]{data.get('action', '').upper()}[/{ac}]",
                f"[{'green' if api_ms < 150 else 'red'}]{api_ms:.1f}[/{'green' if api_ms < 150 else 'red'}]",
                "[green]YES[/green]" if has_security else "[red]NO[/red]",
            )

        # Test feedback endpoint
        fb = client.post("/v1/feedback", json={"alert_id": "a-1", "predicted_fraud": True, "actual_fraud": False})
        console.print(f"  [green]POST /v1/feedback (analyst RL feedback) → HTTP {fb.status_code}[/green]")

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    console.print(Panel.fit(
        "[bold red]FraudGraphX[/bold red] – Enterprise Fraud Detection Platform\n"
        "[dim]End-to-End Demo · No external services required[/dim]",
        border_style="red",
    ))

    with tempfile.TemporaryDirectory() as model_dir:
        model_path = Path(model_dir)

        # Patch settings to use temp dir
        os.environ["MODEL_DIR"] = str(model_path)
        os.environ["LOGREG_MODEL_PATH"] = str(model_path / "logreg_fraud.pkl")
        os.environ["XGBOOST_MODEL_PATH"] = str(model_path / "xgboost_fraud.ubj")
        os.environ["GNN_MODEL_PATH"] = str(model_path / "gnn_fraud.pt")

        # Clear the lru_cache so env changes take effect
        from src.common.config import get_settings
        get_settings.cache_clear()

        # Step 1: Generate data
        console.print("\n[bold cyan]── Generating Synthetic Dataset ──[/bold cyan]")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as prog:
            prog.add_task("Generating 2,150 transactions (150 fraud)…", total=None)
            df = _generate_demo_dataset(n_normal=2000, n_fraud=150)
        fraud_pct = df["is_fraud"].mean() * 100
        console.print(f"  [green]✓[/green] {len(df):,} transactions generated — fraud rate: [red]{fraud_pct:.1f}%[/red]")

        # Step 2: Feature engineering
        _demo_feature_engineering()

        # Step 3: Train models
        trained = _demo_model_training(df, model_path)

        # Step 4: Ensemble scoring
        _demo_ensemble_scoring(trained)

        # Step 5: SHAP
        _demo_shap(trained, df)

        # Step 6: Security validation
        _demo_security()

        # Step 7: API smoke test
        _demo_api(trained)

    console.print(Panel.fit(
        "[bold green]Demo complete![/bold green]\n\n"
        "Next steps:\n"
        "  [cyan]make up[/cyan]              — start all services (Docker Compose)\n"
        "  [cyan]make generate-data[/cyan]   — generate full 200k transaction dataset\n"
        "  [cyan]make train[/cyan]           — train all models on real data\n"
        "  [cyan]make api-dev[/cyan]         — run FastAPI server (localhost:8000/docs)\n"
        "  [cyan]make frontend-dev[/cyan]    — run React UI (localhost:3000)\n"
        "  [cyan]make test[/cyan]            — run full test suite",
        border_style="green",
    ))


if __name__ == "__main__":
    main()

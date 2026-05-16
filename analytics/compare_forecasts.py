"""Backtest Prophet vs SARIMA on the same compliance series.

Train on first N-7 days, forecast next 7 days, compare MAPE on held-out window.
Logs to MLflow (file:./mlruns), saves per-type plot HTML + JSON to
evaluation/forecast_compare/. Populates ADR-008 + model card.

CLI:
    python -m analytics.compare_forecasts
    python -m analytics.compare_forecasts --types "NO-Hardhat" "No_Harness"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from analytics.forecast import load_compliance_series

warnings.filterwarnings("ignore")
load_dotenv()
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

DB_PATH = Path(os.getenv("SAFETYVISION_DB", "/tmp/violations.db"))
OUT_DIR = Path("evaluation/forecast_compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TYPES = ["NO-Hardhat", "NO-Safety Vest", "No_Harness"]


def _prophet_forecast(train_df: pd.DataFrame, horizon: int) -> np.ndarray:
    m = Prophet(
        weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False,
    )
    m.fit(train_df)
    future = m.make_future_dataframe(periods=horizon)
    return m.predict(future)["yhat"].tail(horizon).values


def _sarima_forecast(train_df: pd.DataFrame, horizon: int) -> np.ndarray:
    series = train_df.set_index("ds")["y"]
    m = SARIMAX(
        series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fit = m.fit(disp=False)
    return fit.get_forecast(steps=horizon).predicted_mean.values


def backtest(violation_type: str, db_path: Path, horizon: int = 7) -> dict:
    """Train/test split: last `horizon` days are held out for MAPE."""
    full = load_compliance_series(violation_type, db_path, days=60)
    if len(full) < 21:
        raise ValueError(
            f"Need >=21 days for backtest (got {len(full)}). "
            "Run `python -m analytics.seed_violations` first."
        )

    train = full.iloc[:-horizon].copy()
    test = full.iloc[-horizon:].copy()

    prophet_pred = _prophet_forecast(train, horizon)
    sarima_pred = _sarima_forecast(train, horizon)
    sarima_pred = np.clip(sarima_pred, 0.0, 1.0)

    y_true = test["y"].values
    prophet_mape = float(mean_absolute_percentage_error(y_true, prophet_pred))
    sarima_mape = float(mean_absolute_percentage_error(y_true, sarima_pred))

    return {
        "violation_type": violation_type,
        "train_days": int(len(train)),
        "test_days": int(len(test)),
        "prophet_mape": prophet_mape,
        "sarima_mape": sarima_mape,
        "winner": "prophet" if prophet_mape < sarima_mape else "sarima",
        "train_df": train,
        "test_df": test,
        "prophet_pred": prophet_pred.tolist(),
        "sarima_pred": sarima_pred.tolist(),
    }


def _plot_compare(result: dict) -> go.Figure:
    train, test = result["train_df"], result["test_df"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train["ds"], y=train["y"], mode="markers+lines",
        name="Train", marker=dict(size=5, color="black"),
        line=dict(color="black", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=test["ds"], y=test["y"], mode="markers",
        name="Test (actual)",
        marker=dict(size=10, color="black", symbol="diamond"),
    ))
    fig.add_trace(go.Scatter(
        x=test["ds"], y=result["prophet_pred"], mode="lines+markers",
        name=f"Prophet (MAPE={result['prophet_mape']:.4f})",
        line=dict(color="rgb(0,100,200)", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=test["ds"], y=result["sarima_pred"], mode="lines+markers",
        name=f"SARIMA  (MAPE={result['sarima_mape']:.4f})",
        line=dict(color="rgb(220,80,40)", width=2),
    ))
    fig.update_layout(
        title=f"Prophet vs SARIMA — {result['violation_type']}",
        xaxis_title="Date", yaxis_title="Compliance Rate",
        height=450, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def main(types: list, horizon: int = 7) -> None:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("safetyvision-forecasting")

    rows = []
    for vtype in types:
        print(f"\n=== {vtype} ===")
        with mlflow.start_run(run_name=f"compare_{_safe(vtype)}"):
            result = backtest(vtype, DB_PATH, horizon)

            mlflow.log_param("violation_type", vtype)
            mlflow.log_param("horizon_days", horizon)
            mlflow.log_param("train_days", result["train_days"])
            mlflow.log_param("sarima_order", "(1,1,1)x(1,1,1,7)")
            mlflow.log_metric("prophet_mape", result["prophet_mape"])
            mlflow.log_metric("sarima_mape", result["sarima_mape"])
            mlflow.set_tag("winner", result["winner"])

            print(f"  Prophet MAPE: {result['prophet_mape']:.4f}")
            print(f"  SARIMA  MAPE: {result['sarima_mape']:.4f}")
            print(f"  Winner:       {result['winner'].upper()}")

            plot_path = OUT_DIR / f"compare_{_safe(vtype)}.html"
            _plot_compare(result).write_html(plot_path)
            mlflow.log_artifact(str(plot_path))

            json_path = OUT_DIR / f"compare_{_safe(vtype)}.json"
            serializable = {k: v for k, v in result.items() if k not in ("train_df", "test_df")}
            json_path.write_text(json.dumps(serializable, indent=2, default=str))
            mlflow.log_artifact(str(json_path))

            rows.append({
                "type": vtype,
                "prophet_mape": result["prophet_mape"],
                "sarima_mape": result["sarima_mape"],
                "winner": result["winner"],
            })

    print("\n" + "=" * 64)
    print("SUMMARY (lower MAPE = better)")
    print("=" * 64)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    avg = {
        "prophet_mape_avg": float(df["prophet_mape"].mean()),
        "sarima_mape_avg": float(df["sarima_mape"].mean()),
        "overall_winner": "prophet" if df["prophet_mape"].mean() < df["sarima_mape"].mean() else "sarima",
    }
    print(f"\nAverage MAPE — Prophet: {avg['prophet_mape_avg']:.4f}  |  "
          f"SARIMA: {avg['sarima_mape_avg']:.4f}  |  "
          f"Winner: {avg['overall_winner'].upper()}")

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps({"per_type": rows, "overall": avg}, indent=2))
    print(f"\nSummary -> {summary_path}")
    print("MLflow UI: mlflow ui --backend-store-uri file:./mlruns")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--types", nargs="+", default=DEFAULT_TYPES)
    p.add_argument("--horizon", type=int, default=7)
    args = p.parse_args()
    main(args.types, args.horizon)

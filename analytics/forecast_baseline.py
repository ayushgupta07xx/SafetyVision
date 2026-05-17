"""SARIMA(1,1,1)(1,1,1,7) baseline forecast (statsmodels) vs Prophet.

Period=7 captures weekly seasonality on daily compliance series.
Same input contract as analytics.forecast — feeds compare_forecasts.py.

CLI:
    python -m analytics.forecast_baseline "NO-Hardhat" --save /tmp/sarima_check.html
"""
from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from statsmodels.tsa.statespace.sarimax import SARIMAX

from analytics.forecast import load_compliance_series

warnings.filterwarnings("ignore")  # statsmodels convergence chatter
load_dotenv()
logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("SAFETYVISION_DB", "/tmp/violations.db"))


def forecast_compliance_sarima(
    violation_type: str,
    db_path: Path = DB_PATH,
    history_days: int = 30,
    horizon_days: int = 7,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 7),
) -> tuple[pd.DataFrame, go.Figure]:
    """Fit SARIMA, return (forecast_df[ds, yhat, yhat_lower, yhat_upper], figure)."""
    df = load_compliance_series(violation_type, db_path, history_days)
    if len(df) < 14:
        raise ValueError(f"Need >=14 days of history (got {len(df)}).")

    series = df.set_index("ds")["y"]
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)

    fc = fit.get_forecast(steps=horizon_days)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.20)  # 80% CI

    future_dates = pd.date_range(
        start=series.index[-1] + pd.Timedelta(days=1),
        periods=horizon_days, freq="D",
    )
    fc_df = pd.DataFrame({
        "ds": future_dates,
        "yhat": mean.values,
        "yhat_lower": ci.iloc[:, 0].values,
        "yhat_upper": ci.iloc[:, 1].values,
    })
    # Clamp to [0, 1] — SARIMA has no native bound
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        fc_df[col] = fc_df[col].clip(lower=0.0, upper=1.0)

    fig = _plot_forecast(df, fc_df, violation_type)
    return fc_df, fig


def _plot_forecast(history: pd.DataFrame, forecast: pd.DataFrame, vtype: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_upper"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_lower"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(220,80,40,0.18)", name="80% CI",
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"],
        mode="lines", line=dict(color="rgb(220,80,40)", width=2), name="SARIMA forecast",
    ))
    fig.add_trace(go.Scatter(
        x=history["ds"], y=history["y"],
        mode="markers", marker=dict(size=6, color="black"), name="History",
    ))
    fig.update_layout(
        title=f"7-Day Compliance Forecast (SARIMA 1,1,1×1,1,1,7) — {vtype}",
        xaxis_title="Date", yaxis_title="Compliance Rate",
        yaxis=dict(range=[0, 1.05]),
        height=420, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("violation_type")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--save", default=None)
    args = p.parse_args()

    forecast, fig = forecast_compliance_sarima(
        args.violation_type, history_days=args.days, horizon_days=args.horizon,
    )
    print(forecast.to_string(index=False))
    if args.save:
        fig.write_html(args.save)
        print(f"\nSaved plot -> {args.save}")

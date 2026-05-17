"""Prophet 7-day compliance forecasting.

Loads daily compliance rate per violation type from SQLite, fits Prophet with
weekly seasonality, returns 7-day forecast + Plotly figure for Gradio.

CLI:
    python -m analytics.forecast "NO-Hardhat"
    python -m analytics.forecast "NO-Hardhat" --save /tmp/prophet_check.html
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from prophet import Prophet

load_dotenv()
logger = logging.getLogger(__name__)

# Quiet Prophet/stan chatter
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

DB_PATH = Path(os.getenv("SAFETYVISION_DB", "/tmp/violations.db"))


def load_compliance_series(
    violation_type: str,
    db_path: Path = DB_PATH,
    days: int = 30,
) -> pd.DataFrame:
    """Return DataFrame[ds, y] where y = 1 - violations/inspections per day.

    Uses LEFT JOIN so days with zero violations of this type still contribute
    (compliance_rate = 1.0 on those days).
    """
    query = """
        WITH daily_v AS (
            SELECT
                date(timestamp_ms/1000, 'unixepoch') AS d,
                COUNT(*) AS v
            FROM violations
            WHERE violation_type = ?
              AND date(timestamp_ms/1000, 'unixepoch') >= date('now', ?)
            GROUP BY d
        )
        SELECT
            i.inspection_date AS ds,
            1.0 - COALESCE(dv.v, 0) * 1.0 / i.total_inspections AS y
        FROM daily_inspections i
        LEFT JOIN daily_v dv ON dv.d = i.inspection_date
        WHERE i.inspection_date >= date('now', ?)
        ORDER BY i.inspection_date
    """
    window = f"-{days} days"
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=(violation_type, window, window))
    df["ds"] = pd.to_datetime(df["ds"])
    # Clamp to [0, 1] in case of any odd noise
    df["y"] = df["y"].clip(lower=0.0, upper=1.0)
    return df


def forecast_compliance(
    violation_type: str,
    db_path: Path = DB_PATH,
    history_days: int = 30,
    horizon_days: int = 7,
) -> tuple[pd.DataFrame, go.Figure]:
    """Fit Prophet, return (full_forecast_df, plotly_figure)."""
    df = load_compliance_series(violation_type, db_path, history_days)
    if len(df) < 14:
        raise ValueError(
            f"Need >=14 days of history for weekly seasonality (got {len(df)}). "
            "Run `python -m analytics.seed_violations` first."
        )

    model = Prophet(
        weekly_seasonality=True,
        daily_seasonality=False,
        yearly_seasonality=False,
        interval_width=0.80,
    )
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)

    fig = _plot_forecast(df, forecast, violation_type)
    return forecast, fig


def _plot_forecast(history: pd.DataFrame, forecast: pd.DataFrame, vtype: str) -> go.Figure:
    fig = go.Figure()
    # CI band
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_upper"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_lower"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(0,100,200,0.18)", name="80% CI",
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"],
        mode="lines", line=dict(color="rgb(0,100,200)", width=2), name="Prophet forecast",
    ))
    fig.add_trace(go.Scatter(
        x=history["ds"], y=history["y"],
        mode="markers", marker=dict(size=6, color="black"), name="History",
    ))
    fig.update_layout(
        title=f"7-Day Compliance Forecast (Prophet) — {vtype}",
        xaxis_title="Date", yaxis_title="Compliance Rate",
        yaxis=dict(range=[0, 1.05]),
        height=420, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("violation_type", help="e.g. 'NO-Hardhat'")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--save", default=None, help="Save plot HTML to this path")
    args = p.parse_args()

    forecast, fig = forecast_compliance(
        args.violation_type, history_days=args.days, horizon_days=args.horizon,
    )
    tail = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(args.horizon)
    print(tail.to_string(index=False))
    if args.save:
        fig.write_html(args.save)
        print(f"\nSaved plot -> {args.save}")

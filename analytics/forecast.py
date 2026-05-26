"""Prophet 7-day compliance forecasting.

Loads daily compliance rate per violation type from SQLite, fits Prophet with
weekly seasonality, returns 7-day forecast + Plotly figure + a plain-language
summary for Gradio.

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
    source: str = "sqlite",
    user_id: str | None = None,
) -> pd.DataFrame:
    """Return DataFrame[ds, y] where y = 1 - violations/inspections per day.

    Uses LEFT JOIN so days with zero violations of this type still contribute
    (compliance_rate = 1.0 on those days).
    """
    if source == "supabase":
        from core import supabase_db  # lazy import — only when Supabase is the source
        df = supabase_db.fetch_compliance_series(violation_type, days=days, user_id=user_id)
        df["y"] = df["y"].clip(lower=0.0, upper=1.0)
        return df
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
    source: str = "sqlite",
    user_id: str | None = None,
) -> tuple[pd.DataFrame, go.Figure, str]:
    """Fit Prophet, return (full_forecast_df, plotly_figure, plain_language_summary)."""
    df = load_compliance_series(violation_type, db_path, history_days, source=source, user_id=user_id)
    if len(df) < 14:
        hint = ("python -m analytics.seed_supabase --user-id <uuid>" if source == "supabase"
                else "python -m analytics.seed_violations")
        raise ValueError(
            f"Need >=14 days of history for weekly seasonality (got {len(df)}). Run `{hint}` first."
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
    summary = _summarize_forecast(df, forecast, violation_type, horizon_days)
    return forecast, fig, summary


def _summarize_forecast(
    history: pd.DataFrame, forecast: pd.DataFrame, vtype: str, horizon_days: int
) -> str:
    """One-line plain-language takeaway for non-technical readers."""
    def _clamp(x: float) -> float:
        return min(100.0, max(0.0, x))
    recent = _clamp(float(history["y"].tail(7).mean()) * 100)
    fut = forecast.tail(horizon_days)
    proj_end = _clamp(float(fut["yhat"].iloc[-1]) * 100)
    delta = _clamp(float(fut["yhat"].mean()) * 100) - recent
    if delta > 2:
        trend = "trending up"
    elif delta < -2:
        trend = "trending down"
    else:
        trend = "roughly stable"
    end_date = fut["ds"].iloc[-1].strftime("%b %d").replace(" 0", " ")
    return (
        f"Recent compliance averaged {recent:.0f}% over the last 7 days. "
        f"The 7-day forecast is {trend}, with about {proj_end:.0f}% projected by {end_date}. "
        f"Compliance rate = the share of checks where the required PPE was present "
        f"(higher is better). Dots = the past 30 days; line = the Prophet forecast; "
        f"shaded band = the 80% uncertainty range."
    )
def _plot_forecast(history: pd.DataFrame, forecast: pd.DataFrame, vtype: str) -> go.Figure:
    split = history["ds"].max()      # last actual day; forecast begins after this
    last = forecast["ds"].max()

    BG = "#0f172a"
    ACCENT = "#60a5fa"
    ACCENT_SOFT = "rgba(96,165,250,0.15)"
    INK = "#e5e7eb"
    MUTED = "#94a3b8"
    GRID = "rgba(255,255,255,0.08)"

    fig = go.Figure()
    # 80% uncertainty band
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_upper"] * 100,
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_lower"] * 100,
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor=ACCENT_SOFT, name="Uncertainty (80%)", hoverinfo="skip",
    ))
    # Forecast line — smoothed
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"] * 100,
        mode="lines", line=dict(color=ACCENT, width=3, shape="spline"),
        name="Forecast", hovertemplate="%{x|%b %d}: %{y:.0f}%<extra></extra>",
    ))
    # Actual history — hollow dots
    fig.add_trace(go.Scatter(
        x=history["ds"], y=history["y"] * 100,
        mode="markers",
        marker=dict(size=7, color=BG, line=dict(color=INK, width=1.5)),
        name="Actual (past 30 days)",
        hovertemplate="%{x|%b %d}: %{y:.0f}%<extra></extra>",
    ))

    # Forecast window shading + divider
    fig.add_shape(type="rect", xref="x", yref="paper", x0=split, x1=last, y0=0, y1=1,
                  fillcolor="rgba(255,255,255,0.04)", line_width=0, layer="below")
    fig.add_shape(type="line", xref="x", yref="paper", x0=split, x1=split, y0=0, y1=1,
                  line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=split, xref="x", y=1.0, yref="paper", text="forecast →",
                       showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=11, color=MUTED))

    fig.update_layout(
        template="plotly_dark",
        font=dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif",
                  size=13, color=INK),
        title=dict(text=f"Compliance forecast — {vtype}",
                   font=dict(size=18, color=INK), x=0.01, xanchor="left", y=0.96),
        paper_bgcolor=BG, plot_bgcolor=BG,
        height=440, margin=dict(l=55, r=24, t=56, b=84),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="", showgrid=False, showline=False, zeroline=False,
                   ticks="outside", tickcolor=GRID, color=MUTED),
        yaxis=dict(title="Compliance rate (%)", range=[0, 105], ticksuffix="%",
                   showgrid=True, gridcolor=GRID, showline=False, zeroline=False,
                   dtick=20, color=MUTED),
        hoverlabel=dict(bgcolor="#1e293b", font=dict(color=INK), bordercolor=GRID),
    )
    return fig

def forecast_json(
    violation_type: str,
    db_path: Path = DB_PATH,
    history_days: int = 30,
    horizon_days: int = 7,
    source: str = "sqlite",
    user_id: str | None = None,
) -> dict:
    """API-friendly forecast for the Mode-2 /forecast endpoint.

    Same Prophet fit as forecast_compliance, but returns JSON-serializable
    history + forecast points + a summary -- no Plotly figure (the API path
    must not build figures).
    """
    df = load_compliance_series(
        violation_type, db_path, history_days, source=source, user_id=user_id
    )
    if len(df) < 14:
        raise ValueError(
            f"Need >=14 days of history for weekly seasonality (got {len(df)})."
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

    fut = forecast.tail(horizon_days)
    points = [
        {
            "ds": ds.strftime("%Y-%m-%d"),
            "yhat": round(float(yh), 4),
            "yhat_lower": round(float(lo), 4),
            "yhat_upper": round(float(hi), 4),
        }
        for ds, yh, lo, hi in zip(
            fut["ds"], fut["yhat"], fut["yhat_lower"], fut["yhat_upper"], strict=True
        )
    ]
    history = [
        {"ds": ds.strftime("%Y-%m-%d"), "y": round(float(y), 4)}
        for ds, y in zip(df["ds"], df["y"], strict=True)
    ]
    return {
        "violation_type": violation_type,
        "history_days": history_days,
        "horizon_days": horizon_days,
        "recent_compliance": round(float(df["y"].tail(7).mean()), 4),
        "summary": _summarize_forecast(df, forecast, violation_type, horizon_days),
        "history": history,
        "forecast": points,
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("violation_type", help="e.g. 'NO-Hardhat'")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--source", default="sqlite", choices=["supabase", "sqlite"])
    p.add_argument("--user-id", default=None, help="auth.users UUID (Supabase source)")
    p.add_argument("--save", default=None, help="Save plot HTML to this path")
    args = p.parse_args()

    forecast, fig, summary = forecast_compliance(
        args.violation_type, history_days=args.days, horizon_days=args.horizon,
        source=args.source, user_id=args.user_id,
    )
    print(summary, "\n")
    tail = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(args.horizon)
    print(tail.to_string(index=False))
    if args.save:
        fig.write_html(args.save)
        print(f"\nSaved plot -> {args.save}")
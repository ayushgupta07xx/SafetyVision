"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { SV_KEY_STORAGE } from "@/lib/keys";
import { getForecast, VIOLATION_TYPES, type ForecastResult } from "@/lib/api";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";

type Row = { ds: string; actual: number | null; predicted: number | null; band: [number, number] | null };

const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
const mmdd = (ds: string) => ds.slice(5);
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const TEAL = "#0d9488";

function toRows(data: ForecastResult): Row[] {
  const rows: Row[] = [
    ...data.history.map((h) => ({ ds: h.ds, actual: clamp(h.y), predicted: null, band: null }) as Row),
    ...data.forecast.map((f) => ({
      ds: f.ds, actual: null, predicted: clamp(f.yhat),
      band: [clamp(f.yhat_lower), clamp(f.yhat_upper)] as [number, number],
    })),
  ];
  if (data.history.length && data.forecast.length) {
    const last = rows[data.history.length - 1];
    last.predicted = last.actual;
    last.band = [last.actual as number, last.actual as number];
  }
  return rows;
}

export default function ForecastPage() {
  const [vt, setVt] = useState<string>("NO-Hardhat");
  const [data, setData] = useState<ForecastResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window !== "undefined") setApiKey(localStorage.getItem(SV_KEY_STORAGE));
  }, []);

  useEffect(() => {
    if (!apiKey) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);
    getForecast(vt, apiKey)
      .then((d) => { if (!cancelled) setData(d); })
      .catch((e) => { if (!cancelled) setError(e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [vt, apiKey]);

  const rows = data ? toRows(data) : [];
  const interval = rows.length > 16 ? Math.floor(rows.length / 8) : 0;

  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">Forecast</p>
          <h1 className="mt-1 text-3xl font-bold tracking-tight">Compliance forecast</h1>
        </div>
        <Select value={vt} onValueChange={setVt}>
          <SelectTrigger className="w-[200px]"><SelectValue /></SelectTrigger>
          <SelectContent>
            {VIOLATION_TYPES.map((t) => (
              <SelectItem key={t} value={t}>{t}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {!apiKey && (
        <div className="mt-8 rounded-xl border border-border bg-card p-10 text-center text-muted-foreground">
          No API key found. Mint one on the <Link href="/account" className="text-primary underline">Account</Link> page first.
        </div>
      )}

      {apiKey && loading && (
        <div className="mt-8 rounded-xl border border-border bg-card p-12 text-center text-muted-foreground">Loading forecast...</div>
      )}

      {apiKey && !loading && error && (
        <div className="mt-8 rounded-xl border border-border bg-card p-10 text-center text-muted-foreground">
          {error.startsWith("422")
            ? "Not enough history yet - Prophet needs at least 14 days of inspections for this violation type."
            : `Forecast failed - ${error}`}
        </div>
      )}

      {apiKey && !loading && !error && data && (
        <div className="mt-8 space-y-6">
          <div className="rounded-xl border border-border bg-card p-6">
            <p className="text-sm font-medium">{data.violation_type} &middot; recent compliance <span className="font-mono text-primary">{pct(data.recent_compliance)}</span></p>
            <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-muted-foreground">{data.summary}</p>
          </div>

          <div className="rounded-xl border border-border bg-card p-6">
            <p className="text-sm font-medium">{data.history_days}-day history + {data.horizon_days}-day forecast (80% CI)</p>
            <div className="mt-5 text-muted-foreground" style={{ width: "100%", height: 360 }}>
              <ResponsiveContainer>
                <ComposedChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.15} />
                  <XAxis dataKey="ds" tickFormatter={mmdd} interval={interval} tick={{ fill: "currentColor", fontSize: 11 }} stroke="currentColor" />
                  <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={pct} tick={{ fill: "currentColor", fontSize: 11 }} stroke="currentColor" width={42} allowDataOverflow />
                  <Tooltip
                    contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 8, color: "var(--foreground)" }}
                    formatter={(value: unknown, name) => {
                      if (Array.isArray(value)) return [`${pct(value[0])} - ${pct(value[1])}`, "80% CI"];
                      return [pct(value as number), name];
                    }}
                  />
                  <Legend />
                  <Area dataKey="band" name="80% CI" stroke="none" fill={TEAL} fillOpacity={0.12} connectNulls />
                  <Line dataKey="actual" name="Compliance" stroke={TEAL} dot={false} strokeWidth={2} connectNulls />
                  <Line dataKey="predicted" name="Forecast" stroke={TEAL} strokeDasharray="5 4" dot={false} strokeWidth={2} connectNulls />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

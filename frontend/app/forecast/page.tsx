"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { SV_KEY_STORAGE } from "@/lib/keys";
import { getForecast, VIOLATION_TYPES, type ForecastResult } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";

type Row = { ds: string; actual: number | null; predicted: number | null; band: [number, number] | null };

const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
const mmdd = (ds: string) => ds.slice(5);
const clamp = (v: number) => Math.max(0, Math.min(1, v));

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
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <h1 className="text-2xl font-semibold">Compliance Forecast</h1>
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
        <Card><CardContent className="py-8 text-center text-muted-foreground">
          No API key found. Mint one on the{" "}
          <Link href="/account" className="underline">Account</Link> page first.
        </CardContent></Card>
      )}

      {apiKey && loading && (
        <Card><CardContent className="py-12 text-center text-muted-foreground">Loading forecast...</CardContent></Card>
      )}

      {apiKey && !loading && error && (
        <Card><CardContent className="py-8 text-center text-muted-foreground">
          {error.startsWith("422")
            ? "Not enough history yet - Prophet needs at least 14 days of inspections for this violation type."
            : `Forecast failed - ${error}`}
        </CardContent></Card>
      )}

      {apiKey && !loading && !error && data && (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                {data.violation_type} - recent compliance {pct(data.recent_compliance)}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground whitespace-pre-wrap">{data.summary}</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle className="text-base">
              {data.history_days}-day history + {data.horizon_days}-day forecast (80% CI)
            </CardTitle></CardHeader>
            <CardContent>
              <div style={{ width: "100%", height: 360 }}>
                <ResponsiveContainer>
                  <ComposedChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                    <XAxis dataKey="ds" tickFormatter={mmdd} interval={interval} fontSize={11} />
                    <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={pct} fontSize={11} width={42} allowDataOverflow />
                    <Tooltip
                      formatter={(value: unknown, name) => {
                        if (Array.isArray(value)) return [`${pct(value[0])} - ${pct(value[1])}`, "80% CI"];
                        return [pct(value as number), name];
                      }}
                    />
                    <Legend />
                    <Area dataKey="band" name="80% CI" stroke="none" fill="#2563eb" fillOpacity={0.15} connectNulls />
                    <Line dataKey="actual" name="Compliance" stroke="#0d9488" dot={false} strokeWidth={2} connectNulls />
                    <Line dataKey="predicted" name="Forecast" stroke="#16a34a" strokeDasharray="5 4" dot={false} strokeWidth={2} connectNulls />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

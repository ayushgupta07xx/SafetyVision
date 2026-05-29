import { Link } from "next-view-transitions";
import { createClient } from "@/lib/supabase/server";
import { RiskBadge } from "@/components/risk-badge";

export const dynamic = "force-dynamic";

const DAY = 86_400_000;

type RecentRow = {
  violation_id: string;
  timestamp_ms: number;
  violation_type: string;
  risk_level: string;
  confidence: number;
  pdf_report_url: string | null;
};

export default async function DashboardPage() {
  const supabase = createClient();
  const now = Date.now();
  const ms7 = now - 7 * DAY;
  const ms14 = now - 14 * DAY;

  const [recentRes, totalVRes, totalIRes, last7Res, trendRes] = await Promise.all([
    supabase.from("violations").select("violation_id, timestamp_ms, violation_type, risk_level, confidence, pdf_report_url").order("timestamp_ms", { ascending: false }).limit(5),
    supabase.from("violations").select("*", { count: "exact", head: true }),
    supabase.from("inspections").select("*", { count: "exact", head: true }),
    supabase.from("violations").select("*", { count: "exact", head: true }).gte("timestamp_ms", ms7),
    supabase.from("violations").select("timestamp_ms").gte("timestamp_ms", ms14),
  ]);

  const recent = (recentRes.data ?? []) as RecentRow[];
  const totalViolations = totalVRes.count ?? 0;
  const totalInspections = totalIRes.count ?? 0;
  const last7 = last7Res.count ?? 0;

  const days = 14;
  const startOfToday = new Date();
  startOfToday.setHours(0, 0, 0, 0);
  const startMs = startOfToday.getTime() - (days - 1) * DAY;
  const buckets = new Array(days).fill(0) as number[];
  for (const v of (trendRes.data ?? []) as { timestamp_ms: number }[]) {
    const idx = Math.floor((v.timestamp_ms - startMs) / DAY);
    if (idx >= 0 && idx < days) buckets[idx] += 1;
  }
  const maxB = Math.max(1, ...buckets);
  const stats = [
    { label: "Total inspections", value: totalInspections },
    { label: "Total violations", value: totalViolations },
    { label: "Last 7 days", value: last7 },
  ];
  const empty = totalInspections === 0 && totalViolations === 0;

  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">Dashboard</p>
          <h1 className="mt-1 text-3xl font-bold tracking-tight">Your safety overview</h1>
        </div>
        <Link href="/upload" className="rounded-lg bg-primary px-5 py-2.5 font-semibold text-primary-foreground transition-all hover:-translate-y-0.5 hover:opacity-90">Analyze new image</Link>
      </div>

      {empty ? (
        <div className="mt-10 rounded-xl border border-border bg-card p-10 text-center">
          <p className="text-muted-foreground">No inspections yet.</p>
          <Link href="/upload" className="mt-4 inline-block rounded-lg bg-primary px-5 py-2.5 font-semibold text-primary-foreground hover:opacity-90">Upload your first image</Link>
        </div>
      ) : (
        <>
          <div className="mt-8 grid gap-4 sm:grid-cols-3">
            {stats.map((s) => (
              <div key={s.label} className="rounded-xl border border-border bg-card p-5">
                <div className="text-sm text-muted-foreground">{s.label}</div>
                <div className="mt-2 font-mono text-3xl font-bold tracking-tight">{s.value}</div>
              </div>
            ))}
          </div>

          <div className="mt-6 rounded-xl border border-border bg-card p-5">
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">Violations, last 14 days</div>
              <Link href="/forecast" className="text-sm font-medium text-primary hover:underline">7-day forecast &rarr;</Link>
            </div>
            <div className="mt-5 flex h-24 items-end gap-1.5">
              {buckets.map((c, i) => (
                <div key={i} className="flex-1 rounded-t bg-primary" style={{ height: `${Math.max(4, (c / maxB) * 100)}%`, opacity: c === 0 ? 0.18 : 1 }} title={`${c}`} />
              ))}
            </div>
          </div>

          <div className="mt-6 rounded-xl border border-border bg-card">
            <div className="flex items-center justify-between border-b border-border p-5">
              <div className="text-sm font-medium">Recent violations</div>
              <Link href="/history" className="text-sm font-medium text-primary hover:underline">View all &rarr;</Link>
            </div>
            <ul className="divide-y divide-border">
              {recent.map((r) => (
                <li key={r.violation_id} className="flex items-center gap-4 p-4">
                  <RiskBadge level={r.risk_level} />
                  <span className="font-medium">{r.violation_type}</span>
                  <span className="font-mono text-sm text-muted-foreground">{(r.confidence <= 1 ? r.confidence * 100 : r.confidence).toFixed(0)}%</span>
                  <span className="ml-auto text-sm text-muted-foreground">{new Date(r.timestamp_ms).toLocaleDateString()}</span>
                  <Link href={`/violations/${r.violation_id}`} className="rounded-md border border-border px-3 py-1.5 text-sm font-medium transition-colors hover:bg-accent">View</Link>
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </main>
  );
}

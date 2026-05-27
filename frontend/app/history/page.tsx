import Link from "next/link";
import { createClient } from "@/lib/supabase/server";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { RiskBadge } from "@/components/risk-badge";

export const dynamic = "force-dynamic";

type ViolationRow = {
  violation_id: string;
  timestamp_ms: number;
  violation_type: string;
  risk_level: string;
  confidence: number;
  pdf_report_url: string | null;
  source: string | null;
};

export default async function HistoryPage() {
  const supabase = createClient();
  const { data, error } = await supabase
    .from("violations")
    .select("violation_id, timestamp_ms, violation_type, risk_level, confidence, pdf_report_url, source")
    .order("timestamp_ms", { ascending: false })
    .limit(200);

  const rows = (data ?? []) as ViolationRow[];

  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">History</p>
          <h1 className="mt-1 text-3xl font-bold tracking-tight">Violation history</h1>
          <p className="mt-2 text-sm text-muted-foreground">{rows.length} record{rows.length === 1 ? "" : "s"}</p>
        </div>
        <Link href="/upload" className="rounded-lg bg-primary px-5 py-2.5 font-semibold text-primary-foreground transition-all hover:-translate-y-0.5 hover:opacity-90">
          Analyze new image
        </Link>
      </div>

      {error && <p className="mt-6 text-sm text-red-600">Failed to load: {error.message}</p>}

      {rows.length === 0 && !error ? (
        <div className="mt-10 rounded-xl border border-border bg-card p-10 text-center text-muted-foreground">
          No violations yet.{" "}
          <Link href="/upload" className="text-primary underline">Upload an image</Link> to get started.
        </div>
      ) : (
        <div className="mt-8 overflow-hidden rounded-xl border border-border bg-card">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Date</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Risk</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>Source</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r) => (
                <TableRow key={r.violation_id}>
                  <TableCell className="whitespace-nowrap text-muted-foreground">{new Date(r.timestamp_ms).toLocaleString()}</TableCell>
                  <TableCell className="font-medium">{r.violation_type}</TableCell>
                  <TableCell><RiskBadge level={r.risk_level} /></TableCell>
                  <TableCell className="font-mono">{(r.confidence <= 1 ? r.confidence * 100 : r.confidence).toFixed(0)}%</TableCell>
                  <TableCell className="text-muted-foreground">{r.source ?? "\u2014"}</TableCell>
                  <TableCell className="text-right space-x-2">
                    <Button asChild variant="outline" size="sm">
                      <Link href={`/violations/${r.violation_id}`}>View</Link>
                    </Button>
                    {r.pdf_report_url && (
                      <Button asChild variant="outline" size="sm">
                        <a href={r.pdf_report_url} target="_blank" rel="noopener noreferrer">PDF</a>
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </main>
  );
}

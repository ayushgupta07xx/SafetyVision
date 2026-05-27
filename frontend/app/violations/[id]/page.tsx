import Link from "next/link";
import { notFound } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { RiskBadge } from "@/components/risk-badge";

export const dynamic = "force-dynamic";

type Props = { params: { id: string } };

export default async function ViolationDetailPage({ params }: Props) {
  const supabase = createClient();
  const { data, error } = await supabase
    .from("violations")
    .select("violation_id, timestamp_ms, violation_type, risk_level, confidence, regulation_cited, summary, pdf_report_url, source")
    .eq("violation_id", params.id)
    .maybeSingle();

  if (error) {
    return <main className="mx-auto max-w-3xl px-6 py-12 text-red-600">Failed to load: {error.message}</main>;
  }
  if (!data) notFound();

  const v = data as {
    violation_id: string;
    timestamp_ms: number;
    violation_type: string;
    risk_level: string;
    confidence: number;
    regulation_cited: string | null;
    summary: string | null;
    pdf_report_url: string | null;
    source: string | null;
  };

  const conf = (v.confidence <= 1 ? v.confidence * 100 : v.confidence).toFixed(0);

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <div className="flex items-center justify-between gap-4">
        <Link href="/history" className="text-sm font-medium text-muted-foreground transition-colors hover:text-foreground">&larr; Back to history</Link>
        {v.pdf_report_url && (
          <a href={v.pdf_report_url} target="_blank" rel="noopener noreferrer" className="rounded-lg bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground transition-all hover:-translate-y-0.5 hover:opacity-90">Download PDF report</a>
        )}
      </div>

      <div className="mt-6 rounded-xl border border-border bg-card p-6">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold tracking-tight">{v.violation_type}</h1>
          <RiskBadge level={v.risk_level} />
        </div>
        <p className="mt-2 font-mono text-sm text-muted-foreground">
          {new Date(v.timestamp_ms).toLocaleString()} &middot; {conf}% confidence &middot; {v.source ?? "\u2014"}
        </p>

        <div className="mt-6 space-y-5">
          {v.regulation_cited && (
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">OSHA citation</p>
              <p className="mt-1 text-sm">{v.regulation_cited}</p>
            </div>
          )}
          {v.summary && (
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">Incident summary</p>
              <p className="mt-1 whitespace-pre-wrap text-sm leading-relaxed">{v.summary}</p>
            </div>
          )}
          {!v.pdf_report_url && (
            <p className="text-xs text-muted-foreground">No PDF stored (reports are generated for the primary violation per inspection). Live GradCAM/SHAP tabs exist only at analyze time.</p>
          )}
        </div>
      </div>
    </main>
  );
}

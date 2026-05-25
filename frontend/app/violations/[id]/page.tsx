import Link from "next/link";
import { notFound } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
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
    return <div className="max-w-3xl mx-auto p-6 text-red-600">Failed to load: {error.message}</div>;
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
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <Button asChild variant="ghost" size="sm">
          <Link href="/history">&larr; Back to history</Link>
        </Button>
        {v.pdf_report_url && (
          <Button asChild>
            <a href={v.pdf_report_url} target="_blank" rel="noopener noreferrer">Download PDF report</a>
          </Button>
        )}
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <CardTitle className="text-xl">{v.violation_type}</CardTitle>
            <RiskBadge level={v.risk_level} />
          </div>
          <p className="text-sm text-muted-foreground">
            {new Date(v.timestamp_ms).toLocaleString()} &middot; {conf}% confidence &middot; {v.source ?? "\u2014"}
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          {v.regulation_cited && (
            <div>
              <h3 className="text-sm font-semibold">OSHA Citation</h3>
              <p className="text-sm">{v.regulation_cited}</p>
            </div>
          )}
          {v.summary && (
            <div>
              <h3 className="text-sm font-semibold">Incident Summary</h3>
              <p className="text-sm whitespace-pre-wrap">{v.summary}</p>
            </div>
          )}
          {!v.pdf_report_url && (
            <p className="text-xs text-muted-foreground">
              No PDF stored (reports are generated for the primary violation per inspection). Live GradCAM/SHAP tabs exist only at analyze time.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

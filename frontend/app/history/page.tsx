import Link from "next/link";
import { createClient } from "@/lib/supabase/server";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      <h1 className="text-2xl font-semibold">Violation History</h1>
      {error && <p className="text-red-600 text-sm">Failed to load: {error.message}</p>}
      {rows.length === 0 && !error ? (
        <Card>
          <CardContent className="py-10 text-center text-muted-foreground">
            No violations yet.{" "}
            <Link href="/upload" className="underline">Upload an image</Link> to get started.
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>{rows.length} record{rows.length === 1 ? "" : "s"}</CardTitle>
          </CardHeader>
          <CardContent>
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
                    <TableCell className="whitespace-nowrap">
                      {new Date(r.timestamp_ms).toLocaleString()}
                    </TableCell>
                    <TableCell className="font-medium">{r.violation_type}</TableCell>
                    <TableCell><RiskBadge level={r.risk_level} /></TableCell>
                    <TableCell>
                      {(r.confidence <= 1 ? r.confidence * 100 : r.confidence).toFixed(0)}%
                    </TableCell>
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
          </CardContent>
        </Card>
      )}
    </div>
  );
}

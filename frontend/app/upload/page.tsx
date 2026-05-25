"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { analyzeImage, type AnalyzeResult } from "@/lib/api";
import { SV_KEY_STORAGE } from "@/lib/keys";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const MAX_MB = 6;

function riskBadge(level?: string) {
  const high = level === "HIGH" || level === "CRITICAL";
  return <Badge variant={high ? "destructive" : "secondary"}>{level ?? "—"}</Badge>;
}

function Img({ b64, alt }: { b64: string; alt: string }) {
  if (!b64) return <p className="p-4 text-sm text-gray-500">Not available.</p>;
  // eslint-disable-next-line @next/next/no-img-element
  return (
    <img
      src={`data:image/png;base64,${b64}`}
      alt={alt}
      className="mt-2 max-h-[480px] w-auto rounded border"
    />
  );
}

function Results({ r }: { r: AnalyzeResult }) {
  const ir = r.incident_report;
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            {r.violations.length} violation{r.violations.length === 1 ? "" : "s"} · {r.processing_time_ms} ms
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {r.violations.length === 0 ? (
            <p className="text-sm text-gray-500">No PPE violations detected.</p>
          ) : (
            <ul className="space-y-1 text-sm">
              {r.violations.map((v) => (
                <li key={v.violation_id} className="flex items-center gap-2">
                  {riskBadge(v.risk_level)}
                  <span className="font-medium">{v.class}</span>
                  <span className="text-gray-500">{(v.confidence * 100).toFixed(0)}%</span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <Tabs defaultValue="annotated">
            <TabsList>
              <TabsTrigger value="annotated">Annotated</TabsTrigger>
              <TabsTrigger value="gradcam">GradCAM</TabsTrigger>
              <TabsTrigger value="shap">SHAP</TabsTrigger>
            </TabsList>
            <TabsContent value="annotated"><Img b64={r.annotated_image_b64} alt="annotated" /></TabsContent>
            <TabsContent value="gradcam"><Img b64={r.gradcam_b64} alt="gradcam" /></TabsContent>
            <TabsContent value="shap"><Img b64={r.shap_chart_b64} alt="shap" /></TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {ir && (
        <Card>
          <CardHeader><CardTitle className="text-base">Incident report</CardTitle></CardHeader>
          <CardContent className="space-y-3 text-sm">
            {ir.error ? (
              <p className="text-red-600">Report unavailable: {ir.error}</p>
            ) : (
              <>
                {ir.summary && <p>{ir.summary}</p>}
                {ir.regulation_cited && (
                  <p>
                    <span className="font-medium">{ir.regulation_cited}</span>
                    {ir.regulation_text ? ` — ${ir.regulation_text}` : ""}
                  </p>
                )}
                {ir.corrective_actions && ir.corrective_actions.length > 0 && (
                  <div>
                    <p className="font-medium">Corrective actions</p>
                    <ul className="list-disc pl-5">
                      {ir.corrective_actions.map((a, i) => <li key={i}>{a}</li>)}
                    </ul>
                  </div>
                )}
                {ir.follow_up_timeline && (
                  <p><span className="font-medium">Follow-up:</span> {ir.follow_up_timeline}</p>
                )}
              </>
            )}
            {r.pdf_report_url && (
              <a href={r.pdf_report_url} target="_blank" rel="noreferrer">
                <Button variant="outline" size="sm">Download PDF report</Button>
              </a>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default function UploadPage() {
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window !== "undefined") setApiKey(localStorage.getItem(SV_KEY_STORAGE));
  }, []);

  const tooBig = file ? file.size > MAX_MB * 1024 * 1024 : false;
  const okType = file
    ? ["image/jpeg", "image/png"].includes(file.type) || /\.(jpe?g|png)$/i.test(file.name)
    : true;

  async function run() {
    if (!file || !apiKey || !okType || tooBig) return;
    setLoading(true);
    setErr(null);
    setResult(null);
    try {
      setResult(await analyzeImage(file, apiKey));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  if (!apiKey) {
    return (
      <main className="mx-auto max-w-2xl p-6">
        <Card>
          <CardContent className="space-y-2 p-6 text-sm">
            <p>No API key found in this browser.</p>
            <p>
              <Link href="/account" className="text-teal-700 underline">Mint one on the Account page</Link> first, then return here.
            </p>
          </CardContent>
        </Card>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-3xl space-y-6 p-6">
      <h1 className="text-2xl font-semibold">Analyze a worksite image</h1>
      <Card>
        <CardContent className="space-y-3 p-6">
          <input
            type="file"
            accept="image/jpeg,image/png"
            onChange={(e) => { setFile(e.target.files?.[0] ?? null); setResult(null); setErr(null); }}
            className="block text-sm"
          />
          {file && (
            <p className="text-sm text-gray-600">
              {file.name} · {(file.size / 1024 / 1024).toFixed(2)} MB
              {tooBig && <span className="text-red-600"> · over {MAX_MB} MB limit</span>}
              {!okType && <span className="text-red-600"> · only JPEG/PNG supported</span>}
            </p>
          )}
          <Button onClick={run} disabled={!file || tooBig || !okType || loading}>
            {loading ? "Analyzing… (~45–60s)" : "Analyze"}
          </Button>
          {loading && (
            <p className="text-sm text-gray-500">Detection + GradCAM/SHAP + OSHA report. First call may cold-start.</p>
          )}
          {err && <p className="text-sm text-red-600">{err}</p>}
        </CardContent>
      </Card>
      {result && <Results r={result} />}
    </main>
  );
}

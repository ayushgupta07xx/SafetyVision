"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { analyzeImage, type AnalyzeResult } from "@/lib/api";
import { SV_KEY_STORAGE } from "@/lib/keys";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RiskBadge } from "@/components/risk-badge";
import { Analyzing } from "@/components/analyzing";

const MAX_MB = 4;

function mapError(err: string): string {
  console.error("[upload] raw error:", err);
  const m = err.match(/Analyze failed \((\d+)\)/);
  if (m) {
    const s = parseInt(m[1], 10);
    if (s === 400) return "Couldn't decode that image. Try a different JPEG or PNG.";
    if (s === 401) return "API key invalid or revoked. Mint a new one on the Account page.";
    if (s === 403) return "Request blocked (likely CORS or revoked permission). Check console.";
    if (s === 413) return "Image too large for the server — try a file under 4MB.";
    if (s === 429) return "Rate limit hit. Wait a moment and try again.";
    if (s >= 500 && s < 600) return "Server error (cold start or response too large). Retry in ~10s.";
    return `Unexpected response (${s}). Check console for details.`;
  }
  if (err.includes("Failed to fetch") || err.includes("NetworkError")) {
    return "Couldn't reach the server (network or CORS issue). Check console for details.";
  }
  return "Something went wrong. Check console for details.";
}

function Img({ b64, alt }: { b64: string; alt: string }) {
  if (!b64) return <p className="p-4 text-sm text-muted-foreground">Not available.</p>;
  // eslint-disable-next-line @next/next/no-img-element
  return (
    <img
      src={`data:image/png;base64,${b64}`}
      alt={alt}
      className="mt-3 max-h-[480px] w-auto rounded-lg border border-border"
    />
  );
}

function Results({ r }: { r: AnalyzeResult }) {
  const ir = r.incident_report;
  return (
    <div className="mt-6 space-y-6">
      <div className="rounded-xl border border-border bg-card p-6">
        <p className="text-sm font-medium">
          {r.violations.length} violation{r.violations.length === 1 ? "" : "s"} <span className="font-mono text-muted-foreground">&middot; {r.processing_time_ms} ms</span>
        </p>
        <div className="mt-4">
          {r.violations.length === 0 ? (
            <p className="text-sm text-muted-foreground">No PPE violations detected.</p>
          ) : (
            <ul className="space-y-2">
              {r.violations.map((v) => (
                <li key={v.violation_id} className="flex items-center gap-3 text-sm">
                  <RiskBadge level={v.risk_level} />
                  <span className="font-medium">{v.class}</span>
                  <span className="font-mono text-muted-foreground">{(v.confidence * 100).toFixed(0)}%</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="rounded-xl border border-border bg-card p-6">
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
      </div>

      {ir && (
        <div className="rounded-xl border border-border bg-card p-6">
          <h2 className="text-sm font-semibold">Incident report</h2>
          <div className="mt-3 space-y-3 text-sm">
            {ir.error ? (
              <p className="text-red-600">Report unavailable: {ir.error}</p>
            ) : (
              <>
                {ir.summary && <p className="leading-relaxed">{ir.summary}</p>}
                {ir.regulation_cited && (
                  <p>
                    <span className="font-medium">{ir.regulation_cited}</span>
                    {ir.regulation_text ? ` — ${ir.regulation_text}` : ""}
                  </p>
                )}
                {ir.corrective_actions && ir.corrective_actions.length > 0 && (
                  <div>
                    <p className="font-medium">Corrective actions</p>
                    <ul className="mt-1 list-disc pl-5">
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
              <a href={r.pdf_report_url} target="_blank" rel="noreferrer" className="inline-block rounded-lg border border-border px-4 py-2 text-sm font-medium transition-colors hover:bg-accent">Download PDF report</a>
            )}
          </div>
        </div>
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
      <main className="mx-auto max-w-2xl px-6 py-12">
        <div className="rounded-xl border border-border bg-card p-6 text-sm">
          <p>No API key found in this browser.</p>
          <p className="mt-2"><Link href="/account" className="text-primary underline">Mint one on the Account page</Link> first, then return here.</p>
        </div>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">Analyze</p>
      <h1 className="mt-1 text-3xl font-bold tracking-tight">Analyze a worksite image</h1>

      <div className="mt-8 rounded-xl border border-border bg-card p-6">
        <input
          type="file"
          accept="image/jpeg,image/png"
          onChange={(e) => { setFile(e.target.files?.[0] ?? null); setResult(null); setErr(null); }}
          className="block w-full text-sm text-muted-foreground file:mr-4 file:rounded-lg file:border-0 file:bg-primary file:px-4 file:py-2 file:text-sm file:font-semibold file:text-primary-foreground hover:file:opacity-90"
        />
        <p className="mt-3 text-sm text-muted-foreground">
          Images only here (4MB limit).{" "}
          <a href="https://huggingface.co/spaces/ayushgupta7777/safetyvision" target="_blank" rel="noopener noreferrer" className="text-primary underline">Need video? Try the open-source demo &rarr;</a>
        </p>
        {file && (
          <p className="mt-2 text-sm text-muted-foreground">
            {file.name} &middot; {(file.size / 1024 / 1024).toFixed(2)} MB
            {tooBig && <span className="text-red-600"> &middot; over {MAX_MB} MB limit</span>}
            {!okType && <span className="text-red-600"> &middot; only JPEG/PNG supported</span>}
          </p>
        )}
        <Button onClick={run} disabled={!file || tooBig || !okType || loading} className="mt-4">
          {loading ? "Analyzing…" : "Analyze"}
        </Button>
        {loading && <Analyzing />}
        {err && <p className="mt-3 text-sm text-red-600">{mapError(err)}</p>}
      </div>

      {result && <Results r={result} />}
    </main>
  );
}

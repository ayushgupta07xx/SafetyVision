import Link from "next/link";
import { ScanEye, Sparkles, FileText, TrendingUp, FileDown, Code2 } from "lucide-react";
import { createClient } from "@/lib/supabase/server";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// External links -- verify these match your deployed targets.
const HF_SPACES = "https://huggingface.co/spaces/ayushgupta7777/safetyvision";
const GITHUB = "https://github.com/ayushgupta07xx/SafetyVision";
const DOCS = `${process.env.NEXT_PUBLIC_LAMBDA_URL}/docs`;

const FEATURES = [
  { icon: ScanEye, title: "PPE detection", desc: "YOLOv8s spots hard hats, vests, masks, gloves, goggles, and fall-protection gaps with bounding boxes." },
  { icon: Sparkles, title: "Explainable by default", desc: "Every detection ships a GradCAM heatmap and SHAP attribution, so you can see why it was flagged." },
  { icon: FileText, title: "OSHA-grounded reports", desc: "Multimodal Gemini plus RAG over 29 CFR 1910 and 1926 writes incident reports citing the actual regulation." },
  { icon: TrendingUp, title: "Compliance forecasting", desc: "Prophet projects each violation type 7 days out with an 80% confidence band, so trends surface early." },
  { icon: FileDown, title: "PDF incident reports", desc: "One-page downloadable reports per violation: annotated image, citation, corrective actions, explainability." },
  { icon: Code2, title: "API and Python SDK", desc: "A documented REST endpoint and a pip-installable client to wire detection into your own stack." },
];

export default async function Home() {
  const supabase = createClient();
  const { data: { user } } = await supabase.auth.getUser();

  return (
    <main className="mx-auto max-w-5xl px-6">
      <section className="py-20 text-center">
        <p className="mb-3 text-sm font-medium uppercase tracking-wide text-teal-700">
          Open-source workplace safety
        </p>
        <h1 className="mx-auto max-w-3xl text-4xl font-bold tracking-tight sm:text-5xl">
          Catch PPE violations before they become incidents
        </h1>
        <p className="mx-auto mt-5 max-w-2xl text-lg text-muted-foreground">
          Upload a worksite photo and get violation detection, a GradCAM and SHAP explanation, an
          OSHA-grounded incident report, a downloadable PDF, and a 7-day compliance forecast.
          Free and self-hostable.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          {user ? (
            <Button asChild size="lg"><Link href="/upload">Open dashboard</Link></Button>
          ) : (
            <Button asChild size="lg"><Link href="/login">Get started for free</Link></Button>
          )}
          <Button asChild size="lg" variant="outline">
            <a href={HF_SPACES} target="_blank" rel="noopener noreferrer">Try the open-source demo</a>
          </Button>
        </div>
        <p className="mt-4 text-xs text-muted-foreground">
          A free alternative to $500-$2,000/month commercial tools. No account needed for the demo.
        </p>
      </section>

      <section className="grid gap-4 pb-16 sm:grid-cols-2 lg:grid-cols-3">
        {FEATURES.map((f) => {
          const Icon = f.icon;
          return (
            <Card key={f.title}>
              <CardHeader>
                <Icon className="h-6 w-6 text-teal-600" />
                <CardTitle className="text-base">{f.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">{f.desc}</p>
              </CardContent>
            </Card>
          );
        })}
      </section>

      <section className="border-t py-12">
        <h2 className="text-center text-xl font-semibold">Four ways to use it</h2>
        <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader><CardTitle className="text-sm">Web app</CardTitle></CardHeader>
            <CardContent className="text-sm">
              {user ? (
                <Link href="/upload" className="text-teal-700 underline">Upload an image &rarr;</Link>
              ) : (
                <Link href="/login" className="text-teal-700 underline">Sign in &rarr;</Link>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle className="text-sm">Open-source demo</CardTitle></CardHeader>
            <CardContent className="text-sm">
              <a href={HF_SPACES} target="_blank" rel="noopener noreferrer" className="text-teal-700 underline">Hugging Face Spaces &rarr;</a>
            </CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle className="text-sm">Production API</CardTitle></CardHeader>
            <CardContent className="text-sm">
              <a href={DOCS} target="_blank" rel="noopener noreferrer" className="text-teal-700 underline">API docs (Swagger) &rarr;</a>
            </CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle className="text-sm">Deploy your own</CardTitle></CardHeader>
            <CardContent className="text-sm">
              <a href={GITHUB} target="_blank" rel="noopener noreferrer" className="text-teal-700 underline">GitHub and Terraform &rarr;</a>
            </CardContent>
          </Card>
        </div>
      </section>

      <footer className="border-t py-8 text-center text-sm text-muted-foreground">
        <p>AI-assisted pre-screening to support human safety officers. Not a replacement for human judgment.</p>
        <div className="mt-2 flex justify-center gap-4">
          <a href={GITHUB} target="_blank" rel="noopener noreferrer" className="underline">GitHub</a>
          <a href={DOCS} target="_blank" rel="noopener noreferrer" className="underline">API docs</a>
          <a href={HF_SPACES} target="_blank" rel="noopener noreferrer" className="underline">Demo</a>
        </div>
      </footer>
    </main>
  );
}

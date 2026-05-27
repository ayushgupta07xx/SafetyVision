import { Link } from "next-view-transitions";
import { createClient } from "@/lib/supabase/server";
import { Reveal } from "@/components/reveal";
import { FeatureScroll } from "@/components/feature-scroll";
import { HeroDetection } from "@/components/hero-detection";

const HF_SPACES = "https://huggingface.co/spaces/ayushgupta7777/safetyvision";
const GITHUB = "https://github.com/ayushgupta07xx/SafetyVision";
const DOCS = `${process.env.NEXT_PUBLIC_LAMBDA_URL}/docs`;

const STEPS = [
  { n: "01", t: "Upload a frame", d: "Drop in a worksite image. Detection runs on a fine-tuned YOLOv8s model, no GPU needed." },
  { n: "02", t: "See why it flagged", d: "Every detection ships a GradCAM heatmap and SHAP attribution, so the call is never a black box." },
  { n: "03", t: "Get the report", d: "An OSHA-grounded incident report and a 7-day compliance forecast, exportable as a PDF." },
];

function FourCard({ title, href, label, internal }: { title: string; href: string; label: string; internal?: boolean }) {
  const inner = (
    <div className="h-full rounded-xl border border-border bg-card p-5 transition-all hover:-translate-y-1 hover:border-primary active:translate-y-0">
      <div className="text-sm font-semibold">{title}</div>
      <div className="mt-6 font-medium text-primary">{label} &rarr;</div>
    </div>
  );
  return internal ? <Link href={href}>{inner}</Link> : <a href={href} target="_blank" rel="noopener noreferrer">{inner}</a>;
}

export default async function Home() {
  const supabase = createClient();
  const { data: { user } } = await supabase.auth.getUser();
  const primaryHref = user ? "/dashboard" : "/login";
  const primaryLabel = user ? "Open dashboard" : "Get started for free";

  return (
    <main>
      <section className="relative overflow-hidden px-6 pb-20 pt-24 text-center">
        <div aria-hidden className="pointer-events-none absolute -left-24 -top-40 h-[480px] w-[480px] rounded-full bg-[#ffd9c2] opacity-50 blur-3xl dark:bg-primary dark:opacity-10" />
        <div aria-hidden className="pointer-events-none absolute -right-20 -top-32 h-[440px] w-[440px] rounded-full bg-[#cfe3f5] opacity-50 blur-3xl dark:bg-[#3a7ca5] dark:opacity-10" />
        <div aria-hidden className="pointer-events-none absolute left-1/3 top-28 h-[360px] w-[360px] rounded-full bg-[#d8efe4] opacity-40 blur-3xl dark:bg-primary dark:opacity-[0.07]" />
        <div className="relative mx-auto max-w-3xl">
          <p className="rise mb-5 text-xs font-semibold uppercase tracking-[0.14em] text-primary">Open-source workplace safety</p>
          <h1 className="rise text-balance text-5xl font-bold leading-[1.05] tracking-tight sm:text-6xl" style={{ animationDelay: ".06s" }}>Catch PPE violations before they become incidents.</h1>
          <p className="rise mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-muted-foreground" style={{ animationDelay: ".12s" }}>Upload a worksite photo and get violation detection, a GradCAM and SHAP explanation, an OSHA-grounded incident report, and a 7-day compliance forecast. Free and self-hostable.</p>
          <div className="rise mt-9 flex flex-wrap items-center justify-center gap-3" style={{ animationDelay: ".18s" }}>
            <Link href={primaryHref} className="rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground transition-all hover:-translate-y-0.5 hover:opacity-90 active:translate-y-0 active:scale-[0.98]">{primaryLabel}</Link>
            <a href={HF_SPACES} target="_blank" rel="noopener noreferrer" className="rounded-lg border border-border bg-card px-6 py-3 font-semibold transition-all duration-150 hover:bg-accent active:scale-[0.98]">Try the open-source demo</a>
          </div>
          <div className="rise mt-16 flex flex-wrap items-center justify-center gap-x-14 gap-y-6" style={{ animationDelay: ".24s" }}>
            <div><div className="text-2xl font-bold tracking-tight">Sub-second</div><div className="mt-1 text-sm text-muted-foreground">warm detection</div></div>
            <div><div className="text-2xl font-bold tracking-tight">7 classes</div><div className="mt-1 text-sm text-muted-foreground">hard hat, vest, harness, more</div></div>
            <div><div className="font-mono text-2xl font-bold tracking-tight">$0</div><div className="mt-1 text-sm text-muted-foreground">forever, self-hosted</div></div>
          </div>
        </div>
        <div className="rise mx-auto mt-16 grid max-w-5xl items-start gap-10 text-left md:grid-cols-2" style={{ animationDelay: ".3s" }}>
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest text-primary">Live detection</p>
            <h2 className="mt-3 text-2xl font-semibold tracking-tight md:text-3xl">See what the model sees.</h2>
            <p className="mt-4 leading-relaxed text-muted-foreground">
              SafetyVision scans any worksite photo, finds each worker, and checks them against the PPE the model is trained to spot &mdash; hard hats, vests, masks, gloves, harnesses. Every item gets a labelled box and a confidence score, and anything missing is flagged in red, like the unprotected worker in this shot.
            </p>
            <ul className="mt-6 space-y-3 text-sm text-muted-foreground">
              <li className="flex items-start gap-3"><span className="mt-0.5 text-primary">&#10003;</span> A labelled bounding box around every worker and piece of PPE</li>
              <li className="flex items-start gap-3"><span className="mt-0.5 text-primary">&#10003;</span> A confidence score on every single detection</li>
              <li className="flex items-start gap-3"><span className="mt-0.5 text-primary">&#10003;</span> Missing gear flagged in red and ranked by risk level</li>
            </ul>
          </div>
          <div className="mx-auto w-fit max-w-full rounded-2xl border border-border bg-card p-2 shadow-xl">
            <HeroDetection />
          </div>
        </div>
      </section>

      <section id="how" className="border-t border-border px-6 py-24">
        <div className="mx-auto max-w-6xl">
          <Reveal>
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">How it works</p>
            <h2 className="mt-3 max-w-2xl text-4xl font-bold tracking-tight">From a single photo to an audit-ready report.</h2>
          </Reveal>
          <div className="mt-12 grid gap-8 sm:grid-cols-3">
            {STEPS.map((s, i) => (
              <Reveal key={s.n} delay={i * 90}>
                <div className="font-mono text-sm font-medium text-primary">{s.n}</div>
                <h3 className="mt-3 text-xl font-semibold">{s.t}</h3>
                <p className="mt-2 leading-relaxed text-muted-foreground">{s.d}</p>
                <div className="mt-5 h-[3px] w-9 rounded-full bg-primary" />
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <FeatureScroll />

      <section className="border-t border-border px-6 py-24">
        <div className="mx-auto max-w-6xl">
          <Reveal><h2 className="text-center text-3xl font-bold tracking-tight">Four ways to use it</h2></Reveal>
          <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Reveal delay={0}><FourCard title="Web app" href={user ? "/upload" : "/login"} label={user ? "Upload an image" : "Sign in"} internal /></Reveal>
            <Reveal delay={70}><FourCard title="Open-source demo" href={HF_SPACES} label="Hugging Face Spaces" /></Reveal>
            <Reveal delay={140}><FourCard title="Production API" href={DOCS} label="API docs (Swagger)" /></Reveal>
            <Reveal delay={210}><FourCard title="Deploy your own" href={GITHUB} label="GitHub and Terraform" /></Reveal>
          </div>
        </div>
      </section>

      <section className="border-t border-border px-6 py-28 text-center">
        <Reveal>
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">Free &amp; self-hostable</p>
          <h2 className="mx-auto mt-3 max-w-2xl text-4xl font-bold tracking-tight">Safety monitoring shouldn&rsquo;t cost $2,000 a month.</h2>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <Link href={primaryHref} className="rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground transition-all hover:-translate-y-0.5 hover:opacity-90 active:translate-y-0 active:scale-[0.98]">{primaryLabel}</Link>
          </div>
        </Reveal>
      </section>

      <footer className="border-t border-border px-6 py-10 text-center text-sm text-muted-foreground">
        <p>AI-assisted pre-screening to support human safety officers. Not a replacement for human judgment.</p>
        <div className="mt-3 flex justify-center gap-5">
          <a href={GITHUB} target="_blank" rel="noopener noreferrer" className="hover:text-foreground">GitHub</a>
          <a href={DOCS} target="_blank" rel="noopener noreferrer" className="hover:text-foreground">API docs</a>
          <a href={HF_SPACES} target="_blank" rel="noopener noreferrer" className="hover:text-foreground">Demo</a>
        </div>
      </footer>
    </main>
  );
}

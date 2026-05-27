"use client";

import { useEffect, useRef, useState } from "react";

type Feature = {
  kicker: string;
  title: string;
  body: string;
  points: string[];
} & (
  | { kind: "pair"; imgs: { src: string; label: string }[] }
  | { kind: "single"; img: string }
);

const FEATURES: Feature[] = [
  {
    kicker: "Explainable by default",
    title: "Detection you can actually trust.",
    body: "Not a black box. Every violation comes with the evidence behind it.",
    points: [
      "GradCAM heatmap over the detection region",
      "SHAP pixel attribution for the flagged class",
      "Confidence, bounding box, and risk level on every call",
    ],
    kind: "pair",
    imgs: [
      { src: "feature-explain-gradcam.png", label: "GradCAM" },
      { src: "feature-explain-shap.png", label: "SHAP attribution" },
    ],
  },
  {
    kicker: "Grounded reports",
    title: "Reports that cite the actual regulation.",
    body: "Multimodal Gemini reads the annotated frame and writes the report, grounded in real OSHA 29 CFR text via RAG.",
    points: [
      "Cites 29 CFR 1910 / 1926 by section",
      "Corrective actions and follow-up timeline",
      "One-click PDF with a human-review footer",
    ],
    kind: "single",
    img: "feature-report.png",
  },
  {
    kicker: "Compliance forecasting",
    title: "See the trend before the incident.",
    body: "Prophet projects each violation type seven days out with an 80% confidence band, so risk surfaces early.",
    points: [
      "Per-type 7-day forecast with uncertainty band",
      "Prophet vs SARIMA baseline",
      "Reads your own violation history",
    ],
    kind: "single",
    img: "feature-forecast.png",
  },
];

export function FeatureScroll() {
  const trackRef = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(0);

  useEffect(() => {
    let raf = 0;
    const update = () => {
      const el = trackRef.current;
      if (!el) return;
      const vh = window.innerHeight;
      const total = Math.max(el.offsetHeight - vh, 1);
      const scrolled = Math.min(Math.max(-el.getBoundingClientRect().top, 0), total);
      const progress = scrolled / total;
      const idx = Math.min(FEATURES.length - 1, Math.floor(progress * FEATURES.length));
      setActive((prev) => (prev === idx ? prev : idx));
    };
    const onScroll = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, []);

  return (
    <section ref={trackRef} id="detect" className="relative border-t border-border" style={{ height: `${FEATURES.length * 100}vh` }}>
      <div className="sticky top-0 flex h-screen items-center overflow-hidden">
        <div className="mx-auto grid w-full max-w-6xl items-center gap-12 px-6 md:grid-cols-2">
          <div className="relative min-h-[320px]">
            {FEATURES.map((f, i) => (
              <div
                key={f.kicker}
                className={`absolute inset-0 transition-all duration-500 ${
                  i === active ? "translate-y-0 opacity-100" : "pointer-events-none translate-y-3 opacity-0"
                }`}
              >
                <p className="text-xs font-semibold uppercase tracking-widest text-primary">{f.kicker}</p>
                <h2 className="mt-3 text-2xl font-semibold tracking-tight md:text-3xl">{f.title}</h2>
                <p className="mt-4 max-w-md leading-relaxed text-muted-foreground">{f.body}</p>
                <ul className="mt-6 space-y-3 text-sm text-muted-foreground">
                  {f.points.map((pt) => (
                    <li key={pt} className="flex items-start gap-3">
                      <span className="mt-0.5 text-primary">&#10003;</span> {pt}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
            <div className="absolute -bottom-10 left-0 flex gap-2">
              {FEATURES.map((f, i) => (
                <span
                  key={f.kicker}
                  className={`h-1.5 rounded-full transition-all duration-300 ${
                    i === active ? "w-8 bg-primary" : "w-1.5 bg-muted-foreground/40"
                  }`}
                />
              ))}
            </div>
          </div>

          <div className="relative h-[360px] [perspective:1200px]">
            {FEATURES.map((f, i) => (
              <div
                key={f.kicker}
                className={`absolute inset-0 flex items-center justify-center transition-all duration-700 ${
                  i === active
                    ? "opacity-100 [transform:rotateY(0deg)_scale(1)]"
                    : "pointer-events-none opacity-0 [transform:rotateY(12deg)_scale(0.96)]"
                }`}
              >
                {f.kind === "pair" ? (
                  <div className="flex justify-center gap-4">
                    {f.imgs.map((im) => (
                      <figure key={im.src} className="m-0">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={`/${im.src}`} alt={im.label} className="block h-64 w-auto rounded-xl border border-border" />
                        <figcaption className="mt-2 text-center text-xs uppercase tracking-wider text-muted-foreground">{im.label}</figcaption>
                      </figure>
                    ))}
                  </div>
                ) : (
                  /* eslint-disable-next-line @next/next/no-img-element */
                  <img src={`/${f.img}`} alt={f.kicker} className="mx-auto block max-h-[340px] w-auto max-w-full rounded-xl border border-border" />
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

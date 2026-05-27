"use client";

import { useEffect, useState } from "react";

const STAGES = [
  "Detecting PPE with YOLOv8",
  "Generating GradCAM + SHAP",
  "Retrieving OSHA regulations",
  "Writing the incident report",
  "Finalizing report + PDF",
];

export function Analyzing() {
  const [stage, setStage] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setStage((s) => Math.min(s + 1, STAGES.length - 1)), 9000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="mt-4 rounded-xl border border-border bg-card p-5">
      <ul className="space-y-3">
        {STAGES.map((label, i) => {
          const state = i < stage ? "done" : i === stage ? "active" : "pending";
          return (
            <li key={label} className="flex items-center gap-3 text-sm">
              <span
                className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full border text-[10px] font-medium ${
                  state === "done"
                    ? "border-primary bg-primary text-primary-foreground"
                    : state === "active"
                    ? "border-primary text-primary"
                    : "border-border text-muted-foreground"
                }`}
              >
                {state === "done" ? "\u2713" : state === "active" ? <span className="h-2 w-2 animate-pulse rounded-full bg-primary" /> : i + 1}
              </span>
              <span className={state === "active" ? "font-medium text-foreground" : "text-muted-foreground"}>{label}</span>
              {state === "active" && <span className="ml-auto text-xs text-muted-foreground">working&hellip;</span>}
            </li>
          );
        })}
      </ul>
      <p className="mt-4 text-xs text-muted-foreground">Full analysis runs ~45&ndash;60s on free-tier CPU. First call may cold-start.</p>
    </div>
  );
}

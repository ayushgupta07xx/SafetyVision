"use client";

import { useEffect, useRef, useState } from "react";

const VB_W = 518;
const VB_H = 880;

const BOXES = [
  { x: 118, y: 150, w: 90, h: 120, label: "NO-Hardhat 0.43", color: "#ef4444" },
  { x: 86, y: 266, w: 166, h: 242, label: "Safety Vest 0.78", color: "#22c55e" },
  { x: 284, y: 118, w: 98, h: 142, label: "Hardhat 0.85", color: "#22c55e" },
  { x: 252, y: 244, w: 182, h: 268, label: "Safety Vest 0.87", color: "#22c55e" },
];

export function HeroDetection() {
  const ref = useRef<HTMLDivElement>(null);
  const [started, setStarted] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setStarted(true);
          io.disconnect();
        }
      },
      { threshold: 0.4 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const fs = 18;
  return (
    <div ref={ref} className={`hero-detection relative w-fit ${started ? "started" : ""}`}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src="/hero-clean.jpg" alt="Worksite PPE detection" className="block max-h-[460px] w-auto rounded-xl" />
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
        {BOXES.map((b, i) => {
          const perim = 2 * (b.w + b.h);
          const boxDelay = 0.1 + i * 0.4;
          const labelDelay = boxDelay + 0.4;
          const lw = b.label.length * fs * 0.6 + 12;
          return (
            <g key={i}>
              <rect x={b.x} y={b.y} width={b.w} height={b.h} rx="6" fill="none" stroke={b.color} strokeWidth="4" className="hero-box" style={{ "--perim": perim, "--bd": `${boxDelay}s` } as React.CSSProperties} />
              <g className="hero-label" style={{ "--ld": `${labelDelay}s` } as React.CSSProperties}>
                <rect x={b.x} y={b.y - fs - 8} width={lw} height={fs + 6} rx="3" fill={b.color} />
                <text x={b.x + 6} y={b.y - 10} fontSize={fs} fontFamily="ui-monospace, monospace" fontWeight="600" fill="#ffffff">{b.label}</text>
              </g>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

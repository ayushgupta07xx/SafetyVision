"use client";
import { ArrowUp, Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

export function ThemeToggle({ hint = false }: { hint?: boolean }) {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [showHint, setShowHint] = useState(false);

  useEffect(() => {
    setMounted(true);
    if (!hint) return;
    const show = setTimeout(() => setShowHint(true), 50);
    const hide = setTimeout(() => setShowHint(false), 3050);
    return () => {
      clearTimeout(show);
      clearTimeout(hide);
    };
  }, [hint]);

  const isDark = resolvedTheme === "dark";

  return (
    <div className="relative">
      <button
        type="button"
        aria-label="Toggle theme"
        onClick={() => {
          setShowHint(false);
          setTheme(isDark ? "light" : "dark");
        }}
        className="inline-flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
      >
        {mounted && isDark ? <Sun className="h-[18px] w-[18px]" /> : <Moon className="h-[18px] w-[18px]" />}
      </button>

      {hint && mounted && (
        <div
          className={`pointer-events-none absolute left-1/2 top-full z-50 mt-1 flex -translate-x-1/2 flex-col items-center transition-all duration-300 ${
            showHint ? "translate-y-0 opacity-100" : "-translate-y-1 opacity-0"
          }`}
        >
          <ArrowUp className="h-3.5 w-3.5 text-foreground" />
          <div className="mt-1.5 whitespace-nowrap rounded-md bg-foreground px-3 py-1.5 text-xs font-medium text-background shadow-lg">
            Switch to {isDark ? "light" : "dark"} mode
          </div>
        </div>
      )}
    </div>
  );
}

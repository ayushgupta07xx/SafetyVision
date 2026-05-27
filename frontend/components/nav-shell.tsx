"use client";

import { useEffect, useState, type ReactNode } from "react";

export function NavShell({ children }: { children: ReactNode }) {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  return (
    <header
      className={`sticky top-0 z-40 bg-background transition-[border-color,box-shadow] duration-200 ${
        scrolled ? "border-b border-border shadow-sm" : "border-b border-transparent"
      }`}
    >
      {children}
    </header>
  );
}

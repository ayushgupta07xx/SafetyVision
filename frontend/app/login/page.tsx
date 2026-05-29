"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { VestIcon } from "@/components/vest-icon";

function GoogleGlyph() {
  return (
    <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]" aria-hidden>
      <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.27-4.74 3.27-8.1Z" />
      <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.99.66-2.26 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84A11 11 0 0 0 12 23Z" />
      <path fill="#FBBC05" d="M5.84 14.1a6.6 6.6 0 0 1 0-4.2V7.06H2.18a11 11 0 0 0 0 9.88l3.66-2.84Z" />
      <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84C6.71 7.31 9.14 5.38 12 5.38Z" />
    </svg>
  );
}

export default function LoginPage() {
  const router = useRouter();
  const supabase = createClient();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [msg, setMsg] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleEmail() {
    setLoading(true);
    setMsg(null);
    const { data, error } =
      mode === "signin"
        ? await supabase.auth.signInWithPassword({ email, password })
        : await supabase.auth.signUp({ email, password });
    setLoading(false);
    if (error) return setMsg(error.message);
    if (mode === "signup" && !data.session)
      return setMsg("Account created — check your email to confirm, then sign in.");
    router.push("/");
    router.refresh();
  }

  async function handleGoogle() {
    await supabase.auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: `${location.origin}/auth/callback` },
    });
  }

  const inputCls =
    "w-full rounded-lg border border-border bg-background px-3.5 py-2.5 text-sm outline-none transition-colors placeholder:text-muted-foreground focus:border-primary";

  return (
    <main className="dark bg-background relative flex min-h-[calc(100vh-4rem)] items-center justify-center overflow-hidden px-6 py-16">
      <div aria-hidden className="pointer-events-none absolute -left-24 -top-20 h-[420px] w-[420px] rounded-full bg-[#ffd9c2] opacity-50 blur-3xl dark:bg-primary dark:opacity-10" />
      <div aria-hidden className="pointer-events-none absolute -right-20 top-10 h-[400px] w-[400px] rounded-full bg-[#cfe3f5] opacity-50 blur-3xl dark:bg-[#3a7ca5] dark:opacity-10" />
      <div aria-hidden className="pointer-events-none absolute bottom-0 left-1/3 h-[320px] w-[320px] rounded-full bg-[#d8efe4] opacity-40 blur-3xl dark:bg-primary dark:opacity-[0.07]" />

      <div className="rise relative w-full max-w-md rounded-2xl border border-border bg-card p-8 shadow-xl">
        <div className="flex items-center gap-2">
          <VestIcon className="text-primary" />
          <span className="text-lg font-bold tracking-tight">Safety<span className="text-primary">Vision</span></span>
        </div>

        <p className="mt-6 text-xs font-semibold uppercase tracking-[0.14em] text-primary">
          {mode === "signin" ? "Welcome back" : "Get started"}
        </p>
        <h1 className="mt-2 text-2xl font-bold tracking-tight">
          {mode === "signin" ? "Sign in to your account" : "Create your account"}
        </h1>
        <p className="mt-1.5 text-sm text-muted-foreground">
          {mode === "signin" ? "Pick up where you left off." : "Free forever. No credit card needed."}
        </p>

        <div className="mt-7 space-y-4">
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">Email</label>
            <input
              className={inputCls}
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">Password</label>
            <input
              className={inputCls}
              type="password"
              placeholder="Your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleEmail()}
            />
          </div>

          <button
            className="w-full rounded-lg bg-primary px-4 py-2.5 text-sm font-semibold text-primary-foreground transition-all hover:-translate-y-0.5 hover:opacity-90 active:translate-y-0 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50"
            onClick={handleEmail}
            disabled={loading}
          >
            {loading ? "Working..." : mode === "signin" ? "Sign in" : "Sign up"}
          </button>
        </div>

        <div className="my-6 flex items-center gap-3">
          <div className="h-px flex-1 bg-border" />
          <span className="text-xs text-muted-foreground">or</span>
          <div className="h-px flex-1 bg-border" />
        </div>

        <button
          className="flex w-full items-center justify-center gap-2.5 rounded-lg border border-border bg-background px-4 py-2.5 text-sm font-semibold transition-all hover:bg-accent active:scale-[0.98]"
          onClick={handleGoogle}
        >
          <GoogleGlyph />
          Continue with Google
        </button>

        {msg && <p className="mt-4 text-sm text-destructive">{msg}</p>}

        <button
          className="mt-6 w-full text-center text-sm text-muted-foreground transition-colors hover:text-foreground"
          onClick={() => {
            setMode(mode === "signin" ? "signup" : "signin");
            setMsg(null);
          }}
        >
          {mode === "signin" ? (
            <>Need an account? <span className="font-medium text-primary">Sign up</span></>
          ) : (
            <>Have an account? <span className="font-medium text-primary">Sign in</span></>
          )}
        </button>
      </div>
    </main>
  );
}

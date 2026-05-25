"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";

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

  return (
    <div className="mx-auto mt-24 max-w-sm space-y-3 p-6">
      <h1 className="text-2xl font-semibold">SafetyVision</h1>
      <p className="text-sm text-gray-500">
        {mode === "signin" ? "Sign in to your account" : "Create an account"}
      </p>
      <input
        className="w-full rounded border px-3 py-2"
        type="email"
        placeholder="you@example.com"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
      />
      <input
        className="w-full rounded border px-3 py-2"
        type="password"
        placeholder="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />
      <button
        className="w-full rounded bg-teal-600 px-3 py-2 font-medium text-white disabled:opacity-50"
        onClick={handleEmail}
        disabled={loading}
      >
        {mode === "signin" ? "Sign in" : "Sign up"}
      </button>
      <button className="w-full rounded border px-3 py-2" onClick={handleGoogle}>
        Continue with Google
      </button>
      <button
        className="block text-sm text-gray-500 underline"
        onClick={() => setMode(mode === "signin" ? "signup" : "signin")}
      >
        {mode === "signin" ? "Need an account? Sign up" : "Have an account? Sign in"}
      </button>
      {msg && <p className="text-sm text-red-600">{msg}</p>}
    </div>
  );
}

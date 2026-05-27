"use client";

import { useEffect, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { generateRawKey, hashKey, SV_KEY_STORAGE } from "@/lib/keys";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

type KeyRow = { key_id: string; label: string | null; revoked: boolean };

export default function AccountPage() {
  const supabase = createClient();
  const [keys, setKeys] = useState<KeyRow[]>([]);
  const [label, setLabel] = useState("");
  const [newKey, setNewKey] = useState<string | null>(null);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [showRevoked, setShowRevoked] = useState(false);

  async function load() {
    const { data, error } = await supabase.from("api_keys").select("key_id,label,revoked");
    if (error) return setMsg(error.message);
    setKeys((data as KeyRow[]) ?? []);
  }

  useEffect(() => {
    load();
    if (typeof window !== "undefined") setActiveKey(localStorage.getItem(SV_KEY_STORAGE));
  }, []);

  async function mint() {
    setLoading(true);
    setMsg(null);
    setNewKey(null);
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      setLoading(false);
      return setMsg("Not signed in.");
    }
    const raw = generateRawKey();
    const key_hash = await hashKey(raw);
    const { error } = await supabase.from("api_keys").insert({ user_id: user.id, key_hash, label: label || null });
    setLoading(false);
    if (error) return setMsg(error.message);
    setNewKey(raw);
    localStorage.setItem(SV_KEY_STORAGE, raw);
    setActiveKey(raw);
    setLabel("");
    load();
  }

  async function revoke(key_id: string) {
    const { error } = await supabase.from("api_keys").update({ revoked: true }).eq("key_id", key_id);
    if (error) return setMsg(error.message);
    load();
  }

  const active = keys.filter((k) => !k.revoked);
  const revoked = keys.filter((k) => k.revoked);

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">Account</p>
      <h1 className="mt-1 text-3xl font-bold tracking-tight">Account &amp; API keys</h1>

      <div className="mt-8 rounded-xl border border-border bg-card p-6">
        <h2 className="text-sm font-semibold">Mint a new API key</h2>
        <div className="mt-4 flex gap-2">
          <Input placeholder="label (optional)" value={label} onChange={(e) => setLabel(e.target.value)} />
          <Button onClick={mint} disabled={loading}>{loading ? "Minting…" : "Mint key"}</Button>
        </div>
        {newKey && (
          <div className="mt-4 rounded-lg border border-primary bg-card p-4 text-sm">
            <p className="font-medium text-primary">Save this key now — it won&apos;t be shown again:</p>
            <code className="mt-2 block break-all font-mono">{newKey}</code>
            <p className="mt-2 text-muted-foreground">Stored in this browser for the Upload page; also use it as your SDK/API key.</p>
          </div>
        )}
        {activeKey && !newKey && (
          <p className="mt-4 text-sm text-muted-foreground">This browser has an active key ending …{activeKey.slice(-6)}</p>
        )}
        {msg && <p className="mt-4 text-sm text-red-600">{msg}</p>}
      </div>

      <div className="mt-6 rounded-xl border border-border bg-card p-6">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Your keys</h2>
          <span className="font-mono text-xs text-muted-foreground">{active.length} active</span>
        </div>

        <div className="mt-4">
          {active.length === 0 ? (
            <p className="text-sm text-muted-foreground">No active keys. Mint one above to use the Upload page and API.</p>
          ) : (
            <ul className="divide-y divide-border">
              {active.map((k) => (
                <li key={k.key_id} className="flex items-center justify-between gap-4 py-3">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{k.label || "Unlabeled key"}</p>
                    <p className="font-mono text-xs text-muted-foreground">id {k.key_id.slice(0, 8)}</p>
                  </div>
                  <Button variant="outline" size="sm" onClick={() => revoke(k.key_id)}>Revoke</Button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {revoked.length > 0 && (
          <div className="mt-4 border-t border-border pt-4">
            <button type="button" onClick={() => setShowRevoked((s) => !s)} className="text-xs font-medium text-muted-foreground transition-colors hover:text-foreground">
              {showRevoked ? "Hide" : "Show"} {revoked.length} revoked key{revoked.length === 1 ? "" : "s"}
            </button>
            {showRevoked && (
              <ul className="mt-3 divide-y divide-border">
                {revoked.map((k) => (
                  <li key={k.key_id} className="flex items-center justify-between gap-4 py-2 opacity-60">
                    <div className="min-w-0">
                      <p className="truncate text-sm">{k.label || "Unlabeled key"}</p>
                      <p className="font-mono text-xs text-muted-foreground">id {k.key_id.slice(0, 8)}</p>
                    </div>
                    <span className="text-xs text-red-600">revoked</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </main>
  );
}

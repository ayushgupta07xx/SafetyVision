"use client";

import { useEffect, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { generateRawKey, hashKey, SV_KEY_STORAGE } from "@/lib/keys";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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

  async function load() {
    const { data, error } = await supabase
      .from("api_keys")
      .select("key_id,label,revoked");
    if (error) return setMsg(error.message);
    setKeys((data as KeyRow[]) ?? []);
  }

  useEffect(() => {
    load();
    if (typeof window !== "undefined")
      setActiveKey(localStorage.getItem(SV_KEY_STORAGE));
  }, []);

  async function mint() {
    setLoading(true);
    setMsg(null);
    setNewKey(null);
    const {
      data: { user },
    } = await supabase.auth.getUser();
    if (!user) {
      setLoading(false);
      return setMsg("Not signed in.");
    }
    const raw = generateRawKey();
    const key_hash = await hashKey(raw);
    const { error } = await supabase
      .from("api_keys")
      .insert({ user_id: user.id, key_hash, label: label || null });
    setLoading(false);
    if (error) return setMsg(error.message);
    setNewKey(raw);
    localStorage.setItem(SV_KEY_STORAGE, raw); // consumed by the Upload page
    setActiveKey(raw);
    setLabel("");
    load();
  }

  async function revoke(key_id: string) {
    const { error } = await supabase
      .from("api_keys")
      .update({ revoked: true })
      .eq("key_id", key_id);
    if (error) return setMsg(error.message);
    load();
  }

  return (
    <main className="mx-auto max-w-3xl space-y-6 p-6">
      <h1 className="text-2xl font-semibold">Account &amp; API keys</h1>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Mint a new API key</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex gap-2">
            <Input
              placeholder="label (optional)"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
            />
            <Button onClick={mint} disabled={loading}>
              {loading ? "Minting…" : "Mint key"}
            </Button>
          </div>
          {newKey && (
            <div className="rounded border border-teal-600 bg-teal-50 p-3 text-sm">
              <p className="font-medium text-teal-800">
                Save this key now — it won&apos;t be shown again:
              </p>
              <code className="mt-1 block break-all">{newKey}</code>
              <p className="mt-1 text-gray-600">
                Stored in this browser for the Upload page; also use it as your
                SDK/API key.
              </p>
            </div>
          )}
          {activeKey && !newKey && (
            <p className="text-sm text-gray-600">
              This browser has an active key ending …{activeKey.slice(-6)}
            </p>
          )}
          {msg && <p className="text-sm text-red-600">{msg}</p>}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Your keys</CardTitle>
        </CardHeader>
        <CardContent>
          {keys.length === 0 ? (
            <p className="text-sm text-gray-500">No keys yet.</p>
          ) : (
            <ul className="space-y-2 text-sm">
              {keys.map((k) => (
                <li
                  key={k.key_id}
                  className="flex items-center justify-between border-b pb-2"
                >
                  <span>
                    {k.label || "(no label)"}
                    {k.revoked && <span className="text-red-600"> · revoked</span>}
                  </span>
                  {!k.revoked && (
                    <Button variant="outline" size="sm" onClick={() => revoke(k.key_id)}>
                      Revoke
                    </Button>
                  )}
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </main>
  );
}

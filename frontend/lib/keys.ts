// Mirrors core/apikeys.py: raw = "sv_" + urlsafe-base64(32 random bytes);
// the STORED value is sha256-hex of the raw string. The Lambda hashes the
// presented key the same way to resolve the user, so only the hash must match.

const KEY_PREFIX = "sv_";

function toBase64Url(bytes: Uint8Array): string {
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function generateRawKey(): string {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return KEY_PREFIX + toBase64Url(bytes);
}

export async function hashKey(raw: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(raw));
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export const SV_KEY_STORAGE = "sv_api_key";

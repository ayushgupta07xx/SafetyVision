"""API-key auth for the Mode-2 Lambda API (Layer 10).

Keys are high-entropy random tokens (sv_<token>); only their SHA-256 hash is
stored in Supabase (api_keys table). The raw key is shown once at mint time and
is unrecoverable afterwards. Resolution at the Lambda uses the service-role
client (bypasses RLS) to map a presented key's hash -> user_id.

Lives in core/ (not the handler) because serving/lambda/ is not importable
(`lambda` is a Python keyword) -- the handler stays thin and calls in here.

CLI (until the Next.js account page exists):
    python -m core.apikeys mint   --user-id <auth-user-uuid> --label "my key"
    python -m core.apikeys revoke --key-id  <key-uuid>
"""
from __future__ import annotations

import argparse
import hashlib
import secrets
from typing import cast

from core import supabase_db

KEY_PREFIX = "sv_"


def hash_key(raw_key: str) -> str:
    """SHA-256 hex of the raw key. Keys are high-entropy, so a fast hash is fine."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_key() -> tuple[str, str]:
    """Return (raw_key, key_hash). Raw key is shown once; only the hash is stored."""
    raw = KEY_PREFIX + secrets.token_urlsafe(32)
    return raw, hash_key(raw)


def mint_key(user_id: str, label: str | None = None) -> str:
    """Create a key for user_id, store its hash, return the raw key (once)."""
    raw, key_hash = generate_key()
    supabase_db._client().table("api_keys").insert(
        {"user_id": user_id, "key_hash": key_hash, "label": label}
    ).execute()
    return raw


def resolve_user_id(raw_key: str | None) -> str | None:
    """Map a presented raw key -> user_id, or None if missing/invalid/revoked."""
    if not raw_key or not raw_key.startswith(KEY_PREFIX):
        return None
    resp = (
        supabase_db._client()
        .table("api_keys")
        .select("user_id")
        .eq("key_hash", hash_key(raw_key))
        .eq("revoked", False)
        .limit(1)
        .execute()
    )
    rows = cast("list[dict]", resp.data or [])
    return str(rows[0]["user_id"]) if rows else None


def revoke_key(key_id: str) -> None:
    supabase_db._client().table("api_keys").update(
        {"revoked": True}
    ).eq("key_id", key_id).execute()


def _main() -> None:
    p = argparse.ArgumentParser(description="SafetyVision API-key admin")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mint", help="create a key for a user")
    m.add_argument("--user-id", required=True, help="auth.users UUID")
    m.add_argument("--label", default=None)
    rv = sub.add_parser("revoke", help="revoke a key by key_id")
    rv.add_argument("--key-id", required=True)
    args = p.parse_args()
    if args.cmd == "mint":
        print("API key (save now -- shown once):")
        print(mint_key(args.user_id, args.label))
    elif args.cmd == "revoke":
        revoke_key(args.key_id)
        print(f"Revoked {args.key_id}")


if __name__ == "__main__":
    _main()

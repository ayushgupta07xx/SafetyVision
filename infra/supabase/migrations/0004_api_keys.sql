-- Layer 10: API-key auth for the Mode-2 Lambda API.
-- Keys are stored as SHA-256 hashes; the raw key is shown once at mint time
-- and is unrecoverable afterwards. The Lambda resolves a presented key's hash
-- to a user_id via the service-role client (bypasses RLS).
create table if not exists api_keys (
    key_id       uuid primary key default gen_random_uuid(),
    user_id      uuid not null references auth.users on delete cascade,
    key_hash     text not null unique,
    label        text,
    created_at   timestamptz not null default now(),
    last_used_at timestamptz,
    revoked      boolean not null default false
);

create index if not exists api_keys_key_hash_idx on api_keys (key_hash);
create index if not exists api_keys_user_id_idx on api_keys (user_id);

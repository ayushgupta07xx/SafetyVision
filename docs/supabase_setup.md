# Supabase Setup — Phase 2 Persistence

Supabase provides SafetyVision's **per-user violation history** (Postgres), **auth**
(email + OAuth, Mode 3), and **PDF report storage** (Layer 9). It is the canonical
user-facing history surface; the DynamoDB table (Mode 2) remains a separate, stateless
audit log.

> Free tier: 500 MB Postgres, 50k MAU auth, free forever.

## 1. Create the project

- supabase.com → **New project** → name `safetyvision`, region **South Asia (Mumbai)**.
- **Postgres Type:** Postgres (default). Not OrioleDB (alpha).
- **Security:** keep *Enable Data API* on; turn on *Enable automatic RLS* (deny-by-default
  safety net on every new public table).
- Save the generated DB password (shown once; needed later for CLI/psql).

## 2. Environment variables

Project URL from the **Connect** button; keys from Settings → **API Keys** (new
publishable/secret format). In `.env`:

```bash
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_ANON_KEY=sb_publishable_...        # publishable — browser-safe, RLS-enforced
SUPABASE_SERVICE_ROLE_KEY=sb_secret_...     # secret — backend ONLY, bypasses RLS
```

The publishable key doubles as `NEXT_PUBLIC_SUPABASE_ANON_KEY` for the Mode 3 frontend.
**Never** expose the secret key to any frontend — it bypasses row-level security.

## 3. Schema + RLS + storage

Apply the migrations in the SQL Editor **in order**:

1. `infra/supabase/migrations/0001_schema.sql` — tables `inspections`, `violations`,
   `reports` + indexes. (`users` = Supabase Auth's built-in `auth.users`.)
2. `infra/supabase/policies/0002_rls.sql` — enables RLS + `auth.uid() = user_id` policies.
3. `infra/supabase/policies/0003_storage.sql` — private `reports` storage bucket + owner-scoped policies.

### RLS model
- **Frontend** (publishable key, `authenticated` role after login): sees only its own rows.
- **Backend** (Lambda, secret/service_role key): bypasses RLS, sets `user_id` explicitly.

## 4. Backend repository — `core/supabase_db.py`

Service-role client (bypasses RLS). Backend-only; never imported by the frontend.

- `fetch_compliance_series(violation_type, days, user_id)` → `DataFrame[ds, y]` for the Prophet forecast.
- `insert_inspection(...)`, `insert_violations(rows)`, `log_inspection(user_id, items, ...)` — writes.

## 5. Forecast data source

`analytics/forecast.py` reads from SQLite by default (local dev, Mode 1 demo, CI tests)
and from Supabase on explicit opt-in:

```bash
# local SQLite (seed with: python -m analytics.seed_violations)
python -m analytics.forecast "NO-Hardhat"

# Supabase per-user (Mode 3 / Lambda /forecast pass these explicitly)
python -m analytics.forecast "NO-Hardhat" --source supabase --user-id <auth-user-uuid>
```

## 6. Seed synthetic history (demo)

The forecast needs ≥14 days of history. `violations.user_id` is a FK to `auth.users`, so:

1. Dashboard → **Authentication → Users → Add user** (email + password) → copy the **UID**.
2. `python -m analytics.seed_supabase --user-id <uid> --days 35`

Deterministic (seed=42); compliance improves across the window.

## 7. Verify connection

```bash
python -c "from dotenv import load_dotenv; load_dotenv('.env'); \
from core import supabase_db; \
print('violations:', supabase_db._client().table('violations').select('*', count='exact').limit(1).execute().count)"
```

---

**Not yet wired (lands with Layer 10 / Mode 3):** the request-flow write — *who* calls
`log_inspection` with *which* `user_id`. `user_id` originates from Lambda API-key auth
(Layer 10) and the Next.js auth session (Mode 3); the per-image caller creates one
`inspections` row + its `violations` there. The backend function is built and verified;
integration lands when those surfaces exist.

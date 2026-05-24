-- SafetyVision Phase 2 — core persistence schema
-- "users" = Supabase Auth built-in auth.users (no table created here).
create extension if not exists "pgcrypto";  -- gen_random_uuid()

create table if not exists public.inspections (
  inspection_id     uuid primary key default gen_random_uuid(),
  user_id           uuid not null references auth.users (id) on delete cascade,
  uploaded_at       timestamptz not null default now(),
  source_type       text not null check (source_type in ('image','video','api')),
  image_url         text,
  total_detections  integer not null default 0,
  total_violations  integer not null default 0
);

create table if not exists public.violations (
  violation_id      uuid primary key default gen_random_uuid(),
  inspection_id     uuid references public.inspections (inspection_id) on delete cascade,
  user_id           uuid not null references auth.users (id) on delete cascade,
  timestamp_ms      bigint not null,
  violation_type    text not null,
  risk_level        text not null,
  confidence        real not null,
  bbox_json         jsonb,
  regulation_cited  text,
  summary           text,
  pdf_report_url    text,
  source            text check (source in ('next_js','hf_spaces','api','synthetic'))
);

create table if not exists public.reports (
  report_id         uuid primary key default gen_random_uuid(),
  violation_id      uuid references public.violations (violation_id) on delete cascade,
  user_id           uuid not null references auth.users (id) on delete cascade,
  generated_at      timestamptz not null default now(),
  pdf_url           text,
  model_version     text,
  regulation_cited  text
);

-- Indexes for history pagination + forecast daily aggregation
create index if not exists idx_inspections_user_time on public.inspections (user_id, uploaded_at desc);
create index if not exists idx_violations_user_time  on public.violations (user_id, timestamp_ms desc);
create index if not exists idx_violations_user_type_time on public.violations (user_id, violation_type, timestamp_ms);
create index if not exists idx_reports_user_time     on public.reports (user_id, generated_at desc);

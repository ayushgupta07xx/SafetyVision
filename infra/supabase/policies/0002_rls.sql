-- SafetyVision Phase 2 — Row-Level Security
-- anon/auth-key path: users see ONLY their own rows.
-- Lambda uses service_role key (BYPASSES RLS), sets user_id explicitly.
alter table public.inspections enable row level security;
alter table public.violations  enable row level security;
alter table public.reports     enable row level security;

drop policy if exists "own_inspections" on public.inspections;
drop policy if exists "own_violations"  on public.violations;
drop policy if exists "own_reports"     on public.reports;

create policy "own_inspections" on public.inspections
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "own_violations" on public.violations
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "own_reports" on public.reports
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

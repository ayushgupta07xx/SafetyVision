-- SafetyVision Phase 2 — private Storage bucket for PDF reports
-- Object path convention: <user_id>/<report_id>.pdf
-- Lambda writes via service_role (bypasses RLS) + serves signed URLs.
insert into storage.buckets (id, name, public)
values ('reports', 'reports', false)
on conflict (id) do nothing;

drop policy if exists "reports_own_read"  on storage.objects;
drop policy if exists "reports_own_write" on storage.objects;

create policy "reports_own_read" on storage.objects
  for select using (bucket_id = 'reports' and (storage.foldername(name))[1] = auth.uid()::text);
create policy "reports_own_write" on storage.objects
  for insert with check (bucket_id = 'reports' and (storage.foldername(name))[1] = auth.uid()::text);

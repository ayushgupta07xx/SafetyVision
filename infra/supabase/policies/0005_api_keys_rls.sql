-- Layer 10: RLS for api_keys. The frontend (anon/publishable key) manages a
-- user's own keys via the account page. The Lambda resolves keys with the
-- service-role client, which bypasses RLS, so resolution is unaffected.
alter table api_keys enable row level security;

create policy "Users manage their own api_keys" on api_keys
    for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

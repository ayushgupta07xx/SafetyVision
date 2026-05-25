import Link from "next/link";
import { createClient } from "@/lib/supabase/server";

export default async function Nav() {
  const supabase = createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return null;

  return (
    <header className="border-b">
      <nav className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
        <Link href="/" className="font-semibold text-teal-700">
          SafetyVision
        </Link>
        <div className="flex items-center gap-4 text-sm">
          <Link href="/upload" className="hover:underline">Upload</Link>
          <Link href="/history" className="hover:underline">History</Link>
          <Link href="/forecast" className="hover:underline">Forecast</Link>
          <Link href="/account" className="hover:underline">Account</Link>
          <form action="/auth/signout" method="post">
            <button className="rounded border px-2 py-1">Sign out</button>
          </form>
        </div>
      </nav>
    </header>
  );
}

import { VestIcon } from "@/components/vest-icon";
import { Link } from "next-view-transitions";
import { createClient } from "@/lib/supabase/server";
import { ThemeToggle } from "@/components/theme-toggle";
import { NavShell } from "@/components/nav-shell";

export default async function Nav() {
  const supabase = createClient();
  const { data: { user } } = await supabase.auth.getUser();

  const link = "rounded-md px-3 py-2 font-medium text-muted-foreground transition-all duration-150 hover:text-primary active:scale-[0.97]";

  return (
    <NavShell>
      <nav className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
        <Link href="/" className="inline-flex items-center gap-2 text-lg font-bold tracking-tight"><VestIcon className="text-primary" /><span>Safety<span className="text-primary">Vision</span></span></Link>
        <div className="flex items-center gap-1 text-sm">
          {user ? (
            <>
              <Link href="/upload" className={link}>Upload</Link>
              <Link href="/history" className={link}>History</Link>
              <Link href="/forecast" className={link}>Forecast</Link>
              <Link href="/account" className={link}>Account</Link>
              <ThemeToggle />
              <form action="/auth/signout" method="post">
                <button className="rounded-md border border-border px-3 py-1.5 font-medium transition-all duration-150 hover:bg-accent active:scale-[0.97]">Sign out</button>
              </form>
            </>
          ) : (
            <>
              <ThemeToggle />
              <Link href="/login" className="rounded-md bg-primary px-4 py-2 font-semibold text-primary-foreground transition-all duration-150 hover:opacity-90 active:scale-[0.97]">Get started</Link>
            </>
          )}
        </div>
      </nav>
    </NavShell>
  );
}

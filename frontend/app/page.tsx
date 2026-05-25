import { createClient } from "@/lib/supabase/server";
import { redirect } from "next/navigation";

export default async function Home() {
  const supabase = createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/login");

  return (
    <main className="mx-auto mt-24 max-w-md space-y-4 p-6">
      <h1 className="text-2xl font-semibold">SafetyVision</h1>
      <p className="text-sm text-gray-600">Signed in as {user.email}</p>
      <form action="/auth/signout" method="post">
        <button className="rounded border px-3 py-2" type="submit">
          Sign out
        </button>
      </form>
    </main>
  );
}

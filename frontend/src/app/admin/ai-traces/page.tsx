"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

import { RequireAuth } from "@/components/auth/RequireAuth";
import { useAuth } from "@/lib/auth";
import { AiTracesPanel } from "@/components/admin/AiTracesPanel";
import { AdminShell, type AdminSectionId } from "@/components/admin/console/AdminShell";
import { AdminSearch } from "@/components/admin/console/AdminSearch";

function AiTracesRoute() {
  const router = useRouter();
  const { user } = useAuth();
  const [mobileNav, setMobileNav] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [nonce, setNonce] = useState(0);

  const navigate = (id: AdminSectionId) => {
    router.push(id === "overview" ? "/admin" : `/admin?section=${id}`);
  };

  return (
    <>
      <AdminShell
        active="traces"
        onNavigate={navigate}
        collapsed={false}
        onToggleCollapsed={() => {
          /* The standalone route keeps the full sidebar. */
        }}
        onOpenSearch={() => setSearchOpen(true)}
        onRefresh={() => setNonce((value) => value + 1)}
        refreshing={false}
        userEmail={user?.email ?? ""}
        userName={user?.username || "Admin"}
        mobileNavOpen={mobileNav}
        onMobileNav={setMobileNav}
      >
        <AiTracesPanel key={nonce} syncUrl />
      </AdminShell>

      <AdminSearch
        open={searchOpen}
        onClose={() => setSearchOpen(false)}
        onNavigate={navigate}
      />
    </>
  );
}

export default function AdminAiTracesPage() {
  return (
    <RequireAuth requireAdmin>
      <AiTracesRoute />
    </RequireAuth>
  );
}

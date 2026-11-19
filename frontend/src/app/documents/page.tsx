"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

/**
 * /documents is kept as a bookmark-safe entry point. It opens the unified
 * Workspace shell with the Documents view selected instead of rendering a
 * second copy of the library.
 */
export default function DocumentsRedirectPage() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/dashboard?view=documents");
  }, [router]);

  return (
    <div
      className="flex min-h-screen items-center justify-center bg-[#f5f7fa] text-sm text-[#6b7487]"
      role="status"
      aria-live="polite"
    >
      Opening your documents…
    </div>
  );
}

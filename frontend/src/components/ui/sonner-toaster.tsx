"use client";

import { Toaster } from "sonner";

export function SonnerToaster() {
  return (
    <Toaster
      richColors
      closeButton
      position="top-right"
      toastOptions={{
        classNames: {
          toast: "rounded-lg border border-border bg-popover shadow-sm",
          title: "font-medium",
          description: "text-muted-foreground",
        },
      }}
    />
  );
}


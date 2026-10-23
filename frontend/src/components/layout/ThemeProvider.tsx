"use client";

import { ThemeProvider as NextThemesProvider } from "next-themes";
import { type ThemeProviderProps } from "next-themes";

/**
 * next-themes injects an inline <script> for FOUC prevention. React 19 / Next 16
 * warns that scripts inside components are not executed on the client — a false
 * positive: the script already ran during SSR. Filter that specific noise in dev.
 * @see https://github.com/pacocoursey/next-themes/issues/385
 */
if (
  typeof window !== "undefined" &&
  process.env.NODE_ENV === "development" &&
  !(globalThis as { __genovaThemeScriptWarnFiltered?: boolean })
    .__genovaThemeScriptWarnFiltered
) {
  (globalThis as { __genovaThemeScriptWarnFiltered?: boolean }).__genovaThemeScriptWarnFiltered =
    true;
  const originalError = console.error;
  console.error = (...args: unknown[]) => {
    const joined = args
      .map((arg) => (typeof arg === "string" ? arg : ""))
      .join(" ");
    if (
      joined.includes("Encountered a script tag while rendering React component")
    ) {
      return;
    }
    originalError.apply(console, args);
  };
}

export function ThemeProvider({ children, ...props }: ThemeProviderProps) {
  return <NextThemesProvider {...props}>{children}</NextThemesProvider>;
}

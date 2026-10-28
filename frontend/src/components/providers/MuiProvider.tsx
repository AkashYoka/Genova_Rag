"use client";

import { ThemeProvider, createTheme, CssBaseline } from "@mui/material";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useTheme } from "next-themes";

/** Material UI theme bridged to Genova tokens — use selectively. */
export default function MuiProvider({ children }: { children: ReactNode }) {
  const { resolvedTheme, theme } = useTheme();
  // CssBaseline injects emotion <style> tags; only after mount so SSR HTML matches.
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  const active = resolvedTheme ?? theme ?? "light";
  const mode = active === "light" ? "light" : "dark";

  const muiTheme = useMemo(
    () =>
      createTheme({
        palette: {
          mode,
          primary: {
            main: mode === "light" ? "#0F7C86" : "#4AA8B5",
          },
          secondary: {
            main: mode === "light" ? "#6B8F3C" : "#8B95A5",
          },
          background: {
            default: "transparent",
            paper: mode === "light" ? "#F3F6FA" : "#151A22",
          },
        },
        typography: {
          fontFamily: "var(--font-sans), system-ui, sans-serif",
          button: { textTransform: "none", fontWeight: 600 },
        },
        shape: { borderRadius: 12 },
      }),
    [mode],
  );

  return (
    <ThemeProvider theme={muiTheme}>
      {mounted ? <CssBaseline enableColorScheme /> : null}
      {children}
    </ThemeProvider>
  );
}

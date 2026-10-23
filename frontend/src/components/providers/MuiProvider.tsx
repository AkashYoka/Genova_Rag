"use client";

import { ThemeProvider, createTheme, CssBaseline } from "@mui/material";
import { useMemo, type ReactNode } from "react";
import { useTheme } from "next-themes";

/** Material UI theme bridged to Genova tokens — use selectively. */
export default function MuiProvider({ children }: { children: ReactNode }) {
  const { resolvedTheme, theme } = useTheme();
  const active = resolvedTheme ?? theme ?? "light";
  const mode = active === "light" ? "light" : "dark";

  const muiTheme = useMemo(
    () =>
      createTheme({
        palette: {
          mode,
          primary: {
            main: mode === "light" ? "#0F7C86" : "#3DD6E0",
          },
          secondary: {
            main: mode === "light" ? "#6B8F3C" : "#B6F05C",
          },
          background: {
            default: "transparent",
            paper: mode === "light" ? "#F3F6FA" : "#12181F",
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
      <CssBaseline enableColorScheme />
      {children}
    </ThemeProvider>
  );
}

import type { Metadata } from "next";
import { Syne, Figtree, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/lib/auth";
import { TooltipProvider } from "@/components/ui/tooltip";
import I18nProvider from "@/components/providers/I18nProvider";
import { ThemeProvider } from "@/components/layout/ThemeProvider";
import MuiProvider from "@/components/providers/MuiProvider";
import { SonnerToaster } from "@/components/ui/sonner-toaster";

const display = Syne({
  variable: "--font-display",
  subsets: ["latin"],
  display: "swap",
});

const sans = Figtree({
  variable: "--font-sans",
  subsets: ["latin"],
  display: "swap",
});

const mono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Genova — Yoka Document Intelligence",
  description:
    "Upload complex PDFs and chat with an AI agent that retrieves precise insights and cites sources using Retrieval-Augmented Generation.",
  keywords: ["RAG", "Document AI", "PDF Analysis", "Genova", "Yoka"],
  icons: {
    icon: [
      { url: "/favicon.png", sizes: "32x32", type: "image/png" },
      { url: "/icon.svg", type: "image/svg+xml" },
      { url: "/logo.png", sizes: "512x512", type: "image/png" },
    ],
    apple: [{ url: "/apple-icon.png", sizes: "180x180", type: "image/png" }],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${display.variable} ${sans.variable} ${mono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="min-h-full flex flex-col bg-background text-foreground font-sans">
        <ThemeProvider
          attribute="class"
          defaultTheme="light"
          enableSystem={false}
          disableTransitionOnChange
          themes={["light", "dark", "ocean", "forest", "sunset"]}
        >
          <MuiProvider>
            <AuthProvider>
              <I18nProvider>
                <TooltipProvider>
                  {children}
                  <SonnerToaster />
                </TooltipProvider>
              </I18nProvider>
            </AuthProvider>
          </MuiProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}

"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import {
  FileText,
  MessageSquare,
  Brain,
  Shield,
  Zap,
  Search,
  ArrowRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import Chip from "@mui/material/Chip";
import Link from "next/link";
import Logo from "@/components/layout/Logo";
import { Aurora } from "@/components/bits/Aurora";
import { SpotlightCard } from "@/components/bits/SpotlightCard";
import { FadeIn } from "@/components/bits/FadeIn";

const FEATURES = [
  {
    icon: FileText,
    title: "Multi-Format Upload",
    desc: "PDF, DOCX, TXT, and Markdown with smart chunking",
  },
  {
    icon: Search,
    title: "Hybrid Retrieval",
    desc: "Vector + BM25 fused with cross-encoder reranking",
  },
  {
    icon: MessageSquare,
    title: "Streaming Chat",
    desc: "Real-time answers with page-level citations",
  },
  {
    icon: Zap,
    title: "Instant Insights",
    desc: "Extract facts, summaries, and grounded data points",
  },
  {
    icon: Shield,
    title: "Isolated Workspaces",
    desc: "Per-user collections and JWT-secured sessions",
  },
  {
    icon: Brain,
    title: "Agentic RAG",
    desc: "Prompt assembly, streaming tokens, source fidelity",
  },
];

export default function HomePage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [mounted, setMounted] = useState(false);

  const docsUrl =
    (process.env.NEXT_PUBLIC_API_URL || "http://localhost:7860") + "/docs";

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (mounted && !loading && user) {
      router.replace("/dashboard");
    }
  }, [user, loading, router, mounted]);

  if (!mounted || loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-pulse-glow h-12 w-12 rounded-full bg-primary/25" />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden">
      <Aurora />

      <header className="relative z-20 flex items-center justify-between px-6 py-5 max-w-6xl mx-auto w-full">
        <div className="flex items-center gap-3">
          <Logo size={34} />
          <div className="leading-tight">
            <p className="font-display text-lg font-semibold tracking-tight">
              Genova
            </p>
            <p className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
              by Yoka
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Link href="/login">
            <Button variant="ghost" className="rounded-full px-4">
              Sign in
            </Button>
          </Link>
          <Link href="/register">
            <Button className="rounded-full px-5 shadow-lg shadow-primary/20">
              Get started
            </Button>
          </Link>
        </div>
      </header>

      <section className="relative z-10 flex-1 flex flex-col items-center justify-center px-6 pb-20 pt-8">
        <FadeIn className="text-center max-w-4xl mx-auto space-y-8">
          <Chip
            label="Enterprise agentic RAG"
            color="primary"
            variant="outlined"
            sx={{
              borderRadius: 999,
              fontWeight: 700,
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              fontSize: 11,
              bgcolor: "transparent",
            }}
          />

          <h1 className="font-display text-5xl sm:text-6xl lg:text-7xl font-bold tracking-tight leading-[0.98]">
            Documents that{" "}
            <span className="bg-gradient-to-r from-primary via-primary to-[oklch(0.86_0.16_125)] bg-clip-text text-transparent">
              answer back
            </span>
          </h1>

          <p className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            Genova_Rag turns financial reports, contracts, and research into a
            cited conversation — hybrid retrieval, streaming answers, and
            page-accurate sources in one workspace.
          </p>

          <div className="flex flex-wrap gap-3 justify-center pt-2">
            <Link href="/register">
              <Button
                size="lg"
                className="h-12 rounded-full px-8 text-base shadow-xl shadow-primary/25 transition-transform hover:-translate-y-0.5 active:scale-[0.98]"
              >
                Open workspace
                <ArrowRight className="ml-1.5 h-4 w-4" />
              </Button>
            </Link>
            <Link href="/login">
              <Button
                size="lg"
                variant="outline"
                className="h-12 rounded-full px-8 text-base border-border/70 bg-background/40 backdrop-blur"
              >
                Sign in
              </Button>
            </Link>
          </div>

          <p className="text-xs text-muted-foreground/70 tracking-wide">
            Upload · Embed · Retrieve · Chat
          </p>
        </FadeIn>

        <div className="relative z-10 mt-16 w-full max-w-5xl mx-auto grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {FEATURES.map(({ icon: Icon, title, desc }, i) => (
            <FadeIn key={title} delay={0.08 + i * 0.05}>
              <SpotlightCard className="h-full p-5">
                <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-xl bg-primary/12 text-primary">
                  <Icon className="h-5 w-5" />
                </div>
                <h3 className="font-display text-base font-semibold tracking-tight">
                  {title}
                </h3>
                <p className="mt-1.5 text-sm text-muted-foreground leading-relaxed">
                  {desc}
                </p>
              </SpotlightCard>
            </FadeIn>
          ))}
        </div>
      </section>

      <footer className="relative z-10 border-t border-border/40 bg-card/30 backdrop-blur-md">
        <div className="max-w-6xl mx-auto px-6 py-12 grid grid-cols-1 md:grid-cols-12 gap-8">
          <div className="md:col-span-5 space-y-4">
            <div className="flex items-center gap-2.5">
              <Logo size={28} />
              <span className="font-display font-semibold text-base tracking-tight">
                Genova_Rag
              </span>
            </div>
            <p className="text-sm text-muted-foreground max-w-sm leading-relaxed">
              Yoka document intelligence — retrieval-augmented chat with
              verifiable citations.
            </p>
          </div>

          <div className="md:col-span-2 space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-[0.16em] text-foreground/80">
              Product
            </h4>
            <ul className="space-y-2.5 text-sm text-muted-foreground">
              <li>
                <Link href="/register" className="hover:text-foreground transition-colors">
                  Get started
                </Link>
              </li>
              <li>
                <Link href="/login" className="hover:text-foreground transition-colors">
                  Sign in
                </Link>
              </li>
              <li>
                <Link href="/dashboard" className="hover:text-foreground transition-colors">
                  Workspace
                </Link>
              </li>
            </ul>
          </div>

          <div className="md:col-span-3 space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-[0.16em] text-foreground/80">
              Resources
            </h4>
            <ul className="space-y-2.5 text-sm text-muted-foreground">
              <li>
                <a
                  href={docsUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-foreground transition-colors"
                >
                  API documentation
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/AkashYoka/Genova_Rag"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-foreground transition-colors"
                >
                  Repository
                </a>
              </li>
            </ul>
          </div>

          <div className="md:col-span-2 space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-[0.16em] text-foreground/80">
              Legal
            </h4>
            <ul className="space-y-2.5 text-sm text-muted-foreground">
              <li>
                <Link href="/privacy" className="hover:text-foreground transition-colors">
                  Privacy
                </Link>
              </li>
              <li>
                <Link href="/terms" className="hover:text-foreground transition-colors">
                  Terms
                </Link>
              </li>
            </ul>
          </div>
        </div>

        <div className="max-w-6xl mx-auto px-6 py-5 border-t border-border/20 flex flex-col sm:flex-row items-center justify-between gap-3">
          <p className="text-xs text-muted-foreground/70">
            © {new Date().getFullYear()} Yoka · Genova_Rag
          </p>
          <p className="text-xs text-muted-foreground/60">
            FastAPI · Next.js · ChromaDB · Hybrid RAG
          </p>
        </div>
      </footer>

    </div>
  );
}

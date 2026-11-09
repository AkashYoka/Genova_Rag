"use client";

import { useEffect, useSyncExternalStore } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import {
  FileText,
  MessageSquare,
  Network,
  StickyNote,
  Shield,
  Search,
  ArrowRight,
  Loader2,
} from "lucide-react";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import Link from "next/link";
import Logo from "@/components/layout/Logo";

const FEATURES = [
  {
    icon: FileText,
    title: "Multi-format upload",
    desc: "PDF, DOCX, TXT and Markdown, chunked with page boundaries kept.",
  },
  {
    icon: Search,
    title: "Hybrid retrieval",
    desc: "Vector search and BM25 fused, then reranked by a cross-encoder.",
  },
  {
    icon: MessageSquare,
    title: "Streaming cited chat",
    desc: "Answers arrive token by token, each claim tagged with its page.",
  },
  {
    icon: Network,
    title: "Knowledge graph",
    desc: "See entities and the links between them across your documents.",
  },
  {
    icon: StickyNote,
    title: "Sticky notes",
    desc: "Pin findings next to the source so context stays with the note.",
  },
  {
    icon: Shield,
    title: "Isolated workspaces",
    desc: "Per-user collections and JWT-secured sessions.",
  },
];

function DeskPreview() {
  return (
    <div
      className="genova-panel overflow-hidden"
      role="img"
      aria-label="Preview of a Genova workspace: a cited answer next to a PDF page strip"
    >
      <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
        <div className="flex items-center gap-2 min-w-0">
          <FileText className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          <span className="truncate font-mono text-xs text-muted-foreground">
            Q3-annual-report.pdf
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <span className="h-1.5 w-1.5 rounded-full bg-signal" />
          <span className="font-mono">Indexed · 64 pages</span>
        </div>
      </div>

      <div className="grid sm:grid-cols-[1fr_8.5rem]">
        <div className="space-y-4 p-5">
          <div className="ml-auto max-w-[85%] rounded-lg bg-[var(--gd-user-bubble)] px-3.5 py-2.5 text-sm">
            What drove operating margin in the second half?
          </div>

          <div className="space-y-3 text-sm leading-relaxed">
            <p>
              Operating margin rose to 18.4%, mostly from lower freight costs
              and a shift toward the services segment
              <span className="cite-keycap ml-1.5 align-middle">p.12</span>
              <span className="cite-keycap ml-1 align-middle">p.31</span>.
            </p>
            <p className="text-muted-foreground">
              Management expects the freight benefit to taper next year, with
              capital spend held flat
              <span className="cite-keycap ml-1.5 align-middle">p.47</span>.
            </p>
          </div>

          <div className="flex items-center gap-2 rounded-lg border border-border px-3 py-2 text-xs text-muted-foreground">
            <span>Ask about this document</span>
            <span className="streaming-caret ml-auto" aria-hidden="true" />
          </div>
        </div>

        <div className="hidden border-l border-border bg-[var(--gd-raised)] p-3 sm:block">
          <p className="mb-2 font-mono text-[11px] text-muted-foreground">
            Cited pages
          </p>
          <div className="space-y-2.5">
            {[12, 31, 47].map((page) => (
              <div key={page} className="space-y-1">
                <div className="space-y-1.5 rounded-md border border-border bg-card p-2">
                  <div className="h-1.5 w-2/3 rounded-sm bg-foreground/20" />
                  <div className="h-1 w-full rounded-sm bg-foreground/10" />
                  <div className="h-1 w-full rounded-sm bg-foreground/10" />
                  <div
                    className={cn(
                      "h-1 w-4/5 rounded-sm",
                      page === 12 ? "bg-primary/40" : "bg-foreground/10"
                    )}
                  />
                  <div className="h-1 w-3/5 rounded-sm bg-foreground/10" />
                </div>
                <span className="font-mono text-[11px] text-muted-foreground">
                  p.{page}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

const subscribe = () => () => {};
const getSnapshot = () => true;
const getServerSnapshot = () => false;

export default function HomePage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const mounted = useSyncExternalStore(
    subscribe,
    getSnapshot,
    getServerSnapshot,
  );

  const docsUrl =
    (process.env.NEXT_PUBLIC_API_URL || "http://localhost:7860") + "/docs";

  useEffect(() => {
    if (mounted && !loading && user) {
      router.replace("/dashboard");
    }
  }, [user, loading, router, mounted]);

  if (!mounted || loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-5">
        <Link href="/" className="flex items-center gap-2.5">
          <Logo size={30} />
          <span className="font-display text-lg font-semibold tracking-tight">
            Genova
          </span>
        </Link>
        <div className="flex items-center gap-1.5">
          <Link
            href="/login"
            className={buttonVariants({
              variant: "ghost",
              className: "rounded-lg px-3",
            })}
          >
            Sign in
          </Link>
          <Link
            href="/register"
            className={buttonVariants({ className: "rounded-lg px-4" })}
          >
            Get started
          </Link>
        </div>
      </header>

      <main className="flex-1">
        <section className="mx-auto grid w-full max-w-6xl items-center gap-12 px-6 pb-20 pt-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)] lg:gap-16 lg:pt-16">
          <div className="space-y-6">
            <h1 className="font-display text-4xl font-semibold leading-[1.05] tracking-tight sm:text-5xl lg:text-[3.5rem]">
              Every answer points back to a page.
            </h1>
            <p className="max-w-lg text-base leading-relaxed text-muted-foreground sm:text-lg">
              Upload reports, contracts and research. Ask questions in plain
              language and get streamed answers with page citations you can
              open and check, backed by a knowledge graph of your documents.
            </p>
            <div className="flex flex-wrap gap-3 pt-1">
              <Link
                href="/register"
                className={buttonVariants({
                  size: "lg",
                  className: "h-11 rounded-lg px-6 text-base",
                })}
              >
                Open workspace
                <ArrowRight className="ml-1.5 h-4 w-4" />
              </Link>
              <Link
                href="/login"
                className={buttonVariants({
                  size: "lg",
                  variant: "outline",
                  className: "h-11 rounded-lg px-6 text-base",
                })}
              >
                Sign in
              </Link>
            </div>
          </div>

          <DeskPreview />
        </section>

        <section className="border-t border-border">
          <div className="mx-auto w-full max-w-6xl px-6 py-16">
            <div className="max-w-xl space-y-2">
              <h2 className="font-display text-2xl font-semibold tracking-tight sm:text-3xl">
                What is in the workspace
              </h2>
              <p className="text-sm leading-relaxed text-muted-foreground sm:text-base">
                Six working parts, each one shipped and in use today.
              </p>
            </div>

            <ul className="mt-10 grid gap-x-10 gap-y-8 sm:grid-cols-2 lg:grid-cols-3">
              {FEATURES.map(({ icon: Icon, title, desc }) => (
                <li key={title} className="flex gap-3.5">
                  <Icon
                    className="mt-0.5 h-5 w-5 shrink-0 text-primary"
                    strokeWidth={1.5}
                  />
                  <div>
                    <h3 className="font-display text-base font-semibold tracking-tight">
                      {title}
                    </h3>
                    <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                      {desc}
                    </p>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </section>
      </main>

      <footer className="border-t border-border">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-4 px-6 py-6 text-sm text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2.5">
            <Logo size={22} />
            <span>
              © {new Date().getFullYear()} Yoka · Genova
            </span>
          </div>
          <nav className="flex flex-wrap items-center gap-x-5 gap-y-2">
            <Link
              href="/register"
              className="transition-colors hover:text-foreground"
            >
              Get started
            </Link>
            <Link
              href="/login"
              className="transition-colors hover:text-foreground"
            >
              Sign in
            </Link>
            <a
              href={docsUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="transition-colors hover:text-foreground"
            >
              API documentation
            </a>
            <a
              href="https://github.com/AkashYoka/Genova_Rag"
              target="_blank"
              rel="noopener noreferrer"
              className="transition-colors hover:text-foreground"
            >
              Repository
            </a>
          </nav>
        </div>
      </footer>
    </div>
  );
}

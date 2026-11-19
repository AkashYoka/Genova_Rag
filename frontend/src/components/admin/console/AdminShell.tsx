"use client";

import type { CSSProperties, ReactNode } from "react";
import {
  ArrowLeft,
  ChevronRight,
  FileText,
  Gauge,
  Menu,
  MessageSquareText,
  PanelLeftClose,
  PanelLeftOpen,
  Plug,
  RefreshCw,
  Route,
  Search,
  ServerCog,
  SlidersHorizontal,
  Sparkles,
  Users,
  X,
} from "lucide-react";

import { AdminButton, IconButton } from "./ConsoleUI";
import "./console.css";

export type AdminSectionId =
  | "overview"
  | "users"
  | "usage"
  | "documents"
  | "feedback"
  | "traces"
  | "styles"
  | "connections"
  | "export"
  | "status";

export type AdminSection = {
  id: AdminSectionId;
  label: string;
  group: string;
  icon: ReactNode;
  title: string;
  description: string;
  keywords: string;
};

export const ADMIN_SECTIONS: AdminSection[] = [
  {
    id: "overview",
    label: "Overview",
    group: "",
    icon: <Gauge />,
    title: "Overview",
    description: "A live picture of your workspace, your documents, and the AI behind every answer.",
    keywords: "home dashboard summary health",
  },
  {
    id: "users",
    label: "Users",
    group: "People & usage",
    icon: <Users />,
    title: "Users",
    description: "Everyone with an account, what they can access, and when they joined.",
    keywords: "people accounts members admins roles",
  },
  {
    id: "usage",
    label: "AI usage",
    group: "People & usage",
    icon: <Sparkles />,
    title: "AI usage",
    description: "How much AI capacity the platform used, and what that usage is estimated to cost.",
    keywords: "tokens cost spend billing models providers requests",
  },
  {
    id: "feedback",
    label: "Feedback",
    group: "People & usage",
    icon: <MessageSquareText />,
    title: "Feedback",
    description: "What people told us about individual answers, so poor answers can be found and fixed.",
    keywords: "ratings likes dislikes satisfaction comments",
  },
  {
    id: "documents",
    label: "Documents",
    group: "Content & answers",
    icon: <FileText />,
    title: "Documents",
    description: "Every file Genova has indexed, plus anything that is still processing or needs attention.",
    keywords: "files pdfs uploads indexing rag chunks failed processed",
  },
  {
    id: "traces",
    label: "Answer traces",
    group: "Content & answers",
    icon: <Route />,
    title: "Answer traces",
    description:
      "A step-by-step record of how each answer was produced — which document passages were used and which model replied.",
    keywords: "traces debug retrieval pipeline latency errors question",
  },
  {
    id: "styles",
    label: "Answer styles",
    group: "Content & answers",
    icon: <SlidersHorizontal />,
    title: "Answer styles",
    description:
      "Control the wording Genova uses for each document specialty. People never pick a specialty — Genova detects it.",
    keywords: "prompts domains templates instructions medical legal payment",
  },
  {
    id: "connections",
    label: "Connections",
    group: "Platform",
    icon: <Plug />,
    title: "Connections",
    description: "Linked Google Drive folders, shared workspaces, and API keys issued to accounts.",
    keywords: "integrations drive sync workspaces api keys oauth",
  },
  {
    id: "export",
    label: "Data export",
    group: "Platform",
    icon: <ServerCog />,
    title: "Data export",
    description: "Download a copy of the platform database for support, migration, or engineering work.",
    keywords: "backup database download json sql redacted",
  },
  {
    id: "status",
    label: "System status",
    group: "Platform",
    icon: <ServerCog />,
    title: "System status",
    description: "Live checks on the services Genova depends on, explained in plain language.",
    keywords: "health database redis celery chromadb ollama uptime services",
  },
];

const GROUP_ORDER = ["", "People & usage", "Content & answers", "Platform"];

export function AdminShell({
  active,
  onNavigate,
  children,
  collapsed,
  onToggleCollapsed,
  onOpenSearch,
  onRefresh,
  refreshing,
  userEmail,
  userName,
  mobileNavOpen,
  onMobileNav,
}: {
  active: AdminSectionId;
  onNavigate: (id: AdminSectionId) => void;
  children: ReactNode;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onOpenSearch: () => void;
  onRefresh: () => void;
  refreshing: boolean;
  userEmail: string;
  userName: string;
  mobileNavOpen: boolean;
  onMobileNav: (open: boolean) => void;
}) {
  const current = ADMIN_SECTIONS.find((s) => s.id === active) ?? ADMIN_SECTIONS[0];
  const initials = (userName || userEmail || "A").slice(0, 2).toUpperCase();

  const navList = (
    <nav className="adm-nav" aria-label="Admin sections">
      {GROUP_ORDER.map((group) => {
        const items = ADMIN_SECTIONS.filter((s) => s.group === group);
        if (!items.length) return null;
        return (
          <div key={group || "primary"} style={{ display: "contents" }}>
            {group ? <div className="adm-nav-group">{group}</div> : null}
            {items.map((item) => (
              <button
                key={item.id}
                type="button"
                className="adm-nav-item"
                aria-current={active === item.id ? "page" : undefined}
                title={collapsed ? item.label : undefined}
                onClick={() => {
                  onNavigate(item.id);
                  onMobileNav(false);
                }}
              >
                {item.icon}
                <span>{item.label}</span>
              </button>
            ))}
          </div>
        );
      })}
    </nav>
  );

  const footer = (
    <div className="adm-side-foot">
      <div className="adm-userchip">
        <span className="adm-avatar" data-tone="accent">
          {initials}
        </span>
        <span className="adm-userchip-text">
          <span className="adm-userchip-name">{userName}</span>
          <span className="adm-userchip-meta">{userEmail}</span>
        </span>
      </div>
      <button
        type="button"
        className="adm-nav-item"
        onClick={() => {
          window.location.href = "/dashboard?view=documents";
        }}
      >
        <ArrowLeft />
        <span>Back to workspace</span>
      </button>
    </div>
  );

  return (
    <div
      className="adm"
      data-collapsed={collapsed ? "true" : "false"}
      style={{ "--adm-side-w": collapsed ? "72px" : "252px" } as CSSProperties}
    >
      <aside className="adm-side">
        <div className="adm-brand">
          <span className="adm-brand-mark" aria-hidden>
            G
          </span>
          <span className="adm-brand-text">
            <span className="adm-brand-name">Genova</span>
            <span className="adm-brand-scope">Admin console</span>
          </span>
        </div>
        {navList}
        {footer}
      </aside>

      <div className="adm-main">
        <header className="adm-topbar">
          <IconButton
            label="Open navigation"
            className="adm-mobile-only"
            onClick={() => onMobileNav(true)}
          >
            <Menu />
          </IconButton>
          <IconButton
            label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            className="adm-collapse"
            onClick={onToggleCollapsed}
          >
            {collapsed ? <PanelLeftOpen /> : <PanelLeftClose />}
          </IconButton>
          <div className="adm-crumbs">
            <a href="/admin">Admin</a>
            <ChevronRight aria-hidden />
            <span aria-current="page">{current.label}</span>
          </div>
          <div className="adm-topbar-actions">
            <button type="button" className="adm-searchbtn" onClick={onOpenSearch}>
              <Search aria-hidden />
              <span>Search admin</span>
              <kbd>Ctrl K</kbd>
            </button>
            <IconButton
              label={refreshing ? "Refreshing data" : "Refresh this page"}
              onClick={onRefresh}
            >
              <RefreshCw className={refreshing ? "animate-spin" : undefined} />
            </IconButton>
          </div>
        </header>

        <main className="adm-content">{children}</main>
      </div>

      {mobileNavOpen ? (
        <>
          <button
            type="button"
            className="adm-drawer-backdrop"
            aria-label="Close navigation"
            style={{ background: "rgb(9 12 20 / 40%)" }}
            onClick={() => onMobileNav(false)}
          />
          <div className="adm-drawer-nav" role="dialog" aria-modal="true" aria-label="Admin navigation">
            <div className="adm-brand">
              <span className="adm-brand-mark" aria-hidden>
                G
              </span>
              <span className="adm-brand-text">
                <span className="adm-brand-name">Genova</span>
                <span className="adm-brand-scope">Admin console</span>
              </span>
              <IconButton
                label="Close navigation"
                onClick={() => onMobileNav(false)}
                className="adm-mobile-only"
                style={{ marginLeft: "auto" }}
              >
                <X />
              </IconButton>
            </div>
            {navList}
            <div className="adm-side-foot">
              <AdminButton
                variant="secondary"
                style={{ width: "100%" }}
                onClick={() => {
                  window.location.href = "/dashboard?view=documents";
                }}
              >
                <ArrowLeft style={{ width: 15, height: 15 }} />
                Back to workspace
              </AdminButton>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}

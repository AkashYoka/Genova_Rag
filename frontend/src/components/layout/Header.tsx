"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { useRouter, usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  LogOut,
  Menu,
  X,
  Sun,
  Moon,
  Settings,
  Columns2,
  StickyNote,
  Network,
} from "lucide-react";
import { useTheme } from "next-themes";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

import { KeyboardShortcutsModal } from "./KeyboardShortcutsModal";
import Logo from "./Logo";

interface HeaderProps {
  sidebarOpen?: boolean;
  onToggleSidebar?: () => void;
  viewerOpen?: boolean;
  onToggleViewer?: () => void;
  notesOpen?: boolean;
  onToggleNotes?: () => void;
  compareOpen?: boolean;
  onToggleCompare?: () => void;
  graphOpen?: boolean;
  onToggleGraph?: () => void;
  /** Pass DocumentSidebar JSX so the mobile sheet can render it */
  mobileSheetContent?: React.ReactNode;
}

const subscribe = () => () => {};
const getSnapshot = () => true;
const getServerSnapshot = () => false;

export default function Header({
  sidebarOpen = false,
  onToggleSidebar,
  viewerOpen = false,
  onToggleViewer,
  notesOpen = false,
  onToggleNotes,
  compareOpen = false,
  onToggleCompare,
  graphOpen = false,
  onToggleGraph,
  mobileSheetContent,
}: HeaderProps) {
  const { user, logout } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const { theme, setTheme } = useTheme();
  const mounted = useSyncExternalStore(
    subscribe,
    getSnapshot,
    getServerSnapshot,
  ); // ← replaces useState + useEffect
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [temperature, setTemperature] = useState(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("temperature");
      return saved ? Number(saved) : 0.5;
    }
    return 0.5;
  });
  useEffect(() => {
    localStorage.setItem("temperature", temperature.toString());
  }, [temperature]);
  const [sheetOpen, setSheetOpen] = useState(false);
  const isDark = theme === "dark";
  const toggleTheme = () => setTheme(isDark ? "light" : "dark");

  const handleLogout = () => {
    logout();
    router.replace("/login");
  };


  const toolBtn =
    "h-8 w-8 text-[var(--gd-muted,#5c636c)] hover:bg-[var(--gd-raised,#eceef2)] hover:text-[var(--gd-text,#14171c)]";

  return (
    <>
      <header className="genova-desk-top">
        <div className="flex min-w-0 items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            className={`${toolBtn} md:hidden`}
            onClick={() => setSheetOpen(true)}
            title="Open sidebar"
            aria-label="Open document navigation"
            aria-expanded={sheetOpen}
            aria-controls="mobile-document-navigation"
          >
            <Menu className="h-4 w-4" />
          </Button>

          {onToggleSidebar ? (
            <Button
              variant="ghost"
              size="icon"
              className={`${toolBtn} hidden md:inline-flex`}
              onClick={onToggleSidebar}
              title={sidebarOpen ? "Close sidebar" : "Open sidebar"}
              aria-label={
                sidebarOpen ? "Close document sidebar" : "Open document sidebar"
              }
              aria-pressed={sidebarOpen}
            >
              {sidebarOpen ? (
                <PanelLeftClose className="h-4 w-4" />
              ) : (
                <PanelLeftOpen className="h-4 w-4" />
              )}
            </Button>
          ) : null}

          <Link
            href="/dashboard"
            className="flex min-w-0 cursor-pointer items-center gap-2.5"
            aria-label="Go to homepage"
          >
            <Logo size={26} />
            <span className="hidden min-w-0 flex-col sm:flex">
              <span className="brand-name">Genova</span>
              <span className="brand-sub">by Yoka</span>
            </span>
          </Link>

          <nav className="ml-3 hidden items-center gap-1 sm:flex" aria-label="Main Navigation">
            <Link
              href="/dashboard"
              className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                pathname === "/dashboard"
                  ? "bg-[var(--gd-active,#e6e8ec)] text-[var(--gd-text,#14171c)] font-semibold"
                  : "text-[var(--gd-muted,#5c636c)] hover:bg-[var(--gd-raised,#eceef2)] hover:text-[var(--gd-text,#14171c)]"
              }`}
            >
              Workspace
            </Link>
            <Link
              href="/documents"
              className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                pathname === "/documents"
                  ? "bg-[var(--gd-active,#e6e8ec)] text-[var(--gd-text,#14171c)] font-semibold"
                  : "text-[var(--gd-muted,#5c636c)] hover:bg-[var(--gd-raised,#eceef2)] hover:text-[var(--gd-text,#14171c)]"
              }`}
            >
              Document Library
            </Link>
          </nav>
        </div>

        <div className="flex items-center gap-1">
          {onToggleViewer ? (
            <Button
              variant="ghost"
              size="icon"
              className={toolBtn}
              onClick={onToggleViewer}
              title={viewerOpen ? "Close viewer" : "Open viewer"}
              aria-label={viewerOpen ? "Close PDF viewer" : "Open PDF viewer"}
              aria-pressed={viewerOpen}
            >
              {viewerOpen ? (
                <PanelRightClose className="h-4 w-4" />
              ) : (
                <PanelRightOpen className="h-4 w-4" />
              )}
            </Button>
          ) : null}
          {onToggleNotes ? (
            <Button
              variant={notesOpen ? "secondary" : "ghost"}
              size="icon"
              className={
                notesOpen
                  ? "h-8 w-8 bg-[var(--gd-active,#e6e8ec)] text-[var(--gd-text,#14171c)]"
                  : toolBtn
              }
              onClick={onToggleNotes}
              title={notesOpen ? "Close notes" : "Open notes"}
              aria-label={notesOpen ? "Close notes panel" : "Open notes panel"}
              aria-pressed={notesOpen}
            >
              <StickyNote className="h-4 w-4" />
            </Button>
          ) : null}
          {onToggleCompare ? (
            <Button
              variant={compareOpen ? "secondary" : "ghost"}
              size="icon"
              className={
                compareOpen
                  ? "h-8 w-8 bg-[var(--gd-active,#e6e8ec)] text-[var(--gd-text,#14171c)]"
                  : toolBtn
              }
              onClick={onToggleCompare}
              title={
                compareOpen
                  ? "Close compare view"
                  : "Compare two PDFs side-by-side"
              }
              aria-label={
                compareOpen ? "Close compare view" : "Open compare view"
              }
              aria-pressed={compareOpen}
            >
              <Columns2 className="h-4 w-4" />
            </Button>
          ) : null}
          {onToggleGraph ? (
            <Button
              variant={graphOpen ? "secondary" : "ghost"}
              size="icon"
              className={
                graphOpen
                  ? "h-8 w-8 bg-[var(--gd-active,#e6e8ec)] text-[var(--gd-text,#14171c)]"
                  : toolBtn
              }
              onClick={onToggleGraph}
              title={graphOpen ? "Close knowledge graph" : "Open knowledge graph"}
              aria-label={
                graphOpen ? "Close knowledge graph" : "Open knowledge graph"
              }
              aria-pressed={graphOpen}
            >
              <Network className="h-4 w-4" />
            </Button>
          ) : null}
          <Button
            variant="ghost"
            size="icon"
            className={toolBtn}
            onClick={() => setSettingsOpen(true)}
            title="LLM Settings"
            aria-label="Open LLM settings"
          >
            <Settings className="h-4 w-4" />
          </Button>

          {mounted && (
            <Button
              variant="ghost"
              size="icon"
              className={toolBtn}
              onClick={toggleTheme}
              title={isDark ? "Light mode" : "Dark mode"}
              aria-label={
                isDark ? "Switch to light mode" : "Switch to dark mode"
              }
            >
              {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
          )}

          <KeyboardShortcutsModal />

          <span className="mx-1 hidden h-5 w-px bg-[var(--gd-border,#e2e5ea)] sm:block" aria-hidden />

          <DropdownMenu>
            <DropdownMenuTrigger
              className="flex h-8 cursor-pointer items-center gap-1.5 rounded-md px-1.5 transition-colors hover:bg-[var(--gd-raised,#eceef2)]"
              aria-label={`Open user menu for ${user?.username ?? "current user"}`}
            >
              <Avatar className="h-6 w-6">
                <AvatarFallback className="bg-[var(--gd-primary,#1a1d21)] text-[10px] text-[var(--gd-primary-ink,#fff)]">
                  {user?.username?.slice(0, 2).toUpperCase() || "U"}
                </AvatarFallback>
              </Avatar>
              <span className="hidden text-[13px] sm:inline">{user?.username}</span>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <div className="px-3 py-2">
                <p className="text-sm font-medium">{user?.username}</p>
                <p className="truncate text-xs text-muted-foreground">
                  {user?.email}
                </p>
              </div>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                className="cursor-pointer text-destructive"
                onClick={handleLogout}
              >
                <LogOut className="mr-2 h-4 w-4" />
                Sign out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      {/* Mobile navigation sheet - backdrop */}
      {sheetOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/40 md:hidden"
          onClick={() => setSheetOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside
        id="mobile-document-navigation"
        className={[
          "fixed inset-y-0 left-0 z-50 w-72 flex flex-col",
          "bg-sidebar border-r border-sidebar-border",
          "transform transition-transform duration-300 ease-in-out md:hidden",
          sheetOpen ? "translate-x-0" : "-translate-x-full",
        ].join(" ")}
        aria-label="Mobile navigation"
        aria-hidden={!sheetOpen}
        {...(!sheetOpen ? { inert: true } : {})}
      >
        <div className="h-14 flex items-center justify-between px-4 border-b border-sidebar-border flex-shrink-0">
          {/* MOBILE LOGO — clicking navigates to dashboard and closes the sheet */}
          <Link
            href="/dashboard"
            className="flex items-center gap-2.5 hover:opacity-85 transition-opacity cursor-pointer"
            aria-label="Go to homepage"
            onClick={() => setSheetOpen(false)}
          >
            <Logo size={28} />
            <span className="font-semibold text-sm">Document AI Analyst</span>
          </Link>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() => setSheetOpen(false)}
            aria-label="Close navigation"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>

        <div className="flex border-b border-sidebar-border p-2 gap-1 bg-sidebar-accent/30">
          <Link
            href="/dashboard"
            onClick={() => setSheetOpen(false)}
            className={`flex-1 text-center py-1.5 px-2 rounded-md text-xs font-medium transition-colors ${
              pathname === "/dashboard"
                ? "bg-primary text-primary-foreground font-semibold"
                : "text-muted-foreground hover:bg-sidebar-accent hover:text-foreground"
            }`}
          >
            Workspace
          </Link>
          <Link
            href="/documents"
            onClick={() => setSheetOpen(false)}
            className={`flex-1 text-center py-1.5 px-2 rounded-md text-xs font-medium transition-colors ${
              pathname === "/documents"
                ? "bg-primary text-primary-foreground font-semibold"
                : "text-muted-foreground hover:bg-sidebar-accent hover:text-foreground"
            }`}
          >
            Document Library
          </Link>
        </div>

        <div className="flex-1 overflow-hidden">
          {sheetOpen ? mobileSheetContent : null}
        </div>
      </aside>

      <Dialog open={settingsOpen} onOpenChange={setSettingsOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>LLM Settings</DialogTitle>
          </DialogHeader>

          <div className="space-y-4">
            <label className="text-sm">Temperature: {temperature}</label>

            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              className="w-full"
            />
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

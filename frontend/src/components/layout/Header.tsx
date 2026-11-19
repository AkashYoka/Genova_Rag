"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
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
  LogOut,
  Menu,
  X,
  Sun,
  Moon,
  Settings,
  Shield,
} from "lucide-react";
import { isAdminUser } from "@/lib/admin";
import { useTheme } from "next-themes";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

import { KeyboardShortcutsModal } from "./KeyboardShortcutsModal";
import Logo from "./Logo";
import GoogleDriveSettings from "@/components/settings/GoogleDriveSettings";

interface HeaderProps {
  sidebarOpen?: boolean;
  onToggleSidebar?: () => void;
  /** Pass sidebar JSX so the mobile sheet can render it */
  mobileSheetContent?: React.ReactNode;
}

const subscribe = () => () => {};
const getSnapshot = () => true;
const getServerSnapshot = () => false;

export default function Header({
  sidebarOpen = false,
  onToggleSidebar,
  mobileSheetContent,
}: HeaderProps) {
  const { user, logout } = useAuth();
  const router = useRouter();
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
    "h-8 w-8 text-[var(--gd-muted)] hover:bg-[var(--gd-raised)] hover:text-[var(--gd-text)]";

  return (
    <>
      <header className="genova-desk-top">
        <div className="flex min-w-0 items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            className={`${toolBtn} md:hidden`}
            onClick={() => setSheetOpen(true)}
            title="Open navigation"
            aria-label="Open navigation"
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
              title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
              aria-label={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
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
            href="/dashboard?view=documents"
            className="flex min-w-0 cursor-pointer items-center gap-2.5"
            aria-label="Genova documents"
          >
            <Logo size={26} />
            <span className="hidden min-w-0 flex-col sm:flex">
              <span className="brand-name">Genova</span>
              <span className="brand-sub">by Yoka</span>
            </span>
          </Link>

        </div>

        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className={toolBtn}
            onClick={() => setSettingsOpen(true)}
            title="Settings"
            aria-label="Open settings"
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

          <span
            className="mx-1 hidden h-5 w-px bg-[var(--gd-border)] sm:block"
            aria-hidden
          />

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
                className="cursor-pointer"
                onClick={() => router.push("/settings")}
              >
                <Settings className="mr-2 h-4 w-4" />
                Settings & usage
              </DropdownMenuItem>
              {isAdminUser(user) ? (
                <DropdownMenuItem
                  className="cursor-pointer"
                  onClick={() => router.push("/admin")}
                >
                  <Shield className="mr-2 h-4 w-4" />
                  Admin dashboard
                </DropdownMenuItem>
              ) : null}
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
          "fixed inset-y-0 left-0 z-50 flex w-72 flex-col",
          "bg-sidebar border-r border-sidebar-border",
          "transform transition-transform duration-300 ease-in-out md:hidden",
          sheetOpen ? "translate-x-0" : "-translate-x-full",
        ].join(" ")}
        aria-label="Mobile navigation"
        aria-hidden={!sheetOpen}
        {...(!sheetOpen ? { inert: true } : {})}
      >
        <div className="flex h-14 shrink-0 items-center justify-between border-b border-sidebar-border px-4">
          <Link
            href="/dashboard?view=documents"
            className="flex items-center gap-2.5 transition-opacity hover:opacity-85"
            aria-label="Genova documents"
            onClick={() => setSheetOpen(false)}
          >
            <Logo size={28} />
            <span className="text-sm font-semibold">Genova</span>
          </Link>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() => setSheetOpen(false)}
            aria-label="Close navigation"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div
          className="min-h-0 flex-1 overflow-hidden"
          onClickCapture={(event) => {
            const target = event.target as HTMLElement | null;
            if (target?.closest("[data-sidebar-nav]")) setSheetOpen(false);
          }}
        >
          {sheetOpen ? mobileSheetContent : null}
        </div>
      </aside>

      <Dialog open={settingsOpen} onOpenChange={setSettingsOpen}>
        <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Settings</DialogTitle>
          </DialogHeader>

          <div className="space-y-6">
            <div className="space-y-4">
              <p className="text-sm font-medium">LLM</p>
              <label className="text-sm text-muted-foreground">
                Temperature: {temperature}
              </label>
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

            <div className="border-t border-border pt-4">
              <GoogleDriveSettings />
            </div>

            <div className="border-t border-border pt-4">
              <p className="mb-2 text-sm font-medium">AI Usage</p>
              <p className="mb-3 text-sm text-muted-foreground">
                Review token consumption and estimated costs for your account.
              </p>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setSettingsOpen(false);
                  router.push("/settings");
                }}
              >
                Open usage dashboard
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

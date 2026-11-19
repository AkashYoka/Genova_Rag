"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  Clock,
  Database,
  DatabaseBackup,
  Download,
  FileText,
  Files,
  HardDrive,
  Info,
  KeyRound,
  MessageSquare,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  Users as UsersIcon,
} from "lucide-react";
import { toast } from "sonner";

import { api, CONNECTION_ERROR_MESSAGE } from "@/lib/api";
import { formatBytes, formatUsd, isAdminUser } from "@/lib/admin";
import { useAuth } from "@/lib/auth";
import { RequireAuth } from "@/components/auth/RequireAuth";
import { AdminShell, type AdminSectionId } from "@/components/admin/console/AdminShell";
import { AdminSearch } from "@/components/admin/console/AdminSearch";
import {
  AdminButton,
  BarChart,
  Badge,
  Callout,
  Card,
  ConfirmDialog,
  DefinitionList,
  Drawer,
  EmptyState,
  Field,
  PageHeader,
  SearchInput,
  SelectInput,
  Stat,
  StatGrid,
  TableSkeleton,
  TableWrap,
  Tooltip,
  type Tone,
} from "@/components/admin/console/ConsoleUI";
import { FeedbackManagementPanel } from "@/components/admin/FeedbackManagementPanel";
import { AiTracesPanel } from "@/components/admin/AiTracesPanel";
import { PromptManagementPanel } from "@/components/admin/PromptManagementPanel";

/* ------------------------------------------------------------------ types */

interface AdminStats {
  total_users: number;
  total_pdfs_uploaded: number;
  total_documents: number;
  total_messages: number;
  average_query_response_time_ms: number;
  query_count: number;
  disk_space_usage: {
    total_bytes: number;
    used_bytes: number;
    free_bytes: number;
    usage_percent: number;
    upload_dir_bytes: number;
  };
}

interface AdminUser {
  id: string;
  username: string;
  email: string;
  role: string;
  is_admin: boolean;
  is_verified: boolean;
  display_name?: string | null;
  created_at: string;
}

interface UsageSummary {
  request_count: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number;
  today: { request_count: number; total_tokens: number; estimated_cost_usd: number };
  month_to_date: { request_count: number; total_tokens: number; estimated_cost_usd: number };
  by_provider: Array<{
    provider: string;
    request_count: number;
    total_tokens: number;
    estimated_cost_usd: number;
  }>;
  by_model: Array<{
    provider: string;
    model_id: string | null;
    request_count: number;
    total_tokens: number;
    estimated_cost_usd: number;
  }>;
  by_user?: Array<{
    user_id: string;
    request_count: number;
    total_tokens: number;
    estimated_cost_usd: number;
  }>;
  cost_disclaimer: string;
}

interface UsageDailyPoint {
  date: string;
  total_tokens: number;
  estimated_cost_usd: number;
  request_count: number;
}

interface UsageEvent {
  id: string;
  created_at: string;
  provider: string;
  model_id: string | null;
  usage_type: string;
  total_tokens: number | null;
  estimated_cost_usd: number | null;
  cost_type: string;
  user_id?: string | null;
}

interface AdminDocumentsResponse {
  total: number;
  status_counts: Record<string, number>;
  items: Array<{
    id: string;
    username?: string | null;
    email?: string | null;
    original_name: string;
    status: string;
    page_count?: number | null;
    chunk_count?: number | null;
    error_message?: string | null;
    processing_stage?: string | null;
    retry_count?: number | null;
    uploaded_at?: string | null;
    completed_at?: string | null;
  }>;
}

interface TracesSummary {
  total: number;
  success: number;
  failed: number;
  partial: number;
  avg_duration_ms: number;
}

interface FeedbackSummary {
  total: number;
  likes: number;
  dislikes: number;
  satisfaction_percentage: number;
}

interface HealthComponent {
  status?: string;
  latency_ms?: number;
  detail?: string;
  error?: string;
  workers?: unknown[];
}

interface HealthStatus {
  overall?: string;
  components?: Record<string, HealthComponent | undefined>;
  database?: HealthComponent;
  redis?: HealthComponent;
  celery?: HealthComponent;
  chromadb?: HealthComponent;
  ollama?: HealthComponent;
}

/* ---------------------------------------------------------------- helpers */

const SECTION_ALIASES: Record<string, AdminSectionId> = {
  overview: "overview",
  users: "users",
  usage: "usage",
  rag: "documents",
  documents: "documents",
  feedback: "feedback",
  "ai-traces": "traces",
  traces: "traces",
  prompts: "styles",
  styles: "styles",
  integrations: "connections",
  connections: "connections",
  database: "export",
  export: "export",
  health: "status",
  status: "status",
};

function sectionFromLocation(): AdminSectionId {
  if (typeof window === "undefined") return "overview";
  const key = new URLSearchParams(window.location.search).get("section") ?? "overview";
  return SECTION_ALIASES[key] ?? "overview";
}

function healthParts(health: HealthStatus | null) {
  const c = health?.components;
  return {
    database: c?.database ?? health?.database,
    redis: c?.redis ?? health?.redis,
    celery: c?.celery ?? health?.celery,
    chromadb: c?.chromadb ?? health?.chromadb,
    ollama: c?.ollama ?? health?.ollama,
  };
}

function toneForStatus(status: string): Tone {
  const s = status.toLowerCase();
  if (["ready", "up", "ok", "healthy", "completed", "active", "success"].includes(s)) return "ok";
  if (["failed", "down", "error", "unhealthy"].includes(s)) return "danger";
  if (["degraded", "processing", "pending", "warning", "partial", "unavailable"].includes(s))
    return "warn";
  return "neutral";
}

function documentStatusLabel(status: string) {
  const s = status.toLowerCase();
  if (s === "ready") return "Ready";
  if (s === "failed") return "Needs attention";
  if (s === "processing") return "Processing";
  if (s === "pending") return "Waiting";
  return status;
}

function serviceState(status?: string) {
  const s = (status || "").toLowerCase();
  if (s === "up") return "Running";
  if (s === "down") return "Not running";
  if (s === "unavailable") return "Unavailable";
  if (s === "skipped") return "Not in use";
  if (s === "degraded") return "Degraded";
  return status || "Unknown";
}

function relativeDay(iso?: string | null) {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso.slice(0, 10);
  const days = Math.floor((Date.now() - then) / 86_400_000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 30) return `${days} days ago`;
  return new Date(iso).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function dateTime(iso?: string | null) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso.replace("T", " ").slice(0, 19);
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function initialsOf(value: string) {
  return value
    .split(/[\s._@-]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("");
}

function primaryProviderName(provider: string) {
  if (provider.toLowerCase() === "bedrock") return "Amazon Bedrock";
  if (provider.toLowerCase() === "ollama") return "Ollama (local)";
  return provider;
}

/* ------------------------------------------------------------------- page */

function AdminConsole() {
  const { user } = useAuth();

  const [section, setSection] = useState<AdminSectionId>("overview");
  const [collapsed, setCollapsed] = useState(false);
  const [mobileNav, setMobileNav] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);

  const [stats, setStats] = useState<AdminStats | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const [usageDaily, setUsageDaily] = useState<UsageDailyPoint[]>([]);
  const [usageEvents, setUsageEvents] = useState<UsageEvent[]>([]);
  const [usageEventPage, setUsageEventPage] = useState(1);
  const [usageEventPages, setUsageEventPages] = useState(1);
  const [usageEventTotal, setUsageEventTotal] = useState(0);
  const [usageMetric, setUsageMetric] = useState<"total_tokens" | "estimated_cost_usd">(
    "total_tokens",
  );
  const [docs, setDocs] = useState<AdminDocumentsResponse | null>(null);
  const [docStatus, setDocStatus] = useState<string>("all");
  const [docQuery, setDocQuery] = useState("");
  const [overviewDocs, setOverviewDocs] = useState<AdminDocumentsResponse | null>(null);
  const [traceSummary, setTraceSummary] = useState<TracesSummary | null>(null);
  const [feedbackSummary, setFeedbackSummary] = useState<FeedbackSummary | null>(null);
  const [drive, setDrive] = useState<{ total: number; items: Array<Record<string, unknown>> } | null>(
    null,
  );
  const [workspaces, setWorkspaces] = useState<{
    total: number;
    items: Array<Record<string, unknown>>;
  } | null>(null);
  const [apiKeys, setApiKeys] = useState<{
    total_keys: number;
    active_keys: number;
    by_user: Array<Record<string, unknown>>;
  } | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [exporting, setExporting] = useState<"json" | "sql" | null>(null);
  const [exportChoice, setExportChoice] = useState<"json" | "sql" | null>(null);

  const [userQuery, setUserQuery] = useState("");
  const [roleFilter, setRoleFilter] = useState<"all" | "admin" | "user">("all");
  const [verifiedFilter, setVerifiedFilter] = useState<"all" | "yes" | "no">("all");
  const [userSort, setUserSort] = useState<"newest" | "oldest" | "name">("newest");
  const [userPage, setUserPage] = useState(1);
  const [openUser, setOpenUser] = useState<AdminUser | null>(null);
  const [openDoc, setOpenDoc] = useState<AdminDocumentsResponse["items"][number] | null>(null);

  const refreshNonce = useRef(0);

  /* ---------------------------------------------------------- data loads */

  const loadOverview = useCallback(async () => {
    const [statsRes, healthRes, usageRes, dailyRes, docsRes, tracesRes, feedbackRes] =
      await Promise.allSettled([
        api.get<AdminStats>("/api/v1/admin/stats"),
        api.get<HealthStatus>("/api/v1/health/status"),
        api.get<UsageSummary>("/api/v1/admin/usage/summary"),
        api.get<{ points: UsageDailyPoint[] }>("/api/v1/admin/usage/daily"),
        api.get<AdminDocumentsResponse>("/api/v1/admin/documents?limit=5"),
        api.get<{ summary: TracesSummary }>("/api/v1/admin/ai-traces?page=1&page_size=1"),
        api.get<{ summary: FeedbackSummary }>("/api/v1/admin/feedback?limit=1"),
      ]);

    if (statsRes.status === "rejected") throw statsRes.reason;
    setStats(statsRes.value);
    if (healthRes.status === "fulfilled") setHealth(healthRes.value);
    if (usageRes.status === "fulfilled") setUsage(usageRes.value);
    if (dailyRes.status === "fulfilled") setUsageDaily(dailyRes.value.points || []);
    if (docsRes.status === "fulfilled") setOverviewDocs(docsRes.value);
    if (tracesRes.status === "fulfilled") setTraceSummary(tracesRes.value.summary);
    if (feedbackRes.status === "fulfilled") setFeedbackSummary(feedbackRes.value.summary);
  }, []);

  const loadUsage = useCallback(async (page = 1) => {
    const [summary, daily, events, usersData] = await Promise.all([
      api.get<UsageSummary>("/api/v1/admin/usage/summary"),
      api.get<{ points: UsageDailyPoint[] }>("/api/v1/admin/usage/daily"),
      api.get<{ items: UsageEvent[]; pages: number; total: number }>(
        `/api/v1/admin/usage/events?limit=20&page=${page}`,
      ),
      api.get<AdminUser[]>("/api/v1/admin/users"),
    ]);
    setUsage(summary);
    setUsageDaily(daily.points || []);
    setUsageEvents(events.items || []);
    setUsageEventPages(events.pages || 1);
    setUsageEventTotal(events.total || 0);
    setUsageEventPage(page);
    setUsers(usersData);
  }, []);

  const loadDocs = useCallback(async (status: string) => {
    const qs = status && status !== "all" ? `&status=${encodeURIComponent(status)}` : "";
    setDocs(await api.get<AdminDocumentsResponse>(`/api/v1/admin/documents?limit=100${qs}`));
  }, []);

  const loadUsers = useCallback(async () => {
    setUsers(await api.get<AdminUser[]>("/api/v1/admin/users"));
  }, []);

  const loadConnections = useCallback(async () => {
    const [driveData, wsData, keysData] = await Promise.all([
      api.get<{ total: number; items: Array<Record<string, unknown>> }>(
        "/api/v1/admin/drive/connections",
      ),
      api.get<{ total: number; items: Array<Record<string, unknown>> }>(
        "/api/v1/admin/workspaces",
      ),
      api.get<{
        total_keys: number;
        active_keys: number;
        by_user: Array<Record<string, unknown>>;
      }>("/api/v1/admin/api-keys/summary"),
    ]);
    setDrive(driveData);
    setWorkspaces(wsData);
    setApiKeys(keysData);
  }, []);

  const loadHealth = useCallback(async () => {
    setHealth(await api.get<HealthStatus>("/api/v1/health/status"));
  }, []);

  const refresh = useCallback(async () => {
    if (!isAdminUser(user)) return;
    const nonce = ++refreshNonce.current;
    try {
      setLoading(true);
      setError("");
      if (section === "overview") await loadOverview();
      else if (section === "users") await loadUsers();
      else if (section === "usage") await loadUsage(usageEventPage);
      else if (section === "documents") await loadDocs(docStatus);
      else if (section === "connections") await loadConnections();
      else if (section === "status") await loadHealth();
      if (nonce === refreshNonce.current) setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : CONNECTION_ERROR_MESSAGE);
    } finally {
      if (nonce === refreshNonce.current) setLoading(false);
    }
  }, [
    user,
    section,
    usageEventPage,
    docStatus,
    loadOverview,
    loadUsage,
    loadDocs,
    loadUsers,
    loadConnections,
    loadHealth,
  ]);

  useEffect(() => {
    void refresh();
    // Panels own their data; they refresh themselves.
  }, [refresh]);

  // Deep links + browser back/forward.
  useEffect(() => {
    setSection(sectionFromLocation());
    const onPop = () => setSection(sectionFromLocation());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const navigate = useCallback((id: AdminSectionId, query?: string) => {
    setSection(id);
    setError("");
    if (id === "users" && query !== undefined) setUserQuery(query);
    if (id === "documents" && query !== undefined) {
      setDocStatus("all");
      setDocQuery(query);
    }
    const url = id === "overview" ? "/admin" : `/admin?section=${id}`;
    if (window.location.pathname + window.location.search !== url) {
      window.history.pushState({ section: id }, "", url);
    }
    window.scrollTo({ top: 0 });
  }, []);

  // Cmd/Ctrl+K opens search.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setSearchOpen((v) => !v);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    try {
      setCollapsed(window.localStorage.getItem("genova.admin.collapsed") === "1");
    } catch {
      /* ignore */
    }
  }, []);

  const toggleCollapsed = () => {
    setCollapsed((v) => {
      const next = !v;
      try {
        window.localStorage.setItem("genova.admin.collapsed", next ? "1" : "0");
      } catch {
        /* ignore */
      }
      return next;
    });
  };

  /* -------------------------------------------------------------- derived */

  const docsCounts = docs?.status_counts ?? overviewDocs?.status_counts ?? {};
  const readyDocs = docsCounts.ready ?? 0;
  const failedDocs = docsCounts.failed ?? 0;
  const pendingDocs = (docsCounts.pending ?? 0) + (docsCounts.processing ?? 0);

  const filteredUsers = useMemo(() => {
    const q = userQuery.trim().toLowerCase();
    const list = users.filter((u) => {
      const role = String(u.role || (u.is_admin ? "admin" : "user")).toLowerCase();
      if (roleFilter === "admin" && !(role === "admin" || u.is_admin)) return false;
      if (roleFilter === "user" && (role === "admin" || u.is_admin)) return false;
      if (verifiedFilter === "yes" && !u.is_verified) return false;
      if (verifiedFilter === "no" && u.is_verified) return false;
      if (!q) return true;
      return (
        u.username?.toLowerCase().includes(q) ||
        u.email?.toLowerCase().includes(q) ||
        u.display_name?.toLowerCase().includes(q)
      );
    });
    return list.sort((a, b) => {
      if (userSort === "name")
        return (a.display_name || a.username).localeCompare(b.display_name || b.username);
      const at = new Date(a.created_at).getTime();
      const bt = new Date(b.created_at).getTime();
      return userSort === "oldest" ? at - bt : bt - at;
    });
  }, [users, userQuery, roleFilter, verifiedFilter, userSort]);

  const usersPerPage = 15;
  const userPageCount = Math.max(1, Math.ceil(filteredUsers.length / usersPerPage));
  const visibleUsers = filteredUsers.slice((userPage - 1) * usersPerPage, userPage * usersPerPage);

  useEffect(() => {
    if (userPage > userPageCount) setUserPage(1);
  }, [userPage, userPageCount]);

  const adminCount = useMemo(
    () => users.filter((u) => isAdminUser(u)).length,
    [users],
  );

  const filteredDocs = useMemo(() => {
    const items = docs?.items ?? [];
    const q = docQuery.trim().toLowerCase();
    if (!q) return items;
    return items.filter(
      (doc) =>
        doc.original_name?.toLowerCase().includes(q) ||
        doc.username?.toLowerCase().includes(q) ||
        doc.email?.toLowerCase().includes(q),
    );
  }, [docs, docQuery]);

  const userLabel = useMemo(() => {
    const map = new Map(users.map((u) => [u.id, u.username || u.email]));
    return (id?: string | null) => (id ? map.get(id) || id.slice(0, 8) : "—");
  }, [users]);

  const exportDb = async (format: "json" | "sql") => {
    try {
      setExporting(format);
      const blob = await api.getBlob(`/api/v1/admin/export-db?format=${format}`);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `db_backup.${format}`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success(`Backup downloaded as ${format.toUpperCase()}.`);
      setExportChoice(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Export failed";
      setError(message);
      toast.error(message);
    } finally {
      setExporting(null);
    }
  };

  const refreshing = loading;

  /* --------------------------------------------------------------- render */

  return (
    <>
      <AdminShell
        active={section}
        onNavigate={navigate}
        collapsed={collapsed}
        onToggleCollapsed={toggleCollapsed}
        onOpenSearch={() => setSearchOpen(true)}
        onRefresh={() => void refresh()}
        refreshing={refreshing}
        userEmail={user?.email ?? ""}
        userName={user?.username || "Admin"}
        mobileNavOpen={mobileNav}
        onMobileNav={setMobileNav}
      >
        {error ? (
          <div style={{ marginBottom: 16 }}>
            <Callout tone="danger" icon={<AlertTriangle />}>
              <strong>We couldn’t load this page.</strong> {error}
              <div style={{ marginTop: 8 }}>
                <AdminButton size="sm" variant="secondary" onClick={() => void refresh()}>
                  <RefreshCw /> Try again
                </AdminButton>
              </div>
            </Callout>
          </div>
        ) : null}

        {section === "overview" ? (
          <OverviewSection
            stats={stats}
            usage={usage}
            usageDaily={usageDaily}
            health={health}
            docs={overviewDocs}
            traceSummary={traceSummary}
            feedbackSummary={feedbackSummary}
            loading={loading}
            lastUpdated={lastUpdated}
            onNavigate={navigate}
            onExport={() => setExportChoice("json")}
          />
        ) : null}

        {section === "users" ? (
          <UsersSection
            users={users}
            loading={loading}
            filtered={filteredUsers}
            visible={visibleUsers}
            query={userQuery}
            roleFilter={roleFilter}
            verifiedFilter={verifiedFilter}
            sort={userSort}
            page={userPage}
            pageCount={userPageCount}
            adminCount={adminCount}
            onQuery={(v) => {
              setUserQuery(v);
              setUserPage(1);
            }}
            onRole={(v) => {
              setRoleFilter(v);
              setUserPage(1);
            }}
            onVerified={(v) => {
              setVerifiedFilter(v);
              setUserPage(1);
            }}
            onSort={(v) => {
              setUserSort(v);
              setUserPage(1);
            }}
            onPage={setUserPage}
            onOpen={setOpenUser}
          />
        ) : null}

        {section === "usage" ? (
          <UsageSection
            usage={usage}
            daily={usageDaily}
            events={usageEvents}
            metric={usageMetric}
            loading={loading}
            page={usageEventPage}
            pages={usageEventPages}
            total={usageEventTotal}
            userLabel={userLabel}
            onMetric={setUsageMetric}
            onPage={(p) => void loadUsage(p)}
          />
        ) : null}

        {section === "documents" ? (
          <DocumentsSection
            docs={docs}
            loading={loading}
            counts={docsCounts}
            ready={readyDocs}
            failed={failedDocs}
            pending={pendingDocs}
            status={docStatus}
            query={docQuery}
            filtered={filteredDocs}
            onStatus={(s) => {
              setDocStatus(s);
              setDocs(null);
              void loadDocs(s).catch((err) =>
                setError(err instanceof Error ? err.message : CONNECTION_ERROR_MESSAGE),
              );
            }}
            onQuery={setDocQuery}
            onOpen={setOpenDoc}
            onClear={() => {
              setDocStatus("all");
              setDocQuery("");
              setDocs(null);
              void loadDocs("all");
            }}
          />
        ) : null}

        {section === "feedback" ? <FeedbackManagementPanel /> : null}

        {section === "traces" ? <AiTracesPanel /> : null}

        {section === "styles" ? <PromptManagementPanel /> : null}

        {section === "connections" ? (
          <ConnectionsSection
            drive={drive}
            workspaces={workspaces}
            apiKeys={apiKeys}
            loading={loading}
          />
        ) : null}

        {section === "export" ? (
          <ExportSection
            userEmail={user?.email ?? ""}
            exporting={exporting}
            onChoose={setExportChoice}
          />
        ) : null}

        {section === "status" ? (
          <StatusSection health={health} loading={loading} lastUpdated={lastUpdated} />
        ) : null}

        <footer
          style={{
            marginTop: 36,
            paddingTop: 16,
            borderTop: "1px solid var(--adm-line-soft)",
            display: "flex",
            flexWrap: "wrap",
            gap: 12,
            fontSize: 12,
            color: "var(--adm-muted)",
          }}
        >
          <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
            <ShieldCheck style={{ width: 13, height: 13 }} />
            Signed in as {user?.email}
          </span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
            <KeyRound style={{ width: 13, height: 13 }} />
            Passwords and secrets are never shown
          </span>
          {lastUpdated ? (
            <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
              <Clock style={{ width: 13, height: 13 }} />
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          ) : null}
        </footer>
      </AdminShell>

      <AdminSearch open={searchOpen} onClose={() => setSearchOpen(false)} onNavigate={navigate} />

      <ConfirmDialog
        open={exportChoice !== null}
        onClose={() => setExportChoice(null)}
        onConfirm={() => exportChoice && void exportDb(exportChoice)}
        title="Download a copy of the database?"
        description="The file contains every table in the platform. Credentials and tokens are replaced with [REDACTED] before download, but it still contains personal data."
        confirmLabel={`Download ${exportChoice?.toUpperCase() ?? "file"}`}
        busy={exporting !== null}
      >
        <Callout tone="warn" icon={<ShieldCheck />}>
          Store the file somewhere approved for personal data, and delete it when you’re done.
          Exports are only available to administrators.
        </Callout>
      </ConfirmDialog>

      {/* Person detail */}
      <Drawer
        open={openUser !== null}
        onClose={() => setOpenUser(null)}
        title={openUser?.display_name || openUser?.username || "Person"}
        description={openUser?.email}
      >
        {openUser ? (
          <div className="adm-stack">
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <span className="adm-avatar" data-tone="accent" style={{ width: 42, height: 42 }}>
                {initialsOf(openUser.display_name || openUser.username || openUser.email)}
              </span>
              <div>
                <div style={{ fontWeight: 600 }}>
                  {openUser.display_name || openUser.username}
                </div>
                <div className="adm-hint">{openUser.email}</div>
              </div>
              <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
                <Badge tone={isAdminUser(openUser) ? "accent" : "neutral"}>
                  {isAdminUser(openUser) ? "Administrator" : "Member"}
                </Badge>
                <Badge tone={openUser.is_verified ? "ok" : "warn"}>
                  {openUser.is_verified ? "Verified" : "Unverified"}
                </Badge>
              </div>
            </div>

            <Card title="Account details">
              <div className="adm-card__body--pad">
                <DefinitionList
                  items={[
                    { term: "Joined", value: dateTime(openUser.created_at) },
                    { term: "Role", value: openUser.role || (openUser.is_admin ? "admin" : "user") },
                    { term: "Email status", value: openUser.is_verified ? "Verified" : "Not verified" },
                    {
                      term: "Account ID",
                      value: <span className="adm-mono">{openUser.id}</span>,
                    },
                  ]}
                />
              </div>
            </Card>

            <Callout icon={<Info />}>
              Roles are managed by backend authorization. This console does not edit accounts, so
              there is nothing here that could accidentally change someone’s access.
            </Callout>
          </div>
        ) : null}
      </Drawer>

      {/* Document detail */}
      <Drawer
        open={openDoc !== null}
        onClose={() => setOpenDoc(null)}
        title={openDoc?.original_name || "Document"}
        description={openDoc ? `${documentStatusLabel(openDoc.status)} · ${openDoc.username || openDoc.email || "Unknown owner"}` : undefined}
      >
        {openDoc ? (
          <div className="adm-stack">
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <Badge tone={toneForStatus(openDoc.status)}>
                {documentStatusLabel(openDoc.status)}
              </Badge>
              <Badge plain>{openDoc.page_count ?? 0} pages</Badge>
              <Badge plain>{openDoc.chunk_count ?? 0} indexed passages</Badge>
            </div>

            {openDoc.status === "failed" && openDoc.error_message ? (
              <Callout tone="danger" icon={<AlertTriangle />}>
                <strong>Processing stopped.</strong> {openDoc.error_message}
                <div style={{ marginTop: 6 }}>
                  The file is still stored. Fixing the source file and uploading it again is the
                  safest recovery.
                </div>
              </Callout>
            ) : null}

            <Card title="Processing details">
              <div className="adm-card__body--pad">
                <DefinitionList
                  items={[
                    { term: "Owner", value: openDoc.email || openDoc.username || "—" },
                    { term: "Uploaded", value: dateTime(openDoc.uploaded_at) },
                    { term: "Finished", value: dateTime(openDoc.completed_at) },
                    { term: "Stage", value: openDoc.processing_stage || "—" },
                    { term: "Retries", value: String(openDoc.retry_count ?? 0) },
                    { term: "Document ID", value: <span className="adm-mono">{openDoc.id}</span> },
                  ]}
                />
              </div>
            </Card>

            <Callout icon={<Info />}>
              Passages (“chunks”) are the pieces of this file Genova can quote when it answers.
            </Callout>
          </div>
        ) : null}
      </Drawer>
    </>
  );
}

/* -------------------------------------------------------------- overview */

function OverviewSection({
  stats,
  usage,
  usageDaily,
  health,
  docs,
  traceSummary,
  feedbackSummary,
  loading,
  lastUpdated,
  onNavigate,
  onExport,
}: {
  stats: AdminStats | null;
  usage: UsageSummary | null;
  usageDaily: UsageDailyPoint[];
  health: HealthStatus | null;
  docs: AdminDocumentsResponse | null;
  traceSummary: TracesSummary | null;
  feedbackSummary: FeedbackSummary | null;
  loading: boolean;
  lastUpdated: Date | null;
  onNavigate: (id: AdminSectionId) => void;
  onExport: () => void;
}) {
  const parts = healthParts(health);
  const counts = docs?.status_counts ?? {};
  const failedDocs = counts.failed ?? 0;
  const pendingDocs = (counts.pending ?? 0) + (counts.processing ?? 0);

  const attention: Array<{
    key: string;
    title: string;
    text: string;
    tone: "danger" | "warn";
    to: AdminSectionId;
  }> = [];

  if (failedDocs > 0) {
    attention.push({
      key: "failed-docs",
      title: `${failedDocs} document${failedDocs === 1 ? "" : "s"} couldn’t be read`,
      text: "Open the file to see why processing stopped, then re-upload it.",
      tone: "danger",
      to: "documents",
    });
  }
  if (pendingDocs > 0) {
    attention.push({
      key: "pending-docs",
      title: `${pendingDocs} document${pendingDocs === 1 ? "" : "s"} still processing`,
      text: "Long waits usually mean the file is large or the extraction service is busy.",
      tone: "warn",
      to: "documents",
    });
  }
  if ((traceSummary?.failed ?? 0) > 0) {
    attention.push({
      key: "failed-traces",
      title: `${traceSummary?.failed} answer${traceSummary?.failed === 1 ? "" : "s"} failed`,
      text: "Open the trace to see which step stopped and what the error was.",
      tone: "danger",
      to: "traces",
    });
  }
  if ((feedbackSummary?.dislikes ?? 0) > 0) {
    attention.push({
      key: "dislikes",
      title: `${feedbackSummary?.dislikes} answer${feedbackSummary?.dislikes === 1 ? "" : "s"} rated unhelpful`,
      text: "Read the question, the answer, and the person’s reason.",
      tone: "warn",
      to: "feedback",
    });
  }
  if ((parts.celery?.status || "").toLowerCase() === "down") {
    attention.push({
      key: "celery",
      title: "Background job worker is not running",
      text: "Uploads still process inside the web app, but separate workers are unavailable.",
      tone: "warn",
      to: "status",
    });
  }

  const services: Array<{ label: string; desc: string; info?: HealthComponent }> = [
    { label: "Database", desc: "Accounts, documents, and conversations", info: parts.database },
    { label: "Search index", desc: "Finds the passages answers are built from", info: parts.chromadb },
    { label: "Cache", desc: "Speeds up repeated work (optional)", info: parts.redis },
  ];

  return (
    <>
      <PageHeader
        title="Overview"
        description="A live picture of your workspace, your documents, and the AI behind every answer."
        actions={
          <>
            <AdminButton variant="secondary" onClick={onExport}>
              <Download /> Export backup
            </AdminButton>
            <AdminButton variant="primary" onClick={() => onNavigate("documents")}>
              <Files /> Review documents
            </AdminButton>
          </>
        }
      />

      {loading && !stats ? (
        <div className="adm-stats" aria-hidden>
          {[1, 2, 3, 4].map((i) => (
            <div className="adm-stat" key={i}>
              <span className="adm-skel" style={{ height: 11, width: 70 }} />
              <span className="adm-skel" style={{ height: 26, width: 90, marginTop: 12 }} />
              <span className="adm-skel" style={{ height: 10, width: 120, marginTop: 10 }} />
            </div>
          ))}
        </div>
      ) : stats ? (
        <StatGrid>
          <Stat
            label="People"
            icon={<UsersIcon />}
            value={stats.total_users.toLocaleString()}
            caption="Accounts on this workspace"
          />
          <Stat
            label="Documents"
            icon={<FileText />}
            value={stats.total_documents.toLocaleString()}
            caption={
              <>
                <strong>{counts.ready ?? 0} ready</strong> · {failedDocs}{" "}
                {failedDocs === 1 ? "needs" : "need"} attention
              </>
            }
          />
          <Stat
            label="Answers"
            icon={<MessageSquare />}
            value={stats.total_messages.toLocaleString()}
            caption={`${stats.query_count.toLocaleString()} measured questions`}
          />
          <Stat
            label="Estimated AI spend"
            icon={<Sparkles />}
            value={formatUsd(usage?.estimated_cost_usd ?? 0)}
            caption={
              usage
                ? `Last 30 days · ${usage.request_count.toLocaleString()} requests`
                : "Usage summary unavailable"
            }
          />
        </StatGrid>
      ) : null}

      <div className="adm-grid-main" style={{ marginTop: 16 }}>
        <div className="adm-stack">
          <Card
            title="What needs attention"
            description="Only real problems found in current data — nothing else is flagged."
            action={
              <AdminButton size="sm" variant="ghost" onClick={() => onNavigate("documents")}>
                Open documents <ArrowUpRight />
              </AdminButton>
            }
          >
            {loading && !docs && !traceSummary ? (
              <TableSkeleton rows={3} />
            ) : attention.length ? (
              <div className="adm-list">
                {attention.map((item) => (
                  <button
                    key={item.key}
                    type="button"
                    className="adm-attention"
                    style={{ border: 0, background: "transparent", width: "100%", textAlign: "left", cursor: "pointer", font: "inherit" }}
                    onClick={() => onNavigate(item.to)}
                  >
                    <span className="adm-attention__icon" data-tone={item.tone}>
                      <AlertTriangle />
                    </span>
                    <span className="adm-attention__body">
                      <span className="adm-attention__title">{item.title}</span>
                      <span className="adm-attention__text">{item.text}</span>
                    </span>
                    <span className="adm-attention__go">
                      <ArrowUpRight />
                    </span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="adm-attention" style={{ cursor: "default" }}>
                <span className="adm-attention__icon" data-tone="ok">
                  <CheckCircle2 />
                </span>
                <span className="adm-attention__body">
                  <span className="adm-attention__title">Everything looks healthy</span>
                  <span className="adm-attention__text">
                    No failed documents, failed answers, or unhelpful ratings in the current data.
                  </span>
                </span>
              </div>
            )}
          </Card>

          <Card
            title="AI usage, last 30 days"
            description="Daily volume across the whole platform. Hover a bar for the exact day."
            action={
              <AdminButton size="sm" variant="ghost" onClick={() => onNavigate("usage")}>
                Full report <ArrowUpRight />
              </AdminButton>
            }
          >
            <BarChart
              points={usageDaily}
              valueKey="total_tokens"
              formatValue={(v) => `${v.toLocaleString()} tokens`}
              ariaLabel="Daily token usage for the last 30 days"
            />
          </Card>

          <Card
            title="Latest documents"
            description="The five most recent uploads."
            action={
              <AdminButton size="sm" variant="ghost" onClick={() => onNavigate("documents")}>
                See all <ArrowUpRight />
              </AdminButton>
            }
          >
            {docs?.items?.length ? (
              <TableWrap height="auto">
                <table className="adm-table">
                  <thead>
                    <tr>
                      <th>Document</th>
                      <th className="adm-col-md">Owner</th>
                      <th>Status</th>
                      <th className="adm-col-sm" data-align="right">
                        Passages
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {docs.items.map((doc) => (
                      <tr key={doc.id}>
                        <td>
                          <div className="adm-cell-title">
                            <span className="adm-cell-icon">
                              <FileText />
                            </span>
                            <span className="adm-cell-stack">
                              <span className="adm-cell-main">{doc.original_name}</span>
                              <span className="adm-cell-sub">{doc.page_count ?? 0} pages</span>
                            </span>
                          </div>
                        </td>
                        <td className="adm-col-md adm-muted">
                          {doc.username || doc.email || "—"}
                        </td>
                        <td>
                          <Badge tone={toneForStatus(doc.status)}>
                            {documentStatusLabel(doc.status)}
                          </Badge>
                        </td>
                        <td className="adm-col-sm adm-num" data-align="right">
                          {doc.chunk_count ?? 0}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </TableWrap>
            ) : (
              <EmptyState
                title="No documents yet"
                text="Uploaded files appear here as soon as processing starts."
                icon={<FileText />}
              />
            )}
          </Card>
        </div>

        <div className="adm-stack">
          <Card
            title="System status"
            description="Live checks on the services answers depend on."
            action={<Badge tone={toneForStatus(health?.overall || "")}>{health?.overall || "Checking"}</Badge>}
          >
            {services.map((service) => {
              const status = service.info?.status || "unknown";
              return (
                <div className="adm-service" key={service.label}>
                  <span className="adm-service__dot" data-tone={toneForStatus(status)} />
                  <span className="adm-service__body">
                    <span className="adm-service__name">{service.label}</span>
                    <span className="adm-service__desc">{service.desc}</span>
                  </span>
                  <span className="adm-service__state">
                    {serviceState(status)}
                    {typeof service.info?.latency_ms === "number" ? (
                      <span style={{ display: "block", fontSize: 11 }}>
                        {Math.round(service.info.latency_ms)} ms
                      </span>
                    ) : null}
                  </span>
                </div>
              );
            })}
            <div className="adm-card__foot">
              <AdminButton size="sm" variant="secondary" onClick={() => onNavigate("status")}>
                View all services
              </AdminButton>
              {lastUpdated ? (
                <span style={{ fontSize: 11 }}>Checked {lastUpdated.toLocaleTimeString()}</span>
              ) : null}
            </div>
          </Card>

          <Card title="Storage" description="Disk space used by the platform.">
            <div className="adm-card__body--pad">
              <div
                style={{
                  height: 6,
                  borderRadius: 999,
                  background: "var(--adm-surface-2)",
                  overflow: "hidden",
                  marginBottom: 12,
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${Math.min(100, stats?.disk_space_usage.usage_percent ?? 0)}%`,
                    background:
                      (stats?.disk_space_usage.usage_percent ?? 0) > 90
                        ? "var(--adm-danger)"
                        : "var(--adm-accent)",
                  }}
                />
              </div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: 12.5,
                  color: "var(--adm-muted)",
                }}
              >
                <span>
                  <strong style={{ color: "var(--adm-text)" }}>
                    {formatBytes(stats?.disk_space_usage.used_bytes ?? 0)}
                  </strong>{" "}
                  used
                </span>
                <span>{formatBytes(stats?.disk_space_usage.free_bytes ?? 0)} free</span>
              </div>
              <div className="adm-hint" style={{ marginTop: 8 }}>
                Uploads take {formatBytes(stats?.disk_space_usage.upload_dir_bytes ?? 0)} of that
                space.
              </div>
            </div>
          </Card>

          <Card title="Shortcuts">
            <div className="adm-list">
              <button
                type="button"
                className="adm-attention"
                style={{ border: 0, background: "transparent", width: "100%", textAlign: "left", cursor: "pointer", font: "inherit" }}
                onClick={() => onNavigate("export")}
              >
                <span className="adm-attention__icon" data-tone="ok">
                  <DatabaseBackup />
                </span>
                <span className="adm-attention__body">
                  <span className="adm-attention__title">Download a database backup</span>
                  <span className="adm-attention__text">JSON or SQL, with secrets redacted.</span>
                </span>
                <span className="adm-attention__go">
                  <ArrowUpRight />
                </span>
              </button>
              <button
                type="button"
                className="adm-attention"
                style={{ border: 0, background: "transparent", width: "100%", textAlign: "left", cursor: "pointer", font: "inherit" }}
                onClick={() => onNavigate("styles")}
              >
                <span className="adm-attention__icon" style={{ background: "var(--adm-accent-soft)", color: "var(--adm-accent)" }}>
                  <Sparkles />
                </span>
                <span className="adm-attention__body">
                  <span className="adm-attention__title">Review answer styles</span>
                  <span className="adm-attention__text">
                    Change how Genova words answers for each specialty.
                  </span>
                </span>
                <span className="adm-attention__go">
                  <ArrowUpRight />
                </span>
              </button>
            </div>
          </Card>
        </div>
      </div>
    </>
  );
}

/* ----------------------------------------------------------------- users */

function UsersSection({
  users,
  loading,
  filtered,
  visible,
  query,
  roleFilter,
  verifiedFilter,
  sort,
  page,
  pageCount,
  adminCount,
  onQuery,
  onRole,
  onVerified,
  onSort,
  onPage,
  onOpen,
}: {
  users: AdminUser[];
  loading: boolean;
  filtered: AdminUser[];
  visible: AdminUser[];
  query: string;
  roleFilter: "all" | "admin" | "user";
  verifiedFilter: "all" | "yes" | "no";
  sort: "newest" | "oldest" | "name";
  page: number;
  pageCount: number;
  adminCount: number;
  onQuery: (v: string) => void;
  onRole: (v: "all" | "admin" | "user") => void;
  onVerified: (v: "all" | "yes" | "no") => void;
  onSort: (v: "newest" | "oldest" | "name") => void;
  onPage: (page: number) => void;
  onOpen: (user: AdminUser) => void;
}) {
  const verifiedCount = users.filter((u) => u.is_verified).length;
  const newThisMonth = users.filter((u) => {
    const d = new Date(u.created_at);
    const now = new Date();
    return d.getMonth() === now.getMonth() && d.getFullYear() === now.getFullYear();
  }).length;

  return (
    <>
      <PageHeader
        title="Users"
        description="Everyone with an account, what they can access, and when they joined."
      />

      <StatGrid>
        <Stat
          label="Accounts"
          icon={<UsersIcon />}
          value={users.length.toLocaleString()}
          caption="People who can sign in"
        />
        <Stat
          label="Administrators"
          icon={<ShieldCheck />}
          value={adminCount.toLocaleString()}
          caption="Full access to this console"
        />
        <Stat
          label="Verified emails"
          icon={<CheckCircle2 />}
          value={verifiedCount.toLocaleString()}
          caption={`${users.length - verifiedCount} still unverified`}
        />
        <Stat
          label="Joined this month"
          icon={<Clock />}
          value={newThisMonth.toLocaleString()}
          caption="New accounts, not activity"
        />
      </StatGrid>

      <div style={{ marginTop: 20 }}>
        <div className="adm-toolbar">
          <SearchInput
            label="Search people"
            value={query}
            onChange={onQuery}
            placeholder="Search name or email"
          />
          <Field label="Role" htmlFor="role-filter">
            <SelectInput
              id="role-filter"
              value={roleFilter}
              onChange={(e) => onRole(e.target.value as "all" | "admin" | "user")}
              style={{ width: 150 }}
            >
              <option value="all">All roles</option>
              <option value="admin">Administrators</option>
              <option value="user">Members</option>
            </SelectInput>
          </Field>
          <Field label="Email" htmlFor="verified-filter">
            <SelectInput
              id="verified-filter"
              value={verifiedFilter}
              onChange={(e) => onVerified(e.target.value as "all" | "yes" | "no")}
              style={{ width: 150 }}
            >
              <option value="all">Any status</option>
              <option value="yes">Verified</option>
              <option value="no">Unverified</option>
            </SelectInput>
          </Field>
          <Field label="Sort" htmlFor="user-sort">
            <SelectInput
              id="user-sort"
              value={sort}
              onChange={(e) => onSort(e.target.value as "newest" | "oldest" | "name")}
              style={{ width: 160 }}
            >
              <option value="newest">Newest first</option>
              <option value="oldest">Oldest first</option>
              <option value="name">Name A–Z</option>
            </SelectInput>
          </Field>
          <span className="adm-toolbar__spacer" />
          <span className="adm-toolbar__meta">
            {filtered.length.toLocaleString()} of {users.length.toLocaleString()} people
          </span>
        </div>

        <Card>
          {loading && !users.length ? (
            <TableSkeleton rows={6} />
          ) : visible.length ? (
            <TableWrap height="auto">
              <table className="adm-table">
                <thead>
                  <tr>
                    <th>Person</th>
                    <th>Role</th>
                    <th className="adm-col-sm">Email status</th>
                    <th className="adm-col-md">Joined</th>
                  </tr>
                </thead>
                <tbody>
                  {visible.map((u) => (
                    <tr
                      key={u.id}
                      data-clickable="true"
                      onClick={() => onOpen(u)}
                      tabIndex={0}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") onOpen(u);
                      }}
                    >
                      <td>
                        <div className="adm-cell-title">
                          <span className="adm-avatar" style={{ width: 28, height: 28 }}>
                            {initialsOf(u.display_name || u.username || u.email)}
                          </span>
                          <span className="adm-cell-stack">
                            <span className="adm-cell-main">
                              {u.display_name || u.username}
                            </span>
                            <span className="adm-cell-sub">{u.email}</span>
                          </span>
                        </div>
                      </td>
                      <td>
                        <Badge tone={isAdminUser(u) ? "accent" : "neutral"}>
                          {isAdminUser(u) ? "Administrator" : "Member"}
                        </Badge>
                      </td>
                      <td className="adm-col-sm">
                        <span
                          style={{
                            fontSize: 12.5,
                            color: u.is_verified ? "var(--adm-muted)" : "var(--adm-warn)",
                          }}
                        >
                          {u.is_verified ? "Verified" : "Unverified"}
                        </span>
                      </td>
                      <td className="adm-col-md adm-muted" style={{ fontSize: 12.5 }}>
                        {relativeDay(u.created_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </TableWrap>
          ) : (
            <EmptyState
              icon={<Search />}
              title="No people match these filters"
              text="Try a different name, or clear the role and email filters."
              action={
                <AdminButton
                  size="sm"
                  variant="secondary"
                  onClick={() => {
                    onQuery("");
                    onRole("all");
                    onVerified("all");
                  }}
                >
                  Clear filters
                </AdminButton>
              }
            />
          )}
          {visible.length ? (
            <div className="adm-card__foot">
              <span>
                Showing {(page - 1) * 15 + 1}–{Math.min(page * 15, filtered.length)} of{" "}
                {filtered.length}
              </span>
              <div className="adm-pager">
                <AdminButton
                  size="sm"
                  variant="secondary"
                  disabled={page <= 1}
                  onClick={() => onPage(page - 1)}
                >
                  Previous
                </AdminButton>
                <span className="adm-pager__label">
                  Page {page} of {pageCount}
                </span>
                <AdminButton
                  size="sm"
                  variant="secondary"
                  disabled={page >= pageCount}
                  onClick={() => onPage(page + 1)}
                >
                  Next
                </AdminButton>
              </div>
            </div>
          ) : null}
        </Card>
      </div>
    </>
  );
}

/* ----------------------------------------------------------------- usage */

function UsageSection({
  usage,
  daily,
  events,
  metric,
  loading,
  page,
  pages,
  total,
  userLabel,
  onMetric,
  onPage,
}: {
  usage: UsageSummary | null;
  daily: UsageDailyPoint[];
  events: UsageEvent[];
  metric: "total_tokens" | "estimated_cost_usd";
  loading: boolean;
  page: number;
  pages: number;
  total: number;
  userLabel: (id?: string | null) => string;
  onMetric: (m: "total_tokens" | "estimated_cost_usd") => void;
  onPage: (page: number) => void;
}) {
  const modelTotal = usage?.total_tokens || 1;

  return (
    <>
      <PageHeader
        title="AI usage"
        description="How much AI capacity the platform used, and what that usage is estimated to cost."
      />

      {usage ? (
        <StatGrid>
          <Stat
            label="Estimated cost"
            icon={<Sparkles />}
            value={formatUsd(usage.estimated_cost_usd)}
            caption={`Today ${formatUsd(usage.today.estimated_cost_usd)} · this month ${formatUsd(usage.month_to_date.estimated_cost_usd)}`}
          />
          <Stat
            label="Total tokens"
            icon={<Files />}
            value={usage.total_tokens.toLocaleString()}
            caption={`${usage.input_tokens.toLocaleString()} sent · ${usage.output_tokens.toLocaleString()} received`}
          />
          <Stat
            label="Requests"
            icon={<RefreshCw />}
            value={usage.request_count.toLocaleString()}
            caption={`Today ${usage.today.request_count} · this month ${usage.month_to_date.request_count}`}
          />
          <Stat
            label="People using AI"
            icon={<UsersIcon />}
            value={String(usage.by_user?.length || 0)}
            caption="Accounts with recorded usage"
          />
        </StatGrid>
      ) : loading ? (
        <TableSkeleton rows={2} />
      ) : (
        <EmptyState title="Usage data is unavailable" text="Try refreshing the page." />
      )}

      <div className="adm-grid-main" style={{ marginTop: 16 }}>
        <Card
          title={metric === "total_tokens" ? "Daily tokens" : "Daily estimated cost"}
          description="Platform-wide volume by day (UTC). Hover a bar for the exact figure."
          action={
            <div className="adm-tabs" role="tablist" aria-label="Chart metric">
              <button
                type="button"
                role="tab"
                className="adm-tab"
                aria-selected={metric === "total_tokens"}
                onClick={() => onMetric("total_tokens")}
              >
                Tokens
              </button>
              <button
                type="button"
                role="tab"
                className="adm-tab"
                aria-selected={metric === "estimated_cost_usd"}
                onClick={() => onMetric("estimated_cost_usd")}
              >
                Cost
              </button>
            </div>
          }
        >
          <BarChart
            points={daily}
            valueKey={metric}
            formatValue={(v) =>
              metric === "total_tokens" ? `${v.toLocaleString()} tokens` : formatUsd(v)
            }
            ariaLabel="Daily AI usage"
          />
        </Card>

        <Card title="Where usage comes from" description="Share of tokens by provider.">
          {(usage?.by_provider || []).length ? (
            <div className="adm-card__body--pad adm-stack" style={{ gap: 14 }}>
              {usage?.by_provider.map((row) => {
                const pct = Math.round((row.total_tokens / modelTotal) * 100);
                return (
                  <div key={row.provider}>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        gap: 10,
                        fontSize: 13,
                        marginBottom: 6,
                      }}
                    >
                      <span style={{ fontWeight: 500 }}>{primaryProviderName(row.provider)}</span>
                      <span className="adm-muted" style={{ fontSize: 12 }}>
                        {pct}% · {formatUsd(row.estimated_cost_usd)}
                      </span>
                    </div>
                    <div className="adm-meter">
                      <div
                        className="adm-meter__fill"
                        style={{ width: `${Math.max(2, pct)}%` }}
                        data-tone={row.provider === "ollama" ? "muted" : undefined}
                      />
                    </div>
                    <div className="adm-hint" style={{ marginTop: 4 }}>
                      {row.total_tokens.toLocaleString()} tokens · {row.request_count.toLocaleString()}{" "}
                      requests
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <EmptyState title="No provider usage yet" text="Usage is recorded per provider as soon as answers are generated." />
          )}
        </Card>
      </div>

      <div className="adm-grid-2" style={{ marginTop: 16 }}>
        <Card title="Models used" description="Tokens and estimated cost by model.">
          {usage?.by_model?.length ? (
            <TableWrap height="short">
              <table className="adm-table" style={{ minWidth: 420 }}>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th data-align="right">Tokens</th>
                    <th data-align="right">Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {usage.by_model.map((row) => (
                    <tr key={`${row.provider}-${row.model_id}`}>
                      <td>
                        <div className="adm-cell-stack">
                          <span className="adm-cell-main" style={{ maxWidth: 220 }}>
                            {row.model_id || "Unknown model"}
                          </span>
                          <span className="adm-cell-sub">
                            {primaryProviderName(row.provider)}
                          </span>
                        </div>
                      </td>
                      <td data-align="right" className="adm-num">
                        {row.total_tokens.toLocaleString()}
                      </td>
                      <td data-align="right" className="adm-num">
                        {formatUsd(row.estimated_cost_usd)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </TableWrap>
          ) : (
            <EmptyState title="No models recorded yet" />
          )}
        </Card>

        <Card title="Top people by tokens" description="Who is using the most AI capacity.">
          {usage?.by_user?.length ? (
            <TableWrap height="short">
              <table className="adm-table" style={{ minWidth: 420 }}>
                <thead>
                  <tr>
                    <th>Person</th>
                    <th data-align="right">Tokens</th>
                    <th data-align="right">Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {usage.by_user.slice(0, 20).map((row) => (
                    <tr key={row.user_id}>
                      <td>{userLabel(row.user_id)}</td>
                      <td data-align="right" className="adm-num">
                        {row.total_tokens.toLocaleString()}
                      </td>
                      <td data-align="right" className="adm-num">
                        {formatUsd(row.estimated_cost_usd)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </TableWrap>
          ) : (
            <EmptyState title="No per-person usage yet" />
          )}
        </Card>
      </div>

      <div style={{ marginTop: 16 }}>
        <Card
          title="Recent requests"
          description="Every AI request, newest first, with the estimated cost of each."
          footer={
            <>
              <span>
                {total.toLocaleString()} requests · page {page} of {Math.max(1, pages)}
              </span>
              <div className="adm-pager">
                <AdminButton
                  size="sm"
                  variant="secondary"
                  disabled={page <= 1}
                  onClick={() => onPage(page - 1)}
                >
                  Previous
                </AdminButton>
                <AdminButton
                  size="sm"
                  variant="secondary"
                  disabled={page >= pages}
                  onClick={() => onPage(page + 1)}
                >
                  Next
                </AdminButton>
              </div>
            </>
          }
        >
          {events.length ? (
            <TableWrap height="short">
              <table className="adm-table">
                <thead>
                  <tr>
                    <th>When</th>
                    <th>Person</th>
                    <th className="adm-col-md">Model</th>
                    <th data-align="right">Tokens</th>
                    <th data-align="right" className="adm-col-sm">
                      Est. cost
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((ev) => (
                    <tr key={ev.id}>
                      <td className="adm-muted" style={{ fontSize: 12.5 }}>
                        {dateTime(ev.created_at)}
                      </td>
                      <td>{userLabel(ev.user_id)}</td>
                      <td className="adm-col-md">
                        <div className="adm-cell-stack">
                          <span className="adm-cell-main" style={{ maxWidth: 200 }}>
                            {ev.model_id || "—"}
                          </span>
                          <span className="adm-cell-sub">{primaryProviderName(ev.provider)}</span>
                        </div>
                      </td>
                      <td data-align="right" className="adm-num">
                        {(ev.total_tokens ?? 0).toLocaleString()}
                      </td>
                      <td data-align="right" className="adm-num adm-col-sm">
                        {formatUsd(ev.estimated_cost_usd)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </TableWrap>
          ) : (
            <EmptyState title="No requests recorded yet" />
          )}
        </Card>

        {usage?.cost_disclaimer ? (
          <div style={{ marginTop: 12 }}>
            <Callout icon={<Info />}>{usage.cost_disclaimer}</Callout>
          </div>
        ) : null}
      </div>
    </>
  );
}

/* ------------------------------------------------------------- documents */

function DocumentsSection({
  docs,
  loading,
  counts,
  ready,
  failed,
  pending,
  status,
  query,
  filtered,
  onStatus,
  onQuery,
  onOpen,
  onClear,
}: {
  docs: AdminDocumentsResponse | null;
  loading: boolean;
  counts: Record<string, number>;
  ready: number;
  failed: number;
  pending: number;
  status: string;
  query: string;
  filtered: AdminDocumentsResponse["items"];
  onStatus: (status: string) => void;
  onQuery: (q: string) => void;
  onOpen: (doc: AdminDocumentsResponse["items"][number]) => void;
  onClear: () => void;
}) {
  const total = docs?.total ?? 0;
  const tabs = [
    { id: "all", label: "All", count: total },
    { id: "ready", label: "Ready", count: ready },
    { id: "processing", label: "Processing", count: pending },
    { id: "failed", label: "Needs attention", count: failed },
  ];

  return (
    <>
      <PageHeader
        title="Documents"
        description="Every file Genova has indexed, plus anything that is still processing or needs attention."
        actions={
          docs && docs.items.length ? (
            <AdminButton variant="secondary" onClick={onClear}>
              <RefreshCw /> Reset filters
            </AdminButton>
          ) : undefined
        }
      />

      <StatGrid>
        <Stat label="Ready to use" icon={<CheckCircle2 />} value={ready.toLocaleString()} caption="Searchable by Genova" />
        <Stat label="Processing" icon={<Clock />} value={pending.toLocaleString()} caption="Waiting or being indexed" />
        <Stat
          label="Needs attention"
          icon={<AlertTriangle />}
          value={failed.toLocaleString()}
          attention={failed > 0}
          caption={failed > 0 ? "Open a file to see why" : "Nothing failed"}
        />
        <Stat label="Total documents" icon={<Files />} value={total.toLocaleString()} caption="Across all accounts" />
      </StatGrid>

      <div style={{ marginTop: 20 }}>
        <div className="adm-toolbar">
          <div className="adm-tabs" role="tablist" aria-label="Document status">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                role="tab"
                className="adm-tab"
                aria-selected={status === tab.id}
                onClick={() => onStatus(tab.id)}
              >
                {tab.label}
                <span className="adm-tab__count">{tab.count}</span>
              </button>
            ))}
          </div>
          <span className="adm-toolbar__spacer" />
          <SearchInput
            label="Search documents"
            value={query}
            onChange={onQuery}
            placeholder="Search file name or owner"
          />
        </div>

        <Card
          title={
            status === "all"
              ? "All documents"
              : status === "failed"
                ? "Documents that need attention"
                : status === "processing"
                  ? "Documents in progress"
                  : "Ready documents"
          }
          description={
            docs
              ? `Showing ${filtered.length} of ${docs.items.length} loaded documents${
                  query ? ` matching “${query}”` : ""
                }.`
              : "Loading documents…"
          }
          footer={
            <span>
              The console loads the 100 most recent documents. Older files can still be found with
              search once loaded.
            </span>
          }
        >
          {loading && !docs ? (
            <TableSkeleton rows={6} />
          ) : filtered.length ? (
            <TableWrap height="auto">
              <table className="adm-table">
                <thead>
                  <tr>
                    <th>Document</th>
                    <th className="adm-col-md">Owner</th>
                    <th>Status</th>
                    <th data-align="right" className="adm-col-sm">
                      Passages
                    </th>
                    <th className="adm-col-md">Uploaded</th>
                    <th aria-label="Actions" />
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((doc) => (
                    <tr
                      key={doc.id}
                      data-clickable="true"
                      onClick={() => onOpen(doc)}
                      tabIndex={0}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") onOpen(doc);
                      }}
                    >
                      <td>
                        <div className="adm-cell-title">
                          <span className="adm-cell-icon">
                            <FileText />
                          </span>
                          <span className="adm-cell-stack">
                            <span className="adm-cell-main">{doc.original_name}</span>
                            <span className="adm-cell-sub">
                              {doc.page_count ?? 0} pages
                              {doc.error_message ? ` · ${doc.error_message}` : ""}
                            </span>
                          </span>
                        </div>
                      </td>
                      <td className="adm-col-md">
                        <div className="adm-cell-stack">
                          <span className="adm-cell-main" style={{ fontWeight: 450 }}>
                            {doc.username || "—"}
                          </span>
                          <span className="adm-cell-sub">{doc.email || ""}</span>
                        </div>
                      </td>
                      <td>
                        <Badge tone={toneForStatus(doc.status)}>
                          {documentStatusLabel(doc.status)}
                        </Badge>
                      </td>
                      <td data-align="right" className="adm-num adm-col-sm">
                        {doc.chunk_count ?? 0}
                      </td>
                      <td className="adm-col-md adm-muted" style={{ fontSize: 12.5 }}>
                        {relativeDay(doc.uploaded_at)}
                      </td>
                      <td data-align="right">
                        <span className="adm-link">Details</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </TableWrap>
          ) : (
            <EmptyState
              icon={status === "failed" ? <CheckCircle2 /> : <Files />}
              title={
                status === "failed"
                  ? "No failed documents"
                  : query
                    ? "No documents match your search"
                    : "No documents in this view"
              }
              text={
                status === "failed"
                  ? "Every uploaded file was read successfully."
                  : query
                    ? "Try part of the file name or the owner’s email."
                    : "Upload a file from the workspace and it will appear here."
              }
              action={
                status !== "all" || query ? (
                  <AdminButton size="sm" variant="secondary" onClick={onClear}>
                    Show all documents
                  </AdminButton>
                ) : undefined
              }
            />
          )}
        </Card>

        <div style={{ marginTop: 12 }}>
          <Callout icon={<Info />}>
            <strong>What processing means:</strong> Genova reads the file, splits it into passages,
            and makes those passages searchable. Failed files stay stored — nothing is deleted.
          </Callout>
        </div>
      </div>
    </>
  );
}

/* ----------------------------------------------------------- connections */

function ConnectionsSection({
  drive,
  workspaces,
  apiKeys,
  loading,
}: {
  drive: { total: number; items: Array<Record<string, unknown>> } | null;
  workspaces: { total: number; items: Array<Record<string, unknown>> } | null;
  apiKeys: {
    total_keys: number;
    active_keys: number;
    by_user: Array<Record<string, unknown>>;
  } | null;
  loading: boolean;
}) {
  return (
    <>
      <PageHeader
        title="Connections"
        description="Linked Google Drive folders, shared workspaces, and API keys issued to accounts."
      />

      <StatGrid>
        <Stat
          label="Drive folders"
          icon={<HardDrive />}
          value={String(drive?.total ?? 0)}
          caption="Folders linked for sync"
        />
        <Stat
          label="Workspaces"
          icon={<UsersIcon />}
          value={String(workspaces?.total ?? 0)}
          caption="Shared spaces for documents"
        />
        <Stat
          label="Active API keys"
          icon={<KeyRound />}
          value={apiKeys ? `${apiKeys.active_keys}/${apiKeys.total_keys}` : "—"}
          caption="Active out of all issued"
        />
        <Stat
          label="Key holders"
          icon={<UsersIcon />}
          value={String(apiKeys?.by_user?.length ?? 0)}
          caption="People with at least one key"
        />
      </StatGrid>

      <div style={{ marginTop: 16 }} className="adm-stack">
        <Card
          title="Google Drive"
          description="Folders Genova watches for new files. Credentials are never shown here."
        >
          {loading && !drive ? (
            <TableSkeleton rows={3} />
          ) : drive?.items?.length ? (
            <TableWrap height="auto">
              <table className="adm-table">
                <thead>
                  <tr>
                    <th>Folder</th>
                    <th className="adm-col-md">Linked by</th>
                    <th className="adm-col-sm">Access</th>
                    <th>Last sync</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {drive.items.map((row) => (
                    <tr key={String(row.id)}>
                      <td>
                        <div className="adm-cell-title">
                          <span className="adm-cell-icon">
                            <HardDrive />
                          </span>
                          <span className="adm-cell-stack">
                            <span className="adm-cell-main">
                              {row.folder_name && String(row.folder_name) !== String(row.folder_id)
                                ? String(row.folder_name)
                                : "Drive folder"}
                            </span>
                            <span className="adm-cell-sub">{String(row.folder_id || "")}</span>
                          </span>
                        </div>
                      </td>
                      <td className="adm-col-md">{String(row.username || row.email || "—")}</td>
                      <td className="adm-col-sm">
                        {String(row.access_mode || "—") === "public" ? "Public link" : "Private"}
                      </td>
                      <td className="adm-muted" style={{ fontSize: 12.5 }}>
                        {relativeDay(row.last_synced_at as string | undefined)}
                      </td>
                      <td>
                        <Badge tone={toneForStatus(String(row.sync_status || ""))}>
                          {String(row.sync_status || "unknown") === "ok"
                            ? "Connected"
                            : String(row.sync_status || "Unknown")}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </TableWrap>
          ) : (
            <EmptyState
              icon={<HardDrive />}
              title="No Drive folders linked"
              text="People can link a Google Drive folder from their workspace settings."
            />
          )}
        </Card>

        <div className="adm-grid-2">
          <Card title="Workspaces" description="Shared spaces that group documents and members.">
            {workspaces?.items?.length ? (
              <TableWrap height="auto">
                <table className="adm-table" style={{ minWidth: 360 }}>
                  <thead>
                    <tr>
                      <th>Workspace</th>
                      <th>Creator</th>
                      <th data-align="right">Members</th>
                    </tr>
                  </thead>
                  <tbody>
                    {workspaces.items.map((ws) => (
                      <tr key={String(ws.id)}>
                        <td style={{ fontWeight: 500 }}>{String(ws.name)}</td>
                        <td className="adm-muted">
                          {String(ws.creator_username || ws.created_by || "—")}
                        </td>
                        <td data-align="right" className="adm-num">
                          {String(ws.member_count ?? 0)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </TableWrap>
            ) : (
              <EmptyState
                icon={<UsersIcon />}
                title="No shared workspaces"
                text="Workspaces appear when someone creates one in the app."
              />
            )}
          </Card>

          <Card title="API keys" description="Issued to people for programmatic access.">
            {apiKeys?.by_user?.length ? (
              <TableWrap height="auto">
                <table className="adm-table" style={{ minWidth: 360 }}>
                  <thead>
                    <tr>
                      <th>Person</th>
                      <th data-align="right">Active</th>
                      <th data-align="right">Total</th>
                      <th className="adm-col-md">Last used</th>
                    </tr>
                  </thead>
                  <tbody>
                    {apiKeys.by_user.map((row) => (
                      <tr key={String(row.user_id)}>
                        <td>
                          <div className="adm-cell-stack">
                            <span className="adm-cell-main" style={{ fontWeight: 450 }}>
                              {String(row.username || row.email || "—")}
                            </span>
                            <span className="adm-cell-sub">{String(row.email || "")}</span>
                          </div>
                        </td>
                        <td data-align="right" className="adm-num">
                          {String(row.active_keys ?? 0)}
                        </td>
                        <td data-align="right" className="adm-num">
                          {String(row.total_keys ?? 0)}
                        </td>
                        <td className="adm-col-md adm-muted" style={{ fontSize: 12.5 }}>
                          {row.last_used_at ? relativeDay(String(row.last_used_at)) : "Never"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </TableWrap>
            ) : (
              <EmptyState icon={<KeyRound />} title="No API keys issued" />
            )}
          </Card>
        </div>

        <Callout icon={<KeyRound />}>
          <strong>Secrets stay private.</strong> Key values and tokens are never rendered in this
          console — only counts and ownership are shown.
        </Callout>
      </div>
    </>
  );
}

/* ---------------------------------------------------------------- export */

function ExportSection({
  userEmail,
  exporting,
  onChoose,
}: {
  userEmail: string;
  exporting: "json" | "sql" | null;
  onChoose: (format: "json" | "sql") => void;
}) {
  return (
    <>
      <PageHeader
        title="Data export"
        description="Download a copy of the platform database for support, migration, or engineering work."
      />

      <Card
        title="What’s included"
        description="Every table in the platform, minus the values that could compromise accounts."
      >
        <div className="adm-card__body--pad adm-stack" style={{ gap: 12 }}>
          <DefinitionList
            items={[
              { term: "Accounts", value: "Usernames, emails, roles, and join dates" },
              { term: "Documents", value: "File names, owners, processing status, and errors" },
              {
                term: "Hidden for safety",
                value: "Passwords, API key secrets, and Google Drive refresh tokens",
              },
            ]}
          />
          <div className="adm-hint">
            Exports are generated on the server and are only available to administrators. Signed in
            as {userEmail}.
          </div>
        </div>
      </Card>

      <div className="adm-grid-2" style={{ marginTop: 16 }}>
        <Card
          title="JSON backup"
          description="Readable, structured copy. Best for support, audits, or moving data to another tool."
        >
          <div className="adm-card__body--pad">
            <AdminButton
              variant="primary"
              onClick={() => onChoose("json")}
              disabled={exporting !== null}
            >
              <Download /> {exporting === "json" ? "Preparing…" : "Download JSON"}
            </AdminButton>
          </div>
        </Card>

        <Card
          title="SQL backup"
          description="A database dump for engineers. Best for restoring or inspecting the schema."
        >
          <div className="adm-card__body--pad">
            <AdminButton
              variant="secondary"
              onClick={() => onChoose("sql")}
              disabled={exporting !== null}
            >
              <DatabaseBackup /> {exporting === "sql" ? "Preparing…" : "Download SQL"}
            </AdminButton>
          </div>
        </Card>
      </div>

      <div style={{ marginTop: 16 }}>
        <Callout tone="warn" icon={<ShieldCheck />}>
          <strong>Handle exports carefully.</strong> They contain personal data even though secrets
          are redacted. Store them somewhere approved and delete them when finished.
        </Callout>
      </div>
    </>
  );
}

/* ---------------------------------------------------------------- status */

function StatusSection({
  health,
  loading,
  lastUpdated,
}: {
  health: HealthStatus | null;
  loading: boolean;
  lastUpdated: Date | null;
}) {
  const parts = healthParts(health);

  const services: Array<{
    id: string;
    name: string;
    desc: string;
    icon: ReactNode;
    info?: HealthComponent;
  }> = [
    {
      id: "database",
      name: "Database",
      desc: "Stores accounts, documents, conversations, and settings.",
      icon: <Database />,
      info: parts.database,
    },
    {
      id: "chromadb",
      name: "Search index",
      desc: "Finds the document passages each answer is built from.",
      icon: <Search />,
      info: parts.chromadb,
    },
    {
      id: "redis",
      name: "Cache",
      desc: "Speeds up repeated work. Optional — answers still work without it.",
      icon: <HardDrive />,
      info: parts.redis,
    },
    {
      id: "celery",
      name: "Background jobs",
      desc: "Runs long uploads in separate workers when configured.",
      icon: <RefreshCw />,
      info: parts.celery,
    },
  ];

  const down = services.filter(
    (s) => ["down", "unavailable"].includes((s.info?.status || "").toLowerCase()),
  );

  return (
    <>
      <PageHeader
        title="System status"
        description="Live checks on the services Genova depends on, explained in plain language."
      />

      <div style={{ marginBottom: 16 }}>
        {health?.overall?.toLowerCase() === "healthy" ? (
          <Callout icon={<CheckCircle2 />}>
            <strong>Genova reports healthy.</strong> The database and search index are responding.
            {down.length
              ? ` ${down.map((s) => s.name).join(" and ")} ${
                  down.length === 1 ? "is" : "are"
                } unavailable, but answers can still be produced without ${
                  down.length === 1 ? "it" : "them"
                }.`
              : " Every configured service is responding."}
          </Callout>
        ) : (
          <Callout tone="warn" icon={<AlertTriangle />}>
            <strong>Genova is reporting a problem.</strong>{" "}
            {down.length
              ? `${down.map((s) => s.name).join(", ")} need attention.`
              : "Check the service details below."}
          </Callout>
        )}
      </div>

      {loading && !health ? (
        <div className="adm-service-grid" aria-hidden>
          {[1, 2, 3, 4].map((i) => (
            <div className="adm-card adm-service-card" key={i}>
              <div className="adm-service-card__top">
                <span className="adm-skel" style={{ width: 32, height: 32, borderRadius: 9 }} />
                <span className="adm-skel" style={{ width: 58, height: 16, borderRadius: 999 }} />
              </div>
              <span className="adm-skel" style={{ height: 13, width: "55%", marginTop: 4 }} />
              <span className="adm-skel" style={{ height: 10, width: "90%" }} />
              <span className="adm-skel" style={{ height: 10, width: "70%" }} />
            </div>
          ))}
        </div>
      ) : (
        <div className="adm-service-grid">
          {services.map((service) => {
            const status = service.info?.status || "unknown";
            const tone = toneForStatus(status);
            return (
              <section className="adm-card adm-service-card" key={service.id}>
                <div className="adm-service-card__top">
                  <span className="adm-service-card__icon" data-tone={tone}>
                    {service.icon}
                  </span>
                  <Badge tone={tone}>{serviceState(status)}</Badge>
                </div>
                <h3 className="adm-service-card__name">
                  {service.name}
                  <Tooltip text={service.desc} />
                </h3>
                <p className="adm-service-card__desc">{service.desc}</p>
                <div
                  className="adm-service-card__foot"
                  data-tone={service.info?.error ? "danger" : undefined}
                  title={service.info?.detail || service.info?.error}
                >
                  <span className="adm-service-card__dot" data-tone={tone} />
                  <span className="adm-service-card__foot-text">
                    {typeof service.info?.latency_ms === "number"
                      ? `${Math.round(service.info.latency_ms)} ms`
                      : service.info?.error || service.info?.detail || "No response time"}
                  </span>
                </div>
              </section>
            );
          })}
        </div>
      )}

      <div className="adm-hint" style={{ marginTop: 10 }}>
        {lastUpdated ? `Last checked ${lastUpdated.toLocaleTimeString()}` : "Checking…"}
      </div>

      <div style={{ marginTop: 16 }}>
        <Callout icon={<Info />}>
          A service marked “Not in use” is intentionally disabled — it is not a failure. Only the
          database and search index are required for answers.
        </Callout>
      </div>
    </>
  );
}

export default function AdminPage() {
  return (
    <RequireAuth requireAdmin>
      <AdminConsole />
    </RequireAuth>
  );
}

"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Activity,
  ArrowLeft,
  Database,
  FileText,
  HardDrive,
  HeartPulse,
  KeyRound,
  MessageSquareText,
  Network,
  RefreshCw,
  Shield,
  Sparkles,
  Users,
} from "lucide-react";

import { api, CONNECTION_ERROR_MESSAGE } from "@/lib/api";
import { formatBytes, formatTime, formatUsd, isAdminUser } from "@/lib/admin";
import { useAuth } from "@/lib/auth";
import { RequireAuth } from "@/components/auth/RequireAuth";
import { PromptManagementPanel } from "@/components/admin/PromptManagementPanel";
import {
  LedgerBarChart,
  LedgerButton,
  LedgerCanvas,
  LedgerEmpty,
  LedgerError,
  LedgerKpiStrip,
  LedgerMobileBar,
  LedgerPanel,
  LedgerScroll,
  LedgerScrollTable,
  LedgerShell,
  LedgerSheet,
  LedgerSidebar,
  LedgerStatus,
  LedgerTitle,
  type LedgerNavItem,
} from "@/components/ledger/LedgerPrimitives";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";

type Section =
  | "overview"
  | "users"
  | "usage"
  | "rag"
  | "prompts"
  | "integrations"
  | "database"
  | "health";

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
  month_to_date: {
    request_count: number;
    total_tokens: number;
    estimated_cost_usd: number;
  };
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
  }>;
}

interface HealthStatus {
  overall?: string;
  components?: {
    database?: Record<string, unknown>;
    redis?: Record<string, unknown>;
    celery?: Record<string, unknown>;
    chromadb?: Record<string, unknown>;
    ollama?: Record<string, unknown>;
  };
  database?: Record<string, unknown>;
  redis?: Record<string, unknown>;
  celery?: Record<string, unknown>;
  chromadb?: Record<string, unknown>;
  ollama?: Record<string, unknown>;
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

const NAV: LedgerNavItem[] = [
  { id: "overview", label: "Overview", icon: <Activity /> },
  { id: "users", label: "Users", icon: <Users /> },
  { id: "usage", label: "AI Usage", icon: <Sparkles /> },
  { id: "rag", label: "RAG", icon: <FileText /> },
  { id: "prompts", label: "Prompt Management", icon: <MessageSquareText /> },
  { id: "integrations", label: "Integrations", icon: <Network /> },
  { id: "database", label: "Database", icon: <Database /> },
  { id: "health", label: "Health", icon: <HeartPulse /> },
];

function statusTone(status: string): "ok" | "bad" | "warn" | "neutral" {
  const s = status.toLowerCase();
  if (["ready", "up", "ok", "healthy", "completed", "active"].includes(s)) return "ok";
  if (["failed", "down", "error", "unhealthy"].includes(s)) return "bad";
  if (["degraded", "processing", "pending", "warning"].includes(s)) return "warn";
  return "neutral";
}

function AdminPageContent() {
  const { user } = useAuth();
  const router = useRouter();
  const [section, setSection] = useState<Section>("overview");
  const [menuOpen, setMenuOpen] = useState(false);
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [userQuery, setUserQuery] = useState("");
  const [roleFilter, setRoleFilter] = useState<"all" | "admin" | "user">("all");
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const [usageDaily, setUsageDaily] = useState<UsageDailyPoint[]>([]);
  const [usageEvents, setUsageEvents] = useState<UsageEvent[]>([]);
  const [docs, setDocs] = useState<AdminDocumentsResponse | null>(null);
  const [docStatus, setDocStatus] = useState("failed");
  const [drive, setDrive] = useState<{
    total: number;
    items: Array<Record<string, unknown>>;
  } | null>(null);
  const [workspaces, setWorkspaces] = useState<{
    total: number;
    items: Array<Record<string, unknown>>;
  } | null>(null);
  const [apiKeys, setApiKeys] = useState<{
    total_keys: number;
    active_keys: number;
    by_user: Array<Record<string, unknown>>;
  } | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [exporting, setExporting] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const loadOverview = useCallback(async () => {
    const [statsData, usersData] = await Promise.all([
      api.get<AdminStats>("/api/v1/admin/stats"),
      api.get<AdminUser[]>("/api/v1/admin/users"),
    ]);
    setStats(statsData);
    setUsers(usersData);
  }, []);

  const loadUsage = useCallback(async () => {
    const [summary, daily, events, usersData] = await Promise.all([
      api.get<UsageSummary>("/api/v1/admin/usage/summary"),
      api.get<{ points: UsageDailyPoint[] }>("/api/v1/admin/usage/daily"),
      api.get<{ items: UsageEvent[] }>("/api/v1/admin/usage/events?limit=50"),
      api.get<AdminUser[]>("/api/v1/admin/users"),
    ]);
    setUsage(summary);
    setUsageDaily(daily.points || []);
    setUsageEvents(events.items || []);
    setUsers(usersData);
  }, []);

  const loadDocs = useCallback(async () => {
    const qs = docStatus
      ? `?status=${encodeURIComponent(docStatus)}&limit=100`
      : "?limit=100";
    setDocs(await api.get<AdminDocumentsResponse>(`/api/v1/admin/documents${qs}`));
  }, [docStatus]);

  const loadIntegrations = useCallback(async () => {
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
    try {
      setLoading(true);
      setError("");
      if (section === "overview" || section === "users" || section === "database") {
        await loadOverview();
        if (section === "overview") {
          try {
            await loadHealth();
          } catch {
            /* health optional on overview */
          }
        }
      }       else if (section === "usage") await loadUsage();
      else if (section === "rag") await loadDocs();
      else if (section === "prompts") {
        /* PromptManagementPanel loads its own data */
      }
      else if (section === "integrations") await loadIntegrations();
      else if (section === "health") await loadHealth();
      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : CONNECTION_ERROR_MESSAGE);
    } finally {
      setLoading(false);
    }
  }, [
    user,
    section,
    loadOverview,
    loadUsage,
    loadDocs,
    loadIntegrations,
    loadHealth,
  ]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const exportDb = async (format: "json" | "sql") => {
    try {
      setExporting(true);
      const blob = await api.getBlob(`/api/v1/admin/export-db?format=${format}`);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `db_backup.${format}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Export failed");
    } finally {
      setExporting(false);
    }
  };

  const userLabel = useMemo(() => {
    const map = new Map(users.map((u) => [u.id, u.username || u.email]));
    return (id?: string | null) => (id ? map.get(id) || id.slice(0, 8) : "—");
  }, [users]);

  const filteredUsers = useMemo(() => {
    const q = userQuery.trim().toLowerCase();
    return users.filter((u) => {
      const role = String(u.role || (u.is_admin ? "admin" : "user")).toLowerCase();
      if (roleFilter === "admin" && role !== "admin" && !u.is_admin) return false;
      if (roleFilter === "user" && (role === "admin" || u.is_admin)) return false;
      if (!q) return true;
      return (
        u.username?.toLowerCase().includes(q) ||
        u.email?.toLowerCase().includes(q) ||
        u.display_name?.toLowerCase().includes(q)
      );
    });
  }, [users, userQuery, roleFilter]);

  const failedDocs = docs?.status_counts?.failed ?? 0;
  const finding = useMemo(() => {
    if (section === "overview") {
      const healthLabel = health?.overall || "status pending";
      return `${stats?.total_users ?? "—"} accounts · ${stats?.total_documents ?? "—"} documents · platform ${healthLabel}${lastUpdated ? ` · updated ${lastUpdated.toLocaleTimeString()}` : ""}`;
    }
    if (section === "usage") {
      return `Estimated ${formatUsd(usage?.estimated_cost_usd)} across ${usage?.request_count?.toLocaleString() ?? "—"} requests · today ${formatUsd(usage?.today.estimated_cost_usd)}`;
    }
    if (section === "rag") {
      return `${docs?.total ?? "—"} indexed documents · ${failedDocs} failed`;
    }
    if (section === "prompts") {
      return "Automatic domain detection · layered Global + Domain + Task prompts";
    }
    if (section === "users") {
      return `${filteredUsers.length} of ${users.length} accounts shown`;
    }
    if (section === "integrations") {
      return `${drive?.total ?? 0} Drive links · ${workspaces?.total ?? 0} workspaces · ${apiKeys?.active_keys ?? 0} active API keys`;
    }
    if (section === "database") {
      return "Credential columns are redacted server-side before download.";
    }
    return health?.overall
      ? `Overall ${health.overall}`
      : "Dependency checks from /api/v1/health/status";
  }, [
    section,
    stats,
    health,
    lastUpdated,
    usage,
    docs,
    failedDocs,
    filteredUsers.length,
    users.length,
    drive,
    workspaces,
    apiKeys,
  ]);

  const selectSection = (id: string) => {
    setSection(id as Section);
    setMenuOpen(false);
  };

  const sidebar = (
    <LedgerSidebar
      brand="Genova"
      scope="Admin desk"
      groupLabel="Operations"
      items={NAV}
      activeId={section}
      onSelect={selectSection}
      footer={
        <LedgerButton
          variant="ghost"
          className="w-full justify-start"
          onClick={() => router.push("/dashboard")}
        >
          <ArrowLeft className="h-4 w-4" />
          Back to workspace
        </LedgerButton>
      }
    />
  );

  return (
    <LedgerShell>
      <div className="ledger-shell">
        {sidebar}
        <LedgerSheet open={menuOpen} onClose={() => setMenuOpen(false)}>
          {sidebar}
        </LedgerSheet>

        <LedgerCanvas>
          <LedgerMobileBar
            title="Admin"
            open={menuOpen}
            onToggle={() => setMenuOpen((v) => !v)}
            action={
              <LedgerButton
                variant="secondary"
                onClick={() => void refresh()}
                disabled={loading}
              >
                <RefreshCw className={loading ? "h-3.5 w-3.5 animate-spin" : "h-3.5 w-3.5"} />
              </LedgerButton>
            }
          />

          <LedgerScroll>
            <LedgerTitle
              title={
                section === "overview"
                  ? "Platform overview"
                  : section === "users"
                    ? "Users"
                    : section === "usage"
                      ? "AI usage"
                      : section === "rag"
                        ? "RAG indexing"
                        : section === "prompts"
                          ? "Prompt Management"
                          : section === "integrations"
                            ? "Integrations"
                            : section === "database"
                              ? "Database"
                              : "System health"
              }
              finding={finding}
              actions={
                <>
                  <LedgerButton
                    variant="secondary"
                    onClick={() => void refresh()}
                    disabled={loading}
                  >
                    <RefreshCw
                      className={loading ? "h-3.5 w-3.5 animate-spin" : "h-3.5 w-3.5"}
                    />
                    Refresh
                  </LedgerButton>
                  {section === "database" || section === "overview" ? (
                    <LedgerButton
                      variant="primary"
                      onClick={() => void exportDb("json")}
                      disabled={exporting}
                    >
                      <HardDrive className="h-3.5 w-3.5" />
                      Export JSON
                    </LedgerButton>
                  ) : null}
                </>
              }
            />

            {error ? <LedgerError>{error}</LedgerError> : null}

            {section === "overview" ? (
              <div className="ledger-stack">
                {loading && !stats ? (
                  <div className="ledger-kpi-strip">
                    {[1, 2, 3, 4].map((i) => (
                      <div key={i} className="ledger-kpi">
                        <Skeleton className="h-3 w-20" />
                        <Skeleton className="mt-3 h-8 w-16" />
                      </div>
                    ))}
                  </div>
                ) : stats ? (
                  <LedgerKpiStrip
                    items={[
                      {
                        label: "Users",
                        value: stats.total_users.toLocaleString(),
                        caption: "Registered accounts",
                      },
                      {
                        label: "Documents",
                        value: stats.total_documents.toLocaleString(),
                        caption: `${stats.total_pdfs_uploaded.toLocaleString()} PDFs uploaded`,
                        emphasis: "data",
                      },
                      {
                        label: "Messages",
                        value: stats.total_messages.toLocaleString(),
                        caption: `${stats.query_count.toLocaleString()} measured queries`,
                      },
                      {
                        label: "Avg response",
                        value: formatTime(stats.average_query_response_time_ms),
                        caption: "In-process latency average",
                      },
                    ]}
                  />
                ) : null}

                <div className="ledger-grid-main-rail">
                  <LedgerPanel
                    title="Disk space"
                    description={`${formatBytes(stats?.disk_space_usage.used_bytes ?? 0)} used of ${formatBytes(stats?.disk_space_usage.total_bytes ?? 0)}`}
                    padded
                  >
                    <div className="mb-3 h-2 overflow-hidden rounded-full bg-[var(--gd-raised)]">
                      <div
                        className="h-full rounded-full bg-[var(--gd-data)]"
                        style={{
                          width: `${Math.min(100, stats?.disk_space_usage.usage_percent ?? 0)}%`,
                        }}
                      />
                    </div>
                    <div className="grid grid-cols-3 gap-3 text-[13px]">
                      <div>
                        <div className="text-[var(--gd-muted)]">Free</div>
                        <div className="font-medium">
                          {formatBytes(stats?.disk_space_usage.free_bytes ?? 0)}
                        </div>
                      </div>
                      <div>
                        <div className="text-[var(--gd-muted)]">Uploads</div>
                        <div className="font-medium">
                          {formatBytes(stats?.disk_space_usage.upload_dir_bytes ?? 0)}
                        </div>
                      </div>
                      <div>
                        <div className="text-[var(--gd-muted)]">Usage</div>
                        <div className="font-medium">
                          {(stats?.disk_space_usage.usage_percent ?? 0).toFixed(2)}%
                        </div>
                      </div>
                    </div>
                  </LedgerPanel>

                  <LedgerPanel title="Health snapshot" padded>
                    <div className="space-y-3 text-[13px]">
                      <div className="flex items-center justify-between">
                        <span className="text-[var(--gd-muted)]">Overall</span>
                        <LedgerStatus
                          label={health?.overall || "—"}
                          tone={statusTone(health?.overall || "")}
                        />
                      </div>
                      {(
                        [
                          ["Database", healthParts(health).database],
                          ["ChromaDB", healthParts(health).chromadb],
                          ["Ollama", healthParts(health).ollama],
                        ] as const
                      ).map(([label, info]) => (
                        <div
                          key={label}
                          className="flex items-center justify-between gap-2"
                        >
                          <span>{label}</span>
                          <LedgerStatus
                            label={String(info?.status || "—")}
                            tone={statusTone(String(info?.status || ""))}
                          />
                        </div>
                      ))}
                      <LedgerButton
                        variant="secondary"
                        className="mt-2 w-full"
                        onClick={() => selectSection("health")}
                      >
                        Open health
                      </LedgerButton>
                    </div>
                  </LedgerPanel>
                </div>
              </div>
            ) : null}

            {section === "users" ? (
              <div className="ledger-stack">
                <div className="ledger-toolbar">
                  <div className="ledger-filters">
                    <div className="ledger-field mb-0">
                      <label htmlFor="user-search">Search</label>
                      <Input
                        id="user-search"
                        value={userQuery}
                        onChange={(e) => setUserQuery(e.target.value)}
                        placeholder="Name or email"
                        className="h-10 w-56 rounded-[10px]"
                      />
                    </div>
                    <div className="ledger-field mb-0">
                      <label htmlFor="role-filter">Role</label>
                      <select
                        id="role-filter"
                        value={roleFilter}
                        onChange={(e) =>
                          setRoleFilter(e.target.value as "all" | "admin" | "user")
                        }
                        className="h-10 rounded-[10px] border border-[#e8e8e8] bg-[var(--gd-surface)] px-3 text-sm"
                      >
                        <option value="all">All roles</option>
                        <option value="admin">Admins</option>
                        <option value="user">Users</option>
                      </select>
                    </div>
                  </div>
                </div>

                <LedgerPanel
                  title="Registered accounts"
                  description="Roles come from backend authorization — passwords are never shown."
                >
                  <LedgerScrollTable>
                    <table className="ledger-table">
                      <thead>
                        <tr>
                          <th>User</th>
                          <th>Email</th>
                          <th>Role</th>
                          <th>Verified</th>
                          <th>Created</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredUsers.map((u) => (
                          <tr key={u.id}>
                            <td className="font-medium">{u.username}</td>
                            <td className="text-[var(--gd-muted)]">{u.email}</td>
                            <td>
                              <LedgerStatus
                                label={u.role || (u.is_admin ? "admin" : "user")}
                                tone={isAdminUser(u) ? "warn" : "neutral"}
                              />
                            </td>
                            <td>{u.is_verified ? "Yes" : "No"}</td>
                            <td className="text-[var(--gd-muted)]">
                              {u.created_at?.slice(0, 10) || "—"}
                            </td>
                          </tr>
                        ))}
                        {!filteredUsers.length ? (
                          <tr>
                            <td colSpan={5}>
                              <LedgerEmpty>No users match this filter.</LedgerEmpty>
                            </td>
                          </tr>
                        ) : null}
                      </tbody>
                    </table>
                  </LedgerScrollTable>
                </LedgerPanel>
              </div>
            ) : null}

            {section === "usage" ? (
              <div className="ledger-stack">
                {usage ? (
                  <>
                    <LedgerKpiStrip
                      items={[
                        {
                          label: "Estimated cost",
                          value: formatUsd(usage.estimated_cost_usd),
                          caption: `Today ${formatUsd(usage.today.estimated_cost_usd)} · MTD ${formatUsd(usage.month_to_date.estimated_cost_usd)}`,
                          emphasis: "money",
                        },
                        {
                          label: "Total tokens",
                          value: usage.total_tokens.toLocaleString(),
                          caption: `${usage.input_tokens.toLocaleString()} in · ${usage.output_tokens.toLocaleString()} out`,
                          emphasis: "data",
                        },
                        {
                          label: "Requests",
                          value: usage.request_count.toLocaleString(),
                          caption: `Today ${usage.today.request_count} · MTD ${usage.month_to_date.request_count}`,
                        },
                        {
                          label: "Active users",
                          value: String(usage.by_user?.length || 0),
                          caption: "With recorded AI usage",
                        },
                      ]}
                    />

                    <div className="ledger-grid-main-rail">
                      <LedgerPanel
                        title="Daily tokens"
                        description="Platform-wide volume by UTC day"
                      >
                        <LedgerBarChart points={usageDaily} />
                      </LedgerPanel>
                      <LedgerPanel title="By provider">
                        {(usage.by_provider || []).map((row) => (
                          <div key={row.provider} className="ledger-list-row">
                            <span className="capitalize">{row.provider}</span>
                            <span className="text-[var(--gd-muted)]">
                              {row.total_tokens.toLocaleString()} ·{" "}
                              {formatUsd(row.estimated_cost_usd)}
                            </span>
                          </div>
                        ))}
                        {!usage.by_provider?.length ? (
                          <LedgerEmpty>No provider usage yet.</LedgerEmpty>
                        ) : null}
                      </LedgerPanel>
                    </div>

                    <div className="ledger-grid-2">
                      <LedgerPanel title="By model">
                        <LedgerScrollTable compact>
                          <table className="ledger-table">
                            <thead>
                              <tr>
                                <th>Model</th>
                                <th data-align="right">Tokens</th>
                                <th data-align="right">Cost</th>
                              </tr>
                            </thead>
                            <tbody>
                              {(usage.by_model || []).map((row) => (
                                <tr key={`${row.provider}-${row.model_id}`}>
                                  <td className="max-w-[200px] truncate">
                                    {row.model_id || "—"}
                                  </td>
                                  <td data-align="right">
                                    {row.total_tokens.toLocaleString()}
                                  </td>
                                  <td data-align="right">
                                    {formatUsd(row.estimated_cost_usd)}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </LedgerScrollTable>
                      </LedgerPanel>
                      <LedgerPanel title="Top users by tokens">
                        <LedgerScrollTable compact>
                          <table className="ledger-table">
                            <thead>
                              <tr>
                                <th>User</th>
                                <th data-align="right">Tokens</th>
                                <th data-align="right">Cost</th>
                              </tr>
                            </thead>
                            <tbody>
                              {(usage.by_user || []).slice(0, 20).map((row) => (
                                <tr key={row.user_id}>
                                  <td>{userLabel(row.user_id)}</td>
                                  <td data-align="right">
                                    {row.total_tokens.toLocaleString()}
                                  </td>
                                  <td data-align="right">
                                    {formatUsd(row.estimated_cost_usd)}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </LedgerScrollTable>
                      </LedgerPanel>
                    </div>

                    <LedgerPanel
                      title="Recent transactions"
                      description={usage.cost_disclaimer}
                    >
                      <LedgerScrollTable>
                        <table className="ledger-table">
                          <thead>
                            <tr>
                              <th>UTC</th>
                              <th>User</th>
                              <th>Model</th>
                              <th data-align="right">Tokens</th>
                              <th data-align="right">Cost</th>
                              <th>Type</th>
                            </tr>
                          </thead>
                          <tbody>
                            {usageEvents.map((ev) => (
                              <tr key={ev.id}>
                                <td className="text-[var(--gd-muted)]">
                                  {ev.created_at.replace("T", " ").replace("Z", "")}
                                </td>
                                <td>{userLabel(ev.user_id)}</td>
                                <td className="max-w-[180px] truncate">
                                  {ev.provider}/{ev.model_id || "—"}
                                </td>
                                <td data-align="right">
                                  {(ev.total_tokens ?? 0).toLocaleString()}
                                </td>
                                <td data-align="right">
                                  {formatUsd(ev.estimated_cost_usd)}
                                </td>
                                <td>
                                  <LedgerStatus label={ev.cost_type} />
                                </td>
                              </tr>
                            ))}
                            {!usageEvents.length ? (
                              <tr>
                                <td colSpan={6}>
                                  <LedgerEmpty>No transactions yet.</LedgerEmpty>
                                </td>
                              </tr>
                            ) : null}
                          </tbody>
                        </table>
                      </LedgerScrollTable>
                    </LedgerPanel>
                  </>
                ) : (
                  <LedgerEmpty>{loading ? "Loading usage…" : "No usage data."}</LedgerEmpty>
                )}
              </div>
            ) : null}

            {section === "rag" ? (
              <div className="ledger-stack">
                <div className="ledger-toolbar">
                  <div className="ledger-filters">
                    <div className="ledger-field mb-0">
                      <label htmlFor="doc-status">Status filter</label>
                      <Input
                        id="doc-status"
                        value={docStatus}
                        onChange={(e) => setDocStatus(e.target.value)}
                        placeholder="failed | ready | processing"
                        className="h-10 w-56 rounded-[10px]"
                      />
                    </div>
                    <LedgerButton variant="secondary" onClick={() => void loadDocs()}>
                      Apply
                    </LedgerButton>
                  </div>
                </div>

                {docs ? (
                  <>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(docs.status_counts || {}).map(([status, count]) => (
                        <LedgerStatus
                          key={status}
                          label={`${status}: ${count}`}
                          tone={statusTone(status)}
                        />
                      ))}
                      <LedgerStatus label={`total: ${docs.total}`} />
                    </div>
                    <LedgerPanel
                      title="Documents"
                      description={`Showing ${docs.items.length} rows${docStatus ? ` with status “${docStatus}”` : ""}`}
                    >
                      <LedgerScrollTable>
                        <table className="ledger-table">
                          <thead>
                            <tr>
                              <th>Document</th>
                              <th>Owner</th>
                              <th>Status</th>
                              <th data-align="right">Chunks</th>
                              <th>Error</th>
                            </tr>
                          </thead>
                          <tbody>
                            {docs.items.map((doc) => (
                              <tr key={doc.id}>
                                <td className="max-w-[220px] truncate font-medium">
                                  {doc.original_name}
                                </td>
                                <td className="text-[var(--gd-muted)]">
                                  {doc.username || doc.email || "—"}
                                </td>
                                <td>
                                  <LedgerStatus
                                    label={doc.status}
                                    tone={statusTone(doc.status)}
                                  />
                                </td>
                                <td data-align="right" className="text-[var(--gd-data)]">
                                  {doc.chunk_count ?? "—"}
                                </td>
                                <td className="max-w-[240px] truncate text-[var(--gd-negative)]">
                                  {doc.error_message || "—"}
                                </td>
                              </tr>
                            ))}
                            {!docs.items.length ? (
                              <tr>
                                <td colSpan={5}>
                                  <LedgerEmpty>No documents for this filter.</LedgerEmpty>
                                </td>
                              </tr>
                            ) : null}
                          </tbody>
                        </table>
                      </LedgerScrollTable>
                    </LedgerPanel>
                  </>
                ) : (
                  <LedgerEmpty>
                    {loading ? "Loading documents…" : "No document data."}
                  </LedgerEmpty>
                )}
              </div>
            ) : null}

            {section === "integrations" ? (
              <div className="ledger-stack">
                <LedgerKpiStrip
                  items={[
                    {
                      label: "Drive connections",
                      value: String(drive?.total ?? "—"),
                      caption: "Google Drive folder links",
                    },
                    {
                      label: "Workspaces",
                      value: String(workspaces?.total ?? "—"),
                      caption: "Collaborative workspaces",
                    },
                    {
                      label: "API keys",
                      value: apiKeys
                        ? `${apiKeys.active_keys}/${apiKeys.total_keys}`
                        : "—",
                      caption: "Active / total · secrets never shown",
                      emphasis: "data",
                    },
                    {
                      label: "Key holders",
                      value: String(apiKeys?.by_user?.length ?? 0),
                      caption: "Accounts with at least one key",
                    },
                  ]}
                />

                <LedgerPanel
                  title="Drive sync status"
                  description="Credentials are never exposed."
                >
                  <LedgerScrollTable>
                    <table className="ledger-table">
                      <thead>
                        <tr>
                          <th>Folder</th>
                          <th>Owner</th>
                          <th>Mode</th>
                          <th>Status</th>
                          <th>Last error</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(drive?.items || []).map((row) => (
                          <tr key={String(row.id)}>
                            <td>{String(row.folder_name || row.folder_id || "—")}</td>
                            <td className="text-[var(--gd-muted)]">
                              {String(row.username || row.email || "—")}
                            </td>
                            <td>{String(row.access_mode || "—")}</td>
                            <td>
                              <LedgerStatus
                                label={String(row.sync_status || "unknown")}
                                tone={statusTone(String(row.sync_status || ""))}
                              />
                            </td>
                            <td className="max-w-[220px] truncate text-[var(--gd-negative)]">
                              {String(row.last_error || "—")}
                            </td>
                          </tr>
                        ))}
                        {!drive?.items?.length ? (
                          <tr>
                            <td colSpan={5}>
                              <LedgerEmpty>No Drive connections.</LedgerEmpty>
                            </td>
                          </tr>
                        ) : null}
                      </tbody>
                    </table>
                  </LedgerScrollTable>
                </LedgerPanel>

                <div className="ledger-grid-2">
                  <LedgerPanel title="Workspaces">
                    <LedgerScrollTable compact>
                      <table className="ledger-table">
                        <thead>
                          <tr>
                            <th>Name</th>
                            <th>Creator</th>
                            <th data-align="right">Members</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(workspaces?.items || []).map((ws) => (
                            <tr key={String(ws.id)}>
                              <td className="font-medium">{String(ws.name)}</td>
                              <td className="text-[var(--gd-muted)]">
                                {String(ws.creator_username || ws.created_by || "—")}
                              </td>
                              <td data-align="right">{String(ws.member_count ?? 0)}</td>
                            </tr>
                          ))}
                          {!workspaces?.items?.length ? (
                            <tr>
                              <td colSpan={3}>
                                <LedgerEmpty>No workspaces.</LedgerEmpty>
                              </td>
                            </tr>
                          ) : null}
                        </tbody>
                      </table>
                    </LedgerScrollTable>
                  </LedgerPanel>

                  <LedgerPanel title="API keys by user" description="Prefixes only">
                    <LedgerScrollTable compact>
                      <table className="ledger-table">
                        <thead>
                          <tr>
                            <th>User</th>
                            <th data-align="right">Active</th>
                            <th data-align="right">Total</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(apiKeys?.by_user || []).map((row) => (
                            <tr key={String(row.user_id)}>
                              <td>{String(row.username || row.email || "—")}</td>
                              <td data-align="right">{String(row.active_keys ?? 0)}</td>
                              <td data-align="right">{String(row.total_keys ?? 0)}</td>
                            </tr>
                          ))}
                          {!apiKeys?.by_user?.length ? (
                            <tr>
                              <td colSpan={3}>
                                <LedgerEmpty>No API keys issued.</LedgerEmpty>
                              </td>
                            </tr>
                          ) : null}
                        </tbody>
                      </table>
                    </LedgerScrollTable>
                  </LedgerPanel>
                </div>
              </div>
            ) : null}

            {section === "database" ? (
              <div className="ledger-stack">
                <LedgerPanel
                  title="Logical backup"
                  description="Downloads every table. hashed_password, API key hashes, Drive refresh tokens and similar columns are replaced with [REDACTED]."
                  padded
                >
                  <div className="flex flex-wrap gap-3">
                    <LedgerButton
                      variant="primary"
                      onClick={() => void exportDb("json")}
                      disabled={exporting}
                    >
                      <HardDrive className="h-3.5 w-3.5" />
                      Export JSON
                    </LedgerButton>
                    <LedgerButton
                      variant="secondary"
                      onClick={() => void exportDb("sql")}
                      disabled={exporting}
                    >
                      <Database className="h-3.5 w-3.5" />
                      Export SQL
                    </LedgerButton>
                  </div>
                  <p className="mt-4 text-[13px] text-[var(--gd-muted)]">
                    Signed in as {user?.email}. Exports require an admin session.
                  </p>
                </LedgerPanel>
              </div>
            ) : null}

            {section === "prompts" ? <PromptManagementPanel /> : null}

            {section === "health" ? (
              <div className="ledger-stack">
                <LedgerPanel
                  title="Dependency status"
                  description={
                    health?.overall
                      ? `Overall ${health.overall}`
                      : "From /api/v1/health/status"
                  }
                  padded
                >
                  <div className="grid gap-3 sm:grid-cols-2">
                    {(
                      [
                        ["database", healthParts(health).database],
                        ["redis", healthParts(health).redis],
                        ["celery", healthParts(health).celery],
                        ["chromadb", healthParts(health).chromadb],
                        ["ollama", healthParts(health).ollama],
                      ] as const
                    ).map(([name, info]) => {
                      const detail = (info || {}) as Record<string, unknown>;
                      return (
                        <div
                          key={name}
                          className="rounded-[8px] border border-[var(--gd-border)] p-3 text-[13px]"
                        >
                          <div className="mb-2 flex items-center justify-between gap-2">
                            <span className="font-medium">
                              {name === "chromadb" ? "ChromaDB" : name}
                            </span>
                            <LedgerStatus
                              label={String(detail.status || "unknown")}
                              tone={statusTone(String(detail.status || ""))}
                            />
                          </div>
                          <p className="text-[var(--gd-muted)]">
                            {typeof detail.latency_ms === "number"
                              ? `${detail.latency_ms} ms`
                              : null}
                            {detail.detail ? ` ${String(detail.detail)}` : null}
                            {detail.error ? ` ${String(detail.error)}` : null}
                            {Array.isArray(detail.workers)
                              ? ` Workers: ${detail.workers.length}`
                              : null}
                            {!info ? "—" : null}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </LedgerPanel>
              </div>
            ) : null}

            <p className="mt-8 flex items-center gap-2 text-[12px] text-[var(--gd-muted)]">
              <Shield className="h-3.5 w-3.5" />
              Admin session for {user?.email}
              <span className="inline-flex items-center gap-1">
                <KeyRound className="h-3 w-3" />
                Secrets never rendered
              </span>
            </p>
          </LedgerScroll>
        </LedgerCanvas>
      </div>
    </LedgerShell>
  );
}

export default function AdminPage() {
  return (
    <RequireAuth requireAdmin>
      <AdminPageContent />
    </RequireAuth>
  );
}

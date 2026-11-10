"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ChevronDown,
  ChevronLeft,
  FileText,
  FolderOpen,
  Loader2,
  RefreshCw,
  Trash2,
  Unplug,
} from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  DropboxMark,
  IntegrationMethod,
  IntegrationProviderCard,
  IntegrationStatusPill,
} from "@/components/settings/IntegrationProviderCard";

type DropboxStatus = {
  connected: boolean;
  configured: boolean;
  account_id?: string | null;
  display_name?: string | null;
  test_shared_link?: string | null;
};

type DropboxEntry = {
  id: string | null;
  name: string;
  path: string;
  is_folder: boolean;
  is_pdf: boolean;
  size?: number | null;
  revision?: string | null;
  modified_at?: string | null;
  document_id?: string | null;
  lifecycle_status?: string | null;
};

type DropboxBrowse = {
  connected: boolean;
  connection_id?: string | null;
  path: string;
  entries: DropboxEntry[];
};

type DropboxConnection = {
  id: string;
  connection_kind: string;
  remote_folder_id: string;
  folder_path: string | null;
  folder_name: string | null;
  folder_url: string | null;
  access_mode: string;
  enabled: boolean;
  sync_status: string;
  task_id?: string | null;
  last_error: string | null;
  last_synced_at: string | null;
  document_count: number;
};

function statusLabel(status: string) {
  switch (status) {
    case "completed":
    case "ok":
      return "Completed";
    case "processing":
    case "syncing":
      return "Syncing…";
    case "queued":
      return "Queued";
    case "failed":
    case "error":
      return "Failed";
    case "idle":
      return "Idle";
    default:
      return status || "—";
  }
}

function isBusy(status: string) {
  return status === "queued" || status === "processing" || status === "syncing";
}

function parentPath(path: string) {
  const trimmed = (path || "").replace(/\/+$/, "");
  if (!trimmed) return "";
  const index = trimmed.lastIndexOf("/");
  return index <= 0 ? "" : trimmed.slice(0, index);
}

function formatBytes(size?: number | null) {
  if (!size || size <= 0) return "";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DropboxSettings() {
  const [status, setStatus] = useState<DropboxStatus>({
    connected: false,
    configured: false,
  });
  const [connections, setConnections] = useState<DropboxConnection[]>([]);
  const [loading, setLoading] = useState(true);
  const [connecting, setConnecting] = useState(false);

  const [path, setPath] = useState("");
  const [entries, setEntries] = useState<DropboxEntry[]>([]);
  const [browsing, setBrowsing] = useState(false);
  const [browseConnectionId, setBrowseConnectionId] = useState<string | null>(
    null,
  );
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [browserOpen, setBrowserOpen] = useState(false);

  const [linkInput, setLinkInput] = useState("");
  const [importingLink, setImportingLink] = useState(false);
  const [importing, setImporting] = useState(false);
  const [syncingId, setSyncingId] = useState<string | null>(null);
  const [syncingAll, setSyncingAll] = useState(false);
  const [manageOpen, setManageOpen] = useState(false);

  const browseConnection = useMemo(
    () => connections.find((c) => c.id === browseConnectionId) || null,
    [connections, browseConnectionId],
  );

  const refreshConnections = useCallback(async () => {
    try {
      const data = await api.get<{ connections: DropboxConnection[] }>(
        "/api/v1/dropbox/connections",
      );
      const next = data.connections || [];
      setConnections(next);
      if (next.length > 0) setManageOpen(true);
      return next;
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Could not load Dropbox sources",
      );
      return [];
    }
  }, []);

  const loadBrowse = useCallback(
    async (targetPath: string, connectionId: string | null) => {
      if (!status.connected && !connectionId) {
        setEntries([]);
        return;
      }
      setBrowsing(true);
      setSelected(new Set());
      try {
        const params = new URLSearchParams();
        if (targetPath) params.set("path", targetPath);
        if (connectionId) params.set("connection_id", connectionId);
        const query = params.toString() ? `?${params.toString()}` : "";
        const data = await api.get<DropboxBrowse>(
          `/api/v1/dropbox/folders${query}`,
        );
        setEntries(data.entries || []);
        setPath(data.path || targetPath || "");
      } catch (err) {
        toast.error(
          err instanceof Error ? err.message : "Could not browse Dropbox",
        );
      } finally {
        setBrowsing(false);
      }
    },
    [status.connected],
  );

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const data = await api.get<DropboxStatus>("/api/v1/dropbox/status");
        if (cancelled) return;
        setStatus(data);
        const loaded = await refreshConnections();
        if (!cancelled && data.connected) {
          await loadBrowse("", null);
        }
        if (!cancelled && loaded.length === 0 && !data.connected) {
          setEntries([]);
        }
      } catch (err) {
        if (!cancelled) {
          toast.error(
            err instanceof Error
              ? err.message
              : "Could not load Dropbox settings",
          );
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [loadBrowse, refreshConnections]);

  const busyKey = connections
    .filter((c) => isBusy(c.sync_status))
    .map((c) => c.id)
    .sort()
    .join(",");

  useEffect(() => {
    if (!busyKey) return;
    const busyIds = busyKey.split(",");
    let cancelled = false;

    const poll = async () => {
      const updates = await Promise.all(
        busyIds.map(async (id) => {
          try {
            return await api.get<{
              connection_id: string;
              sync_status: string;
              task_id?: string | null;
              last_error: string | null;
              last_synced_at: string | null;
              document_count: number;
            }>(`/api/v1/dropbox/connections/${id}/status`);
          } catch {
            return null;
          }
        }),
      );
      if (cancelled) return;
      setConnections((prev) =>
        prev.map((connection) => {
          const hit = updates.find(
            (u) => u && u.connection_id === connection.id,
          );
          if (!hit) return connection;
          return {
            ...connection,
            sync_status: hit.sync_status,
            task_id: hit.task_id,
            last_error: hit.last_error,
            last_synced_at: hit.last_synced_at,
            document_count: hit.document_count,
          };
        }),
      );
    };

    void poll();
    const timer = window.setInterval(() => void poll(), 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [busyKey]);

  const handleConnect = async () => {
    setConnecting(true);
    try {
      const data = await api.get<{ auth_url: string }>(
        "/api/v1/dropbox/connect",
      );
      window.location.assign(data.auth_url);
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Failed to start Dropbox connect",
      );
      setConnecting(false);
    }
  };

  const handleDisconnect = async () => {
    setConnecting(true);
    try {
      await api.delete("/api/v1/dropbox/disconnect");
      setStatus((prev) => ({
        ...prev,
        connected: false,
        account_id: null,
        display_name: null,
      }));
      setEntries([]);
      setPath("");
      setBrowserOpen(false);
      toast.success("Dropbox disconnected");
      await refreshConnections();
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Failed to disconnect Dropbox",
      );
    } finally {
      setConnecting(false);
    }
  };

  const toggleSelected = (entry: DropboxEntry) => {
    const key = entry.id || entry.path;
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const handleImportFolder = async () => {
    if (selected.size === 0 && !path) {
      toast.message("Open a folder to import");
      return;
    }
    if (!status.connected) {
      toast.error("Connect Dropbox to import folders");
      return;
    }
    const freshIds = entries
      .filter(
        (entry) =>
          selected.has(entry.id || entry.path) &&
          entry.is_pdf &&
          entry.lifecycle_status !== "USER_DELETED" &&
          Boolean(entry.id),
      )
      .map((entry) => entry.id as string);

    setImporting(true);
    try {
      await api.post("/api/v1/dropbox/connections", {
        folder_path: path,
        file_ids: freshIds,
      });
      toast.success(
        freshIds.length
          ? `Import queued for ${freshIds.length} PDF${freshIds.length === 1 ? "" : "s"}`
          : "Folder import queued",
      );
      setSelected(new Set());
      setManageOpen(true);
      await refreshConnections();
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Could not import folder",
      );
    } finally {
      setImporting(false);
    }
  };

  const resolveReimportConnectionId = (
    deletedEntries: DropboxEntry[],
  ): string | null => {
    if (browseConnectionId) return browseConnectionId;
    if (!deletedEntries.length || !connections.length) return null;

    for (const connection of connections) {
      const base = (connection.folder_path || "").replace(/\/+$/, "");
      const covers = deletedEntries.every((entry) => {
        if (!base) return true;
        return entry.path === base || entry.path.startsWith(`${base}/`);
      });
      if (covers) return connection.id;
    }
    return connections[0]?.id ?? null;
  };

  const handleReimport = async () => {
    const deletedEntries = entries.filter(
      (entry) =>
        selected.has(entry.id || entry.path) &&
        entry.is_pdf &&
        entry.lifecycle_status === "USER_DELETED" &&
        Boolean(entry.id),
    );
    if (deletedEntries.length === 0) {
      toast.message("Select at least one PDF marked Deleted in Genova");
      return;
    }

    const connectionId = resolveReimportConnectionId(deletedEntries);
    if (!connectionId) {
      toast.error(
        "No Dropbox source found. Open Manage sources → Restore deleted files first.",
      );
      return;
    }

    setImporting(true);
    try {
      await api.post(`/api/v1/dropbox/connections/${connectionId}/import`, {
        file_ids: deletedEntries.map((entry) => entry.id as string),
      });
      toast.success("Re-import queued");
      setSelected(new Set());
      await loadBrowse(path, browseConnectionId);
      await refreshConnections();
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Could not re-import files",
      );
    } finally {
      setImporting(false);
    }
  };

  const handleImportLink = async () => {
    const link = linkInput.trim();
    if (!link) {
      toast.message("Paste a Dropbox shared link");
      return;
    }
    setImportingLink(true);
    try {
      await api.post("/api/v1/dropbox/connections", { shared_link: link });
      toast.success("Dropbox link import queued");
      setLinkInput("");
      setManageOpen(true);
      await refreshConnections();
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Could not import Dropbox link",
      );
    } finally {
      setImportingLink(false);
    }
  };

  const handleSyncOne = async (connection: DropboxConnection) => {
    setSyncingId(connection.id);
    try {
      const result = await api.post<{ already_running?: boolean }>(
        `/api/v1/dropbox/connections/${connection.id}/sync`,
      );
      toast.success(
        result.already_running ? "Sync already running" : "Sync queued",
      );
      await refreshConnections();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Sync failed");
      await refreshConnections();
    } finally {
      setSyncingId(null);
    }
  };

  const handleSyncAll = async () => {
    if (!connections.length) return;
    setSyncingAll(true);
    try {
      const result = await api.post<{
        already_running?: boolean;
        message?: string;
      }>("/api/v1/dropbox/sync");
      toast.success(
        result.already_running
          ? "Sync already running"
          : result.message || "Dropbox sync queued",
      );
      await refreshConnections();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Sync failed");
    } finally {
      setSyncingAll(false);
    }
  };

  const handleRemove = async (connection: DropboxConnection) => {
    try {
      await api.delete(`/api/v1/dropbox/connections/${connection.id}`);
      if (browseConnectionId === connection.id) {
        setBrowseConnectionId(null);
        setEntries([]);
        setPath("");
      }
      toast.success("Dropbox source removed");
      await refreshConnections();
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Could not remove Dropbox source",
      );
    }
  };

  const openConnectionBrowse = async (connection: DropboxConnection) => {
    setBrowseConnectionId(connection.id);
    setPath(connection.folder_path || "");
    setBrowserOpen(true);
    setManageOpen(true);
    await loadBrowse(connection.folder_path || "", connection.id);
  };

  const exitConnectionBrowse = async () => {
    setBrowseConnectionId(null);
    setSelected(new Set());
    setPath("");
    await loadBrowse("", null);
  };

  const selectedEntries = entries.filter((entry) =>
    selected.has(entry.id || entry.path),
  );
  const deletedSelected = selectedEntries.filter(
    (entry) => entry.is_pdf && entry.lifecycle_status === "USER_DELETED",
  );
  const importableSelected = selectedEntries.filter(
    (entry) => entry.is_pdf && entry.lifecycle_status !== "USER_DELETED",
  );
  const reimportMode = Boolean(browseConnectionId);
  const anyBusy = connections.some((c) => isBusy(c.sync_status));

  if (loading) {
    return (
      <div className="flex items-center gap-2 py-8 text-sm text-[var(--gd-muted)]">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading Dropbox…
      </div>
    );
  }

  return (
    <IntegrationProviderCard
      name="Dropbox"
      description="Bring PDFs from Dropbox into Genova for search and chat."
      icon={<DropboxMark />}
      status={
        status.connected ? (
          <IntegrationStatusPill tone="ok">Connected</IntegrationStatusPill>
        ) : anyBusy ? (
          <IntegrationStatusPill tone="busy">Syncing</IntegrationStatusPill>
        ) : connections.length > 0 ? (
          <IntegrationStatusPill tone="idle">
            {connections.length} source{connections.length === 1 ? "" : "s"}
          </IntegrationStatusPill>
        ) : (
          <IntegrationStatusPill tone="idle">Not connected</IntegrationStatusPill>
        )
      }
    >
      <IntegrationMethod
        label="Connect your account"
        hint="Sign in with Dropbox to browse folders and keep them in sync."
      >
        {!status.configured ? (
          <p className="text-[12px] text-[var(--gd-warning)]">
            Dropbox OAuth is not configured on the server. Public links still
            work.
          </p>
        ) : null}
        {status.connected ? (
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-[13px] text-[var(--gd-muted)]">
              {status.display_name
                ? `Signed in as ${status.display_name}`
                : "Dropbox account linked"}
            </p>
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-8"
              onClick={() => {
                setBrowserOpen(true);
                void loadBrowse(path || "", browseConnectionId);
              }}
            >
              <FolderOpen className="mr-1.5 h-3.5 w-3.5" />
              Browse folders
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-8 text-[var(--gd-muted)]"
              disabled={connecting}
              onClick={() => void handleDisconnect()}
            >
              {connecting ? (
                <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
              ) : (
                <Unplug className="mr-1.5 h-3.5 w-3.5" />
              )}
              Disconnect
            </Button>
          </div>
        ) : (
          <Button
            type="button"
            size="sm"
            className="h-9"
            disabled={connecting || !status.configured}
            onClick={() => void handleConnect()}
          >
            {connecting ? (
              <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
            ) : (
              <DropboxMark className="mr-1.5 h-4 w-4" />
            )}
            Connect Dropbox
          </Button>
        )}
      </IntegrationMethod>

      <IntegrationMethod
        label="Import with a link"
        hint="Paste a public Dropbox shared folder or file link."
      >
        <div className="flex flex-col gap-2 sm:flex-row sm:items-stretch">
          <input
            value={linkInput}
            onChange={(event) => setLinkInput(event.target.value)}
            placeholder="https://www.dropbox.com/scl/fo/…"
            className="h-9 w-full flex-1 rounded-md border border-[var(--gd-border)] bg-[var(--gd-ground)] px-3 text-[13px] text-[var(--gd-text)] outline-none placeholder:text-[var(--gd-faint)] focus-visible:ring-2 focus-visible:ring-[var(--gd-accent)]"
            disabled={importingLink}
          />
          <Button
            type="button"
            size="sm"
            className="h-9 shrink-0"
            disabled={importingLink || !linkInput.trim()}
            onClick={() => void handleImportLink()}
          >
            {importingLink ? (
              <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
            ) : null}
            Import
          </Button>
        </div>
      </IntegrationMethod>

      {status.connected && browserOpen ? (
        <div className="space-y-2 rounded-[10px] border border-[var(--gd-border)] bg-[var(--gd-ground)] p-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex min-w-0 items-center gap-1.5">
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                title="Back"
                disabled={browsing || (!path && !reimportMode)}
                onClick={() => {
                  if (reimportMode) {
                    void loadBrowse(parentPath(path), browseConnectionId);
                  } else {
                    void loadBrowse(parentPath(path), null);
                  }
                }}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <p className="truncate text-[12px] text-[var(--gd-muted)]">
                {reimportMode
                  ? `Restore deleted — ${browseConnection?.folder_name || "Dropbox"}`
                  : path || "Dropbox"}
              </p>
            </div>
            <div className="flex items-center gap-2">
              {reimportMode ? (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 text-xs"
                  onClick={() => void exitConnectionBrowse()}
                >
                  Exit restore
                </Button>
              ) : null}
              {deletedSelected.length > 0 || reimportMode ? (
                <Button
                  type="button"
                  size="sm"
                  className="h-7 text-xs"
                  disabled={importing || deletedSelected.length === 0}
                  onClick={() => void handleReimport()}
                >
                  {importing ? (
                    <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
                  )}
                  Re-import ({deletedSelected.length})
                </Button>
              ) : null}
              {!reimportMode ? (
                <Button
                  type="button"
                  size="sm"
                  className="h-7 text-xs"
                  disabled={
                    importing || (!path && importableSelected.length === 0)
                  }
                  onClick={() => void handleImportFolder()}
                >
                  {importing ? (
                    <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <FolderOpen className="mr-1.5 h-3.5 w-3.5" />
                  )}
                  {importableSelected.length
                    ? `Import (${importableSelected.length})`
                    : "Import folder"}
                </Button>
              ) : null}
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-7 text-xs text-[var(--gd-muted)]"
                onClick={() => setBrowserOpen(false)}
              >
                Close
              </Button>
            </div>
          </div>

          {browsing ? (
            <div className="flex items-center gap-2 py-3 text-[12px] text-[var(--gd-muted)]">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Loading…
            </div>
          ) : entries.length === 0 ? (
            <p className="rounded-md border border-dashed border-[var(--gd-border)] px-3 py-3 text-center text-[12px] text-[var(--gd-muted)]">
              {reimportMode
                ? "No deleted PDFs here"
                : "No folders or PDFs here"}
            </p>
          ) : (
            <ul className="max-h-56 space-y-1 overflow-y-auto">
              {entries.map((entry) => {
                const key = entry.id || entry.path;
                const deleted = entry.lifecycle_status === "USER_DELETED";
                const selectable =
                  entry.is_pdf && (reimportMode ? deleted : true);
                return (
                  <li
                    key={key}
                    className="flex items-center justify-between gap-2 rounded px-2 py-1.5 hover:bg-[var(--gd-raised)]"
                  >
                    <div className="flex min-w-0 items-center gap-2">
                      {selectable ? (
                        <input
                          type="checkbox"
                          className="h-3.5 w-3.5 accent-current"
                          checked={selected.has(key)}
                          onChange={() => toggleSelected(entry)}
                          aria-label={
                            deleted
                              ? `Select ${entry.name} to re-import`
                              : `Select ${entry.name}`
                          }
                        />
                      ) : (
                        <span className="h-3.5 w-3.5" />
                      )}
                      {entry.is_folder ? (
                        <FolderOpen className="h-3.5 w-3.5 shrink-0 text-[var(--gd-muted)]" />
                      ) : (
                        <FileText className="h-3.5 w-3.5 shrink-0 text-[var(--gd-muted)]" />
                      )}
                      {entry.is_folder ? (
                        <button
                          type="button"
                          className="truncate text-left text-[12px] hover:underline"
                          onClick={() =>
                            void loadBrowse(entry.path, browseConnectionId)
                          }
                        >
                          {entry.name}
                        </button>
                      ) : (
                        <span className="truncate text-[12px]">
                          {entry.name}
                          <span className="ml-1 text-[10px] text-[var(--gd-muted)]">
                            {formatBytes(entry.size)}
                          </span>
                        </span>
                      )}
                    </div>
                    <div className="shrink-0 text-[10px]">
                      {deleted ? (
                        <button
                          type="button"
                          className="text-[var(--gd-warning)] underline-offset-2 hover:underline"
                          title="Select to re-import"
                          onClick={() => toggleSelected(entry)}
                        >
                          Deleted in Genova
                        </button>
                      ) : entry.document_id ? (
                        <span className="text-[var(--gd-positive)]">
                          Imported
                        </span>
                      ) : entry.is_pdf ? (
                        <span className="text-[var(--gd-muted)]">PDF</span>
                      ) : null}
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      ) : null}

      <div className="mt-auto border-t border-[var(--gd-rule)] pt-4">
        <button
          type="button"
          className="flex w-full items-center justify-between text-left text-[13px] font-medium text-[var(--gd-text)]"
          onClick={() => setManageOpen((open) => !open)}
          aria-expanded={manageOpen}
        >
          <span>
            Manage sources
            {connections.length > 0 ? (
              <span className="ml-1.5 font-normal text-[var(--gd-muted)]">
                ({connections.length})
              </span>
            ) : null}
          </span>
          <ChevronDown
            className={`h-4 w-4 text-[var(--gd-muted)] transition-transform ${manageOpen ? "rotate-180" : ""}`}
          />
        </button>

        {manageOpen ? (
          <div className="mt-3 space-y-3">
            {connections.length > 0 ? (
              <div className="flex justify-end">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8"
                  disabled={syncingAll}
                  onClick={() => void handleSyncAll()}
                >
                  {syncingAll ? (
                    <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
                  )}
                  Sync all
                </Button>
              </div>
            ) : null}

            {connections.length === 0 ? (
              <p className="rounded-md border border-dashed border-[var(--gd-border)] px-3 py-4 text-center text-[12px] text-[var(--gd-muted)]">
                No Dropbox sources yet
              </p>
            ) : (
              <ul className="max-h-52 space-y-2 overflow-y-auto">
                {connections.map((connection) => (
                  <li
                    key={connection.id}
                    className="rounded-md border border-[var(--gd-border)] bg-[var(--gd-ground)] px-3 py-2.5"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate text-[13px] font-medium text-[var(--gd-text)]">
                          {connection.folder_name || connection.remote_folder_id}
                        </p>
                        <p className="mt-0.5 text-[11px] text-[var(--gd-muted)]">
                          {connection.connection_kind === "link"
                            ? "Public link"
                            : "Connected account"}
                          {" · "}
                          <span
                            className={
                              connection.sync_status === "failed" ||
                              connection.sync_status === "error"
                                ? "text-[var(--gd-negative)]"
                                : isBusy(connection.sync_status)
                                  ? "text-[var(--gd-accent)]"
                                  : undefined
                            }
                          >
                            {statusLabel(connection.sync_status)}
                          </span>
                          {connection.document_count > 0
                            ? ` · ${connection.document_count} PDF${connection.document_count === 1 ? "" : "s"}`
                            : ""}
                        </p>
                        {connection.last_error ? (
                          <p className="mt-1 line-clamp-2 text-[11px] text-[var(--gd-negative)]">
                            {connection.last_error}
                          </p>
                        ) : null}
                      </div>
                      <div className="flex shrink-0 items-center gap-0.5">
                        {connection.connection_kind === "oauth" &&
                        status.connected ? (
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            className="h-7 w-7"
                            title="Restore deleted files"
                            onClick={() => void openConnectionBrowse(connection)}
                          >
                            <FolderOpen className="h-3.5 w-3.5" />
                          </Button>
                        ) : null}
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          title="Sync source"
                          disabled={
                            syncingId === connection.id ||
                            isBusy(connection.sync_status)
                          }
                          onClick={() => void handleSyncOne(connection)}
                        >
                          {syncingId === connection.id ||
                          isBusy(connection.sync_status) ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          ) : (
                            <RefreshCw className="h-3.5 w-3.5" />
                          )}
                        </Button>
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 text-[var(--gd-negative)] hover:text-[var(--gd-negative)]"
                          title="Remove source"
                          onClick={() => void handleRemove(connection)}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        ) : null}
      </div>
    </IntegrationProviderCard>
  );
}

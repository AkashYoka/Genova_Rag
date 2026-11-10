"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ChevronLeft,
  FileText,
  FolderOpen,
  Link2,
  Loader2,
  RefreshCw,
  Trash2,
  Unplug,
} from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";

const FALLBACK_TEST_LINK =
  "https://www.dropbox.com/scl/fo/bc9imczz3bxl5ya8l7cil/AN5l1yDnWejG9jYIR_fD6A0?rlkey=rp14xvckn7zin1bw4axe50bjt&st=tzxfo2y1&e=1&dl=0";

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
      return "Processing…";
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

function statusClass(status: string) {
  switch (status) {
    case "completed":
    case "ok":
      return "text-emerald-600 dark:text-emerald-400";
    case "processing":
    case "syncing":
    case "queued":
      return "text-sky-600 dark:text-sky-400";
    case "failed":
    case "error":
      return "text-destructive";
    default:
      return "text-muted-foreground";
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

  const [linkInput, setLinkInput] = useState("");
  const [importingLink, setImportingLink] = useState(false);
  const [importing, setImporting] = useState(false);
  const [syncingId, setSyncingId] = useState<string | null>(null);
  const [syncingAll, setSyncingAll] = useState(false);

  const testLink = status.test_shared_link || FALLBACK_TEST_LINK;
  const browseConnection = useMemo(
    () => connections.find((c) => c.id === browseConnectionId) || null,
    [connections, browseConnectionId],
  );

  const refreshConnections = useCallback(async () => {
    try {
      const data = await api.get<{ connections: DropboxConnection[] }>(
        "/api/v1/dropbox/connections",
      );
      setConnections(data.connections || []);
      return data.connections || [];
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

  // Poll lightweight status while any connection sync is in flight.
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
    setImporting(true);
    try {
      await api.post("/api/v1/dropbox/connections", {
        folder_path: path,
        file_ids: Array.from(selected),
      });
      toast.success(
        selected.size
          ? `Import queued for ${selected.size} PDF${selected.size === 1 ? "" : "s"}`
          : "Folder import queued — new PDFs will sync automatically",
      );
      setSelected(new Set());
      await refreshConnections();
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Could not import folder",
      );
    } finally {
      setImporting(false);
    }
  };

  const handleReimport = async () => {
    if (!browseConnectionId || selected.size === 0) return;
    setImporting(true);
    try {
      await api.post(`/api/v1/dropbox/connections/${browseConnectionId}/import`, {
        file_ids: Array.from(selected),
      });
      toast.success("Re-import queued — the selected PDFs will be searchable again");
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
        result.already_running
          ? "Sync already running — progress updates here"
          : "Sync queued",
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
      const result = await api.post<{ already_running?: boolean; message?: string }>(
        "/api/v1/dropbox/sync",
      );
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
      toast.success("Dropbox source detached");
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
    await loadBrowse(connection.folder_path || "", connection.id);
  };

  const exitConnectionBrowse = async () => {
    setBrowseConnectionId(null);
    setSelected(new Set());
    setPath("");
    await loadBrowse("", null);
  };

  if (loading) {
    return (
      <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading Dropbox settings…
      </div>
    );
  }

  const canBrowse = status.connected;
  const selectedEntries = entries.filter((entry) =>
    selected.has(entry.id || entry.path),
  );
  const reimportMode = Boolean(browseConnectionId);

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-medium">Dropbox</p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Connect Dropbox to browse folders and import PDFs, or paste a public
            shared link. Ingestion uses the same RAG pipeline and runs in the
            background.
            {status.connected
              ? ` · Connected${status.display_name ? ` as ${status.display_name}` : ""}.`
              : ""}
          </p>
          {!status.configured ? (
            <p className="mt-1 text-[11px] text-amber-600 dark:text-amber-400">
              Dropbox OAuth is not configured on the server. Shared-link imports
              still work.
            </p>
          ) : null}
        </div>
        {status.connected ? (
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="shrink-0"
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
        ) : (
          <Button
            type="button"
            size="sm"
            className="shrink-0"
            disabled={connecting || !status.configured}
            onClick={() => void handleConnect()}
          >
            {connecting ? (
              <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
            ) : (
              <Link2 className="mr-1.5 h-3.5 w-3.5" />
            )}
            Connect Dropbox
          </Button>
        )}
      </div>

      {canBrowse ? (
        <div className="space-y-2 rounded-md border border-border bg-muted/20 p-3">
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
                    void loadBrowse(
                      parentPath(path),
                      browseConnectionId,
                    );
                  } else {
                    void loadBrowse(parentPath(path), null);
                  }
                }}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <p className="truncate text-xs text-muted-foreground">
                {reimportMode
                  ? `Restore deleted — ${browseConnection?.folder_name || "Dropbox"}`
                  : path || "Dropbox (root)"}
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
                  Exit restore view
                </Button>
              ) : null}
              {reimportMode ? (
                <Button
                  type="button"
                  size="sm"
                  className="h-7 text-xs"
                  disabled={importing || selected.size === 0}
                  onClick={() => void handleReimport()}
                >
                  {importing ? (
                    <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
                  )}
                  Re-import selected ({selected.size})
                </Button>
              ) : (
                <Button
                  type="button"
                  size="sm"
                  className="h-7 text-xs"
                  disabled={importing || !path}
                  onClick={() => void handleImportFolder()}
                >
                  {importing ? (
                    <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <FolderOpen className="mr-1.5 h-3.5 w-3.5" />
                  )}
                  {selected.size
                    ? `Import selected (${selected.size})`
                    : "Import this folder"}
                </Button>
              )}
            </div>
          </div>

          {browsing ? (
            <div className="flex items-center gap-2 py-3 text-xs text-muted-foreground">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Loading Dropbox…
            </div>
          ) : entries.length === 0 ? (
            <p className="rounded-md border border-dashed border-border px-3 py-3 text-center text-xs text-muted-foreground">
              {reimportMode
                ? "No files here. Deleted PDFs appear in this view for manual restore."
                : "No folders or PDFs in this location."}
            </p>
          ) : (
            <ul className="max-h-64 space-y-1 overflow-y-auto">
              {entries.map((entry) => {
                const key = entry.id || entry.path;
                const deleted = entry.lifecycle_status === "USER_DELETED";
                const selectable = reimportMode
                  ? entry.is_pdf && deleted
                  : entry.is_pdf && !deleted;
                return (
                  <li
                    key={key}
                    className="flex items-center justify-between gap-2 rounded px-2 py-1.5 hover:bg-muted/60"
                  >
                    <div className="flex min-w-0 items-center gap-2">
                      {selectable ? (
                        <input
                          type="checkbox"
                          className="h-3.5 w-3.5 accent-current"
                          checked={selected.has(key)}
                          onChange={() => toggleSelected(entry)}
                          aria-label={`Select ${entry.name}`}
                        />
                      ) : (
                        <span className="h-3.5 w-3.5" />
                      )}
                      {entry.is_folder ? (
                        <FolderOpen className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                      ) : (
                        <FileText className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                      )}
                      {entry.is_folder ? (
                        <button
                          type="button"
                          className="truncate text-left text-xs hover:underline"
                          onClick={() =>
                            void loadBrowse(entry.path, browseConnectionId)
                          }
                        >
                          {entry.name}
                        </button>
                      ) : (
                        <span className="truncate text-xs">
                          {entry.name}
                          <span className="ml-1 text-[10px] text-muted-foreground">
                            {formatBytes(entry.size)}
                          </span>
                        </span>
                      )}
                    </div>
                    <div className="shrink-0 text-[10px]">
                      {deleted ? (
                        <span className="text-amber-600 dark:text-amber-400">
                          Deleted in Genova
                        </span>
                      ) : entry.document_id ? (
                        <span className="text-emerald-600 dark:text-emerald-400">
                          Imported
                        </span>
                      ) : entry.is_pdf ? (
                        <span className="text-muted-foreground">PDF</span>
                      ) : null}
                    </div>
                  </li>
                );
              })}
            </ul>
          )}

          {!reimportMode && selectedEntries.length > 0 ? (
            <p className="text-[10px] text-muted-foreground">
              Selected PDFs import now; new PDFs added to this folder sync
              automatically.
            </p>
          ) : null}
        </div>
      ) : null}

      <div className="space-y-2">
        <label className="text-xs font-medium text-muted-foreground">
          Import using Dropbox link
        </label>
        <textarea
          value={linkInput}
          onChange={(event) => setLinkInput(event.target.value)}
          placeholder="https://www.dropbox.com/scl/fo/…"
          rows={2}
          className="w-full resize-y rounded-md border border-input bg-background px-3 py-2 text-sm outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring"
          disabled={importingLink}
        />
        <div className="flex flex-wrap gap-2">
          <Button
            type="button"
            size="sm"
            disabled={importingLink || !linkInput.trim()}
            onClick={() => void handleImportLink()}
          >
            {importingLink ? (
              <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
            ) : (
              <Link2 className="mr-1.5 h-3.5 w-3.5" />
            )}
            Import link
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={importingLink}
            onClick={() => setLinkInput(testLink)}
          >
            Use test folder link
          </Button>
        </div>
        <p className="text-[11px] text-muted-foreground">
          Works without connecting an account when the link is publicly
          accessible. Shared folders are imported as ZIP archives.
        </p>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between gap-2">
          <p className="text-xs font-medium text-muted-foreground">
            Dropbox sources ({connections.length})
          </p>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-7 text-xs"
            disabled={syncingAll || connections.length === 0}
            onClick={() => void handleSyncAll()}
          >
            {syncingAll ? (
              <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
            ) : (
              <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
            )}
            Sync Now
          </Button>
        </div>

        {connections.length === 0 ? (
          <p className="rounded-md border border-dashed border-border px-3 py-4 text-center text-xs text-muted-foreground">
            No Dropbox sources yet. Connect Dropbox to browse folders, or paste
            a shared link above.
          </p>
        ) : (
          <ul className="max-h-64 space-y-2 overflow-y-auto">
            {connections.map((connection) => (
              <li
                key={connection.id}
                className="rounded-md border border-border bg-muted/30 px-3 py-2"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">
                      {connection.folder_name || connection.remote_folder_id}
                    </p>
                    <p className="text-[11px] text-muted-foreground">
                      {connection.connection_kind === "link"
                        ? "Shared link"
                        : "OAuth folder"}
                      <span className={statusClass(connection.sync_status)}>
                        {" · "}
                        {statusLabel(connection.sync_status)}
                      </span>
                      {connection.document_count > 0
                        ? ` · ${connection.document_count} PDF${connection.document_count === 1 ? "" : "s"}`
                        : ""}
                      {connection.last_synced_at
                        ? ` · ${new Date(connection.last_synced_at).toLocaleString()}`
                        : ""}
                    </p>
                    {connection.last_error ? (
                      <p className="mt-0.5 line-clamp-2 text-[11px] text-destructive">
                        {connection.last_error}
                      </p>
                    ) : null}
                  </div>
                  <div className="flex shrink-0 items-center gap-0.5">
                    {connection.connection_kind === "oauth" && status.connected ? (
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
                        syncingId === connection.id || isBusy(connection.sync_status)
                      }
                      onClick={() => void handleSyncOne(connection)}
                    >
                      {syncingId === connection.id || isBusy(connection.sync_status) ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <RefreshCw className="h-3.5 w-3.5" />
                      )}
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-destructive hover:text-destructive"
                      title="Detach source"
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
    </div>
  );
}

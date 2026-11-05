"use client";

import { useCallback, useEffect, useState } from "react";
import {
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

type DriveFolder = {
  id: string;
  folder_id: string;
  folder_name: string | null;
  folder_url: string | null;
  access_mode?: string;
  enabled: boolean;
  sync_status: string;
  task_id?: string | null;
  last_error: string | null;
  last_synced_at: string | null;
  document_count: number;
};

type DriveSyncStatus = {
  connection_id: string;
  sync_status: string;
  task_id?: string | null;
  last_error: string | null;
  last_synced_at: string | null;
  document_count: number;
  folder_name?: string | null;
  access_mode?: string;
};

function normalizeStatus(status: string) {
  switch (status) {
    case "ok":
      return "completed";
    case "syncing":
      return "processing";
    case "error":
      return "failed";
    default:
      return status || "idle";
  }
}

function statusLabel(status: string) {
  switch (normalizeStatus(status)) {
    case "completed":
      return "Completed";
    case "processing":
      return "Processing…";
    case "queued":
      return "Queued";
    case "failed":
      return "Failed";
    case "idle":
      return "Idle";
    default:
      return status || "—";
  }
}

function statusClass(status: string) {
  switch (normalizeStatus(status)) {
    case "completed":
      return "text-emerald-600 dark:text-emerald-400";
    case "processing":
    case "queued":
      return "text-sky-600 dark:text-sky-400";
    case "failed":
      return "text-destructive";
    default:
      return "text-muted-foreground";
  }
}

function accessLabel(mode?: string) {
  return mode === "oauth" ? "Private (OAuth)" : "Public link";
}

function isBusy(status: string) {
  const normalized = normalizeStatus(status);
  return normalized === "queued" || normalized === "processing";
}

export default function GoogleDriveSettings() {
  const [connected, setConnected] = useState(false);
  const [folders, setFolders] = useState<DriveFolder[]>([]);
  const [loading, setLoading] = useState(true);
  const [connecting, setConnecting] = useState(false);
  const [folderInput, setFolderInput] = useState("");
  const [adding, setAdding] = useState(false);
  const [syncingId, setSyncingId] = useState<string | null>(null);
  const [syncingAll, setSyncingAll] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const data = await api.get<{
        connected: boolean;
        folders: DriveFolder[];
      }>("/api/v1/drive/folders");
      setConnected(data.connected);
      setFolders(data.folders || []);
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Could not load Drive folders",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const busyKey = folders
    .filter((f) => isBusy(f.sync_status))
    .map((f) => f.id)
    .sort()
    .join(",");

  // Poll lightweight per-folder status while any sync is active
  useEffect(() => {
    if (!busyKey) return;
    const busyIds = busyKey.split(",");
    let cancelled = false;

    const poll = async () => {
      try {
        const updates = await Promise.all(
          busyIds.map(async (id) => {
            try {
              return await api.get<DriveSyncStatus>(
                `/api/v1/drive/folders/${id}/status`,
              );
            } catch {
              return null;
            }
          }),
        );
        if (cancelled) return;
        setFolders((prev) =>
          prev.map((folder) => {
            const hit = updates.find(
              (u) => u && u.connection_id === folder.id,
            );
            if (!hit) return folder;
            return {
              ...folder,
              sync_status: hit.sync_status,
              task_id: hit.task_id,
              last_error: hit.last_error,
              last_synced_at: hit.last_synced_at,
              document_count: hit.document_count,
              folder_name: hit.folder_name ?? folder.folder_name,
            };
          }),
        );
      } catch {
        // ignore transient poll errors
      }
    };

    void poll();
    const timer = window.setInterval(() => {
      void poll();
    }, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [busyKey]);

  const handleConnect = async () => {
    setConnecting(true);
    try {
      const data = await api.get<{ auth_url: string }>(
        "/api/v1/auth/google-drive/connect",
      );
      window.location.assign(data.auth_url);
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Failed to start Drive connect",
      );
      setConnecting(false);
    }
  };

  const handleDisconnect = async () => {
    setConnecting(true);
    try {
      await api.delete("/api/v1/auth/google-drive/disconnect");
      setConnected(false);
      toast.success("Google Drive disconnected");
      await refresh();
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Failed to disconnect Drive",
      );
    } finally {
      setConnecting(false);
    }
  };

  const parseUrls = (raw: string): string[] =>
    raw
      .split(/[\n,]+/)
      .map((s) => s.trim())
      .filter(Boolean);

  const handleAddFolders = async () => {
    const urls = parseUrls(folderInput);
    if (!urls.length) {
      toast.message("Paste one or more Google Drive folder links");
      return;
    }
    setAdding(true);
    let ok = 0;
    let oauthNeeded = false;
    try {
      for (const url of urls) {
        try {
          await api.post("/api/v1/drive/folders", { url });
          ok += 1;
        } catch (err) {
          const message =
            err instanceof Error ? err.message : `Could not add folder`;
          if (
            /private|connect google drive|oauth/i.test(message) ||
            message.includes("DRIVE_OAUTH_REQUIRED")
          ) {
            oauthNeeded = true;
          }
          toast.error(message);
        }
      }
      if (ok) {
        toast.success(
          ok === 1
            ? "Folder attached — sync queued in background"
            : `${ok} folders attached — sync queued in background`,
        );
        setFolderInput("");
        await refresh();
      } else if (oauthNeeded && !connected) {
        toast.message(
          "Private folder — connect Google Drive, then attach the link again",
        );
      }
    } finally {
      setAdding(false);
    }
  };

  const handleRemove = async (folder: DriveFolder) => {
    try {
      await api.delete(`/api/v1/drive/folders/${folder.id}`);
      toast.success("Folder detached");
      await refresh();
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Could not remove folder",
      );
    }
  };

  const handleSyncOne = async (folder: DriveFolder) => {
    if (folder.access_mode === "oauth" && !connected) {
      toast.error("Connect Google Drive to sync this private folder");
      return;
    }
    setSyncingId(folder.id);
    try {
      const result = await api.post<{
        message?: string;
        sync_status?: string;
        already_running?: boolean;
      }>(`/api/v1/drive/folders/${folder.id}/sync`);
      toast.success(
        result.already_running
          ? "Sync already running — progress will update here"
          : "Sync queued — you can keep using the app",
      );
      await refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Sync failed");
      await refresh();
    } finally {
      setSyncingId(null);
    }
  };

  const handleSyncAll = async () => {
    if (!folders.length) return;
    setSyncingAll(true);
    try {
      const result = await api.post<{
        message?: string;
        error?: string;
        already_running?: boolean;
      }>("/api/v1/drive/sync");
      if (result.error) {
        toast.error(result.error);
      } else {
        toast.success(
          result.already_running
            ? "Sync already running"
            : result.message || "All folder syncs queued",
        );
      }
      await refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Sync failed");
    } finally {
      setSyncingAll(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading Drive settings…
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-medium">Google Drive</p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Public shared folders sync without Google login. Sync runs in the
            background so you can keep working.
            {connected ? " · OAuth connected." : ""}
          </p>
        </div>
        {connected ? (
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
            disabled={connecting}
            onClick={() => void handleConnect()}
          >
            {connecting ? (
              <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
            ) : (
              <Link2 className="mr-1.5 h-3.5 w-3.5" />
            )}
            Connect Drive
          </Button>
        )}
      </div>

      <div className="space-y-2">
        <label className="text-xs font-medium text-muted-foreground">
          Folder links (one per line)
        </label>
        <textarea
          value={folderInput}
          onChange={(e) => setFolderInput(e.target.value)}
          placeholder="https://drive.google.com/drive/folders/…"
          rows={3}
          className="w-full resize-y rounded-md border border-input bg-background px-3 py-2 text-sm outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring"
          disabled={adding}
        />
        <div className="flex flex-wrap gap-2">
          <Button
            type="button"
            size="sm"
            disabled={adding || !folderInput.trim()}
            onClick={() => void handleAddFolders()}
          >
            {adding ? (
              <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
            ) : (
              <FolderOpen className="mr-1.5 h-3.5 w-3.5" />
            )}
            Attach folders
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={syncingAll || folders.length === 0}
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
      </div>

      <div className="space-y-2">
        <p className="text-xs font-medium text-muted-foreground">
          Attached folders ({folders.length})
        </p>
        {folders.length === 0 ? (
          <p className="rounded-md border border-dashed border-border px-3 py-4 text-center text-xs text-muted-foreground">
            No folders yet. Paste a public or private Drive folder link above.
          </p>
        ) : (
          <ul className="max-h-56 space-y-2 overflow-y-auto">
            {folders.map((folder) => (
              <li
                key={folder.id}
                className="rounded-md border border-border bg-muted/30 px-3 py-2"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">
                      {folder.folder_name || folder.folder_id}
                    </p>
                    <p className="text-[11px] text-muted-foreground">
                      {accessLabel(folder.access_mode)}
                      <span className={statusClass(folder.sync_status)}>
                        {" · "}
                        {statusLabel(folder.sync_status)}
                      </span>
                      {folder.document_count > 0
                        ? ` · ${folder.document_count} PDF${folder.document_count === 1 ? "" : "s"}`
                        : ""}
                      {folder.last_synced_at
                        ? ` · ${new Date(folder.last_synced_at).toLocaleString()}`
                        : ""}
                    </p>
                    {folder.last_error ? (
                      <p className="mt-0.5 line-clamp-2 text-[11px] text-destructive">
                        {folder.last_error}
                      </p>
                    ) : null}
                  </div>
                  <div className="flex shrink-0 items-center gap-0.5">
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      title="Sync folder"
                      disabled={
                        syncingId === folder.id || isBusy(folder.sync_status)
                      }
                      onClick={() => void handleSyncOne(folder)}
                    >
                      {syncingId === folder.id || isBusy(folder.sync_status) ? (
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
                      title="Detach folder"
                      onClick={() => void handleRemove(folder)}
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

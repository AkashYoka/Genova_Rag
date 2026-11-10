"use client";

import { useCallback, useEffect, useState } from "react";
import { ChevronDown, Loader2, RefreshCw, Trash2, Unplug } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  GoogleDriveMark,
  IntegrationMethod,
  IntegrationProviderCard,
  IntegrationStatusPill,
} from "@/components/settings/IntegrationProviderCard";

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
      return "Syncing…";
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

function accessLabel(mode?: string) {
  return mode === "oauth" ? "Connected account" : "Public link";
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
  const [manageOpen, setManageOpen] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const data = await api.get<{
        connected: boolean;
        folders: DriveFolder[];
      }>("/api/v1/drive/folders");
      setConnected(data.connected);
      setFolders(data.folders || []);
      if ((data.folders || []).length > 0) setManageOpen(true);
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Could not load Drive folders",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const data = await api.get<{
          connected: boolean;
          folders: DriveFolder[];
        }>("/api/v1/drive/folders");
        if (cancelled) return;
        setConnected(data.connected);
        setFolders(data.folders || []);
        if ((data.folders || []).length > 0) setManageOpen(true);
      } catch (err) {
        if (!cancelled) {
          toast.error(
            err instanceof Error ? err.message : "Could not load Drive folders",
          );
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const busyKey = folders
    .filter((f) => isBusy(f.sync_status))
    .map((f) => f.id)
    .sort()
    .join(",");

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
      toast.message("Paste a Google Drive folder link");
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
            err instanceof Error ? err.message : "Could not add folder";
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
            ? "Folder attached — sync started"
            : `${ok} folders attached — sync started`,
        );
        setFolderInput("");
        setManageOpen(true);
        await refresh();
      } else if (oauthNeeded && !connected) {
        toast.message(
          "Private folder — connect Google Drive, then import the link again",
        );
      }
    } finally {
      setAdding(false);
    }
  };

  const handleRemove = async (folder: DriveFolder) => {
    try {
      await api.delete(`/api/v1/drive/folders/${folder.id}`);
      toast.success("Folder removed");
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
          ? "Sync already running"
          : "Sync queued",
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
            : result.message || "Sync queued",
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
      <div className="flex items-center gap-2 py-8 text-sm text-[var(--gd-muted)]">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading Google Drive…
      </div>
    );
  }

  const anyBusy = folders.some((f) => isBusy(f.sync_status));

  return (
    <IntegrationProviderCard
      name="Google Drive"
      description="Bring PDFs from Drive into Genova for search and chat."
      icon={<GoogleDriveMark />}
      status={
        connected ? (
          <IntegrationStatusPill tone="ok">Connected</IntegrationStatusPill>
        ) : anyBusy ? (
          <IntegrationStatusPill tone="busy">Syncing</IntegrationStatusPill>
        ) : folders.length > 0 ? (
          <IntegrationStatusPill tone="idle">
            {folders.length} source{folders.length === 1 ? "" : "s"}
          </IntegrationStatusPill>
        ) : (
          <IntegrationStatusPill tone="idle">Not connected</IntegrationStatusPill>
        )
      }
    >
      <IntegrationMethod
        label="Connect your account"
        hint="Sign in with Google to sync private Drive folders."
      >
        {connected ? (
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-[13px] text-[var(--gd-muted)]">
              Google account linked
            </p>
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
            disabled={connecting}
            onClick={() => void handleConnect()}
          >
            {connecting ? (
              <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
            ) : (
              <GoogleDriveMark className="mr-1.5 h-4 w-4" />
            )}
            Connect Google Drive
          </Button>
        )}
      </IntegrationMethod>

      <IntegrationMethod
        label="Import with a link"
        hint="Paste a public Drive folder link. Private folders need Connect first."
      >
        <div className="flex flex-col gap-2 sm:flex-row sm:items-stretch">
          <input
            value={folderInput}
            onChange={(e) => setFolderInput(e.target.value)}
            placeholder="https://drive.google.com/drive/folders/…"
            className="h-9 w-full flex-1 rounded-md border border-[var(--gd-border)] bg-[var(--gd-ground)] px-3 text-[13px] text-[var(--gd-text)] outline-none placeholder:text-[var(--gd-faint)] focus-visible:ring-2 focus-visible:ring-[var(--gd-accent)]"
            disabled={adding}
          />
          <Button
            type="button"
            size="sm"
            className="h-9 shrink-0"
            disabled={adding || !folderInput.trim()}
            onClick={() => void handleAddFolders()}
          >
            {adding ? (
              <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
            ) : null}
            Import
          </Button>
        </div>
      </IntegrationMethod>

      <div className="mt-auto border-t border-[var(--gd-rule)] pt-4">
        <button
          type="button"
          className="flex w-full items-center justify-between text-left text-[13px] font-medium text-[var(--gd-text)]"
          onClick={() => setManageOpen((open) => !open)}
          aria-expanded={manageOpen}
        >
          <span>
            Manage sources
            {folders.length > 0 ? (
              <span className="ml-1.5 font-normal text-[var(--gd-muted)]">
                ({folders.length})
              </span>
            ) : null}
          </span>
          <ChevronDown
            className={`h-4 w-4 text-[var(--gd-muted)] transition-transform ${manageOpen ? "rotate-180" : ""}`}
          />
        </button>

        {manageOpen ? (
          <div className="mt-3 space-y-3">
            {folders.length > 0 ? (
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

            {folders.length === 0 ? (
              <p className="rounded-md border border-dashed border-[var(--gd-border)] px-3 py-4 text-center text-[12px] text-[var(--gd-muted)]">
                No Drive folders attached yet
              </p>
            ) : (
              <ul className="max-h-52 space-y-2 overflow-y-auto">
                {folders.map((folder) => (
                  <li
                    key={folder.id}
                    className="rounded-md border border-[var(--gd-border)] bg-[var(--gd-ground)] px-3 py-2.5"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate text-[13px] font-medium text-[var(--gd-text)]">
                          {folder.folder_name || folder.folder_id}
                        </p>
                        <p className="mt-0.5 text-[11px] text-[var(--gd-muted)]">
                          {accessLabel(folder.access_mode)}
                          {" · "}
                          <span
                            className={
                              normalizeStatus(folder.sync_status) === "failed"
                                ? "text-[var(--gd-negative)]"
                                : isBusy(folder.sync_status)
                                  ? "text-[var(--gd-accent)]"
                                  : undefined
                            }
                          >
                            {statusLabel(folder.sync_status)}
                          </span>
                          {folder.document_count > 0
                            ? ` · ${folder.document_count} PDF${folder.document_count === 1 ? "" : "s"}`
                            : ""}
                        </p>
                        {folder.last_error ? (
                          <p className="mt-1 line-clamp-2 text-[11px] text-[var(--gd-negative)]">
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
                          {syncingId === folder.id ||
                          isBusy(folder.sync_status) ? (
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
                          title="Remove folder"
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
        ) : null}
      </div>
    </IntegrationProviderCard>
  );
}

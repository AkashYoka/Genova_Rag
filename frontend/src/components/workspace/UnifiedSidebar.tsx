"use client";

/**
 * UnifiedSidebar
 *
 * Navigation for the desk: start a conversation, open the source library,
 * and switch between recent chats. Active / Trash filtering lives in the
 * documents view itself — not duplicated here.
 */

import { useEffect, useMemo, useState } from "react";
import {
  Edit2,
  FolderOpen,
  Loader2,
  MessageSquare,
  Plus,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { useChatStore, type ChatSession } from "@/store/chat-store";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import type {
  DocStatusFilter,
  DocTab,
  DocumentLibrary,
} from "@/hooks/useDocumentLibrary";

export type WorkspaceView = "workspace" | "documents";

interface UnifiedSidebarProps {
  view: WorkspaceView;
  docTab: DocTab;
  docFilter: DocStatusFilter;
  counts: DocumentLibrary["counts"];
  onOpenWorkspace: () => void;
  onOpenDocuments: (opts: { tab: DocTab; filter: DocStatusFilter }) => void;
  mobile?: boolean;
}

const ROW_BASE =
  "group flex h-8 w-full items-center gap-2 rounded-md px-2 text-left text-[13px] leading-4 outline-none transition-colors focus-visible:ring-2 focus-visible:ring-[var(--gd-accent)]/30";
const ROW_IDLE =
  "text-[var(--gd-muted)] hover:bg-[var(--gd-raised)] hover:text-[var(--gd-text)]";
const ROW_ACTIVE = "bg-[var(--gd-active)] font-medium text-[var(--gd-text)]";

export default function UnifiedSidebar({
  view,
  counts,
  onOpenWorkspace,
  onOpenDocuments,
  mobile = false,
}: UnifiedSidebarProps) {
  const sessions = useChatStore((state) => state.sessions);
  const activeSessionId = useChatStore((state) => state.activeSessionId);
  const fetchSessions = useChatStore((state) => state.fetchSessions);
  const createSession = useChatStore((state) => state.createSession);
  const renameSession = useChatStore((state) => state.renameSession);
  const deleteSession = useChatStore((state) => state.deleteSession);
  const setActiveSessionId = useChatStore((state) => state.setActiveSessionId);
  const fetchSessionHistory = useChatStore(
    (state) => state.fetchSessionHistory,
  );

  const [sessionQuery, setSessionQuery] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    void fetchSessions();
  }, [fetchSessions]);

  const filteredSessions = useMemo(() => {
    const q = sessionQuery.trim().toLowerCase();
    if (!q) return sessions;
    return sessions.filter((s) => s.title.toLowerCase().includes(q));
  }, [sessions, sessionQuery]);

  const inDocuments = view === "documents";
  const indexingCount = counts.indexing;

  const handleCreate = async () => {
    if (creating) return;
    setCreating(true);
    try {
      const title = `Chat ${sessions.length + 1}`;
      const id = await createSession(title);
      setEditingId(id);
      setEditTitle(title);
      onOpenWorkspace();
    } catch (err) {
      console.error(err);
    } finally {
      setCreating(false);
    }
  };

  const handleSaveRename = async (id: string, e?: React.FormEvent) => {
    e?.preventDefault();
    const nextTitle = editTitle.trim();
    setEditingId(null);
    if (!nextTitle) return;
    try {
      await renameSession(id, nextTitle);
    } catch (err) {
      console.error(err);
    }
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm("Delete this chat session? Its messages will be removed.")) {
      return;
    }
    try {
      await deleteSession(id);
    } catch (err) {
      console.error(err);
    }
  };

  const handleSelectSession = async (session: ChatSession) => {
    setEditingId(null);
    setActiveSessionId(session.id);
    await fetchSessionHistory(session.id);
    if (useChatStore.getState().activeSessionId !== session.id) {
      setActiveSessionId(session.id);
    }
    onOpenWorkspace();
  };

  return (
    <div
      className={cn(
        "flex h-full min-h-0 w-full flex-col overflow-hidden",
        mobile && "pb-[env(safe-area-inset-bottom,0px)]",
      )}
    >
      <div className="shrink-0 px-2.5 pt-2.5">
        <Button
          onClick={() => void handleCreate()}
          disabled={creating}
          data-sidebar-nav
          className="h-8 w-full justify-start gap-2 rounded-md bg-[var(--gd-primary)] px-2.5 text-[13px] font-medium text-[var(--gd-primary-ink)] shadow-none hover:bg-[var(--gd-primary)]/90"
        >
          {creating ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
          ) : (
            <Plus className="h-3.5 w-3.5" aria-hidden />
          )}
          New conversation
        </Button>
      </div>

      <nav className="shrink-0 px-2.5 pt-4" aria-label="Sources">
        <p className="px-2 pb-1.5 text-[11px] font-medium text-[var(--gd-faint)]">
          Sources
        </p>
        <div className="flex flex-col gap-0.5">
          <button
            type="button"
            data-sidebar-nav
            title="Browse and manage the files the assistant can search"
            aria-current={inDocuments ? "page" : undefined}
            onClick={() => onOpenDocuments({ tab: "active", filter: "all" })}
            className={cn(ROW_BASE, inDocuments ? ROW_ACTIVE : ROW_IDLE)}
          >
            <FolderOpen className="h-3.5 w-3.5 shrink-0" aria-hidden />
            <span className="min-w-0 flex-1 truncate">Browse library</span>
            {counts.active > 0 ? (
              <span className="num shrink-0 text-[11px] text-[var(--gd-faint)]">
                {counts.active}
              </span>
            ) : null}
          </button>

          {indexingCount > 0 ? (
            <div
              className="mx-1 mt-0.5 flex items-center gap-1.5 rounded-md px-2 py-1.5 text-[11px] text-[var(--gd-warning)]"
              role="status"
              aria-live="polite"
            >
              <Loader2
                className="h-3 w-3 shrink-0 animate-spin motion-reduce:animate-none"
                aria-hidden
              />
              <span className="min-w-0 flex-1 truncate">
                {indexingCount} preparing for search
              </span>
            </div>
          ) : null}
        </div>
      </nav>

      <div
        className={cn(
          "mt-4 flex min-h-0 flex-1 flex-col border-t border-[var(--gd-border)] px-2.5 pt-3",
          mobile && "pb-3",
        )}
      >
        <div className="flex items-center justify-between px-2 pb-1.5">
          <p className="text-[11px] font-medium text-[var(--gd-faint)]">
            Recent chats
          </p>
          <span className="num text-[11px] text-[var(--gd-faint)]">
            {sessions.length}
          </span>
        </div>

        <div className="relative px-1 pb-2">
          <Search
            className="pointer-events-none absolute left-3 top-[40%] h-3.5 w-3.5 -translate-y-1/2 text-[var(--gd-faint)]"
            aria-hidden
          />
          <Input
            value={sessionQuery}
            onChange={(e) => setSessionQuery(e.target.value)}
            placeholder="Search chats"
            aria-label="Search chat sessions"
            className="h-7 rounded-md border-[var(--gd-border)] bg-[var(--gd-surface)] pl-7 pr-7 text-xs"
          />
          {sessionQuery ? (
            <button
              type="button"
              onClick={() => setSessionQuery("")}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 rounded p-0.5 text-[var(--gd-faint)] hover:text-[var(--gd-text)]"
              aria-label="Clear chat search"
            >
              <X className="h-3 w-3" />
            </button>
          ) : null}
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto pb-2">
          {filteredSessions.length === 0 ? (
            <p className="px-2 py-3 text-xs leading-relaxed text-[var(--gd-muted)]">
              {sessions.length === 0
                ? "No conversations yet. Start one to ask about a document."
                : `No chats match “${sessionQuery.trim()}”.`}
            </p>
          ) : (
            <div className="flex flex-col gap-0.5">
              {filteredSessions.map((session) => {
                const isActive =
                  session.id === activeSessionId && view === "workspace";
                const isEditing = session.id === editingId;
                return (
                  <div
                    key={session.id}
                    role="button"
                    tabIndex={isEditing ? -1 : 0}
                    data-sidebar-nav
                    aria-current={isActive ? "true" : undefined}
                    aria-label={`Open chat session ${session.title}`}
                    onClick={() => void handleSelectSession(session)}
                    onKeyDown={(e) => {
                      if (isEditing) return;
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        void handleSelectSession(session);
                      }
                    }}
                    className={cn(
                      ROW_BASE,
                      "relative cursor-pointer pr-1",
                      isActive ? ROW_ACTIVE : ROW_IDLE,
                    )}
                  >
                    <MessageSquare
                      className="h-3.5 w-3.5 shrink-0"
                      aria-hidden
                    />
                    {isEditing ? (
                      <form
                        onSubmit={(e) => void handleSaveRename(session.id, e)}
                        className="flex min-w-0 flex-1 items-center gap-1"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <Input
                          value={editTitle}
                          onChange={(e) => setEditTitle(e.target.value)}
                          autoFocus
                          onBlur={() => void handleSaveRename(session.id)}
                          onKeyDown={(e) => {
                            if (e.key === "Escape") {
                              e.preventDefault();
                              e.stopPropagation();
                              setEditingId(null);
                            }
                          }}
                          aria-label={`Rename chat session ${session.title}`}
                          className="h-6 w-full border-[var(--gd-border)] bg-[var(--gd-surface)] px-1 py-0 text-xs"
                        />
                      </form>
                    ) : (
                      <span className="min-w-0 flex-1 truncate">
                        {session.title}
                      </span>
                    )}

                    {!isEditing ? (
                      <span className="flex shrink-0 items-center opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100 group-focus-within:opacity-100">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setEditingId(session.id);
                            setEditTitle(session.title);
                          }}
                          className="rounded p-1 text-[var(--gd-faint)] hover:bg-[var(--gd-surface)] hover:text-[var(--gd-text)]"
                          aria-label={`Rename chat session ${session.title}`}
                        >
                          <Edit2 className="h-3 w-3" />
                        </button>
                        <button
                          type="button"
                          onClick={(e) => void handleDelete(session.id, e)}
                          className="rounded p-1 text-[var(--gd-faint)] hover:bg-[var(--gd-surface)] hover:text-[var(--gd-negative)]"
                          aria-label={`Delete chat session ${session.title}`}
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      </span>
                    ) : null}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

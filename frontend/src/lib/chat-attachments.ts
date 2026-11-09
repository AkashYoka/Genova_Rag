import type { AttachedDocument } from "@/store/chat-store";

export const ATTACH_STORAGE_PREFIX = "genova-chat-attachments:";
const PENDING_DOC_CHAT_KEY = "genova:pending-doc-chat";

/** In-memory guard so React Strict Mode cannot consume the same nonce twice. */
const consumedPendingNonces = new Set<string>();

export interface PendingDocChat {
  docId: string;
  filename: string;
  nonce: string;
}

export function loadSessionAttachments(
  sessionId: string | null,
): AttachedDocument[] {
  if (!sessionId || typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(`${ATTACH_STORAGE_PREFIX}${sessionId}`);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const valid = parsed.filter(
      (item): item is AttachedDocument =>
        !!item &&
        typeof item.id === "string" &&
        typeof item.filename === "string",
    );
    const byName = new Map<string, AttachedDocument>();
    for (const item of valid) {
      byName.set(item.filename.toLowerCase(), item);
    }
    return Array.from(byName.values()).slice(0, 20);
  } catch {
    return [];
  }
}

export function saveSessionAttachments(
  sessionId: string | null,
  docs: AttachedDocument[],
): void {
  if (!sessionId || typeof window === "undefined") return;
  try {
    localStorage.setItem(
      `${ATTACH_STORAGE_PREFIX}${sessionId}`,
      JSON.stringify(docs),
    );
  } catch {
    // ignore quota / private mode
  }
}

export function chatTitleFromFilename(filename: string): string {
  const base = filename.replace(/\.[^.]+$/, "").trim() || filename.trim();
  if (!base) return "New chat";
  return base.length > 48 ? `${base.slice(0, 45)}…` : base;
}

/** Queue a document so the dashboard opens a fresh session with it attached. */
export function queueDocChat(doc: {
  id: string;
  filename: string;
}): void {
  if (typeof window === "undefined") return;
  const pending: PendingDocChat = {
    docId: doc.id,
    filename: doc.filename,
    nonce: crypto.randomUUID(),
  };
  try {
    sessionStorage.setItem(PENDING_DOC_CHAT_KEY, JSON.stringify(pending));
  } catch {
    // sessionStorage unavailable — dashboard will still open the PDF via docId
  }
}

/** Take a queued doc-chat request once (safe across Strict Mode remounts). */
export function takePendingDocChat(): PendingDocChat | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = sessionStorage.getItem(PENDING_DOC_CHAT_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<PendingDocChat>;
    if (
      !parsed ||
      typeof parsed.docId !== "string" ||
      typeof parsed.filename !== "string" ||
      typeof parsed.nonce !== "string"
    ) {
      sessionStorage.removeItem(PENDING_DOC_CHAT_KEY);
      return null;
    }
    if (consumedPendingNonces.has(parsed.nonce)) {
      return null;
    }
    consumedPendingNonces.add(parsed.nonce);
    sessionStorage.removeItem(PENDING_DOC_CHAT_KEY);
    return parsed as PendingDocChat;
  } catch {
    return null;
  }
}

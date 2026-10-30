"use client";

import { toast } from "sonner";
import { useState, useRef, useEffect } from "react";
import { useTranslation } from "react-i18next";
import type { DocInfo } from "@/app/dashboard/page";
import { api, API_BASE } from "@/lib/api";
import {
  useChatStore,
  type AttachedDocument,
  type ChatMsg,
  type LlmUsage,
  type SourceBoundingBox,
  type SourceChunk,
} from "@/store/chat-store";
import { buttonVariants } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import MessageBubble from "./MessageBubble";
import SourceCard from "./SourceCard";
import QuizCard from "./QuizCard";
import type { QuizPayload } from "@/store/chat-store";
import {
  Send,
  Loader2,
  Trash2,
  MessageSquare,
  Download,
  Mic,
  MicOff,
  ChevronDown,
  Paperclip,
  Cpu,
  FileText,
  Check,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

const ATTACH_STORAGE_PREFIX = "genova-chat-attachments:";

function loadSessionAttachments(sessionId: string | null): AttachedDocument[] {
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
    // Keep one chip per filename (last wins) so restored sessions aren't duplicated.
    const byName = new Map<string, AttachedDocument>();
    for (const item of valid) {
      byName.set(item.filename.toLowerCase(), item);
    }
    return Array.from(byName.values()).slice(0, 20);
  } catch {
    return [];
  }
}

function saveSessionAttachments(
  sessionId: string | null,
  docs: AttachedDocument[],
) {
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

interface ISpeechRecognitionEvent {
  resultIndex: number;
  results: {
    length: number;
    [index: number]: {
      [index: number]: {
        transcript: string;
      };
      isFinal: boolean;
    };
  };
}

interface ISpeechRecognitionErrorEvent {
  error: string;
  message: string;
}

interface ISpeechRecognition {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: ISpeechRecognitionEvent) => void) | null;
  onerror: ((event: ISpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
}

interface WindowWithSpeech extends Window {
  SpeechRecognition?: new () => ISpeechRecognition;
  webkitSpeechRecognition?: new () => ISpeechRecognition;
}

interface CitationTarget {
  page: number;
  highlightRects?: SourceBoundingBox[];
}

interface Props {
  activeDoc: DocInfo | null;
  documents?: DocInfo[];
  onCitationClick: (target: CitationTarget) => void;
}

export default function ChatPanel({
  activeDoc,
  documents = [],
  onCitationClick,
}: Props) {
  const { t, i18n } = useTranslation();
  const messages = useChatStore((state) => state.messages);
  const input = useChatStore((state) => state.input);
  const streaming = useChatStore((state) => state.streaming);
  const isTyping = useChatStore((state) => state.isTyping);
  const historyLoading = useChatStore((state) => state.historyLoading);
  const activeSessionId = useChatStore((state) => state.activeSessionId);
  const setMessages = useChatStore((state) => state.setMessages);
  const setInput = useChatStore((state) => state.setInput);
  const setStreaming = useChatStore((state) => state.setStreaming);
  const setIsTyping = useChatStore((state) => state.setIsTyping);
  const fetchSessionHistory = useChatStore(
    (state) => state.fetchSessionHistory,
  );

  const [showExportMenu, setShowExportMenu] = useState(false);
  const [attachedDocs, setAttachedDocs] = useState<AttachedDocument[]>([]);
  const [attachPickerOpen, setAttachPickerOpen] = useState(false);
  const [attachFilter, setAttachFilter] = useState("");
  const MAX_CHARACTERS = 2000;
  const [isRecording, setIsRecording] = useState(false);
  const [speechError, setSpeechError] = useState<string | null>(null);
  const [llmProvider, setLlmProvider] = useState<"ollama" | "bedrock">(
    "ollama",
  );

  const applyUsageToMessage = (
    assistantId: string,
    usage: LlmUsage | undefined,
    extra: Partial<ChatMsg> = {},
  ) => {
    if (!usage && Object.keys(extra).length === 0) return;
    setMessages((prev) =>
      prev.map((m) =>
        m.id === assistantId
          ? {
            ...m,
            ...extra,
            usage: usage ?? m.usage,
            llm_provider: usage?.provider ?? m.llm_provider ?? llmProvider,
          }
          : m,
      ),
    );
  };

  const applyQuizToMessage = (assistantId: string, quiz: QuizPayload) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === assistantId ? { ...m, quiz } : m)),
    );
  };

  const ensureAssistantPlaceholder = (
    assistantId: string,
    assistantCreated: boolean,
    seedContent = "",
  ) => {
    if (assistantCreated) return true;
    setIsTyping(false);
    const assistantMsg: ChatMsg = {
      id: assistantId,
      role: "assistant",
      content: seedContent,
      sources: [],
      isStreaming: true,
      llm_provider: llmProvider,
    };
    setMessages((prev) => [...prev, assistantMsg]);
    return true;
  };

  const [showScrollButton, setShowScrollButton] = useState(false);

  const recognitionRef = useRef<ISpeechRecognition | null>(null);
  const initialInputRef = useRef<string>("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const prevDocId = useRef<string | null>(null);
  const exportMenuRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const showEmptyState = messages.length === 0 && !isTyping && !historyLoading;

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    textarea.style.height = "auto";
    const computedMaxHeight = Number.parseFloat(
      window.getComputedStyle(textarea).maxHeight,
    );
    const maxHeight = Number.isFinite(computedMaxHeight)
      ? computedMaxHeight
      : textarea.scrollHeight;
    const nextHeight = Math.min(textarea.scrollHeight, maxHeight);
    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY =
      textarea.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [input]);

  // Auto-scroll to bottom whenever messages change
  useEffect(() => {
    if (containerRef.current) {
      const { scrollHeight, scrollTop, clientHeight } = containerRef.current;
      if (scrollHeight - scrollTop - clientHeight < 150) {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
      }
    } else {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    setShowScrollButton(scrollTop < scrollHeight - clientHeight - 100);
  };

  const scrollToBottom = () => {
    if (containerRef.current) {
      containerRef.current.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  };

  // Abort in-flight generation on unmount. Do not call resetChat() here —
  // it clears the shared Zustand store (sessions + messages). React Strict Mode
  // remounts in dev would wipe the Chat Sessions sidebar after fetchSessions.
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
      setStreaming(false);
      setIsTyping(false);
    };
  }, [setStreaming, setIsTyping]);

  const seededAttachSessionRef = useRef<string | null>(null);

  // Restore session-scoped attachments when switching chats
  useEffect(() => {
    seededAttachSessionRef.current = null;
    setAttachedDocs(loadSessionAttachments(activeSessionId));
    setAttachPickerOpen(false);
    setAttachFilter("");
  }, [activeSessionId]);

  useEffect(() => {
    saveSessionAttachments(activeSessionId, attachedDocs);
  }, [activeSessionId, attachedDocs]);

  // After history loads, seed attachments from the latest user message if storage empty
  useEffect(() => {
    if (!activeSessionId) return;
    if (seededAttachSessionRef.current === activeSessionId) return;
    if (attachedDocs.length > 0) {
      seededAttachSessionRef.current = activeSessionId;
      return;
    }
    if (historyLoading || messages.length === 0) return;
    seededAttachSessionRef.current = activeSessionId;
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const msg = messages[i];
      if (msg.role === "user" && msg.attachments && msg.attachments.length > 0) {
        setAttachedDocs(msg.attachments);
        break;
      }
    }
  }, [messages, activeSessionId, attachedDocs.length, historyLoading]);

  // One ready row per filename so the attach picker can't show chemistry twice.
  const readyDocs = (() => {
    const byName = new Map<string, DocInfo>();
    for (const doc of documents.filter((d) => d.status === "ready")) {
      const key = doc.original_name.toLowerCase();
      const existing = byName.get(key);
      if (!existing) {
        byName.set(key, doc);
        continue;
      }
      const betterChunks =
        (doc.chunk_count || 0) > (existing.chunk_count || 0);
      const sameChunksNewer =
        (doc.chunk_count || 0) === (existing.chunk_count || 0) &&
        (doc.uploaded_at || "") > (existing.uploaded_at || "");
      if (betterChunks || sameChunksNewer) {
        byName.set(key, doc);
      }
    }
    return Array.from(byName.values());
  })();
  const attachQuery = input.trim().toLowerCase();
  const isAttachCommand =
    attachQuery === "/attach" || attachQuery.startsWith("/attach ");
  const attachSearch = isAttachCommand
    ? input.trim().slice("/attach".length).trim().toLowerCase()
    : attachFilter.trim().toLowerCase();
  const attachCandidates = readyDocs.filter((d) =>
    attachSearch
      ? d.original_name.toLowerCase().includes(attachSearch)
      : true,
  );

  const toggleAttachedDoc = (doc: DocInfo) => {
    setAttachedDocs((prev) => {
      const sameName = (d: AttachedDocument) =>
        d.filename.toLowerCase() === doc.original_name.toLowerCase();
      const existsSameId = prev.some((d) => d.id === doc.id);
      if (existsSameId) {
        return prev.filter((d) => d.id !== doc.id);
      }
      // Collapse any other chip with the same display name, then add this ID.
      const withoutNameDupes = prev.filter((d) => !sameName(d));
      return [
        ...withoutNameDupes,
        { id: doc.id, filename: doc.original_name },
      ];
    });
  };

  const removeAttachedDoc = (id: string) => {
    setAttachedDocs((prev) => prev.filter((d) => d.id !== id));
  };

  // Load history on activeSessionId or fallback to activeDoc change
  useEffect(() => {
    if (activeSessionId) {
      fetchSessionHistory(activeSessionId);
      return;
    }

    if (!activeDoc) {
      prevDocId.current = null;
      setMessages([]);
      return;
    }

    if (activeDoc.id === prevDocId.current) return;

    const documentId = activeDoc.id;
    prevDocId.current = documentId;
    setMessages([]);
    let cancelled = false;

    api
      .get<{
        messages: Array<{
          id: string;
          role: string;
          content: string;
          sources?: SourceChunk[];
        }>;
      }>(`/api/v1/chat/history/${documentId}`)
      .then((data) => {
        if (cancelled || prevDocId.current !== documentId) return;

        setMessages(
          data.messages.map((m) => ({
            id: m.id,
            role: m.role as "user" | "assistant",
            content: m.content,
            sources: m.sources || [],
          })),
        );
      })
      .catch(() => {
        if (cancelled || prevDocId.current !== documentId) return;
        setMessages([]);
      });

    return () => {
      cancelled = true;
    };
  }, [activeSessionId, activeDoc, fetchSessionHistory, setMessages]);

  const handleStop = () => {
    abortControllerRef.current?.abort();
    setStreaming(false);
    setIsTyping(false);
  };

  const handleSend = async () => {
    if (!input.trim() || streaming) return;

    const question = input.trim();
    // /attach is a composer command, not a chat question
    if (question === "/attach" || question.toLowerCase().startsWith("/attach ")) {
      setAttachPickerOpen(true);
      setAttachFilter(question.slice("/attach".length).trim());
      setInput("");
      return;
    }

    setInput("");
    setAttachPickerOpen(false);

    const attachmentsSnapshot = [...attachedDocs];
    const documentIds =
      attachmentsSnapshot.length > 0
        ? attachmentsSnapshot.map((d) => d.id)
        : null;

    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    // Add user message
    const userMsg: ChatMsg = {
      id: `user-${Date.now()}`,
      role: "user",
      content: question,
      sources: [],
      attachments:
        attachmentsSnapshot.length > 0 ? attachmentsSnapshot : undefined,
    };
    setMessages((prev) => [...prev, userMsg]);

    const assistantId = `assistant-${Date.now()}`;
    // Create the assistant bubble immediately so tokens paint progressively
    // (ChatGPT-style) instead of waiting for the first SSE/WS event.
    let assistantCreated = ensureAssistantPlaceholder(assistantId, false);

    setStreaming(true);
    setIsTyping(false);

    // Coalesce rapid token events into one React update per animation frame
    // so the UI stays responsive while still streaming live.
    let pendingTokens = "";
    let rafHandle: number | null = null;
    const flushPendingTokens = () => {
      rafHandle = null;
      if (!pendingTokens) return;
      const chunk = pendingTokens;
      pendingTokens = "";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId ? { ...m, content: m.content + chunk } : m,
        ),
      );
    };
    const appendToken = (piece: string) => {
      if (!piece) return;
      if (!assistantCreated) {
        assistantCreated = ensureAssistantPlaceholder(
          assistantId,
          assistantCreated,
          piece,
        );
        return;
      }
      pendingTokens += piece;
      if (rafHandle == null) {
        rafHandle = window.requestAnimationFrame(flushPendingTokens);
      }
    };
    const flushTokensNow = () => {
      if (rafHandle != null) {
        window.cancelAnimationFrame(rafHandle);
        rafHandle = null;
      }
      flushPendingTokens();
    };

    try {
      // Try WebSocket first for real-time agentic thought streaming
      const token =
        typeof window !== "undefined" ? localStorage.getItem("token") : null;
      const base = API_BASE || window.location.origin;
      const wsScheme = base.startsWith("https")
        ? "wss"
        : base.startsWith("http")
          ? "ws"
          : "wss";
      const host = base.replace(/^https?:/, "");
      const wsUrl = `${wsScheme}:${host}/api/v1/chat/ws${token ? `?token=${encodeURIComponent(token)}` : ""}`;

      const ws = new WebSocket(wsUrl);

      let onAbort: (() => void) | null = null;

      const wsDone = new Promise<void>((resolve, reject) => {
        onAbort = () => {
          try {
            ws.close();
          } catch {
            // ignore
          }
          reject(new DOMException("The user aborted a request.", "AbortError"));
        };
        abortController.signal.addEventListener("abort", onAbort);

        ws.onopen = () => {
          // No attachments → search all indexed docs (omit document_id).
          // Attachments → restrict retrieval to those document_ids only.
          ws.send(
            JSON.stringify({
              question,
              document_id: null,
              document_ids: documentIds,
              attachments: attachmentsSnapshot,
              session_id: activeSessionId,
              llm_provider: llmProvider,
            }),
          );
        };

        // If WS doesn't open within 800ms, treat as failure and fallback
        const connectTimeout = setTimeout(() => {
          try {
            ws.close();
          } catch {
            // ignore
          }
          reject(new Error("WebSocket connection timeout"));
        }, 800);

        ws.onmessage = (ev) => {
          clearTimeout(connectTimeout);
          try {
            const event = JSON.parse(ev.data);
            if (event.type === "token") {
              appendToken(event.data as string);
            } else if (event.type === "status") {
              // Retrieving / generating — keep the streaming bubble visible.
              setIsTyping(false);
            } else if (event.type === "sources") {
              if (!assistantCreated) {
                assistantCreated = ensureAssistantPlaceholder(
                  assistantId,
                  assistantCreated,
                );
              }
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, sources: event.data as SourceChunk[] }
                    : m,
                ),
              );
            } else if (event.type === "quiz") {
              flushTokensNow();
              if (!assistantCreated) {
                assistantCreated = ensureAssistantPlaceholder(
                  assistantId,
                  assistantCreated,
                );
              }
              applyQuizToMessage(assistantId, event.data as QuizPayload);
            } else if (event.type === "thought") {
              appendToken(`\n[thought] ${event.data}`);
            } else if (event.type === "usage") {
              applyUsageToMessage(assistantId, event.data as LlmUsage);
            } else if (event.type === "error") {
              flushTokensNow();
              setIsTyping(false);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? {
                      ...m,
                      content: `Error: ${event.data}`,
                      isStreaming: false,
                    }
                    : m,
                ),
              );
              ws.close();
              reject(new Error(String(event.data)));
            } else if (event.type === "done") {
              flushTokensNow();
              const persistedId =
                typeof (event as { message_id?: string }).message_id === "string"
                  ? (event as { message_id: string }).message_id
                  : undefined;
              applyUsageToMessage(
                assistantId,
                (event as { usage?: LlmUsage }).usage,
                {
                  isStreaming: false,
                  response_time_ms: event.response_time_ms,
                  ...(persistedId ? { id: persistedId } : {}),
                },
              );
              ws.close();
              resolve();
            }
          } catch {
            // ignore malformed messages
          }
        };

        ws.onerror = () => {
          clearTimeout(connectTimeout);
          reject(new Error("WebSocket error"));
        };

        ws.onclose = () => {
          resolve();
        };
      });

      try {
        await wsDone;
      } finally {
        if (onAbort) {
          abortController.signal.removeEventListener("abort", onAbort);
        }
      }
    } catch (err) {
      if (
        err instanceof Error &&
        (err.name === "AbortError" ||
          err.message === "The user aborted a request.")
      ) {
        return;
      }
      // Fallback to existing SSE stream if WebSocket fails
      try {
        const stream = api.streamPost(
          "/api/v1/chat/ask/stream",
          {
            question,
            document_id: null,
            document_ids: documentIds,
            session_id: activeSessionId,
            llm_provider: llmProvider,
          },
          abortController.signal,
        );

        for await (const event of stream) {
          if (event.type === "token") {
            appendToken(event.data as string);
          } else if (event.type === "status") {
            setIsTyping(false);
          } else if (event.type === "sources") {
            if (!assistantCreated) {
              assistantCreated = ensureAssistantPlaceholder(
                assistantId,
                assistantCreated,
              );
            }
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, sources: event.data as SourceChunk[] }
                  : m,
              ),
            );
          } else if (event.type === "quiz") {
            flushTokensNow();
            if (!assistantCreated) {
              assistantCreated = ensureAssistantPlaceholder(
                assistantId,
                assistantCreated,
              );
            }
            applyQuizToMessage(assistantId, event.data as QuizPayload);
          } else if (event.type === "usage") {
            applyUsageToMessage(assistantId, event.data as LlmUsage);
          } else if (event.type === "error") {
            flushTokensNow();
            setIsTyping(false);
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                    ...m,
                    content: `Error: ${event.data}`,
                    isStreaming: false,
                  }
                  : m,
              ),
            );
          } else if (event.type === "done") {
            flushTokensNow();
            const doneEvent = event as unknown as {
              usage?: LlmUsage;
              response_time_ms?: number;
              message_id?: string;
            };
            const persistedId =
              typeof doneEvent.message_id === "string"
                ? doneEvent.message_id
                : undefined;
            applyUsageToMessage(assistantId, doneEvent.usage, {
              isStreaming: false,
              response_time_ms: doneEvent.response_time_ms,
              ...(persistedId ? { id: persistedId } : {}),
            });
          }
        }
      } catch (err2) {
        setIsTyping(false);
        if (
          err2 instanceof Error &&
          (err2.name === "AbortError" ||
            err2.message === "The user aborted a request.")
        ) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, isStreaming: false } : m,
            ),
          );
          return;
        }
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                ...m,
                content: t("chat.fallbackError", {
                  message:
                    err2 instanceof Error ? err2.message : "Unknown error",
                }),
                isStreaming: false,
              }
              : m,
          ),
        );
      }
    } finally {
      flushTokensNow();
      setStreaming(false);
      setIsTyping(false);
    }
  };

  const handleClear = async () => {
    if (!activeDoc || !confirm(t("chat.clearConfirm"))) return;
    try {
      await api.delete(`/api/v1/chat/history/${activeDoc.id}`);
      setMessages([]);
      toast.info("Chat history cleared");
    } catch {
      // silent fail preserved; no additional toast for this scenario
    }
  };

  const handleExport = (format: "md" | "txt" | "pdf") => {
    setShowExportMenu(false);
    const token = localStorage.getItem("token");
    if (!token || token === "cookie") {
      toast.error("Sign in required to export chat");
      return;
    }

    // Chat is session-scoped; document export is only a fallback.
    const base = API_BASE || window.location.origin;
    let url: string | null = null;
    if (activeSessionId) {
      url = `${base}/api/v1/chat/export/session/${activeSessionId}?format=${encodeURIComponent(format)}&token=${encodeURIComponent(token)}`;
    } else if (activeDoc) {
      url = `${base}/api/v1/chat/export/${activeDoc.id}?format=${encodeURIComponent(format)}&token=${encodeURIComponent(token)}`;
    } else {
      toast.error("Select a chat session to export");
      return;
    }

    const a = document.createElement("a");
    a.href = url;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // Close export dropdown on outside click
  useEffect(() => {
    if (!showExportMenu) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (
        exportMenuRef.current &&
        !exportMenuRef.current.contains(e.target as Node)
      ) {
        setShowExportMenu(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showExportMenu]);

  // Cleanup speech recognition on unmount
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  const startRecording = () => {
    const SpeechRecognitionAPI =
      typeof window !== "undefined"
        ? (window as unknown as WindowWithSpeech).SpeechRecognition ||
        (window as unknown as WindowWithSpeech).webkitSpeechRecognition
        : null;

    if (!SpeechRecognitionAPI) {
      setSpeechError(
        t("chat.speechNotSupported", {
          defaultValue: "Speech recognition is not supported in this browser.",
        }),
      );
      return;
    }

    try {
      const recognition = new SpeechRecognitionAPI();
      recognition.continuous = true;
      recognition.interimResults = true;

      const currentLang = i18n.language || "en";
      const langMap: Record<string, string> = {
        en: "en-US",
        hi: "hi-IN",
        es: "es-ES",
        fr: "fr-FR",
      };
      recognition.lang = langMap[currentLang] || "en-US";

      initialInputRef.current = input;
      setSpeechError(null);
      setIsRecording(true);

      recognition.onresult = (event: ISpeechRecognitionEvent) => {
        let sessionTranscript = "";
        for (let i = 0; i < event.results.length; ++i) {
          sessionTranscript += event.results[i][0].transcript;
        }
        setInput(
          initialInputRef.current +
          (initialInputRef.current ? " " : "") +
          sessionTranscript.trim(),
        );
      };

      recognition.onerror = (event: ISpeechRecognitionErrorEvent) => {
        const errorCode = event.error;
        if (errorCode === "aborted") return; // ignore manual aborts

        let msg = t("chat.speechError", {
          defaultValue: `Speech recognition error: ${errorCode}`,
        });
        if (errorCode === "not-allowed") {
          msg = t("chat.micPermissionDenied", {
            defaultValue:
              "Microphone access denied. Please enable permissions in settings.",
          });
        } else if (errorCode === "no-speech") {
          msg = t("chat.noSpeechDetected", {
            defaultValue: "No speech was detected. Please try again.",
          });
        } else if (errorCode === "audio-capture") {
          msg = t("chat.audioCaptureError", {
            defaultValue: "No microphone found or microphone is not working.",
          });
        } else if (errorCode === "network") {
          msg = t("chat.networkError", {
            defaultValue: "Network error occurred during speech recognition.",
          });
        }
        setSpeechError(msg);
        setIsRecording(false);
      };

      recognition.onend = () => {
        setIsRecording(false);
      };

      recognitionRef.current = recognition;
      recognition.start();
    } catch (err) {
      setSpeechError(
        err instanceof Error
          ? err.message
          : "Failed to start speech recognition.",
      );
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape" && (attachPickerOpen || isAttachCommand)) {
      e.preventDefault();
      setAttachPickerOpen(false);
      if (isAttachCommand) setInput("");
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Open attach picker while typing /attach in the composer
  useEffect(() => {
    if (isAttachCommand) {
      setAttachPickerOpen(true);
    }
  }, [isAttachCommand]);

  const handleExportMenuKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      setShowExportMenu(false);
    }
  };

  // ── NEW KEYBOARD SHORTCUTS ENGINE EFFECT ──────────────────────────
  // ── KEYBOARD SHORTCUTS ENGINE EFFECT ──────────────────────────
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      const isCmdOrCtrl = e.metaKey || e.ctrlKey;
      const isMobile = /Mobi|Android/i.test(navigator.userAgent);

      // No-op on mobile devices
      if (isMobile) return;

      // Shortcut 1: Ctrl/Cmd + Enter → Send Message (when textarea focused)
      if (isCmdOrCtrl && e.key === "Enter") {
        if (document.activeElement === textareaRef.current) {
          e.preventDefault();
          handleSend();
        }
      }

      // Shortcut 2: Escape → Abort SSE stream OR clear input OR close modal
      if (e.key === "Escape") {
        if (streaming) {
          e.preventDefault();
          handleStop();
          toast.info("Response cancelled");
        } else if (document.activeElement === textareaRef.current) {
          e.preventDefault();
          setInput("");
        } else if (showExportMenu) {
          setShowExportMenu(false);
        }
      }

      // Shortcut 3: Ctrl/Cmd + K → Focus chat input from anywhere
      if (isCmdOrCtrl && (e.key === "k" || e.key === "K")) {
        e.preventDefault();
        textareaRef.current?.focus();
      }

      // Shortcut 5: Ctrl/Cmd + Shift + C → Clear chat history
      if (isCmdOrCtrl && e.shiftKey && (e.key === "c" || e.key === "C")) {
        e.preventDefault();
        handleClear();
      }

      // Shortcut 6: Ctrl/Cmd + Shift + E → Toggle export menu
      if (isCmdOrCtrl && e.shiftKey && (e.key === "e" || e.key === "E")) {
        e.preventDefault();
        if (messages.length > 0) {
          setShowExportMenu((prev) => !prev);
        }
      }

      // Shortcut 7: Ctrl/Cmd + Shift + M → Toggle mic recording
      if (isCmdOrCtrl && e.shiftKey && (e.key === "m" || e.key === "M")) {
        e.preventDefault();
        if (!streaming) {
          toggleRecording();
        }
      }
    };

    window.addEventListener("keydown", handleGlobalKeyDown);
    return () => {
      window.removeEventListener("keydown", handleGlobalKeyDown);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input, streaming, showExportMenu, messages]); // Dependencies updated to capture fresh state data

  return (
    <div className="h-full flex flex-col relative">
      {/* ── Chat Messages ──────────────────────────── */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="flex-1 px-4 overflow-y-auto custom-scrollbar"
        aria-busy={historyLoading}
      >
        {historyLoading ? (
          <div
            className="py-6 space-y-5 max-w-3xl mx-auto"
            aria-label="Loading chat history"
          >
            {Array.from({ length: 4 }).map((_, index) => (
              <div
                key={index}
                className={cn(
                  "flex gap-3",
                  index % 2 === 0 ? "justify-end" : "justify-start",
                )}
              >
                {index % 2 !== 0 && (
                  <Skeleton className="mt-1 h-8 w-8 rounded-full" />
                )}
                <div
                  className={cn(
                    "space-y-2",
                    index % 2 === 0 ? "w-2/3" : "w-3/4",
                  )}
                >
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-5/6" />
                  <Skeleton className="h-4 w-2/3" />
                </div>
              </div>
            ))}
          </div>
        ) : showEmptyState ? (
          <div className="h-full flex flex-col items-center justify-center py-20 px-6">
            <div className="relative mb-5">
              <div className="absolute inset-0 rounded-3xl bg-primary/20 blur-xl" />
              <div className="relative w-16 h-16 rounded-3xl bg-gradient-to-br from-primary/25 to-[oklch(0.86_0.16_125/0.2)] border border-primary/20 flex items-center justify-center">
                <MessageSquare className="w-8 h-8 text-primary" />
              </div>
            </div>
            <h3 className="font-display text-xl font-semibold mb-1.5 tracking-tight">
              {activeDoc
                ? t("chat.askAboutDocument")
                : t("chat.selectDocument")}
            </h3>
            <p className="text-sm text-muted-foreground text-center max-w-sm leading-relaxed">
              {activeDoc
                ? t("chat.readyPrompt", { name: activeDoc.original_name })
                : t("chat.uploadPrompt")}
            </p>
          </div>
        ) : (
          <div className="py-4 space-y-1 max-w-3xl mx-auto">
            {messages.map((msg) => (
              <div key={msg.id}>
                <MessageBubble message={msg} />
                {msg.role === "assistant" && msg.quiz && (
                  <div className="ml-10 mt-2 mb-3 max-w-xl">
                    <QuizCard
                      quiz={msg.quiz}
                      messageId={msg.id}
                      onQuizUpdate={(quiz) =>
                        setMessages((prev) =>
                          prev.map((m) =>
                            m.id === msg.id ? { ...m, quiz } : m,
                          ),
                        )
                      }
                    />
                  </div>
                )}
                {msg.role === "assistant" && (msg.sources?.length ?? 0) > 0 && (
                  <div className="ml-10 mt-1 mb-3">
                    <SourceCard
                      sources={msg.sources}
                      onPageClick={onCitationClick}
                    />
                  </div>
                )}
              </div>
            ))}
            {isTyping && (
              <div className="flex items-center gap-1 ml-10 py-2">
                <span className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce [animation-delay:-0.3s]" />
                <span className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce [animation-delay:-0.15s]" />
                <span className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce" />
              </div>
            )}
          </div>
        )}
        <div ref={bottomRef} className="h-4" />
      </div>

      {/* Scroll to bottom button */}
      <button
        type="button"
        onClick={scrollToBottom}
        aria-label="Scroll to bottom"
        className={cn(
          "absolute right-4 bottom-20 z-50 rounded-full p-2 bg-primary text-primary-foreground shadow-lg transition-all duration-200",
          showScrollButton
            ? "opacity-100 translate-y-0"
            : "opacity-0 translate-y-2 pointer-events-none",
        )}
      >
        <ChevronDown className="h-5 w-5" />
      </button>

      {/* ── Composer ───────────────────────────────── */}
      <div className="relative border-t border-border/40 bg-card/45 p-3 pl-16 backdrop-blur-xl sm:p-4 sm:pl-4">
        <div className="mx-auto max-w-3xl">
          <div className="overflow-hidden rounded-2xl border border-border/60 bg-background/70 shadow-sm transition-[border-color,box-shadow] duration-200 focus-within:border-primary/35 focus-within:shadow-md">
            {/* Speech status strip */}
            {(isRecording || speechError) && (
              <div className="flex items-center justify-between gap-2 border-b border-border/40 bg-muted/40 px-3 py-1.5 text-xs">
                <div className="flex min-w-0 items-center gap-2">
                  {isRecording ? (
                    <>
                      <span className="relative flex h-2 w-2 shrink-0">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                        <span className="relative inline-flex h-2 w-2 rounded-full bg-red-500" />
                      </span>
                      <span className="truncate font-medium text-foreground/80">
                        {t("chat.listening", {
                          defaultValue: "Listening… speak now.",
                        })}
                      </span>
                    </>
                  ) : (
                    <span className="truncate font-medium text-destructive">
                      {speechError}
                    </span>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => {
                    if (isRecording) {
                      stopRecording();
                    } else {
                      setSpeechError(null);
                    }
                  }}
                  className="shrink-0 rounded px-1.5 py-0.5 font-semibold text-muted-foreground transition-colors hover:bg-background hover:text-foreground"
                  aria-label={
                    isRecording
                      ? "Stop speech recording"
                      : "Dismiss speech error"
                  }
                >
                  {isRecording ? (
                    t("chat.stop", { defaultValue: "Stop" })
                  ) : (
                    <X className="h-3 w-3" />
                  )}
                </button>
              </div>
            )}

            {/* Control rail — model, scope, utilities */}
            <div className="flex items-center gap-1 px-2 py-1.5">
              <label className="relative flex items-center">
                <Cpu
                  className="pointer-events-none absolute left-2 h-3.5 w-3.5 text-muted-foreground"
                  aria-hidden="true"
                />
                <select
                  value={llmProvider}
                  disabled={streaming}
                  onChange={(e) =>
                    setLlmProvider(e.target.value as "ollama" | "bedrock")
                  }
                  className="h-7 cursor-pointer appearance-none rounded-lg bg-muted/40 py-0 pr-6 pl-7 text-xs font-medium text-foreground outline-none transition-colors hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring/40 disabled:cursor-not-allowed disabled:opacity-50"
                  aria-label="LLM provider"
                >
                  <option value="ollama">Ollama (local)</option>
                  <option value="bedrock">Nova 2 Lite (Bedrock)</option>
                </select>
                <ChevronDown
                  className="pointer-events-none absolute right-1.5 h-3 w-3 text-muted-foreground"
                  aria-hidden="true"
                />
              </label>

              <span
                className="mx-1 h-4 w-px shrink-0 bg-border/60"
                aria-hidden="true"
              />

              <button
                type="button"
                disabled={streaming || readyDocs.length === 0}
                onClick={() => setAttachPickerOpen((v) => !v)}
                className={cn(
                  "inline-flex h-7 shrink-0 items-center gap-1.5 rounded-lg px-2 text-xs font-medium transition-colors outline-none select-none focus-visible:ring-2 focus-visible:ring-ring/40 disabled:pointer-events-none disabled:opacity-50",
                  attachPickerOpen
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
                aria-label="Attach documents"
                aria-expanded={attachPickerOpen}
              >
                <Paperclip className="h-3.5 w-3.5" aria-hidden="true" />
                <span className="hidden sm:inline">Attach</span>
                {attachedDocs.length > 0 && (
                  <span
                    className={cn(
                      "inline-flex h-4 min-w-4 items-center justify-center rounded-full px-1 text-[10px] leading-none font-semibold tabular-nums",
                      attachPickerOpen
                        ? "bg-primary/20 text-primary"
                        : "bg-muted text-foreground",
                    )}
                  >
                    {attachedDocs.length}
                  </span>
                )}
              </button>

              {messages.length > 0 && (
                <div className="ml-auto flex items-center gap-0.5">
                  {/* Export history */}
                  <div className="relative" ref={exportMenuRef}>
                    <Tooltip>
                      <TooltipTrigger
                        id="export-chat-btn"
                        type="button"
                        onClick={() => setShowExportMenu((v) => !v)}
                        className={cn(
                          buttonVariants({ variant: "ghost", size: "icon" }),
                          "h-7 w-7 rounded-lg text-muted-foreground hover:text-foreground",
                        )}
                        aria-label={t("chat.exportTitle")}
                        aria-expanded={showExportMenu}
                        aria-controls="chat-export-menu"
                        aria-haspopup="menu"
                      >
                        <Download className="h-3.5 w-3.5" />
                      </TooltipTrigger>
                      <TooltipContent>{t("chat.exportTitle")}</TooltipContent>
                    </Tooltip>
                    {showExportMenu && (
                      <div
                        id="chat-export-menu"
                        role="menu"
                        aria-label="Export chat"
                        onKeyDown={handleExportMenuKeyDown}
                        className="absolute bottom-full right-0 z-50 mb-2 min-w-[160px] max-w-[calc(100vw-2rem)] rounded-lg border border-border bg-popover p-1 shadow-lg animate-in fade-in slide-in-from-bottom-2"
                      >
                        <button
                          id="export-md-btn"
                          type="button"
                          role="menuitem"
                          onClick={() => handleExport("md")}
                          className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors hover:bg-accent"
                        >
                          <span className="text-base">📝</span>
                          {t("chat.markdown")}
                        </button>
                        <button
                          id="export-txt-btn"
                          type="button"
                          role="menuitem"
                          onClick={() => handleExport("txt")}
                          className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors hover:bg-accent"
                        >
                          <span className="text-base">📄</span>
                          {t("chat.plainText")}
                        </button>
                        <button
                          id="export-pdf-btn"
                          type="button"
                          role="menuitem"
                          onClick={() => handleExport("pdf")}
                          className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors hover:bg-accent"
                        >
                          <span className="text-base">📕</span>
                          {t("chat.pdf")}
                        </button>
                      </div>
                    )}
                  </div>
                  {/* Clear history */}
                  <Tooltip>
                    <TooltipTrigger
                      type="button"
                      onClick={handleClear}
                      className={cn(
                        buttonVariants({ variant: "ghost", size: "icon" }),
                        "h-7 w-7 rounded-lg text-muted-foreground hover:text-destructive",
                      )}
                      aria-label="Clear chat history"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </TooltipTrigger>
                    <TooltipContent>Clear chat history</TooltipContent>
                  </Tooltip>
                </div>
              )}
            </div>

            {/* Attached document chips */}
            {attachedDocs.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5 border-t border-border/40 px-2.5 py-2">
                {attachedDocs.map((doc) => (
                  <span
                    key={doc.id}
                    className="inline-flex max-w-[15rem] items-center gap-1.5 rounded-lg border border-border/60 bg-muted/40 py-1 pr-1 pl-2 text-[11px] font-medium text-foreground"
                    title={doc.filename}
                  >
                    <FileText
                      className="h-3 w-3 shrink-0 text-muted-foreground"
                      aria-hidden="true"
                    />
                    <span className="truncate">{doc.filename}</span>
                    <button
                      type="button"
                      className="rounded p-0.5 text-muted-foreground/70 transition-colors hover:bg-background hover:text-destructive"
                      aria-label={`Remove ${doc.filename}`}
                      onClick={() => removeAttachedDoc(doc.id)}
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </span>
                ))}
                <button
                  type="button"
                  className="rounded-md px-1.5 py-1 text-[11px] font-medium text-muted-foreground transition-colors hover:text-destructive"
                  onClick={() => setAttachedDocs([])}
                >
                  Clear all
                </button>
              </div>
            )}

            {/* Attach picker */}
            {attachPickerOpen && (
              <div className="border-t border-border/40">
                <div className="flex items-center justify-between px-3 py-1.5">
                  <span className="text-[10px] font-semibold tracking-wider text-muted-foreground uppercase">
                    Attach PDFs
                  </span>
                  <span className="text-[10px] text-muted-foreground tabular-nums">
                    {attachedDocs.length} selected
                  </span>
                </div>
                <div
                  className="max-h-48 overflow-y-auto px-1 pb-1"
                  role="listbox"
                  aria-label="Attach documents"
                >
                  {attachCandidates.length === 0 ? (
                    <p className="px-3 py-3 text-xs text-muted-foreground">
                      {readyDocs.length === 0
                        ? "No ready documents to attach."
                        : "No matching PDFs."}
                    </p>
                  ) : (
                    attachCandidates.map((doc) => {
                      const selected = attachedDocs.some((d) => d.id === doc.id);
                      return (
                        <button
                          key={doc.id}
                          type="button"
                          role="option"
                          aria-selected={selected}
                          onClick={() => toggleAttachedDoc(doc)}
                          className={cn(
                            "flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-xs transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring/40",
                            selected
                              ? "bg-primary/10 font-medium text-foreground"
                              : "text-foreground hover:bg-accent",
                          )}
                        >
                          <span
                            className={cn(
                              "flex h-4 w-4 shrink-0 items-center justify-center rounded border transition-colors",
                              selected
                                ? "border-primary bg-primary text-primary-foreground"
                                : "border-border/70 bg-background/60",
                            )}
                            aria-hidden="true"
                          >
                            {selected && <Check className="h-3 w-3" />}
                          </span>
                          <span className="flex-1 truncate">
                            {doc.original_name}
                          </span>
                        </button>
                      );
                    })
                  )}
                </div>
                <div className="flex items-center justify-between gap-2 border-t border-border/40 px-3 py-1.5">
                  <span className="text-[10px] text-muted-foreground">
                    {attachedDocs.length === 0
                      ? "None selected → search all indexed docs"
                      : `Search limited to ${attachedDocs.length} PDF${attachedDocs.length === 1 ? "" : "s"}`}
                  </span>
                  <button
                    type="button"
                    className="h-6 shrink-0 rounded-md bg-primary px-2.5 text-[11px] font-semibold text-primary-foreground shadow-sm transition-colors hover:bg-primary/90 focus-visible:ring-2 focus-visible:ring-ring/40 focus-visible:outline-none"
                    onClick={() => {
                      setAttachPickerOpen(false);
                      if (isAttachCommand) setInput("");
                    }}
                  >
                    Done
                  </button>
                </div>
              </div>
            )}

            {/* Writing area */}
            <div className="flex items-end gap-1.5 px-2 pb-2">
              <Textarea
                ref={textareaRef}
                id="chat-input"
                maxLength={MAX_CHARACTERS}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  attachedDocs.length > 0
                    ? `Ask about ${attachedDocs.length} attached PDF${attachedDocs.length === 1 ? "" : "s"}…`
                    : "Ask across all indexed documents…"
                }
                disabled={streaming}
                className="min-h-[40px] max-h-32 flex-1 resize-none border-0 bg-transparent px-1.5 py-2 shadow-none focus-visible:border-0 focus-visible:ring-0 dark:bg-transparent"
                rows={1}
                aria-label="Chat message"
                aria-describedby="chat-input-hint"
              />

              <Tooltip>
                <TooltipTrigger
                  id="mic-btn"
                  type="button"
                  disabled={streaming}
                  onClick={toggleRecording}
                  className={cn(
                    buttonVariants({ variant: "ghost", size: "icon" }),
                    "mb-0.5 h-8 w-8 shrink-0 rounded-lg text-muted-foreground transition-all duration-200",
                    isRecording
                      ? "animate-pulse bg-red-500/20 text-red-500 hover:bg-red-500/30 hover:text-red-600"
                      : "hover:bg-accent hover:text-primary",
                  )}
                  aria-label={
                    isRecording
                      ? t("chat.stopRecording", {
                        defaultValue: "Stop recording",
                      })
                      : t("chat.startRecording", {
                        defaultValue: "Start recording",
                      })
                  }
                  aria-pressed={isRecording}
                >
                  {isRecording ? (
                    <MicOff className="h-4 w-4" />
                  ) : (
                    <Mic className="h-4 w-4" />
                  )}
                </TooltipTrigger>
                <TooltipContent>
                  {isRecording
                    ? t("chat.stopRecording", {
                      defaultValue: "Stop recording",
                    })
                    : t("chat.startRecording", {
                      defaultValue: "Start recording",
                    })}
                </TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger
                  id="send-btn"
                  type="button"
                  onClick={streaming ? handleStop : handleSend}
                  disabled={!streaming && !input.trim()}
                  className={cn(
                    buttonVariants({ size: "icon" }),
                    "mb-0.5 h-9 w-9 shrink-0 rounded-xl shadow-sm shadow-primary/20",
                  )}
                  aria-label={streaming ? "Stop generating" : "Send message"}
                >
                  {streaming ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </TooltipTrigger>
                <TooltipContent>
                  {streaming ? "Stop generating" : "Send message"}
                </TooltipContent>
              </Tooltip>
            </div>
          </div>

          <p id="chat-input-hint" className="sr-only">
            Press Enter to send. Press Shift and Enter for a new line. Press
            Ctrl+Enter to send from anywhere. Press Escape to cancel streaming.
          </p>
        </div>
      </div>
    </div>
  );
}

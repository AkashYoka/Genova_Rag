"use client";

import { toast } from "sonner";
import { useState, useRef, useEffect } from "react";
import { useTranslation } from "react-i18next";
import type { DocInfo } from "@/app/dashboard/page";
import { api, API_BASE } from "@/lib/api";
import {
  useChatStore,
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
} from "lucide-react";
import { cn } from "@/lib/utils";
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
  onCitationClick: (target: CitationTarget) => void;
}

export default function ChatPanel({ activeDoc, onCitationClick }: Props) {
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
    setInput("");

    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    // Add user message
    const userMsg: ChatMsg = {
      id: `user-${Date.now()}`,
      role: "user",
      content: question,
      sources: [],
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
          // Send initial payload
          ws.send(
            JSON.stringify({
              question,
              document_id: activeDoc?.id || null,
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
            document_id: activeDoc?.id || null,
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
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

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

      {/* ── Input Area ─────────────────────────────── */}
      <div className="border-t border-border/40 p-4 pl-16 sm:pl-4 bg-card/45 backdrop-blur-xl relative">
        <div className="max-w-3xl mx-auto relative">
          <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <label className="flex items-center gap-2">
              <span className="font-medium text-foreground/80">Model</span>
              <select
                value={llmProvider}
                disabled={streaming}
                onChange={(e) =>
                  setLlmProvider(e.target.value as "ollama" | "bedrock")
                }
                className="h-8 rounded-md border border-border/60 bg-background/70 px-2 text-xs text-foreground"
                aria-label="LLM provider"
              >
                <option value="ollama">Ollama (local)</option>
                <option value="bedrock">Nova 2 Lite (Bedrock)</option>
              </select>
            </label>
          </div>
          {/* Status / Error Message Area */}
          {(isRecording || speechError) && (
            <div className="absolute bottom-full mb-2 left-0 right-0 flex items-center justify-between bg-card border border-border/80 shadow-md rounded-lg px-3 py-1.5 text-xs animate-in fade-in slide-in-from-bottom-1 z-40 max-w-3xl mx-auto">
              <div className="flex items-center gap-2">
                {isRecording ? (
                  <>
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
                    </span>
                    <span className="font-medium text-muted-foreground">
                      {t("chat.listening", {
                        defaultValue: "Listening... Speak now.",
                      })}
                    </span>
                  </>
                ) : (
                  <span className="text-destructive font-medium">
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
                className="text-muted-foreground hover:text-foreground font-semibold px-1.5 py-0.5 rounded hover:bg-muted transition-colors"
                aria-label={
                  isRecording ? "Stop speech recording" : "Dismiss speech error"
                }
              >
                {isRecording ? t("chat.stop", { defaultValue: "Stop" }) : "✕"}
              </button>
            </div>
          )}

          <div className="flex gap-2 items-end">
            <div className="relative flex-1 flex items-center">
              <Textarea
                ref={textareaRef}
                id="chat-input"
                maxLength={MAX_CHARACTERS}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  activeDoc
                    ? t("chat.askPlaceholder", {
                      name: activeDoc.original_name,
                    })
                    : t("chat.selectPlaceholder")
                }
                disabled={streaming}
                className="min-h-[44px] max-h-32 resize-none bg-background/50 border-border/50 pr-11"
                rows={1}
                aria-label="Chat message"
                aria-describedby="chat-input-hint"
              />

              {/* Mic — inner-right of composer, vertically centered */}
              <Tooltip>
                <TooltipTrigger
                  id="mic-btn"
                  type="button"
                  disabled={streaming}
                  onClick={toggleRecording}
                  className={cn(
                    buttonVariants({ variant: "ghost", size: "icon" }),
                    "absolute right-2 top-1/2 h-7 w-7 -translate-y-1/2 rounded-md text-muted-foreground transition-all duration-200",
                    isRecording
                      ? "bg-red-500/20 text-red-500 hover:bg-red-500/30 hover:text-red-600 animate-pulse"
                      : "hover:text-primary hover:bg-accent",
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
            </div>

            <div className="flex gap-1.5 shrink-0">
              <Tooltip>
                <TooltipTrigger
                  id="send-btn"
                  type="button"
                  onClick={streaming ? handleStop : handleSend}
                  disabled={!streaming && !input.trim()}
                  className={cn(
                    buttonVariants({ size: "icon" }),
                    "h-10 w-10 sm:h-[44px] sm:w-[44px]",
                  )}
                  aria-label={streaming ? "Stop generating" : "Send message"}
                >
                  {streaming ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Send className="w-4 h-4" />
                  )}
                </TooltipTrigger>
                <TooltipContent>
                  {streaming ? "Stop generating" : "Send message"}
                </TooltipContent>
              </Tooltip>
              {messages.length > 0 && (
                <>
                  {/* Export dropdown */}
                  <div className="relative" ref={exportMenuRef}>
                    <Tooltip>
                      <TooltipTrigger
                        id="export-chat-btn"
                        type="button"
                        onClick={() => setShowExportMenu((v) => !v)}
                        className={cn(
                          buttonVariants({ variant: "ghost", size: "icon" }),
                          "h-10 w-10 sm:h-[44px] sm:w-[44px] text-muted-foreground hover:text-primary",
                        )}
                        aria-label={t("chat.exportTitle")}
                        aria-expanded={showExportMenu}
                        aria-controls="chat-export-menu"
                        aria-haspopup="menu"
                      >
                        <Download className="w-4 h-4" />
                      </TooltipTrigger>
                      <TooltipContent>{t("chat.exportTitle")}</TooltipContent>
                    </Tooltip>
                    {showExportMenu && (
                      <div
                        id="chat-export-menu"
                        role="menu"
                        aria-label="Export chat"
                        onKeyDown={handleExportMenuKeyDown}
                        className="absolute bottom-full mb-2 right-0 min-w-[160px] max-w-[calc(100vw-2rem)] rounded-lg border border-border bg-popover p-1 shadow-lg animate-in fade-in slide-in-from-bottom-2 z-50"
                      >
                        <button
                          id="export-md-btn"
                          type="button"
                          role="menuitem"
                          onClick={() => handleExport("md")}
                          className="w-full flex items-center gap-2 rounded-md px-3 py-2 text-sm hover:bg-accent transition-colors text-left"
                        >
                          <span className="text-base">📝</span>
                          {t("chat.markdown")}
                        </button>
                        <button
                          id="export-txt-btn"
                          type="button"
                          role="menuitem"
                          onClick={() => handleExport("txt")}
                          className="w-full flex items-center gap-2 rounded-md px-3 py-2 text-sm hover:bg-accent transition-colors text-left"
                        >
                          <span className="text-base">📄</span>
                          {t("chat.plainText")}
                        </button>
                        <button
                          id="export-pdf-btn"
                          type="button"
                          role="menuitem"
                          onClick={() => handleExport("pdf")}
                          className="w-full flex items-center gap-2 rounded-md px-3 py-2 text-sm hover:bg-accent transition-colors text-left"
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
                        "h-10 w-10 sm:h-[44px] sm:w-[44px] text-muted-foreground hover:text-destructive",
                      )}
                      aria-label="Clear chat history"
                    >
                      <Trash2 className="w-4 h-4" />
                    </TooltipTrigger>
                    <TooltipContent>Clear chat history</TooltipContent>
                  </Tooltip>
                </>
              )}
            </div>
          </div>
        </div>
        <p id="chat-input-hint" className="sr-only">
          Press Enter to send. Press Shift and Enter for a new line. Press
          Ctrl+Enter to send from anywhere. Press Escape to cancel streaming.
        </p>
      </div>
    </div>
  );
}

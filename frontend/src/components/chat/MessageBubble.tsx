"use client";

import { useState, useRef, useEffect, useSyncExternalStore } from "react";
import { formatDistanceToNow } from "date-fns";
import type { ChatMsg } from "@/store/chat-store";
import { api } from "@/lib/api";
import {
  Brain,
  User,
  Copy,
  Check,
  Share2,
  Link2,
  X,
  Play,
  Pause,
  GitBranch,
  StickyNote,
} from "lucide-react";
import { toast } from "sonner";
import { buttonVariants } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { useChatStore } from "@/store/chat-store";
import { useSettingsStore } from "@/store/settings-store";
import { useNotesStore } from "@/store/notes-store";
import {
  GENOVA_NOTE_MIME,
  encodeNotePayload,
} from "@/lib/notes-mime";
import MarkdownContent from "@/components/chat/MarkdownContent";

const subscribe = () => () => {};
const getSnapshot = () => true;
const getServerSnapshot = () => false;

/** Local client ids look like `user-1712345678901`; API history ids are UUIDs. */
function timestampFromMessageId(id: string): Date | null {
  const match = /^(?:user|assistant)-(\d+)$/.exec(id);
  if (!match) return null;
  const ms = Number(match[1]);
  if (!Number.isFinite(ms)) return null;
  const date = new Date(ms);
  return Number.isNaN(date.getTime()) ? null : date;
}

interface Props {
  message: ChatMsg;
}

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";
  const messageTime = timestampFromMessageId(message.id);
  const [copied, setCopied] = useState(false);
  const [shared, setShared] = useState(false);
  const [shareFailed, setShareFailed] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const copiedTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sharedTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  // Cancel speech synthesis when the component unmounts
  useEffect(() => {
    return () => {
      if (utteranceRef.current) {
        window.speechSynthesis.cancel();
        utteranceRef.current = null;
      }
    };
  }, []);

  const setCurrentBranchId = useChatStore((s) => s.setCurrentBranchId);
  const appendMessage = useNotesStore((s) => s.appendMessage);
  const openNotes = useNotesStore((s) => s.openNotes);

  const mounted = useSyncExternalStore(
    subscribe,
    getSnapshot,
    getServerSnapshot,
  );
  const fontSize = useSettingsStore((s) => s.fontSize);

  const activeFontSize = mounted ? fontSize : "medium";
  const fontSizeClass = {
    small: "text-xs",
    medium: "text-sm",
    large: "text-base",
  }[activeFontSize];

  const handleCopy = async () => {
    if (!message.content) return;
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      if (copiedTimeoutRef.current) clearTimeout(copiedTimeoutRef.current);
      copiedTimeoutRef.current = setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };

  const handleShare = async () => {
    if (!message.content || message.isStreaming) return;
    try {
      const data = await api.post<{ message_id: string; share_url: string }>(
        `/api/v1/chat/share/${message.id}`,
      );
      await navigator.clipboard.writeText(
        `${window.location.origin}${data.share_url}`,
      );
      setShared(true);
      setShareFailed(false);
      if (sharedTimeoutRef.current) clearTimeout(sharedTimeoutRef.current);
      sharedTimeoutRef.current = setTimeout(() => {
        setShared(false);
        setShareFailed(false);
      }, 2000);
    } catch {
      setShareFailed(true);
      setShared(false);
      if (sharedTimeoutRef.current) clearTimeout(sharedTimeoutRef.current);
      sharedTimeoutRef.current = setTimeout(() => {
        setShareFailed(false);
      }, 2000);
    }
  };

  const handleSpeech = () => {
    if (!message.content || message.isStreaming) return;
    if (typeof window === "undefined" || !window.speechSynthesis) return;

    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      utteranceRef.current = null;
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(message.content);
    utteranceRef.current = utterance;

    utterance.onend = () => {
      setIsSpeaking(false);
      utteranceRef.current = null;
    };
    utterance.onerror = () => {
      setIsSpeaking(false);
      utteranceRef.current = null;
    };

    setIsSpeaking(true);
    window.speechSynthesis.speak(utterance);
  };

  const handleBranch = () => {
    setCurrentBranchId(message.id);

    console.log("Branch created from:", message.id);
  };

  const notePayload = {
    id: message.id,
    role: message.role,
    content: message.content,
  };

  const handleAddToNotes = () => {
    if (!message.content || message.isStreaming) return;
    appendMessage(notePayload);
    openNotes();
    toast.success("Added to notes");
  };

  const handleDragStart = (e: React.DragEvent) => {
    if (!message.content || message.isStreaming) {
      e.preventDefault();
      return;
    }
    const encoded = encodeNotePayload(notePayload);
    e.dataTransfer.setData(GENOVA_NOTE_MIME, encoded);
    e.dataTransfer.setData("text/plain", encoded);
    e.dataTransfer.effectAllowed = "copy";
  };

  return (
    <div
      className={`flex gap-3 py-3 animate-fade-in-up ${isUser ? "justify-end" : "justify-start"}`}
    >
      {!isUser && (
        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-primary/25 to-[oklch(0.86_0.16_125/0.2)] border border-primary/20 flex items-center justify-center shrink-0 mt-0.5">
          <Brain className="w-4 h-4 text-primary" />
        </div>
      )}

      <div
        draggable={Boolean(message.content) && !message.isStreaming}
        onDragStart={handleDragStart}
        className={`group relative max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? "bg-primary text-primary-foreground rounded-br-md shadow-lg shadow-primary/20"
            : "bg-card/80 border border-border/50 rounded-bl-md backdrop-blur-sm"
        } ${message.content && !message.isStreaming ? "cursor-grab active:cursor-grabbing" : ""}`}
      >
        {isUser ? (
          <>
            {message.content ? (
              <div className="absolute top-2 left-2 flex opacity-0 pointer-events-none transition-opacity group-hover:opacity-100 group-hover:pointer-events-auto">
                <Tooltip>
                  <TooltipTrigger
                    type="button"
                    className={cn(
                      buttonVariants({ variant: "ghost", size: "icon-xs" }),
                      "bg-primary-foreground/15 text-primary-foreground hover:bg-primary-foreground/25",
                    )}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleAddToNotes();
                    }}
                    aria-label="Add to notes"
                  >
                    <StickyNote className="w-3.5 h-3.5" />
                  </TooltipTrigger>
                  <TooltipContent>Add to notes</TooltipContent>
                </Tooltip>
              </div>
            ) : null}
            <p
              className={`leading-relaxed whitespace-pre-wrap ${fontSizeClass} ${message.content ? "pt-5" : ""}`}
            >
              {message.content}
            </p>
          </>
        ) : (
          <>
            {message.content && (
              <>
                {/* Action buttons row — shown on hover */}
                <div className="absolute top-2 right-2 flex items-center gap-0.5 opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto transition-opacity">
                  {/* Branch button */}
                  <Tooltip>
                    <TooltipTrigger
                      type="button"
                      className={cn(
                        buttonVariants({ variant: "ghost", size: "icon-xs" }),
                        "text-muted-foreground hover:text-foreground",
                      )}
                      onClick={handleBranch}
                      aria-label="Branch conversation"
                    >
                      <GitBranch className="w-3.5 h-3.5" />
                    </TooltipTrigger>
                    <TooltipContent>Branch conversation</TooltipContent>
                  </Tooltip>

                  {/* Speech button */}
                  {!message.isStreaming && (
                    <Tooltip>
                      <TooltipTrigger
                        type="button"
                        className={cn(
                          buttonVariants({ variant: "ghost", size: "icon-xs" }),
                          isSpeaking
                            ? "text-primary"
                            : "text-muted-foreground hover:text-foreground",
                        )}
                        onClick={handleSpeech}
                        aria-label={
                          isSpeaking ? "Stop reading" : "Read response aloud"
                        }
                        aria-pressed={isSpeaking}
                      >
                        {isSpeaking ? (
                          <Pause className="w-3.5 h-3.5" />
                        ) : (
                          <Play className="w-3.5 h-3.5" />
                        )}
                      </TooltipTrigger>
                      <TooltipContent>
                        {isSpeaking ? "Stop reading" : "Read response aloud"}
                      </TooltipContent>
                    </Tooltip>
                  )}

                  {/* Copy button */}
                  <Tooltip>
                    <TooltipTrigger
                      type="button"
                      className={cn(
                        buttonVariants({ variant: "ghost", size: "icon-xs" }),
                        "text-muted-foreground hover:text-foreground",
                        copied && "opacity-100",
                      )}
                      onClick={handleCopy}
                      aria-label={copied ? "Copied" : "Copy response"}
                    >
                      {copied ? (
                        <Check className="w-3.5 h-3.5 text-emerald-400" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </TooltipTrigger>
                    <TooltipContent>
                      {copied ? "Copied" : "Copy response"}
                    </TooltipContent>
                  </Tooltip>

                  {/* Add to notes */}
                  {!message.isStreaming && (
                    <Tooltip>
                      <TooltipTrigger
                        type="button"
                        className={cn(
                          buttonVariants({ variant: "ghost", size: "icon-xs" }),
                          "text-muted-foreground hover:text-foreground",
                        )}
                        onClick={handleAddToNotes}
                        aria-label="Add to notes"
                      >
                        <StickyNote className="w-3.5 h-3.5" />
                      </TooltipTrigger>
                      <TooltipContent>Add to notes</TooltipContent>
                    </Tooltip>
                  )}

                  {/* Share button */}
                  {!message.isStreaming && (
                    <Tooltip>
                      <TooltipTrigger
                        type="button"
                        className={cn(
                          buttonVariants({ variant: "ghost", size: "icon-xs" }),
                          "text-muted-foreground hover:text-foreground",
                        )}
                        onClick={handleShare}
                        aria-label={
                          shared
                            ? "Link copied"
                            : shareFailed
                              ? "Share failed"
                              : "Share response"
                        }
                      >
                        {shared ? (
                          <Link2 className="w-3.5 h-3.5 text-emerald-400" />
                        ) : shareFailed ? (
                          <X className="w-3.5 h-3.5 text-destructive" />
                        ) : (
                          <Share2 className="w-3.5 h-3.5" />
                        )}
                      </TooltipTrigger>
                      <TooltipContent>
                        {shared
                          ? "Link copied"
                          : shareFailed
                            ? "Share failed"
                            : "Share response"}
                      </TooltipContent>
                    </Tooltip>
                  )}
                </div>

                {copied && (
                  <div
                    className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-zinc-800 text-white text-xs rounded-md whitespace-nowrap pointer-events-none"
                    role="status"
                    aria-live="polite"
                  >
                    Copied!
                  </div>
                )}
              </>
            )}

            <div
              className={`prose-chat ${fontSizeClass} ${message.content ? "pr-24" : ""}`}
            >
              {message.content ? (
                <MarkdownContent
                  content={message.content}
                  isStreaming={Boolean(message.isStreaming)}
                />
              ) : message.isStreaming ? (
                <div className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-primary/60 animate-bounce [animation-delay:0ms]" />
                  <span className="w-1.5 h-1.5 rounded-full bg-primary/60 animate-bounce [animation-delay:150ms]" />
                  <span className="w-1.5 h-1.5 rounded-full bg-primary/60 animate-bounce [animation-delay:300ms]" />
                </div>
              ) : null}
              {message.isStreaming && message.content && (
                <span className="inline-block w-0.5 h-4 bg-primary/60 animate-pulse ml-0.5 align-text-bottom" />
              )}
            </div>
          </>
        )}

        <div
          className={`flex flex-wrap items-center gap-2 mt-2 text-xs text-muted-foreground ${
            isUser ? "justify-end" : "justify-end"
          }`}
        >
          {messageTime && (
            <div title={messageTime.toLocaleString()}>
              {formatDistanceToNow(messageTime, { addSuffix: true })}
            </div>
          )}
          {!isUser && (
            <span className="text-[10px] rounded-md border border-border/50 px-1.5 py-0.5">
              {message.usage?.model_id ||
                (message.llm_provider === "bedrock"
                  ? "us.amazon.nova-2-lite-v1:0"
                  : message.llm_provider === "ollama"
                    ? "Ollama"
                    : message.llm_provider || "LLM")}
            </span>
          )}
          {!isUser && message.usage && (
            <span
              className="text-[10px] text-right"
              title={
                message.usage.pricing_note ||
                "Estimated cost from provider usage metadata — not an AWS bill."
              }
            >
              in {message.usage.input_tokens ?? "—"} · out{" "}
              {message.usage.output_tokens ?? "—"} · tot{" "}
              {message.usage.total_tokens ?? "—"}
              {typeof message.usage.estimated_cost_usd === "number"
                ? ` · est $${
                    message.usage.estimated_cost_usd < 0.0001
                      ? message.usage.estimated_cost_usd.toFixed(6)
                      : message.usage.estimated_cost_usd < 0.01
                        ? message.usage.estimated_cost_usd.toFixed(4)
                        : message.usage.estimated_cost_usd.toFixed(3)
                  } (est. ≠ billed)`
                : ""}
            </span>
          )}
          {!isUser && message.response_time_ms && (
            <span className="text-[10px]">
              ⚡ {(message.response_time_ms / 1000).toFixed(1)}s
            </span>
          )}
        </div>
      </div>
      {isUser && (
        <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center shrink-0 mt-0.5">
          <User className="w-4 h-4 text-primary-foreground" />
        </div>
      )}
    </div>
  );
}

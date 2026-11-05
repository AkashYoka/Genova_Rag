"use client";

import { useState, useRef, useEffect, useSyncExternalStore } from "react";
import { formatDistanceToNow } from "date-fns";
import type { ChatMsg } from "@/store/chat-store";
import {
  Copy,
  Check,
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
import MarkdownContent from "@/components/chat/MarkdownContent";
import StreamingText from "@/components/chat/StreamingText";
import { prepareMarkdownForMath } from "@/lib/math-markdown";

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
  const [isSpeaking, setIsSpeaking] = useState(false);
  const copiedTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

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

  const actionBtn = cn(
    buttonVariants({ variant: "ghost", size: "icon-xs" }),
    "h-7 w-7 rounded-md text-[var(--gd-muted)] hover:bg-[var(--gd-raised)] hover:text-[var(--gd-text)]",
  );

  return (
    <div
      className={cn(
        "gc-msg animate-fade-in-up",
        isUser ? "is-user" : "is-assistant",
      )}
    >
      <div className="gc-msg-body select-text">
        {isUser ? (
          <>
            <div className="gc-user-bubble relative group">
              {message.content ? (
                <div className="absolute -left-9 top-1/2 -translate-y-1/2 opacity-0 pointer-events-none transition-opacity group-hover:opacity-100 group-hover:pointer-events-auto">
                  <Tooltip>
                    <TooltipTrigger
                      type="button"
                      className={actionBtn}
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
              <p className={cn("m-0", fontSizeClass)}>{message.content}</p>
            </div>

            {(message.attachments?.length || messageTime) && (
              <div className="gc-msg-meta justify-end">
                {message.attachments?.map((doc) => (
                  <span
                    key={doc.id}
                    className="inline-flex max-w-[10rem] truncate rounded-md border border-[var(--gd-border)] bg-[var(--gd-raised)] px-1.5 py-0.5 text-[10px] text-[var(--gd-muted)]"
                    title={doc.filename}
                  >
                    {doc.filename}
                  </span>
                ))}
                {messageTime && (
                  <div title={messageTime.toLocaleString()}>
                    {formatDistanceToNow(messageTime, { addSuffix: true })}
                  </div>
                )}
              </div>
            )}
          </>
        ) : (
          <>
            <div className={cn("gc-assistant-prose genova-message-prose prose-chat", fontSizeClass)}>
              {message.content ? (
                message.isStreaming ? (
                  <StreamingText
                    liveText={prepareMarkdownForMath(message.content, {
                      isStreaming: true,
                    })}
                    streaming
                    loop={false}
                    fill
                    showActions={false}
                    showFollowUps={false}
                    followUps={[]}
                    sources={[]}
                  />
                ) : (
                  <MarkdownContent content={message.content} />
                )
              ) : message.isStreaming ? (
                <div className="flex items-center gap-1.5 py-1" aria-label="Generating">
                  <span className="h-1.5 w-1.5 rounded-full bg-[var(--gd-accent)]/70 animate-bounce [animation-delay:0ms]" />
                  <span className="h-1.5 w-1.5 rounded-full bg-[var(--gd-accent)]/70 animate-bounce [animation-delay:150ms]" />
                  <span className="h-1.5 w-1.5 rounded-full bg-[var(--gd-accent)]/70 animate-bounce [animation-delay:300ms]" />
                </div>
              ) : null}
            </div>

            {message.content && (
              <div className="gc-msg-actions">
                <Tooltip>
                  <TooltipTrigger
                    type="button"
                    className={actionBtn}
                    onClick={handleBranch}
                    aria-label="Branch conversation"
                  >
                    <GitBranch className="w-3.5 h-3.5" />
                  </TooltipTrigger>
                  <TooltipContent>Branch conversation</TooltipContent>
                </Tooltip>

                {!message.isStreaming && (
                  <Tooltip>
                    <TooltipTrigger
                      type="button"
                      className={cn(
                        actionBtn,
                        isSpeaking && "text-[var(--gd-accent)]",
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

                <Tooltip>
                  <TooltipTrigger
                    type="button"
                    className={cn(actionBtn, copied && "opacity-100")}
                    onClick={handleCopy}
                    aria-label={copied ? "Copied" : "Copy response"}
                  >
                    {copied ? (
                      <Check className="w-3.5 h-3.5 text-[var(--gd-positive)]" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                  </TooltipTrigger>
                  <TooltipContent>
                    {copied ? "Copied" : "Copy response"}
                  </TooltipContent>
                </Tooltip>

                {!message.isStreaming && (
                  <Tooltip>
                    <TooltipTrigger
                      type="button"
                      className={actionBtn}
                      onClick={handleAddToNotes}
                      aria-label="Add to notes"
                    >
                      <StickyNote className="w-3.5 h-3.5" />
                    </TooltipTrigger>
                    <TooltipContent>Add to notes</TooltipContent>
                  </Tooltip>
                )}
              </div>
            )}

            {(messageTime ||
              message.usage ||
              message.response_time_ms ||
              message.llm_provider) && (
              <div className="gc-msg-meta">
                {messageTime && (
                  <div title={messageTime.toLocaleString()}>
                    {formatDistanceToNow(messageTime, { addSuffix: true })}
                  </div>
                )}
                <span className="rounded-md border border-[var(--gd-border)] px-1.5 py-0.5 text-[10px]">
                  {message.usage?.model_id ||
                    (message.llm_provider === "bedrock"
                      ? "us.amazon.nova-2-lite-v1:0"
                      : message.llm_provider === "ollama"
                        ? "Ollama"
                        : message.llm_provider || "LLM")}
                </span>
                {message.usage && (
                  <span
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
                {message.response_time_ms && (
                  <span>{(message.response_time_ms / 1000).toFixed(1)}s</span>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

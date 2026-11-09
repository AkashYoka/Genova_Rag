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
  ThumbsUp,
  ThumbsDown,
  Loader2,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { buttonVariants } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { api } from "@/lib/api";
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

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** Feedback needs the persisted assistant message id (not a local placeholder). */
function isPersistedMessageId(id: string): boolean {
  return UUID_RE.test(id);
}

const FEEDBACK_REASONS: Record<
  "up" | "down",
  Array<{ id: string; label: string }>
> = {
  up: [
    { id: "accurate", label: "Accurate" },
    { id: "helpful", label: "Helpful" },
    { id: "clear", label: "Clear" },
    { id: "well_cited", label: "Well cited" },
    { id: "other", label: "Other" },
  ],
  down: [
    { id: "inaccurate", label: "Inaccurate" },
    { id: "incomplete", label: "Incomplete" },
    { id: "not_relevant", label: "Not relevant" },
    { id: "unclear", label: "Unclear" },
    { id: "poor_citation", label: "Poor citations" },
    { id: "too_long", label: "Too long" },
    { id: "other", label: "Other" },
  ],
};

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
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [feedbackType, setFeedbackType] = useState<"up" | "down">("up");
  const [feedbackReason, setFeedbackReason] = useState("");
  const [feedbackComment, setFeedbackComment] = useState("");
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);
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
  const setMessages = useChatStore((s) => s.setMessages);
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

  const showFeedbackControls =
    !isUser &&
    !message.isStreaming &&
    Boolean(message.content) &&
    (isPersistedMessageId(message.id) || Boolean(message.feedback));

  const openFeedback = (type: "up" | "down") => {
    if (message.feedback) return;
    setFeedbackType(type);
    setFeedbackReason("");
    setFeedbackComment("");
    setFeedbackOpen(true);
  };

  const submitFeedback = async () => {
    if (message.feedback || feedbackSubmitting) return;
    if (feedbackType === "down" && !feedbackReason) {
      toast.error("Choose a reason before submitting feedback.");
      return;
    }
    setFeedbackSubmitting(true);
    try {
      await api.patch(
        `/api/v1/chat/feedback/${encodeURIComponent(message.id)}`,
        {
          feedback: feedbackType,
          reason: feedbackReason || null,
          comment: feedbackComment.trim() || null,
        },
      );
      setMessages((prev) =>
        prev.map((m) =>
          m.id === message.id ? { ...m, feedback: feedbackType } : m,
        ),
      );
      setFeedbackOpen(false);
      toast.success("Thanks — your feedback was saved.");
    } catch (err) {
      const detail = err instanceof Error ? err.message : "";
      toast.error(
        /already|exists|submitted/i.test(detail)
          ? "You already submitted feedback for this response."
          : detail || "Could not save feedback",
      );
    } finally {
      setFeedbackSubmitting(false);
    }
  };

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
                <div
                  className="flex items-center gap-2 py-1 text-[12px] text-[var(--gd-muted)]"
                  aria-label="Generating"
                >
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Generating
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

                {showFeedbackControls && (
                  <>
                    <Tooltip>
                      <TooltipTrigger
                        type="button"
                        className={cn(
                          actionBtn,
                          message.feedback === "up" &&
                            "bg-[var(--gd-accent-soft)] text-[var(--gd-accent)]",
                        )}
                        onClick={() => openFeedback("up")}
                        disabled={
                          Boolean(message.feedback) || feedbackSubmitting
                        }
                        aria-label="Like response"
                        aria-pressed={message.feedback === "up"}
                      >
                        <ThumbsUp className="w-3.5 h-3.5" />
                      </TooltipTrigger>
                      <TooltipContent>
                        {message.feedback === "up" ? "Liked" : "Like response"}
                      </TooltipContent>
                    </Tooltip>

                    <Tooltip>
                      <TooltipTrigger
                        type="button"
                        className={cn(
                          actionBtn,
                          message.feedback === "down" &&
                            "bg-[var(--gd-negative)]/10 text-[var(--gd-negative)]",
                        )}
                        onClick={() => openFeedback("down")}
                        disabled={
                          Boolean(message.feedback) || feedbackSubmitting
                        }
                        aria-label="Dislike response"
                        aria-pressed={message.feedback === "down"}
                      >
                        <ThumbsDown className="w-3.5 h-3.5" />
                      </TooltipTrigger>
                      <TooltipContent>
                        {message.feedback === "down"
                          ? "Disliked"
                          : "Dislike response"}
                      </TooltipContent>
                    </Tooltip>

                    {message.feedback ? (
                      <span className="ml-1 inline-flex items-center gap-1 text-[11px] text-[var(--gd-muted)]">
                        <Check className="h-3 w-3 text-[var(--gd-positive)]" />
                        Feedback saved
                      </span>
                    ) : null}
                  </>
                )}
              </div>
            )}

            {feedbackOpen && !message.feedback ? (
              <div className="mt-2 rounded-[10px] border border-[var(--gd-border)] bg-[var(--gd-surface)] p-3">
                <div className="mb-2 flex items-start justify-between gap-3">
                  <div>
                    <p className="text-[13px] font-medium">
                      {feedbackType === "up"
                        ? "What did you like about this response?"
                        : "What went wrong with this response?"}
                    </p>
                    <p className="mt-0.5 text-[11px] text-[var(--gd-muted)]">
                      {feedbackType === "up"
                        ? "Pick a reason or add a note — both are optional."
                        : "Choose a reason so we can improve. A note is optional."}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setFeedbackOpen(false)}
                    aria-label="Close feedback"
                    className="rounded p-0.5 text-[var(--gd-muted)] transition-colors hover:bg-[var(--gd-raised)] hover:text-[var(--gd-text)]"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>

                <div
                  className="mb-2 flex flex-wrap gap-1.5"
                  role="group"
                  aria-label="Feedback reason"
                >
                  {FEEDBACK_REASONS[feedbackType].map((reason) => (
                    <button
                      key={reason.id}
                      type="button"
                      aria-pressed={feedbackReason === reason.id}
                      onClick={() =>
                        setFeedbackReason((current) =>
                          current === reason.id ? "" : reason.id,
                        )
                      }
                      className={cn(
                        "rounded-md border px-2.5 py-1 text-[12px] transition-colors",
                        feedbackReason === reason.id
                          ? "border-[var(--gd-accent)] bg-[var(--gd-accent-soft)] text-[var(--gd-accent)]"
                          : "border-[var(--gd-border)] text-[var(--gd-muted)] hover:border-[var(--gd-accent)]/40 hover:text-[var(--gd-text)]",
                      )}
                    >
                      {reason.label}
                    </button>
                  ))}
                </div>

                <textarea
                  className="min-h-[64px] w-full resize-y rounded-[8px] border border-[var(--gd-border)] bg-transparent p-2 text-[13px] leading-relaxed outline-none focus:border-[var(--gd-accent)]"
                  maxLength={2000}
                  value={feedbackComment}
                  onChange={(e) => setFeedbackComment(e.target.value)}
                  placeholder={
                    feedbackType === "up"
                      ? "What worked well? (optional)"
                      : "Tell us more (optional)"
                  }
                />

                <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
                  <span className="text-[11px] text-[var(--gd-muted)]">
                    {feedbackType === "down" && !feedbackReason
                      ? "A reason is required for dislikes."
                      : `${feedbackComment.length}/2000`}
                  </span>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => setFeedbackOpen(false)}
                      disabled={feedbackSubmitting}
                      className="h-7 rounded-md px-2 text-[12px] text-[var(--gd-muted)] transition-colors hover:bg-[var(--gd-raised)] hover:text-[var(--gd-text)] disabled:opacity-50"
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      onClick={() => void submitFeedback()}
                      disabled={
                        feedbackSubmitting ||
                        (feedbackType === "down" && !feedbackReason)
                      }
                      className="inline-flex h-7 items-center gap-1.5 rounded-md bg-[var(--gd-primary)] px-2.5 text-[12px] font-medium text-[var(--gd-primary-ink)] transition-opacity disabled:opacity-50"
                    >
                      {feedbackSubmitting ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : null}
                      Submit
                    </button>
                  </div>
                </div>
              </div>
            ) : null}

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
                {(message.usage ||
                  message.response_time_ms ||
                  message.llm_provider) && (
                  <details className="group/meta">
                    <summary className="cursor-pointer list-none text-[11px] text-[var(--gd-faint)] marker:content-none hover:text-[var(--gd-muted)] [&::-webkit-details-marker]:hidden">
                      Details
                    </summary>
                    <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-[10px] text-[var(--gd-faint)]">
                      <span>
                        {message.usage?.model_id ||
                          (message.llm_provider === "bedrock"
                            ? "nova-2-lite"
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
                              }`
                            : ""}
                        </span>
                      )}
                      {message.response_time_ms && (
                        <span>
                          {(message.response_time_ms / 1000).toFixed(1)}s
                        </span>
                      )}
                    </div>
                  </details>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

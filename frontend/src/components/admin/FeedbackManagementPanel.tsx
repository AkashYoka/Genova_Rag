"use client";

import { useCallback, useEffect, useState } from "react";
import {
  ArrowRight,
  Loader2,
  RefreshCw,
  ThumbsDown,
  ThumbsUp,
  X,
} from "lucide-react";

import { api } from "@/lib/api";
import {
  LedgerButton,
  LedgerEmpty,
  LedgerError,
  LedgerKpiStrip,
  LedgerPanel,
  LedgerScrollTable,
  LedgerStatus,
} from "@/components/ledger/LedgerPrimitives";
import { Input } from "@/components/ui/input";

type FeedbackType = "up" | "down";

interface FeedbackItem {
  id: string;
  user_id: string;
  username?: string | null;
  email?: string | null;
  question: string;
  response: string;
  feedback: FeedbackType;
  reason?: string | null;
  comment?: string | null;
  session_id?: string | null;
  message_id: string;
  question_message_id?: string | null;
  created_at?: string | null;
}

interface FeedbackSummary {
  total: number;
  likes: number;
  dislikes: number;
  satisfaction_percentage: number;
}

interface FeedbackResponse {
  total: number;
  page: number;
  page_size: number;
  pages: number;
  summary: FeedbackSummary;
  items: FeedbackItem[];
}

const PAGE_SIZE = 20;

const REASON_LABELS: Record<string, string> = {
  accurate: "Accurate",
  helpful: "Helpful",
  clear: "Clear",
  well_cited: "Well cited",
  inaccurate: "Inaccurate",
  incomplete: "Incomplete",
  not_relevant: "Not relevant",
  unclear: "Unclear",
  poor_citation: "Poor citations",
  too_long: "Too long",
  other: "Other",
};

function formatReason(reason?: string | null): string {
  if (!reason) return "—";
  return (
    REASON_LABELS[reason] ||
    reason.replace(/_/g, " ").replace(/^./, (char) => char.toUpperCase())
  );
}

function formatWhen(iso?: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function userLabel(item: FeedbackItem): string {
  return item.username || item.email || `User ${item.user_id.slice(0, 8)}`;
}

function DetailBlock({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <h3 className="mb-1.5 text-[12px] font-medium text-[var(--gd-muted)]">
        {label}
      </h3>
      <div className="rounded-[8px] border border-[var(--gd-border)] bg-[var(--gd-ground)]/40 p-3 text-[13px] leading-relaxed">
        {children}
      </div>
    </section>
  );
}

export function FeedbackManagementPanel() {
  const [data, setData] = useState<FeedbackResponse | null>(null);
  const [selected, setSelected] = useState<FeedbackItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [feedbackFilter, setFeedbackFilter] = useState<"all" | FeedbackType>(
    "all",
  );
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [page, setPage] = useState(1);

  // Debounce free-text search so each keystroke doesn't hit the API.
  useEffect(() => {
    const timer = window.setTimeout(() => {
      setSearch(searchInput.trim());
      setPage(1);
    }, 350);
    return () => window.clearTimeout(timer);
  }, [searchInput]);

  useEffect(() => {
    if (!selected) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSelected(null);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selected]);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams({
        page: String(page),
        page_size: String(PAGE_SIZE),
      });
      if (feedbackFilter !== "all") params.set("feedback", feedbackFilter);
      if (search) params.set("search", search);
      if (start) params.set("start", start);
      if (end) params.set("end", end);
      const result = await api.get<FeedbackResponse>(
        `/api/v1/admin/feedback?${params.toString()}`,
      );
      setData(result);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not load user feedback",
      );
    } finally {
      setLoading(false);
    }
  }, [page, feedbackFilter, search, start, end]);

  useEffect(() => {
    void load();
  }, [load]);

  const summary = data?.summary;
  const pages = Math.max(1, data?.pages ?? 1);
  const total = data?.total ?? 0;

  return (
    <div className="ledger-stack">
      {error ? <LedgerError>{error}</LedgerError> : null}

      <LedgerKpiStrip
        items={[
          {
            label: "Total feedback",
            value: summary ? String(summary.total) : "—",
            caption: "Rated assistant responses",
          },
          {
            label: "Likes",
            value: summary ? String(summary.likes) : "—",
            caption: "Positive ratings",
            emphasis: "data",
          },
          {
            label: "Dislikes",
            value: summary ? String(summary.dislikes) : "—",
            caption: "Negative ratings",
          },
          {
            label: "Satisfaction",
            value: summary
              ? `${summary.satisfaction_percentage.toFixed(1)}%`
              : "—",
            caption: "Likes ÷ all ratings",
          },
        ]}
      />

      <div className="ledger-toolbar">
        <div className="ledger-filters">
          <div className="ledger-field mb-0">
            <label htmlFor="feedback-search">Search</label>
            <Input
              id="feedback-search"
              value={searchInput}
              onChange={(event) => setSearchInput(event.target.value)}
              placeholder="User, question, or response"
              className="h-10 w-64 rounded-[10px]"
            />
          </div>
          <div className="ledger-field mb-0">
            <label htmlFor="feedback-type">Feedback</label>
            <select
              id="feedback-type"
              value={feedbackFilter}
              onChange={(event) => {
                setFeedbackFilter(event.target.value as "all" | FeedbackType);
                setPage(1);
              }}
              className="h-10 rounded-[10px] border border-[#e8e8e8] bg-[var(--gd-surface)] px-3 text-sm"
            >
              <option value="all">All feedback</option>
              <option value="up">Likes</option>
              <option value="down">Dislikes</option>
            </select>
          </div>
          <div className="ledger-field mb-0">
            <label htmlFor="feedback-start">From</label>
            <Input
              id="feedback-start"
              type="date"
              value={start}
              onChange={(event) => {
                setStart(event.target.value);
                setPage(1);
              }}
              className="h-10 w-40 rounded-[10px]"
            />
          </div>
          <div className="ledger-field mb-0">
            <label htmlFor="feedback-end">To</label>
            <Input
              id="feedback-end"
              type="date"
              value={end}
              onChange={(event) => {
                setEnd(event.target.value);
                setPage(1);
              }}
              className="h-10 w-40 rounded-[10px]"
            />
          </div>
          <LedgerButton onClick={() => void load()} disabled={loading}>
            {loading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <RefreshCw className="h-3.5 w-3.5" />
            )}
            Refresh
          </LedgerButton>
        </div>
        <p className="text-[12px] text-[var(--gd-muted)]">
          {loading
            ? "Loading feedback…"
            : `${total} matching record${total === 1 ? "" : "s"}`}
        </p>
      </div>

      <LedgerPanel
        title="Feedback ledger"
        description="Click a row to read the full question, response, and reason."
      >
        <LedgerScrollTable
          footer={
            <div className="ledger-foot">
              <span>
                Page {page} of {pages}
                {data?.page_size
                  ? ` · ${data.page_size} rows per page`
                  : ""}
              </span>
              <div className="flex gap-2">
                <LedgerButton
                  onClick={() => setPage((value) => Math.max(1, value - 1))}
                  disabled={loading || page <= 1}
                >
                  Previous
                </LedgerButton>
                <LedgerButton
                  onClick={() => setPage((value) => value + 1)}
                  disabled={loading || page >= pages}
                >
                  Next
                  <ArrowRight className="h-3.5 w-3.5" />
                </LedgerButton>
              </div>
            </div>
          }
        >
          <table className="ledger-table" style={{ minWidth: 1000 }}>
            <thead>
              <tr>
                <th>User</th>
                <th>Date and time</th>
                <th>Question</th>
                <th>Response</th>
                <th>Feedback</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {(data?.items || []).map((item) => (
                <tr
                  key={item.id}
                  role="button"
                  tabIndex={0}
                  onClick={() => setSelected(item)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      setSelected(item);
                    }
                  }}
                  className="cursor-pointer"
                >
                  <td>
                    <div className="max-w-[160px]">
                      <p className="truncate font-medium">{userLabel(item)}</p>
                      {item.email ? (
                        <p className="truncate text-[11px] text-[var(--gd-muted)]">
                          {item.email}
                        </p>
                      ) : null}
                    </div>
                  </td>
                  <td className="whitespace-nowrap text-[var(--gd-muted)]">
                    {formatWhen(item.created_at)}
                  </td>
                  <td>
                    <p
                      className="max-w-[240px] truncate"
                      title={item.question}
                    >
                      {item.question || "—"}
                    </p>
                  </td>
                  <td>
                    <p
                      className="max-w-[280px] truncate text-[var(--gd-muted)]"
                      title={item.response}
                    >
                      {item.response || "—"}
                    </p>
                  </td>
                  <td>
                    <LedgerStatus
                      label={item.feedback === "up" ? "Liked" : "Disliked"}
                      tone={item.feedback === "up" ? "ok" : "bad"}
                    />
                  </td>
                  <td className="max-w-[160px]">
                    <span className="truncate">{formatReason(item.reason)}</span>
                  </td>
                </tr>
              ))}
              {!loading && !(data?.items || []).length ? (
                <tr>
                  <td colSpan={6}>
                    <LedgerEmpty>
                      No feedback matches these filters yet.
                    </LedgerEmpty>
                  </td>
                </tr>
              ) : null}
              {loading && !data ? (
                <tr>
                  <td colSpan={6}>
                    <LedgerEmpty>Loading feedback…</LedgerEmpty>
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </LedgerScrollTable>
      </LedgerPanel>

      {selected ? (
        <div
          className="fixed inset-0 z-50 flex justify-end bg-black/40"
          role="dialog"
          aria-modal="true"
          aria-label="Feedback detail"
          onClick={(event) => {
            if (event.target === event.currentTarget) setSelected(null);
          }}
        >
          <aside className="flex h-full w-full max-w-2xl flex-col border-l border-[var(--gd-border)] bg-[var(--gd-surface)] shadow-2xl">
            <div className="flex items-start justify-between gap-3 border-b border-[var(--gd-border)] px-5 py-4">
              <div>
                <p className="text-[12px] text-[var(--gd-muted)]">
                  Feedback detail
                </p>
                <h2 className="text-[18px] font-semibold tracking-tight">
                  {userLabel(selected)}
                </h2>
                <p className="mt-1 flex flex-wrap items-center gap-2 text-[12px] text-[var(--gd-muted)]">
                  {formatWhen(selected.created_at)}
                  <LedgerStatus
                    label={selected.feedback === "up" ? "Liked" : "Disliked"}
                    tone={selected.feedback === "up" ? "ok" : "bad"}
                  />
                </p>
              </div>
              <div className="flex items-center gap-2">
                {selected.feedback === "up" ? (
                  <ThumbsUp className="h-4 w-4 text-[var(--gd-positive)]" />
                ) : (
                  <ThumbsDown className="h-4 w-4 text-[var(--gd-negative)]" />
                )}
                <button
                  type="button"
                  onClick={() => setSelected(null)}
                  aria-label="Close feedback detail"
                  className="rounded p-1 text-[var(--gd-muted)] transition-colors hover:bg-[var(--gd-raised)] hover:text-[var(--gd-text)]"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>

            <div className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
              <DetailBlock label="User's question">
                <p className="whitespace-pre-wrap break-words">
                  {selected.question || "—"}
                </p>
              </DetailBlock>

              <DetailBlock label="LLM response">
                <p className="whitespace-pre-wrap break-words">
                  {selected.response || "—"}
                </p>
              </DetailBlock>

              <div className="grid gap-4 sm:grid-cols-2">
                <DetailBlock label="Feedback">
                  <LedgerStatus
                    label={selected.feedback === "up" ? "Liked" : "Disliked"}
                    tone={selected.feedback === "up" ? "ok" : "bad"}
                  />
                  <p className="mt-2 text-[13px]">
                    Reason: {formatReason(selected.reason)}
                  </p>
                </DetailBlock>

                <DetailBlock label="Additional comments">
                  <p className="whitespace-pre-wrap break-words">
                    {selected.comment || "No additional comment provided."}
                  </p>
                </DetailBlock>
              </div>

              <DetailBlock label="References">
                <dl className="grid gap-2 text-[12px]">
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <dt className="text-[var(--gd-muted)]">Conversation</dt>
                    <dd className="ledger-mono break-all">
                      {selected.session_id || "—"}
                    </dd>
                  </div>
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <dt className="text-[var(--gd-muted)]">Response message</dt>
                    <dd className="ledger-mono break-all">
                      {selected.message_id}
                    </dd>
                  </div>
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <dt className="text-[var(--gd-muted)]">Question message</dt>
                    <dd className="ledger-mono break-all">
                      {selected.question_message_id || "—"}
                    </dd>
                  </div>
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <dt className="text-[var(--gd-muted)]">User ID</dt>
                    <dd className="ledger-mono break-all">{selected.user_id}</dd>
                  </div>
                </dl>
              </DetailBlock>
            </div>

            <div className="flex justify-end border-t border-[var(--gd-border)] px-5 py-4">
              <LedgerButton onClick={() => setSelected(null)}>Close</LedgerButton>
            </div>
          </aside>
        </div>
      ) : null}
    </div>
  );
}

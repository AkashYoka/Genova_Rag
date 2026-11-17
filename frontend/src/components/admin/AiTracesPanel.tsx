"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import {
  ArrowRight,
  Loader2,
  RefreshCw,
  Search,
  X,
} from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";

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
import {
  AiTraceDetail,
  type AITraceDetail as AITraceDetailType,
} from "@/components/admin/AiTraceDetail";

interface AITraceListItem {
  trace_id: string;
  user_id?: string | null;
  username?: string | null;
  email?: string | null;
  question: string;
  request_type: string;
  status: string;
  created_at?: string | null;
  duration_ms?: number | null;
  total_tokens?: number | null;
  retrieved_chunk_count?: number | null;
  final_answer_status?: string | null;
  early_exit?: boolean;
  model_name?: string | null;
  provider?: string | null;
}

interface AITraceListResponse {
  total: number;
  page: number;
  page_size: number;
  pages: number;
  summary: {
    total: number;
    success: number;
    failed: number;
    partial: number;
    avg_duration_ms?: number | null;
  };
  items: AITraceListItem[];
}

const PAGE_SIZE = 20;

function formatWhen(iso?: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function formatMs(ms?: number | null): string {
  if (ms == null || Number.isNaN(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function truncate(text: string, n = 80): string {
  if (!text) return "—";
  return text.length > n ? `${text.slice(0, n)}…` : text;
}

function userLabel(item: AITraceListItem): string {
  return item.username || item.email || (item.user_id ? item.user_id.slice(0, 8) : "—");
}

function statusTone(status: string): "positive" | "negative" | "neutral" {
  if (status === "success") return "positive";
  if (status === "failed") return "negative";
  return "neutral";
}

function AiTracesPanelInner({
  syncUrl = false,
  initialId = null,
}: {
  syncUrl?: boolean;
  initialId?: string | null;
}) {
  const router = useRouter();

  const [data, setData] = useState<AITraceListResponse | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(initialId);
  const [detail, setDetail] = useState<AITraceDetailType | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [lookupError, setLookupError] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [lookupInput, setLookupInput] = useState(initialId || "");
  const [traceIdInput, setTraceIdInput] = useState("");
  const [traceId, setTraceId] = useState("");
  const [userInput, setUserInput] = useState("");
  const [userQ, setUserQ] = useState("");
  const [questionInput, setQuestionInput] = useState("");
  const [questionQ, setQuestionQ] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [requestType, setRequestType] = useState("all");
  const [failedOnly, setFailedOnly] = useState(false);
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [page, setPage] = useState(1);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setTraceId(traceIdInput.trim());
      setUserQ(userInput.trim());
      setQuestionQ(questionInput.trim());
      setPage(1);
    }, 350);
    return () => window.clearTimeout(timer);
  }, [traceIdInput, userInput, questionInput]);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams({
        page: String(page),
        page_size: String(PAGE_SIZE),
      });
      if (traceId) params.set("trace_id", traceId);
      if (userQ) params.set("user", userQ);
      if (questionQ) params.set("q", questionQ);
      if (start) params.set("start", start);
      if (end) params.set("end", end);
      if (failedOnly) params.set("failed_only", "true");
      else if (statusFilter !== "all") params.set("status", statusFilter);
      if (requestType !== "all") params.set("request_type", requestType);
      const result = await api.get<AITraceListResponse>(
        `/api/v1/admin/ai-traces?${params.toString()}`,
      );
      setData(result);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not load AI traces",
      );
    } finally {
      setLoading(false);
    }
  }, [
    page,
    traceId,
    userQ,
    questionQ,
    start,
    end,
    statusFilter,
    requestType,
    failedOnly,
  ]);

  useEffect(() => {
    void load();
  }, [load]);

  const openDetail = useCallback(
    (id: string) => {
      setLookupError("");
      setSelectedId(id);
      setLookupInput(id);
      if (syncUrl) {
        router.replace(`/admin/ai-traces?id=${encodeURIComponent(id)}`);
      }
    },
    [router, syncUrl],
  );

  const closeDetail = useCallback(() => {
    setSelectedId(null);
    setDetail(null);
    setLookupError("");
    if (syncUrl) {
      router.replace("/admin/ai-traces");
    }
  }, [router, syncUrl]);

  const lookupTrace = useCallback(async () => {
    const id = lookupInput.trim();
    if (!id) {
      setLookupError("Enter a trace_id to look up.");
      return;
    }
    setLookupError("");
    setDetailLoading(true);
    try {
      const result = await api.get<AITraceDetailType>(
        `/api/v1/admin/ai-traces/${encodeURIComponent(id)}`,
      );
      setDetail(result);
      setSelectedId(result.trace_id);
      if (syncUrl) {
        router.replace(
          `/admin/ai-traces?id=${encodeURIComponent(result.trace_id)}`,
        );
      }
    } catch (err) {
      setDetail(null);
      setSelectedId(null);
      setLookupError(
        err instanceof Error ? err.message : "Trace not found",
      );
    } finally {
      setDetailLoading(false);
    }
  }, [lookupInput, router, syncUrl]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setDetailLoading(true);
    setLookupError("");
    void (async () => {
      try {
        const result = await api.get<AITraceDetailType>(
          `/api/v1/admin/ai-traces/${encodeURIComponent(selectedId)}`,
        );
        if (!cancelled) setDetail(result);
      } catch (err) {
        if (!cancelled) {
          setDetail(null);
          setLookupError(
            err instanceof Error ? err.message : "Could not load trace detail",
          );
        }
      } finally {
        if (!cancelled) setDetailLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  useEffect(() => {
    if (!selectedId) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeDetail();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selectedId, closeDetail]);

  const summary = data?.summary;
  const pages = Math.max(1, data?.pages ?? 1);
  const total = data?.total ?? 0;

  return (
    <div className="ledger-stack">
      {error ? <LedgerError>{error}</LedgerError> : null}

      <section className="rounded-[12px] border border-[var(--gd-border)] bg-[var(--gd-surface)] p-4 shadow-sm">
        <h2 className="text-[15px] font-semibold tracking-tight">
          Processing Trace Lookup
        </h2>
        <p className="mt-1 text-[13px] text-[var(--gd-muted)]">
          Enter a <strong>trace_id</strong> from a chat answer (WebSocket meta /
          done event) to view the full RAG processing pipeline — retrieval,
          prompts, and model output.
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          <Input
            value={lookupInput}
            onChange={(e) => setLookupInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void lookupTrace();
            }}
            placeholder="e.g. 5a43a10f-7175-41f5-80dd-5d93d0a13485"
            className="h-10 min-w-[280px] flex-1 rounded-[10px] font-mono text-[13px]"
          />
          <LedgerButton onClick={() => void lookupTrace()} disabled={detailLoading}>
            {detailLoading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Search className="h-3.5 w-3.5" />
            )}
            Lookup Trace
          </LedgerButton>
          {selectedId ? (
            <LedgerButton onClick={closeDetail}>
              <X className="h-3.5 w-3.5" />
              Clear
            </LedgerButton>
          ) : null}
        </div>
        {lookupError ? (
          <p className="mt-2 text-[13px] text-[var(--gd-negative)]">{lookupError}</p>
        ) : null}
      </section>

      {detailLoading && selectedId ? (
        <div className="flex justify-center py-10">
          <Loader2 className="h-6 w-6 animate-spin text-[var(--gd-muted)]" />
        </div>
      ) : null}

      {!detailLoading && detail ? <AiTraceDetail detail={detail} /> : null}

      <LedgerKpiStrip
        items={[
          {
            label: "Traces",
            value: summary ? String(summary.total) : "—",
            caption: "Matching requests",
          },
          {
            label: "Success",
            value: summary ? String(summary.success) : "—",
            emphasis: "data",
          },
          {
            label: "Failed",
            value: summary ? String(summary.failed) : "—",
          },
          {
            label: "Avg duration",
            value: summary?.avg_duration_ms != null
              ? formatMs(summary.avg_duration_ms)
              : "—",
            caption: "Across filtered set",
          },
        ]}
      />

      <div className="ledger-toolbar">
        <div className="ledger-filters flex flex-wrap gap-2">
          <div className="ledger-field mb-0">
            <label htmlFor="trace-id">Filter Trace ID</label>
            <Input
              id="trace-id"
              value={traceIdInput}
              onChange={(e) => setTraceIdInput(e.target.value)}
              placeholder="UUID prefix"
              className="h-10 w-48 rounded-[10px]"
            />
          </div>
          <div className="ledger-field mb-0">
            <label htmlFor="trace-user">User</label>
            <Input
              id="trace-user"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              placeholder="Username or email"
              className="h-10 w-44 rounded-[10px]"
            />
          </div>
          <div className="ledger-field mb-0">
            <label htmlFor="trace-q">Question</label>
            <Input
              id="trace-q"
              value={questionInput}
              onChange={(e) => setQuestionInput(e.target.value)}
              placeholder="Search question text"
              className="h-10 w-56 rounded-[10px]"
            />
          </div>
          <div className="ledger-field mb-0">
            <label htmlFor="trace-status">Status</label>
            <select
              id="trace-status"
              value={statusFilter}
              onChange={(e) => {
                setStatusFilter(e.target.value);
                setFailedOnly(false);
                setPage(1);
              }}
              className="h-10 rounded-[10px] border border-[#e8e8e8] bg-[var(--gd-surface)] px-3 text-sm"
            >
              <option value="all">All</option>
              <option value="success">Success</option>
              <option value="failed">Failed</option>
              <option value="partial">Partial</option>
            </select>
          </div>
          <div className="ledger-field mb-0">
            <label htmlFor="trace-type">Type</label>
            <select
              id="trace-type"
              value={requestType}
              onChange={(e) => {
                setRequestType(e.target.value);
                setPage(1);
              }}
              className="h-10 rounded-[10px] border border-[#e8e8e8] bg-[var(--gd-surface)] px-3 text-sm"
            >
              <option value="all">All types</option>
              <option value="rag">RAG</option>
              <option value="quiz">Quiz</option>
              <option value="special_turn">Special turn</option>
            </select>
          </div>
          <div className="ledger-field mb-0">
            <label htmlFor="trace-start">From</label>
            <Input
              id="trace-start"
              type="date"
              value={start}
              onChange={(e) => {
                setStart(e.target.value);
                setPage(1);
              }}
              className="h-10 w-40 rounded-[10px]"
            />
          </div>
          <div className="ledger-field mb-0">
            <label htmlFor="trace-end">To</label>
            <Input
              id="trace-end"
              type="date"
              value={end}
              onChange={(e) => {
                setEnd(e.target.value);
                setPage(1);
              }}
              className="h-10 w-40 rounded-[10px]"
            />
          </div>
          <label className="mb-0 flex items-end gap-2 pb-2 text-[12px]">
            <input
              type="checkbox"
              checked={failedOnly}
              onChange={(e) => {
                setFailedOnly(e.target.checked);
                setPage(1);
              }}
            />
            Failed only
          </label>
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
            ? "Loading traces…"
            : `${total} matching record${total === 1 ? "" : "s"}`}
        </p>
      </div>

      <LedgerPanel
        title="Recent AI processing traces"
        description="One row per user question. Open a Trace ID for the full pipeline view above."
      >
        <LedgerScrollTable
          footer={
            <div className="ledger-foot">
              <span>
                Page {page} of {pages}
                {data?.page_size ? ` · ${data.page_size} rows per page` : ""}
              </span>
              <div className="flex gap-2">
                <LedgerButton
                  onClick={() => setPage((v) => Math.max(1, v - 1))}
                  disabled={loading || page <= 1}
                >
                  Previous
                </LedgerButton>
                <LedgerButton
                  onClick={() => setPage((v) => v + 1)}
                  disabled={loading || page >= pages}
                >
                  Next
                  <ArrowRight className="h-3.5 w-3.5" />
                </LedgerButton>
              </div>
            </div>
          }
        >
          <table className="ledger-table" style={{ minWidth: 1100 }}>
            <thead>
              <tr>
                <th>Trace ID</th>
                <th>User</th>
                <th>Question</th>
                <th>Type</th>
                <th>Status</th>
                <th>Time</th>
                <th>Duration</th>
                <th>Tokens</th>
                <th>Chunks</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {loading && !data ? (
                <tr>
                  <td colSpan={10} className="py-8 text-center text-[var(--gd-muted)]">
                    <Loader2 className="mx-auto h-5 w-5 animate-spin" />
                  </td>
                </tr>
              ) : null}
              {!loading && (data?.items.length ?? 0) === 0 ? (
                <tr>
                  <td colSpan={10}>
                    <LedgerEmpty>No traces match these filters.</LedgerEmpty>
                  </td>
                </tr>
              ) : null}
              {(data?.items || []).map((item) => (
                <tr
                  key={item.trace_id}
                  className="cursor-pointer"
                  onClick={() => openDetail(item.trace_id)}
                >
                  <td className="font-mono text-[11px]">
                    {item.trace_id.slice(0, 8)}…
                  </td>
                  <td>{userLabel(item)}</td>
                  <td title={item.question}>{truncate(item.question, 72)}</td>
                  <td>{item.request_type}</td>
                  <td>
                    <LedgerStatus tone={statusTone(item.status)}>
                      {item.status}
                    </LedgerStatus>
                  </td>
                  <td className="whitespace-nowrap text-[12px]">
                    {formatWhen(item.created_at)}
                  </td>
                  <td>{formatMs(item.duration_ms)}</td>
                  <td>{item.total_tokens ?? "—"}</td>
                  <td>{item.retrieved_chunk_count ?? "—"}</td>
                  <td>
                    <LedgerButton
                      onClick={(e) => {
                        e.stopPropagation();
                        openDetail(item.trace_id);
                      }}
                    >
                      View
                    </LedgerButton>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </LedgerScrollTable>
      </LedgerPanel>
    </div>
  );
}

function AiTracesPanelWithUrl() {
  const searchParams = useSearchParams();
  return (
    <AiTracesPanelInner syncUrl initialId={searchParams.get("id")} />
  );
}

export function AiTracesPanel({ syncUrl = false }: { syncUrl?: boolean }) {
  if (syncUrl) {
    return (
      <Suspense
        fallback={
          <div className="p-4 text-sm text-[var(--gd-muted)]">Loading…</div>
        }
      >
        <AiTracesPanelWithUrl />
      </Suspense>
    );
  }
  return <AiTracesPanelInner syncUrl={false} />;
}

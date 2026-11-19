"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, RefreshCw, Route, Search, Timer } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";

import { api } from "@/lib/api";
import {
  AdminButton,
  Badge,
  Callout,
  Card,
  Drawer,
  EmptyState,
  Field,
  PageHeader,
  Pager,
  SearchInput,
  SelectInput,
  Stat,
  StatGrid,
  TableSkeleton,
  TableWrap,
  TextInput,
  type Tone,
} from "@/components/admin/console/ConsoleUI";
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
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatMs(ms?: number | null): string {
  if (ms == null || Number.isNaN(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function userLabel(item: AITraceListItem): string {
  return item.username || item.email || (item.user_id ? item.user_id.slice(0, 8) : "—");
}

function toneFor(status: string): Tone {
  const s = (status || "").toLowerCase();
  if (s === "success") return "ok";
  if (s === "failed") return "danger";
  if (s === "partial") return "warn";
  return "neutral";
}

function statusLabel(status: string) {
  const s = (status || "").toLowerCase();
  if (s === "success") return "Succeeded";
  if (s === "failed") return "Failed";
  if (s === "partial") return "Partial";
  return status || "Unknown";
}

function requestTypeLabel(type: string) {
  if (type === "rag") return "Document answer";
  if (type === "quiz") return "Quiz";
  if (type === "special_turn") return "Special turn";
  return type || "—";
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
      if (statusFilter !== "all") params.set("status", statusFilter);
      if (requestType !== "all") params.set("request_type", requestType);
      const result = await api.get<AITraceListResponse>(
        `/api/v1/admin/ai-traces?${params.toString()}`,
      );
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load answer traces");
    } finally {
      setLoading(false);
    }
  }, [page, traceId, userQ, questionQ, start, end, statusFilter, requestType]);

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
      setLookupError("Paste a trace ID first.");
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
        router.replace(`/admin/ai-traces?id=${encodeURIComponent(result.trace_id)}`);
      }
    } catch (err) {
      setDetail(null);
      setSelectedId(null);
      setLookupError(err instanceof Error ? err.message : "That trace ID was not found.");
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
            err instanceof Error ? err.message : "Could not load the full trace.",
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

  const summary = data?.summary;
  const pages = Math.max(1, data?.pages ?? 1);
  const total = data?.total ?? 0;
  const items = data?.items ?? [];
  const hasFilters = Boolean(
    traceId || userQ || questionQ || start || end || statusFilter !== "all" || requestType !== "all",
  );

  const clearFilters = () => {
    setTraceIdInput("");
    setUserInput("");
    setQuestionInput("");
    setTraceId("");
    setUserQ("");
    setQuestionQ("");
    setStatusFilter("all");
    setRequestType("all");
    setStart("");
    setEnd("");
    setPage(1);
  };

  const lookupPanel = useMemo(
    () => (
      <Card
        title="Have a trace ID?"
        description="Paste an ID from an answer’s diagnostics to jump straight to that record."
      >
        <div className="adm-card__body--pad">
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            <TextInput
              value={lookupInput}
              onChange={(e) => setLookupInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void lookupTrace();
              }}
              placeholder="e.g. 5a43a10f-7175-41f5-80dd-5d93d0a13485"
              aria-label="Trace ID to open"
              style={{ flex: "1 1 280px", fontFamily: "var(--adm-mono)", fontSize: 12.5 }}
            />
            <AdminButton
              variant="secondary"
              onClick={() => void lookupTrace()}
              disabled={detailLoading}
            >
              <Search /> Open trace
            </AdminButton>
          </div>
          {lookupError ? (
            <p className="adm-hint" style={{ color: "var(--adm-danger)", marginTop: 8 }}>
              {lookupError}
            </p>
          ) : (
            <p className="adm-hint" style={{ marginTop: 8 }}>
              Only administrators can read these records. Nothing here changes the platform.
            </p>
          )}
        </div>
      </Card>
    ),
    [lookupInput, detailLoading, lookupError, lookupTrace],
  );

  return (
    <>
      <PageHeader
        title="Answer traces"
        description="A step-by-step record of how each answer was produced — which document passages were used and which model replied."
        actions={
          <AdminButton variant="secondary" onClick={() => void load()} disabled={loading}>
            <RefreshCw className={loading ? "animate-spin" : undefined} /> Refresh
          </AdminButton>
        }
      />

      {error ? (
        <div style={{ marginBottom: 16 }}>
          <Callout tone="danger" icon={<AlertTriangle />}>
            <strong>Answer traces could not be loaded.</strong> {error}
          </Callout>
        </div>
      ) : null}

      <StatGrid>
        <Stat
          label="Traces"
          icon={<Route />}
          value={summary ? summary.total.toLocaleString() : "—"}
          caption="Matching requests"
        />
        <Stat
          label="Succeeded"
          icon={<Route />}
          value={summary ? summary.success.toLocaleString() : "—"}
          caption="Produced a full answer"
        />
        <Stat
          label="Failed"
          icon={<AlertTriangle />}
          value={summary ? summary.failed.toLocaleString() : "—"}
          caption={summary?.failed ? "Open one to see why" : "Nothing failed"}
          attention={(summary?.failed ?? 0) > 0}
        />
        <Stat
          label="Average time"
          icon={<Timer />}
          value={summary?.avg_duration_ms != null ? formatMs(summary.avg_duration_ms) : "—"}
          caption="Across the filtered set"
        />
      </StatGrid>

      <div style={{ marginTop: 20 }}>
        <div className="adm-toolbar">
          <SearchInput
            label="Search question text"
            value={questionInput}
            onChange={setQuestionInput}
            placeholder="Search the question"
          />
          <Field label="Person" htmlFor="trace-user">
            <TextInput
              id="trace-user"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              placeholder="Name or email"
              style={{ width: 170 }}
            />
          </Field>
          <Field label="Trace ID" htmlFor="trace-id">
            <TextInput
              id="trace-id"
              value={traceIdInput}
              onChange={(e) => setTraceIdInput(e.target.value)}
              placeholder="ID prefix"
              style={{ width: 160, fontFamily: "var(--adm-mono)", fontSize: 12.5 }}
            />
          </Field>
          <Field label="Outcome" htmlFor="trace-status">
            <SelectInput
              id="trace-status"
              value={statusFilter}
              onChange={(e) => {
                setStatusFilter(e.target.value);
                setPage(1);
              }}
              style={{ width: 150 }}
            >
              <option value="all">Any outcome</option>
              <option value="success">Succeeded</option>
              <option value="failed">Failed</option>
              <option value="partial">Partial</option>
            </SelectInput>
          </Field>
          <Field label="Request type" htmlFor="trace-type">
            <SelectInput
              id="trace-type"
              value={requestType}
              onChange={(e) => {
                setRequestType(e.target.value);
                setPage(1);
              }}
              style={{ width: 170 }}
            >
              <option value="all">All types</option>
              <option value="rag">Document answer</option>
              <option value="quiz">Quiz</option>
              <option value="special_turn">Special turn</option>
            </SelectInput>
          </Field>
          <Field label="From" htmlFor="trace-start">
            <TextInput
              id="trace-start"
              type="date"
              value={start}
              onChange={(e) => {
                setStart(e.target.value);
                setPage(1);
              }}
              style={{ width: 158 }}
            />
          </Field>
          <Field label="To" htmlFor="trace-end">
            <TextInput
              id="trace-end"
              type="date"
              value={end}
              onChange={(e) => {
                setEnd(e.target.value);
                setPage(1);
              }}
              style={{ width: 158 }}
            />
          </Field>
          {hasFilters ? (
            <AdminButton variant="ghost" onClick={clearFilters}>
              Clear filters
            </AdminButton>
          ) : null}
          <span className="adm-toolbar__spacer" />
          <span className="adm-toolbar__meta">
            {loading ? "Loading…" : `${total.toLocaleString()} matching trace${total === 1 ? "" : "s"}`}
          </span>
        </div>

        <div className="adm-stack">
          {lookupPanel}

          <Card
            title="Recent answer traces"
            description="One row per question. Select a row to see every step it went through."
          >
            {loading && !data ? (
              <TableSkeleton rows={6} />
            ) : items.length ? (
              <TableWrap height="auto">
                <table className="adm-table">
                  <thead>
                    <tr>
                      <th>Trace</th>
                      <th>Person</th>
                      <th>Question</th>
                      <th className="adm-col-md">Type</th>
                      <th>Outcome</th>
                      <th className="adm-col-md">When</th>
                      <th data-align="right" className="adm-col-sm">
                        Time
                      </th>
                      <th data-align="right" className="adm-col-sm">
                        Tokens
                      </th>
                      <th aria-label="Open" />
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((item) => (
                      <tr
                        key={item.trace_id}
                        data-clickable="true"
                        tabIndex={0}
                        onClick={() => openDetail(item.trace_id)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault();
                            openDetail(item.trace_id);
                          }
                        }}
                      >
                        <td>
                          <div className="adm-cell-stack">
                            <span className="adm-cell-main adm-mono" style={{ fontWeight: 450 }}>
                              {item.trace_id.slice(0, 8)}
                            </span>
                            <span className="adm-cell-sub">
                              {item.model_name || item.provider || "No model recorded"}
                            </span>
                          </div>
                        </td>
                        <td className="adm-nowrap">{userLabel(item)}</td>
                        <td>
                          <span
                            className="adm-clamp"
                            style={{ display: "-webkit-box", maxWidth: 440 }}
                            title={item.question}
                          >
                            {item.question || "No question recorded"}
                          </span>
                        </td>
                        <td className="adm-col-md">
                          <span style={{ fontSize: 12.5 }}>{requestTypeLabel(item.request_type)}</span>
                        </td>
                        <td>
                          <Badge tone={toneFor(item.status)}>{statusLabel(item.status)}</Badge>
                        </td>
                        <td className="adm-col-md adm-muted adm-nowrap" style={{ fontSize: 12.5 }}>
                          {formatWhen(item.created_at)}
                        </td>
                        <td data-align="right" className="adm-num adm-col-sm">
                          {formatMs(item.duration_ms)}
                        </td>
                        <td data-align="right" className="adm-num adm-col-sm">
                          {(item.total_tokens ?? 0).toLocaleString()}
                        </td>
                        <td data-align="right">
                          <span className="adm-link">Open</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </TableWrap>
            ) : (
              <EmptyState
                icon={<Route />}
                title={hasFilters ? "No traces match these filters" : "No answer traces yet"}
                text={
                  hasFilters
                    ? "Try a wider date range, or clear the person and ID filters."
                    : "Traces appear as soon as someone asks Genova a question."
                }
                action={
                  hasFilters ? (
                    <AdminButton size="sm" variant="secondary" onClick={clearFilters}>
                      Clear filters
                    </AdminButton>
                  ) : undefined
                }
              />
            )}
            {items.length ? (
              <div className="adm-card__foot">
                <span>
                  Showing {(page - 1) * PAGE_SIZE + 1}–{Math.min(page * PAGE_SIZE, total)} of{" "}
                  {total.toLocaleString()}
                </span>
                <Pager
                  page={page}
                  pages={pages}
                  total={total}
                  onPage={setPage}
                  noun="traces"
                  nounOne="trace"
                />
              </div>
            ) : null}
          </Card>

          <Callout icon={<Route />}>
            <strong>What a trace shows:</strong> the exact search queries used, the document passages
            that were retrieved, the instructions sent to the model, and how long each step took.
          </Callout>
        </div>
      </div>

      <Drawer
        open={selectedId !== null}
        onClose={closeDetail}
        wide
        title="Trace detail"
        description={
          detail
            ? `${detail.username || detail.email || "Unknown person"} · ${formatWhen(detail.created_at)}`
            : selectedId
              ? `Loading ${selectedId.slice(0, 8)}…`
              : undefined
        }
        footer={
          <AdminButton variant="secondary" onClick={closeDetail}>
            Close
          </AdminButton>
        }
      >
        {detailLoading && !detail ? (
          <div className="adm-stack">
            <TableSkeleton rows={3} />
            <TableSkeleton rows={4} />
          </div>
        ) : detail ? (
          <AiTraceDetail detail={detail} />
        ) : lookupError ? (
          <Callout tone="danger" icon={<AlertTriangle />}>
            <strong>Trace could not be opened.</strong> {lookupError}
          </Callout>
        ) : (
          <EmptyState
            icon={<Route />}
            title="Nothing selected"
            text="Pick a row from the table to inspect how an answer was produced."
          />
        )}
      </Drawer>
    </>
  );
}

function AiTracesPanelWithUrl() {
  const searchParams = useSearchParams();
  return <AiTracesPanelInner syncUrl initialId={searchParams.get("id")} />;
}

export function AiTracesPanel({ syncUrl = false }: { syncUrl?: boolean }) {
  if (syncUrl) {
    return (
      <Suspense fallback={<TableSkeleton rows={6} />}>
        <AiTracesPanelWithUrl />
      </Suspense>
    );
  }
  return <AiTracesPanelInner syncUrl={false} />;
}

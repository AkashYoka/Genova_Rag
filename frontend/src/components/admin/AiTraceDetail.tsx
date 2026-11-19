"use client";

import { useMemo, useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  Copy,
  MessageSquare,
  Sparkles,
} from "lucide-react";

import {
  Badge,
  Callout,
  Card,
  DefinitionList,
  EmptyState,
  type Tone,
} from "@/components/admin/console/ConsoleUI";

export interface AITraceStep {
  id: string;
  sequence: number;
  name: string;
  status: string;
  started_at?: string | null;
  finished_at?: string | null;
  duration_ms?: number | null;
  input_summary?: string | null;
  output_summary?: string | null;
  error_message?: string | null;
  metadata?: unknown;
}

export interface AITraceRetrieval {
  id: string;
  sequence: number;
  search_query?: string | null;
  search_method?: string | null;
  document_ids?: unknown[];
  document_names?: unknown[];
  chunk_ids?: unknown[];
  chunks?: Array<Record<string, unknown>>;
  filters?: unknown;
  rejected?: unknown;
  selected_chunk_ids?: unknown[];
  result_count?: number | null;
  duration_ms?: number | null;
  metadata?: unknown;
}

export interface AITraceLLMCall {
  id: string;
  sequence: number;
  model_name?: string | null;
  provider?: string | null;
  system_prompt_version?: string | null;
  status?: string | null;
  latency_ms?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  cached_tokens?: number | null;
  total_tokens?: number | null;
  estimated_cost_usd?: number | null;
  cost_type?: string | null;
  pricing_note?: string | null;
  error_message?: string | null;
  system_prompt?: string | null;
  user_prompt?: string | null;
  context_text?: string | null;
  raw_response?: string | null;
  validation_result?: unknown;
  chunks_sent_to_llm?: Array<Record<string, unknown>>;
}

export interface AITraceDetail {
  trace_id: string;
  user_id?: string | null;
  username?: string | null;
  email?: string | null;
  session_id?: string | null;
  message_id?: string | null;
  question: string;
  request_type: string;
  status: string;
  final_answer_status?: string | null;
  final_answer?: string | null;
  error_summary?: string | null;
  early_exit?: boolean;
  early_exit_reason?: string | null;
  created_at?: string | null;
  duration_ms?: number | null;
  model_name?: string | null;
  provider?: string | null;
  total_tokens?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  cached_tokens?: number | null;
  retrieved_chunk_count?: number | null;
  steps: AITraceStep[];
  retrievals: AITraceRetrieval[];
  llm_calls: AITraceLLMCall[];
  feedback?: Array<{
    feedback: string;
    reason?: string | null;
    comment?: string | null;
    created_at?: string | null;
  }>;
  metadata?: Record<string, unknown>;
  duration_breakdown?: Array<{
    name: string;
    duration_ms?: number | null;
    status?: string;
  }>;
}

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
    second: "2-digit",
  });
}

function formatMs(ms?: number | null): string {
  if (ms == null || Number.isNaN(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function prettyJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function stepDetailsPayload(step: AITraceStep): Record<string, unknown> {
  const payload: Record<string, unknown> = {};
  if (step.input_summary) payload.input_summary = step.input_summary;
  if (step.output_summary) payload.output_summary = step.output_summary;
  if (step.error_message) payload.error_message = step.error_message;
  if (step.metadata && typeof step.metadata === "object") {
    Object.assign(payload, step.metadata as Record<string, unknown>);
  }
  return payload;
}

function toneFor(status?: string | null): Tone {
  const s = (status || "").toLowerCase();
  if (["success", "completed", "ok"].includes(s)) return "ok";
  if (["failed", "error"].includes(s)) return "danger";
  if (["partial", "warning"].includes(s)) return "warn";
  return "neutral";
}

function formatCost(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return "—";
  if (value === 0) return "$0.00";
  if (Math.abs(value) < 0.0001) return `$${value.toFixed(8)}`;
  return `$${value.toFixed(6)}`;
}

function humanRequestType(type: string) {
  if (type === "rag") return "Document answer";
  if (type === "quiz") return "Quiz";
  if (type === "special_turn") return "Special turn";
  return type || "—";
}

function statusLabel(status?: string | null) {
  const s = (status || "").toLowerCase();
  if (s === "success") return "Succeeded";
  if (s === "failed") return "Failed";
  if (s === "partial") return "Partial";
  if (s === "completed") return "Completed";
  return status || "Unknown";
}

const STEP_NAMES: Record<string, string> = {
  vector_search_hybrid_retrieval: "Find relevant passages",
  retrieval: "Find relevant passages",
  vector_search: "Find relevant passages",
  domain_detection: "Detect the document specialty",
  prompt_assembly: "Assemble the prompt",
  llm_call: "Ask the model",
  response_validation: "Check the answer",
  validation: "Check the answer",
  save_message: "Save the answer",
  guardrail_check: "Safety check",
  context_build: "Prepare document context",
};

function humanStepName(name: string) {
  if (!name) return "Step";
  return STEP_NAMES[name] || name.replace(/_/g, " ");
}

function CopyValue({ value, label }: { value: string; label: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="adm-link"
      style={{ fontFamily: "var(--adm-mono)", fontSize: 11.5, fontWeight: 400 }}
      title={`Copy ${label}`}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1200);
        } catch {
          /* ignore */
        }
      }}
    >
      {copied ? <Check style={{ width: 11, height: 11 }} /> : <Copy style={{ width: 11, height: 11 }} />}
      {value}
    </button>
  );
}

function PreBlock({ children }: { children: React.ReactNode }) {
  return <pre className="adm-pre">{children}</pre>;
}

function DetailsBlock({
  summary,
  children,
  defaultOpen = false,
}: {
  summary: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details className="adm-details" open={defaultOpen}>
      <summary>{summary}</summary>
      <div className="adm-details__body">{children}</div>
    </details>
  );
}

function ChunkCard({ chunk, index }: { chunk: Record<string, unknown>; index: number }) {
  const title = [
    `#${index + 1}`,
    chunk.filename ? String(chunk.filename) : null,
    chunk.page != null ? `page ${chunk.page}` : null,
    chunk.score != null ? `relevance ${Number(chunk.score).toFixed(3)}` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <details className="adm-chunk">
      <summary>
        <ChevronDown style={{ width: 13, height: 13, color: "var(--adm-faint)" }} />
        {title || `Passage ${index + 1}`}
      </summary>
      <div className="adm-chunk__body">
        {chunk.text ? (
          <PreBlock>{String(chunk.text)}</PreBlock>
        ) : (
          <p className="adm-hint">No text preview was stored for this passage.</p>
        )}
        <div className="adm-hint" style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
          <span>
            ID <span className="adm-inline-code">{String(chunk.id || "—")}</span>
          </span>
          {chunk.char_count != null ? <span>{String(chunk.char_count)} characters</span> : null}
          {chunk.section ? <span>Section {String(chunk.section)}</span> : null}
        </div>
        <DetailsBlock summary="Raw passage data">
          <PreBlock>{prettyJson(chunk)}</PreBlock>
        </DetailsBlock>
      </div>
    </details>
  );
}

export function AiTraceDetail({ detail }: { detail: AITraceDetail }) {
  const userLabel = detail.username || detail.email || detail.user_id?.slice(0, 8) || "—";
  const promptAssembly = (detail.metadata?.prompt_assembly || null) as Record<string, unknown> | null;

  const chunksToLlm = useMemo(() => {
    const fromCalls = detail.llm_calls.flatMap((c) => c.chunks_sent_to_llm || []);
    if (fromCalls.length) return fromCalls;
    const fromMeta = (promptAssembly?.chunks_sent_to_llm || []) as Array<Record<string, unknown>>;
    if (fromMeta.length) return fromMeta;
    return detail.retrievals.flatMap((r) => r.chunks || []);
  }, [detail, promptAssembly]);

  const tokenCostTotal = useMemo(() => {
    const fromMeta = detail.metadata?.estimated_cost_usd_total;
    if (typeof fromMeta === "number") return fromMeta;
    const sum = detail.llm_calls.reduce(
      (acc, call) => acc + (typeof call.estimated_cost_usd === "number" ? call.estimated_cost_usd : 0),
      0,
    );
    return detail.llm_calls.some((c) => c.estimated_cost_usd != null) ? sum : null;
  }, [detail]);

  const pipelineSteps = useMemo(() => {
    if (detail.steps.length > 0) return detail.steps;
    const synthetic: AITraceStep[] = [];
    detail.retrievals.forEach((ret, idx) => {
      synthetic.push({
        id: `synth-ret-${ret.id || idx}`,
        sequence: synthetic.length + 1,
        name: `Search documents (${ret.search_method || "hybrid"})`,
        status: "success",
        duration_ms: ret.duration_ms,
        output_summary: prettyJson({
          result_count: ret.result_count ?? (ret.chunks || []).length,
          documents: ret.document_names || [],
        }),
      });
    });
    detail.llm_calls.forEach((call, idx) => {
      synthetic.push({
        id: `synth-llm-${call.id || idx}`,
        sequence: synthetic.length + 1,
        name: `Ask the model (${call.provider || "?"}/${call.model_name || "?"})`,
        status: call.status || "success",
        duration_ms: call.latency_ms,
        output_summary: prettyJson({
          tokens: {
            input: call.input_tokens,
            output: call.output_tokens,
            cached: call.cached_tokens,
            total: call.total_tokens,
          },
          estimated_cost_usd: call.estimated_cost_usd,
          validation: call.validation_result,
        }),
        error_message: call.error_message,
      });
    });
    if (detail.final_answer) {
      synthetic.push({
        id: "synth-answer",
        sequence: synthetic.length + 1,
        name: "Write the final answer",
        status: detail.final_answer_status || detail.status || "success",
        output_summary: detail.final_answer.slice(0, 2000),
      });
    }
    return synthetic;
  }, [detail]);

  return (
    <div className="adm-stack">
      {detail.error_summary ? (
        <Callout tone="danger" icon={<AlertTriangle />}>
          <strong>This answer failed.</strong> {detail.error_summary}
        </Callout>
      ) : null}

      <Card title="The question">
        <div className="adm-card__body--pad" style={{ display: "grid", gap: 14 }}>
          <div style={{ fontSize: 14, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
            {detail.question || "No question recorded."}
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            <Badge tone={toneFor(detail.status)}>{statusLabel(detail.status)}</Badge>
            <Badge plain>{humanRequestType(detail.request_type)}</Badge>
            <Badge plain>{formatWhen(detail.created_at)}</Badge>
            {detail.early_exit ? <Badge tone="warn">Stopped early</Badge> : null}
          </div>
          {detail.early_exit ? (
            <p className="adm-hint">
              The pipeline stopped before a full answer: {detail.early_exit_reason || "no reason recorded"}.
            </p>
          ) : null}
        </div>
      </Card>

      <Card title="At a glance" description="Everything recorded about this request, in one place.">
        <div className="adm-card__body--pad">
          <DefinitionList
            items={[
              { term: "Person", value: userLabel },
              { term: "Duration", value: formatMs(detail.duration_ms) },
              {
                term: "Model",
                value: detail.model_name ? `${detail.model_name}${detail.provider ? ` · ${detail.provider}` : ""}` : "—",
              },
              {
                term: "Tokens",
                value: `${(detail.total_tokens ?? 0).toLocaleString()} total · ${(detail.input_tokens ?? 0).toLocaleString()} in · ${(detail.output_tokens ?? 0).toLocaleString()} out`,
              },
              { term: "Estimated cost", value: formatCost(tokenCostTotal) },
              { term: "Passages used", value: String(detail.retrieved_chunk_count ?? chunksToLlm.length) },
              { term: "Trace ID", value: <CopyValue value={detail.trace_id} label="trace ID" /> },
              {
                term: "Conversation",
                value: detail.session_id ? (
                  <CopyValue value={detail.session_id} label="conversation ID" />
                ) : (
                  "—"
                ),
              },
              {
                term: "Answer message",
                value: detail.message_id ? (
                  <CopyValue value={detail.message_id} label="message ID" />
                ) : (
                  "—"
                ),
              },
            ]}
          />
        </div>
      </Card>

      <Card
        title={`How the answer was produced (${pipelineSteps.length} steps)`}
        description="Each step in order, with how long it took and whether it succeeded."
      >
        {pipelineSteps.length === 0 ? (
          <EmptyState
            icon={<Sparkles />}
            title="No processing steps recorded"
            text="Older traces may not include step-level detail."
          />
        ) : (
          <div className="adm-steps">
            {pipelineSteps.map((step, idx) => {
              const payload = stepDetailsPayload(step);
              return (
                <div className="adm-step" key={step.id} data-tone={toneFor(step.status)}>
                  <span className="adm-step__index">{idx + 1}</span>
                  <div className="adm-step__body">
                    <div className="adm-step__title">
                      {humanStepName(step.name)}
                      <Badge tone={toneFor(step.status)}>{statusLabel(step.status)}</Badge>
                    </div>
                    <div className="adm-step__meta">{formatMs(step.duration_ms)}</div>
                    {step.error_message ? (
                      <div className="adm-step__meta" style={{ color: "var(--adm-danger)" }}>
                        {step.error_message}
                      </div>
                    ) : null}
                    {Object.keys(payload).length > 0 ? (
                      <DetailsBlock summary="Step detail">
                        <PreBlock>{prettyJson(payload)}</PreBlock>
                      </DetailsBlock>
                    ) : null}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Card>

      <Card
        title={`Passages used to answer (${chunksToLlm.length})`}
        description="These document excerpts were given to the model as evidence."
      >
        {chunksToLlm.length === 0 ? (
          <EmptyState
            title="No passages stored"
            text="This trace did not record the excerpt text sent to the model."
          />
        ) : (
          <div className="adm-card__body--pad adm-stack" style={{ gap: 8 }}>
            {chunksToLlm.map((chunk, idx) => (
              <ChunkCard key={String(chunk.id || idx)} chunk={chunk} index={idx} />
            ))}
          </div>
        )}
      </Card>

      <Card title="Model calls" description="Tokens, cost, and latency for every model request.">
        {detail.llm_calls.length === 0 ? (
          <EmptyState title="No model calls recorded" />
        ) : (
          <>
            <table className="adm-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th data-align="right">In</th>
                  <th data-align="right" className="adm-col-sm">Out</th>
                  <th data-align="right" className="adm-col-md">Total</th>
                  <th data-align="right">Cost</th>
                  <th data-align="right" className="adm-col-sm">Time</th>
                </tr>
              </thead>
              <tbody>
                {detail.llm_calls.map((call) => (
                  <tr key={call.id}>
                    <td>
                      <div className="adm-cell-stack">
                        <span className="adm-cell-main">{call.model_name || "Unknown model"}</span>
                        <span className="adm-cell-sub">{call.provider || "—"}</span>
                      </div>
                    </td>
                    <td data-align="right" className="adm-num">{(call.input_tokens ?? 0).toLocaleString()}</td>
                    <td data-align="right" className="adm-num adm-col-sm">{(call.output_tokens ?? 0).toLocaleString()}</td>
                    <td data-align="right" className="adm-num adm-col-md">{(call.total_tokens ?? 0).toLocaleString()}</td>
                    <td data-align="right" className="adm-num">{formatCost(call.estimated_cost_usd)}</td>
                    <td data-align="right" className="adm-num adm-col-sm">{formatMs(call.latency_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="adm-card__body--pad adm-stack" style={{ gap: 8 }}>
              {detail.llm_calls.map((call) => (
                <DetailsBlock
                  key={`call-${call.id}`}
                  summary={`${call.provider || "?"} / ${call.model_name || "?"} — prompts and response`}
                >
                  {call.error_message ? (
                    <Callout tone="danger" icon={<AlertTriangle />}>{call.error_message}</Callout>
                  ) : null}
                  {call.system_prompt ? (
                    <>
                      <strong style={{ fontSize: 12 }}>Instructions sent</strong>
                      <PreBlock>{call.system_prompt}</PreBlock>
                    </>
                  ) : null}
                  {call.context_text ? (
                    <>
                      <strong style={{ fontSize: 12 }}>Document context</strong>
                      <PreBlock>{call.context_text}</PreBlock>
                    </>
                  ) : null}
                  {call.user_prompt ? (
                    <>
                      <strong style={{ fontSize: 12 }}>Full prompt</strong>
                      <PreBlock>{call.user_prompt}</PreBlock>
                    </>
                  ) : null}
                  {call.raw_response ? (
                    <>
                      <strong style={{ fontSize: 12 }}>Model response</strong>
                      <PreBlock>{call.raw_response}</PreBlock>
                    </>
                  ) : null}
                  {call.pricing_note ? (
                    <p className="adm-hint">{call.pricing_note}</p>
                  ) : null}
                </DetailsBlock>
              ))}
            </div>
          </>
        )}
      </Card>

      <Card title={`Searches (${detail.retrievals.length})`} description="How the platform looked for relevant passages.">
        {detail.retrievals.length === 0 ? (
          <EmptyState title="No searches recorded" />
        ) : (
          <div className="adm-card__body--pad adm-stack" style={{ gap: 8 }}>
            {detail.retrievals.map((ret) => (
              <DetailsBlock
                key={ret.id}
                summary={`${ret.search_method || "Search"} · ${ret.result_count ?? (ret.chunks || []).length} matches · ${formatMs(ret.duration_ms)}`}
              >
                {ret.search_query ? (
                  <>
                    <strong style={{ fontSize: 12 }}>Query</strong>
                    <PreBlock>{ret.search_query}</PreBlock>
                  </>
                ) : null}
                {ret.document_names && (ret.document_names as string[]).length ? (
                  <p className="adm-hint">
                    <strong>Documents:</strong> {(ret.document_names as string[]).join(", ")}
                  </p>
                ) : null}
                {(ret.chunks || []).map((chunk, idx) => (
                  <ChunkCard key={`${ret.id}-${String(chunk.id || idx)}`} chunk={chunk} index={idx} />
                ))}
                {ret.filters ? (
                  <DetailsBlock summary="Search filters">
                    <PreBlock>{prettyJson(ret.filters)}</PreBlock>
                  </DetailsBlock>
                ) : null}
                {ret.rejected ? (
                  <DetailsBlock summary="Rejected matches">
                    <PreBlock>{prettyJson(ret.rejected)}</PreBlock>
                  </DetailsBlock>
                ) : null}
              </DetailsBlock>
            ))}
          </div>
        )}
      </Card>

      <Card
        title="Prompt assembly"
        description="How passages and the question were combined before asking the model."
      >
        {promptAssembly ? (
          <div className="adm-card__body--pad adm-stack" style={{ gap: 10 }}>
            <p className="adm-hint">
              {String(
                promptAssembly.assembly || "Passages become numbered context blocks inside the prompt.",
              )}
            </p>
            <DefinitionList
              items={[
                { term: "Passages", value: String(promptAssembly.chunk_count ?? chunksToLlm.length) },
                { term: "Context size", value: `${String(promptAssembly.context_char_count ?? "—")} characters` },
                {
                  term: "Prompt size",
                  value: `${String(promptAssembly.user_prompt_char_count ?? "—")} characters`,
                },
              ]}
            />
            {promptAssembly.system_prompt || promptAssembly.system_prompt_preview ? (
              <DetailsBlock summary="Instructions sent to the model">
                <PreBlock>
                  {String(promptAssembly.system_prompt || promptAssembly.system_prompt_preview)}
                </PreBlock>
              </DetailsBlock>
            ) : null}
            {promptAssembly.context_text || promptAssembly.context_preview ? (
              <DetailsBlock summary="Assembled document context">
                <PreBlock>
                  {String(promptAssembly.context_text || promptAssembly.context_preview)}
                </PreBlock>
              </DetailsBlock>
            ) : null}
            {promptAssembly.user_prompt || promptAssembly.user_prompt_preview ? (
              <DetailsBlock summary="Final prompt (context + question)">
                <PreBlock>
                  {String(promptAssembly.user_prompt || promptAssembly.user_prompt_preview)}
                </PreBlock>
              </DetailsBlock>
            ) : null}
          </div>
        ) : (
          <EmptyState
            title="Assembly detail not recorded"
            text="Newer answers store the exact instructions, context, and prompt used."
          />
        )}
      </Card>

      <Card
        title="Final answer"
        description={`Status: ${statusLabel(detail.final_answer_status || detail.status)}`}
      >
        <div className="adm-card__body--pad">
          <div style={{ fontSize: 13.5, lineHeight: 1.65, whiteSpace: "pre-wrap" }}>
            {detail.final_answer || "No final answer was recorded."}
          </div>
          {detail.feedback && detail.feedback.length > 0 ? (
            <>
              <hr className="adm-hr" style={{ margin: "14px 0" }} />
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
                <MessageSquare style={{ width: 14, height: 14, color: "var(--adm-muted)" }} />
                <strong style={{ fontSize: 12.5 }}>Feedback on this answer</strong>
              </div>
              <div className="adm-stack" style={{ gap: 8 }}>
                {detail.feedback.map((fb, i) => (
                  <div key={i} style={{ fontSize: 13 }}>
                    <Badge tone={fb.feedback === "up" ? "ok" : "danger"}>
                      {fb.feedback === "up" ? "Liked" : "Disliked"}
                    </Badge>
                    {fb.reason ? <span className="adm-muted"> · {fb.reason.replace(/_/g, " ")}</span> : null}
                    {fb.comment ? <div className="adm-hint">{fb.comment}</div> : null}
                  </div>
                ))}
              </div>
            </>
          ) : null}
        </div>
      </Card>

      {detail.duration_breakdown && detail.duration_breakdown.length > 0 ? (
        <Card title="Time breakdown">
          <div className="adm-card__body--pad adm-stack" style={{ gap: 10 }}>
            {detail.duration_breakdown.map((part) => (
              <div key={part.name} style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}>
                <span>{part.name}</span>
                <span className="adm-muted">{formatMs(part.duration_ms)}</span>
              </div>
            ))}
          </div>
        </Card>
      ) : null}

      {detail.metadata && Object.keys(detail.metadata).length > 0 ? (
        <Card title="Raw metadata" description="Advanced detail for engineering and support.">
          <div className="adm-card__body--pad">
            <PreBlock>{prettyJson(detail.metadata)}</PreBlock>
          </div>
        </Card>
      ) : null}
    </div>
  );
}

"use client";

import { useMemo, useState } from "react";
import { Check, Copy } from "lucide-react";

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
  temperature?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  cached_tokens?: number | null;
  total_tokens?: number | null;
  estimated_cost_usd?: number | null;
  cost_type?: string | null;
  pricing_note?: string | null;
  latency_ms?: number | null;
  retry_count?: number | null;
  validation_result?: string | null;
  status: string;
  error_message?: string | null;
  system_prompt?: string | null;
  user_prompt?: string | null;
  context_text?: string | null;
  raw_response?: string | null;
  chunks_sent_to_llm?: Array<Record<string, unknown>> | null;
  metadata?: unknown;
}

export interface AITraceDetail {
  trace_id: string;
  user_id?: string | null;
  username?: string | null;
  email?: string | null;
  session_id?: string | null;
  message_id?: string | null;
  document_id?: string | null;
  question: string;
  request_type: string;
  status: string;
  early_exit: boolean;
  early_exit_reason?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  created_at?: string | null;
  duration_ms?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  cached_tokens?: number | null;
  total_tokens?: number | null;
  model_name?: string | null;
  provider?: string | null;
  retrieved_chunk_count?: number | null;
  final_answer?: string | null;
  final_answer_status?: string | null;
  error_summary?: string | null;
  warnings?: unknown[];
  metadata?: Record<string, unknown>;
  steps: AITraceStep[];
  retrievals: AITraceRetrieval[];
  llm_calls: AITraceLLMCall[];
  feedback?: Array<{
    feedback?: string | null;
    reason?: string | null;
    comment?: string | null;
  }>;
  duration_breakdown?: Array<{
    name: string;
    duration_ms?: number | null;
    status?: string;
  }>;
}

function formatWhen(iso?: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toISOString();
  } catch {
    return iso;
  }
}

function formatMs(ms?: number | null): string {
  if (ms == null || Number.isNaN(ms)) return "—";
  return String(Math.round(ms));
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

function CopyId({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="inline-flex items-center gap-1 font-mono text-[12px] text-[var(--gd-muted)] hover:text-[var(--gd-ink)]"
      title="Copy Trace ID"
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
      {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
      {value}
    </button>
  );
}

function Card({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-[12px] border border-[var(--gd-border)] bg-[var(--gd-surface)] p-4 shadow-sm">
      <h2 className="mb-3 text-[15px] font-semibold tracking-tight">{title}</h2>
      {children}
    </section>
  );
}

function PreBlock({ children }: { children: React.ReactNode }) {
  return (
    <pre className="max-h-[28rem] overflow-auto whitespace-pre-wrap rounded-[8px] border border-[var(--gd-border)] bg-[var(--gd-ground)]/50 p-3 font-mono text-[11px] leading-relaxed text-[var(--gd-ink)]">
      {children}
    </pre>
  );
}

function DetailsBlock({
  summary,
  children,
}: {
  summary: string;
  children: React.ReactNode;
}) {
  return (
    <details className="mt-2 rounded-[8px] border border-[var(--gd-border)] bg-[var(--gd-ground)]/20">
      <summary className="cursor-pointer px-3 py-2 text-[13px] font-medium">
        {summary}
      </summary>
      <div className="border-t border-[var(--gd-border)] p-3">{children}</div>
    </details>
  );
}

function formatUsd(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return "—";
  if (value === 0) return "$0.00";
  if (Math.abs(value) < 0.0001) return `$${value.toFixed(8)}`;
  return `$${value.toFixed(6)}`;
}

function ChunkCard({
  chunk,
  index,
}: {
  chunk: Record<string, unknown>;
  index: number;
}) {
  const title = [
    `#${index + 1}`,
    chunk.filename ? String(chunk.filename) : null,
    chunk.page != null ? `p.${chunk.page}` : null,
    chunk.score != null ? `score=${chunk.score}` : null,
    chunk.confidence != null ? `conf=${chunk.confidence}` : null,
    chunk.chunk_type ? String(chunk.chunk_type) : null,
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <DetailsBlock summary={title || `Chunk ${index + 1}`}>
      <div className="mb-2 grid gap-1 text-[12px] sm:grid-cols-2">
        <div>
          <strong>id:</strong>{" "}
          <span className="font-mono">{String(chunk.id || "—")}</span>
        </div>
        <div>
          <strong>chars:</strong> {String(chunk.char_count ?? "—")}
        </div>
        <div>
          <strong>section:</strong> {String(chunk.section || "—")}
        </div>
        <div>
          <strong>table_id:</strong> {String(chunk.table_id || "—")}
        </div>
      </div>
      {chunk.text ? (
        <>
          <p className="mb-1 text-[12px] font-medium">Chunk text</p>
          <PreBlock>{String(chunk.text)}</PreBlock>
        </>
      ) : (
        <p className="text-[12px] text-[var(--gd-muted)]">
          No text preview stored for this chunk.
        </p>
      )}
      <DetailsBlock summary="Raw chunk JSON">
        <PreBlock>{prettyJson(chunk)}</PreBlock>
      </DetailsBlock>
    </DetailsBlock>
  );
}

export function AiTraceDetail({ detail }: { detail: AITraceDetail }) {
  const userLabel =
    detail.username || detail.email || detail.user_id?.slice(0, 8) || "—";

  const promptAssembly = (detail.metadata?.prompt_assembly || null) as Record<
    string,
    unknown
  > | null;

  const chunksToLlm = useMemo(() => {
    const fromCalls = detail.llm_calls.flatMap(
      (c) => c.chunks_sent_to_llm || [],
    );
    if (fromCalls.length) return fromCalls;
    const fromMeta = (promptAssembly?.chunks_sent_to_llm || []) as Array<
      Record<string, unknown>
    >;
    if (fromMeta.length) return fromMeta;
    return detail.retrievals.flatMap((r) => r.chunks || []);
  }, [detail, promptAssembly]);

  const tokenCostTotal = useMemo(() => {
    const fromMeta = detail.metadata?.estimated_cost_usd_total;
    if (typeof fromMeta === "number") return fromMeta;
    const sum = detail.llm_calls.reduce((acc, call) => {
      return acc + (typeof call.estimated_cost_usd === "number" ? call.estimated_cost_usd : 0);
    }, 0);
    return detail.llm_calls.some((c) => c.estimated_cost_usd != null) ? sum : null;
  }, [detail]);

  const pipelineSteps = useMemo(() => {
    if (detail.steps.length > 0) return detail.steps;
    const synthetic: AITraceStep[] = [];
    detail.retrievals.forEach((ret, idx) => {
      synthetic.push({
        id: `synth-ret-${ret.id || idx}`,
        sequence: synthetic.length + 1,
        name: `Vector search (${ret.search_method || "hybrid"})`,
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
        name: `LLM generation (${call.provider || "?"}/${call.model_name || "?"})`,
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
        name: "Final answer",
        status: detail.final_answer_status || detail.status || "success",
        output_summary: detail.final_answer.slice(0, 2000),
      });
    }
    return synthetic;
  }, [detail]);

  return (
    <div className="trace-page space-y-4">
      <Card title="Trace Summary">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <div>
            <strong>Trace ID:</strong> <CopyId value={detail.trace_id} />
          </div>
          <div>
            <strong>Type:</strong> {detail.request_type}
          </div>
          <div>
            <strong>Status:</strong> {detail.status}
          </div>
          <div>
            <strong>Created:</strong> {formatWhen(detail.created_at)}
          </div>
          <div>
            <strong>Duration (ms):</strong> {formatMs(detail.duration_ms)}
          </div>
          <div>
            <strong>Early exit:</strong>{" "}
            {detail.early_exit ? detail.early_exit_reason || "Yes" : "No"}
          </div>
          <div>
            <strong>Total tokens:</strong> {detail.total_tokens ?? "—"}
          </div>
          <div>
            <strong>Input tokens:</strong> {detail.input_tokens ?? "—"}
          </div>
          <div>
            <strong>Output tokens:</strong> {detail.output_tokens ?? "—"}
          </div>
          <div>
            <strong>Cached tokens:</strong> {detail.cached_tokens ?? "—"}
          </div>
          <div>
            <strong>Est. cost (total):</strong> {formatUsd(tokenCostTotal)}
          </div>
          <div>
            <strong>Model:</strong> {detail.model_name || "—"}
            {detail.provider ? ` (${detail.provider})` : ""}
          </div>
          <div>
            <strong>Retrieved chunks:</strong>{" "}
            {detail.retrieved_chunk_count ?? "—"}
          </div>
          <div>
            <strong>User:</strong> {userLabel}
          </div>
          <div>
            <strong>Session:</strong>{" "}
            <span className="font-mono text-[12px]">
              {detail.session_id || "—"}
            </span>
          </div>
          <div>
            <strong>Message:</strong>{" "}
            <span className="font-mono text-[12px]">
              {detail.message_id || "—"}
            </span>
          </div>
        </div>
        <div className="mt-4">
          <strong>Question</strong>
          <PreBlock>{detail.question || "—"}</PreBlock>
        </div>
        {detail.error_summary ? (
          <p className="mt-3 text-[13px] text-[var(--gd-negative)]">
            {detail.error_summary}
          </p>
        ) : null}
      </Card>

      <Card title={`Processing Steps (${pipelineSteps.length})`}>
        {pipelineSteps.length === 0 ? (
          <p className="text-[13px] text-[var(--gd-muted)]">
            No processing steps were recorded for this trace.
          </p>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="trace-table w-full min-w-[520px] border-collapse text-left text-[13px]">
                <thead>
                  <tr className="border-b border-[var(--gd-border)] text-[12px] text-[var(--gd-muted)]">
                    <th className="py-2 pr-3 font-medium">Step</th>
                    <th className="py-2 pr-3 font-medium">Duration (ms)</th>
                    <th className="py-2 font-medium">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {pipelineSteps.map((step) => (
                    <tr
                      key={step.id}
                      className="border-b border-[var(--gd-border)]/70"
                    >
                      <td className="py-2 pr-3">{step.name}</td>
                      <td className="py-2 pr-3 font-mono">
                        {formatMs(step.duration_ms)}
                      </td>
                      <td className="py-2">{step.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {pipelineSteps.map((step) => {
              const payload = stepDetailsPayload(step);
              if (Object.keys(payload).length === 0) return null;
              return (
                <DetailsBlock
                  key={`details-${step.id}`}
                  summary={`${step.name} — details`}
                >
                  <PreBlock>{prettyJson(payload)}</PreBlock>
                </DetailsBlock>
              );
            })}
          </>
        )}
      </Card>

      <Card title="Token & cost by model">
        {detail.llm_calls.length === 0 ? (
          <p className="text-[13px] text-[var(--gd-muted)]">
            No LLM usage recorded yet for this trace.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="trace-table w-full min-w-[720px] border-collapse text-left text-[13px]">
              <thead>
                <tr className="border-b border-[var(--gd-border)] text-[12px] text-[var(--gd-muted)]">
                  <th className="py-2 pr-3 font-medium">Provider / model</th>
                  <th className="py-2 pr-3 font-medium">In</th>
                  <th className="py-2 pr-3 font-medium">Out</th>
                  <th className="py-2 pr-3 font-medium">Cached</th>
                  <th className="py-2 pr-3 font-medium">Total</th>
                  <th className="py-2 pr-3 font-medium">Est. USD</th>
                  <th className="py-2 pr-3 font-medium">Latency</th>
                  <th className="py-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {detail.llm_calls.map((call) => (
                  <tr
                    key={`cost-${call.id}`}
                    className="border-b border-[var(--gd-border)]/70"
                  >
                    <td className="py-2 pr-3">
                      {call.provider || "—"} / {call.model_name || "—"}
                    </td>
                    <td className="py-2 pr-3 font-mono">
                      {call.input_tokens ?? "—"}
                    </td>
                    <td className="py-2 pr-3 font-mono">
                      {call.output_tokens ?? "—"}
                    </td>
                    <td className="py-2 pr-3 font-mono">
                      {call.cached_tokens ?? "—"}
                    </td>
                    <td className="py-2 pr-3 font-mono">
                      {call.total_tokens ?? "—"}
                    </td>
                    <td className="py-2 pr-3 font-mono">
                      {formatUsd(call.estimated_cost_usd)}
                    </td>
                    <td className="py-2 pr-3 font-mono">
                      {formatMs(call.latency_ms)} ms
                    </td>
                    <td className="py-2">{call.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {detail.llm_calls.map((call) =>
              call.pricing_note ? (
                <p
                  key={`note-${call.id}`}
                  className="mt-2 text-[12px] text-[var(--gd-muted)]"
                >
                  {call.provider}/{call.model_name}: {call.pricing_note}
                  {call.cost_type ? ` (${call.cost_type})` : ""}
                </p>
              ) : null,
            )}
          </div>
        )}
      </Card>

      <Card title={`Chunks sent to the LLM (${chunksToLlm.length})`}>
        <p className="mb-2 text-[13px] text-[var(--gd-muted)]">
          These are the retrieved chunk cards that were formatted into the RAG{" "}
          <code>context</code> block and injected into the user prompt.
        </p>
        {chunksToLlm.length === 0 ? (
          <p className="text-[13px] text-[var(--gd-muted)]">
            No chunk payloads were stored. Ask a new question with the updated
            server to capture per-chunk text.
          </p>
        ) : (
          chunksToLlm.map((chunk, idx) => (
            <ChunkCard
              key={String(chunk.id || idx)}
              chunk={chunk}
              index={idx}
            />
          ))
        )}
      </Card>

      <Card title="How chunks become the prompt">
        {promptAssembly ? (
          <>
            <p className="mb-2 text-[13px]">
              {String(
                promptAssembly.assembly ||
                  "Chunks → numbered context blocks → RAG user prompt template.",
              )}
            </p>
            <div className="mb-3 grid gap-2 text-[13px] sm:grid-cols-3">
              <div>
                <strong>Chunks:</strong>{" "}
                {String(promptAssembly.chunk_count ?? chunksToLlm.length)}
              </div>
              <div>
                <strong>Context chars:</strong>{" "}
                {String(promptAssembly.context_char_count ?? "—")}
              </div>
              <div>
                <strong>User prompt chars:</strong>{" "}
                {String(promptAssembly.user_prompt_char_count ?? "—")}
              </div>
            </div>
            {promptAssembly.system_prompt || promptAssembly.system_prompt_preview ? (
              <DetailsBlock summary="System prompt (sent to model)">
                <PreBlock>
                  {String(
                    promptAssembly.system_prompt ||
                      promptAssembly.system_prompt_preview,
                  )}
                </PreBlock>
              </DetailsBlock>
            ) : null}
            {promptAssembly.context_text || promptAssembly.context_preview ? (
              <DetailsBlock summary="Assembled context (from chunks)">
                <PreBlock>
                  {String(
                    promptAssembly.context_text ||
                      promptAssembly.context_preview,
                  )}
                </PreBlock>
              </DetailsBlock>
            ) : null}
            {promptAssembly.user_prompt || promptAssembly.user_prompt_preview ? (
              <DetailsBlock summary="Final user prompt (context + question)">
                <PreBlock>
                  {String(
                    promptAssembly.user_prompt ||
                      promptAssembly.user_prompt_preview,
                  )}
                </PreBlock>
              </DetailsBlock>
            ) : null}
          </>
        ) : (
          <p className="text-[13px] text-[var(--gd-muted)]">
            Prompt assembly detail was not recorded for this older trace. New
            questions store the system prompt, assembled context, and final user
            prompt here.
          </p>
        )}
      </Card>

      <Card title={`Vector Searches (${detail.retrievals.length})`}>
        {detail.retrievals.length === 0 ? (
          <p className="text-[13px] text-[var(--gd-muted)]">
            No vector searches were recorded.
          </p>
        ) : (
          detail.retrievals.map((ret) => (
            <DetailsBlock
              key={ret.id}
              summary={`${ret.search_method || "retrieval"} — ${ret.result_count ?? (ret.chunks || []).length} hits — ${formatMs(ret.duration_ms)}ms`}
            >
              {ret.search_query ? (
                <>
                  <p className="mb-1 text-[12px] font-medium">Query:</p>
                  <PreBlock>{ret.search_query}</PreBlock>
                </>
              ) : null}
              {ret.document_names && (ret.document_names as string[]).length ? (
                <p className="mt-2 text-[12px]">
                  <strong>Documents:</strong>{" "}
                  {(ret.document_names as string[]).join(", ")}
                </p>
              ) : null}
              {ret.filters ? (
                <>
                  <p className="mb-1 mt-2 text-[12px] font-medium">Filters:</p>
                  <PreBlock>{prettyJson(ret.filters)}</PreBlock>
                </>
              ) : null}
              <p className="mb-1 mt-2 text-[12px] font-medium">
                Retrieved chunks ({(ret.chunks || []).length})
              </p>
              {(ret.chunks || []).map((chunk, idx) => (
                <ChunkCard
                  key={`${ret.id}-${String(chunk.id || idx)}`}
                  chunk={chunk}
                  index={idx}
                />
              ))}
              {ret.rejected ? (
                <>
                  <p className="mb-1 mt-2 text-[12px] font-medium">Rejected:</p>
                  <PreBlock>{prettyJson(ret.rejected)}</PreBlock>
                </>
              ) : null}
            </DetailsBlock>
          ))
        )}
      </Card>

      <Card title={`LLM Calls (${detail.llm_calls.length})`}>
        {detail.llm_calls.length === 0 ? (
          <p className="text-[13px] text-[var(--gd-muted)]">
            No LLM calls were recorded. Restart the API and ask a new question —
            the WebSocket stream path now persists LLM rows with prompts, chunks,
            tokens, and cost.
          </p>
        ) : (
          detail.llm_calls.map((call) => {
            const title = [
              call.system_prompt_version || "llm",
              call.model_name || call.provider || "model",
              `in:${call.input_tokens ?? "—"}`,
              `out:${call.output_tokens ?? "—"}`,
              `cost:${formatUsd(call.estimated_cost_usd)}`,
              `${formatMs(call.latency_ms)}ms`,
            ].join(" — ");
            return (
              <DetailsBlock key={call.id} summary={title}>
                {call.error_message ? (
                  <p className="mb-2 text-[12px] text-[var(--gd-negative)]">
                    {call.error_message}
                  </p>
                ) : null}
                <div className="mb-2 grid gap-1 text-[12px] sm:grid-cols-2">
                  <div>
                    <strong>Tokens in/out/cached/total:</strong>{" "}
                    {call.input_tokens ?? "—"} / {call.output_tokens ?? "—"} /{" "}
                    {call.cached_tokens ?? "—"} / {call.total_tokens ?? "—"}
                  </div>
                  <div>
                    <strong>Est. cost:</strong> {formatUsd(call.estimated_cost_usd)}{" "}
                    {call.cost_type ? `(${call.cost_type})` : ""}
                  </div>
                </div>
                {call.system_prompt ? (
                  <>
                    <p className="mb-1 text-[12px] font-medium">System prompt:</p>
                    <PreBlock>{call.system_prompt}</PreBlock>
                  </>
                ) : null}
                {call.context_text ? (
                  <>
                    <p className="mb-1 mt-2 text-[12px] font-medium">
                      Context block (from chunks):
                    </p>
                    <PreBlock>{call.context_text}</PreBlock>
                  </>
                ) : null}
                {call.user_prompt ? (
                  <>
                    <p className="mb-1 mt-2 text-[12px] font-medium">
                      User prompt (dynamic assembly):
                    </p>
                    <PreBlock>{call.user_prompt}</PreBlock>
                  </>
                ) : null}
                {call.raw_response ? (
                  <>
                    <p className="mb-1 mt-2 text-[12px] font-medium">Response:</p>
                    <PreBlock>{call.raw_response}</PreBlock>
                  </>
                ) : null}
                {(call.chunks_sent_to_llm || []).length ? (
                  <>
                    <p className="mb-1 mt-2 text-[12px] font-medium">
                      Chunks attached to this call (
                      {call.chunks_sent_to_llm!.length})
                    </p>
                    {call.chunks_sent_to_llm!.map((chunk, idx) => (
                      <ChunkCard
                        key={`${call.id}-c-${idx}`}
                        chunk={chunk}
                        index={idx}
                      />
                    ))}
                  </>
                ) : null}
              </DetailsBlock>
            );
          })
        )}
      </Card>

      <Card title="Final Answer">
        <p className="mb-2 text-[12px] text-[var(--gd-muted)]">
          Status: {detail.final_answer_status || detail.status}
        </p>
        <PreBlock>{detail.final_answer || "—"}</PreBlock>
        {detail.feedback && detail.feedback.length > 0 ? (
          <div className="mt-3 space-y-1 text-[13px]">
            <strong>User feedback</strong>
            {detail.feedback.map((fb, i) => (
              <p key={i}>
                {fb.feedback}
                {fb.reason ? ` · ${fb.reason}` : ""}
                {fb.comment ? ` — ${fb.comment}` : ""}
              </p>
            ))}
          </div>
        ) : null}
      </Card>

      {detail.metadata && Object.keys(detail.metadata).length > 0 ? (
        <Card title="Trace Metadata">
          <DetailsBlock summary="metadata JSON">
            <PreBlock>{prettyJson(detail.metadata)}</PreBlock>
          </DetailsBlock>
        </Card>
      ) : null}
    </div>
  );
}

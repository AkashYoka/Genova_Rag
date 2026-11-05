"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";

import { api, CONNECTION_ERROR_MESSAGE } from "@/lib/api";
import { formatUsd } from "@/lib/admin";
import {
  LedgerBarChart,
  LedgerButton,
  LedgerEmpty,
  LedgerError,
  LedgerKpiStrip,
  LedgerPanel,
  LedgerScrollTable,
  LedgerStatus,
} from "@/components/ledger/LedgerPrimitives";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";

export interface UsageSummary {
  start: string;
  end: string;
  request_count: number;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number;
  today: {
    request_count: number;
    total_tokens: number;
    estimated_cost_usd: number;
  };
  month_to_date: {
    request_count: number;
    total_tokens: number;
    estimated_cost_usd: number;
  };
  by_provider: Array<{
    provider: string;
    request_count: number;
    total_tokens: number;
    estimated_cost_usd: number;
  }>;
  by_model: Array<{
    provider: string;
    model_id: string | null;
    request_count: number;
    total_tokens: number;
    estimated_cost_usd: number;
  }>;
  by_usage_type: Array<{
    usage_type: string;
    request_count: number;
    total_tokens: number;
    estimated_cost_usd: number;
  }>;
  cost_disclaimer: string;
}

export interface UsageDailyPoint {
  date: string;
  request_count: number;
  total_tokens: number;
  estimated_cost_usd: number;
  input_tokens: number;
  output_tokens: number;
}

export interface UsageEvent {
  id: string;
  created_at: string;
  provider: string;
  model_id: string | null;
  usage_type: string;
  input_tokens: number | null;
  output_tokens: number | null;
  cache_read_tokens: number | null;
  cache_write_tokens: number | null;
  total_tokens: number | null;
  estimated_cost_usd: number | null;
  cost_type: string;
  pricing_note: string | null;
}

interface UsageEventsResponse {
  total: number;
  page: number;
  limit: number;
  pages: number;
  items: UsageEvent[];
}

function utcDateOffset(days: number): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

function formatTokens(value: number | null | undefined): string {
  if (value == null) return "—";
  return value.toLocaleString();
}

function formatUtc(iso: string): string {
  try {
    return new Date(iso)
      .toISOString()
      .replace("T", " ")
      .replace(/\.\d+Z$/, " UTC");
  } catch {
    return iso;
  }
}

function costTypeLabel(costType: string): string {
  switch (costType) {
    case "local_zero":
      return "Local $0";
    case "provider_reported":
      return "Provider";
    case "estimated":
      return "Estimated";
    case "unavailable":
      return "N/A";
    default:
      return costType;
  }
}

export default function AiUsagePanel() {
  const [start, setStart] = useState(() => utcDateOffset(-29));
  const [end, setEnd] = useState(() => utcDateOffset(0));
  const [summary, setSummary] = useState<UsageSummary | null>(null);
  const [daily, setDaily] = useState<UsageDailyPoint[]>([]);
  const [events, setEvents] = useState<UsageEvent[]>([]);
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const query = useMemo(
    () => `start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`,
    [start, end],
  );

  const load = useCallback(
    async (nextPage = 1) => {
      try {
        setLoading(true);
        const [summaryData, dailyData, eventsData] = await Promise.all([
          api.get<UsageSummary>(`/api/v1/usage/summary?${query}`),
          api.get<{ points: UsageDailyPoint[] }>(`/api/v1/usage/daily?${query}`),
          api.get<UsageEventsResponse>(
            `/api/v1/usage/events?${query}&page=${nextPage}&limit=25`,
          ),
        ]);
        setSummary(summaryData);
        setDaily(dailyData.points || []);
        setEvents(eventsData.items || []);
        setPage(eventsData.page);
        setPages(eventsData.pages);
        setTotal(eventsData.total);
        setError("");
      } catch (err) {
        setError(err instanceof Error ? err.message : CONNECTION_ERROR_MESSAGE);
      } finally {
        setLoading(false);
      }
    },
    [query],
  );

  useEffect(() => {
    void load(1);
  }, [load]);

  return (
    <div className="ledger-stack !mt-0">
      <div className="ledger-toolbar !mt-0">
        <div>
          <p className="text-[14px] text-[var(--gd-muted)]">
            Authenticated request ledger — tokens and estimated cost in UTC.
          </p>
        </div>
        <div className="ledger-filters">
          <div className="ledger-field mb-0">
            <label htmlFor="usage-from">From</label>
            <Input
              id="usage-from"
              type="date"
              value={start}
              onChange={(e) => setStart(e.target.value)}
              className="h-10 w-[150px] rounded-[10px]"
            />
          </div>
          <div className="ledger-field mb-0">
            <label htmlFor="usage-to">To</label>
            <Input
              id="usage-to"
              type="date"
              value={end}
              onChange={(e) => setEnd(e.target.value)}
              className="h-10 w-[150px] rounded-[10px]"
            />
          </div>
          <LedgerButton
            variant="secondary"
            onClick={() => void load(1)}
            disabled={loading}
          >
            <RefreshCw className={loading ? "h-3.5 w-3.5 animate-spin" : "h-3.5 w-3.5"} />
            Refresh
          </LedgerButton>
        </div>
      </div>

      {error ? <LedgerError>{error}</LedgerError> : null}

      {loading && !summary ? (
        <div className="ledger-kpi-strip">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="ledger-kpi">
              <Skeleton className="h-3 w-20" />
              <Skeleton className="mt-3 h-8 w-16" />
            </div>
          ))}
        </div>
      ) : (
        <LedgerKpiStrip
          items={[
            {
              label: "Estimated cost",
              value: formatUsd(summary?.estimated_cost_usd),
              caption: `Today ${formatUsd(summary?.today.estimated_cost_usd)} · MTD ${formatUsd(summary?.month_to_date.estimated_cost_usd)}`,
              emphasis: "money",
            },
            {
              label: "Total tokens",
              value: formatTokens(summary?.total_tokens),
              caption: `${formatTokens(summary?.input_tokens)} in · ${formatTokens(summary?.output_tokens)} out`,
              emphasis: "data",
            },
            {
              label: "Requests",
              value: formatTokens(summary?.request_count),
              caption: `Today ${summary?.today.request_count ?? 0} · MTD ${summary?.month_to_date.request_count ?? 0}`,
            },
            {
              label: "Cached tokens",
              value: formatTokens(summary?.cache_read_tokens),
              caption: "Provider-reported cache reads when available",
            },
          ]}
        />
      )}

      <LedgerPanel
        title="Daily usage"
        description="Token volume by UTC day in the selected range"
      >
        {daily.length === 0 ? (
          <LedgerEmpty>No usage in this range.</LedgerEmpty>
        ) : (
          <LedgerBarChart points={daily} />
        )}
      </LedgerPanel>

      {(summary?.by_provider?.length || summary?.by_model?.length) ? (
        <div className="ledger-grid-2">
          <LedgerPanel title="By provider">
            <LedgerScrollTable compact>
              <table className="ledger-table">
                <thead>
                  <tr>
                    <th>Provider</th>
                    <th data-align="right">Tokens</th>
                    <th data-align="right">Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {(summary?.by_provider || []).map((row) => (
                    <tr key={row.provider}>
                      <td className="capitalize font-medium">{row.provider}</td>
                      <td data-align="right">{formatTokens(row.total_tokens)}</td>
                      <td data-align="right">{formatUsd(row.estimated_cost_usd)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </LedgerScrollTable>
          </LedgerPanel>
          <LedgerPanel title="By model">
            <LedgerScrollTable compact>
              <table className="ledger-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th data-align="right">Requests</th>
                    <th data-align="right">Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {(summary?.by_model || []).map((row) => (
                    <tr key={`${row.provider}-${row.model_id || "unknown"}`}>
                      <td className="max-w-[200px] truncate font-medium">
                        {row.model_id || "unknown"}
                      </td>
                      <td data-align="right">{row.request_count}</td>
                      <td data-align="right">{formatUsd(row.estimated_cost_usd)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </LedgerScrollTable>
          </LedgerPanel>
        </div>
      ) : null}

      <LedgerPanel
        title="Usage transactions"
        description={`${total} request${total === 1 ? "" : "s"} in range`}
      >
        <LedgerScrollTable
          footer={
            pages > 1 ? (
              <div className="ledger-foot">
                <span>
                  Page {page} of {pages}
                </span>
                <div className="flex gap-2">
                  <LedgerButton
                    variant="secondary"
                    disabled={page <= 1 || loading}
                    onClick={() => void load(page - 1)}
                  >
                    Previous
                  </LedgerButton>
                  <LedgerButton
                    variant="secondary"
                    disabled={page >= pages || loading}
                    onClick={() => void load(page + 1)}
                  >
                    Next
                  </LedgerButton>
                </div>
              </div>
            ) : null
          }
        >
          <table className="ledger-table">
            <thead>
              <tr>
                <th>Date (UTC)</th>
                <th>Type</th>
                <th>Model</th>
                <th data-align="right">Input</th>
                <th data-align="right">Output</th>
                <th data-align="right">Cached</th>
                <th data-align="right">Cost</th>
                <th>Cost type</th>
              </tr>
            </thead>
            <tbody>
              {events.length === 0 ? (
                <tr>
                  <td colSpan={8}>
                    <LedgerEmpty>No AI requests recorded yet.</LedgerEmpty>
                  </td>
                </tr>
              ) : (
                events.map((event) => (
                  <tr key={event.id}>
                    <td className="text-[var(--gd-muted)]">
                      {formatUtc(event.created_at)}
                    </td>
                    <td className="capitalize">{event.usage_type}</td>
                    <td className="max-w-[180px] truncate">
                      <span className="text-[var(--gd-muted)]">{event.provider}/</span>
                      {event.model_id || "—"}
                    </td>
                    <td data-align="right">{formatTokens(event.input_tokens)}</td>
                    <td data-align="right">{formatTokens(event.output_tokens)}</td>
                    <td data-align="right">
                      {formatTokens(event.cache_read_tokens)}
                    </td>
                    <td data-align="right">{formatUsd(event.estimated_cost_usd)}</td>
                    <td>
                      <LedgerStatus label={costTypeLabel(event.cost_type)} />
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </LedgerScrollTable>
      </LedgerPanel>

      {summary?.cost_disclaimer ? (
        <p className="text-[12px] text-[var(--gd-muted)]">{summary.cost_disclaimer}</p>
      ) : null}
    </div>
  );
}

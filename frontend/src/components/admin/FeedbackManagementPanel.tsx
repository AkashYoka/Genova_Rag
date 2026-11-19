"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { MessageSquare, RefreshCw, Search, ThumbsDown, ThumbsUp } from "lucide-react";

import { api } from "@/lib/api";
import {
  AdminButton,
  Badge,
  Callout,
  Card,
  DefinitionList,
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
} from "@/components/admin/console/ConsoleUI";

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
  if (!reason) return "No reason given";
  return (
    REASON_LABELS[reason] ||
    reason.replace(/_/g, " ").replace(/^./, (char) => char.toUpperCase())
  );
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
  });
}

function userLabel(item: FeedbackItem): string {
  return item.username || item.email || `User ${item.user_id.slice(0, 8)}`;
}

function initialsOf(value: string) {
  return value
    .split(/[\s._@-]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("");
}

export function FeedbackManagementPanel() {
  const [data, setData] = useState<FeedbackResponse | null>(null);
  const [selected, setSelected] = useState<FeedbackItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [feedbackFilter, setFeedbackFilter] = useState<"all" | FeedbackType>("all");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [page, setPage] = useState(1);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setSearch(searchInput.trim());
      setPage(1);
    }, 350);
    return () => window.clearTimeout(timer);
  }, [searchInput]);

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
      const result = await api.get<FeedbackResponse>(`/api/v1/admin/feedback?${params}`);
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load feedback");
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
  const items = data?.items ?? [];
  const hasFilters = Boolean(search || start || end || feedbackFilter !== "all");

  const satisfactionCaption = useMemo(() => {
    if (!summary || summary.total === 0) return "No ratings yet";
    return `${summary.likes.toLocaleString()} liked · ${summary.dislikes.toLocaleString()} disliked`;
  }, [summary]);

  return (
    <>
      <PageHeader
        title="Feedback"
        description="What people told us about individual answers, so poor answers can be found and fixed."
        actions={
          <AdminButton variant="secondary" onClick={() => void load()} disabled={loading}>
            <RefreshCw className={loading ? "animate-spin" : undefined} /> Refresh
          </AdminButton>
        }
      />

      {error ? (
        <div style={{ marginBottom: 14 }}>
          <Callout tone="danger">
            <strong>Feedback could not be loaded.</strong> {error}
          </Callout>
        </div>
      ) : null}

      <StatGrid>
        <Stat
          label="Ratings received"
          icon={<MessageSquare />}
          value={summary ? summary.total.toLocaleString() : "—"}
          caption="Across the whole platform"
        />
        <Stat
          label="Liked"
          icon={<ThumbsUp />}
          value={summary ? summary.likes.toLocaleString() : "—"}
          caption="Marked as helpful"
        />
        <Stat
          label="Disliked"
          icon={<ThumbsDown />}
          value={summary ? summary.dislikes.toLocaleString() : "—"}
          caption="Marked as unhelpful"
          attention={(summary?.dislikes ?? 0) > 0}
        />
        <Stat
          label="Satisfaction"
          icon={<ThumbsUp />}
          value={summary ? `${summary.satisfaction_percentage.toFixed(1)}%` : "—"}
          caption={satisfactionCaption}
        />
      </StatGrid>

      <div style={{ marginTop: 20 }}>
        <div className="adm-toolbar">
          <SearchInput
            label="Search feedback"
            value={searchInput}
            onChange={setSearchInput}
            placeholder="Search person, question, or answer"
          />
          <Field label="Rating" htmlFor="feedback-type">
            <SelectInput
              id="feedback-type"
              value={feedbackFilter}
              onChange={(event) => {
                setFeedbackFilter(event.target.value as "all" | FeedbackType);
                setPage(1);
              }}
              style={{ width: 150 }}
            >
              <option value="all">All ratings</option>
              <option value="up">Liked</option>
              <option value="down">Disliked</option>
            </SelectInput>
          </Field>
          <Field label="From" htmlFor="feedback-start">
            <TextInput
              id="feedback-start"
              type="date"
              value={start}
              onChange={(event) => {
                setStart(event.target.value);
                setPage(1);
              }}
              style={{ width: 158 }}
            />
          </Field>
          <Field label="To" htmlFor="feedback-end">
            <TextInput
              id="feedback-end"
              type="date"
              value={end}
              onChange={(event) => {
                setEnd(event.target.value);
                setPage(1);
              }}
              style={{ width: 158 }}
            />
          </Field>
          {hasFilters ? (
            <AdminButton
              variant="ghost"
              onClick={() => {
                setSearchInput("");
                setSearch("");
                setStart("");
                setEnd("");
                setFeedbackFilter("all");
                setPage(1);
              }}
            >
              Clear filters
            </AdminButton>
          ) : null}
          <span className="adm-toolbar__spacer" />
          <span className="adm-toolbar__meta">
            {loading ? "Loading…" : `${total.toLocaleString()} matching rating${total === 1 ? "" : "s"}`}
          </span>
        </div>

        <Card
          title="Ratings"
          description="Select a row to read the full question, the answer we gave, and what the person said."
        >
          {loading && !data ? (
            <TableSkeleton rows={5} />
          ) : items.length ? (
            <TableWrap height="auto">
              <table className="adm-table">
                <thead>
                  <tr>
                    <th>Person</th>
                    <th>Rating</th>
                    <th>Question</th>
                    <th className="adm-col-md">When</th>
                    <th className="adm-col-sm">Reason</th>
                    <th aria-label="Open" />
                  </tr>
                </thead>
                <tbody>
                  {items.map((item) => (
                    <tr
                      key={item.id}
                      data-clickable="true"
                      tabIndex={0}
                      onClick={() => setSelected(item)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          setSelected(item);
                        }
                      }}
                    >
                      <td>
                        <div className="adm-cell-title">
                          <span className="adm-avatar" style={{ width: 28, height: 28 }}>
                            {initialsOf(userLabel(item))}
                          </span>
                          <span className="adm-cell-stack">
                            <span className="adm-cell-main">{userLabel(item)}</span>
                            <span className="adm-cell-sub">{item.email || ""}</span>
                          </span>
                        </div>
                      </td>
                      <td>
                        <Badge tone={item.feedback === "up" ? "ok" : "danger"}>
                          {item.feedback === "up" ? "Liked" : "Disliked"}
                        </Badge>
                      </td>
                      <td>
                        <div className="adm-cell-stack">
                          <span className="adm-cell-main" style={{ whiteSpace: "normal", maxWidth: 420 }}>
                            {item.question || "No question recorded"}
                          </span>
                          <span className="adm-cell-sub" style={{ maxWidth: 420 }}>
                            {item.response || "No answer recorded"}
                          </span>
                        </div>
                      </td>
                      <td className="adm-col-md adm-muted" style={{ fontSize: 12.5, whiteSpace: "nowrap" }}>
                        {formatWhen(item.created_at)}
                      </td>
                      <td className="adm-col-sm adm-muted" style={{ fontSize: 12.5 }}>
                        {formatReason(item.reason)}
                      </td>
                      <td data-align="right">
                        <span className="adm-link">Read</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </TableWrap>
          ) : (
            <EmptyState
              icon={<Search />}
              title={hasFilters ? "No ratings match these filters" : "No ratings yet"}
              text={
                hasFilters
                  ? "Try a wider date range, or clear the search and rating filters."
                  : "When people rate an answer in the workspace, it appears here."
              }
              action={
                hasFilters ? (
                  <AdminButton
                    size="sm"
                    variant="secondary"
                    onClick={() => {
                      setSearchInput("");
                      setSearch("");
                      setStart("");
                      setEnd("");
                      setFeedbackFilter("all");
                      setPage(1);
                    }}
                  >
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
                noun="ratings"
                nounOne="rating"
              />
            </div>
          ) : null}
        </Card>

        <div style={{ marginTop: 12 }}>
          <Callout icon={<MessageSquare />}>
            Use dislikes as a starting point, not a verdict. Read the question and the answer together
            — then check the answer trace if you need to see which document passages were used.
          </Callout>
        </div>
      </div>

      <Drawer
        open={selected !== null}
        onClose={() => setSelected(null)}
        title={selected ? userLabel(selected) : "Feedback"}
        description={
          selected
            ? `${formatWhen(selected.created_at)} · ${selected.feedback === "up" ? "Marked helpful" : "Marked unhelpful"}`
            : undefined
        }
        footer={
          <AdminButton variant="secondary" onClick={() => setSelected(null)}>
            Close
          </AdminButton>
        }
      >
        {selected ? (
          <div className="adm-stack">
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <Badge tone={selected.feedback === "up" ? "ok" : "danger"}>
                {selected.feedback === "up" ? "Liked" : "Disliked"}
              </Badge>
              <Badge plain>{formatReason(selected.reason)}</Badge>
            </div>

            <Card title="The question">
              <div className="adm-card__body--pad" style={{ fontSize: 13.5, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
                {selected.question || "No question recorded."}
              </div>
            </Card>

            <Card title="The answer Genova gave">
              <div className="adm-card__body--pad" style={{ fontSize: 13.5, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
                {selected.response || "No answer recorded."}
              </div>
            </Card>

            <Card title="What the person said">
              <div className="adm-card__body--pad">
                <DefinitionList
                  items={[
                    { term: "Rating", value: selected.feedback === "up" ? "Helpful" : "Unhelpful" },
                    { term: "Reason", value: formatReason(selected.reason) },
                    {
                      term: "Comment",
                      value: selected.comment ? (
                        <span style={{ whiteSpace: "pre-wrap" }}>{selected.comment}</span>
                      ) : (
                        "No additional comment"
                      ),
                    },
                  ]}
                />
              </div>
            </Card>

            <Card title="References" description="Useful when matching this rating to a detailed trace.">
              <div className="adm-card__body--pad">
                <DefinitionList
                  items={[
                    { term: "Conversation", value: <span className="adm-mono">{selected.session_id || "—"}</span> },
                    { term: "Answer message", value: <span className="adm-mono">{selected.message_id}</span> },
                    {
                      term: "Question message",
                      value: <span className="adm-mono">{selected.question_message_id || "—"}</span>,
                    },
                    { term: "Person", value: <span className="adm-mono">{selected.user_id}</span> },
                  ]}
                />
              </div>
            </Card>
          </div>
        ) : null}
      </Drawer>
    </>
  );
}

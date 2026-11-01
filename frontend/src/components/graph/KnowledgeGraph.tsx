"use client";
/**
 * KnowledgeGraph — interactive workspace knowledge-graph explorer.
 *
 * Backed by the GraphRAG endpoints in `backend/app/routes/graph.py`:
 *   GET /api/v1/graph/documents            — documents + graph indexing status
 *   GET /api/v1/graph/overview             — counts, entity/relationship types
 *   GET /api/v1/graph/neighborhood         — bounded subgraph (seed, depth, scope)
 *   GET /api/v1/graph/search               — entity search
 *   GET /api/v1/graph/entities/{id}        — entity details + evidence
 *   GET /api/v1/graph/relationships/{id}   — relationship details + evidence
 *
 * Only forced relationship types are drawn; every edge carries the quoted
 * source text that produced it, shown in the details panel.
 *
 * The graph is loaded lazily and bounded: a neighbourhood is fetched, never the
 * whole store, and expansion merges a single hop at a time.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  MarkerType,
  type ReactFlowInstance,
  useNodesState,
  useEdgesState,
  type Edge,
  type Node,
  type NodeMouseHandler,
  type EdgeMouseHandler,
  type OnNodesChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import {
  ArrowRight,
  FileText,
  Info,
  Loader2,
  Network,
  RefreshCw,
  RotateCcw,
  Search,
  X,
} from "lucide-react";
import { useTheme } from "next-themes";
import { api } from "@/lib/api";

// ── API shapes ────────────────────────────────────────────────────────────────

interface GraphEntity {
  id: string;
  canonical_name: string;
  normalized_name?: string;
  entity_type: string;
  description: string;
  aliases: string[];
  mention_count: number;
  document_count: number;
  confidence: number;
  possible_duplicate_of?: string | null;
}

interface GraphRelationship {
  id: string;
  source: { id: string; name: string; type: string };
  target: { id: string; name: string; type: string };
  relation_type: string;
  description: string;
  confidence: number;
  evidence_count: number;
  evidence: GraphEvidence[];
}

interface GraphEvidence {
  document_id: string;
  document_name: string;
  chunk_id?: string;
  page?: number | null;
  section?: string;
  evidence: string;
  confidence: number;
}

interface Neighborhood {
  nodes: GraphEntity[];
  edges: GraphRelationship[];
  truncated: boolean;
  seed_entity_ids: string[];
}

interface Overview {
  available: boolean;
  scope: string;
  entity_count: number;
  relationship_count: number;
  document_count: number;
  entity_types: { type: string; count: number }[];
  relationship_types: { type: string; count: number }[];
}

interface GraphDocument {
  document_id: string;
  document_name: string;
  status: string;
  entity_count: number;
  relationship_count: number;
  error?: string | null;
}

interface EntityDetail extends GraphEntity {
  mentions: GraphEvidence[];
  documents: { id: string; name: string }[];
  connected: {
    id: string;
    name: string;
    type: string;
    relation_type: string;
    direction: "outgoing" | "incoming";
    confidence: number;
  }[];
  relationships: GraphRelationship[];
}

// ── Presentation helpers ─────────────────────────────────────────────────────

const NODE_LIMIT = 120;

const LIGHT_PALETTE = [
  { bg: "#e0f2fe", border: "#0284c7", text: "#075985" },
  { bg: "#dcfce7", border: "#16a34a", text: "#166534" },
  { bg: "#fef3c7", border: "#d97706", text: "#92400e" },
  { bg: "#ede9fe", border: "#7c3aed", text: "#4c1d95" },
  { bg: "#ffe4e6", border: "#e11d48", text: "#9f1239" },
  { bg: "#e0e7ff", border: "#4f46e5", text: "#312e81" },
  { bg: "#ccfbf1", border: "#0d9488", text: "#115e59" },
  { bg: "#f1f5f9", border: "#475569", text: "#0f172a" },
];

/** Deep fills with light text keep nodes legible on the dark canvas. */
const DARK_PALETTE = [
  { bg: "#0b3a53", border: "#38bdf8", text: "#e0f2fe" },
  { bg: "#14532d", border: "#4ade80", text: "#dcfce7" },
  { bg: "#713f12", border: "#fbbf24", text: "#fef3c7" },
  { bg: "#4c1d95", border: "#a78bfa", text: "#ede9fe" },
  { bg: "#841337", border: "#fb7185", text: "#ffe4e6" },
  { bg: "#312e81", border: "#818cf8", text: "#e0e7ff" },
  { bg: "#134e4a", border: "#2dd4bf", text: "#ccfbf1" },
  { bg: "#1e293b", border: "#94a3b8", text: "#f1f5f9" },
];

function colourForType(type: string, dark = false) {
  let hash = 0;
  for (let i = 0; i < type.length; i += 1) {
    hash = (hash * 31 + type.charCodeAt(i)) % 997;
  }
  const palette = dark ? DARK_PALETTE : LIGHT_PALETTE;
  return palette[hash % palette.length];
}

function prettyLabel(value: string): string {
  return (value || "unknown").replace(/[_-]+/g, " ").toLowerCase().replace(/^\w/, (c) => c.toUpperCase());
}

function confidencePct(value: number): string {
  return `${Math.round(Math.max(0, Math.min(1, value || 0)) * 100)}%`;
}

/**
 * Deterministic sunflower (phyllotaxis) layout: nodes spread evenly instead of
 * stacking on a single ring, so edges and labels stay readable. The same
 * subgraph always reloads into the same positions.
 */
function radialPosition(index: number, total: number) {
  if (total <= 1) return { x: 0, y: 0 };
  const spacing = Math.max(165, Math.min(240, 3400 / total));
  const radius = spacing * Math.sqrt(index + 0.5);
  const angle = index * 2.399963229728653; // golden angle in radians
  return { x: radius * Math.cos(angle), y: radius * Math.sin(angle) };
}

// ── Component ────────────────────────────────────────────────────────────────

interface KnowledgeGraphProps {
  onClose: () => void;
  onMount?: () => void;
}

export default function KnowledgeGraph({ onClose, onMount }: KnowledgeGraphProps) {
  const { resolvedTheme, theme } = useTheme();
  // ocean/forest/sunset are dark surfaces too, so anything but "light" is dark.
  const isDark = (resolvedTheme ?? theme) !== undefined && (resolvedTheme ?? theme) !== "light";

  const [documents, setDocuments] = useState<GraphDocument[]>([]);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [depth, setDepth] = useState(1);
  const [showLabels, setShowLabels] = useState(true);

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [flowInstance, setFlowInstance] = useState<ReactFlowInstance | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [truncated, setTruncated] = useState(false);
  const [expandingId, setExpandingId] = useState<string | null>(null);

  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<GraphEntity[]>([]);
  const [searching, setSearching] = useState(false);

  const [typeFilter, setTypeFilter] = useState<string[]>([]);
  const [relationFilter, setRelationFilter] = useState("");

  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [selectedRelationshipId, setSelectedRelationshipId] = useState<string | null>(null);
  const [entityDetail, setEntityDetail] = useState<EntityDetail | null>(null);
  const [relationshipDetail, setRelationshipDetail] = useState<GraphRelationship | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  // ── Graph loading ──────────────────────────────────────────────────────────

  const loadGraph = useCallback(
    async (seedEntityId?: string) => {
      setLoading(!seedEntityId);
      setError(null);
      setSelectedEntityId(null);
      setSelectedRelationshipId(null);
      setEntityDetail(null);
      setRelationshipDetail(null);
      try {
        const params = new URLSearchParams({
          depth: String(depth),
          limit: String(NODE_LIMIT),
        });
        if (seedEntityId) params.set("entity_id", seedEntityId);
        const data = await api.get<Neighborhood>(
          `/api/v1/graph/neighborhood?${params.toString()}`,
        );

        const total = data.nodes?.length ?? 0;
        const nextNodes: Node[] = (data.nodes ?? []).map((entity, index) =>
          buildNode(entity, radialPosition(index, total), isDark),
        );
        const nextEdges: Edge[] = (data.edges ?? []).map((relationship) =>
          buildEdge(relationship, showLabels, isDark),
        );

        setNodes(nextNodes);
        setEdges(nextEdges);
        setTruncated(Boolean(data.truncated));
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load the graph");
        setNodes([]);
        setEdges([]);
      } finally {
        setLoading(false);
      }
    },
    [depth, isDark, setNodes, setEdges, showLabels],
  );

  const loadOverview = useCallback(async () => {
    try {
      const data = await api.get<Overview>("/api/v1/graph/overview");
      setOverview(data);
    } catch {
      setOverview(null);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await api.get<{ documents: GraphDocument[] }>(
          "/api/v1/graph/documents",
        );
        if (!cancelled) setDocuments(data.documents ?? []);
      } catch {
        if (!cancelled) setDocuments([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    // Defer so the first render paints the loading state before the fetch.
    const timer = setTimeout(() => {
      void loadGraph();
      void loadOverview();
    }, 0);
    return () => clearTimeout(timer);
  }, [loadGraph, loadOverview]);

  // Refit whenever a neighbourhood replaces the graph so every node stays in view.
  useEffect(() => {
    if (!flowInstance || nodes.length === 0) return;
    const frame = requestAnimationFrame(() =>
      flowInstance.fitView({ padding: 0.2, duration: 250 }),
    );
    return () => cancelAnimationFrame(frame);
  }, [flowInstance, nodes.length]);

  // Let the host fit the canvas once the first neighbourhood is on screen.
  const mountNotified = useRef(false);
  useEffect(() => {
    if (mountNotified.current || loading || nodes.length === 0) return;
    mountNotified.current = true;
    onMount?.();
  }, [loading, nodes.length, onMount]);

  // Keep edge labels in sync with the toggle without refetching.
  useEffect(() => {
    setEdges((current) =>
      current.map((edge) => ({
        ...edge,
        label: showLabels
          ? prettyLabel((edge.data as { relationType?: string })?.relationType ?? "")
          : undefined,
      })),
    );
  }, [showLabels, setEdges]);

  // ── Entity search ──────────────────────────────────────────────────────────

  useEffect(() => {
    const term = query.trim();
    let cancelled = false;
    const timer = setTimeout(async () => {
      if (term.length < 2) {
        if (!cancelled) {
          setSearchResults([]);
          setSearching(false);
        }
        return;
      }
      setSearching(true);
      try {
        const params = new URLSearchParams({ q: term, limit: "12" });
        const data = await api.get<{ results: GraphEntity[] }>(
          `/api/v1/graph/search?${params.toString()}`,
        );
        if (!cancelled) setSearchResults(data.results ?? []);
      } catch {
        if (!cancelled) setSearchResults([]);
      } finally {
        if (!cancelled) setSearching(false);
      }
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [query]);

  // ── Selection / details ────────────────────────────────────────────────────

  const selectEntity = useCallback(async (entityId: string) => {
    setSelectedEntityId(entityId);
    setSelectedRelationshipId(null);
    setRelationshipDetail(null);
    setEntityDetail(null);
    setDetailLoading(true);
    try {
      const detail = await api.get<EntityDetail>(
        `/api/v1/graph/entities/${encodeURIComponent(entityId)}`,
      );
      setEntityDetail(detail);
    } catch {
      setEntityDetail(null);
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const selectRelationship = useCallback(async (relationshipId: string) => {
    setSelectedRelationshipId(relationshipId);
    setSelectedEntityId(null);
    setEntityDetail(null);
    setRelationshipDetail(null);
    setDetailLoading(true);
    try {
      const detail = await api.get<GraphRelationship>(
        `/api/v1/graph/relationships/${encodeURIComponent(relationshipId)}`,
      );
      setRelationshipDetail(detail);
    } catch {
      setRelationshipDetail(null);
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const expandEntity = useCallback(
    async (entityId: string) => {
      if (expandingId) return;
      setExpandingId(entityId);
      try {
        const params = new URLSearchParams({ entity_id: entityId, depth: "1", limit: "25" });
        const data = await api.get<Neighborhood>(
          `/api/v1/graph/neighborhood?${params.toString()}`,
        );

        setNodes((current) => {
          const seed = current.find((node) => node.id === entityId);
          const base = seed?.position ?? { x: 0, y: 0 };
          const existing = new Set(current.map((node) => node.id));
          const additions = (data.nodes ?? []).filter((entity) => !existing.has(entity.id));
          const total = additions.length;
          const newNodes = additions.map((entity, index) => {
            const offset = radialPosition(index, Math.max(total, 3));
            return buildNode(
              entity,
              { x: base.x + offset.x * 0.6, y: base.y + offset.y * 0.6 },
              isDark,
            );
          });
          return [...current, ...newNodes].slice(0, NODE_LIMIT + 40);
        });

        const existingEdges = new Set(edges.map((edge) => edge.id));
        const newEdges = (data.edges ?? [])
          .filter((relationship) => !existingEdges.has(relationship.id))
          .map((relationship) => buildEdge(relationship, showLabels, isDark));
        setEdges((current) => [...current, ...newEdges]);
      } catch {
        /* expansion is best-effort; the current graph stays usable */
      } finally {
        setExpandingId(null);
      }
    },
    [edges, expandingId, isDark, setEdges, setNodes, showLabels],
  );

  const onNodeClick = useCallback<NodeMouseHandler>(
    (_event, node) => {
      void selectEntity(node.id);
    },
    [selectEntity],
  );

  const onEdgeClick = useCallback<EdgeMouseHandler>(
    (_event, edge) => {
      void selectRelationship(edge.id);
    },
    [selectRelationship],
  );

  const onReset = useCallback(() => {
    setQuery("");
    setSearchResults([]);
    setTypeFilter([]);
    setRelationFilter("");
    setDepth(1);
    setSelectedEntityId(null);
    setSelectedRelationshipId(null);
    setEntityDetail(null);
    setRelationshipDetail(null);
    void loadGraph();
    void loadOverview();
  }, [loadGraph, loadOverview]);

  const retryDocument = useCallback(async (documentId: string) => {
    try {
      await api.post(`/api/v1/graph/documents/${encodeURIComponent(documentId)}/reindex`, {});
      setDocuments((current) =>
        current.map((doc) =>
          doc.document_id === documentId ? { ...doc, status: "queued" } : doc,
        ),
      );
    } catch {
      /* surfaced by the status dot staying on "failed" */
    }
  }, []);

  // ── Derived data ───────────────────────────────────────────────────────────

  const presentTypes = useMemo(
    () =>
      Array.from(
        new Set(
          nodes
            .map((node) => (node.data as { entityType?: string })?.entityType)
            .filter((value): value is string => Boolean(value)),
        ),
      ).sort(),
    [nodes],
  );

  const presentRelations = useMemo(
    () =>
      Array.from(
        new Set(
          edges
            .map((edge) => (edge.data as { relationType?: string })?.relationType)
            .filter((value): value is string => Boolean(value)),
        ),
      ).sort(),
    [edges],
  );

  const displayNodes = useMemo(() => {
    const allowedTypes = typeFilter.length ? new Set(typeFilter) : null;
    return nodes.map((node) => {
      const entityType = (node.data as { entityType?: string })?.entityType ?? "";
      return {
        ...node,
        hidden: allowedTypes ? !allowedTypes.has(entityType) : false,
        selected: node.id === selectedEntityId,
      };
    });
  }, [nodes, typeFilter, selectedEntityId]);

  const displayEdges = useMemo(() => {
    const allowedTypes = typeFilter.length ? new Set(typeFilter) : null;
    return edges.map((edge) => {
      const relationType = (edge.data as { relationType?: string })?.relationType ?? "";
      const sourceHidden = allowedTypes
        ? !allowedTypes.has((edge.data as { sourceType?: string })?.sourceType ?? "")
        : false;
      const targetHidden = allowedTypes
        ? !allowedTypes.has((edge.data as { targetType?: string })?.targetType ?? "")
        : false;
      return {
        ...edge,
        hidden:
          sourceHidden ||
          targetHidden ||
          (relationFilter ? relationType !== relationFilter : false),
        selected: edge.id === selectedRelationshipId,
      };
    });
  }, [edges, typeFilter, relationFilter, selectedRelationshipId]);

  const busy = loading || detailLoading;
  const hasGraph = nodes.length > 0;

  const toggleType = (type: string) => {
    setTypeFilter((current) =>
      current.includes(type) ? current.filter((item) => item !== type) : [...current, type],
    );
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-background text-foreground">
      {/* Header */}
      <div className="flex shrink-0 items-start justify-between gap-3 border-b border-border/60 px-4 py-3">
        <div className="min-w-0">
          <h2 className="flex items-center gap-2 text-sm font-semibold">
            <Network className="h-4 w-4 text-primary" />
            Knowledge Graph
            {overview && overview.available ? (
              <span className="text-xs font-normal text-muted-foreground">
                {overview.entity_count} entities · {overview.relationship_count} relationships
              </span>
            ) : null}
          </h2>
          <p className="mt-0.5 truncate text-xs text-muted-foreground">
            {documents.length
              ? `Entire workspace · ${documents.length} document${documents.length === 1 ? "" : "s"} linked`
              : "Entire workspace · every indexed document"}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <button
            type="button"
            onClick={() => void loadGraph()}
            disabled={loading}
            className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:opacity-50"
            aria-label="Refresh graph"
            title="Refresh"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          </button>
          <button
            type="button"
            onClick={onReset}
            className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            aria-label="Reset filters"
            title="Reset filters"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            aria-label="Close knowledge graph"
            title="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="flex min-h-0 flex-1">
        {/* Controls rail */}
        <aside className="flex w-60 shrink-0 flex-col gap-4 overflow-y-auto border-r border-border/60 bg-card/40 p-3">
          <div>
            <label
              htmlFor="graph-search"
              className="mb-1 block text-[11px] font-medium uppercase tracking-wide text-muted-foreground"
            >
              Search entities
            </label>
            <div className="relative">
              <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
              <input
                id="graph-search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Name, alias or description"
                className="h-8 w-full rounded-md border border-border/60 bg-background pl-7 pr-2 text-xs outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/40"
              />
              {searching ? (
                <Loader2 className="absolute right-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 animate-spin text-muted-foreground" />
              ) : null}
            </div>
            {searchResults.length > 0 ? (
              <ul className="mt-2 max-h-48 space-y-1 overflow-y-auto" aria-label="Search results">
                {searchResults.map((entity) => (
                  <li key={entity.id}>
                    <button
                      type="button"
                      onClick={() => {
                        setQuery("");
                        setSearchResults([]);
                        void loadGraph(entity.id).then(() => selectEntity(entity.id));
                      }}
                      className="flex w-full items-center justify-between gap-2 rounded-md px-2 py-1.5 text-left text-xs hover:bg-accent"
                    >
                      <span className="truncate font-medium">{entity.canonical_name}</span>
                      <span className="shrink-0 text-[10px] text-muted-foreground">
                        {prettyLabel(entity.entity_type)}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            ) : null}
          </div>

          <div>
            <span className="mb-1 block text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Traversal depth
            </span>
            <div className="flex gap-1" role="group" aria-label="Traversal depth">
              {[1, 2, 3].map((value) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setDepth(value)}
                  aria-pressed={depth === value}
                  className={`h-7 flex-1 rounded-md border text-xs transition-colors ${
                    depth === value
                      ? "border-primary/40 bg-primary/15 font-semibold text-foreground"
                      : "border-border/60 text-muted-foreground hover:bg-accent"
                  }`}
                >
                  {value}
                </button>
              ))}
            </div>
            <p className="mt-1 text-[10px] text-muted-foreground">
              Depth 3 can pull in a wide neighbourhood; results stay bounded.
            </p>
          </div>

          <div>
            <span className="mb-1 block text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Documents in this graph
            </span>
            {documents.length === 0 ? (
              <p className="text-[11px] text-muted-foreground">
                No graph-indexed documents yet.
              </p>
            ) : (
              <ul className="space-y-1">
                {documents.map((document) => {
                  const failed = document.status === "failed";
                  return (
                    <li
                      key={document.document_id}
                      className="flex items-start gap-1.5 rounded-md px-1.5 py-1 text-xs"
                    >
                      <span className="min-w-0 flex-1">
                        <span className="block truncate font-medium" title={document.document_name}>
                          {document.document_name}
                        </span>
                        <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                          <span
                            className={`h-1.5 w-1.5 rounded-full ${
                              failed
                                ? "bg-destructive"
                                : document.status === "ready"
                                  ? "bg-emerald-500"
                                  : "bg-amber-500"
                            }`}
                            aria-hidden
                          />
                          {document.status} · {document.entity_count} entities
                          {failed ? (
                            <button
                              type="button"
                              onClick={() => void retryDocument(document.document_id)}
                              className="ml-1 underline underline-offset-2 hover:text-foreground"
                            >
                              retry
                            </button>
                          ) : null}
                        </span>
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
            <p className="mt-1 text-[10px] text-muted-foreground">
              Entities shared between PDFs are merged, so cross-document relations appear as
              single edges.
            </p>
          </div>

          <div>
            <span className="mb-1 block text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Entity types
            </span>
            {presentTypes.length === 0 ? (
              <p className="text-[11px] text-muted-foreground">No entities loaded.</p>
            ) : (
              <div className="flex flex-wrap gap-1">
                {presentTypes.map((type) => {
                  const colour = colourForType(type, isDark);
                  const active = typeFilter.includes(type);
                  return (
                    <button
                      key={type}
                      type="button"
                      onClick={() => toggleType(type)}
                      aria-pressed={active}
                      className={`rounded-full border px-2 py-0.5 text-[10px] font-medium transition-opacity ${
                        active || typeFilter.length === 0 ? "opacity-100" : "opacity-40"
                      }`}
                      style={{
                        backgroundColor: colour.bg,
                        borderColor: colour.border,
                        color: colour.text,
                      }}
                    >
                      {prettyLabel(type)}
                    </button>
                  );
                })}
              </div>
            )}
            {typeFilter.length > 0 ? (
              <button
                type="button"
                onClick={() => setTypeFilter([])}
                className="mt-1 text-[10px] text-muted-foreground underline underline-offset-2 hover:text-foreground"
              >
                Clear type filter
              </button>
            ) : null}
          </div>

          <div>
            <label
              htmlFor="graph-relation-filter"
              className="mb-1 block text-[11px] font-medium uppercase tracking-wide text-muted-foreground"
            >
              Relationship type
            </label>
            <select
              id="graph-relation-filter"
              value={relationFilter}
              onChange={(event) => setRelationFilter(event.target.value)}
              className="h-8 w-full rounded-md border border-border/60 bg-background px-2 text-xs"
            >
              <option value="">All relationships</option>
              {presentRelations.map((type) => (
                <option key={type} value={type}>
                  {prettyLabel(type)}
                </option>
              ))}
            </select>
            <label className="mt-2 flex items-center gap-2 text-[11px] text-muted-foreground">
              <input
                type="checkbox"
                checked={showLabels}
                onChange={(event) => setShowLabels(event.target.checked)}
              />
              Show relationship labels
            </label>
          </div>
        </aside>

        {/* Canvas */}
        <div className="relative min-w-0 flex-1 h-full overflow-hidden">
          {truncated ? (
            <div className="absolute inset-x-0 top-0 z-20 border-b border-amber-500/30 bg-amber-500/10 px-3 py-1.5 text-center text-[11px] text-amber-700 dark:text-amber-300">
              Showing the {nodes.length} best-connected entities. Search for an entity to explore
              its neighbourhood.
            </div>
          ) : null}

          {loading ? (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-background/85">
              <RefreshCw className="h-7 w-7 animate-spin text-primary/60" />
              <p className="text-sm text-muted-foreground">Loading graph neighbourhood…</p>
            </div>
          ) : null}

          {!loading && error ? (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 px-8 text-center">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-muted">
                <Info className="h-5 w-5 text-muted-foreground" />
              </div>
              <div>
                <p className="mb-1 text-sm font-medium">Could not load the graph</p>
                <p className="text-xs text-muted-foreground">{error}</p>
              </div>
              <button
                type="button"
                onClick={() => void loadGraph()}
                className="rounded-md bg-primary px-3 py-1.5 text-xs text-primary-foreground hover:bg-primary/90"
              >
                Try again
              </button>
            </div>
          ) : null}

          {!loading && !error && !hasGraph ? (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 px-10 text-center">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-muted">
                <Network className="h-5 w-5 text-muted-foreground" />
              </div>
              <div>
                <p className="mb-1 text-sm font-medium">No graph yet</p>
                <p className="text-xs text-muted-foreground">
                  Entities and relationships are extracted from each document while it is indexed.
                  Upload or re-index a document, then refresh. Graphs are built only from text that
                  supports them, so documents without extractable entities stay empty.
                </p>
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => void loadGraph()}
                  className="rounded-md border border-border/60 px-3 py-1.5 text-xs hover:bg-accent"
                >
                  Refresh
                </button>
                {documents.some((doc) => doc.status === "failed") ? (
                  <button
                    type="button"
                    onClick={() => {
                      documents
                        .filter((doc) => doc.status === "failed")
                        .forEach((doc) => void retryDocument(doc.document_id));
                    }}
                    className="rounded-md bg-primary px-3 py-1.5 text-xs text-primary-foreground hover:bg-primary/90"
                  >
                    Retry failed documents
                  </button>
                ) : null}
              </div>
            </div>
          ) : null}

          {!loading && !error && hasGraph ? (
            <ReactFlow
              nodes={displayNodes}
              edges={displayEdges}
              onNodesChange={onNodesChange as OnNodesChange<Node>}
              onEdgesChange={onEdgesChange}
              onNodeClick={onNodeClick}
              onEdgeClick={onEdgeClick}
              onNodeDoubleClick={(_event, node) => void expandEntity(node.id)}
              onPaneClick={() => {
                setSelectedEntityId(null);
                setSelectedRelationshipId(null);
                setEntityDetail(null);
                setRelationshipDetail(null);
              }}
              onInit={setFlowInstance}
              fitView
              fitViewOptions={{ padding: 0.2 }}
              minZoom={0.2}
              maxZoom={2.5}
              nodesConnectable={false}
              elementsSelectable
              deleteKeyCode={["Delete", "Backspace"]}
              attributionPosition="bottom-center"
              colorMode={isDark ? "dark" : "light"}
            >
              <Background
                variant={BackgroundVariant.Dots}
                gap={22}
                size={1.4}
                color={isDark ? "rgba(148,163,184,0.4)" : "rgba(100,116,139,0.4)"}
              />
              <Controls
                showInteractive={false}
                className="!border-border !bg-card !shadow-sm [&>button]:!border-border [&>button]:!bg-card [&>button]:!text-foreground"
                position="bottom-left"
              />
              <MiniMap
                nodeColor={(node) => {
                  const type = (node.data as { entityType?: string })?.entityType ?? "";
                  return colourForType(type, isDark).border;
                }}
                maskColor={isDark ? "rgba(0,0,0,0.55)" : "rgba(0,0,0,0.08)"}
                className="!rounded-lg !border-border !bg-card overflow-hidden"
                position="bottom-right"
                zoomable
                pannable
                draggable
                style={{ pointerEvents: "auto" }}
              />
            </ReactFlow>
          ) : null}
        </div>

        {/* Details */}
        <aside
          className="flex w-64 shrink-0 flex-col overflow-y-auto border-l border-border/60 bg-card/40 p-3"
          data-testid="graph-details-panel"
        >
          <div className="flex min-h-0 flex-col">
            {detailLoading ? (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading details…
              </div>
            ) : null}

            {!detailLoading && !selectedEntityId && !selectedRelationshipId ? (
              <div className="space-y-2 text-xs text-muted-foreground">
                <p className="font-medium text-foreground">Details</p>
                <p>
                  Select an entity to see its documents, source passages and connections, or a
                  relationship to see the exact quote that supports it.
                </p>
                {overview?.available ? (
                  <div className="mt-3 space-y-1 rounded-lg border border-border/60 bg-background/60 p-2">
                    <p className="text-[11px] font-medium uppercase tracking-wide">Graph scope</p>
                    <p>
                      {overview.document_count} document
                      {overview.document_count === 1 ? "" : "s"} · {overview.entity_count} entities
                    </p>
                    <p>{overview.relationship_count} evidence-backed relationships</p>
                  </div>
                ) : null}
              </div>
            ) : null}

            {!detailLoading && entityDetail ? (
              <EntityPanel
                entity={entityDetail}
                busy={busy}
                onSelectEntity={(id) => void selectEntity(id)}
                onExpand={(id) => void expandEntity(id)}
              />
            ) : null}

            {!detailLoading && relationshipDetail ? (
              <RelationshipPanel relationship={relationshipDetail} />
            ) : null}
          </div>
        </aside>
      </div>
    </div>
  );
}

// ── React Flow builders ──────────────────────────────────────────────────────

function buildNode(
  entity: GraphEntity,
  position: { x: number; y: number },
  dark = false,
): Node {
  const colour = colourForType(entity.entity_type, dark);
  return {
    id: entity.id,
    position,
    data: {
      entityType: entity.entity_type,
      label: (
        <div className="flex flex-col items-center leading-tight">
          <span
            className="max-w-[9.5rem] truncate text-[11px] font-semibold"
            title={entity.canonical_name}
            style={{ color: colour.text }}
          >
            {entity.canonical_name}
          </span>
          <span className="text-[9px] uppercase tracking-wide opacity-70" style={{ color: colour.text }}>
            {prettyLabel(entity.entity_type)}
          </span>
        </div>
      ),
    },
    style: {
      borderRadius: 10,
      border: `1.5px solid ${colour.border}`,
      backgroundColor: colour.bg,
      padding: "5px 9px",
      boxShadow: dark ? "0 2px 8px rgba(0,0,0,0.55)" : "0 1px 3px rgba(0,0,0,0.12)",
      cursor: "pointer",
      width: "auto",
      minWidth: 120,
    },
  };
}

function buildEdge(
  relationship: GraphRelationship,
  showLabels: boolean,
  dark = false,
): Edge {
  const label = prettyLabel(relationship.relation_type);
  const stroke = dark ? "rgba(148,163,184,0.9)" : "rgba(71,85,105,0.8)";
  const labelBg = dark ? "#0f172a" : "#f8fafc";
  const labelText = dark ? "#e2e8f0" : "#1e293b";
  return {
    id: relationship.id,
    source: relationship.source.id,
    target: relationship.target.id,
    type: "default",
    label: showLabels ? label : undefined,
    labelStyle: { fontSize: 10, fill: labelText },
    labelBgStyle: { fill: labelBg, fillOpacity: 0.95 },
    labelBgPadding: [4, 2] as [number, number],
    labelBgBorderRadius: 3,
    markerEnd: {
      type: MarkerType.ArrowClosed,
      width: 16,
      height: 16,
      color: stroke,
    },
    style: {
      stroke,
      strokeWidth: 1.5 + Math.max(0, Math.min(1, relationship.confidence)) * 1.5,
    },
    data: {
      relationType: relationship.relation_type,
      sourceType: relationship.source.type,
      targetType: relationship.target.type,
    },
  };
}

// ── Detail panels ────────────────────────────────────────────────────────────

function EvidenceList({ items }: { items: GraphEvidence[] }) {
  if (!items.length) {
    return <p className="text-[11px] text-muted-foreground">No stored evidence.</p>;
  }
  return (
    <ul className="space-y-1.5">
      {items.map((item, index) => (
        <li
          key={`${item.document_id}-${item.chunk_id ?? index}`}
          className="rounded-md border border-border/60 bg-background/70 p-2"
        >
          <p className="mb-1 flex items-center gap-1 text-[10px] text-muted-foreground">
            <FileText className="h-3 w-3" />
            <span className="truncate" title={item.document_name}>
              {item.document_name}
            </span>
            {item.page ? <span>· p.{item.page}</span> : null}
            {item.section ? <span className="truncate">· {item.section}</span> : null}
          </p>
          <p className="text-[11px] leading-snug">“{item.evidence}”</p>
        </li>
      ))}
    </ul>
  );
}

function EntityPanel({
  entity,
  busy,
  onSelectEntity,
  onExpand,
}: {
  entity: EntityDetail;
  busy: boolean;
  onSelectEntity: (id: string) => void;
  onExpand: (id: string) => void;
}) {
  const { resolvedTheme, theme } = useTheme();
  const colour = colourForType(entity.entity_type, (resolvedTheme ?? theme) !== "light");
  return (
    <div className="space-y-3">
      <div>
        <span
          className="inline-block rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide"
          style={{ backgroundColor: colour.bg, color: colour.text }}
        >
          {prettyLabel(entity.entity_type)}
        </span>
        <h3 className="mt-1 break-words text-sm font-semibold">{entity.canonical_name}</h3>
        {entity.description ? (
          <p className="mt-1 text-[11px] leading-snug text-muted-foreground">{entity.description}</p>
        ) : null}
      </div>

      <div className="grid grid-cols-2 gap-1 text-[11px]">
        <Stat label="Mentions" value={String(entity.mention_count)} />
        <Stat label="Documents" value={String(entity.document_count)} />
        <Stat label="Confidence" value={confidencePct(entity.confidence)} />
        <Stat label="Relationships" value={String(entity.relationships.length)} />
      </div>

      {entity.possible_duplicate_of ? (
        <p className="rounded-md border border-amber-500/40 bg-amber-500/10 p-2 text-[10px] text-amber-700 dark:text-amber-300">
          Possible duplicate of another entity — kept separate until evidence confirms the match.
        </p>
      ) : null}

      {entity.aliases.length ? (
        <div>
          <p className="mb-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
            Aliases
          </p>
          <p className="text-[11px]">{entity.aliases.join(", ")}</p>
        </div>
      ) : null}

      <div>
        <p className="mb-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
          Documents
        </p>
        <ul className="space-y-0.5 text-[11px]">
          {entity.documents.map((document) => (
            <li key={document.id} className="truncate" title={document.name}>
              {document.name}
            </li>
          ))}
        </ul>
      </div>

      <div>
        <p className="mb-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
          Connected entities
        </p>
        <ul className="space-y-0.5">
          {entity.connected.map((neighbour) => (
            <li key={`${neighbour.id}-${neighbour.relation_type}`}>
              <button
                type="button"
                onClick={() => onSelectEntity(neighbour.id)}
                className="flex w-full items-center gap-1 rounded px-1 py-0.5 text-left text-[11px] hover:bg-accent"
              >
                <span className="text-muted-foreground">
                  {neighbour.direction === "outgoing" ? "→" : "←"}
                </span>
                <span className="truncate font-medium">{neighbour.name}</span>
                <span className="ml-auto shrink-0 text-[9px] text-muted-foreground">
                  {prettyLabel(neighbour.relation_type)}
                </span>
              </button>
            </li>
          ))}
          {entity.connected.length === 0 ? (
            <li className="text-[11px] text-muted-foreground">
              No relationships stored for this entity.
            </li>
          ) : null}
        </ul>
      </div>

      <div>
        <p className="mb-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
          Source evidence
        </p>
        <EvidenceList items={entity.mentions} />
      </div>

      <button
        type="button"
        onClick={() => onExpand(entity.id)}
        disabled={busy}
        className="w-full rounded-md border border-border/60 px-2 py-1.5 text-[11px] hover:bg-accent disabled:opacity-50"
      >
        Expand one more hop
      </button>
    </div>
  );
}

function RelationshipPanel({ relationship }: { relationship: GraphRelationship }) {
  return (
    <div className="space-y-3">
      <div>
        <p className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
          Relationship
        </p>
        <div className="mt-1 flex flex-wrap items-center gap-1 text-[11px]">
          <span className="rounded bg-muted px-1.5 py-0.5 font-medium">
            {relationship.source.name}
          </span>
          <ArrowRight className="h-3 w-3 text-muted-foreground" />
          <span className="rounded bg-primary/15 px-1.5 py-0.5 font-medium">
            {prettyLabel(relationship.relation_type)}
          </span>
          <ArrowRight className="h-3 w-3 text-muted-foreground" />
          <span className="rounded bg-muted px-1.5 py-0.5 font-medium">
            {relationship.target.name}
          </span>
        </div>
        {relationship.description ? (
          <p className="mt-1 text-[11px] leading-snug text-muted-foreground">
            {relationship.description}
          </p>
        ) : null}
      </div>

      <div className="grid grid-cols-2 gap-1 text-[11px]">
        <Stat label="Confidence" value={confidencePct(relationship.confidence)} />
        <Stat label="Evidence" value={`${relationship.evidence_count} source(s)`} />
      </div>

      <p className="rounded-md border border-border/60 bg-background/60 p-2 text-[10px] text-muted-foreground">
        Direction is stated by the source text; the reverse relationship is not inferred. Confidence
        reflects the extraction certainty, not document authority.
      </p>

      <div>
        <p className="mb-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
          Supporting passages
        </p>
        <EvidenceList items={relationship.evidence} />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border/60 bg-background/60 px-2 py-1">
      <p className="text-[9px] uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="text-[11px] font-semibold">{value}</p>
    </div>
  );
}

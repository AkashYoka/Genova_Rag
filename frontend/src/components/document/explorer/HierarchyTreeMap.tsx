"use client";

import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  type CSSProperties,
  type MouseEvent as ReactMouseEvent,
} from "react";
import {
  Background,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  useViewport,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useTheme } from "next-themes";
import { cn } from "@/lib/utils";
import {
  buildEnrichedTree,
  countVisibleNodes,
  type EnrichedTreeNode,
  type SectionChunksLoading,
  type SectionChunksMap,
  type TreeNodeKind,
} from "@/lib/hierarchy-tree-model";
import type { SectionTreeNode } from "@/types/document-hierarchy";

const FLOW_THEME_VARS = {
  "--xy-background-color": "var(--gd-ground)",
  "--xy-edge-stroke-default": "var(--gd-border)",
  "--xy-edge-stroke-selected-default": "var(--gd-accent)",
  "--xy-minimap-background-color-default": "var(--gd-surface)",
  "--xy-minimap-mask-background-color-default":
    "color-mix(in oklab, var(--gd-ground) 72%, transparent)",
  "--xy-minimap-node-background-color-default": "var(--gd-raised)",
  "--xy-controls-button-background-color-default": "var(--gd-surface)",
  "--xy-controls-button-background-color-hover-default": "var(--gd-raised)",
  "--xy-controls-button-color-default": "var(--gd-text)",
  "--xy-controls-button-color-hover-default": "var(--gd-text)",
  "--xy-controls-button-border-color-default": "var(--gd-border)",
  "--xy-controls-box-shadow-default": "var(--gd-shadow)",
  "--xy-attribution-background-color-default":
    "color-mix(in oklab, var(--gd-surface) 80%, transparent)",
} as CSSProperties;

const ROOT_ID = "__doc_root__";
const NODE_W = 268;
const GAP_X = 26;
const GAP_Y = 44;
const PAD = 36;

function nodeHeight(kind: TreeNodeKind): number {
  if (kind === "document") return 88;
  if (kind === "table") return 128;
  if (kind === "paragraph") return 118;
  if (kind === "loading") return 64;
  return 108;
}

type MapNodeData = {
  kind: TreeNodeKind;
  title: string;
  subtitle?: string;
  pageLabel?: string | null;
  preview?: string;
  tableHeaders?: string[];
  tableShape?: string;
  childCount?: number;
  sectionId?: string;
  chunkId?: string;
  selected?: boolean;
  hasChildren?: boolean;
  expanded?: boolean;
  onSelect?: () => void;
  onToggleExpand?: () => void;
};

type MapNode = Node<MapNodeData, "docPart">;

type TreeSelection =
  | { kind: "section"; sectionId: string }
  | { kind: "chunk"; sectionId: string; chunkId: string };

function layoutTree(
  root: EnrichedTreeNode,
  opts: {
    expandedIds: Set<string>;
    selected: TreeSelection | null;
    onSelect: (selection: TreeSelection) => void;
    onToggleExpand: (sectionId: string) => void;
  },
): { nodes: MapNode[]; edges: Edge[] } {
  const nodes: MapNode[] = [];
  const edges: Edge[] = [];
  let nextLeafX = PAD;
  const depthMaxHeight = new Map<number, number>();

  const selectedId =
    opts.selected?.kind === "chunk"
      ? `chunk:${opts.selected.chunkId}`
      : opts.selected?.kind === "section"
        ? opts.selected.sectionId
        : null;

  const measure = (node: EnrichedTreeNode, depth: number) => {
    const height = nodeHeight(node.kind);
    depthMaxHeight.set(
      depth,
      Math.max(depthMaxHeight.get(depth) ?? 0, height),
    );
    for (const child of node.children) measure(child, depth + 1);
  };
  measure(root, 0);

  const depthY = (depth: number): number => {
    let y = PAD;
    for (let d = 0; d < depth; d += 1) {
      y += (depthMaxHeight.get(d) ?? nodeHeight("section")) + GAP_Y;
    }
    return y;
  };

  const walk = (
    node: EnrichedTreeNode,
    parentId: string | null,
    depth: number,
  ): { x: number } => {
    const height = nodeHeight(node.kind);
    const visibleChildren = node.children;
    let x: number;

    if (visibleChildren.length === 0) {
      x = nextLeafX;
      nextLeafX += NODE_W + GAP_X;
    } else {
      const childXs = visibleChildren.map(
        (child) => walk(child, node.id, depth + 1).x,
      );
      x = (childXs[0] + childXs[childXs.length - 1]) / 2;
    }

    const isSection = node.kind === "section";
    const expanded = isSection && opts.expandedIds.has(node.id);
    const canExpand =
      isSection &&
      (node.childCount > 0 || (node.children?.length ?? 0) > 0);

    nodes.push({
      id: node.id,
      type: "docPart",
      position: {
        x,
        y: depthY(depth),
      },
      style: { width: NODE_W, height },
      data: {
        kind: node.kind,
        title: node.title,
        subtitle: node.subtitle,
        pageLabel: node.pageLabel,
        preview: node.preview,
        tableHeaders: node.tableHeaders,
        tableShape: node.tableShape,
        childCount: node.childCount,
        sectionId: node.sectionId,
        chunkId: node.chunkId,
        selected:
          selectedId === node.id ||
          (node.kind === "document" && !selectedId),
        hasChildren: canExpand || visibleChildren.length > 0,
        expanded: node.kind === "document" ? true : expanded,
        onSelect: () => {
          if (node.kind === "document") return;
          if (node.kind === "loading") return;
          if (node.chunkId && node.sectionId) {
            opts.onSelect({
              kind: "chunk",
              sectionId: node.sectionId,
              chunkId: node.chunkId,
            });
            return;
          }
          if (node.sectionId) {
            opts.onSelect({ kind: "section", sectionId: node.sectionId });
          }
        },
        onToggleExpand:
          canExpand && node.sectionId
            ? () => opts.onToggleExpand(node.sectionId!)
            : undefined,
      },
      draggable: false,
      selectable: node.kind !== "document" && node.kind !== "loading",
    });

    if (parentId) {
      edges.push({
        id: `${parentId}->${node.id}`,
        source: parentId,
        target: node.id,
        type: "smoothstep",
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 12,
          height: 12,
          color: "var(--gd-border)",
        },
        style: {
          stroke: "var(--gd-border)",
          strokeWidth: 1.4,
        },
      });
    }

    return { x };
  };

  walk(root, null, 0);
  return { nodes, edges };
}

function KindBadge({ kind }: { kind: TreeNodeKind }) {
  const label =
    kind === "document"
      ? "Document"
      : kind === "section"
        ? "Section"
        : kind === "paragraph"
          ? "Paragraph"
          : kind === "table"
            ? "Table"
            : "…";
  const tone =
    kind === "table"
      ? "border-[color-mix(in_oklab,var(--gd-accent)_35%,var(--gd-border))] text-[var(--gd-accent)]"
      : kind === "paragraph"
        ? "border-[var(--gd-border)] text-[var(--gd-muted)]"
        : kind === "document"
          ? "border-[var(--gd-accent)] text-[var(--gd-accent)]"
          : "border-[var(--gd-border)] text-[var(--gd-muted)]";
  return (
    <span
      className={cn(
        "rounded-md border px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-[0.06em]",
        tone,
      )}
    >
      {label}
    </span>
  );
}

function DocPartNodeView({ data }: NodeProps<MapNode>) {
  const isRoot = data.kind === "document";
  // Overview zoom: compact copy so Fit View stays readable without relayout.
  const { zoom } = useViewport();
  const compact = zoom < 0.55;

  return (
    <div
      className={cn(
        "h-full w-[268px] overflow-hidden rounded-xl border bg-[var(--gd-surface)] px-3.5 py-3 shadow-sm transition-colors",
        data.selected
          ? "border-[var(--gd-accent)] ring-2 ring-[var(--gd-accent-soft)]"
          : "border-[var(--gd-border)] hover:border-[var(--gd-muted)]",
        data.kind === "table" &&
          "bg-[color-mix(in_oklab,var(--gd-surface)_92%,var(--gd-accent)_8%)]",
        data.kind === "loading" && "opacity-80",
      )}
    >
      <Handle
        type="target"
        position={Position.Top}
        className="!h-2 !w-2 !border-0 !bg-[var(--gd-border)]"
      />
      <button
        type="button"
        className="flex h-full w-full cursor-pointer flex-col text-left"
        onClick={() => data.onSelect?.()}
        disabled={isRoot || data.kind === "loading"}
      >
        <div className="flex items-start justify-between gap-2">
          <KindBadge kind={data.kind} />
          {data.pageLabel ? (
            <span className="shrink-0 text-[10px] tabular-nums text-[var(--gd-muted)]">
              {data.pageLabel}
            </span>
          ) : null}
        </div>
        <p
          className={cn(
            "mt-1.5 text-[12px] leading-snug text-[var(--gd-text)]",
            compact ? "line-clamp-1" : "line-clamp-2",
            data.selected || isRoot ? "font-semibold" : "font-medium",
          )}
          title={data.title}
        >
          {data.title}
        </p>
        {!compact && data.subtitle ? (
          <p className="mt-1 line-clamp-1 text-[11px] text-[var(--gd-muted)]">
            {data.subtitle}
          </p>
        ) : null}
        {!compact && data.kind === "paragraph" && data.preview ? (
          <p className="mt-1.5 line-clamp-2 text-[11px] leading-snug text-[var(--gd-muted)]">
            {data.preview}
          </p>
        ) : null}
        {!compact && data.kind === "table" ? (
          <div className="mt-1.5 space-y-1">
            {data.tableHeaders && data.tableHeaders.length > 0 ? (
              <div className="flex flex-wrap gap-1">
                {data.tableHeaders.map((header, headerIdx) => (
                  <span
                    key={`${headerIdx}:${header}`}
                    className="max-w-[7.5rem] truncate rounded border border-[var(--gd-border)] px-1 py-0.5 text-[9px] text-[var(--gd-muted)]"
                    title={header}
                  >
                    {header}
                  </span>
                ))}
              </div>
            ) : data.preview ? (
              <pre className="line-clamp-2 overflow-hidden whitespace-pre-wrap font-[family-name:var(--font-geist-mono),ui-monospace,monospace] text-[9px] leading-snug text-[var(--gd-muted)]">
                {data.preview}
              </pre>
            ) : null}
          </div>
        ) : null}
        {!compact &&
        data.kind === "section" &&
        data.preview &&
        !data.subtitle ? (
          <p className="mt-1.5 line-clamp-2 text-[11px] text-[var(--gd-muted)]">
            {data.preview}
          </p>
        ) : null}
      </button>
      {data.hasChildren && data.onToggleExpand ? (
        <button
          type="button"
          className="mt-1.5 text-[10px] text-[var(--gd-accent)] hover:underline cursor-pointer"
          onClick={(e) => {
            e.stopPropagation();
            data.onToggleExpand?.();
          }}
        >
          {data.expanded
            ? "Collapse"
            : `Expand${data.childCount ? ` (${data.childCount})` : ""}`}
        </button>
      ) : null}
      <Handle
        type="source"
        position={Position.Bottom}
        className="!h-2 !w-2 !border-0 !bg-[var(--gd-border)]"
      />
    </div>
  );
}

const nodeTypes = { docPart: memo(DocPartNodeView) };

function HierarchyTreeMapCanvas({
  documentName,
  pageCount,
  partCount,
  sections,
  expandedIds,
  selectedSectionId,
  selectedChunkId,
  chunksBySection,
  loadingBySection,
  onSelectSection,
  onSelectChunk,
  onToggleExpand,
  onExpandAll,
  onCollapseAll,
}: {
  documentName: string;
  pageCount?: number | null;
  partCount: number;
  sections: SectionTreeNode[];
  expandedIds: Set<string>;
  selectedSectionId: string | null;
  selectedChunkId?: string | null;
  chunksBySection: SectionChunksMap;
  loadingBySection: SectionChunksLoading;
  onSelectSection: (sectionId: string) => void;
  onSelectChunk: (sectionId: string, chunkId: string) => void;
  onToggleExpand: (sectionId: string) => void;
  onExpandAll?: () => void;
  onCollapseAll?: () => void;
}) {
  const { fitView, getNodes, getViewport, setViewport, setCenter } =
    useReactFlow();
  const { resolvedTheme } = useTheme();
  const colorMode = resolvedTheme === "dark" ? "dark" : "light";

  const selection = useMemo<TreeSelection | null>(() => {
    if (!selectedSectionId) return null;
    if (selectedChunkId) {
      return {
        kind: "chunk",
        sectionId: selectedSectionId,
        chunkId: selectedChunkId,
      };
    }
    return { kind: "section", sectionId: selectedSectionId };
  }, [selectedSectionId, selectedChunkId]);

  const enrichedRoot = useMemo(
    () =>
      buildEnrichedTree({
        documentName,
        pageCount,
        sectionCount: partCount,
        sections,
        expandedIds,
        chunksBySection,
        loadingBySection,
      }),
    [
      documentName,
      pageCount,
      partCount,
      sections,
      expandedIds,
      chunksBySection,
      loadingBySection,
    ],
  );

  const visibleCount = useMemo(
    () => countVisibleNodes(enrichedRoot),
    [enrichedRoot],
  );

  const onSelect = useCallback(
    (next: TreeSelection) => {
      if (next.kind === "chunk") {
        onSelectChunk(next.sectionId, next.chunkId);
        return;
      }
      onSelectSection(next.sectionId);
    },
    [onSelectChunk, onSelectSection],
  );

  const graph = useMemo(
    () =>
      layoutTree(enrichedRoot, {
        expandedIds,
        selected: selection,
        onSelect,
        onToggleExpand,
      }),
    [enrichedRoot, expandedIds, selection, onSelect, onToggleExpand],
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(graph.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(graph.edges);
  const initialFitDoneRef = useRef(false);

  /** Readable opening zoom — matches the collapsed Tree screenshot (~0.62). */
  const INITIAL_ZOOM = 0.62;

  const centerOnRoot = useCallback(
    (zoom: number, duration = 220) => {
      const visible = typeof getNodes === "function" ? getNodes() : [];
      const root = visible.find((node) => node.id === ROOT_ID);
      if (root && typeof setCenter === "function") {
        void setCenter(
          root.position.x + NODE_W / 2,
          root.position.y + 120,
          { zoom, duration },
        );
        return;
      }
      if (typeof setViewport === "function") {
        void setViewport({ x: 0, y: 0, zoom }, { duration });
      }
    },
    [getNodes, setCenter, setViewport],
  );

  const fitReadable = useCallback(
    (mode: "overview" | "branch" | "initial" = "overview") => {
      const visible = typeof getNodes === "function" ? getNodes() : [];
      if (!visible.length) {
        void fitView({
          padding: 0.2,
          duration: 280,
          minZoom: INITIAL_ZOOM,
          maxZoom: INITIAL_ZOOM,
        });
        return;
      }

      if (mode === "initial") {
        // Collapsed opening view: keep labels readable; sides may clip (pan/minimap).
        centerOnRoot(INITIAL_ZOOM, 0);
        return;
      }

      if (mode === "branch" && selectedSectionId) {
        const focusNodes = visible.filter((node) => {
          if (node.id === ROOT_ID || node.id === selectedSectionId) return true;
          const data = node.data as MapNodeData;
          return data.sectionId === selectedSectionId;
        });
        void fitView({
          nodes: focusNodes.length ? focusNodes : visible,
          padding: 0.28,
          duration: 280,
          minZoom: 0.45,
          maxZoom: 1.25,
        });
        return;
      }

      // Explicit Fit view: try full visible graph, then enforce a readable floor.
      const padding = visibleCount > 60 ? 0.1 : visibleCount > 25 ? 0.14 : 0.2;
      const readableFloor = visibleCount > 30 ? 0.45 : 0.5;
      void fitView({
        nodes: visible,
        padding,
        duration: 280,
        minZoom: 0.08,
        maxZoom: visibleCount > 80 ? 0.95 : 1.2,
      }).then(() => {
        const vp =
          typeof getViewport === "function"
            ? getViewport()
            : { x: 0, y: 0, zoom: 1 };
        if (vp.zoom >= readableFloor) return;
        centerOnRoot(readableFloor, 220);
      });
    },
    [
      fitView,
      getNodes,
      getViewport,
      centerOnRoot,
      selectedSectionId,
      visibleCount,
    ],
  );

  useEffect(() => {
    setNodes(graph.nodes);
    setEdges(graph.edges);
  }, [graph.nodes, graph.edges, setNodes, setEdges]);

  // One-shot readable initial viewport for the collapsed tree.
  useEffect(() => {
    if (initialFitDoneRef.current) return;
    if (!graph.nodes.length) return;
    const timer = window.setTimeout(() => {
      fitReadable("initial");
      initialFitDoneRef.current = true;
    }, 50);
    return () => window.clearTimeout(timer);
  }, [graph.nodes.length, fitReadable]);

  useEffect(() => {
    const onResize = () => {
      if (!initialFitDoneRef.current) return;
      window.setTimeout(() => fitReadable("initial"), 120);
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [fitReadable]);

  const onNodeClick = useCallback(
    (_: ReactMouseEvent, node: Node) => {
      if (node.id === ROOT_ID) return;
      const data = node.data as MapNodeData;
      data.onSelect?.();
    },
    [],
  );

  if (!sections.length) {
    return (
      <p className="px-4 py-10 text-center text-sm text-[var(--gd-muted)]">
        No sections indexed for this document.
      </p>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--gd-rule)] px-4 py-2">
        <p className="text-[11px] text-[var(--gd-muted)]">
          Drag to pan · scroll to zoom · expand branches for paragraphs & tables
          · {visibleCount} visible nodes
        </p>
        <div className="flex flex-wrap items-center gap-3">
          {onExpandAll ? (
            <button
              type="button"
              className="text-[11px] text-[var(--gd-muted)] hover:text-[var(--gd-text)] cursor-pointer"
              onClick={onExpandAll}
            >
              Expand all
            </button>
          ) : null}
          {onCollapseAll ? (
            <button
              type="button"
              className="text-[11px] text-[var(--gd-muted)] hover:text-[var(--gd-text)] cursor-pointer"
              onClick={() => {
                onCollapseAll();
                window.setTimeout(() => fitReadable("initial"), 80);
              }}
            >
              Collapse all
            </button>
          ) : null}
          {selectedSectionId ? (
            <button
              type="button"
              className="text-[11px] text-[var(--gd-muted)] hover:text-[var(--gd-text)] cursor-pointer"
              onClick={() => fitReadable("branch")}
            >
              Focus branch
            </button>
          ) : null}
          <button
            type="button"
            className="text-[11px] text-[var(--gd-muted)] hover:text-[var(--gd-text)] cursor-pointer"
            onClick={() => fitReadable("overview")}
          >
            Fit view
          </button>
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-hidden bg-[var(--gd-ground)] text-[var(--gd-text)]">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          nodeTypes={nodeTypes}
          colorMode={colorMode}
          className="hierarchy-tree-map"
          style={FLOW_THEME_VARS}
          minZoom={0.05}
          maxZoom={1.9}
          defaultViewport={{ x: 0, y: 0, zoom: 0.62 }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable
          panOnDrag
          panOnScroll
          zoomOnScroll
          zoomOnPinch
          preventScrolling={false}
          proOptions={{ hideAttribution: true }}
          defaultEdgeOptions={{
            type: "smoothstep",
          }}
          onlyRenderVisibleElements={visibleCount > 120}
        >
          <Background
            gap={22}
            size={1}
            bgColor="var(--gd-ground)"
            color="color-mix(in oklab, var(--gd-border) 55%, transparent)"
          />
          <Controls
            showInteractive={false}
            position="bottom-left"
            className="!overflow-hidden !rounded-lg !border !border-[var(--gd-border)] !bg-[var(--gd-surface)] !shadow-sm"
          />
          <MiniMap
            pannable
            zoomable
            position="bottom-right"
            className="!overflow-hidden !rounded-lg !border !border-[var(--gd-border)] !bg-[var(--gd-surface)]"
            bgColor="var(--gd-surface)"
            maskColor="color-mix(in oklab, var(--gd-ground) 70%, transparent)"
            nodeColor={(node) => {
              const kind = (node.data as MapNodeData | undefined)?.kind;
              if (node.id === selectedSectionId || node.id === ROOT_ID) {
                return "var(--gd-accent)";
              }
              if (kind === "table") return "color-mix(in oklab, var(--gd-accent) 55%, var(--gd-raised))";
              if (kind === "paragraph") return "var(--gd-muted)";
              return "var(--gd-raised)";
            }}
            nodeStrokeColor="var(--gd-border)"
          />
        </ReactFlow>
      </div>
    </div>
  );
}

/** Visual document hierarchy map — recursive sections + content leaves. */
export function HierarchyTreeMap(props: {
  documentName: string;
  pageCount?: number | null;
  partCount: number;
  sections: SectionTreeNode[];
  expandedIds: Set<string>;
  selectedSectionId: string | null;
  selectedChunkId?: string | null;
  chunksBySection: SectionChunksMap;
  loadingBySection: SectionChunksLoading;
  onSelectSection: (sectionId: string) => void;
  onSelectChunk: (sectionId: string, chunkId: string) => void;
  onToggleExpand: (sectionId: string) => void;
  onExpandAll?: () => void;
  onCollapseAll?: () => void;
}) {
  return (
    <ReactFlowProvider>
      <HierarchyTreeMapCanvas {...props} />
    </ReactFlowProvider>
  );
}

"use client";

import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  type CSSProperties,
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
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useTheme } from "next-themes";
import { cn } from "@/lib/utils";
import {
  formatPageRange,
  sectionSignalChips,
} from "@/lib/document-hierarchy";
import type { SectionTreeNode } from "@/types/document-hierarchy";

/** Map xyflow chrome to Genova desk tokens (light + dark). */
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
const NODE_W = 260;
const NODE_H = 104;
/** Horizontal gap between sibling branches (vertical tree). */
const GAP_X = 28;
/** Vertical gap between depth levels. */
const GAP_Y = 56;
const PAD = 32;
/** Keep the map readable — never shrink below this on fit. */
const FIT_MIN_ZOOM = 0.55;
const FIT_MAX_ZOOM = 1.15;

type PartNodeData = {
  kind: "root" | "section";
  title: string;
  subtitle?: string;
  pageLabel?: string | null;
  partIndex?: number;
  chips?: string[];
  selected?: boolean;
  hasChildren?: boolean;
  expanded?: boolean;
  onSelect?: () => void;
  onToggleExpand?: () => void;
};

type PartNode = Node<PartNodeData, "part">;

/** Top-down tree: depth grows downward, siblings fan left→right. */
function buildGraph(opts: {
  documentName: string;
  partCount: number;
  sections: SectionTreeNode[];
  flatIndex: Map<string, number>;
  expandedIds: Set<string>;
  selectedSectionId: string | null;
  onSelectSection: (sectionId: string) => void;
  onToggleExpand: (sectionId: string) => void;
}): { nodes: PartNode[]; edges: Edge[] } {
  const {
    documentName,
    partCount,
    sections,
    flatIndex,
    expandedIds,
    selectedSectionId,
    onSelectSection,
    onToggleExpand,
  } = opts;

  const nodes: PartNode[] = [];
  const edges: Edge[] = [];
  let nextLeafX = PAD;

  const layoutSection = (
    node: SectionTreeNode,
    parentId: string,
    depth: number,
  ): { x: number } => {
    const children = node.children ?? [];
    const expanded = expandedIds.has(node.section_id);
    const visibleChildren =
      children.length && expanded ? children : ([] as SectionTreeNode[]);

    let x: number;
    if (visibleChildren.length === 0) {
      x = nextLeafX;
      nextLeafX += NODE_W + GAP_X;
    } else {
      const childXs = visibleChildren.map((child) =>
        layoutSection(child, node.section_id, depth + 1).x,
      );
      x = (childXs[0] + childXs[childXs.length - 1]) / 2;
    }

    const chips = sectionSignalChips(node).slice(0, 2);
    const partIndex = (flatIndex.get(node.section_id) ?? 0) + 1;

    nodes.push({
      id: node.section_id,
      type: "part",
      position: {
        x,
        y: PAD + depth * (NODE_H + GAP_Y),
      },
      data: {
        kind: "section",
        title: node.title || "Untitled section",
        pageLabel: formatPageRange(node.page_start, node.page_end),
        partIndex,
        chips,
        selected: selectedSectionId === node.section_id,
        hasChildren: children.length > 0,
        expanded,
        onSelect: () => onSelectSection(node.section_id),
        onToggleExpand: children.length
          ? () => onToggleExpand(node.section_id)
          : undefined,
      },
      draggable: false,
    });

    edges.push({
      id: `${parentId}->${node.section_id}`,
      source: parentId,
      target: node.section_id,
      type: "smoothstep",
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 14,
        height: 14,
        color: "var(--gd-border)",
      },
      style: {
        stroke: "var(--gd-border)",
        strokeWidth: 1.5,
      },
    });

    return { x };
  };

  const topXs: number[] = [];
  for (const section of sections) {
    topXs.push(layoutSection(section, ROOT_ID, 1).x);
  }

  const rootX =
    topXs.length > 0 ? (topXs[0] + topXs[topXs.length - 1]) / 2 : PAD;

  nodes.unshift({
    id: ROOT_ID,
    type: "part",
    position: { x: rootX, y: PAD },
    data: {
      kind: "root",
      title: documentName,
      subtitle: `${partCount} part${partCount === 1 ? "" : "s"}`,
      selected: !selectedSectionId,
      hasChildren: sections.length > 0,
      expanded: true,
      onSelect: undefined,
    },
    draggable: false,
    selectable: false,
  });

  return { nodes, edges };
}

function PartNodeView({ data }: NodeProps<PartNode>) {
  const isRoot = data.kind === "root";
  return (
    <div
      className={cn(
        "w-[260px] rounded-xl border bg-[var(--gd-surface)] px-3.5 py-3 shadow-sm transition-colors",
        data.selected
          ? "border-[var(--gd-accent)] ring-2 ring-[var(--gd-accent-soft)]"
          : "border-[var(--gd-border)] hover:border-[var(--gd-muted)]",
      )}
    >
      <Handle
        type="target"
        position={Position.Top}
        className="!h-2 !w-2 !border-0 !bg-[var(--gd-border)]"
      />
      <button
        type="button"
        className="w-full cursor-pointer text-left"
        onClick={() => data.onSelect?.()}
        disabled={isRoot}
      >
        <div className="flex items-start justify-between gap-2">
          <p
            className={cn(
              "line-clamp-2 text-[12px] leading-snug text-[var(--gd-text)]",
              data.selected || isRoot ? "font-semibold" : "font-medium",
            )}
            title={data.title}
          >
            {data.title}
          </p>
          {data.partIndex != null ? (
            <span className="shrink-0 text-[10px] tabular-nums text-[var(--gd-muted)]">
              Part {data.partIndex}
            </span>
          ) : null}
        </div>
        {data.subtitle ? (
          <p className="mt-1 text-[11px] text-[var(--gd-muted)]">{data.subtitle}</p>
        ) : null}
        {data.pageLabel ? (
          <p className="mt-1 text-[11px] tabular-nums text-[var(--gd-muted)]">
            {data.pageLabel}
          </p>
        ) : null}
        {data.chips && data.chips.length > 0 ? (
          <div className="mt-1.5 flex flex-wrap gap-1">
            {data.chips.map((chip, idx) => (
              <span
                key={`${chip}:${idx}`}
                className="max-w-full truncate rounded-full border border-[var(--gd-border)] px-1.5 py-0.5 text-[10px] text-[var(--gd-muted)]"
                title={chip}
              >
                {chip}
              </span>
            ))}
          </div>
        ) : null}
      </button>
      {data.hasChildren && data.onToggleExpand ? (
        <button
          type="button"
          className="mt-2 text-[10px] text-[var(--gd-accent)] hover:underline cursor-pointer"
          onClick={(e) => {
            e.stopPropagation();
            data.onToggleExpand?.();
          }}
        >
          {data.expanded ? "Collapse" : "Expand"}
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

const nodeTypes = { part: memo(PartNodeView) };

function HierarchyTreeMapCanvas({
  documentName,
  partCount,
  sections,
  flatSections,
  expandedIds,
  selectedSectionId,
  onSelectSection,
  onToggleExpand,
  onExpandAll,
  onCollapseAll,
}: {
  documentName: string;
  partCount: number;
  sections: SectionTreeNode[];
  flatSections: SectionTreeNode[];
  expandedIds: Set<string>;
  selectedSectionId: string | null;
  onSelectSection: (sectionId: string) => void;
  onToggleExpand: (sectionId: string) => void;
  onExpandAll?: () => void;
  onCollapseAll?: () => void;
}) {
  const { fitView, setViewport, getViewport } = useReactFlow();
  const { resolvedTheme } = useTheme();
  const colorMode = resolvedTheme === "dark" ? "dark" : "light";
  const flatIndex = useMemo(() => {
    const map = new Map<string, number>();
    flatSections.forEach((section, idx) => map.set(section.section_id, idx));
    return map;
  }, [flatSections]);

  const graph = useMemo(
    () =>
      buildGraph({
        documentName,
        partCount,
        sections,
        flatIndex,
        expandedIds,
        selectedSectionId,
        onSelectSection,
        onToggleExpand,
      }),
    [
      documentName,
      partCount,
      sections,
      flatIndex,
      expandedIds,
      selectedSectionId,
      onSelectSection,
      onToggleExpand,
    ],
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(graph.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(graph.edges);
  const fitKeyRef = useRef("");

  const fitReadable = useCallback(() => {
    void fitView({
      padding: 0.18,
      duration: 280,
      minZoom: FIT_MIN_ZOOM,
      maxZoom: FIT_MAX_ZOOM,
    }).then(() => {
      const vp = getViewport();
      if (vp.zoom < FIT_MIN_ZOOM) {
        void setViewport({ ...vp, zoom: FIT_MIN_ZOOM }, { duration: 180 });
      }
    });
  }, [fitView, getViewport, setViewport]);

  useEffect(() => {
    setNodes(graph.nodes);
    setEdges(graph.edges);
  }, [graph.nodes, graph.edges, setNodes, setEdges]);

  useEffect(() => {
    const key = `${expandedIds.size}:${selectedSectionId ?? ""}:${sections.length}`;
    if (fitKeyRef.current === key) return;
    fitKeyRef.current = key;
    const timer = window.setTimeout(() => {
      fitReadable();
    }, 60);
    return () => window.clearTimeout(timer);
  }, [expandedIds, selectedSectionId, sections.length, fitReadable]);

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      if (node.id === ROOT_ID) return;
      onSelectSection(node.id);
    },
    [onSelectSection],
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
      <div className="flex items-center justify-between gap-3 border-b border-[var(--gd-rule)] px-4 py-2">
        <p className="text-[11px] text-[var(--gd-muted)]">
          Drag to pan (x/y) · scroll to zoom · click a part for details
        </p>
        <div className="flex items-center gap-3">
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
              onClick={onCollapseAll}
            >
              Collapse all
            </button>
          ) : null}
          <button
            type="button"
            className="text-[11px] text-[var(--gd-muted)] hover:text-[var(--gd-text)] cursor-pointer"
            onClick={fitReadable}
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
          fitView
          fitViewOptions={{
            padding: 0.18,
            minZoom: FIT_MIN_ZOOM,
            maxZoom: FIT_MAX_ZOOM,
          }}
          minZoom={0.4}
          maxZoom={1.85}
          defaultViewport={{ x: 0, y: 0, zoom: 0.85 }}
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
            nodeColor={(node) =>
              node.id === selectedSectionId || node.id === ROOT_ID
                ? "var(--gd-accent)"
                : "var(--gd-raised)"
            }
            nodeStrokeColor="var(--gd-border)"
          />
        </ReactFlow>
      </div>
    </div>
  );
}

/** Visual document-part hierarchy map (Tree mode) — top-down sections only. */
export function HierarchyTreeMap(props: {
  documentName: string;
  partCount: number;
  sections: SectionTreeNode[];
  flatSections: SectionTreeNode[];
  expandedIds: Set<string>;
  selectedSectionId: string | null;
  onSelectSection: (sectionId: string) => void;
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

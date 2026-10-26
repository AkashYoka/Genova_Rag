"use client";

import { useMemo, type ComponentProps } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
// Chemical equation macros (\ce{...}, arrows, charges)
import "katex/contrib/mhchem";
import { prepareMarkdownForMath } from "@/lib/math-markdown";

export const markdownComponents: Components = {
  table: ({ children }) => (
    <div className="my-3 overflow-x-auto rounded-lg border border-border/70">
      <table className="min-w-full border-collapse text-left text-sm">
        {children}
      </table>
    </div>
  ),
  thead: ({ children }) => (
    <thead className="bg-muted/60 text-foreground">{children}</thead>
  ),
  th: ({ children }) => (
    <th className="border-b border-border/70 px-3 py-2 font-semibold">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border-b border-border/50 px-3 py-2 align-top">
      {children}
    </td>
  ),
  pre: ({ children }) => (
    <pre className="not-prose my-3 overflow-x-auto rounded-lg border border-border/70 bg-zinc-950 p-3 text-sm text-zinc-100">
      {children}
    </pre>
  ),
  code: ({ className, children, ...props }) => {
    const language = /language-(\w+)/.exec(className ?? "")?.[1];
    return (
      <code className={className} data-language={language} {...props}>
        {children}
      </code>
    );
  },
};

const remarkPlugins: NonNullable<
  ComponentProps<typeof ReactMarkdown>["remarkPlugins"]
> = [remarkGfm, remarkMath];

const rehypePlugins: NonNullable<
  ComponentProps<typeof ReactMarkdown>["rehypePlugins"]
> = [
  rehypeHighlight,
  [
    rehypeKatex,
    {
      throwOnError: false,
      strict: "ignore",
      output: "html",
      trust: false,
    },
  ],
];

interface MarkdownContentProps {
  content: string;
  /** When true, incomplete math delimiters are held back to avoid flicker. */
  isStreaming?: boolean;
  className?: string;
}

export default function MarkdownContent({
  content,
  isStreaming = false,
  className,
}: MarkdownContentProps) {
  const prepared = useMemo(
    () => prepareMarkdownForMath(content, { isStreaming }),
    [content, isStreaming],
  );

  return (
    <div className={className}>
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
        components={markdownComponents}
      >
        {prepared}
      </ReactMarkdown>
    </div>
  );
}

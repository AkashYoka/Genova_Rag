"use client";

import { useEffect, useState, type ReactNode } from "react";

/* ─────────────────────────────────────────────────────────
 * STREAMING TEXT (Beautiful UI)
 * Words resolve out of blur, inline citations appear in
 * context, then actions and follow-up prompts become usable.
 * ───────────────────────────────────────────────────────── */

const WORD_MS = 55;
const HOLD_MS = 3400;

/** one streamed word, or a `cite` placeholder that renders an inline source chip */
export type StreamingToken = { text: string; cite?: boolean };

const TOKENS: StreamingToken[] = [
  ..."Pistachio is your fastest-growing flavor — sales are up 23% this month and margins beat vanilla by 8 points."
    .split(" ")
    .map((text) => ({ text })),
  { text: "", cite: true },
  ..."Stone-fruit flavors are trending in the same range."
    .split(" ")
    .map((text) => ({ text })),
];

const FOLLOW_UPS = [
  "Which flavors sell best in winter",
  "Compare gelato and soft serve margins",
];

const SOURCE_IMAGES = {
  scoop:
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='16' fill='%231f7a5f'/%3E%3Cpath d='M20 36c0 7 5.4 12 12 12s12-5 12-12H20Z' fill='%23fff'/%3E%3Ccircle cx='32' cy='25' r='11' fill='%23bff3dd'/%3E%3Cpath d='M24 24c4-7 13-7 17 0' fill='none' stroke='%231f7a5f' stroke-width='4' stroke-linecap='round'/%3E%3C/svg%3E",
  trends:
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='16' fill='%232f6fec'/%3E%3Cpath d='M15 43 27 31l8 7 14-18' fill='none' stroke='%23fff' stroke-width='7' stroke-linecap='round' stroke-linejoin='round'/%3E%3Ccircle cx='49' cy='20' r='5' fill='%23bfe0ff'/%3E%3C/svg%3E",
  market:
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='16' fill='%23e56d24'/%3E%3Cpath d='M17 45V25h8v20h-8Zm11 0V16h8v29h-8Zm11 0V30h8v15h-8Z' fill='%23fff'/%3E%3Cpath d='M16 49h32' stroke='%23ffd6b8' stroke-width='4' stroke-linecap='round'/%3E%3C/svg%3E",
};

/** one cited source rendered as an inline chip and in the sources list */
export type StreamingSource = {
  name: string;
  domain: string;
  href: string;
  image: string;
};

const SOURCES: StreamingSource[] = [
  {
    name: "Scoop Data",
    domain: "scoopdata.io",
    href: "https://scoopdata.io/",
    image: SOURCE_IMAGES.scoop,
  },
  {
    name: "Trends Index",
    domain: "trends.google.com",
    href: "https://trends.google.com/trends/",
    image: SOURCE_IMAGES.trends,
  },
  {
    name: "Market Basket",
    domain: "marketbasket.io",
    href: "https://marketbasket.io/",
    image: SOURCE_IMAGES.market,
  },
];

function sourceImage(source: StreamingSource) {
  return source.image;
}

function SourceChip({ source }: { source?: StreamingSource }) {
  if (!source) return null;
  return (
    <a
      href={source.href}
      target="_blank"
      rel="noreferrer"
      className="ml-0 mr-1 inline-flex h-4.5 translate-y-[-1px] items-center gap-1 rounded-[5px] bg-muted pr-[3px] pl-[3px] align-middle font-mono text-[10.5px] text-muted-foreground shadow-[0_0_0_1px_var(--border)] transition-colors duration-150 hover:bg-accent hover:text-foreground"
      style={{
        animation: "streaming-pop-in 250ms cubic-bezier(0.23,1,0.32,1) both",
      }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={sourceImage(source)}
        alt=""
        className="streaming-source-avatar size-3 rounded-[3px]"
      />
      <span>{source.domain}</span>
    </a>
  );
}

const ACTION_ICONS: ReactNode[] = [
  <g key="copy">
    <rect x="9" y="9" width="12" height="12" rx="2.5" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </g>,
  <path key="retry" d="M21 12a9 9 0 1 1-2.64-6.36M21 3v6h-6" />,
  <path
    key="up"
    d="M7 10v12M15 5.88L14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88z"
  />,
  <path
    key="down"
    d="M17 14V2M9 18.12L10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88z"
  />,
];

export type StreamingLabels = {
  /** label on the collapsed sources toggle */
  sources: string;
  /** heading above the follow-up prompts */
  followUps: string;
};

const DEFAULT_LABELS: StreamingLabels = {
  sources: "10 sources",
  followUps: "Follow-ups",
};

/** Split plain text into word tokens for the animated reveal. */
export function textToStreamingTokens(text: string): StreamingToken[] {
  return text
    .split(/\s+/)
    .filter(Boolean)
    .map((word) => ({ text: word }));
}

export default function StreamingText({
  content = TOKENS,
  sources = SOURCES,
  followUps = FOLLOW_UPS,
  labels,
  loop = true,
  fill = false,
  liveText,
  streaming = false,
  showActions = true,
  showFollowUps = true,
  onDone,
  onFollowUp,
  onAction,
}: {
  variant?: string;
  /** the streamed tokens; `cite` tokens render an inline source chip */
  content?: StreamingToken[];
  /** cited sources shown in the chip, avatar stack, and expanded list */
  sources?: StreamingSource[];
  /** follow-up prompt suggestions shown once the stream completes */
  followUps?: string[];
  /** prominent copy strings */
  labels?: Partial<StreamingLabels>;
  /** restart the stream after a hold; turn off when embedding in a real thread */
  loop?: boolean;
  /** fill the parent width instead of the gallery's fixed measure */
  fill?: boolean;
  /**
   * Live SSE / WebSocket text. When set, words are shown as they arrive
   * (no artificial word timer). Prefer this in real chat threads.
   */
  liveText?: string;
  /** caret stays solid while true; actions/follow-ups wait until false */
  streaming?: boolean;
  /** hide the demo action/sources row (chat already has its own actions) */
  showActions?: boolean;
  showFollowUps?: boolean;
  onDone?: () => void;
  /** fired when a follow-up prompt is chosen */
  onFollowUp?: (text: string, index: number) => void;
  /** fired when a footer action is clicked (0 copy, 1 retry, 2 up, 3 down) */
  onAction?: (index: number) => void;
} = {}) {
  const isLive = typeof liveText === "string";
  const tokens = isLive ? textToStreamingTokens(liveText) : content;
  const l = {
    ...DEFAULT_LABELS,
    sources:
      sources.length > 0
        ? `${sources.length} source${sources.length === 1 ? "" : "s"}`
        : DEFAULT_LABELS.sources,
    ...labels,
  };
  const [count, setCount] = useState(isLive ? tokens.length : 0);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const done = isLive
    ? !streaming
    : count >= tokens.length && tokens.length > 0;

  useEffect(() => {
    if (isLive) {
      setCount(tokens.length);
      return;
    }
    if (done && !loop) {
      onDone?.();
      return;
    }
    const t = setTimeout(
      () => setCount((c) => (c >= tokens.length ? 0 : c + 1)),
      done ? HOLD_MS : WORD_MS,
    );
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [count, done, loop, isLive, tokens.length, liveText]);

  useEffect(() => {
    if (isLive && done) onDone?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLive, done]);

  const visible = isLive ? tokens : tokens.slice(0, count);
  const showChrome = done && !streaming;

  return (
    <div className={fill ? "w-full" : "min-h-[15.5rem] w-full max-w-95"}>
      <p className="text-[13px] leading-relaxed text-foreground whitespace-pre-wrap break-words">
        {isLive ? (
          <>
            {liveText}
            {streaming && (
              <span
                className="streaming-caret is-streaming ml-0.5 inline-block align-text-bottom"
                aria-hidden
              />
            )}
          </>
        ) : (
          <>
            {visible.map((token, i) =>
              token.cite ? (
                <SourceChip key={i} source={sources[0]} />
              ) : (
                <span key={i} className="inline">
                  {token.text}{" "}
                </span>
              ),
            )}
            {!done && (
              <span
                className="streaming-caret is-streaming ml-0.5 inline-block align-text-bottom"
                aria-hidden
              />
            )}
          </>
        )}
      </p>

      {showActions && (
        <>
          <div
            className="mt-2 flex items-center gap-0.5 transition-opacity duration-400"
            style={{
              opacity: showChrome ? 1 : 0,
              pointerEvents: showChrome ? "auto" : "none",
            }}
          >
            {ACTION_ICONS.map((icon, i) => (
              <button
                key={i}
                type="button"
                aria-label={
                  (["Copy", "Retry", "Thumbs up", "Thumbs down"] as const)[i]
                }
                onClick={() => onAction?.(i)}
                className="flex size-6 items-center justify-center rounded-[6px] text-muted-foreground/70 transition-colors duration-100 hover:bg-muted hover:text-muted-foreground"
              >
                <svg
                  width="15"
                  height="15"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  {icon}
                </svg>
              </button>
            ))}
            {sources.length > 0 && (
              <button
                type="button"
                aria-expanded={sourcesOpen}
                onClick={() => setSourcesOpen((current) => !current)}
                className="ml-1.5 flex items-center gap-1.5 rounded-[6px] px-1 py-0.5 text-left transition-colors duration-150 hover:bg-muted"
              >
                <span className="flex -space-x-1">
                  {sources.slice(0, 3).map((source) => (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      key={source.domain + source.name}
                      src={sourceImage(source)}
                      alt=""
                      className="streaming-source-avatar size-3.5 rounded-full bg-card shadow-[0_0_0_1.5px_var(--background)]"
                    />
                  ))}
                </span>
                <span className="text-[12px] text-muted-foreground">
                  {l.sources}
                </span>
              </button>
            )}
          </div>

          {sources.length > 0 && (
            <div
              className="grid transition-[grid-template-rows,opacity] duration-300"
              style={{
                gridTemplateRows: showChrome && sourcesOpen ? "1fr" : "0fr",
                opacity: showChrome && sourcesOpen ? 1 : 0,
                transitionTimingFunction: "cubic-bezier(0.23, 1, 0.32, 1)",
              }}
            >
              <div className="overflow-hidden">
                <div className="mt-1.5 flex flex-col rounded-[10px] bg-muted p-1 shadow-[0_0_0_1px_var(--border)]">
                  {sources.map((source) => (
                    <a
                      key={source.domain + source.name}
                      href={source.href}
                      target="_blank"
                      rel="noreferrer"
                      className="flex items-center gap-2 rounded-[6px] px-1.5 py-1 text-[12px] text-muted-foreground transition-colors duration-150 hover:bg-accent hover:text-foreground"
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={sourceImage(source)}
                        alt=""
                        className="streaming-source-avatar size-4 rounded-[4px]"
                      />
                      <span className="streaming-animated-underline">
                        {source.name}
                      </span>
                      <span className="ml-auto font-mono text-[10.5px] text-muted-foreground/70">
                        {source.domain}
                      </span>
                    </a>
                  ))}
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {showFollowUps && followUps.length > 0 && (
        <div
          className="mt-2.5 transition-opacity duration-400"
          style={{
            opacity: showChrome ? 1 : 0,
            pointerEvents: showChrome ? "auto" : "none",
          }}
        >
          <p className="text-[12px] font-medium text-muted-foreground">
            {l.followUps}
          </p>
          <div className="mt-0.5 flex flex-col">
            {followUps.map((text, i) => (
              <button
                key={text}
                type="button"
                onClick={() => onFollowUp?.(text, i)}
                className="-mx-1.5 flex items-center gap-2 rounded-[7px] border-b border-border px-1.5 py-1.5 text-left text-[12.5px] text-foreground transition-colors duration-100 hover:bg-muted"
                style={
                  showChrome
                    ? {
                        animation: `streaming-fade-up 350ms cubic-bezier(0.23,1,0.32,1) ${i * 90}ms both`,
                      }
                    : { opacity: 0 }
                }
              >
                <svg
                  width="11"
                  height="11"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="shrink-0 text-muted-foreground/70"
                >
                  <path d="M9 10l-5 5 5 5" />
                  <path d="M20 4v7a4 4 0 0 1-4 4H4" />
                </svg>
                {text}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

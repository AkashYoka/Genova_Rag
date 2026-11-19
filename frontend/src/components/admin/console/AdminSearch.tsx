"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { CornerDownLeft, FileText, Search, UserRound } from "lucide-react";
import { api } from "@/lib/api";
import { ADMIN_SECTIONS, type AdminSectionId } from "./AdminShell";
import { Modal } from "./ConsoleUI";
import "./console.css";

type Person = { id: string; username: string; email: string };
type Doc = { id: string; original_name: string; username?: string | null; status: string };

type Result =
  | { kind: "page"; id: AdminSectionId; label: string; hint: string; icon: ReactNode }
  | { kind: "person"; id: string; label: string; hint: string; icon: ReactNode }
  | { kind: "doc"; id: string; label: string; hint: string; icon: ReactNode };

export function AdminSearch({
  open,
  onClose,
  onNavigate,
}: {
  open: boolean;
  onClose: () => void;
  onNavigate: (id: AdminSectionId, query?: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [people, setPeople] = useState<Person[]>([]);
  const [docs, setDocs] = useState<Doc[]>([]);
  const [loading, setLoading] = useState(false);
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    setQuery("");
    setCursor(0);
    const t = window.setTimeout(() => inputRef.current?.focus(), 30);
    return () => window.clearTimeout(t);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    Promise.allSettled([
      api.get<Person[]>("/api/v1/admin/users"),
      api.get<{ items: Doc[] }>("/api/v1/admin/documents?limit=100"),
    ]).then(([usersRes, docsRes]) => {
      if (cancelled) return;
      if (usersRes.status === "fulfilled") setPeople(usersRes.value ?? []);
      if (docsRes.status === "fulfilled") setDocs(docsRes.value?.items ?? []);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [open]);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    const pages: Result[] = ADMIN_SECTIONS.filter((s) => {
      if (!q) return true;
      return (
        s.label.toLowerCase().includes(q) ||
        s.title.toLowerCase().includes(q) ||
        s.keywords.includes(q)
      );
    }).map((s) => ({
      kind: "page",
      id: s.id,
      label: s.label,
      hint: s.title,
      icon: s.icon,
    }));

    const peopleResults: Result[] = q
      ? people
          .filter(
            (p) =>
              p.username?.toLowerCase().includes(q) || p.email?.toLowerCase().includes(q),
          )
          .slice(0, 5)
          .map((p) => ({
            kind: "person",
            id: p.id,
            label: p.username || p.email,
            hint: p.email,
            icon: <UserRound />,
          }))
      : [];

    const docResults: Result[] = q
      ? docs
          .filter((d) => d.original_name?.toLowerCase().includes(q))
          .slice(0, 5)
          .map((d) => ({
            kind: "doc",
            id: d.id,
            label: d.original_name,
            hint: d.username ? `Uploaded by ${d.username}` : d.status,
            icon: <FileText />,
          }))
      : [];

    return { pages: pages.slice(0, q ? 6 : 10), peopleResults, docResults };
  }, [query, people, docs]);

  const flat = useMemo(
    () => [...results.pages, ...results.peopleResults, ...results.docResults],
    [results],
  );

  useEffect(() => {
    setCursor(0);
  }, [query]);

  const choose = (item: Result) => {
    if (item.kind === "person") onNavigate("users", item.label);
    else if (item.kind === "doc") onNavigate("documents", item.label);
    else onNavigate(item.id);
    onClose();
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Search admin"
      closeLabel="Close search"
      wide
    >
      <div style={{ margin: "-16px -18px" }}>
        <div className="adm-palette__search" style={{ borderBottom: "1px solid var(--adm-line)" }}>
          <Search aria-hidden />
          <input
            ref={inputRef}
            className="adm-palette__input"
            placeholder="Search pages, people, or documents…"
            value={query}
            aria-label="Search pages, people, or documents"
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") {
                e.preventDefault();
                setCursor((c) => Math.min(flat.length - 1, c + 1));
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setCursor((c) => Math.max(0, c - 1));
              } else if (e.key === "Enter" && flat[cursor]) {
                e.preventDefault();
                choose(flat[cursor]);
              }
            }}
          />
        </div>

        <div className="adm-palette__results">
          {results.pages.length > 0 ? (
            <>
              <div className="adm-palette__group">Pages</div>
              {results.pages.map((item, i) => (
                <button
                  key={`page-${item.id}`}
                  type="button"
                  className="adm-palette__item"
                  data-active={cursor === i}
                  onMouseEnter={() => setCursor(i)}
                  onClick={() => choose(item)}
                >
                  {item.icon}
                  <span>{item.label}</span>
                  <span className="adm-palette__hint">{item.hint}</span>
                </button>
              ))}
            </>
          ) : null}

          {results.peopleResults.length > 0 ? (
            <>
              <div className="adm-palette__group">People</div>
              {results.peopleResults.map((item, i) => (
                <button
                  key={`person-${item.id}`}
                  type="button"
                  className="adm-palette__item"
                  data-active={cursor === results.pages.length + i}
                  onMouseEnter={() => setCursor(results.pages.length + i)}
                  onClick={() => choose(item)}
                >
                  {item.icon}
                  <span>{item.label}</span>
                  <span className="adm-palette__hint">{item.hint}</span>
                </button>
              ))}
            </>
          ) : null}

          {results.docResults.length > 0 ? (
            <>
              <div className="adm-palette__group">Documents</div>
              {results.docResults.map((item, i) => (
                <button
                  key={`doc-${item.id}`}
                  type="button"
                  className="adm-palette__item"
                  data-active={
                    cursor === results.pages.length + results.peopleResults.length + i
                  }
                  onMouseEnter={() =>
                    setCursor(results.pages.length + results.peopleResults.length + i)
                  }
                  onClick={() => choose(item)}
                >
                  {item.icon}
                  <span className="adm-truncate" style={{ maxWidth: 320 }}>
                    {item.label}
                  </span>
                  <span className="adm-palette__hint">{item.hint}</span>
                </button>
              ))}
            </>
          ) : null}

          {flat.length === 0 ? (
            <div className="adm-palette__empty">
              {loading
                ? "Searching…"
                : `No results for “${query}”. Try a person's name, a file name, or a page.`}
            </div>
          ) : null}
        </div>

        <div className="adm-palette__foot">
          <span>
            <CornerDownLeft style={{ width: 10, height: 10, verticalAlign: "-1px" }} /> to open
          </span>
          <span>↑ ↓ to move</span>
          <span>Esc to close</span>
        </div>
      </div>
    </Modal>
  );
}

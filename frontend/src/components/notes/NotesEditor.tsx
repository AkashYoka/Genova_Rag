"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CKEditor } from "@ckeditor/ckeditor5-react";
import {
  Bold,
  ClassicEditor,
  Essentials,
  Heading,
  Italic,
  List,
  ModelDocumentFragment,
  Paragraph,
  Undo,
  BlockQuote,
  Link,
  Indent,
} from "ckeditor5";
import type { EditorConfig } from "ckeditor5";
import "ckeditor5/ckeditor5.css";
import { messageToNoteHtml, type GenovaNotePayload } from "@/lib/notes-mime";
import {
  createNoteDocument,
  normalizeNoteDocument,
  type NoteDocument,
  type NoteModelNode,
} from "@/lib/notes-document";
import type { CkEditorDocument } from "@/store/notes-store";

/** The subset of CKEditor's model node API used for serialization. */
interface EditorNodeLike {
  is: (name: string) => boolean;
  name?: string;
  data?: string;
  getAttributes: () => Iterable<[string, unknown]>;
  getChildren: () => Iterable<EditorNodeLike>;
}

function serializeNode(node: EditorNodeLike): NoteModelNode {
  const attributes = Object.fromEntries(node.getAttributes());
  const serialized: NoteModelNode = node.is("text")
    ? { data: node.data ?? "" }
    : {
        name: node.name ?? "paragraph",
        children: Array.from(node.getChildren(), serializeNode),
      };

  if (Object.keys(attributes).length > 0) {
    serialized.attributes = attributes;
  }
  return serialized;
}

/** Serialize the editor model into the persisted structured document. */
function readNoteDocument(editor: ClassicEditor): CkEditorDocument {
  const root = editor.model.document.getRoot();
  if (!root) return createNoteDocument([]);

  const children = root.getChildren() as unknown as Iterable<EditorNodeLike>;
  return createNoteDocument(Array.from(children, serializeNode));
}

/** Replace editor content with a structured document, preserving formatting. */
function applyNoteDocument(editor: ClassicEditor, document: NoteDocument): void {
  const root = editor.model.document.getRoot();
  if (!root) return;

  const fragment = ModelDocumentFragment.fromJSON(document.main);

  editor.model.change((writer) => {
    writer.remove(writer.createRangeIn(root));
    writer.insert(fragment, root, "end");
  });
}

interface NotesEditorProps {
  /** CKEditor JSON loaded from the user's S3 notes file. */
  document: CkEditorDocument | null;
  /** Legacy local notes that only exist as HTML (migrated on first edit). */
  legacyHtml?: string;
  /** Chat messages queued for insertion. */
  pendingAppends: GenovaNotePayload[];
  /** Bumped to request an empty document. */
  clearToken: number;
  /** User edit — the store persists this JSON. */
  onDocumentChange: (document: CkEditorDocument, html: string) => void;
  /** Initial/programmatic sync — persisted only when migrating legacy notes. */
  onDocumentReady: (document: CkEditorDocument, html: string) => void;
  onAppendsHandled: () => void;
}

export default function NotesEditor({
  document,
  legacyHtml = "",
  pendingAppends,
  clearToken,
  onDocumentChange,
  onDocumentReady,
  onAppendsHandled,
}: NotesEditorProps) {
  const editorRef = useRef<ClassicEditor | null>(null);
  const skipNextChange = useRef(false);
  const appliedInitial = useRef(false);
  const appliedClearToken = useRef(clearToken);
  const seededDocument = useRef<CkEditorDocument | null>(null);
  const [ready, setReady] = useState(false);

  // Capture the initial payloads only — later arrivals are handled below.
  // A lazy initial state keeps this out of render-time ref reads.
  const [initialPayload] = useState(() => ({ document, html: legacyHtml }));
  const callbacks = useRef({ onDocumentChange, onDocumentReady, onAppendsHandled });

  useEffect(() => {
    callbacks.current = { onDocumentChange, onDocumentReady, onAppendsHandled };
  });

  const config = useMemo(
    (): EditorConfig => ({
      licenseKey: "GPL",
      plugins: [
        Essentials,
        Paragraph,
        Heading,
        Bold,
        Italic,
        List,
        BlockQuote,
        Link,
        Indent,
        Undo,
      ],
      toolbar: [
        "undo",
        "redo",
        "|",
        "heading",
        "|",
        "bold",
        "italic",
        "link",
        "|",
        "bulletedList",
        "numberedList",
        "outdent",
        "indent",
        "|",
        "blockQuote",
      ],
      placeholder: "Drop chat messages here, or refine notes…",
    }),
    [],
  );

  const applyData = useCallback((editor: ClassicEditor, data: unknown): CkEditorDocument => {
    skipNextChange.current = true;

    const { document: stored, html } = normalizeNoteDocument(data);
    if (stored) {
      try {
        applyNoteDocument(editor, stored);
      } catch {
        // Unsupported or corrupt content: start empty instead of leaving the
        // editor in a partially-applied state.
        editor.data.set("");
      }
    } else {
      editor.data.set(html ?? "");
    }

    const json = readNoteDocument(editor);
    seededDocument.current = json;
    return json;
  }, []);

  // A saved document arriving after mount (login restore) replaces the editor.
  useEffect(() => {
    const editor = editorRef.current;
    if (!ready || !editor || !document) return;
    if (seededDocument.current === document) return;
    const json = applyData(editor, document);
    callbacks.current.onDocumentReady(json, editor.getData());
  }, [ready, document, applyData]);

  // Append queued chat messages (Add to notes / drag & drop).
  useEffect(() => {
    const editor = editorRef.current;
    if (!ready || !editor || pendingAppends.length === 0) return;

    let html = editor.getData();
    for (const payload of pendingAppends) {
      html += messageToNoteHtml(payload);
    }
    const json = applyData(editor, html);
    callbacks.current.onDocumentChange(json, editor.getData());
    callbacks.current.onAppendsHandled();
  }, [ready, pendingAppends, applyData]);

  // Clear request from the toolbar.
  useEffect(() => {
    const editor = editorRef.current;
    if (!ready || !editor || clearToken === appliedClearToken.current) return;
    appliedClearToken.current = clearToken;
    const json = applyData(editor, "");
    callbacks.current.onDocumentChange(json, editor.getData());
  }, [ready, clearToken, applyData]);

  return (
    <div className="genova-notes-editor min-h-0 flex-1 overflow-hidden">
      <CKEditor
        editor={ClassicEditor}
        config={config}
        data={initialPayload.document ? undefined : initialPayload.html}
        onReady={(editor) => {
          editorRef.current = editor;
          setReady(true);
          if (appliedInitial.current) return;
          appliedInitial.current = true;

          const seed = initialPayload.document ?? initialPayload.html;
          const json = seed
            ? applyData(editor, seed)
            : readNoteDocument(editor);
          seededDocument.current = json;
          callbacks.current.onDocumentReady(json, editor.getData());
        }}
        onChange={(_event, editor) => {
          if (skipNextChange.current) {
            skipNextChange.current = false;
            return;
          }
          const json = readNoteDocument(editor);
          seededDocument.current = json;
          callbacks.current.onDocumentChange(json, editor.getData());
        }}
      />
    </div>
  );
}

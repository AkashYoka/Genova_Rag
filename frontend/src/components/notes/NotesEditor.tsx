"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CKEditor } from "@ckeditor/ckeditor5-react";
import {
  Alignment,
  BlockQuote,
  Bold,
  ClassicEditor,
  Code,
  CodeBlock,
  Essentials,
  Heading,
  HorizontalLine,
  Indent,
  Italic,
  Link,
  List,
  ModelDocumentFragment,
  Paragraph,
  Strikethrough,
  Subscript,
  Superscript,
  Table,
  TableCellProperties,
  TableProperties,
  TableToolbar,
  Undo,
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
  /** Active note id — switching tabs applies that note's content. */
  noteId: string;
  /** CKEditor JSON loaded from the user's S3 notes file. */
  document: CkEditorDocument | null;
  /** HTML mirror, used for exports and as the restore fallback. */
  html?: string;
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
  noteId,
  document,
  html = "",
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
  const appliedNoteId = useRef(noteId);
  const appliedDocument = useRef<CkEditorDocument | null>(document);
  const [ready, setReady] = useState(false);

  // Capture the initial payloads only — later arrivals are handled below.
  // A lazy initial state keeps this out of render-time ref reads.
  const [initialPayload] = useState(() => ({ document, html }));
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
        Strikethrough,
        Code,
        Subscript,
        Superscript,
        List,
        BlockQuote,
        Link,
        Indent,
        Alignment,
        HorizontalLine,
        CodeBlock,
        Table,
        TableToolbar,
        TableProperties,
        TableCellProperties,
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
        "strikethrough",
        "code",
        "|",
        "link",
        "|",
        "bulletedList",
        "numberedList",
        "outdent",
        "indent",
        "|",
        "blockQuote",
        "insertTable",
        "horizontalLine",
      ],
      heading: {
        options: [
          { model: "paragraph", title: "Paragraph", class: "ck-heading_paragraph" },
          { model: "heading1", view: "h1", title: "Heading 1", class: "ck-heading_heading1" },
          { model: "heading2", view: "h2", title: "Heading 2", class: "ck-heading_heading2" },
          { model: "heading3", view: "h3", title: "Heading 3", class: "ck-heading_heading3" },
          { model: "heading4", view: "h4", title: "Heading 4", class: "ck-heading_heading4" },
          { model: "heading5", view: "h5", title: "Heading 5", class: "ck-heading_heading5" },
          { model: "heading6", view: "h6", title: "Heading 6", class: "ck-heading_heading6" },
        ],
      },
      table: {
        contentToolbar: [
          "tableColumn",
          "tableRow",
          "mergeTableCells",
          "tableProperties",
          "tableCellProperties",
        ],
      },
      placeholder: "Drop chat messages here, or refine notes…",
    }),
    [],
  );

  const applyData = useCallback(
    (editor: ClassicEditor, data: unknown, htmlFallback = ""): CkEditorDocument => {
      skipNextChange.current = true;

      const { document: stored, html: storedHtml } = normalizeNoteDocument(data);
      const fallback = storedHtml ?? htmlFallback;
      if (stored) {
        try {
          applyNoteDocument(editor, stored);
        } catch (error) {
          // Unsupported or corrupt model JSON: fall back to the HTML mirror so
          // the note is still readable instead of silently opening empty.
          console.warn(
            "[notes] Could not restore the structured note; using its HTML mirror.",
            error,
          );
          editor.data.set(fallback);
        }
      } else {
        editor.data.set(fallback);
      }

      const json = readNoteDocument(editor);
      return json;
    },
    [],
  );

  // A saved document arriving after mount (login restore) or a different note
  // selected in the notes panel replaces the editor content.
  useEffect(() => {
    const editor = editorRef.current;
    if (!ready || !editor) return;

    const noteChanged = appliedNoteId.current !== noteId;
    const documentChanged = appliedDocument.current !== document;
    if (!noteChanged && !documentChanged) return;

    appliedNoteId.current = noteId;
    const json = applyData(editor, document ?? html, html);
    appliedDocument.current = json;
    callbacks.current.onDocumentReady(json, editor.getData());
  }, [ready, noteId, document, html, applyData]);

  // Append queued chat messages (Add to notes / drag & drop).
  useEffect(() => {
    const editor = editorRef.current;
    if (!ready || !editor || pendingAppends.length === 0) return;

    let html = editor.getData();
    for (const payload of pendingAppends) {
      html += messageToNoteHtml(payload);
    }
    const json = applyData(editor, html);
    appliedDocument.current = json;
    callbacks.current.onDocumentChange(json, editor.getData());
    callbacks.current.onAppendsHandled();
  }, [ready, pendingAppends, applyData]);

  // Clear request from the toolbar.
  useEffect(() => {
    const editor = editorRef.current;
    if (!ready || !editor || clearToken === appliedClearToken.current) return;
    appliedClearToken.current = clearToken;
    const json = applyData(editor, "");
    appliedDocument.current = json;
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
            ? applyData(editor, seed, initialPayload.html)
            : readNoteDocument(editor);
          appliedNoteId.current = noteId;
          appliedDocument.current = json;
          callbacks.current.onDocumentReady(json, editor.getData());
        }}
        onChange={(_event, editor) => {
          if (skipNextChange.current) {
            skipNextChange.current = false;
            return;
          }
          const json = readNoteDocument(editor);
          // Mark the new object as applied so the restore effect does not
          // re-apply the content the user just typed.
          appliedDocument.current = json;
          callbacks.current.onDocumentChange(json, editor.getData());
        }}
      />
    </div>
  );
}

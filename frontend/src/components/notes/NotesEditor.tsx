"use client";

import { useEffect, useMemo, useRef } from "react";
import { CKEditor } from "@ckeditor/ckeditor5-react";
import {
  Bold,
  ClassicEditor,
  Essentials,
  Heading,
  Italic,
  List,
  Paragraph,
  Undo,
  BlockQuote,
  Link,
  Indent,
} from "ckeditor5";
import type { EditorConfig } from "ckeditor5";
import "ckeditor5/ckeditor5.css";

interface NotesEditorProps {
  /** Shared notes HTML (persisted across chat sessions) */
  html: string;
  onChange: (html: string) => void;
}

export default function NotesEditor({ html, onChange }: NotesEditorProps) {
  const editorRef = useRef<ClassicEditor | null>(null);
  const skipNextChange = useRef(false);
  const initialHtml = useRef(html);

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

  // External updates (append from Chat 1 / Chat 2) → push into the editor
  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    if (editor.getData() === html) return;
    skipNextChange.current = true;
    editor.setData(html);
  }, [html]);

  return (
    <div className="genova-notes-editor min-h-0 flex-1 overflow-hidden">
      <CKEditor
        editor={ClassicEditor}
        config={config}
        data={initialHtml.current}
        onReady={(editor) => {
          editorRef.current = editor;
          if (html && editor.getData() !== html) {
            skipNextChange.current = true;
            editor.setData(html);
          }
        }}
        onChange={(_event, editor) => {
          if (skipNextChange.current) {
            skipNextChange.current = false;
            return;
          }
          onChange(editor.getData());
        }}
      />
    </div>
  );
}

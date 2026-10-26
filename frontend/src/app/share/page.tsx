"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Brain } from "lucide-react";
import { api } from "@/lib/api";
import MarkdownContent from "@/components/chat/MarkdownContent";

interface SharedSource {
  text: string;
  filename: string;
  page: number;
  score: number;
  confidence: number;
}

interface SharedAnswer {
  id: string;
  content: string;
  created_at: string;
  sources: SharedSource[];
}

function ShareAnswerContent() {
  const searchParams = useSearchParams();
  const messageId = searchParams.get("message_id");
  const missingMessageId = !messageId;
  const [answer, setAnswer] = useState<SharedAnswer | null>(null);
  const [error, setError] = useState("");
  const loading = !error && !answer && !missingMessageId;

  useEffect(() => {
    if (missingMessageId) {
      return;
    }

    let cancelled = false;

    void api
      .get<SharedAnswer>(`/api/v1/chat/share/${messageId}`)
      .then((data) => {
        if (cancelled) return;
        setAnswer(data);
        setError("");
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setAnswer(null);
        setError(err instanceof Error ? err.message : "Shared answer not found");
      });

    return () => {
      cancelled = true;
    };
  }, [messageId, missingMessageId]);

  if (missingMessageId) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4">
        <div className="rounded-xl border border-border/50 bg-card/80 px-6 py-5 text-center">
          <p className="text-lg font-semibold mb-1">Shared answer unavailable</p>
          <p className="text-sm text-muted-foreground">This shared answer could not be found.</p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4">
        <div className="text-sm text-muted-foreground">Loading shared answer...</div>
      </div>
    );
  }

  if (error || !answer) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4">
        <div className="rounded-xl border border-border/50 bg-card/80 px-6 py-5 text-center">
          <p className="text-lg font-semibold mb-1">Shared answer unavailable</p>
          <p className="text-sm text-muted-foreground">{error || "This shared answer could not be found."}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen px-4 py-10 bg-background text-foreground">
      <div className="max-w-3xl mx-auto">
        <div className="rounded-2xl border border-border/50 bg-card/80 backdrop-blur-sm p-6">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-10 h-10 rounded-xl bg-primary/15 flex items-center justify-center">
              <Brain className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="text-xl font-semibold">Shared AI Answer</h1>
              <p className="text-sm text-muted-foreground">
                {new Date(answer.created_at).toLocaleString()}
              </p>
            </div>
          </div>

          <MarkdownContent
            content={answer.content}
            className="prose-chat text-sm"
          />

          {answer.sources.length > 0 && (
            <div className="mt-6 border-t border-border/50 pt-4">
              <h2 className="text-sm font-semibold mb-3">Sources</h2>
              <div className="space-y-2">
                {answer.sources.map((source, index) => (
                  <div
                    key={`${answer.id}-${index}`}
                    className="rounded-lg border border-border/50 bg-background/60 p-3"
                  >
                    <p className="text-xs font-medium mb-1">
                      {source.filename} • Page {source.page}
                    </p>
                    <p className="text-xs text-muted-foreground">{source.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function ShareAnswerPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center px-4">
          <div className="text-sm text-muted-foreground">Loading shared answer...</div>
        </div>
      }
    >
      <ShareAnswerContent />
    </Suspense>
  );
}

"use client";

import { useCallback, useMemo, useState } from "react";
import { Check, ChevronLeft, ChevronRight, CircleHelp, RotateCcw, X } from "lucide-react";
import { api } from "@/lib/api";
import type { QuizPayload, QuizQuestion } from "@/store/chat-store";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface Props {
  quiz: QuizPayload;
  messageId: string;
  onQuizUpdate?: (quiz: QuizPayload) => void;
}

const modeLabel = (mode: string) => {
  switch (mode) {
    case "conversation":
      return "From conversation";
    case "document":
      return "From document";
    case "summary":
      return "After summary";
    default:
      return "Quiz";
  }
};

function scoreQuiz(quiz: QuizPayload): { correct: number; total: number } {
  const answers = quiz.state?.answers ?? {};
  let correct = 0;
  for (const q of quiz.questions) {
    if (answers[q.id] && answers[q.id] === q.correct_option_id) correct += 1;
  }
  return { correct, total: quiz.questions.length };
}

export default function QuizCard({ quiz, messageId, onQuizUpdate }: Props) {
  const questions = quiz.questions ?? [];
  const [localQuiz, setLocalQuiz] = useState<QuizPayload>(() => ({
    ...quiz,
    state: {
      current_index: quiz.state?.current_index ?? 0,
      answers: { ...(quiz.state?.answers ?? {}) },
      completed: quiz.state?.completed ?? false,
    },
  }));
  const [persistError, setPersistError] = useState<string | null>(null);

  const state = localQuiz.state ?? {
    current_index: 0,
    answers: {},
    completed: false,
  };
  const index = Math.min(
    Math.max(state.current_index, 0),
    Math.max(questions.length - 1, 0),
  );
  const current: QuizQuestion | undefined = questions[index];
  const selected = current ? state.answers[current.id] : undefined;
  const revealed = Boolean(selected);
  const { correct, total } = useMemo(() => scoreQuiz(localQuiz), [localQuiz]);
  const progressPct = total === 0 ? 0 : Math.round(((index + (revealed ? 1 : 0)) / total) * 100);

  const persistState = useCallback(
    async (next: QuizPayload) => {
      // Client-generated streaming ids are not durable until history reload.
      if (!messageId || messageId.startsWith("assistant-")) return;
      try {
        setPersistError(null);
        await api.patch(`/api/v1/chat/messages/${encodeURIComponent(messageId)}/quiz-state`, {
          current_index: next.state?.current_index ?? 0,
          answers: next.state?.answers ?? {},
          completed: next.state?.completed ?? false,
        });
      } catch (err) {
        setPersistError(
          err instanceof Error ? err.message : "Could not save quiz progress",
        );
      }
    },
    [messageId],
  );

  const commit = useCallback(
    (next: QuizPayload) => {
      setLocalQuiz(next);
      onQuizUpdate?.(next);
      void persistState(next);
    },
    [onQuizUpdate, persistState],
  );

  if (!current || questions.length === 0) return null;

  const selectOption = (optionId: string) => {
    if (state.answers[current.id]) return;
    const answers = { ...state.answers, [current.id]: optionId };
    const answeredCount = Object.keys(answers).length;
    commit({
      ...localQuiz,
      state: {
        current_index: index,
        answers,
        completed: answeredCount >= questions.length,
      },
    });
  };

  const goNext = () => {
    if (index >= questions.length - 1) {
      commit({
        ...localQuiz,
        state: { ...state, current_index: index, completed: true },
      });
      return;
    }
    commit({
      ...localQuiz,
      state: { ...state, current_index: index + 1, completed: false },
    });
  };

  const goPrev = () => {
    if (index <= 0) return;
    commit({
      ...localQuiz,
      state: { ...state, current_index: index - 1, completed: false },
    });
  };

  const restart = () => {
    commit({
      ...localQuiz,
      state: { current_index: 0, answers: {}, completed: false },
    });
  };

  if (state.completed && Object.keys(state.answers).length >= questions.length) {
    return (
      <div className="rounded-2xl border border-primary/15 bg-card/70 overflow-hidden shadow-[0_12px_40px_-28px_oklch(0.5_0.08_195/0.55)]">
        <div className="px-4 py-3 border-b border-border/30 flex items-center justify-between gap-2">
          <div>
            <p className="text-sm font-semibold tracking-tight">{localQuiz.title}</p>
            <p className="text-[11px] text-muted-foreground">{modeLabel(localQuiz.mode)}</p>
          </div>
          <Badge variant="outline" className="text-[10px]">
            Results
          </Badge>
        </div>
        <div className="px-4 py-5 space-y-3">
          <p className="text-2xl font-semibold tabular-nums">
            {correct}
            <span className="text-muted-foreground text-base font-medium"> / {total}</span>
          </p>
          <p className="text-sm text-muted-foreground">
            {correct === total
              ? "Perfect score — every answer matched the source material."
              : correct === 0
                ? "No correct answers this round. Review the explanations and try again."
                : `You answered ${correct} of ${total} correctly.`}
          </p>
          <div className="space-y-2 pt-1">
            {questions.map((q, i) => {
              const chosen = state.answers[q.id];
              const ok = chosen === q.correct_option_id;
              return (
                <div
                  key={q.id}
                  className="rounded-xl border border-border/40 px-3 py-2 text-[12px]"
                >
                  <div className="flex items-start gap-2">
                    {ok ? (
                      <Check className="w-3.5 h-3.5 text-emerald-600 mt-0.5 shrink-0" />
                    ) : (
                      <X className="w-3.5 h-3.5 text-red-500 mt-0.5 shrink-0" />
                    )}
                    <div className="min-w-0">
                      <p className="font-medium text-foreground/90">
                        {i + 1}. {q.question}
                      </p>
                      <p className="text-muted-foreground mt-0.5">
                        Correct:{" "}
                        {q.options.find((o) => o.id === q.correct_option_id)?.text ??
                          q.correct_option_id}
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="mt-2"
            onClick={restart}
          >
            <RotateCcw className="w-3.5 h-3.5 mr-1.5" />
            Retake quiz
          </Button>
          {persistError && (
            <p className="text-[11px] text-amber-600">{persistError}</p>
          )}
        </div>
      </div>
    );
  }

  const sourceLabel = current.source?.filename
    ? `${current.source.filename}${
        typeof current.source.page === "number" ? `, p. ${current.source.page + 1}` : ""
      }`
    : null;

  return (
    <div className="rounded-2xl border border-primary/15 bg-card/70 overflow-hidden shadow-[0_12px_40px_-28px_oklch(0.5_0.08_195/0.55)]">
      <div className="px-4 py-3 border-b border-border/30 flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-semibold tracking-tight truncate">
            {localQuiz.title}
          </p>
          <p className="text-[11px] text-muted-foreground">
            {modeLabel(localQuiz.mode)} · Question {index + 1} of {total}
          </p>
        </div>
        <Badge variant="outline" className="text-[10px] shrink-0 tabular-nums">
          Score {correct}/{Object.keys(state.answers).length || 0}
        </Badge>
      </div>

      <div className="h-1 bg-muted">
        <div
          className="h-full bg-primary transition-all duration-300"
          style={{ width: `${Math.min(100, Math.max(progressPct, (index / Math.max(total, 1)) * 100))}%` }}
        />
      </div>

      <div className="px-4 py-4 space-y-3">
        <p className="text-sm font-medium leading-relaxed">{current.question}</p>

        <div className="space-y-2" role="listbox" aria-label="Answer options">
          {current.options.map((opt) => {
            const isSelected = selected === opt.id;
            const isCorrect = opt.id === current.correct_option_id;
            let tone =
              "border-border/50 hover:border-primary/40 hover:bg-accent/30 cursor-pointer";
            if (revealed) {
              if (isCorrect) {
                tone = "border-emerald-500/40 bg-emerald-500/10";
              } else if (isSelected) {
                tone = "border-red-500/40 bg-red-500/10";
              } else {
                tone = "border-border/30 opacity-70";
              }
            } else if (isSelected) {
              tone = "border-primary/50 bg-primary/10";
            }

            return (
              <button
                key={opt.id}
                type="button"
                role="option"
                aria-selected={isSelected}
                aria-label={`Option ${opt.id}: ${opt.text}`}
                disabled={revealed}
                onClick={() => selectOption(opt.id)}
                className={cn(
                  "w-full text-left rounded-xl border px-3 py-2.5 text-[13px] transition-colors flex items-start gap-2.5",
                  tone,
                  revealed && "cursor-default",
                )}
              >
                <span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-md border border-border/60 text-[10px] font-semibold uppercase text-muted-foreground">
                  {opt.id}
                </span>
                <span className="flex-1 leading-snug">{opt.text}</span>
                {revealed && isCorrect && (
                  <Check className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                )}
                {revealed && isSelected && !isCorrect && (
                  <X className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
                )}
              </button>
            );
          })}
        </div>

        {revealed && (
          <div
            className={cn(
              "rounded-xl px-3 py-2.5 text-[12px] leading-relaxed border",
              selected === current.correct_option_id
                ? "border-emerald-500/25 bg-emerald-500/5 text-emerald-900 dark:text-emerald-100"
                : "border-red-500/25 bg-red-500/5 text-red-900 dark:text-red-100",
            )}
          >
            <p className="font-medium mb-1 flex items-center gap-1.5">
              <CircleHelp className="w-3.5 h-3.5" />
              {selected === current.correct_option_id ? "Correct" : "Incorrect"}
            </p>
            <p className="text-foreground/80">
              {current.explanation ||
                "Review the cited source material for the supporting detail."}
            </p>
            {sourceLabel && (
              <p className="mt-1.5 text-[11px] text-muted-foreground">
                Source: {sourceLabel}
              </p>
            )}
          </div>
        )}

        <div className="flex items-center justify-between gap-2 pt-1">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 px-2"
            onClick={goPrev}
            disabled={index === 0}
          >
            <ChevronLeft className="w-4 h-4 mr-1" />
            Prev
          </Button>
          <p className="text-[11px] text-muted-foreground tabular-nums">
            {Object.keys(state.answers).length} answered
          </p>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 px-2"
            onClick={goNext}
            disabled={!revealed && !state.answers[current.id]}
          >
            {index >= questions.length - 1 ? "Results" : "Next"}
            <ChevronRight className="w-4 h-4 ml-1" />
          </Button>
        </div>
        {persistError && (
          <p className="text-[11px] text-amber-600">{persistError}</p>
        )}
      </div>
    </div>
  );
}

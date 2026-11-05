"use client";

import { useCallback, useMemo, useState } from "react";
import { Check, ChevronLeft, ChevronRight, CircleHelp, RotateCcw, X } from "lucide-react";
import { api } from "@/lib/api";
import type { QuizPayload, QuizQuestion } from "@/store/chat-store";
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
  const progressPct =
    total === 0
      ? 0
      : Math.round(((index + (revealed ? 1 : 0)) / total) * 100);

  const persistState = useCallback(
    async (next: QuizPayload) => {
      if (!messageId || messageId.startsWith("assistant-")) return;
      try {
        setPersistError(null);
        await api.patch(
          `/api/v1/chat/messages/${encodeURIComponent(messageId)}/quiz-state`,
          {
            current_index: next.state?.current_index ?? 0,
            answers: next.state?.answers ?? {},
            completed: next.state?.completed ?? false,
          },
        );
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
      <div className="gc-card">
        <div className="gc-card-header">
          <div>
            <p className="gc-card-title">{localQuiz.title}</p>
            <p className="gc-card-sub">{modeLabel(localQuiz.mode)}</p>
          </div>
          <span className="rounded-md border border-[var(--gd-border)] px-2 py-0.5 text-[10px] font-medium text-[var(--gd-muted)]">
            Results
          </span>
        </div>
        <div className="space-y-3 px-3 py-4">
          <p className="text-2xl font-semibold tabular-nums tracking-tight text-[var(--gd-text)]">
            {correct}
            <span className="text-base font-medium text-[var(--gd-muted)]">
              {" "}
              / {total}
            </span>
          </p>
          <p className="text-sm text-[var(--gd-muted)]">
            {correct === total
              ? "Perfect score — every answer matched the source material."
              : correct === 0
                ? "No correct answers this round. Review the explanations and try again."
                : `You answered ${correct} of ${total} correctly.`}
          </p>
          <div className="space-y-1.5 pt-1">
            {questions.map((q, i) => {
              const chosen = state.answers[q.id];
              const ok = chosen === q.correct_option_id;
              return (
                <div
                  key={q.id}
                  className="rounded-lg border border-[var(--gd-border)] px-3 py-2 text-[12px]"
                >
                  <div className="flex items-start gap-2">
                    {ok ? (
                      <Check className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[var(--gd-positive)]" />
                    ) : (
                      <X className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[var(--gd-negative)]" />
                    )}
                    <div className="min-w-0">
                      <p className="font-medium text-[var(--gd-text)]">
                        {i + 1}. {q.question}
                      </p>
                      <p className="mt-0.5 text-[var(--gd-muted)]">
                        Correct:{" "}
                        {q.options.find((o) => o.id === q.correct_option_id)
                          ?.text ?? q.correct_option_id}
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
            className="mt-1 border-[var(--gd-border)]"
            onClick={restart}
          >
            <RotateCcw className="mr-1.5 h-3.5 w-3.5" />
            Retake quiz
          </Button>
          {persistError && (
            <p className="text-[11px] text-[var(--gd-warning)]">{persistError}</p>
          )}
        </div>
      </div>
    );
  }

  const sourceLabel = current.source?.filename
    ? `${current.source.filename}${
        typeof current.source.page === "number"
          ? `, p. ${current.source.page + 1}`
          : ""
      }`
    : null;

  return (
    <div className="gc-card">
      <div className="gc-card-header">
        <div className="min-w-0">
          <p className="gc-card-title truncate">{localQuiz.title}</p>
          <p className="gc-card-sub">
            {modeLabel(localQuiz.mode)} · Question {index + 1} of {total}
          </p>
        </div>
        <span className="shrink-0 rounded-md border border-[var(--gd-border)] px-2 py-0.5 text-[10px] font-medium tabular-nums text-[var(--gd-muted)]">
          Score {correct}/{Object.keys(state.answers).length || 0}
        </span>
      </div>

      <div className="h-0.5 bg-[var(--gd-raised)]">
        <div
          className="h-full bg-[var(--gd-accent)] transition-all duration-300"
          style={{
            width: `${Math.min(
              100,
              Math.max(progressPct, (index / Math.max(total, 1)) * 100),
            )}%`,
          }}
        />
      </div>

      <div className="space-y-3 px-3 py-4">
        <p className="text-[15px] font-medium leading-relaxed text-[var(--gd-text)]">
          {current.question}
        </p>

        <div className="space-y-1.5" role="listbox" aria-label="Answer options">
          {current.options.map((opt) => {
            const isSelected = selected === opt.id;
            const isCorrect = opt.id === current.correct_option_id;
            let tone =
              "border-[var(--gd-border)] hover:border-[var(--gd-accent)]/40 hover:bg-[var(--gd-raised)] cursor-pointer";
            if (revealed) {
              if (isCorrect) {
                tone =
                  "border-[var(--gd-positive)]/35 bg-[color-mix(in_oklab,var(--gd-positive)_10%,transparent)]";
              } else if (isSelected) {
                tone =
                  "border-[var(--gd-negative)]/35 bg-[color-mix(in_oklab,var(--gd-negative)_10%,transparent)]";
              } else {
                tone = "border-[var(--gd-border)] opacity-60";
              }
            } else if (isSelected) {
              tone =
                "border-[var(--gd-accent)]/45 bg-[var(--gd-accent-soft)]";
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
                  "flex w-full items-start gap-2.5 rounded-lg border px-3 py-2.5 text-left text-[13px] transition-colors",
                  tone,
                  revealed && "cursor-default",
                )}
              >
                <span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-md border border-[var(--gd-border)] text-[10px] font-semibold uppercase text-[var(--gd-muted)]">
                  {opt.id}
                </span>
                <span className="flex-1 leading-snug text-[var(--gd-text)]">
                  {opt.text}
                </span>
                {revealed && isCorrect && (
                  <Check className="mt-0.5 h-4 w-4 shrink-0 text-[var(--gd-positive)]" />
                )}
                {revealed && isSelected && !isCorrect && (
                  <X className="mt-0.5 h-4 w-4 shrink-0 text-[var(--gd-negative)]" />
                )}
              </button>
            );
          })}
        </div>

        {revealed && (
          <div
            className={cn(
              "rounded-lg border px-3 py-2.5 text-[12px] leading-relaxed",
              selected === current.correct_option_id
                ? "border-[var(--gd-positive)]/25 bg-[color-mix(in_oklab,var(--gd-positive)_8%,transparent)]"
                : "border-[var(--gd-negative)]/25 bg-[color-mix(in_oklab,var(--gd-negative)_8%,transparent)]",
            )}
          >
            <p className="mb-1 flex items-center gap-1.5 font-medium text-[var(--gd-text)]">
              <CircleHelp className="h-3.5 w-3.5" />
              {selected === current.correct_option_id ? "Correct" : "Incorrect"}
            </p>
            <p className="text-[var(--gd-muted)]">
              {current.explanation ||
                "Review the cited source material for the supporting detail."}
            </p>
            {sourceLabel && (
              <p className="mt-1.5 text-[11px] text-[var(--gd-faint)]">
                Source: {sourceLabel}
              </p>
            )}
          </div>
        )}

        <div className="flex items-center justify-between gap-2 pt-0.5">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 px-2 text-[var(--gd-muted)]"
            onClick={goPrev}
            disabled={index === 0}
          >
            <ChevronLeft className="mr-1 h-4 w-4" />
            Prev
          </Button>
          <p className="text-[11px] tabular-nums text-[var(--gd-muted)]">
            {Object.keys(state.answers).length} answered
          </p>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 px-2 text-[var(--gd-muted)]"
            onClick={goNext}
            disabled={!revealed && !state.answers[current.id]}
          >
            {index >= questions.length - 1 ? "Results" : "Next"}
            <ChevronRight className="ml-1 h-4 w-4" />
          </Button>
        </div>
        {persistError && (
          <p className="text-[11px] text-[var(--gd-warning)]">{persistError}</p>
        )}
      </div>
    </div>
  );
}

import { describe, expect, it, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import QuizCard from "./QuizCard";
import type { QuizPayload } from "@/store/chat-store";

vi.mock("@/lib/api", () => ({
  api: {
    patch: vi.fn().mockResolvedValue({}),
  },
}));

const baseQuiz: QuizPayload = {
  id: "quiz-1",
  mode: "document",
  title: "Document check",
  questions: [
    {
      id: "q1",
      question: "What interval is recommended?",
      options: [
        { id: "a", text: "Every 2 weeks" },
        { id: "b", text: "Every 4 weeks" },
        { id: "c", text: "Daily" },
        { id: "d", text: "Monthly only" },
      ],
      correct_option_id: "b",
      explanation: "The label recommends every 4 weeks.",
      source: { filename: "label.pdf", page: 8 },
    },
    {
      id: "q2",
      question: "Second question?",
      options: [
        { id: "a", text: "One" },
        { id: "b", text: "Two" },
        { id: "c", text: "Three" },
        { id: "d", text: "Four" },
      ],
      correct_option_id: "a",
      explanation: "Conversation stated one.",
    },
  ],
  state: { current_index: 0, answers: {}, completed: false },
};

describe("QuizCard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("reveals correctness and explanation on option click", () => {
    render(
      <QuizCard quiz={baseQuiz} messageId="msg-1" onQuizUpdate={vi.fn()} />,
    );

    fireEvent.click(
      screen.getByRole("option", { name: /Option a: Every 2 weeks/i }),
    );
    expect(screen.getByText("Incorrect")).toBeInTheDocument();
    expect(
      screen.getByText("The label recommends every 4 weeks."),
    ).toBeInTheDocument();
    expect(screen.getByText(/Source: label\.pdf/i)).toBeInTheDocument();
  });

  it("navigates to next question and shows results summary", () => {
    render(
      <QuizCard quiz={baseQuiz} messageId="msg-1" onQuizUpdate={vi.fn()} />,
    );

    fireEvent.click(
      screen.getByRole("option", { name: /Option b: Every 4 weeks/i }),
    );
    expect(screen.getByText("Correct")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Next/i }));
    expect(screen.getByText("Second question?")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("option", { name: /Option a: One/i }));
    // Answering the final question completes the quiz immediately.
    expect(screen.getByText("Results")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Retake quiz/i })).toBeInTheDocument();
    expect(
      screen.getByText(/Perfect score — every answer matched the source material/i),
    ).toBeInTheDocument();
  });
});

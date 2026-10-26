import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import MessageBubble from "./MessageBubble";
import type { ChatMsg } from "@/store/chat-store";

vi.mock("@/lib/api", () => ({
  api: {
    patch: vi.fn(),
    post: vi.fn(),
  },
}));

const makeMessage = (overrides: Partial<ChatMsg>): ChatMsg => ({
  id: "msg-1",
  role: "assistant",
  content: "Assistant response",
  sources: [],
  ...overrides,
});

describe("MessageBubble", () => {
  it("renders a user message without assistant actions", () => {
    render(
      <MessageBubble
        message={makeMessage({
          role: "user",
          content: "Summarize my uploaded report",
        })}
      />,
    );

    expect(
      screen.getByText("Summarize my uploaded report"),
    ).toBeInTheDocument();
    expect(screen.queryByText("Was this helpful?")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Copy response")).not.toBeInTheDocument();
  });

  it("renders an assistant message with markdown and response controls", () => {
    const { container } = render(
      <MessageBubble
        message={makeMessage({
          content: "Here is the answer:\n\n```ts\nconst ok = true;\n```",
        })}
      />,
    );

    expect(screen.getByText("Here is the answer:")).toBeInTheDocument();
    expect(container.querySelector("pre")).toHaveTextContent(
      "const ok = true;",
    );
    expect(screen.getByLabelText("Copy response")).toBeInTheDocument();
    expect(screen.getByLabelText("Share response")).toBeInTheDocument();
  });

  it("renders inline and display math with KaTeX", () => {
    const { container } = render(
      <MessageBubble
        message={makeMessage({
          content:
            "Energy \\(E = mc^2\\).\n\n\\[\n\\int_0^1 x^2\\,dx = \\frac{1}{3}\n\\]\n\nCite [Source: physics.pdf, Page 2].",
        })}
      />,
    );

    expect(container.querySelectorAll(".katex").length).toBeGreaterThan(0);
    expect(container.querySelector(".katex-display")).not.toBeNull();
    expect(screen.getByText(/Source: physics.pdf, Page 2/)).toBeInTheDocument();
  });

  it("renders chemical equations without crashing", () => {
    const { container } = render(
      <MessageBubble
        message={makeMessage({
          content:
            "Water forms via \\(2H_2 + O_2 \\rightarrow 2H_2O\\) and calcination \\(\\mathrm{CaCO_3 \\rightarrow CaO + CO_2}\\). Also \\(\\ce{H2SO4 ->[heat] SO3 + H2O}\\).",
        })}
      />,
    );

    expect(container.querySelectorAll(".katex").length).toBeGreaterThan(0);
  });

  it("renders matrices, summations, and multi-line display math", () => {
    const { container } = render(
      <MessageBubble
        message={makeMessage({
          content: [
            "Matrix:",
            "\\[",
            "\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}",
            "\\]",
            "Sum: \\(\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}\\)",
          ].join("\n"),
        })}
      />,
    );

    expect(container.querySelector(".katex-display")).not.toBeNull();
    expect(container.querySelectorAll(".katex").length).toBeGreaterThan(1);
  });

  it("holds back incomplete streamed math without duplicating content", () => {
    const { container, rerender } = render(
      <MessageBubble
        message={makeMessage({
          content: "Partial \\(E = mc",
          isStreaming: true,
        })}
      />,
    );

    expect(container.querySelector(".katex")).toBeNull();
    expect(container.textContent).toContain("Partial");
    expect(container.textContent).not.toContain("E = mc");

    rerender(
      <MessageBubble
        message={makeMessage({
          content: "Partial \\(E = mc^2\\) done.",
          isStreaming: true,
        })}
      />,
    );

    expect(container.querySelector(".katex")).not.toBeNull();
    expect(container.textContent).toContain("done.");
  });

  it("handles malformed LaTeX without crashing the chat", () => {
    expect(() =>
      render(
        <MessageBubble
          message={makeMessage({
            content: "Broken \\(\\frac{a{b}\\) still readable.",
          })}
        />,
      ),
    ).not.toThrow();

    expect(screen.getByText(/still readable/)).toBeInTheDocument();
  });
});

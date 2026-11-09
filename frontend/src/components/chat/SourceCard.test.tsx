import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import SourceCard from "./SourceCard";
import type { SourceChunk } from "@/store/chat-store";

const sources: SourceChunk[] = [
  {
    filename: "annual-report.pdf",
    page: 3,
    text: "Revenue increased after the new pricing plan rolled out.",
    score: 0.91,
    confidence: 0.76,
    highlightRects: [{ left: 10, top: 20, width: 30, height: 40 }],
  },
  {
    filename: "board-notes.pdf",
    page: 5,
    text: "Risk review notes mention customer concentration and renewal timing.",
    score: 0.42,
  },
  {
    filename: "summary.txt",
    page: 0,
    text: "A".repeat(220),
    score: 0.2,
  },
];

describe("SourceCard", () => {
  it("renders a collapsed source summary and forwards page clicks", async () => {
    const onPageClick = vi.fn();
    const user = userEvent.setup();

    render(<SourceCard sources={sources} onPageClick={onPageClick} />);

    expect(
      screen.getByRole("button", { name: "Expand 3 cited sources" }),
    ).toBeInTheDocument();
    expect(screen.getByText("3 sources cited")).toBeInTheDocument();
    expect(
      screen.getByLabelText("Go to source page 3. Confidence Medium"),
    ).toHaveTextContent("p.3");
    expect(
      screen.getByLabelText("Go to source page 5. Confidence Low"),
    ).toHaveTextContent("p.5");

    await user.click(
      screen.getByLabelText("Go to source page 3. Confidence Medium"),
    );

    expect(onPageClick).toHaveBeenCalledWith({
      page: 3,
      filename: "annual-report.pdf",
      excerpt: sources[0].text,
      highlightRects: sources[0].highlightRects,
    });
  });

  it("expands to show source metadata, View, and excerpt toggle", async () => {
    const onPageClick = vi.fn();
    const user = userEvent.setup();

    render(<SourceCard sources={sources} onPageClick={onPageClick} />);

    await user.click(screen.getByRole("button", { name: "Expand 3 cited sources" }));

    expect(
      screen.getByRole("button", { name: "Collapse 3 cited sources" }),
    ).toBeInTheDocument();
    expect(screen.getByText("annual-report.pdf")).toBeInTheDocument();
    expect(screen.getByText("Page 3")).toBeInTheDocument();
    expect(screen.getByText("Score: High")).toBeInTheDocument();
    expect(screen.getByText("Confidence: Medium")).toBeInTheDocument();
    expect(
      screen.getByText("Revenue increased after the new pricing plan rolled out."),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "View source page 3" }));
    expect(onPageClick).toHaveBeenCalledWith({
      page: 3,
      filename: "annual-report.pdf",
      excerpt: sources[0].text,
      highlightRects: sources[0].highlightRects,
    });

    const showExcerpt = screen.getByRole("button", { name: "Show excerpt" });
    expect(showExcerpt).toHaveAttribute("aria-expanded", "false");
    await user.click(showExcerpt);
    expect(screen.getByRole("button", { name: "Hide excerpt" })).toHaveAttribute(
      "aria-expanded",
      "true",
    );
  });

  it("does not render when there are no sources", () => {
    const { container } = render(
      <SourceCard sources={[]} onPageClick={vi.fn()} />,
    );

    expect(container).toBeEmptyDOMElement();
  });
});

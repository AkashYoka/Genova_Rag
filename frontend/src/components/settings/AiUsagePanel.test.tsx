import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

import AiUsagePanel from "./AiUsagePanel";

const getMock = vi.fn();

vi.mock("@/lib/api", () => ({
  api: {
    get: (...args: unknown[]) => getMock(...args),
  },
  CONNECTION_ERROR_MESSAGE: "Could not connect to the server. Please try again later.",
}));

describe("AiUsagePanel", () => {
  beforeEach(() => {
    getMock.mockReset();
    getMock.mockImplementation(async (path: string) => {
      if (path.includes("/summary")) {
        return {
          start: "2026-01-01T00:00:00Z",
          end: "2026-01-31T00:00:00Z",
          request_count: 2,
          input_tokens: 100,
          output_tokens: 50,
          cache_read_tokens: 10,
          total_tokens: 160,
          estimated_cost_usd: 0.0123,
          today: { request_count: 1, total_tokens: 80, estimated_cost_usd: 0.01 },
          month_to_date: {
            request_count: 2,
            total_tokens: 160,
            estimated_cost_usd: 0.0123,
          },
          by_provider: [
            {
              provider: "bedrock",
              request_count: 2,
              total_tokens: 160,
              estimated_cost_usd: 0.0123,
            },
          ],
          by_model: [
            {
              provider: "bedrock",
              model_id: "us.amazon.nova-2-lite-v1:0",
              request_count: 2,
              total_tokens: 160,
              estimated_cost_usd: 0.0123,
            },
          ],
          by_usage_type: [],
          cost_disclaimer: "Costs are estimated.",
        };
      }
      if (path.includes("/daily")) {
        return {
          points: [
            {
              date: "2026-01-05",
              request_count: 2,
              total_tokens: 160,
              estimated_cost_usd: 0.0123,
              input_tokens: 100,
              output_tokens: 50,
            },
          ],
        };
      }
      return {
        total: 1,
        page: 1,
        limit: 25,
        pages: 1,
        items: [
          {
            id: "u1",
            created_at: "2026-01-05T12:00:00Z",
            provider: "bedrock",
            model_id: "us.amazon.nova-2-lite-v1:0",
            usage_type: "chat",
            input_tokens: 100,
            output_tokens: 50,
            cache_read_tokens: 10,
            cache_write_tokens: null,
            total_tokens: 160,
            estimated_cost_usd: 0.0123,
            cost_type: "estimated",
            pricing_note: "Estimated",
          },
        ],
      };
    });
  });

  it("renders summary cards and transaction rows", async () => {
    render(<AiUsagePanel />);

    await waitFor(() => {
      expect(screen.getByText("Estimated cost")).toBeInTheDocument();
    });

    expect(screen.getByText("Total tokens")).toBeInTheDocument();
    expect(screen.getByText("Usage transactions")).toBeInTheDocument();
    expect(await screen.findByText("chat")).toBeInTheDocument();
    expect(screen.getByText("Estimated")).toBeInTheDocument();
    expect(screen.getByText("Costs are estimated.")).toBeInTheDocument();
    expect(getMock).toHaveBeenCalled();
  });
});

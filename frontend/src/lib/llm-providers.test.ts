import { describe, expect, it } from "vitest";
import {
  fallbackAvailableLlmProviders,
  fallbackDefaultLlmProvider,
  isProductionDeploymentHost,
  llmProviderSelectOptions,
  resolveLlmProviderOptions,
} from "./llm-providers";

describe("llm-providers", () => {
  it("treats rag.genova-ai.yokaiq.in as production", () => {
    expect(isProductionDeploymentHost("rag.genova-ai.yokaiq.in")).toBe(true);
    expect(isProductionDeploymentHost("localhost")).toBe(false);
  });

  it("hides Ollama on the production host fallback", () => {
    expect(fallbackAvailableLlmProviders("rag.genova-ai.yokaiq.in")).toEqual([
      "bedrock",
    ]);
    expect(fallbackDefaultLlmProvider("rag.genova-ai.yokaiq.in")).toBe(
      "bedrock",
    );
    expect(fallbackAvailableLlmProviders("localhost")).toEqual([
      "ollama",
      "bedrock",
    ]);
  });

  it("uses backend config but still strips Ollama on production host", () => {
    const resolved = resolveLlmProviderOptions(
      {
        environment: "production",
        default_llm_provider: "ollama",
        available_llm_providers: ["ollama", "bedrock"],
      },
      "rag.genova-ai.yokaiq.in",
    );
    expect(resolved.available).toEqual(["bedrock"]);
    expect(resolved.defaultProvider).toBe("bedrock");
  });

  it("keeps Ollama locally when backend allows it", () => {
    const resolved = resolveLlmProviderOptions(
      {
        environment: "development",
        default_llm_provider: "ollama",
        available_llm_providers: ["ollama", "bedrock"],
      },
      "localhost",
    );
    expect(resolved.available).toEqual(["ollama", "bedrock"]);
    expect(resolved.defaultProvider).toBe("ollama");
    expect(llmProviderSelectOptions(resolved.available).map((o) => o.value)).toEqual(
      ["ollama", "bedrock"],
    );
  });
});

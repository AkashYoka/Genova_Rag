/**
 * Environment-aware LLM provider options for the chat UI.
 * Backend /api/v1/auth/config is the source of truth; hostname is a fallback
 * so production (rag.genova-ai.yokaiq.in) never briefly shows Ollama.
 */

export type LlmProviderId = "ollama" | "bedrock";

export type PublicLlmConfig = {
  environment?: string;
  default_llm_provider?: string;
  available_llm_providers?: string[];
};

const PRODUCTION_HOSTS = new Set(["rag.genova-ai.yokaiq.in"]);

const ALL_OPTIONS: { value: LlmProviderId; label: string }[] = [
  { value: "ollama", label: "Ollama" },
  { value: "bedrock", label: "Amazon Nova 2 Lite" },
];

export function isProductionDeploymentHost(hostname?: string): boolean {
  if (typeof window === "undefined" && !hostname) return false;
  const host = (hostname ?? window.location.hostname).toLowerCase();
  return PRODUCTION_HOSTS.has(host);
}

export function fallbackAvailableLlmProviders(
  hostname?: string,
): LlmProviderId[] {
  if (isProductionDeploymentHost(hostname)) {
    return ["bedrock"];
  }
  return ["ollama", "bedrock"];
}

export function fallbackDefaultLlmProvider(hostname?: string): LlmProviderId {
  return fallbackAvailableLlmProviders(hostname)[0];
}

export function normalizeLlmProviderId(
  value: string | undefined | null,
  available: LlmProviderId[],
): LlmProviderId {
  const normalized = (value || "").trim().toLowerCase();
  if (normalized === "ollama" || normalized === "bedrock") {
    if (available.includes(normalized)) return normalized;
  }
  return available[0] ?? "bedrock";
}

export function resolveLlmProviderOptions(
  config?: PublicLlmConfig | null,
  hostname?: string,
): { available: LlmProviderId[]; defaultProvider: LlmProviderId } {
  const fallbackAvailable = fallbackAvailableLlmProviders(hostname);
  const raw = config?.available_llm_providers;
  const available =
    Array.isArray(raw) && raw.length > 0
      ? (raw
          .map((p) => p.trim().toLowerCase())
          .filter((p): p is LlmProviderId => p === "ollama" || p === "bedrock")
          // Production host must never expose Ollama even if misconfigured.
          .filter((p) => !(isProductionDeploymentHost(hostname) && p === "ollama")) as LlmProviderId[])
      : fallbackAvailable;

  const resolvedAvailable =
    available.length > 0 ? available : fallbackAvailable;

  const defaultProvider = normalizeLlmProviderId(
    config?.default_llm_provider,
    resolvedAvailable,
  );

  return { available: resolvedAvailable, defaultProvider };
}

export function llmProviderSelectOptions(available: LlmProviderId[]) {
  return ALL_OPTIONS.filter((opt) => available.includes(opt.value));
}

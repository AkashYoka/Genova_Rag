"use client";

import { useCallback, useEffect, useState } from "react";
import { Check, Copy, Plus, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { api, CONNECTION_ERROR_MESSAGE } from "@/lib/api";
import {
  LedgerButton,
  LedgerEmpty,
  LedgerScrollTable,
} from "@/components/ledger/LedgerPrimitives";

interface ApiKey {
  id: string;
  key_prefix: string;
  created_at: string;
  last_used: string | null;
  last_used_at?: string | null;
}

export default function ApiKeysPanel() {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [newKey, setNewKey] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const fetchKeys = useCallback(async () => {
    try {
      setLoading(true);
      const data = await api.get<ApiKey[]>("/api/v1/auth/api-keys");
      setKeys(data || []);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : CONNECTION_ERROR_MESSAGE);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchKeys();
  }, [fetchKeys]);

  const generateKey = async () => {
    try {
      setLoading(true);
      const data = await api.post<{ key: string; api_key: ApiKey }>(
        "/api/v1/auth/api-keys",
      );
      setNewKey(data.key);
      setKeys((prev) => [...prev, data.api_key]);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : CONNECTION_ERROR_MESSAGE);
    } finally {
      setLoading(false);
    }
  };

  const revokeKey = async (id: string) => {
    if (
      !confirm(
        "Revoke this key? Integrations using it will stop working immediately.",
      )
    ) {
      return;
    }
    try {
      await api.delete(`/api/v1/auth/api-keys/${id}`);
      setKeys((prev) => prev.filter((k) => k.id !== id));
      toast.success("API key revoked");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : CONNECTION_ERROR_MESSAGE);
    }
  };

  const copyToClipboard = async () => {
    if (!newKey) return;
    await navigator.clipboard.writeText(newKey);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-[13px] text-[var(--gd-muted)]">
          Keys grant programmatic access to your RAG workspace. The full secret is
          shown once.
        </p>
        <LedgerButton
          variant="primary"
          onClick={() => void generateKey()}
          disabled={loading}
        >
          <Plus className="h-3.5 w-3.5" />
          Generate key
        </LedgerButton>
      </div>

      {newKey ? (
        <div className="rounded-[10px] border border-[var(--gd-border)] bg-[var(--gd-surface)] p-4">
          <p className="text-[13px] font-medium">Copy this key now</p>
          <p className="mt-1 text-[12px] text-[var(--gd-muted)]">
            It will not be shown again.
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <code className="ledger-mono flex-1 break-all rounded-[8px] border border-[var(--gd-border)] bg-[var(--gd-raised)] px-3 py-2 text-[12px]">
              {newKey}
            </code>
            <LedgerButton variant="secondary" onClick={() => void copyToClipboard()}>
              {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
              {copied ? "Copied" : "Copy"}
            </LedgerButton>
          </div>
        </div>
      ) : null}

      <div className="overflow-hidden rounded-[10px] border border-[var(--gd-border)] bg-[var(--gd-surface)]">
        <LedgerScrollTable compact>
          <table className="ledger-table">
            <thead>
              <tr>
                <th>Prefix</th>
                <th>Created</th>
                <th>Last used</th>
                <th data-align="right"> </th>
              </tr>
            </thead>
            <tbody>
              {keys.map((key) => {
                const lastUsed = key.last_used_at || key.last_used;
                return (
                  <tr key={key.id}>
                    <td className="ledger-mono">{key.key_prefix}••••</td>
                    <td className="text-[var(--gd-muted)]">
                      {new Date(key.created_at).toLocaleDateString()}
                    </td>
                    <td className="text-[var(--gd-muted)]">
                      {lastUsed ? new Date(lastUsed).toLocaleDateString() : "Never"}
                    </td>
                    <td data-align="right">
                      <LedgerButton
                        variant="ghost"
                        aria-label={`Revoke ${key.key_prefix}`}
                        onClick={() => void revokeKey(key.id)}
                      >
                        <Trash2 className="h-3.5 w-3.5 text-[var(--gd-negative)]" />
                      </LedgerButton>
                    </td>
                  </tr>
                );
              })}
              {!keys.length ? (
                <tr>
                  <td colSpan={4}>
                    <LedgerEmpty>
                      {loading ? "Loading keys…" : "No API keys yet."}
                    </LedgerEmpty>
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </LedgerScrollTable>
      </div>
    </div>
  );
}

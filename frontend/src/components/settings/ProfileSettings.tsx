"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";

import { api, CONNECTION_ERROR_MESSAGE } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { LedgerButton } from "@/components/ledger/LedgerPrimitives";

export default function ProfileSettings() {
  const { user, initializeAuth } = useAuth();
  const [username, setUsername] = useState(user?.username || "");
  const [email, setEmail] = useState(user?.email || "");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);
  const [savingPassword, setSavingPassword] = useState(false);

  useEffect(() => {
    setUsername(user?.username || "");
    setEmail(user?.email || "");
  }, [user?.username, user?.email]);

  const saveProfile = async () => {
    try {
      setSavingProfile(true);
      await api.put("/api/v1/auth/update", {
        username: username.trim() || undefined,
        email: email.trim() || undefined,
      });
      await initializeAuth();
      toast.success("Profile updated");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : CONNECTION_ERROR_MESSAGE);
    } finally {
      setSavingProfile(false);
    }
  };

  const savePassword = async () => {
    if (newPassword !== confirmPassword) {
      toast.error("New password and confirmation do not match");
      return;
    }
    try {
      setSavingPassword(true);
      await api.put("/api/v1/auth/password", {
        password: newPassword,
        confirm_password: confirmPassword,
        current_password: currentPassword || undefined,
      });
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      toast.success("Password updated");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : CONNECTION_ERROR_MESSAGE);
    } finally {
      setSavingPassword(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <div className="ledger-field">
          <label htmlFor="settings-username">Username</label>
          <input
            id="settings-username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
          />
        </div>
        <div className="ledger-field">
          <label htmlFor="settings-email">Email</label>
          <input
            id="settings-email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
          />
        </div>
        <LedgerButton
          variant="primary"
          onClick={() => void saveProfile()}
          disabled={savingProfile}
        >
          {savingProfile ? "Saving…" : "Save profile"}
        </LedgerButton>
      </div>

      <div className="border-t border-[var(--gd-border)] pt-5">
        <p className="mb-4 text-[13px] text-[var(--gd-muted)]">
          Password changes are hashed server-side and never shown again.
        </p>
        <div className="ledger-field">
          <label htmlFor="settings-current-password">Current password</label>
          <input
            id="settings-current-password"
            type="password"
            value={currentPassword}
            onChange={(e) => setCurrentPassword(e.target.value)}
            autoComplete="current-password"
          />
        </div>
        <div className="ledger-field">
          <label htmlFor="settings-new-password">New password</label>
          <input
            id="settings-new-password"
            type="password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            autoComplete="new-password"
          />
        </div>
        <div className="ledger-field">
          <label htmlFor="settings-confirm-password">Confirm new password</label>
          <input
            id="settings-confirm-password"
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            autoComplete="new-password"
          />
        </div>
        <LedgerButton
          variant="secondary"
          onClick={() => void savePassword()}
          disabled={savingPassword || !newPassword}
        >
          {savingPassword ? "Updating…" : "Update password"}
        </LedgerButton>
      </div>
    </div>
  );
}

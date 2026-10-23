import React from "react";

interface LogoProps {
  className?: string;
  size?: number;
}

/** Genova mark — geometric G with signal node. */
export default function Logo({ className = "w-7 h-7", size = 28 }: LogoProps) {
  return (
    <div
      className={`relative flex items-center justify-center shrink-0 rounded-xl overflow-hidden ${className}`}
      style={{ width: size, height: size }}
      aria-hidden="true"
    >
      <svg viewBox="0 0 32 32" fill="none" className="h-full w-full">
        <defs>
          <linearGradient id="genovaMark" x1="4" y1="2" x2="28" y2="30" gradientUnits="userSpaceOnUse">
            <stop stopColor="oklch(0.78 0.13 195)" />
            <stop offset="0.55" stopColor="oklch(0.7 0.11 210)" />
            <stop offset="1" stopColor="oklch(0.86 0.16 125)" />
          </linearGradient>
        </defs>
        <rect width="32" height="32" rx="9" fill="url(#genovaMark)" />
        <path
          d="M22.2 11.2A7.4 7.4 0 1 0 22.4 20.1"
          stroke="oklch(0.16 0.03 230)"
          strokeWidth="2.4"
          strokeLinecap="round"
        />
        <circle cx="22.6" cy="16" r="2.2" fill="oklch(0.16 0.03 230)" />
        <circle cx="22.6" cy="16" r="0.9" fill="oklch(0.9 0.16 125)" />
      </svg>
    </div>
  );
}

import React from "react";

interface LogoProps {
  className?: string;
  size?: number;
}

/** Genova mark — charcoal tile with teal G and citation node. */
export default function Logo({ className = "w-7 h-7", size = 28 }: LogoProps) {
  return (
    // eslint-disable-next-line @next/next/no-img-element -- brand mark asset, not content photography
    <img
      src="/logo.png"
      alt=""
      width={size}
      height={size}
      className={`shrink-0 rounded-[22%] object-cover ${className}`}
      style={{ width: size, height: size }}
      aria-hidden="true"
      draggable={false}
    />
  );
}

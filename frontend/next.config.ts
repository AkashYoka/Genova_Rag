import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  transpilePackages: ["ckeditor5", "@ckeditor/ckeditor5-react"],
  // Never ship browser source maps with the static export (DevTools "Page Source"
  // / Sources must not reconstruct original TS with comments or env wiring).
  productionBrowserSourceMaps: false,
  // Turbopack config (Next.js 16 default bundler)
  turbopack: {},
};

export default nextConfig;

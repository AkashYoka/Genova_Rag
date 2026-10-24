import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  transpilePackages: ["ckeditor5", "@ckeditor/ckeditor5-react"],
  // Turbopack config (Next.js 16 default bundler)
  turbopack: {},
};

export default nextConfig;

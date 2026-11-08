/**
 * Copy pdf.js worker into /public so Next can serve it as a static module.
 * Avoids Turbopack hashing the worker into /_next/static/media/, which breaks
 * pdf.js dynamic import ("Setting up fake worker failed").
 */
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const publicDir = path.join(root, "public");

const pdfjsRoot = path.dirname(require.resolve("pdfjs-dist/package.json"));
const src = path.join(pdfjsRoot, "build", "pdf.worker.min.mjs");
const dest = path.join(publicDir, "pdf.worker.min.mjs");

if (!fs.existsSync(src)) {
  console.warn(`[copy-pdf-worker] missing ${src}`);
  process.exit(0);
}

fs.mkdirSync(publicDir, { recursive: true });
fs.copyFileSync(src, dest);
console.log(`[copy-pdf-worker] wrote ${path.relative(root, dest)}`);

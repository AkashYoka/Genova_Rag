#!/usr/bin/env python3
"""Run STE on xolair_prescribing.pdf and write a 3-step demo HTML.

Steps shown:
  1. Textract AnalyzeDocument TABLES (page images → cells / rows / cols / bboxes)
  2. Bedrock LLM reconstruction (headers, gaps, multipage)
  3. Python validation + HTML / JSON / CSV exports

Also optionally re-indexes the document into Chroma/SQL (default: on).

Usage (from backend/):
  python scripts/demo_ste_xolair_pipeline.py
  python scripts/demo_ste_xolair_pipeline.py --pages 5,6,19
  python scripts/demo_ste_xolair_pipeline.py --no-reindex
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PDF = ROOT / "data" / "sample_subsets" / "xolair_prescribing.pdf"
DEFAULT_PAGES = "5,6,7,8,19"
OUT_DIR = ROOT / "data" / "layout_reports"
# When --all-pages, leave PDF_STE_PAGES blank; AnalyzeDocument still gates
# to tabular pages via tables_only=True inside ste.pipeline.

logger = logging.getLogger("demo_ste_xolair")


@dataclass
class LogEvent:
    ts: float
    level: str
    logger_name: str
    message: str
    step: int  # 0=general, 1=textract, 2=bedrock, 3=python


@dataclass
class StepCapture:
    events: List[LogEvent] = field(default_factory=list)

    def classify(self, logger_name: str, message: str) -> int:
        name = (logger_name or "").lower()
        msg = (message or "").lower()
        if (
            "textract" in name
            or "analyze_document" in msg
            or "analyzedocument" in msg
            or "textract_analyze" in msg
            or name.endswith("textract_analyze_backend")
            or "ocr_textract" in name
        ):
            return 1
        if (
            "bedrock" in name
            or "bedrock" in msg
            or "nova" in msg
            or "reconstruction" in msg
        ):
            return 2
        if (
            "validat" in msg
            or "write_" in msg
            or "export" in msg
            or "reconstructed.html" in msg
            or "validation_report" in msg
            or name.endswith("validator")
            or name.endswith("exporters")
        ):
            return 3
        return 0


class CaptureHandler(logging.Handler):
    def __init__(self, capture: StepCapture) -> None:
        super().__init__(level=logging.INFO)
        self.capture = capture

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        step = self.capture.classify(record.name, msg)
        self.capture.events.append(
            LogEvent(
                ts=record.created,
                level=record.levelname,
                logger_name=record.name,
                message=msg,
                step=step,
            )
        )


def _parse_pages(raw: str) -> List[int]:
    pages: List[int] = []
    for part in (raw or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            pages.append(int(part))
        except ValueError:
            continue
    return sorted(set(p for p in pages if p > 0))


def _b64_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _grid_html(grid: List[List[Any]], *, max_rows: int = 8, max_cols: int = 8) -> str:
    if not grid:
        return "<p class='muted'>Empty grid</p>"
    rows = grid[:max_rows]
    cols = max((len(r) for r in rows), default=0)
    cols = min(cols, max_cols)
    parts = ["<table class='grid'>"]
    for i, row in enumerate(rows):
        parts.append("<tr>")
        cells = list(row)[:cols]
        tag = "th" if i == 0 else "td"
        for cell in cells:
            parts.append(f"<{tag}>{html.escape(str(cell or ''))}</{tag}>")
        if len(row) > cols:
            parts.append(f"<{tag} class='muted'>+{len(row) - cols}</{tag}>")
        parts.append("</tr>")
    parts.append("</table>")
    if len(grid) > max_rows:
        parts.append(f"<p class='muted'>+{len(grid) - max_rows} more rows</p>")
    return "".join(parts)


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _run_ste(
    *,
    pages: List[int],
    capture: StepCapture,
    out_root: Path,
) -> Dict[str, Any]:
    from ste.pipeline import extract_pdf

    run_dir = out_root / f"demo_run_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Force Textract STE path; region from env / Genova settings.
    from app.config import get_settings

    settings = get_settings()
    region = (
        (getattr(settings, "PDF_STE_AWS_REGION", "") or "").strip()
        or (getattr(settings, "PDF_TEXTRACT_REGION", "") or "").strip()
        or (getattr(settings, "BEDROCK_REGION", "") or "").strip()
        or os.environ.get("AWS_REGION")
        or "us-east-1"
    )
    model_id = (
        (getattr(settings, "PDF_STE_BEDROCK_MODEL_ID", "") or "").strip()
        or (getattr(settings, "BEDROCK_MODEL_ID", "") or "").strip()
        or "us.amazon.nova-2-lite-v1:0"
    )

    logger.info("STEP 1 — Textract AnalyzeDocument TABLES on page images (pages=%s)", pages)
    capture.events.append(
        LogEvent(
            time.time(),
            "INFO",
            "demo_ste_xolair",
            f"STEP 1 start: AnalyzeDocument FeatureTypes=['TABLES'] pages={pages} region={region}",
            1,
        )
    )
    logger.info("STEP 2 — Bedrock LLM reconstruction enabled (model=%s)", model_id)
    capture.events.append(
        LogEvent(
            time.time(),
            "INFO",
            "demo_ste_xolair",
            f"STEP 2 armed: Bedrock model={model_id} region={region}",
            2,
        )
    )
    logger.info("STEP 3 — Python validation + HTML/JSON/CSV exports → %s", run_dir)
    capture.events.append(
        LogEvent(
            time.time(),
            "INFO",
            "demo_ste_xolair",
            f"STEP 3 armed: validate_table + exporters → {run_dir}",
            3,
        )
    )

    t0 = time.perf_counter()
    result = extract_pdf(
        PDF,
        output_dir=run_dir,
        backend="textract",
        tables_only=True,  # AnalyzeDocument only on tabular pages
        pages=pages or None,  # None = whole PDF, still gated
        use_cache=False,
        stitch_multipage=True,
        bedrock=True,
        bedrock_model_id=model_id,
        aws_region=region,
        dpi=200,
    )
    elapsed = round(time.perf_counter() - t0, 2)

    capture.events.append(
        LogEvent(
            time.time(),
            "INFO",
            "demo_ste_xolair",
            f"Pipeline finished: tables={len(result.tables)} elapsed={elapsed}s "
            f"backend={result.ocr_backend} errors={len(result.errors)}",
            0,
        )
    )
    return {
        "run_dir": str(run_dir),
        "elapsed_s": elapsed,
        "table_count": len(result.tables),
        "page_count": result.page_count,
        "processed_pages": list(result.processed_pages or []),
        "skipped_pages": list(result.skipped_pages or []),
        "errors": list(result.errors or []),
        "backend": result.ocr_backend,
        "meta": dict(result.meta or {}),
        "region": region,
        "model_id": model_id,
        "pages": pages,
    }


def _collect_tables(run_dir: Path) -> List[Dict[str, Any]]:
    tables_dir = run_dir / "tables"
    if not tables_dir.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for table_dir in sorted(tables_dir.iterdir()):
        if not table_dir.is_dir():
            continue
        original = _load_json(table_dir / "original_extraction.json") or {}
        reconstructed = _load_json(table_dir / "reconstructed.json") or {}
        universal = _load_json(table_dir / "universal.json") or {}
        validation = _load_json(table_dir / "validation_report.json") or {}
        preview_b64 = _b64_file(table_dir / "preview.png")
        csv_preview = ""
        csv_path = table_dir / "reconstructed.csv"
        if csv_path.is_file():
            csv_preview = csv_path.read_text(encoding="utf-8", errors="replace")[:1200]

        cells_meta = list(original.get("cells_meta") or [])[:6]
        bbox_samples = []
        for cell in cells_meta:
            bbox_samples.append(
                {
                    "row": cell.get("row"),
                    "col": cell.get("col"),
                    "text": cell.get("text"),
                    "bbox": cell.get("bbox"),
                    "confidence": cell.get("confidence"),
                }
            )

        orig_grid = original.get("grid") or reconstructed.get("original_grid") or []
        recon_headers = (
            universal.get("reconstructed_headers")
            or reconstructed.get("headers")
            or []
        )
        recon_rows = reconstructed.get("rows") or []
        recon_grid = [list(recon_headers)] + [list(r) for r in recon_rows] if recon_headers else recon_rows

        out.append(
            {
                "table_id": table_dir.name,
                "page": original.get("page")
                or reconstructed.get("page")
                or universal.get("page_number"),
                "source": original.get("source") or "textract_analyze_tables",
                "n_rows": original.get("n_rows") or len(orig_grid),
                "n_cols": original.get("n_cols")
                or (len(orig_grid[0]) if orig_grid else 0),
                "header_row_count": original.get("header_row_count"),
                "confidence": original.get("confidence"),
                "bbox": original.get("bbox"),
                "original_grid_html": _grid_html(
                    [list(r) for r in orig_grid] if isinstance(orig_grid, list) else []
                ),
                "reconstructed_grid_html": _grid_html(
                    [list(r) for r in recon_grid] if isinstance(recon_grid, list) else []
                ),
                "bbox_samples": bbox_samples,
                "bedrock": (reconstructed.get("provenance") or {}).get("bedrock")
                or {},
                "validation": validation,
                "preview_b64": preview_b64,
                "csv_preview": csv_preview,
                "exports": {
                    "original_extraction.json": (table_dir / "original_extraction.json").is_file(),
                    "reconstructed.json": (table_dir / "reconstructed.json").is_file(),
                    "universal.json": (table_dir / "universal.json").is_file(),
                    "reconstructed.html": (table_dir / "reconstructed.html").is_file(),
                    "reconstructed.csv": (table_dir / "reconstructed.csv").is_file(),
                    "validation_report.json": (table_dir / "validation_report.json").is_file(),
                    "preview.png": (table_dir / "preview.png").is_file(),
                },
            }
        )
    return out


def _render_demo_html(
    *,
    run_meta: Dict[str, Any],
    tables: List[Dict[str, Any]],
    capture: StepCapture,
    reindex_report: Optional[Dict[str, Any]],
) -> str:
    t0 = capture.events[0].ts if capture.events else time.time()

    def log_block(step: int) -> str:
        rows = []
        for ev in capture.events:
            if step and ev.step != step and not (
                step == 0 and ev.step == 0
            ):
                if step != 0 and ev.step != step:
                    continue
            if step == 0 and ev.step != 0:
                continue
            rel = round(ev.ts - t0, 2)
            rows.append(
                "<div class='log-line'>"
                f"<span class='t'>+{rel}s</span>"
                f"<span class='lvl'>{_esc(ev.level)}</span>"
                f"<span class='msg'>{_esc(ev.message)}</span>"
                "</div>"
            )
        # Also include uncategorized lines that mention the step keyword for step panels
        if step and not rows:
            for ev in capture.events:
                if step == 1 and ("textract" in ev.message.lower() or "analyze" in ev.message.lower()):
                    pass
                else:
                    continue
        if not rows:
            # Fallback: show a few general lines
            for ev in capture.events[:8]:
                rel = round(ev.ts - t0, 2)
                rows.append(
                    "<div class='log-line'>"
                    f"<span class='t'>+{rel}s</span>"
                    f"<span class='lvl'>{_esc(ev.level)}</span>"
                    f"<span class='msg'>{_esc(ev.message)}</span>"
                    "</div>"
                )
        return "".join(rows) or "<p class='muted'>No log lines captured for this step.</p>"

    def step_logs(step: int) -> str:
        lines = [ev for ev in capture.events if ev.step == step]
        if not lines:
            lines = [ev for ev in capture.events if ev.step == 0][:5]
        parts = []
        for ev in lines:
            rel = round(ev.ts - t0, 2)
            parts.append(
                "<div class='log-line'>"
                f"<span class='t'>+{rel}s</span>"
                f"<span class='lvl'>{_esc(ev.level)}</span>"
                f"<span class='msg'>{_esc(ev.message)}</span>"
                "</div>"
            )
        return "".join(parts) or "<p class='muted'>No classified log lines — see full log below.</p>"

    usage = (run_meta.get("meta") or {}).get("bedrock_usage") or {}
    backend_meta = (run_meta.get("meta") or {}).get("backend_meta") or {}

    table_cards = []
    for t in tables:
        preview = ""
        if t.get("preview_b64"):
            preview = (
                f"<img class='preview' alt='table preview' "
                f"src='data:image/png;base64,{t['preview_b64']}' />"
            )
        bbox_rows = "".join(
            "<tr>"
            f"<td>{_esc(b.get('row'))}</td><td>{_esc(b.get('col'))}</td>"
            f"<td>{_esc(b.get('text'))}</td><td><code>{_esc(b.get('bbox'))}</code></td>"
            f"<td>{_esc(b.get('confidence'))}</td>"
            "</tr>"
            for b in (t.get("bbox_samples") or [])
        )
        exports = "".join(
            f"<li class='{'ok' if ok else 'missing'}'>{_esc(name)}</li>"
            for name, ok in (t.get("exports") or {}).items()
        )
        val = t.get("validation") or {}
        bedrock = t.get("bedrock") or {}
        table_cards.append(
            f"""
<section class="table-card">
  <header>
    <h3>{_esc(t.get('table_id'))} · page {_esc(t.get('page'))}</h3>
    <p class="muted">{_esc(t.get('source'))} · {_esc(t.get('n_rows'))}×{_esc(t.get('n_cols'))} ·
       conf {_esc(t.get('confidence'))}</p>
  </header>
  <div class="table-grid">
    <div>
      <h4>Step 1 — Textract original grid</h4>
      {t.get('original_grid_html') or ''}
      <h4>Cell bbox samples</h4>
      <table class="grid compact">
        <tr><th>r</th><th>c</th><th>text</th><th>bbox</th><th>conf</th></tr>
        {bbox_rows or '<tr><td colspan="5" class="muted">none</td></tr>'}
      </table>
    </div>
    <div>
      <h4>Step 2 — Bedrock reconstructed</h4>
      {t.get('reconstructed_grid_html') or ''}
      <p class="muted">bedrock applied={_esc(bedrock.get('applied'))}
         tokens={_esc(bedrock.get('total_tokens'))}
         model={_esc(bedrock.get('model_id'))}</p>
      <h4>Step 3 — Validation &amp; exports</h4>
      <p>validation ok=<strong>{_esc(val.get('ok', val.get('passed', '—')))}</strong>
         warnings={_esc(len(val.get('warnings') or []))}</p>
      <ul class="exports">{exports}</ul>
      <pre class="csv">{_esc(t.get('csv_preview') or '')}</pre>
    </div>
    <div class="preview-wrap">{preview}</div>
  </div>
</section>
"""
        )

    reindex_html = ""
    if reindex_report:
        insp = reindex_report.get("inspection") or {}
        reindex_html = f"""
<section class="panel">
  <h2>Re-index result</h2>
  <p>document_id=<code>{_esc(reindex_report.get('document_id'))}</code>
     status=<strong>{_esc(reindex_report.get('status'))}</strong>
     chunks={_esc(reindex_report.get('chunk_count'))}
     elapsed={_esc(reindex_report.get('ingest_elapsed_s'))}s</p>
  <p class="muted">chroma_ste_table_chunks={_esc(insp.get('chroma_ste_table_chunks'))}
     sql_table_count={_esc(insp.get('sql_table_count'))}
     chroma_chunk_types={_esc(insp.get('chroma_chunk_types'))}</p>
</section>
"""

    all_logs = []
    for ev in capture.events:
        rel = round(ev.ts - t0, 2)
        badge = {0: "gen", 1: "s1", 2: "s2", 3: "s3"}.get(ev.step, "gen")
        all_logs.append(
            f"<div class='log-line {badge}'>"
            f"<span class='t'>+{rel}s</span>"
            f"<span class='step'>S{ev.step or '·'}</span>"
            f"<span class='lvl'>{_esc(ev.level)}</span>"
            f"<span class='src'>{_esc(ev.logger_name)}</span>"
            f"<span class='msg'>{_esc(ev.message)}</span>"
            "</div>"
        )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>STE Demo — Xolair · Textract → Bedrock → Python</title>
<style>
  :root {{
    --bg: #0f1419;
    --panel: #1a222c;
    --ink: #e8eef4;
    --muted: #8b9aab;
    --accent: #3d9cf0;
    --s1: #2bb673;
    --s2: #f0a030;
    --s3: #c084fc;
    --line: #2a3542;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: "Segoe UI", system-ui, sans-serif;
    background: radial-gradient(1200px 600px at 10% -10%, #1a3a4a 0%, var(--bg) 55%);
    color: var(--ink); line-height: 1.45;
  }}
  header.hero {{
    padding: 2.2rem 2rem 1.4rem; border-bottom: 1px solid var(--line);
  }}
  header.hero h1 {{ margin: 0 0 .4rem; font-size: 1.75rem; letter-spacing: -.02em; }}
  header.hero p {{ margin: .2rem 0; color: var(--muted); max-width: 70ch; }}
  .wrap {{ padding: 1.25rem 2rem 3rem; }}
  .steps {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1.25rem 0 1.75rem;
  }}
  @media (max-width: 960px) {{ .steps {{ grid-template-columns: 1fr; }} }}
  .step-card {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 1rem 1.1rem; min-height: 12rem;
  }}
  .step-card h2 {{ margin: 0 0 .5rem; font-size: 1.05rem; }}
  .step-card.s1 h2 {{ color: var(--s1); }}
  .step-card.s2 h2 {{ color: var(--s2); }}
  .step-card.s3 h2 {{ color: var(--s3); }}
  .step-card ul {{ margin: .4rem 0 .8rem; padding-left: 1.1rem; color: var(--muted); font-size: .92rem; }}
  .log {{
    background: #0b1015; border: 1px solid var(--line); border-radius: 8px;
    padding: .55rem .65rem; max-height: 160px; overflow: auto; font-family: ui-monospace, Consolas, monospace;
    font-size: .75rem;
  }}
  .log-line {{ display: grid; grid-template-columns: 52px 36px 48px minmax(80px,140px) 1fr; gap: .4rem; padding: .12rem 0; border-bottom: 1px solid #162029; }}
  .log .log-line {{ grid-template-columns: 52px 42px 1fr; }}
  .log-line .t {{ color: var(--muted); }}
  .log-line.s1 .msg {{ color: #9be7c0; }}
  .log-line.s2 .msg {{ color: #ffd399; }}
  .log-line.s3 .msg {{ color: #e0c4ff; }}
  .panel {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 1rem 1.2rem; margin-bottom: 1.25rem;
  }}
  .muted {{ color: var(--muted); font-size: .9rem; }}
  code {{ background: #0b1015; padding: .1rem .35rem; border-radius: 4px; }}
  .table-card {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 1rem 1.2rem; margin-bottom: 1.25rem;
  }}
  .table-card header h3 {{ margin: 0; }}
  .table-grid {{
    display: grid; grid-template-columns: 1.1fr 1.1fr .7fr; gap: 1rem; margin-top: .8rem;
  }}
  @media (max-width: 1100px) {{ .table-grid {{ grid-template-columns: 1fr; }} }}
  table.grid {{
    width: 100%; border-collapse: collapse; font-size: .78rem; background: #0b1015;
  }}
  table.grid th, table.grid td {{
    border: 1px solid var(--line); padding: .25rem .35rem; vertical-align: top;
  }}
  table.grid th {{ background: #15202b; }}
  table.compact {{ margin-top: .4rem; }}
  img.preview {{
    width: 100%; border-radius: 8px; border: 1px solid var(--line); background: #000;
  }}
  ul.exports {{ list-style: none; padding: 0; margin: .4rem 0; font-size: .85rem; }}
  ul.exports .ok::before {{ content: "✓ "; color: var(--s1); }}
  ul.exports .missing::before {{ content: "✗ "; color: #f07178; }}
  pre.csv {{
    background: #0b1015; border: 1px solid var(--line); border-radius: 6px;
    padding: .5rem; font-size: .72rem; max-height: 120px; overflow: auto; white-space: pre-wrap;
  }}
  .full-log {{ max-height: 420px; }}
  .kpi {{ display: flex; flex-wrap: wrap; gap: .75rem 1.25rem; margin: .6rem 0 0; }}
  .kpi span {{ background: #0b1015; border: 1px solid var(--line); border-radius: 999px; padding: .25rem .7rem; font-size: .85rem; }}
</style>
</head>
<body>
  <header class="hero">
    <h1>Genova STE demo — Xolair prescribing info</h1>
    <p>Live pipeline proof: <strong>Textract AnalyzeDocument</strong> →
       <strong>Bedrock LLM reconstruction</strong> →
       <strong>Python validate + HTML/JSON/CSV</strong>. Docling is not used.</p>
    <p class="muted">Generated { _esc(generated) } · PDF {_esc(PDF.name)} ·
       pages {_esc(run_meta.get('pages'))} · run {_esc(run_meta.get('run_dir'))}</p>
    <div class="kpi">
      <span>tables {_esc(run_meta.get('table_count'))}</span>
      <span>elapsed {_esc(run_meta.get('elapsed_s'))}s</span>
      <span>backend {_esc(run_meta.get('backend'))}</span>
      <span>AnalyzeDocument pages {_esc(backend_meta.get('analyze_document_tables_pages'))}</span>
      <span>Bedrock tokens {_esc(usage.get('total_tokens') or (usage.get('totals') or {}).get('total_tokens'))}</span>
    </div>
  </header>

  <div class="wrap">
    <div class="steps">
      <article class="step-card s1">
        <h2>1 · Textract AnalyzeDocument</h2>
        <ul>
          <li>Render each gated PDF page to an image</li>
          <li><code>AnalyzeDocument</code> with <code>FeatureTypes=["TABLES"]</code></li>
          <li>Rows, columns, cells, spans, bounding boxes</li>
        </ul>
        <div class="log">{step_logs(1)}</div>
      </article>
      <article class="step-card s2">
        <h2>2 · Bedrock LLM</h2>
        <ul>
          <li>Model {_esc(run_meta.get('model_id'))}</li>
          <li>Rebuild multi-level headers / missing regions</li>
          <li>Uses page crop images + Textract extract</li>
        </ul>
        <div class="log">{step_logs(2)}</div>
      </article>
      <article class="step-card s3">
        <h2>3 · Python validate &amp; export</h2>
        <ul>
          <li>Validate structured JSON</li>
          <li>Write HTML, JSON, CSV per table</li>
          <li>Feeds TableRAG / re-index</li>
        </ul>
        <div class="log">{step_logs(3)}</div>
      </article>
    </div>

    {reindex_html}

    <section class="panel">
      <h2>Per-table walkthrough</h2>
      <p class="muted">Left: Textract original · Center: Bedrock reconstructed + validation · Right: crop preview</p>
    </section>
    {''.join(table_cards) or '<p class="muted">No tables extracted.</p>'}

    <section class="panel">
      <h2>Full captured log</h2>
      <div class="log full-log">{''.join(all_logs)}</div>
    </section>
  </div>
</body>
</html>
"""


def _run_reindex(
    *,
    pages: str,
    document_id: str = "",
    all_pages: bool = False,
) -> Dict[str, Any]:
    """Invoke the existing reindex script logic in-process."""
    import importlib.util

    mod_path = ROOT / "scripts" / "reindex_xolair_ste.py"
    spec = importlib.util.spec_from_file_location("reindex_xolair_ste", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {mod_path}")
    reindex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reindex)

    reindex._configure_env(all_pages=bool(all_pages), pages=pages or "")
    # Ensure Textract STE backend even if env had old docling.
    os.environ["PDF_STE_BACKEND"] = "textract"
    settings = reindex._clear_settings()
    settings.PDF_STE_BACKEND = "textract"
    doc = reindex._resolve_doc(document_id.strip() or None)
    if doc is None:
        return {"ok": False, "error": "No xolair document row found"}

    document_id = str(doc.id)
    user_id = str(doc.user_id)
    from app.database import SessionLocal
    from app.models import Document
    from app.services.document_ingestion import ingest_document

    local = reindex._ensure_local_copy(user_id, PDF)
    db = SessionLocal()
    try:
        row = db.query(Document).filter(Document.id == document_id).first()
        row.filename = local.name
        row.original_name = PDF.name
        row.file_size = local.stat().st_size
        row.status = "pending"
        row.error_message = None
        row.is_deleted = False
        row.storage_backend = "local"
        row.s3_key = None
        row.chunk_strategy = "semantic_llm"
        db.commit()
    finally:
        db.close()

    t0 = time.perf_counter()
    logger.info("Re-index ingest starting for %s …", document_id)
    ingest_document(document_id, str(local), PDF.name, user_id)
    elapsed = round(time.perf_counter() - t0, 2)

    report: Dict[str, Any] = {
        "ok": True,
        "document_id": document_id,
        "user_id": user_id,
        "ingest_elapsed_s": elapsed,
        "ste_pages": settings.PDF_STE_PAGES or pages,
    }
    db = SessionLocal()
    try:
        row = db.query(Document).filter(Document.id == document_id).first()
        report["status"] = row.status
        report["page_count"] = row.page_count
        report["chunk_count"] = row.chunk_count
        report["error_message"] = row.error_message
    finally:
        db.close()
    report["inspection"] = reindex._inspect_vectors(document_id, user_id)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", default=DEFAULT_PAGES, help="1-based STE pages")
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Scan whole PDF; AnalyzeDocument still only hits tabular pages",
    )
    parser.add_argument(
        "--no-reindex",
        action="store_true",
        help="Only run STE + write demo HTML (skip Chroma/SQL re-index)",
    )
    parser.add_argument("--document-id", default="", help="Optional document UUID")
    args = parser.parse_args()

    if not PDF.is_file():
        logger.error("Missing PDF: %s", PDF)
        return 1

    if args.all_pages:
        # None → extract_pdf processes every page through the table gate.
        pages = []
        logger.info(
            "Full-document mode: all pages gated; AnalyzeDocument TABLES only "
            "on pages with tabular signal (tables_only=True)"
        )
    else:
        pages = _parse_pages(args.pages)
        if not pages:
            logger.error("No valid pages")
            return 1

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    capture = StepCapture()
    handler = CaptureHandler(capture)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    for name in (
        "ste",
        "ste.pipeline",
        "ste.backends.textract_analyze_backend",
        "ste.ocr_textract_tables",
        "ste.bedrock_reconstructor",
        "ste.validator",
        "demo_ste_xolair",
    ):
        logging.getLogger(name).setLevel(logging.INFO)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ste_out = ROOT / "data" / "ste_runs" / "xolair_demo"
    ste_out.mkdir(parents=True, exist_ok=True)

    logger.info("=== STE demo start pages=%s ===", pages)
    run_meta = _run_ste(pages=pages, capture=capture, out_root=ste_out)
    run_dir = Path(run_meta["run_dir"])
    tables = _collect_tables(run_dir)
    logger.info("Collected %s tables from %s", len(tables), run_dir)

    reindex_report = None
    if not args.no_reindex:
        try:
            reindex_report = _run_reindex(
                pages="" if args.all_pages else ",".join(str(p) for p in pages),
                document_id=args.document_id,
                all_pages=bool(args.all_pages),
            )
            capture.events.append(
                LogEvent(
                    time.time(),
                    "INFO",
                    "demo_ste_xolair",
                    f"Re-index done status={reindex_report.get('status')} "
                    f"chunks={reindex_report.get('chunk_count')} "
                    f"ste_tables={((reindex_report.get('inspection') or {}).get('chroma_ste_table_chunks'))}",
                    3,
                )
            )
        except Exception as exc:
            logger.exception("Re-index failed: %s", exc)
            reindex_report = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    html_path = OUT_DIR / f"xolair_ste_demo_{int(time.time())}.html"
    html_path.write_text(
        _render_demo_html(
            run_meta=run_meta,
            tables=tables,
            capture=capture,
            reindex_report=reindex_report,
        ),
        encoding="utf-8",
    )
    meta_path = html_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "run_meta": {
                    k: v
                    for k, v in run_meta.items()
                    if k != "meta"
                },
                "backend_meta": (run_meta.get("meta") or {}).get("backend_meta"),
                "bedrock_usage": (run_meta.get("meta") or {}).get("bedrock_usage"),
                "table_ids": [t["table_id"] for t in tables],
                "reindex": reindex_report,
                "html": str(html_path),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print("\n=== STE Xolair demo ===")
    print(f"tables: {run_meta.get('table_count')}")
    print(f"elapsed: {run_meta.get('elapsed_s')}s")
    print(f"run_dir: {run_dir}")
    print(f"demo_html: {html_path}")
    if reindex_report:
        print(
            f"reindex: status={reindex_report.get('status')} "
            f"chunks={reindex_report.get('chunk_count')} "
            f"doc={reindex_report.get('document_id')}"
        )
    return 0 if int(run_meta.get("table_count") or 0) >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

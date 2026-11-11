"""Retrieval + context-assembly eval for all ready PDFs except XOLAIR."""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.database import SessionLocal
from app.models import Document
from app.rag.llm.agent import _format_context, _sources_from_chunks
from app.rag.retrieval import retriever as retriever_mod
from app.rag.retrieval.retriever import retrieve

OUT = Path(__file__).resolve().parents[1] / "_all_pdfs_retrieval_eval.json"
USER = "f3d989fb-79af-4fe2-9360-12b06772348c"
TOP_K = 8
SKIP_NAME_SUBSTR = ("xolair",)


def _joined(chunks: List[Dict[str, Any]], n: int = 8) -> str:
    return "\n".join((c.get("text") or "") for c in chunks[:n]).lower()


def _types(chunks: List[Dict[str, Any]]) -> List[str]:
    return [
        str(c.get("chunk_type") or c.get("content_type") or "unknown").lower()
        for c in chunks
    ]


def _any_one(chunks: List[Dict[str, Any]], *needles: str, top_n: int = 8) -> bool:
    blob = _joined(chunks, top_n)
    return any(n.lower() in blob for n in needles)


def _question_bank() -> Dict[str, List[Tuple[str, List[str], bool]]]:
    """filename substring -> list of (question, needles, want_table)."""
    return {
        "01_vendor_invoice_goods": [
            ("What is the vendor / seller name on this tax invoice?", ["yenpu", "invoice"], False),
            ("What is the taxable value on the invoice?", ["taxable", "7400000", "rs"], True),
            ("What is the CGST amount and rate?", ["cgst", "9%", "666000"], True),
            ("What is the SGST amount?", ["sgst", "666000"], True),
            ("What is the grand total?", ["grand", "total"], True),
            ("List the goods / line items described on the invoice.", ["description", "hsn", "qty"], True),
            ("What is the HSN code on the invoice line items?", ["hsn"], True),
            ("Is this invoice charged with CGST and SGST?", ["cgst", "sgst"], True),
            ("What is the quantity and unit of measure for the goods?", ["qty", "uom"], True),
            ("Extract rate and taxable amount for the goods line.", ["rate", "taxable"], True),
            ("Who issued the tax invoice?", ["yenpu", "limited"], False),
            ("Summarize all tax components shown on the invoice.", ["cgst", "sgst", "taxable"], True),
            ("Is GST shown as CGST+SGST or IGST on this invoice?", ["cgst", "sgst"], True),
            ("What currency amounts appear near Grand Total?", ["grand", "rs"], True),
            ("Find table facts or tax breakdown cells for CGST/SGST.", ["cgst", "sgst"], True),
            ("What is the invoice document type heading?", ["tax invoice"], False),
            ("Are there multiple goods lines or a single goods description?", ["description", "goods"], False),
            ("What percentage CGST is applied?", ["9%", "cgst"], True),
            ("What percentage SGST is applied?", ["9%", "sgst"], True),
            ("Provide the full tax calculation path taxable → CGST → SGST → total.", ["taxable", "cgst", "sgst"], True),
        ],
        "03_vendor_invoice_nonpo": [
            ("What is the vendor name on this non-PO invoice?", ["techprint", "invoice"], False),
            ("What is the GSTIN on the invoice?", ["gstin", "36aadft"], False),
            ("What is the PAN mentioned?", ["pan", "aadft"], False),
            ("What is the vendor address / city?", ["hyderabad", "madhapur"], False),
            ("What phone number is listed?", ["phone", "+91"], False),
            ("Is this labeled as a TAX INVOICE?", ["tax invoice"], False),
            ("Extract key header fields from the invoice table.", ["tax invoice", "techprint"], True),
            ("What state is the vendor in?", ["telangana"], False),
            ("Summarize the invoice header party details.", ["techprint", "gstin"], False),
            ("Find shop / floor address details.", ["shop", "ground floor"], False),
            ("What is the PIN / postal code?", ["500081"], False),
            ("Does the document contain a structured invoice table?", ["tax invoice"], True),
            ("List contact details present on the invoice.", ["phone", "gstin"], False),
            ("Who is the supplier for this FB60 / non-PO invoice?", ["techprint"], False),
            ("Extract any invoice title cells from the table.", ["tax invoice"], True),
            ("Is Madhapur mentioned as the location?", ["madhapur"], False),
            ("What legal business name appears at the top?", ["techprint solutions"], False),
            ("Find GSTIN and PAN together from the header.", ["gstin", "pan"], False),
            ("What kind of document is this (invoice type)?", ["invoice"], False),
            ("Pull the main identifying fields useful for AP posting.", ["gstin", "techprint", "invoice"], False),
        ],
        "ai_engineer_roadmap": [
            ("What is the title of this roadmap document?", ["ai engineer", "roadmap", "2026"], False),
            ("How long is the mastery roadmap (months)?", ["6-month", "6 month"], False),
            ("What does week 21 cover in the roadmap table?", ["week 21", "agent"], True),
            ("Find agent planning / tool calling topics in the schedule.", ["agent", "tool"], True),
            ("What is the overall goal of the 6-month plan?", ["production-ready", "ai"], False),
            ("List a few weekly themes from the roadmap table.", ["week", "roadmap"], True),
            ("Does the roadmap mention agentic workflows?", ["agentic", "workflow"], True),
            ("What skills are emphasized for production AI systems?", ["production", "ai"], False),
            ("Summarize early weeks focus if present.", ["week", "roadmap"], False),
            ("Is there a table structuring the weekly plan?", ["week", "table"], True),
            ("What year is the roadmap targeting?", ["2026"], False),
            ("Find mentions of reasoning or tool calling.", ["reasoning", "tool"], False),
            ("Extract schedule rows related to agents.", ["agent", "week"], True),
            ("What does 'Building Production-Ready AI Systems' refer to?", ["production-ready", "2026"], False),
            ("Are workflows mentioned in later weeks?", ["workflow", "week"], True),
            ("Give the document subtitle / tagline.", ["mastery", "roadmap"], False),
            ("Find any figure caption on page 1.", ["roadmap", "figure"], False),
            ("Which week numbers appear in the retrieved plan?", ["week"], True),
            ("Does the plan include tool calling practice?", ["tool calling", "tool"], True),
            ("Summarize what an AI engineer would learn from this PDF.", ["ai engineer", "roadmap"], False),
        ],
        "graph1": [
            ("What does Graph1 say about graph database scaling?", ["graph", "scaling"], False),
            ("How do graph databases optimize relationship queries?", ["relationship", "graph"], False),
            ("What is mentioned about flexibility in graph data?", ["flexibility", "graph"], False),
            ("Find discussion of sub-graph concepts.", ["sub-graph", "graph"], False),
            ("What future work is suggested regarding computational overhead?", ["overhead", "memory"], False),
            ("Are compression methods discussed for graph models?", ["compression", "graph"], False),
            ("Summarize retrieval advantages of graph databases.", ["retrieval", "graph"], False),
            ("What efficiency concerns are raised?", ["efficiency", "graph"], False),
            ("Find knowledge-related graph statements.", ["knowledge", "graph"], False),
            ("Does the paper discuss faster retrieval via relationships?", ["relationship", "retrieval"], False),
            ("Extract a claim about graph database optimization.", ["optimize", "graph"], False),
            ("What limitations or future investigations are proposed?", ["future work", "investigate"], False),
            ("Find text about memory requirements.", ["memory", "requirements"], False),
            ("Is image/sub-graph structure mentioned?", ["image", "graph"], False),
            ("Summarize the scaling-up narrative for graph DBs.", ["scaling", "graph"], False),
            ("What does the document say about comparable methods?", ["comparable", "compression"], False),
            ("Locate discussion of relationship-based queries.", ["relationship-based", "queries"], False),
            ("Are graph databases said to optimize for relationships?", ["optimize", "relationship"], False),
            ("Find any note on reducing computational overhead.", ["reduce", "overhead"], False),
            ("Give key takeaways about graph data flexibility.", ["flexibility", "graph data"], False),
        ],
        "graph2": [
            ("What framework is described for representing graph data components?", ["graph data", "framework"], False),
            ("How do researchers explore code-like formats for graph structure?", ["code", "graph"], False),
            ("What does section 7.2.1 discuss?", ["7.2.1", "graph"], False),
            ("Why use code-like formats with language models for graphs?", ["code", "lm"], False),
            ("Find discussion of 2-D structure of graph data.", ["2-d", "graph"], False),
            ("What uniform syntax/semantic framework is prescribed?", ["syntax", "semantic"], False),
            ("Summarize LM code comprehension applied to graphs.", ["code comprehension", "graph"], False),
            ("Are language models mentioned for graph representation?", ["language", "graph"], False),
            ("Extract claims about representing graph components.", ["components", "graph"], False),
            ("Find references to researchers exploring formats.", ["researchers", "formats"], False),
            ("What is specifically crafted to characterize graph data?", ["characterize", "graph"], False),
            ("Discuss robustness of code comprehension capabilities.", ["comprehension", "capabilities"], False),
            ("Locate text about uniform semantic framework.", ["uniform", "semantic"], False),
            ("How is graph data representation approached in this paper?", ["representing", "graph data"], False),
            ("Find mentions of code-like formats.", ["code-like", "formats"], False),
            ("What section numbers appear near graph representation discussion?", ["7.2", "graph"], False),
            ("Summarize why syntax frameworks matter for graphs.", ["syntax", "graph"], False),
            ("Are LMs used to explore graph structure encodings?", ["lm", "graph"], False),
            ("Extract a sentence about 2-D graph structure.", ["2-d structure", "graph"], False),
            ("Give the main idea of graph data characterization here.", ["characterize", "represent"], False),
        ],
        "avastin_prescribing": [
            ("What is Avastin indicated for in hepatocellular carcinoma?", ["hepatocellular", "atezolizumab"], False),
            ("What doses of Avastin were used every 2 or 3 weeks in studies?", ["10 mg/kg", "15 mg/kg"], False),
            ("Find dosing schedule every 2 weeks vs every 3 weeks.", ["every 2 weeks", "every 3 weeks"], False),
            ("What combination partner is mentioned with Avastin for HCC?", ["atezolizumab", "avastin"], False),
            ("Is Avastin indicated for unresectable or metastatic HCC?", ["unresectable", "metastatic"], False),
            ("Summarize a clinical study randomization involving Avastin doses.", ["randomized", "avastin"], False),
            ("What prior chemotherapy constraint appears in a study description?", ["prior", "chemotherapy"], False),
            ("Locate indication section text for hepatocellular carcinoma.", ["1.7", "hepatocellular"], False),
            ("What mg/kg strengths are described?", ["mg/kg"], False),
            ("Find safety or study references near Avastin dosing.", ["patients", "avastin"], False),
            ("Does the label mention atezolizumab combination?", ["atezolizumab"], False),
            ("Extract recommended or studied dosing intervals.", ["every", "weeks"], False),
            ("What disease area is covered under indication 1.7?", ["hepatocellular carcinoma"], False),
            ("Find text about metastatic hepatocellular carcinoma.", ["metastatic", "hepatocellular"], False),
            ("How frequently was 15 mg/kg Avastin given in the study?", ["15 mg/kg", "every 3 weeks"], False),
            ("How frequently was 10 mg/kg Avastin given?", ["10 mg/kg", "every 2 weeks"], False),
            ("Summarize key HCC indication sentence.", ["avastin", "hepatocellular"], False),
            ("Are patients with unresectable HCC discussed?", ["unresectable"], False),
            ("Find chemotherapy lines / prior therapy limits if stated.", ["prior", "chemo"], False),
            ("Pull dosing numbers useful for a prescribing question.", ["mg/kg", "weeks"], False),
        ],
        "calculus-volume-1": [
            ("Who are the senior contributing authors of Calculus Volume 1?", ["strang", "herman"], False),
            ("What course is this calculus book designed for?", ["calculus", "semester"], False),
            ("Is this Volume 1 of the calculus series?", ["volume", "1"], False),
            ("What innovative features are mentioned for student learning?", ["student learning", "calculus"], False),
            ("Find the book title and volume number.", ["calculus volume 1"], False),
            ("Which universities are associated with the authors?", ["massachusetts", "university"], False),
            ("Summarize the purpose of the calculus textbook.", ["calculus", "course"], False),
            ("Does the preface mention two- or three-semester courses?", ["two-", "three-semester"], False),
            ("Extract author names Gilbert Strang and Edwin Herman.", ["gilbert strang", "herman"], False),
            ("What subject does this PDF teach?", ["calculus"], False),
            ("Find front-matter description of Calculus Volume 1.", ["calculus", "volume"], False),
            ("Is MIT mentioned with an author affiliation?", ["massachusetts", "institute"], False),
            ("What learning enhancements does the book claim?", ["enhance", "learning"], False),
            ("Identify the document as an OpenStax-style calculus text if stated.", ["calculus"], False),
            ("Give a one-line description from early pages.", ["calculus", "designed"], False),
            ("Who is Edwin Jed Herman according to the cover/front matter?", ["herman"], False),
            ("Find 'SENIOR CONTRIBUTING AUTHORS' heading content.", ["senior contributing authors"], False),
            ("What volume number is printed prominently?", ["volume", "1"], False),
            ("Summarize intended audience for the book.", ["student", "calculus"], False),
            ("Extract any statement about general calculus course design.", ["general calculus", "course"], False),
        ],
        "gartley-hm-profits": [
            ("What does the appendix say about Firestone vs U.S. Rubber management?", ["firestone", "rubber"], False),
            ("Which company had greater gross sales according to the text?", ["u.s. rubber", "gross sales"], False),
            ("Find the Investograph method appendix title.", ["investograph", "appraising"], True),
            ("Compare Firestone and U.S. Rubber from the narrative.", ["firestone", "rubber"], False),
            ("Is management quality discussed relative to sales?", ["management", "sales"], False),
            ("Extract table headers related to Firestone and U.S. Rubber.", ["firestone", "rubber"], True),
            ("What is APPENDIX 2 about?", ["appendix", "investograph"], False),
            ("Find appraisal of security values method name.", ["investograph", "security"], True),
            ("Does the text claim Firestone management was superior?", ["firestone", "superior"], False),
            ("Summarize the comparison conclusion between the two companies.", ["firestone", "rubber"], False),
            ("Locate numeric/table appraisal content for the companies.", ["firestone", "u.s. rubber"], True),
            ("What method is used for appraising security values?", ["investograph", "appraising"], False),
            ("Find mentions of gross sales in the comparison.", ["gross sales"], False),
            ("Is U.S. Rubber said to enjoy greater gross sales?", ["greater gross sales"], False),
            ("Extract a key investing/appraisal takeaway.", ["security", "values"], False),
            ("Does a table present both U.S. Rubber and Firestone columns?", ["u.s. rubber", "firestone"], True),
            ("Identify the appendix number discussing Investograph.", ["appendix 2"], False),
            ("Find qualitative judgment about management superiority.", ["management", "superior"], False),
            ("Pull any tabular cells useful for company comparison.", ["firestone", "rubber"], True),
            ("Summarize why Firestone is contrasted with U.S. Rubber.", ["firestone", "rubber"], False),
        ],
        "upsc-text": [
            ("What exam year branding appears in the UPSC prelims material?", ["upsc", "2026"], False),
            ("Find a multiple-choice answer key pattern like Answer: (c).", ["answer:", "(c)"], False),
            ("What website/brand is associated with UnlockIAS?", ["unlockias"], False),
            ("Locate a question with options (a)(b)(c)(d).", ["(a)", "(b)", "(c)", "(d)"], False),
            ("Is this document related to UPSC prelims PYQ?", ["upsc", "prelims"], False),
            ("Extract a sample MCQ answer letter from the text.", ["answer"], False),
            ("Find page markers such as Page 15 content.", ["upsc", "page"], False),
            ("What does UnlockIAS publish in this PDF?", ["upsc", "prelims"], False),
            ("Locate option-matching questions with codes like 3 1 2 4.", ["3 1 2 4", "answer"], False),
            ("Summarize the document type (PYQ practice).", ["upsc", "pyq"], False),
            ("Find www.unlockias.in reference.", ["unlockias.in"], False),
            ("Are answer keys explicitly labeled?", ["answer:"], False),
            ("Extract any UPSC 2026 heading text.", ["upsc 2026"], False),
            ("Find a four-option matching question block.", ["(a)", "(d)"], False),
            ("What practice exam franchise branding appears?", ["unlockias"], False),
            ("Locate prelims PYQ wording.", ["prelims-pyq", "prelims"], False),
            ("Give an example of a selected answer choice from the PDF.", ["answer: (c)"], False),
            ("Is option order permutation used in matching questions?", ["(a)", "3 1 2 4"], False),
            ("Find text useful for verifying an MCQ answer.", ["answer", "upsc"], False),
            ("Summarize what a student would use this PDF for.", ["upsc", "prelims"], False),
        ],
    }


def _match_bank_key(filename: str) -> Optional[str]:
    n = filename.lower()
    for key in _question_bank():
        if key.replace("_", "") in n.replace("_", "").replace("-", "").replace(" ", ""):
            return key
        # softer contains
        token = key.split("_")[0]
        if token and token in n:
            # disambiguate invoices
            if key.startswith("01_") and "01_" in n:
                return key
            if key.startswith("03_") and "03_" in n:
                return key
            if key == "graph1" and "graph1" in n:
                return key
            if key == "graph2" and "graph2" in n:
                return key
            if key.startswith("ai_") and "ai_engineer" in n:
                return key
            if key.startswith("avastin") and "avastin" in n:
                return key
            if key.startswith("calculus") and "calculus" in n:
                return key
            if key.startswith("gartley") and "gartley" in n:
                return key
            if key.startswith("upsc") and "upsc" in n:
                return key
    return None


def _list_target_docs() -> List[Dict[str, str]]:
    db = SessionLocal()
    try:
        rows = (
            db.query(Document)
            .filter(
                Document.user_id == USER,
                Document.is_deleted.is_(False),
                Document.status == "ready",
            )
            .order_by(Document.original_name.asc())
            .all()
        )
        out = []
        for d in rows:
            name = d.original_name or ""
            if any(s in name.lower() for s in SKIP_NAME_SUBSTR):
                continue
            key = _match_bank_key(name)
            if not key:
                continue
            out.append({"id": str(d.id), "name": name, "bank_key": key})
        return out
    finally:
        db.close()


def _assess_context_assembly(chunks: List[Dict[str, Any]], filename: str) -> Dict[str, Any]:
    context = _format_context(chunks)
    sources = _sources_from_chunks(chunks)
    checks = {
        "non_empty": bool(context.strip()),
        "has_numbered_sources": bool(re.search(r"\[1\] Source:", context)),
        "has_filename": filename.lower() in context.lower() if filename else True,
        "has_page_labels": "Page " in context,
        "chunk_count_matches_sources": len(sources) == len(chunks),
        "no_empty_chunk_bodies": all(bool((c.get("text") or "").strip()) for c in chunks),
        "type_meta_when_available": True,
    }
    # If any chunk has type, assembled context should expose Type:
    if any((c.get("chunk_type") or c.get("content_type")) for c in chunks):
        checks["type_meta_when_available"] = "Type:" in context

    # Length sanity: enough signal, not absurdly empty.
    checks["min_length_ok"] = len(context) >= 80
    checks["max_length_ok"] = len(context) <= 120_000

    passed = sum(1 for v in checks.values() if v)
    return {
        "ok": all(checks.values()),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "context_chars": len(context),
        "context_preview": context[:500],
        "source_count": len(sources),
    }


def _score_case(
    *,
    doc: Dict[str, str],
    qid: str,
    question: str,
    needles: List[str],
    want_table: bool,
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    hits = [n for n in needles if _any_one(chunks, n)]
    missed = [n for n in needles if n not in hits]
    table_ok = True
    if want_table:
        table_ok = (
            any("table" in t for t in _types(chunks))
            or _any_one(chunks, "table facts:", "tax invoice", "| --- |")
        )
    needle_ok = len(hits) >= max(1, (len(needles) + 1) // 2)
    assembly = _assess_context_assembly(chunks, doc["name"])
    return {
        "id": qid,
        "document_id": doc["id"],
        "document_name": doc["name"],
        "category": doc["bank_key"],
        "question": question,
        "want_table": want_table,
        "retrieved_count": len(chunks),
        "chunk_type_counts": dict(Counter(_types(chunks))),
        "needle_hits": hits,
        "needle_missed": missed,
        "table_ok": table_ok,
        "needle_ok": needle_ok,
        "retrieval_ok": bool(chunks) and needle_ok and table_ok,
        "context_assembly": assembly,
        "ok": bool(chunks) and needle_ok and table_ok and assembly["ok"],
        "top_hits": [
            {
                "rank": i,
                "page": c.get("page"),
                "chunk_type": c.get("chunk_type") or c.get("content_type"),
                "score": c.get("score"),
                "boost_reason": c.get("boost_reason"),
                "text_preview": re.sub(r"\s+", " ", (c.get("text") or ""))[:200],
            }
            for i, c in enumerate(chunks[:5], 1)
        ],
    }


def main() -> None:
    retriever_mod._generate_query_variants = lambda _q: []  # type: ignore[assignment]
    docs = _list_target_docs()
    bank = _question_bank()
    print(f"docs={len(docs)} -> {[d['name'] for d in docs]}", flush=True)

    results: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    total_q = sum(len(bank[d["bank_key"]]) for d in docs)
    done = 0

    for doc in docs:
        cases = bank[doc["bank_key"]]
        for idx, (question, needles, want_table) in enumerate(cases, 1):
            done += 1
            qid = f"{doc['bank_key']}_Q{idx:02d}"
            print(f"[{done}/{total_q}] {qid} ...", flush=True)
            started = time.perf_counter()
            try:
                chunks = retrieve(
                    question,
                    user_id=USER,
                    document_id=doc["id"],
                    top_k=TOP_K,
                )
                scored = _score_case(
                    doc=doc,
                    qid=qid,
                    question=question,
                    needles=needles,
                    want_table=want_table,
                    chunks=chunks,
                )
                scored["elapsed_sec"] = round(time.perf_counter() - started, 2)
                scored["error"] = None
            except Exception as exc:  # noqa: BLE001
                scored = {
                    "id": qid,
                    "document_id": doc["id"],
                    "document_name": doc["name"],
                    "ok": False,
                    "retrieval_ok": False,
                    "error": str(exc),
                    "elapsed_sec": round(time.perf_counter() - started, 2),
                    "context_assembly": {"ok": False},
                }
            status = "PASS" if scored.get("ok") else "FAIL"
            ca = (scored.get("context_assembly") or {}).get("ok")
            print(
                f"  {status} retrieval={scored.get('retrieval_ok')} "
                f"assembly={ca} n={scored.get('retrieved_count')} "
                f"miss={scored.get('needle_missed')}",
                flush=True,
            )
            results.append(scored)
            if done % 15 == 0:
                _write(results, docs, t0, partial=True)

    payload = _write(results, docs, t0, partial=False)
    summary = {
        "saved": str(OUT),
        "documents": payload["documents_summary"],
        "n_questions": payload["n_questions"],
        "retrieval_pass": payload["retrieval_pass"],
        "retrieval_pass_rate": payload["retrieval_pass_rate"],
        "assembly_pass": payload["assembly_pass"],
        "assembly_pass_rate": payload["assembly_pass_rate"],
        "overall_pass": payload["overall_pass"],
        "overall_pass_rate": payload["overall_pass_rate"],
        "elapsed_sec": payload["elapsed_sec"],
        "sample_failures": payload["sample_failures"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _write(
    results: List[Dict[str, Any]],
    docs: List[Dict[str, str]],
    t0: float,
    *,
    partial: bool,
) -> Dict[str, Any]:
    retrieval_pass = sum(1 for r in results if r.get("retrieval_ok"))
    assembly_pass = sum(
        1 for r in results if (r.get("context_assembly") or {}).get("ok")
    )
    overall_pass = sum(1 for r in results if r.get("ok"))
    by_doc: Dict[str, Dict[str, Any]] = {}
    for r in results:
        name = r.get("document_name") or "?"
        slot = by_doc.setdefault(
            name,
            {"n": 0, "retrieval_pass": 0, "assembly_pass": 0, "overall_pass": 0},
        )
        slot["n"] += 1
        slot["retrieval_pass"] += int(bool(r.get("retrieval_ok")))
        slot["assembly_pass"] += int(bool((r.get("context_assembly") or {}).get("ok")))
        slot["overall_pass"] += int(bool(r.get("ok")))

    fails = [r for r in results if not r.get("ok")]
    payload = {
        "user_id": USER,
        "top_k": TOP_K,
        "excluded": list(SKIP_NAME_SUBSTR),
        "partial": partial,
        "documents": docs,
        "n_documents": len(docs),
        "n_questions": len(results),
        "retrieval_pass": retrieval_pass,
        "retrieval_pass_rate": round(retrieval_pass / max(len(results), 1), 3),
        "assembly_pass": assembly_pass,
        "assembly_pass_rate": round(assembly_pass / max(len(results), 1), 3),
        "overall_pass": overall_pass,
        "overall_pass_rate": round(overall_pass / max(len(results), 1), 3),
        "documents_summary": by_doc,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "sample_failures": [
            {
                "id": r.get("id"),
                "document": r.get("document_name"),
                "retrieval_ok": r.get("retrieval_ok"),
                "assembly_ok": (r.get("context_assembly") or {}).get("ok"),
                "missed": r.get("needle_missed"),
                "assembly_checks": (r.get("context_assembly") or {}).get("checks"),
                "top1": (r.get("top_hits") or [None])[0],
            }
            for r in fails[:20]
        ],
        "results": results,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()

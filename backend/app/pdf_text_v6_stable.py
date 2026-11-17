# -*- coding: utf-8 -*-
"""
lambda_function.py  —  Pharma PDF Extraction Pipeline
=======================================================
ARCHITECTURE: SQS  →  Lambda (trigger)  →  Step Functions  →  Lambda (tasks)

The single Lambda handles FOUR task modes, routed by the event["task"] key
set by Step Functions, plus the SQS/S3/manual entry points:

  task = "render"          → render_task()    PyMuPDF + parallel Textract
  task = "classify"        → classify_task()  Prompt 1 (classification)
  task = "extract_prompt"  → extract_task()   One prompt (called 7× in Map state)
  task = "merge"           → merge_task()     Build final JSON + notify Archilles
  (no task key)            → entry point: SQS / direct S3 / manual test

KEY IMPROVEMENTS over the previous version:
  1. Textract runs in PARALLEL across all pages  (ThreadPoolExecutor)
  2. Prompts 2-8 run in PARALLEL via Step Functions Map state
  3. Each step has its own timeout + 3-retry backoff — no silent failures
  4. SQS visibility timeout is irrelevant: trigger Lambda returns in <5s
  5. Lambda memory 3008 MB gives 3 vCPUs for faster PyMuPDF rendering
  6. In-memory cache (_CONFIG_CACHE) skips redundant S3 reads on warm containers
  7. bedrock_call_with_backoff handles ThrottlingException with jittered backoff

v7 — COMPLETENESS CHANGES (no data capping):
  8. List prompts that produce many rows (labs by default) run on PAGE WINDOWS
     in parallel and the results are merged + de-duplicated, so the model never
     has to emit hundreds of rows in one answer.          → extract_by_page_windows()
  9. When a needle prompt (drug/event/medical condition/lab) is truncated, it is
     re-run in NEEDLE BATCHES over the FULL document, plus one "remainder" call
     for items the classifier missed.                      → extract_by_needle_batches()
 10. The cached Bedrock path now reports stop_reason, so truncation is detected
     on both paths.                                        → _bedrock_invoke_cached()
 11. CIOMS "13. Lab Data" is read deterministically (row numbers 1..N are
     verified) and the LLM handles only the remaining text. → parse_cioms_lab_table()
 12. A completeness report is written per list prompt.   → completeness_<prompt>.json
 13. render_task also saves per-page text (raw_pages.json) for page windowing.
 14. Fixed classify_chunked cut-point bug (rfind() == -1 is truthy).
"""

import os
import json
import random
import boto3
import urllib.parse
import urllib.request
import base64
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.config import Config
import pymupdf as fitz
import time

# ====================================================================
# Configuration
# ====================================================================

BUCKET                = os.getenv("BUCKET_NAME", "eai-genova")

# ====================================================================
# Multi-tenant folder layout
# ====================================================================
# Every client shares the same folder skeleton under clients/<client_id>/:
#
#   clients/<client_id>/pdf/              ← incoming PDFs (S3 trigger folder)
#   clients/<client_id>/raw-json/         ← rendered text, intermediate + final JSON
#   clients/<client_id>/custom-prompts/   ← prompt files
#   clients/<client_id>/run-book/configuration.json
#   clients/<client_id>/sh-json/...       ← display-standard files (paths for
#                                            these come FROM configuration.json,
#                                            they are not built here)
#
# client_id is derived at runtime from the S3 key that triggered the pipeline
# (e.g. "clients/pharma_1/pdf/x.pdf" → "pharma_1", "clients/dev/pdf/x.pdf" →
# "dev"). Onboarding a new client is just dropping a configuration.json +
# prompts under a new clients/<new_client>/ prefix — no code change needed.
CLIENTS_ROOT_FOLDER   = os.getenv("CLIENTS_ROOT_FOLDER", "clients/")
PDF_SUBFOLDER_NAME    = os.getenv("PDF_SUBFOLDER_NAME", "pdf")        # renamed from "input-pdf"
RAW_JSON_SUBFOLDER    = os.getenv("RAW_JSON_SUBFOLDER", "raw-json")
PROMPTS_SUBFOLDER     = os.getenv("PROMPTS_SUBFOLDER", "custom-prompts")
CONFIG_SUBFOLDER      = os.getenv("CONFIG_SUBFOLDER", "run-book")
CONFIG_FILE_NAME      = os.getenv("CONFIG_FILE_NAME", "configuration.json")

# Matches clients/<client_id>/pdf/<anything> and captures <client_id>.
_CLIENT_ID_RE = re.compile(
    rf"^{re.escape(CLIENTS_ROOT_FOLDER)}([^/]+)/{re.escape(PDF_SUBFOLDER_NAME)}/"
)


def input_pdf_folder(client_id: str) -> str:
    return f"{CLIENTS_ROOT_FOLDER}{client_id}/{PDF_SUBFOLDER_NAME}/"

def raw_json_folder(client_id: str) -> str:
    return f"{CLIENTS_ROOT_FOLDER}{client_id}/{RAW_JSON_SUBFOLDER}/"

def prompts_folder(client_id: str) -> str:
    return f"{CLIENTS_ROOT_FOLDER}{client_id}/{PROMPTS_SUBFOLDER}/"

def configuration_key(client_id: str) -> str:
    return f"{CLIENTS_ROOT_FOLDER}{client_id}/{CONFIG_SUBFOLDER}/{CONFIG_FILE_NAME}"


def get_client_id_from_pdf_key(key: str) -> str:
    """
    STRICT: derive client_id from an INPUT PDF key, validating it landed in
    the expected trigger folder: clients/<client_id>/pdf/<file>.pdf
    Raises ValueError if the key doesn't match this layout, so callers can
    reject/DLQ an unexpected upload instead of silently guessing a client.
    """
    m = _CLIENT_ID_RE.match(key)
    if not m:
        raise ValueError(
            f"Could not determine client_id from key '{key}' — expected "
            f"'{CLIENTS_ROOT_FOLDER}<client_id>/{PDF_SUBFOLDER_NAME}/<file>.pdf'"
        )
    return m.group(1)


def get_client_id_from_path(path: str) -> str:
    """
    LENIENT: derive client_id from any S3 key/prefix of the form
    clients/<client_id>/... Used for prefixes WE constructed ourselves
    (out_prefix under raw-json/, etc.) as a self-contained fallback when
    client_id wasn't explicitly carried through Step Functions state.
    """
    m = re.match(rf"^{re.escape(CLIENTS_ROOT_FOLDER)}([^/]+)/", path)
    if not m:
        raise ValueError(
            f"Could not determine client_id from path '{path}' — expected it "
            f"to start with '{CLIENTS_ROOT_FOLDER}<client_id>/'"
        )
    return m.group(1)


def list_client_ids() -> list:
    """Discover onboarded client_ids by listing common prefixes under clients/."""
    ids  = []
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=CLIENTS_ROOT_FOLDER, Delimiter="/")
    for cp in resp.get("CommonPrefixes", []):
        cid = cp["Prefix"][len(CLIENTS_ROOT_FOLDER):].strip("/")
        if cid:
            ids.append(cid)
    return ids


STEP_FUNCTIONS_ARN    = os.getenv("STEP_FUNCTIONS_ARN", "")   # set in Lambda env vars

FORM_ID               = None
RENDER_DPI            = 300
PRESIGNED_EXPIRY      = 604800

BEDROCK_REGION        = os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "us-east-1"))

_BEDROCK_TIMEOUT_CFG  = Config(read_timeout=3600, connect_timeout=10,
                                retries={"max_attempts": 1})   # backoff handled in code
_TEXTRACT_TIMEOUT_CFG = Config(read_timeout=60,   connect_timeout=10,
                                retries={"max_attempts": 2})

TEMPERATURE           = float(os.getenv("MODEL_TEMPERATURE", "0.0"))
# top_p = 1.0 with temperature 0 = greedy decoding (always pick the single most
# likely token). top_p = 0.1 with temp 0 is an UNSTABLE regime: the nucleus is so
# narrow that tiny floating-point differences (inherent to Bedrock's fleet) flip
# which tokens survive the cutoff, amplifying run-to-run variance. For
# deterministic extraction, top_p MUST be 1.0 so greedy decoding is clean.
TOP_P                 = float(os.getenv("MODEL_TOP_P", "1.0"))
CONTEXT_WINDOW        = int(os.getenv("CTX_WINDOW", "1000000"))
OUTPUT_HARD_CAP       = int(os.getenv("OUTPUT_HARD_CAP", "32000"))
# Bedrock retry budget — raised from 5 to 8 for high-concurrency (50+ docs).
# At 8 attempts the cumulative backoff window is ~2 minutes, enough to ride out
# a throttle storm when 400 calls compete for the account RPM quota.
BEDROCK_MAX_ATTEMPTS  = int(os.getenv("BEDROCK_MAX_ATTEMPTS", "8"))


class _TransientBedrockError(Exception):
    """
    Raised when a Bedrock call fails for a TRANSIENT reason (throttle, timeout,
    service unavailable) after exhausting in-process retries. Distinguished from
    parse/size errors so classify_smart can propagate it for Step Functions
    retry instead of silently switching to the chunked path (which would change
    the extracted field count and cause run-to-run inconsistency).
    """
    pass


class _TruncatedOutput(ValueError):
    """
    v7: raised when the model hit max_tokens. Carries the best-effort repaired
    partial result so callers can split the work (page windows / needle batches)
    and, only as a last resort, keep the partial data instead of re-calling.
    """
    def __init__(self, msg: str, partial=None):
        super().__init__(msg)
        self.partial = partial or {}


SAFETY_RESERVE        = int(os.getenv("SAFETY_RESERVE", "4000"))
# USER_MAX_TOKENS_CAP raised to 32000: the data_classification prompt emits
# 53 fields (~4.5K tokens) but other prompts (event, lab tables) can emit far
# more. A too-low cap silently truncates the JSON mid-output.
USER_MAX_TOKENS_CAP   = int(os.getenv("MODEL_MAX_TOKENS_CAP", "32000"))
HIGH_OUTPUT_TOKENS    = int(os.getenv("HIGH_OUTPUT_TOKENS", "32000"))

# ── Stability toggles ────────────────────────────────────────────────
# ENABLE_PROMPT_CACHE: prompt caching reorders the message so raw_text comes
#   BEFORE the prompt instruction (required for prefix caching). Nova 2 Lite
#   under-attends to the document in this order and returns empty fields.
#   DEFAULT OFF — uses the proven PROMPT-first format that matches the
#   Bedrock Playground. Only enable after validating extraction quality.
ENABLE_PROMPT_CACHE   = os.getenv("ENABLE_PROMPT_CACHE", "false").lower() in ("1", "true", "yes")
# KEEP_INTERMEDIATE_JSON: when true, per-prompt Bedrock JSON outputs are NOT
#   deleted after merge — useful for validating raw model responses in S3.
KEEP_INTERMEDIATE_JSON = os.getenv("KEEP_INTERMEDIATE_JSON", "true").lower() in ("1", "true", "yes")

# HIGH_OUTPUT_PROMPTS: matched by base name (version-agnostic).
# Uses _prompt_base_name() which strips _vN suffix before checking.
# Add base names of any prompts that produce large outputs (32K tokens).
HIGH_OUTPUT_PROMPT_BASES = {
    "patient_laboratory_prompt",
    "event_prompt",
    "data_classification_prompt",
    "drug_prompt",
    "patient_medical_condition",
}

# ── v7: completeness controls ────────────────────────────────────────
# PAGE_CHUNKED_PROMPTS: prompts whose items are self-contained rows (one lab
#   result per row). These always run on page windows when the document is
#   longer than one window. Comma-separated base names.
PAGE_CHUNKED_PROMPT_BASES = {p.strip() for p in os.getenv(
    "PAGE_CHUNKED_PROMPTS", "patient_laboratory_prompt").split(",") if p.strip()}
PAGE_CHUNK_SIZE      = max(1, int(os.getenv("PAGE_CHUNK_SIZE", "3")))      # pages per window
PAGE_CHUNK_OVERLAP   = max(0, int(os.getenv("PAGE_CHUNK_OVERLAP", "0")))   # pages shared by neighbouring windows
CHUNK_WORKERS        = max(1, int(os.getenv("CHUNK_WORKERS", "4")))        # parallel Bedrock calls per prompt
# ENABLE_CIOMS_LAB_PARSER: read CIOMS "13. Lab Data" rows deterministically.
#   Only used when every row 1..N is found; otherwise the LLM result is kept.
ENABLE_CIOMS_LAB_PARSER = os.getenv("ENABLE_CIOMS_LAB_PARSER", "true").lower() in ("1", "true", "yes")
LAB_SECTION_KEY      = "laboratory_test_carried_out"

# ====================================================================
# AWS Clients
# ====================================================================
s3       = boto3.client("s3")
bedrock  = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION,
                         config=_BEDROCK_TIMEOUT_CFG)
textract = boto3.client("textract",         region_name=BEDROCK_REGION,
                         config=_TEXTRACT_TIMEOUT_CFG)
sfn      = boto3.client("stepfunctions",    region_name=BEDROCK_REGION)
HAS_CONVERSE = hasattr(bedrock, "converse")

# ====================================================================
# In-memory cache — skips S3 reads on warm Lambda containers.
# Keyed by client_id: each tenant's configuration.json, extraction prompt
# sequence, and Archilles credentials are loaded and cached independently,
# so pharma_1 and dev (and any future client) never share state.
# ====================================================================
_CONFIG_CACHE = {}   # client_id -> {config, extraction_prompt_sequence,
                     #               archilles_api_url, archilles_access_token,
                     #               prompt_to_section_map}

ARCHILLES_COOKIE = os.getenv("ARCHILLES_COOKIE")   # not client-specific


def load_configuration(client_id: str) -> dict:
    """Read configuration.json for one client_id from S3 (uncached read)."""
    key    = configuration_key(client_id)
    obj    = s3.get_object(Bucket=BUCKET, Key=key)
    config = json.loads(obj["Body"].read().decode("utf-8", errors="ignore"))
    print(f"Loaded configuration for client_id='{client_id}' from {key}")
    return config


def get_client_config(client_id: str) -> dict:
    """
    Lazily load + cache everything derived from one client's configuration.json.
    Returns:
      {
        "config":                     <raw configuration.json dict>,
        "extraction_prompt_sequence": [S3 keys of prompts, prompt[0] = classification],
        "archilles_api_url":          str,
        "archilles_access_token":     str,
        "prompt_to_section_map":      {prompt_base_name: section_name},
      }
    """
    if client_id in _CONFIG_CACHE:
        return _CONFIG_CACHE[client_id]

    config                      = load_configuration(client_id)
    extraction_prompt_sequence = config.get("extraction_prompt_sequence", [])

    entry = {
        "config":                     config,
        "extraction_prompt_sequence": extraction_prompt_sequence,
        "archilles_api_url":          config.get("record_creation_endpoint", ""),
        "archilles_access_token":     config.get("record_creation_token", ""),
        "prompt_to_section_map":      _build_prompt_to_section_map(extraction_prompt_sequence),
    }
    _CONFIG_CACHE[client_id] = entry

    print(f"  Extraction prompt sequence for '{client_id}' ({len(extraction_prompt_sequence)} prompts):")
    for idx, p in enumerate(extraction_prompt_sequence):
        print(f"    {idx+1}. {p}")
    return entry

# ====================================================================
# Needle Registry
# ====================================================================
NEEDLE_REGISTRY = {
    "drug_prompt":               {"inject_key": "product_details_needle",  "source_key": "product_names"},
    "event_prompt":              {"inject_key": "adverse_event_needle",     "source_key": "adverse_event_names"},
    "patient_medical_condition": {"inject_key": "medical_condition_needle", "source_key": "medical_condition"},
    "patient_laboratory_prompt": {"inject_key": "laboratory_test_needle",   "source_key": "diagnostic_report"},
}

# ====================================================================
# Section name map
# ====================================================================
SECTION_NAME_MAP = {
    "data_classification_prompt": "general_info",
    "reporter_prompt":            "reporter_details",
    "patient_demographics_prompt":"patient_generic_info",   # display stand key
    "patient_laboratory_prompt":  "laboratory_test_carried_out",
    "patient_medical_condition":  "medical_conditions",
    "drug_prompt":                "drug_information",
    "event_prompt":               "event_details",
    "narrative":                  "narrative_summary",
}

# ====================================================================
# Dynamic skeleton extraction from prompts
#
# Skeletons are parsed directly from each prompt's output structure block
# at Lambda cold start. No hardcoded field lists in Lambda code — if the
# prompt schema changes, the skeleton updates automatically.
# ====================================================================

def _extract_skeleton_from_prompt_text(text: str) -> dict:
    """
    Parse the JSON output skeleton embedded in a prompt file.
    Handles single-quoted pseudo-JSON, inline comments, missing commas,
    and bare-word remnants left after comment stripping.
    Returns the parsed skeleton dict, or {} if parsing fails.
    """
    markers = ["## output structure", "# Output Format", "# output format", "# Output format"]
    start = -1
    for m in markers:
        idx = text.lower().find(m.lower())
        if idx != -1:
            start = idx
            break
    if start == -1:
        return {}

    snippet    = text[start:]
    brace_open = snippet.find("{")
    if brace_open == -1:
        return {}

    # Walk to find the matching closing brace
    depth, in_str, str_char, end_pos = 0, False, None, -1
    for i, ch in enumerate(snippet[brace_open:], brace_open):
        if in_str:
            if ch == str_char:
                in_str = False
        else:
            if ch in ('"', "'"):
                in_str   = True
                str_char = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_pos = i + 1
                    break

    if end_pos == -1:
        return {}

    raw = snippet[brace_open:end_pos]

    # Normalise to valid JSON
    raw = raw.replace("\r", "").replace("\t", "  ")
    raw = re.sub(r"//[^\n]*",  "",  raw)         # strip // comments
    raw = re.sub(r"##[^\n]*",  "",  raw)         # strip ## comments
    # Remove bare unquoted words left after comment stripping (e.g. yes/no/unknown)
    raw = re.sub(r"(?<=[,\{\[\s])([a-zA-Z][a-zA-Z0-9_/\-]*)\s*(?=[,\}\]\n])", "", raw)
    raw = re.sub(r"'([^']*)'", r'"\1"', raw)     # single → double quotes
    raw = re.sub(r",\s*([\}\]])", r"\1", raw)    # trailing commas
    # Insert missing commas between adjacent keys/values
    raw = re.sub(r'(")\s*\n(\s*")',  r'",\n\2',  raw)
    raw = re.sub(r'(\})\s*\n(\s*")', r'},\n\2',  raw)
    raw = re.sub(r'(\])\s*\n(\s*")', r'],\n\2',  raw)
    raw = re.sub(r",\s*([\}\]])", r"\1", raw)    # trailing commas again
    raw = re.sub(r"\n\s*\n", "\n", raw)          # collapse blank lines

    try:
        return json.loads(raw)
    except Exception:
        # Last resort: blank all values and retry
        raw = re.sub(r':\s*"[^"]*"', ': ""', raw)
        raw = re.sub(r",\s*([\}\]])", r"\1", raw)
        try:
            return json.loads(raw)
        except Exception as e:
            print(f"  Warning: skeleton parse failed: {e}")
            return {}


def _normalise_skeleton_values(obj):
    """
    Walk a parsed skeleton and set every leaf value to "" (empty string).
    Prompts may have leftover non-empty strings from inline examples —
    this ensures every field starts blank for manual entry.
    Preserves list structure (one item per array section).
    """
    if isinstance(obj, dict):
        return {k: _normalise_skeleton_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict):
            return [_normalise_skeleton_values(obj[0])]  # keep exactly one item
        return obj
    # Leaf value — always empty for skeleton
    return ""


# Build skeleton cache at cold start by reading every prompt from S3
# Populated lazily in get_skeleton_for_prompt() to avoid blocking init
_PROMPT_SKELETON_CACHE: dict = {}   # prompt_s3_key → parsed skeleton dict


def get_skeleton_for_prompt(prompt_s3_key: str) -> dict:
    """
    Return the empty skeleton for a prompt (cached after first read).
    The skeleton is extracted directly from the prompt's output structure block —
    no hardcoding anywhere in Lambda code.
    """
    import copy
    if prompt_s3_key in _PROMPT_SKELETON_CACHE:
        return copy.deepcopy(_PROMPT_SKELETON_CACHE[prompt_s3_key])

    try:
        prompt_text = read_s3_text(prompt_s3_key)
        raw_skeleton = _extract_skeleton_from_prompt_text(prompt_text)
        normalised   = _normalise_skeleton_values(raw_skeleton)
        _PROMPT_SKELETON_CACHE[prompt_s3_key] = normalised
        sections = list(normalised.keys())
        print(f"  Skeleton parsed from {os.path.basename(prompt_s3_key)}: {sections}")
        return copy.deepcopy(normalised)
    except Exception as e:
        print(f"  Warning: could not get skeleton for {prompt_s3_key}: {e}")
        return {}


# ====================================================================
# Helpers
# ====================================================================
def _geo_prefix_for_region(region: str) -> str:
    for k, v in {"us-": "us", "eu-": "eu", "ap-": "ap"}.items():
        if region.startswith(k):
            return v
    return "us"

#MODEL_ID = f"{_geo_prefix_for_region(BEDROCK_REGION)}.amazon.nova-premier-v1:0"
MODEL_ID = f"{_geo_prefix_for_region(BEDROCK_REGION)}.amazon.nova-2-lite-v1:0"

def _unquote_key(k: str) -> str:
    return urllib.parse.unquote_plus(k)

def _prompt_base_name(s3_key: str) -> str:
    fname = os.path.splitext(os.path.basename(s3_key))[0]
    return re.sub(r'_v\d+$', '', fname)

def get_clean_filename(s3_key: str) -> str:
    return f"{_prompt_base_name(s3_key)}.json"

def _canonical_path(path: str) -> str:
    if not path:
        return ""
    parts = [p.strip() for p in re.split(r'\s*->\s*', str(path))]
    return " -> ".join("0" if p.isdigit() or p == "0" else p for p in parts).strip()

def _build_prompt_to_section_map(extraction_prompt_sequence: list) -> dict:
    mapping = {}
    for s3_key in extraction_prompt_sequence:
        base    = _prompt_base_name(s3_key)
        section = SECTION_NAME_MAP.get(base, base)
        mapping[base] = section
    return mapping
# NOTE: this used to be built once at cold start from a single hardcoded
# EXTRACTION_PROMPT_SEQUENCE. It is now built per-client inside
# get_client_config() and cached there (client_config["prompt_to_section_map"]),
# since each client can have its own prompt sequence.

def read_s3_text(key: str) -> str:
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode("utf-8", errors="ignore")

def read_s3_json(key: str):
    return json.loads(read_s3_text(key))

def upload_bytes(key: str, data: bytes, content_type: str):
    s3.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType=content_type)

# ====================================================================
# Display stand data (cached)
# ====================================================================
# Per-doc-type display stand cache: "doc_type_lower" → {lookup_dict, form_id}
_DISPLAY_STAND_CACHE: dict = {}


def _build_lookup_from_data(data: list) -> tuple:
    """
    Build a d_keys lookup dict and extract form_id from display stand JSON data.
    Returns (lookup_dict, form_id).
    """
    form_id = None
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and "record_id" in first:
            form_id = first["record_id"]
    lookup = {}
    for item in (data if isinstance(data, list) else []):
        dkey = item.get("d_keys")
        if dkey:
            dkey = dkey.strip()
            norm = _canonical_path(dkey)
            for k in (dkey, norm, dkey.replace(" -> ", "->"), norm.replace(" -> ", "->")):
                lookup[k] = item
    return lookup, form_id


def get_display_lookup_for_doc_type(doc_type: str, client_id: str) -> tuple:
    """
    Return (lookup_dict, form_id) for the given doc_type + client_id.

    Reads the S3 path from that client's configuration.json["display_standard"]
    [doc_type_lower]. Results are cached per (client_id, doc_type) so each path
    is only read once per container, and different clients never share a cache
    entry even if they happen to use the same doc_type name.

    Falls back gracefully:
      - If doc_type not in display_standard → tries "isi" as default
      - If no match at all → returns ({}, None)
    """
    doc_key = (doc_type or "unknown").lower().strip()
    key     = f"{client_id}:{doc_key}"

    if key in _DISPLAY_STAND_CACHE:
        cached = _DISPLAY_STAND_CACHE[key]
        return cached["lookup"], cached["form_id"]

    config           = get_client_config(client_id)["config"]
    display_standard = config.get("display_standard", {})

    if not display_standard:
        print("  Warning: 'display_standard' not found in configuration.json. "
              "Add it with doc_type->S3 path mappings e.g. "
              "{'isi': 'clients/.../isi.json', 'medwatch': 'clients/.../med.json'}")

    # Case-insensitive match
    s3_path = None
    for k, v in display_standard.items():
        if k.lower() == doc_key:
            s3_path = v
            break

    # Fallback: try ISI as the universal default when doc_type has no dedicated entry.
    # ISI is chosen because it is the most common form type in pharma PV processing.
    if not s3_path and display_standard:
        # First try explicit "default" key in config
        for k, v in display_standard.items():
            if k.lower() == "default":
                s3_path = v
                print(f"  [{client_id}] display_standard: no entry for '{doc_key}' — falling back to 'default'")
                break
        # Then try "isi"
        if not s3_path:
            for k, v in display_standard.items():
                if k.lower() == "isi":
                    s3_path = v
                    print(f"  [{client_id}] display_standard: no entry for '{doc_key}' — falling back to 'isi'")
                    break
        # Last resort: use the first entry in the mapping
        if not s3_path:
            first_key, first_val = next(iter(display_standard.items()))
            s3_path = first_val
            print(f"  [{client_id}] display_standard: no entry for '{doc_key}' — using first entry '{first_key}' as fallback")

    if not s3_path:
        print(f"  [{client_id}] Warning: no display_standard path for doc_type '{doc_key}' — enrichment will use empty metadata")
        _DISPLAY_STAND_CACHE[key] = {"lookup": {}, "form_id": None}
        return {}, None

    try:
        data          = read_s3_json(s3_path)
        lookup, form_id = _build_lookup_from_data(data)
        _DISPLAY_STAND_CACHE[key] = {"lookup": lookup, "form_id": form_id}
        print(f"  [{client_id}] Loaded display stand for '{doc_key}': {len(lookup)} entries, form_id={form_id} from {s3_path}")
        return lookup, form_id
    except Exception as e:
        print(f"  [{client_id}] Warning: could not load display stand for '{doc_key}' from {s3_path}: {e}")
        _DISPLAY_STAND_CACHE[key] = {"lookup": {}, "form_id": None}
        return {}, None


# DISPLAY_LOOKUP is no longer a module-level variable.
# All enrichment uses get_display_lookup_for_doc_type(doc_type) which
# reads the correct display stand file from S3 based on the document type
# declared in configuration.json["display_standard"].
# FORM_ID is also per-doc-type — read from the display stand file itself.
FORM_ID = None   # kept for Archilles fallback only

# ====================================================================
# Token + Bedrock helpers
# ====================================================================
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

STRICT_JSON_LINE = ("You MUST return STRICT, VALID JSON only (RFC 8259): double quotes around every key "
                    "and string value, no comments, no trailing commas. No markdown.")


def build_instruction(prompt_text: str) -> str:
    return f"{STRICT_JSON_LINE}\n\n<<PROMPT>>\n{prompt_text}\n\n<<RAW_TEXT>>\n"

def detect_document_type(raw_text: str) -> str:
    t = raw_text.lower()
    if any(k in t for k in ["doi:", "doi.org", "pubmed", "abstract\n", "references\n"]):
        return "LITERATURE"
    if any(k in t for k in ["medwatch", "form fda 3500", "describe event, problem"]):
        return "MEDWATCH"
    if any(k in t for k in ["cioms", "reaction onset", "date of birth\nday\nmonth\nyear"]):
        return "CIOMS"
    if any(k in t for k in ["isi form", "solicited", "patient support program"]):
        return "ISI"
    return "UNSTRUCTURED"

def build_document_preamble(doc_type: str) -> str:
    return {
        "LITERATURE": (
            "[DOCUMENT TYPE: PUBLISHED LITERATURE / CASE REPORT]\n"
            "This document is a peer-reviewed journal article or clinical case report. "
            "Extract patient-specific data from case sections and tables. "
            "For laboratory tables: treat every (row x column) cell as a distinct observation.\n\n"
        ),
        "MEDWATCH": "[DOCUMENT TYPE: FDA MEDWATCH FORM]\n\n",
        "CIOMS":    "[DOCUMENT TYPE: CIOMS FORM]\n\n",
        "ISI":      "[DOCUMENT TYPE: ISI / SOLICITED FORM]\n\n",
        "UNKNOWN":  "",
    }.get(doc_type, "")

def _build_user_prompt(prompt_text: str, raw_chunk: str, doc_type: str = "") -> str:
    preamble = build_document_preamble(doc_type) if doc_type else ""
    return (
        f"{STRICT_JSON_LINE}\n\n"
        f"<<PROMPT>>\n{prompt_text}\n\n"
        f"<<RAW_TEXT>>\n{preamble}{raw_chunk}"
    )

def _get_output_cap(fname: str) -> int:
    """Return the correct output token cap for a given prompt filename."""
    fname_lower = fname.lower() if fname else ""
    return HIGH_OUTPUT_TOKENS if _prompt_base_name(fname_lower) in HIGH_OUTPUT_PROMPT_BASES else USER_MAX_TOKENS_CAP


def bedrock_call(prompt_text: str, max_out: int, fname: str = "") -> str:
    """
    Standard Bedrock call — single combined prompt+raw_text message.
    Returns model text. Warns (does not raise) when output was truncated.
    """
    text, stop_reason = _bedrock_invoke(prompt_text, max_out, fname)
    if stop_reason == "max_tokens":
        print(f"  WARNING: {fname} hit max_tokens — output truncated, JSON may be incomplete")
    return text


def _bedrock_invoke(prompt_text: str, max_out: int, fname: str = "") -> tuple:
    """
    Low-level Bedrock call returning (text, stop_reason).
    stop_reason: "end_turn" (complete), "max_tokens" (truncated), or "" (unknown).
    Lets callers detect truncated JSON instead of silently accepting a partial parse.
    """
    cap        = _get_output_cap(fname)
    out_tokens = min(max_out, cap, OUTPUT_HARD_CAP)
    if HAS_CONVERSE:
        resp = bedrock.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt_text}]}],
            inferenceConfig={"maxTokens": out_tokens, "temperature": TEMPERATURE, "topP": TOP_P}
        )
        text        = resp["output"]["message"]["content"][0]["text"]
        stop_reason = resp.get("stopReason", "")
        return text, stop_reason
    body = json.dumps({"prompt": prompt_text, "max_gen_len": out_tokens,
                       "temperature": TEMPERATURE, "top_p": TOP_P})
    resp = bedrock.invoke_model(modelId=MODEL_ID, body=body,
                                accept="application/json", contentType="application/json")
    parsed      = json.loads(resp["body"].read())
    text        = parsed.get("generation", "")
    stop_reason = parsed.get("stop_reason", parsed.get("stopReason", ""))
    return text, stop_reason


def bedrock_call_cached(prompt_text: str, raw_text: str, doc_type: str,
                        max_out: int, fname: str = "") -> str:
    """Text-only wrapper kept for backward compatibility (see _bedrock_invoke_cached)."""
    text, stop_reason = _bedrock_invoke_cached(prompt_text, raw_text, doc_type, max_out, fname)
    if stop_reason == "max_tokens":
        print(f"  WARNING: {fname} hit max_tokens (cached path) — output truncated")
    return text


def _bedrock_invoke_cached(prompt_text: str, raw_text: str, doc_type: str,
                           max_out: int, fname: str = "") -> tuple:
    """
    v7: returns (text, stop_reason) so truncation is detectable on the cached
    path too (previously the cached path never checked stop_reason).

    Bedrock call with prompt caching on the raw_text prefix.

    HOW IT WORKS:
    The converse API accepts a cachePoint block inside the message content.
    Everything before the cachePoint is cached by Bedrock on the first call
    and reused on subsequent calls for the same document — the KV attention
    state of the raw text is computed once and shared across all 7 prompts.

    Structure sent to Bedrock:
      user message:
        [text]      ← document preamble + raw text         ← CACHED (90% cost reduction)
        [cachePoint type=default]                           ← cache boundary marker
        [text]      ← "You MUST return JSON..." + prompt   ← NOT cached (prompt-specific)

    Cost saving: raw_text tokens (the bulk of input cost) are only billed at
    ~10% for calls 2-8 on the same document. Prompt 1 always pays full price
    since it runs before the cache is warm.

    Minimum cacheable tokens for Nova Premier: 2048 tokens.
    If raw_text is shorter than that, falls back to standard bedrock_call.
    """
    if not HAS_CONVERSE:
        # converse API required for caching — fall back to standard call
        return _bedrock_invoke(
            _build_user_prompt(prompt_text, raw_text, doc_type), max_out, fname
        )

    cap        = _get_output_cap(fname)
    out_tokens = min(max_out, cap, OUTPUT_HARD_CAP)

    # Bedrock requires at least 2048 tokens before a cachePoint.
    # For short documents, skip caching and use the standard path.
    preamble    = build_document_preamble(doc_type)
    cached_text = f"{preamble}{raw_text}"
    if estimate_tokens(cached_text) < 2048:
        return _bedrock_invoke(
            _build_user_prompt(prompt_text, raw_text, doc_type), max_out, fname
        )

    # Prompt instruction goes after the cachePoint (not cached — prompt-specific)
    prompt_instruction = (
        f"{STRICT_JSON_LINE}\n\n"
        f"<<PROMPT>>\n{prompt_text}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                # ── Cached prefix: document preamble + raw text ──────────
                {"text": f"<<RAW_TEXT>>\n{cached_text}"},
                # ── Cache boundary — everything above is cached ──────────
                {"cachePoint": {"type": "default"}},
                # ── Prompt instruction — NOT cached, varies per prompt ───
                {"text": prompt_instruction},
            ]
        }
    ]

    resp = bedrock.converse(
        modelId=MODEL_ID,
        messages=messages,
        inferenceConfig={"maxTokens": out_tokens, "temperature": TEMPERATURE, "topP": TOP_P}
    )
    return resp["output"]["message"]["content"][0]["text"], resp.get("stopReason", "")


def bedrock_call_with_backoff(prompt_text: str, max_out: int, fname: str = "",
                               raw_text: str = "", doc_type: str = "",
                               use_cache: bool = False) -> str:
    """
    Exponential backoff + jitter on ThrottlingException. Up to 5 attempts.

    When use_cache=True and raw_text is provided, uses bedrock_call_cached
    (splits raw_text from prompt for prompt caching).
    Otherwise uses the standard bedrock_call path.
    """
    max_attempts = BEDROCK_MAX_ATTEMPTS
    for attempt in range(max_attempts):
        try:
            if use_cache and raw_text:
                return bedrock_call_cached(prompt_text, raw_text, doc_type, max_out, fname)
            return bedrock_call(prompt_text, max_out, fname)
        except Exception as exc:
            err = str(exc)
            throttled = any(x in err for x in
                            ["ThrottlingException", "Too Many Requests",
                             "Rate exceeded", "slowdown", "ServiceUnavailable",
                             "ModelTimeoutException", "InternalServerException"])
            if throttled and attempt == max_attempts - 1:
                # Exhausted retries on a transient error — raise typed exception
                raise _TransientBedrockError(f"{fname} throttled after {max_attempts} attempts: {err}")
            if not throttled:
                raise
            # Cap exponential growth at 30s; add jitter to de-synchronise the burst
            wait = min(2 ** attempt, 30) + random.uniform(0, 2)
            print(f"Bedrock throttled ({fname}) — retry in {wait:.1f}s (attempt {attempt+1}/{max_attempts})")
            time.sleep(wait)


def _bedrock_invoke_with_backoff(prompt_text: str, max_out: int, fname: str = "",
                                 raw_text: str = "", doc_type: str = "",
                                 use_cache: bool = False) -> tuple:
    """
    Like bedrock_call_with_backoff but returns (text, stop_reason) so callers
    can detect truncation. Backoff on throttling.
    v7: also supports the cached path (use_cache=True + raw_text), in which case
    prompt_text is the bare prompt (raw_text is sent separately before the cachePoint).
    """
    max_attempts = BEDROCK_MAX_ATTEMPTS
    for attempt in range(max_attempts):
        try:
            if use_cache and raw_text:
                return _bedrock_invoke_cached(prompt_text, raw_text, doc_type, max_out, fname)
            return _bedrock_invoke(prompt_text, max_out, fname)
        except Exception as exc:
            err = str(exc)
            throttled = any(x in err for x in
                            ["ThrottlingException", "Too Many Requests",
                             "Rate exceeded", "slowdown", "ServiceUnavailable",
                             "ModelTimeoutException", "InternalServerException"])
            if throttled and attempt == max_attempts - 1:
                raise _TransientBedrockError(f"{fname} throttled after {max_attempts} attempts: {err}")
            if not throttled:
                raise
            wait = min(2 ** attempt, 30) + random.uniform(0, 2)
            print(f"Bedrock throttled ({fname}) — retry in {wait:.1f}s (attempt {attempt+1}/{max_attempts})")
            time.sleep(wait)

def extract_json(text: str):
    """
    Parse JSON from model output. Strips markdown fences, then tries strict parse.
    If that fails (e.g. truncated output), attempts to REPAIR the JSON by closing
    any unterminated strings/objects/arrays so partial data is still recovered
    rather than discarded entirely.
    """
    if not text:
        return {}

    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)

    # Isolate the JSON object
    i = cleaned.find("{")
    if i == -1:
        return {}
    candidate = cleaned[i:]

    # Strict parse first
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Trailing-} parse (handles trailing prose after the JSON)
    j = candidate.rfind("}")
    if j != -1:
        try:
            return json.loads(candidate[:j + 1])
        except Exception:
            pass

    # v7.1: the model sometimes copies the prompt's pseudo-JSON style —
    # single-quoted keys/strings, // or ## comments, trailing commas,
    # True/False/None. Normalise to strict JSON and parse again.
    normalised = _normalize_pseudo_json(candidate)
    if normalised != candidate:
        try:
            parsed = json.loads(normalised)
            print("  extract_json: parsed after normalising non-strict JSON (quotes/comments/trailing commas)")
            return parsed
        except Exception:
            candidate = normalised          # repair below works on the normalised text

    # Repair truncated JSON: walk the string tracking depth, close what is open
    repaired = _repair_truncated_json(candidate)
    if repaired:
        try:
            return json.loads(repaired)
        except Exception as e:
            print(f"  extract_json: repair attempt failed: {e} — walking back to an earlier safe point")

    # v7: the single-shot repair fails when the cut lands inside a key or
    # half-written object. Walk back through earlier structural boundaries
    # (after '}' / ']' or before ',') until a prefix parses, so the rows that
    # were complete are kept instead of losing the whole answer.
    recovered = _repair_by_walking_back(candidate)
    if recovered:
        return recovered
    return {}


_JSON_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "/": "/", "\\": "\\"}


def _normalize_pseudo_json(s: str) -> str:
    """
    v7.1: convert model output written in the prompts' pseudo-JSON style into
    strict JSON:
      • 'single' or "double" quoted strings → JSON double-quoted strings
      • // and # comments outside strings   → removed
      • trailing commas before } or ]       → removed
      • True/False/None (Python style)      → true/false/null
      • other stray bare words (comment leftovers such as yes/no) → removed
    An unterminated string at the end is left open so the truncation repair can
    still recover the complete part.
    """
    out, i, n = [], 0, len(s)

    def drop_trailing_comma():
        k = len(out) - 1
        while k >= 0 and out[k].isspace():
            k -= 1
        if k >= 0 and out[k] == ",":
            del out[k]

    while i < n:
        ch = s[i]
        if ch in ('"', "'"):
            q, j, buf = ch, i + 1, []
            while j < n:
                c = s[j]
                if c == "\\" and j + 1 < n:
                    nxt = s[j + 1]
                    if nxt in ('"', "'"):
                        buf.append(nxt); j += 2; continue
                    if nxt == "u" and j + 5 < n and re.fullmatch(r"[0-9a-fA-F]{4}", s[j + 2:j + 6]):
                        buf.append(chr(int(s[j + 2:j + 6], 16))); j += 6; continue
                    buf.append(_JSON_ESCAPES.get(nxt, nxt)); j += 2; continue
                if c == q:
                    break
                buf.append(c); j += 1
            text = json.dumps("".join(buf), ensure_ascii=False)
            if j >= n:                       # unterminated (truncated output)
                out.append(text[:-1])
                break
            out.append(text)
            i = j + 1
            continue
        if ch == "/" and i + 1 < n and s[i + 1] == "/":
            while i < n and s[i] != "\n":
                i += 1
            continue
        if ch == "#":
            while i < n and s[i] != "\n":
                i += 1
            continue
        if ch in "}]":
            drop_trailing_comma()
            out.append(ch); i += 1
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            word = s[i:j]
            prev = next((c for c in reversed("".join(out[-3:])) if not c.isspace()), "")
            if prev.isdigit() or prev == ".":           # exponent, e.g. 1e5
                out.append(word)
            else:
                mapped = {"true": "true", "True": "true", "false": "false", "False": "false",
                          "null": "null", "None": "null"}.get(word)
                if mapped:
                    out.append(mapped)
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _repair_by_walking_back(s: str, max_tries: int = 400):
    """v7: try progressively shorter prefixes ending at structural boundaries."""
    in_str, escape, cuts = False, False, []
    for idx, ch in enumerate(s):
        if escape:
            escape = False; continue
        if ch == "\\":
            escape = True; continue
        if in_str:
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "}]":
            cuts.append(idx + 1)
        elif ch == ",":
            cuts.append(idx)
    for cut in reversed(cuts[-max_tries:]):
        fixed = _repair_truncated_json(s[:cut])
        if not fixed:
            continue
        try:
            parsed = json.loads(fixed)
            if parsed:
                print(f"  extract_json: recovered partial JSON at char {cut} of {len(s)}")
                return parsed
        except Exception:
            continue
    return None


def _repair_truncated_json(s: str) -> str:
    """
    Best-effort repair of truncated JSON. Tracks string/escape state and
    bracket depth, drops any dangling partial key/value at the end, and
    appends the closing brackets needed to make it parseable.
    Returns repaired string, or "" if it cannot be repaired.
    """
    in_str   = False
    escape   = False
    stack    = []          # holds '{' and '[' to close later
    last_safe = -1         # index just after the last completed value/pair

    for idx, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_str:
            if ch == '"':
                in_str = False
                # closing quote of a string value is a reasonable safe point
                last_safe = idx + 1
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            last_safe = idx + 1
        elif ch in ",":
            last_safe = idx       # before the comma — safe to truncate here

    if not stack and not in_str:
        return s   # already balanced

    # Truncate to last safe point, drop a trailing comma, then close open brackets
    truncated = s[:last_safe].rstrip().rstrip(",")

    # Recount open brackets in the truncated string
    in_str = False; escape = False; depth_stack = []
    for ch in truncated:
        if escape:
            escape = False; continue
        if ch == "\\":
            escape = True; continue
        if in_str:
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth_stack.append(ch)
        elif ch in "}]":
            if depth_stack:
                depth_stack.pop()

    closing = "".join("}" if b == "{" else "]" for b in reversed(depth_stack))
    return truncated + closing

def deep_merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        merged = dict(a)
        for k, v in b.items():
            merged[k] = deep_merge(merged.get(k), v) if k in merged else v
        return merged
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    return b

def classify_smart(prompt_text: str, raw_text: str, fname: str = "",
                   doc_type: str = "", use_cache: bool = False):
    """
    Route to classify_single, falling back to classify_chunked ONLY for
    size/truncation reasons — never for transient throttle errors.

    DETERMINISM: which path runs determines the output, so path selection
    must be stable across runs. Transient (throttle/timeout) exceptions are
    re-raised so Step Functions retries the SAME path, rather than silently
    switching to chunked (which would extract a different field count and
    cause the run-to-run inconsistency seen in single uploads).
    """
    try:
        result = classify_single(prompt_text, raw_text, fname, doc_type, use_cache=use_cache)
        print(f"  {fname}: extracted via SINGLE path")
        return result
    except _TransientBedrockError:
        # Throttle/timeout — do NOT fall back to chunked (that changes the
        # output). Propagate so Step Functions retries the single path.
        raise
    except Exception as e:
        # Genuine size/truncation/parse reason — chunked is the correct,
        # deterministic recovery path for oversized documents.
        print(f"  {fname}: SINGLE path unavailable ({e}) — using CHUNKED path")
        return classify_chunked(prompt_text, raw_text, fname, doc_type)

def classify_single(prompt_text: str, raw_text: str, fname: str = "",
                    doc_type: str = "", use_cache: bool = False):
    """
    Single-shot Bedrock call with truncation detection.

    If the model output is truncated (stop_reason == "max_tokens") OR the JSON
    fails to parse cleanly, the result is unreliable — we raise so classify_smart
    falls back to the chunked path which uses smaller pieces and merges results.
    """
    cap = _get_output_cap(fname)
    est = estimate_tokens(build_instruction(prompt_text)) + estimate_tokens(raw_text) + cap
    if est > CONTEXT_WINDOW:
        raise ValueError("Input too large — falling back to chunked")

    if use_cache:
        # v7: cached path now returns stop_reason as well, so truncation is
        # detected here exactly like the non-cached path below.
        out, stop_reason = _bedrock_invoke_with_backoff(
            prompt_text, cap, fname,
            raw_text=raw_text, doc_type=doc_type, use_cache=True
        )
    else:
        # Non-cached path: get stop_reason so we can detect truncation
        full_prompt = _build_user_prompt(prompt_text, raw_text, doc_type)
        out, stop_reason = _bedrock_invoke_with_backoff(full_prompt, cap, fname)

    # Strict parse
    try:
        parsed = json.loads(out)
    except Exception:
        parsed = extract_json(out)

    # If truncated, the parse is partial/repaired — not trustworthy for a
    # high-field-count prompt. Raise so classify_smart uses chunked merge.
    if stop_reason == "max_tokens":
        # v7: typed exception carrying the partial result, so callers can
        # split the work (page windows / needle batches) instead of
        # re-running the identical call.
        raise _TruncatedOutput(f"{fname} truncated at max_tokens", partial=parsed)

    if not parsed:
        print(f"  {fname}: model output could not be parsed — first 400 chars: {out[:400]!r}")
        raise ValueError(f"{fname} produced empty/unparseable JSON — falling back to chunked")

    return parsed

def classify_chunked(prompt_text: str, raw_text: str, fname: str = "", doc_type: str = ""):
    """
    Chunked fallback for documents too large for a single context window.
    Chunks are processed sequentially and results are merged.
    Note: chunks are individually too short to meet the 2048-token caching
    minimum, so standard (non-cached) calls are used here.
    """
    cap       = _get_output_cap(fname)
    instr_tok = estimate_tokens(build_instruction(prompt_text))
    room      = max(5_000, CONTEXT_WINDOW - cap - SAFETY_RESERVE - instr_tok)
    per_chunk = max(40_000, room * 4)
    chunks, start, n = [], 0, len(raw_text)
    while start < n:
        end = min(n, start + per_chunk)
        # v7 fix: rfind() returns -1 (truthy) when not found, so the old
        # `a or b or c` chain never fell through. Try each separator in turn.
        cut = -1
        if end < n:
            for sep in ("\n\n", "\n", " "):
                pos = raw_text.rfind(sep, start, end)
                if pos > start:
                    cut = pos
                    break
        if cut == -1 or cut <= start:
            cut = end
        chunk = raw_text[start:cut]
        if chunk.strip():
            chunks.append(chunk)
        start = cut
    merged = {}
    for ch in chunks:
        out    = bedrock_call_with_backoff(_build_user_prompt(prompt_text, ch, doc_type), cap, fname)
        # v7.1: never json.loads() model output unguarded — a single non-strict
        # character used to crash the whole task. extract_json tries strict
        # parsing first, then normalisation and truncation repair.
        result = extract_json(out)
        if not result:
            print(f"  {fname}: chunk output could not be parsed — first 400 chars: {out[:400]!r}")
            continue
        merged = deep_merge(merged, result)
    return merged

# ====================================================================
# PDF Rendering — PARALLEL Textract across all pages
# ====================================================================
def _render_single_page(page_data: tuple) -> dict:
    """
    Render one PDF page, upload PNG, call Textract synchronously.
    Runs in a thread — one thread per page, all pages in parallel.
    Returns a dict with keys: page_idx, raw_text, layout_entry, file_info_entry.
    """
    page, idx, out_prefix, scale = page_data
    pix      = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img_key  = f"{out_prefix}images/page-{idx:04d}.png"
    png_bytes = pix.tobytes("png")
    width, height = pix.width, pix.height

    # Upload PNG to S3
    upload_bytes(img_key, png_bytes, "image/png")

    # Textract — DetectDocumentText (Option A).
    # Replaces analyze_document(FeatureTypes=["LAYOUT","FORMS"]) which cost
    # $0.10/page ($0.05 LAYOUT + $0.05 FORMS billed separately).
    # DetectDocumentText costs $0.0015/page — 98% cheaper.
    # Output is identical for our use case: WORD blocks with Text, Geometry,
    # and Confidence are all we consume downstream. LAYOUT and FORMS blocks
    # were never read by any code in this pipeline.
    # Retries with backoff on ThrottlingException (up to 4 attempts).
    textract_resp = {"Blocks": []}
    for t_attempt in range(4):
        try:
            textract_resp = textract.detect_document_text(
                Document={"Bytes": png_bytes}
            )
            break
        except Exception as e:
            err = str(e)
            is_throttle = any(x in err for x in
                              ["ThrottlingException", "ProvisionedThroughputExceededException",
                               "Too Many Requests", "slowdown"])
            if not is_throttle or t_attempt == 3:
                print(f"  Textract page {idx} failed after {t_attempt+1} attempts: {e}")
                break
            wait = (2 ** t_attempt) + random.uniform(0, 1)
            print(f"  Textract page {idx} throttled — retry in {wait:.1f}s (attempt {t_attempt+1}/4)")
            time.sleep(wait)

    words = []
    for block in textract_resp.get("Blocks", []):
        if block.get("BlockType") != "WORD":
            continue
        polygon     = block["Geometry"]["Polygon"]
        pixel_points = []
        for p in polygon[:4]:
            pixel_points.extend([int(p["X"] * width), int(p["Y"] * height)])
        while len(pixel_points) < 8:
            pixel_points.extend(pixel_points[-2:])
        words.append({
            "text":       block["Text"],
            "polygon_px": pixel_points[:8],
            "confidence": block.get("Confidence", 95.0),
        })

    return {
        "page_idx":        idx,
        "raw_text":        page.get_text("text", sort=True),
        "layout_entry": {
            "page_index":   idx,
            "image_width":  width,
            "image_height": height,
            "words":        words,
        },
        "file_info_entry": {f"page-{idx}": {"value": img_key}},
    }


def render_pdf_pages(doc, out_prefix: str, scale: float):
    """
    Render ALL pages in parallel using ThreadPoolExecutor.
    Previous version: sequential — 10 pages = 10 × Textract wait = ~50s
    Now:              parallel   — 10 pages = max(Textract wait) = ~5s
    """
    page_data = [(page, idx + 1, out_prefix, scale) for idx, page in enumerate(doc)]
    results   = [None] * len(page_data)

    max_workers = min(len(page_data), 10)   # cap at 10 concurrent Textract calls
    print(f"  Rendering {len(page_data)} pages with {max_workers} parallel workers")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {pool.submit(_render_single_page, pd): pd[1] - 1
                         for pd in page_data}
        for future in as_completed(future_to_idx):
            slot = future_to_idx[future]
            try:
                results[slot] = future.result()
            except Exception as e:
                print(f"  Page {slot+1} render failed: {e}")
                results[slot] = None

    raw_pages  = []
    layout     = []
    file_info  = {}
    for r in results:
        if r is None:
            continue
        raw_pages.append(r["raw_text"])
        layout.append(r["layout_entry"])
        file_info.update(r["file_info_entry"])

    # v7: also return the per-page texts so list prompts can run on page windows.
    # raw.txt (the joined text) is unchanged, so every other prompt sees
    # exactly the same input as before.
    return "\n".join(raw_pages), layout, file_info, len(doc), raw_pages

# ====================================================================
# Region finder + Composite Score Engine (unchanged from stable version)
# ====================================================================
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[^a-zA-Z0-9\s\-@./:,]', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()

def find_region_for_value(value: str, layout: list) -> dict:
    empty = {"page_info": {}, "data_list": [], "match_quality": 0.0}
    if not value or not layout:
        return empty
    original_value = str(value).strip()
    if not original_value:
        return empty
    search_norm   = normalize_text(original_value)
    search_tokens = [t for t in re.split(r'[\s,]+', search_norm) if t]
    if not search_tokens:
        return empty

    if len(search_tokens) == 1:
        for page in layout:
            for w in page.get("words", []):
                if normalize_text(w["text"]) == search_norm:
                    p   = w["polygon_px"]
                    pad = 2
                    merged = [p[0]-pad, p[1]-pad, p[2]+pad, p[3]-pad,
                              p[4]+pad, p[5]+pad, p[6]-pad, p[7]+pad]
                    return {
                        "page_info": {"pageNumber": page["page_index"], "angle": 0,
                                      "width": page["image_width"], "height": page["image_height"],
                                      "unit": "pixel"},
                        "data_list": [{"content": original_value, "polygon": merged,
                                       "confidence": round(w.get("confidence", 95.0), 2),
                                       "offset": 0, "length": len(original_value)}],
                        "match_quality": 100.0,
                    }

    best_span, best_score, best_page, best_mq = None, -1, None, 0.0
    for page in layout:
        words = page.get("words", [])
        if not words:
            continue
        word_list = [(normalize_text(w["text"]), w) for w in words if normalize_text(w["text"])]
        for i in range(len(word_list)):
            for length in range(len(search_tokens),
                                min(len(search_tokens) + 8, len(word_list) - i + 1)):
                phrase_tokens = [word_list[i + j][0] for j in range(length)]
                phrase_norm   = " ".join(phrase_tokens)
                is_exact      = (phrase_norm == search_norm)
                is_all_tok    = (all(t in phrase_norm for t in search_tokens)
                                 and phrase_norm.count(" ") + 1 == len(search_tokens))
                if is_exact or is_all_tok:
                    span       = [word_list[i + j][1] for j in range(length)]
                    span_score = len(span) * 100 + sum(w.get("confidence", 90) for w in span)
                    if span_score > best_score:
                        best_score, best_span, best_page = span_score, span, page
                        best_mq = 90.0 if is_exact else 70.0
                    break

    if best_span and best_page:
        all_x = [c for w in best_span for c in [w["polygon_px"][0], w["polygon_px"][2],
                                                   w["polygon_px"][4], w["polygon_px"][6]]]
        all_y = [c for w in best_span for c in [w["polygon_px"][1], w["polygon_px"][3],
                                                   w["polygon_px"][5], w["polygon_px"][7]]]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        pad     = 2
        polygon = [min_x-pad, min_y-pad, max_x+pad, min_y-pad,
                   max_x+pad, max_y+pad, min_x-pad, max_y+pad]
        avg_conf = round(sum(w.get("confidence", 95.0) for w in best_span) / len(best_span), 2)
        return {
            "page_info": {"pageNumber": best_page["page_index"], "angle": 0,
                          "width": best_page["image_width"], "height": best_page["image_height"],
                          "unit": "pixel"},
            "data_list": [{"content": original_value, "polygon": polygon,
                           "confidence": avg_conf, "offset": 0, "length": len(original_value)}],
            "match_quality": best_mq,
        }
    return empty

# Composite score weights
_W1, _W2       = 0.35, 0.65
_W1_3, _W2_3, _W3_3 = 0.25, 0.40, 0.35

def compute_composite_score(ocr_confidence: float, match_quality: float,
                             llm_confidence: float = -1) -> str:
    if match_quality == 0.0:
        return str(round(min(llm_confidence, 100.0), 2)) if llm_confidence >= 0 else "50.0"
    if llm_confidence >= 0:
        composite = ocr_confidence*_W1_3 + match_quality*_W2_3 + llm_confidence*_W3_3
    else:
        composite = ocr_confidence*_W1 + match_quality*_W2
    return str(round(min(composite, 100.0), 2))

def attach_regions(body: dict, layout: list):
    def recurse(item):
        if isinstance(item, dict):
            if "value" in item and str(item.get("value", "")).strip():
                new_item        = dict(item)
                existing_region = item.get("region", [])
                existing_conf   = None
                if existing_region and isinstance(existing_region, list):
                    dl = existing_region[0].get("data_list", [])
                    if dl and dl[0].get("confidence") is not None:
                        existing_conf = float(dl[0]["confidence"])
                if existing_conf is not None:
                    ocr_conf, match_quality, region_list = existing_conf, 90.0, existing_region
                else:
                    region_dict   = find_region_for_value(str(item["value"]), layout)
                    match_quality = region_dict.pop("match_quality", 0.0)
                    region_list   = ([region_dict]
                                     if region_dict.get("data_list") and region_dict["data_list"]
                                     else [])
                    ocr_conf = 0.0
                    if region_list:
                        dl = region_list[0].get("data_list", [])
                        if dl and dl[0].get("confidence") is not None:
                            ocr_conf = float(dl[0]["confidence"])
                new_item["region"] = region_list
                llm_conf = float(item["llm_confidence"]) if "llm_confidence" in item else -1
                new_item["score"]  = compute_composite_score(ocr_conf, match_quality, llm_conf)
                return new_item
            return {k: recurse(v) for k, v in item.items()}
        if isinstance(item, list):
            return [recurse(x) for x in item]
        return item
    return recurse(body)

# ====================================================================
# Enrich helpers
# ====================================================================
def enrich_field(value, lookup_path: str, lookup: dict = None):
    # lookup must always be passed explicitly from merge_task via recursive_enrich.
    # If not provided (should not happen in production), use empty dict — fields
    # will still be enriched structurally but without display stand metadata.
    if lookup is None:
        lookup = {}
    meta = lookup.get(lookup_path) or lookup.get(_canonical_path(lookup_path)) or {}
    raw_display      = meta.get("display_ivp_val", "")
    is_display_value = "yes" if str(raw_display).strip() in ("1", "yes", "true", "True") else "no"
    if meta.get("custom_php_ivp", ""):
        decoded_bytes = base64.b64decode(meta.get("custom_php_ivp", ""))
        decoded_string = decoded_bytes.decode('utf-8')
    else:
        decoded_string = ""
    return {
        "value":         value,
        "source_label_name": meta.get("label", ""),
        "ext_rationale": "",
        "lang":          "",
        "score":         "",
        "is_redect":     "no",
        "is_display":    is_display_value,
        "threshold":     meta.get("threshold_ivp_val", "85"),
        "req_status":    meta.get("Mand_data", 0),
        "display_order": meta.get("order_ivp_val", ""),
        "is_trans_reqd": meta.get("translation_required", "no"),
        "is_just_reqd":  meta.get("justification_required", "no"),
        "is_computed":   "yes",
        "trans_value":   "",
        "type":          meta.get("type", ""),   # data type from display_standard
        "region":        [],
        "display_name":  meta.get("label", ""),
        "option_data":  decoded_string,
    }

def recursive_enrich(data, real_path="", lookup_path="", lookup: dict = None):
    if isinstance(data, dict):
        enriched = {}
        for key, value in data.items():
            new_real   = f"{real_path} -> {key}" if real_path else key
            new_lookup = f"{lookup_path} -> {key}" if lookup_path else key
            if key == "narrative":
                raw_value = value.get("value", value) if isinstance(value, dict) else value
                if isinstance(raw_value, (str, int, float)):
                    meta = enrich_field(str(raw_value), _canonical_path(new_lookup), lookup)
                    meta["value"]        = str(raw_value)
                    meta["region"]       = []
                    meta["display_name"] = "Narrative "
                    enriched[key]        = meta
                else:
                    enriched[key] = value
                continue
            # Signal 3: prompt returned {"value": "...", "confidence": 95}
            if (isinstance(value, dict) and "value" in value and "confidence" in value
                    and "score" not in value and "region" not in value
                    and "source_label_name" not in value):
                ef = enrich_field(value["value"], _canonical_path(new_lookup), lookup)
                try:
                    ef["llm_confidence"] = float(value["confidence"])
                except (TypeError, ValueError):
                    pass
                enriched[key] = ef
                continue
            if isinstance(value, (dict, list)):
                enriched[key] = recursive_enrich(value, new_real, new_lookup, lookup)
            else:
                enriched[key] = enrich_field(value, _canonical_path(new_lookup), lookup)
        return enriched
    if isinstance(data, list):
        result = []
        for idx, item in enumerate(data):
            new_real   = f"{real_path} -> {idx}" if real_path else str(idx)
            new_lookup = f"{lookup_path} -> 0"   if lookup_path else "0"
            if isinstance(item, (dict, list)):
                result.append(recursive_enrich(item, new_real, new_lookup, lookup))
            else:
                result.append(enrich_field(item, _canonical_path(new_lookup), lookup))
        return result
    return data

# ====================================================================
# Needle injection
# ====================================================================
def get_needle_values(prompt_s3_key: str, classification: dict) -> list:
    """v7: the needle list (from the classification prompt) for a prompt, as a clean list."""
    meta = NEEDLE_REGISTRY.get(_prompt_base_name(prompt_s3_key))
    if not meta or not meta.get("inject_key"):
        return []
    values = (classification or {}).get("general_info", {}).get(meta["source_key"], [])
    if not isinstance(values, list):
        values = [values] if values else []
    out = []
    for v in values:
        if isinstance(v, dict):
            v = v.get("value", "")
        if v not in (None, "") and v not in out:
            out.append(v)
    return out


def inject_needle_values(prompt_text: str, prompt_s3_key: str, classification: dict,
                         values_override: list = None) -> str:
    """
    Replace the needle placeholder (e.g. laboratory_test_needle) with the
    classification list. v7: values_override lets needle batching inject a
    subset of the list.
    """
    base = _prompt_base_name(prompt_s3_key)
    meta = NEEDLE_REGISTRY.get(base)
    if not meta or not meta.get("inject_key"):
        return prompt_text
    if values_override is not None:
        values = values_override
    else:
        values = classification.get("general_info", {}).get(meta["source_key"], [])
    replaced = prompt_text.replace(meta["inject_key"], json.dumps(values, ensure_ascii=False))
    print(f"  Injected needle '{meta['inject_key']}' with "
          f"{len(values) if isinstance(values, list) else 1} value(s)")
    return replaced


# ====================================================================
# v7: COMPLETE extraction for list prompts
#
# Problem fixed: one Bedrock call was asked to emit every row of a long
# list (e.g. 229 lab rows ≈ 35-45K output tokens). The model stopped early
# (25 rows) or hit max_tokens, and the old chunked fallback re-sent the SAME
# whole document, so the loss was silent.
#
# Strategy:
#   1. Row-type prompts (PAGE_CHUNKED_PROMPTS, labs by default) run on page
#      windows in parallel → each call returns a small, complete slice.
#   2. Needle prompts that still truncate are re-run in needle batches over
#      the FULL document (full context kept), plus one remainder call for
#      items the classifier missed.
#   3. Results from multiple calls are merged and de-duplicated.
#   4. CIOMS "13. Lab Data" is parsed deterministically when rows 1..N verify.
#   5. completeness_<prompt>.json records what happened, for audit.
# ====================================================================

BATCH_SCOPE_NOTE = (
    "\n\n**BATCH SCOPE (this overrides the coverage instructions above)**: "
    "This request is one of several batches over the same document. Output entries "
    "ONLY for these items: {items}. Every other item is handled in a separate request "
    "— do not output it. Apply all other rules unchanged.\n"
)
REMAINDER_SCOPE_NOTE = (
    "\n\n**REMAINDER SCOPE (this overrides the coverage instructions above)**: "
    "The items {items} are handled in separate requests — do NOT output them. "
    "Output entries ONLY for qualifying items of the same kind that are present in the "
    "document but NOT in that list. If there are none, return the JSON structure with "
    "an empty list.\n"
)
PAGE_WINDOW_HEADER = (
    "[PARTIAL DOCUMENT — pages {start}-{end} of {total}. Extract EVERY qualifying item "
    "that appears in this text. The other pages are processed in separate requests.]\n"
)

# Keys used to recognise the same item coming back from two calls
# (overlapping page windows, needle batch + remainder, parser + LLM).
_DEDUPE_KEYS = {
    "laboratory_test_carried_out": ("name", "date", "time", "results", "unit"),
    "event_details":               ("adverse_event", "date_of_reaction_started"),
    "medical_conditions":          ("condition", "start_date"),
    "drug_information":            ("product_name", "type"),
}


def _fv(item, key):
    """Field value whether the field is {'value': x, 'confidence': n} or a bare value."""
    v = item.get(key) if isinstance(item, dict) else None
    if isinstance(v, dict):
        v = v.get("value", "")
    return v


def _norm(v) -> str:
    s = str(v if v is not None else "")
    s = s.replace("　", " ").replace("−", "-").lower()
    return re.sub(r"\s+", " ", s).strip()


def _strip_confidence(obj):
    if isinstance(obj, dict):
        return {k: _strip_confidence(v) for k, v in obj.items() if k != "confidence"}
    if isinstance(obj, list):
        return [_strip_confidence(x) for x in obj]
    return obj


def _canon(obj) -> str:
    return json.dumps(_strip_confidence(obj), sort_keys=True, ensure_ascii=False)


def _is_empty(v) -> bool:
    if v in (None, "", [], {}):
        return True
    if isinstance(v, dict) and set(v.keys()) <= {"value", "confidence"}:
        return str(v.get("value", "")).strip() == ""
    return False


def _dedupe_exact(items: list) -> list:
    seen, out = set(), []
    for it in items:
        c = _canon(it)
        if c not in seen:
            seen.add(c)
            out.append(it)
    return out


def _combine_items(a: dict, b: dict) -> dict:
    """Same item from two calls: keep a, fill its empty fields from b, union nested lists."""
    merged = dict(a)
    for k, v in b.items():
        if k not in merged or _is_empty(merged[k]):
            merged[k] = v
        elif isinstance(merged[k], list) and isinstance(v, list):
            merged[k] = _dedupe_exact(merged[k] + v)
    return merged


def dedupe_list_sections(result: dict) -> dict:
    """De-duplicate every list section of a merged multi-call result."""
    if not isinstance(result, dict):
        return result
    for sec, items in list(result.items()):
        if not isinstance(items, list):
            continue
        keys, out, index = _DEDUPE_KEYS.get(sec), [], {}
        for it in items:
            if isinstance(it, dict) and keys:
                k = tuple(_norm(_fv(it, f)) for f in keys)
                if not any(k):
                    k = ("__exact__", _canon(it))
            else:
                k = ("__exact__", _canon(it))
            if k in index:
                if isinstance(it, dict) and isinstance(out[index[k]], dict):
                    out[index[k]] = _combine_items(out[index[k]], it)
            else:
                index[k] = len(out)
                out.append(it)
        if len(out) != len(items):
            print(f"  dedupe '{sec}': {len(items)} → {len(out)} item(s)")
        result[sec] = out
    return result


def _merge_partial_results(parts: list) -> dict:
    """Concatenate list sections in call order; for dict/scalar sections the first non-empty value wins."""
    merged = {}
    for part in parts:
        if not isinstance(part, dict):
            continue
        for k, v in part.items():
            if isinstance(v, list):
                merged.setdefault(k, [])
                if isinstance(merged[k], list):
                    merged[k].extend(v)
            elif isinstance(v, dict):
                cur = merged.get(k)
                if not isinstance(cur, dict):
                    merged[k] = dict(v)
                else:
                    for fk, fv in v.items():
                        if fk not in cur or _is_empty(cur[fk]):
                            cur[fk] = fv
            elif k not in merged or _is_empty(merged[k]):
                merged[k] = v
    return merged


def _count_rows(result: dict) -> dict:
    return {k: len(v) for k, v in (result or {}).items() if isinstance(v, list)}


def _build_page_windows(n_pages: int) -> list:
    step, windows, start = max(1, PAGE_CHUNK_SIZE - PAGE_CHUNK_OVERLAP), [], 0
    while start < n_pages:
        end = min(n_pages, start + PAGE_CHUNK_SIZE)
        windows.append((start, end))
        if end >= n_pages:
            break
        start += step
    return windows


def _one_call(prompt_text: str, text: str, fname: str, doc_type: str, ctx: dict,
              label: str, use_cache: bool = False) -> dict:
    """
    One extraction call. Retries ONCE on empty/unparseable JSON.
    Raises _TruncatedOutput (caller splits) and _TransientBedrockError (Step Functions retries).
    """
    for attempt in (1, 2):
        ctx["calls"] += 1
        try:
            return classify_single(prompt_text, text, fname, doc_type, use_cache=use_cache)
        except (_TransientBedrockError, _TruncatedOutput):
            raise
        except ValueError as e:
            if attempt == 2:
                msg = f"{label}: no parseable JSON after 2 attempts ({e}) — slice skipped"
                print(f"  WARNING: {msg}")
                ctx["warnings"].append(msg)
                return {}
            print(f"  {label}: unparseable/empty JSON — retrying once")


def extract_by_page_windows(prompt_text: str, pages: list, fname: str,
                            doc_type: str, ctx: dict) -> dict:
    """
    Run a list prompt on windows of PAGE_CHUNK_SIZE pages in parallel.
    A window that still truncates is split in half (down to one page); a single
    page that still truncates keeps its partial result and is flagged.
    """
    n = len(pages)
    windows = _build_page_windows(n)
    print(f"  {fname}: page-window extraction — {n} pages, {len(windows)} window(s) "
          f"of {PAGE_CHUNK_SIZE} (overlap {PAGE_CHUNK_OVERLAP}), {CHUNK_WORKERS} worker(s)")

    def _halve_text(text: str):
        """Split one page's text near the middle, on a blank line if possible."""
        lines = text.splitlines()
        mid   = len(lines) // 2
        blank = lambda i: not lines[i].strip()
        # prefer a double blank line (gap between two rows/blocks), then a single one
        cut   = next((i for i in range(max(mid, 1), len(lines)) if blank(i) and blank(i - 1)), None)
        if cut is None:
            cut = next((i for i in range(mid, len(lines)) if blank(i)), None)
        if cut is None:
            cut = next((i for i in range(mid, 0, -1) if not lines[i].strip()), mid)
        cut = max(1, min(cut, len(lines) - 1))
        return "\n".join(lines[:cut]), "\n".join(lines[cut:])

    def run(s: int, e: int, texts: list, part: str = "") -> list:
        label = f"{fname} pages {s+1}-{e}{part}"
        text  = PAGE_WINDOW_HEADER.format(start=s + 1, end=e, total=n) + "\n".join(texts)
        try:
            return [_one_call(prompt_text, text, fname, doc_type, ctx, label)]
        except _TruncatedOutput as t:
            if len(texts) > 1:
                mid = len(texts) // 2
                print(f"  {label}: truncated — splitting into pages {s+1}-{s+mid} and {s+mid+1}-{e}")
                return run(s, s + mid, texts[:mid], part) + run(s + mid, e, texts[mid:], part)
            if len(texts[0]) > 1500 and texts[0].count("\n") > 4:
                a, b = _halve_text(texts[0])
                print(f"  {label}: single page truncated — splitting the page text in half")
                return run(s, e, [a], part + " (a)") + run(s, e, [b], part + " (b)")
            msg = f"{label}: still truncated at max_tokens and too small to split — partial rows kept"
            print(f"  WARNING: {msg}")
            ctx["warnings"].append(msg)
            return [t.partial]

    with ThreadPoolExecutor(max_workers=min(CHUNK_WORKERS, len(windows))) as pool:
        futures = [pool.submit(run, s, e, pages[s:e]) for s, e in windows]
        parts   = [p for f in futures for p in f.result()]   # window order preserved

    ctx["method"] = "page_windows"
    return dedupe_list_sections(_merge_partial_results(parts))


def extract_by_needle_batches(raw_prompt_text: str, prompt_s3_key: str, classification: dict,
                              raw_text: str, fname: str, doc_type: str,
                              values: list, ctx: dict) -> dict:
    """
    Re-run a needle prompt over the FULL document with the needle split into
    batches (halved again on truncation), plus one remainder call for items of
    the same kind that are not in the needle (classifier misses).
    """
    print(f"  {fname}: needle-batch extraction over {len(values)} needle value(s)")

    def run_batch(batch: list) -> list:
        label  = f"{fname} batch[{len(batch)}: {str(batch[0])[:30]}…]"
        prompt = (inject_needle_values(raw_prompt_text, prompt_s3_key, classification, values_override=batch)
                  + BATCH_SCOPE_NOTE.format(items=json.dumps(batch, ensure_ascii=False)))
        try:
            return [_one_call(prompt, raw_text, fname, doc_type, ctx, label)]
        except _TruncatedOutput as t:
            if len(batch) > 1:
                mid = len(batch) // 2
                return run_batch(batch[:mid]) + run_batch(batch[mid:])
            msg = f"{label}: one needle item still truncated — partial result kept"
            print(f"  WARNING: {msg}")
            ctx["warnings"].append(msg)
            return [t.partial]

    def run_remainder() -> list:
        label  = f"{fname} remainder"
        prompt = (inject_needle_values(raw_prompt_text, prompt_s3_key, classification, values_override=values)
                  + REMAINDER_SCOPE_NOTE.format(items=json.dumps(values, ensure_ascii=False)))
        try:
            return [_one_call(prompt, raw_text, fname, doc_type, ctx, label)]
        except _TruncatedOutput as t:
            msg = f"{label}: truncated — partial result kept"
            print(f"  WARNING: {msg}")
            ctx["warnings"].append(msg)
            return [t.partial]

    mid   = max(1, len(values) // 2)
    tasks = [values[:mid], values[mid:]] if len(values) > 1 else [values]
    with ThreadPoolExecutor(max_workers=min(CHUNK_WORKERS, len(tasks) + 1)) as pool:
        futures = [pool.submit(run_batch, b) for b in tasks if b] + [pool.submit(run_remainder)]
        parts   = [p for f in futures for p in f.result()]

    ctx["method"] = "needle_batches"
    return dedupe_list_sections(_merge_partial_results(parts))


def _combine_lab_parser_rows(result: dict, parser_rows: list) -> dict:
    """
    Parser rows (complete CIOMS table) first, then LLM rows from the rest of the
    document. An LLM row for a table analyte is dropped when it has no result or
    repeats a result already in the table (prevents duplicates).
    """
    result   = dict(result or {})
    llm_rows = result.get(LAB_SECTION_KEY, []) if isinstance(result.get(LAB_SECTION_KEY), list) else []
    table_results = {}
    for r in parser_rows:
        table_results.setdefault(_norm(_fv(r, "name")), set()).add(_norm(_fv(r, "results")))
    extra = []
    for r in llm_rows:
        if not isinstance(r, dict):
            continue
        name, res = _norm(_fv(r, "name")), _norm(_fv(r, "results"))
        if name in table_results and (not res or res in table_results[name]):
            continue
        extra.append(r)
    result[LAB_SECTION_KEY] = list(parser_rows) + extra
    return dedupe_list_sections(result)


def write_completeness_report(out_prefix: str, base: str, report: dict):
    """Audit file per list prompt — kept even when intermediate JSONs are cleaned up."""
    try:
        upload_bytes(f"{out_prefix}completeness_{base}.json",
                     json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
                     "application/json")
    except Exception as e:
        print(f"  completeness report not written (non-fatal): {e}")


def run_prompt_extraction(prompt_text: str, prompt_s3_key: str, classification_json: dict,
                          raw_text: str, raw_pages: list, fname: str, doc_type: str,
                          out_prefix: str) -> dict:
    """
    v7 router used by extract_task.
      • non-list prompts (reporter, demographics, narrative …) → classify_smart, unchanged
      • list prompts → page windows / needle batches / CIOMS lab parser as needed
    """
    base        = _prompt_base_name(prompt_s3_key)
    is_needle   = base in NEEDLE_REGISTRY
    is_paged    = base in PAGE_CHUNKED_PROMPT_BASES
    is_lab      = base == "patient_laboratory_prompt"
    updated     = inject_needle_values(prompt_text, prompt_s3_key, classification_json)

    if not (is_needle or is_paged):
        return classify_smart(updated, raw_text, fname, doc_type, use_cache=ENABLE_PROMPT_CACHE)

    ctx    = {"calls": 0, "warnings": [], "method": "single"}
    report = {"prompt": base, "doc_type": doc_type, "pages": len(raw_pages or [])}
    llm_text, llm_pages, lab_parse = raw_text, raw_pages, None

    # ── CIOMS lab table: deterministic, self-verifying ───────────────
    if is_lab and ENABLE_CIOMS_LAB_PARSER:
        lab_parse = parse_cioms_lab_table(raw_text)
        report["cioms_lab_table"] = {"source_rows": lab_parse["max_row"],
                                     "parser_complete": lab_parse["complete"]}
        if lab_parse["complete"]:
            print(f"  {fname}: CIOMS '13. Lab Data' parsed — {lab_parse['max_row']} rows (1..N verified); "
                  f"LLM will read the remaining text only")
            llm_text  = strip_cioms_lab_section(raw_text)
            llm_pages = None          # remaining text is small → single call
        elif lab_parse["max_row"]:
            msg = (f"CIOMS lab table found but rows are not contiguous 1..{lab_parse['max_row']} "
                   f"— parser NOT used, LLM result kept")
            print(f"  WARNING: {fname}: {msg}")
            ctx["warnings"].append(msg)

    # ── LLM pass ─────────────────────────────────────────────────────
    needle = get_needle_values(prompt_s3_key, classification_json) if is_needle else []
    if is_paged and llm_pages and len(llm_pages) > PAGE_CHUNK_SIZE:
        result = extract_by_page_windows(updated, llm_pages, fname, doc_type, ctx)
    else:
        try:
            ctx["calls"] += 1
            result = classify_single(updated, llm_text, fname, doc_type, use_cache=ENABLE_PROMPT_CACHE)
        except _TransientBedrockError:
            raise
        except _TruncatedOutput as t:
            print(f"  {fname}: single call truncated — splitting the work")
            if len(needle) > 1:
                result = extract_by_needle_batches(prompt_text, prompt_s3_key, classification_json,
                                                   llm_text, fname, doc_type, needle, ctx)
            elif llm_pages and len(llm_pages) > 1:
                result = extract_by_page_windows(updated, llm_pages, fname, doc_type, ctx)
            else:
                msg = "truncated and cannot be split further — partial result kept"
                print(f"  WARNING: {fname}: {msg}")
                ctx["warnings"].append(msg)
                result = t.partial
        except Exception as e:
            # input too large / unparseable — previous behaviour (chunked path)
            print(f"  {fname}: SINGLE path unavailable ({e}) — using CHUNKED path")
            ctx["method"] = "chunked"
            result = dedupe_list_sections(classify_chunked(updated, llm_text, fname, doc_type))

    if lab_parse and lab_parse["complete"]:
        llm_extra = len(result.get(LAB_SECTION_KEY, []) or []) if isinstance(result, dict) else 0
        result    = _combine_lab_parser_rows(result, lab_parse["rows"])
        report["cioms_lab_table"].update({
            "rows_from_parser":   len(lab_parse["rows"]),
            "llm_rows_returned":  llm_extra,
            "rows_final":         len(result.get(LAB_SECTION_KEY, [])),
        })
        ctx["method"] = f"cioms_parser+{ctx['method']}"

    report.update({"method": ctx["method"], "bedrock_calls": ctx["calls"],
                   "rows_extracted": _count_rows(result), "warnings": ctx["warnings"]})
    print(f"  {fname}: completeness — {report['rows_extracted']} via {ctx['method']} "
          f"({ctx['calls']} call(s), {len(ctx['warnings'])} warning(s))")
    write_completeness_report(out_prefix, base, report)
    return result


# ====================================================================
# CIOMS "13. Lab Data" deterministic parser
# ====================================================================
# CIOMS line listings print lab data as numbered rows:
#
#     #   Date   Test / Assessment / Notes          Results   Normal High / Low
#     14         C-reactive protein                            1.4      <- high
#                                                              0        <- low
#                Screening (Day −17)2.5mg/L                             <- visit + result + unit
#
# Because every row carries a sequence number (#), the parser can prove it
# found every row (1..N contiguous). It is ONLY trusted when that check
# passes; otherwise it returns [] and the LLM output is used unchanged.

_LAB_SECTION_RE   = re.compile(r"^\s*13\.\s*Lab\s*Data\b", re.I)
_LAB_HEADER_RE    = re.compile(r"^\s*#\s+Date\s+Test", re.I)
_LAB_ROW_RE       = re.compile(r"^\s{0,12}(\d{1,4})\s{2,}(.+?)\s*$")
_LAB_DATE_TOKEN   = re.compile(r"^(\d{1,2}[-/ ][A-Za-z]{3}[-/ ]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4})\s{2,}(.+)$")
_NUM_ONLY_RE      = re.compile(r"^\s*[<>≤≥]?\s*[-−]?\d+(?:[.,]\d+)?\s*$")
_SECTION_END_RE   = re.compile(r"^\s*\d{1,2}(?:\s*[-+]\s*\d{1,2})?\.\s+[A-Z]")   # e.g. "14-19. SUSPECT", "22. CONCOMITANT"
_PAGE_NOISE_RE    = re.compile(
    r"^\s*(Page\s+\d+\s+of\s+\d+|Mfr\.\s*Control\s*Number.*|ADDITIONAL\s+INFORMATION"
    r"|\d{1,2}-[A-Za-z]{3}-\d{4}\s+\d{1,2}:\d{2})\s*$", re.I)
_VISIT_RESULT_RE  = re.compile(r"^(?P<visit>.*\))\s*(?P<rest>.*)$")
_RESULT_UNIT_RE   = re.compile(r"^(?P<res>[<>≤≥]?\s*[-−]?\d+(?:[.,]\d+)?)\s*(?P<unit>.*)$")
_TIME_IN_VISIT_RE = re.compile(r"^\(\s*(?P<day>[^\s　)]+)[\s　].*?(?P<time>\d{1,2}:\d{2})\s*\)$")


def _split_cols(line: str) -> list:
    """Split a sort=True PyMuPDF line into visual columns (2+ spaces apart)."""
    return [c for c in re.split(r"\s{2,}", line.strip()) if c]


def _lab_wrap(v, conf=99):
    return {"value": v, "confidence": conf}


def _parse_lab_row(num: int, head_cols: list, body_lines: list) -> dict:
    """Turn one numbered row (header columns + following lines) into a lab entry."""
    name, date_col, results_col, high, low = "", "", "", "", ""
    cols = list(head_cols)

    # Optional date column right after the row number
    if cols and _LAB_DATE_TOKEN.match(cols[0] + "  x"):
        date_col = cols.pop(0)
    if cols:
        name = cols.pop(0)
    # Remaining header columns: [results] [normal-high]
    nums = [c for c in cols if _NUM_ONLY_RE.match(c)]
    rest = [c for c in cols if not _NUM_ONLY_RE.match(c)]
    if rest:
        results_col = " ".join(rest)
    if nums:
        high = nums[-1].strip()
        if len(nums) > 1 and not results_col:
            results_col = nums[0].strip()

    notes_lines = []
    for ln in body_lines:
        s = ln.strip()
        if not s:
            continue
        if _NUM_ONLY_RE.match(s) and high and not low:
            low = s
            continue
        notes_lines.append(s)

    visit, time_, result, unit = date_col, "", results_col, ""
    extra_notes = []
    for s in notes_lines:
        m = _VISIT_RESULT_RE.match(s)
        if m and not result:
            visit_part = m.group("visit").strip()
            rest_part  = m.group("rest").strip()
            tm = _TIME_IN_VISIT_RE.match(visit_part)
            if tm:                                   # "(Day2　13:24)" → date Day2, time 13:24
                visit, time_ = tm.group("day").strip(), tm.group("time")
            else:
                visit = visit or visit_part
            rm = _RESULT_UNIT_RE.match(rest_part)
            if rm:
                result, unit = rm.group("res").replace(" ", ""), rm.group("unit").strip()
            else:
                result = rest_part
        else:
            extra_notes.append(s)

    if result and not unit:
        rm = _RESULT_UNIT_RE.match(result)
        if rm and rm.group("unit"):
            result, unit = rm.group("res").replace(" ", ""), rm.group("unit").strip()

    entry = {"diagnosis_type": _lab_wrap("test"), "name": _lab_wrap(name)}
    if visit:  entry["date"]    = _lab_wrap(visit)
    if time_:  entry["time"]    = _lab_wrap(time_)
    if result: entry["results"] = _lab_wrap(result)
    if unit:   entry["unit"]    = _lab_wrap(unit)
    if low:    entry["value_low"]  = _lab_wrap(low)
    if high:   entry["value_high"] = _lab_wrap(high)
    if extra_notes:
        entry["notes"] = _lab_wrap(" ".join(extra_notes), 90)
    entry["_row_no"] = num
    return entry


def parse_cioms_lab_table(raw_text: str) -> dict:
    """
    Parse every numbered row of CIOMS '13. Lab Data' (across all continuation pages).
    Returns {"rows": [...], "max_row": N, "complete": bool}.
    'complete' is True only when rows 1..N were all found exactly once.
    """
    rows, in_section, cur_num, cur_head, cur_body = {}, False, None, None, []

    def flush():
        if cur_num is not None and cur_num not in rows:
            rows[cur_num] = _parse_lab_row(cur_num, cur_head, cur_body)

    for line in raw_text.splitlines():
        if _LAB_SECTION_RE.match(line):
            in_section = True
            continue
        if not in_section:
            continue
        if _LAB_HEADER_RE.match(line) or _PAGE_NOISE_RE.match(line) or not line.strip():
            continue
        if _SECTION_END_RE.match(line):                  # next CIOMS section → stop
            flush(); cur_num = None; in_section = False
            continue
        m = _LAB_ROW_RE.match(line)
        cols = _split_cols(line)
        # A new row must carry the NEXT sequence number; anything else
        # (e.g. a wrapped note starting with a number) stays in the body.
        expected = (cur_num + 1) if cur_num is not None else (max(rows) + 1 if rows else 1)
        if m and len(cols) >= 2 and cols[0].isdigit() and int(cols[0]) == expected:
            flush()
            cur_num, cur_head, cur_body = int(cols[0]), cols[1:], []
            continue
        if cur_num is not None:
            cur_body.append(line)
    flush()

    if not rows:
        return {"rows": [], "max_row": 0, "complete": False}
    max_row  = max(rows)
    complete = sorted(rows) == list(range(1, max_row + 1)) and all(
        r["name"]["value"] for r in rows.values())
    ordered  = [rows[k] for k in sorted(rows)]
    for r in ordered:
        r.pop("_row_no", None)
    return {"rows": ordered, "max_row": max_row, "complete": complete}


def strip_cioms_lab_section(raw_text: str) -> str:
    """
    Return raw_text with the CIOMS '13. Lab Data' table rows removed (one
    placeholder line per section). Used so the LLM lab pass only reads the
    narrative/other text once the table has been parsed deterministically.
    """
    out, in_section = [], False
    for line in raw_text.splitlines():
        if _LAB_SECTION_RE.match(line):
            in_section = True
            out.append("[13. Lab Data table: rows already extracted separately — do not extract them again]")
            continue
        if in_section:
            if _SECTION_END_RE.match(line):
                in_section = False
                out.append(line)
            elif _PAGE_NOISE_RE.match(line):
                out.append(line)
            continue
        out.append(line)
    return "\n".join(out)


# ====================================================================
# Archilles
# ====================================================================
def update_final_json_with_nid(final_key: str, nid):
    try:
        obj        = s3.get_object(Bucket=BUCKET, Key=final_key)
        final_json = json.loads(obj["Body"].read().decode("utf-8"))
        final_json["record_id"] = nid
        upload_bytes(final_key,
                     json.dumps(final_json, ensure_ascii=False, indent=2).encode("utf-8"),
                     "application/json")
        print(f"Updated final JSON with record_id = {nid}")
    except Exception as e:
        print(f"Failed to update final JSON with nid: {e}")

def notify_archilles_api(client_id: str, original_stem: str, final_key: str,
                          no_of_pages: int = 0, form_id=None):
    client_config          = get_client_config(client_id)
    archilles_api_url      = client_config["archilles_api_url"]
    archilles_access_token = client_config["archilles_access_token"]

    if not archilles_api_url:
        print(f"  [{client_id}] Archilles API URL not configured — skipping")
        return
    try:
        json_folder    = raw_json_folder(client_id)
        json_reference = (final_key[len(json_folder):]
                          if final_key.startswith(json_folder) else final_key)
        payload = {
            "title":                       f"{original_stem}.pdf",
            "field_actual_json_reference": json_reference,
            "field_file_name":             os.path.basename(final_key),
            "access_token":                archilles_access_token,
            # source_doc: record_id from the display stand JSON (sh-json/ folder)
            # page_count: total pages of the processed PDF
            "source_doc":                  form_id if form_id is not None else FORM_ID,
            "page_count":                  no_of_pages,
        }
        data    = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if ARCHILLES_COOKIE:
            headers["Cookie"] = ARCHILLES_COOKIE
        req = urllib.request.Request(archilles_api_url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as response:
            resp_json = json.loads(response.read().decode("utf-8"))
            nid       = resp_json.get("nid")
            if nid is not None:
                update_final_json_with_nid(final_key, nid)
                print(f"  [{client_id}] Archilles notified: source_doc={form_id if form_id is not None else FORM_ID}, page_count={no_of_pages}, nid={nid}")
            else:
                print(f"  [{client_id}] Archilles response did not contain 'nid'")
    except Exception as e:
        print(f"  [{client_id}] Failed to notify Archilles API: {e}")

# ====================================================================
# Intermediate file cleanup
# ====================================================================
def delete_intermediate_json_files(out_prefix: str):
    KEEP = {"Classification.json", "raw.txt", "layout.json", "raw_pages.json"}
    try:
        paginator = s3.get_paginator("list_objects_v2")
        to_delete = []
        for page in paginator.paginate(Bucket=BUCKET, Prefix=out_prefix):
            for obj in page.get("Contents", []):
                key      = obj["Key"]
                basename = os.path.basename(key)
                if "/images/" in key:
                    continue
                if basename.startswith("output_") and basename.endswith(".json"):
                    continue
                if basename.startswith("completeness_"):      # v7 audit reports
                    continue
                if basename in KEEP:
                    continue
                if basename.endswith(".json"):
                    to_delete.append({"Key": key})
        if to_delete:
            for i in range(0, len(to_delete), 1000):
                s3.delete_objects(Bucket=BUCKET, Delete={"Objects": to_delete[i:i+1000]})
            print(f"Deleted {len(to_delete)} intermediate file(s) from {out_prefix}")
    except Exception as e:
        print(f"Cleanup failed (non-fatal): {e}")

# ====================================================================
# ====================================================================
#  STEP FUNCTIONS TASK HANDLERS  (rebuilt for correctness)
#
#  DESIGN PRINCIPLES:
#  1. render_task   — stores ALL metadata in layout.json so merge_task
#                     never depends on Step Functions state passing
#  2. classify_task — builds prompt_items for the Map state dynamically
#                     from config; never hardcoded in ASL
#  3. extract_task  — runs ONE prompt, writes result to S3, returns only
#                     a completion signal (not the full result payload)
#  4. merge_task    — reads EVERYTHING from S3; assembles only when all
#                     prompt files exist; notifies Archilles LAST after
#                     the complete final JSON is confirmed written to S3
# ====================================================================
# ====================================================================

def render_task(event: dict) -> dict:
    """
    STATE: Render
    Receives: task, pdf_key, out_prefix, original_stem, timestamp, extraction_prompts
    Returns:  raw_text_key, doc_type, out_prefix, layout_key, file_info,
              no_of_pages, original_stem, timestamp, extraction_prompts, client_id
    Stores in layout.json: pages, file_info, no_of_pages, original_stem,
                           timestamp, client_id — so merge_task can reconstruct
                           everything from S3 without relying on Step Functions
                           state passing.
    """
    pdf_key          = event["pdf_key"]
    out_prefix       = event["out_prefix"]
    original_stem    = event["original_stem"]
    timestamp        = event["timestamp"]
    # client_id is normally passed through explicitly (set by _start_step_functions),
    # but fall back to deriving it from the pdf_key so this task stays self-contained.
    client_id        = event.get("client_id") or get_client_id_from_pdf_key(pdf_key)
    client_config    = get_client_config(client_id)
    extraction_prompts = event.get("extraction_prompts", client_config["extraction_prompt_sequence"][1:])

    pdf_bytes = s3.get_object(Bucket=BUCKET, Key=pdf_key)["Body"].read()
    doc       = fitz.open(stream=pdf_bytes, filetype="pdf")
    scale     = RENDER_DPI / 72.0

    raw_text, layout, file_info, no_of_pages, raw_pages = render_pdf_pages(doc, out_prefix, scale)
    doc_type     = detect_document_type(raw_text)
    raw_text_key = f"{out_prefix}raw.txt"
    layout_key   = f"{out_prefix}layout.json"
    raw_pages_key = f"{out_prefix}raw_pages.json"     # v7

    upload_bytes(raw_text_key, raw_text.encode("utf-8"), "text/plain; charset=utf-8")
    # v7: per-page text for page-window extraction (list prompts)
    upload_bytes(raw_pages_key,
                 json.dumps(raw_pages, ensure_ascii=False).encode("utf-8"),
                 "application/json")

    # Store ALL metadata needed by merge_task inside layout.json.
    # This makes merge_task fully self-contained — it never needs
    # original_stem or timestamp from the Step Functions event.
    layout_meta = {
        "pdf_s3_key":      pdf_key,
        "created_utc":     datetime.utcnow().isoformat() + "Z",
        "dpi":             RENDER_DPI,
        "doc_type":        doc_type,
        "file_info":       file_info,
        "no_of_pages":     no_of_pages,
        "original_stem":   original_stem,
        "timestamp":       timestamp,
        "client_id":       client_id,
        "pages":           layout,
    }
    upload_bytes(layout_key,
                 json.dumps(layout_meta, ensure_ascii=False).encode("utf-8"),
                 "application/json")
    print(f"  [{client_id}] Render complete: {no_of_pages} pages, doc_type={doc_type}")

    return {
        "raw_text_key":       raw_text_key,
        "raw_pages_key":      raw_pages_key,
        "doc_type":           doc_type,
        "out_prefix":         out_prefix,
        "layout_key":         layout_key,
        "file_info":          file_info,
        "no_of_pages":        no_of_pages,
        "original_stem":      original_stem,
        "timestamp":          timestamp,
        "pdf_key":            pdf_key,
        "client_id":          client_id,
        "extraction_prompts": extraction_prompts,
    }


def classify_task(event: dict) -> dict:
    """
    STATE: Classify
    Receives: task, raw_text_key, doc_type, out_prefix, extraction_prompts
              (+ layout_key, file_info, no_of_pages, original_stem, timestamp
               carried through from render_task for PreparePromptInputs)
    Returns:  classification_json, prompt_items (for the Map state),
              and all other keys passed through.

    Builds prompt_items dynamically — each item is a self-contained event
    for one extract_task Lambda call.  No prompt keys are in the ASL.
    """
    out_prefix         = event["out_prefix"]
    raw_text_key       = event["raw_text_key"]
    doc_type           = event.get("doc_type", "UNKNOWN")
    # Fall back to deriving client_id from out_prefix (clients/<id>/raw-json/...)
    # so this task stays self-contained even if Step Functions didn't forward it.
    client_id          = event.get("client_id") or get_client_id_from_path(out_prefix)
    client_config      = get_client_config(client_id)
    extraction_prompt_sequence = client_config["extraction_prompt_sequence"]
    extraction_prompts = event.get("extraction_prompts", extraction_prompt_sequence[1:])

    raw_text      = read_s3_text(raw_text_key)
    prompt_s3_key = extraction_prompt_sequence[0]
    fname         = os.path.basename(prompt_s3_key)
    prompt_text   = read_s3_text(prompt_s3_key)

    print(f"  [{client_id}] Running classify prompt: {fname}")
    try:
        result = classify_smart(prompt_text, raw_text, fname, doc_type)
    except Exception as e:
        # Transient failure (throttle/timeout) — re-raise so Step Functions
        # retries the whole Classify state. Classification feeds needle values
        # into all 7 downstream prompts, so a partial/empty classification
        # would cascade into empty extraction everywhere. Never swallow it.
        print(f"  [{client_id}] Classification failed with transient error: {e} — re-raising for Step Functions retry")
        raise

    if not result:
        print(f"  [{client_id}] Empty classification result — using default schema")
        result = {}

    # Save classification output to two keys:
    # 1. Named file (e.g. data_classification_prompt.json) for consistency
    # 2. Classification.json — canonical name that merge_task always reads
    clean_file = get_clean_filename(prompt_s3_key)
    upload_bytes(f"{out_prefix}{clean_file}",
                 json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json")
    upload_bytes(f"{out_prefix}Classification.json",
                 json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json")
    print(f"  [{client_id}] Classification saved to S3")

    # Build prompt_items: each item is a complete, self-contained input
    # for one extract_task Lambda. classification_json is embedded here
    # so the Map state Lambdas do not need to read it from S3 themselves.
    prompt_items = [
        {
            "task":               "extract_prompt",
            "prompt_s3_key":      ps3,
            "raw_text_key":       raw_text_key,
            "raw_pages_key":      event.get("raw_pages_key", f"{out_prefix}raw_pages.json"),
            "doc_type":           doc_type,
            "out_prefix":         out_prefix,
            "classification_json": result,
            "client_id":          client_id,
        }
        for ps3 in extraction_prompts
    ]

    return {
        "classification_json": result,
        "prompt_items":        prompt_items,
        "out_prefix":          out_prefix,
        "doc_type":            doc_type,
        "raw_text_key":        raw_text_key,
        "raw_pages_key":       event.get("raw_pages_key", f"{out_prefix}raw_pages.json"),
        "client_id":           client_id,
        # Pass through everything PreparePromptInputs and Merge need
        "layout_key":          event.get("layout_key",    f"{out_prefix}layout.json"),
        "file_info":           event.get("file_info",     {}),
        "no_of_pages":         event.get("no_of_pages",   0),
        "original_stem":       event.get("original_stem", ""),
        "timestamp":           event.get("timestamp",     ""),
        "extraction_prompts":  extraction_prompts,
    }


def extract_task(event: dict) -> dict:
    """
    STATE: ExtractParallel — Map state, one Lambda per prompt 2-8.
    Receives: task, prompt_s3_key, raw_text_key, doc_type,
              out_prefix, classification_json
    Returns:  { done: true, prompt_s3_key, out_key }
              — Merge reads results from S3, not from this return value.

    The result is ALWAYS written to S3 — even on failure.
    On failure: writes {"error": "..."} so merge_task can detect it
    and substitute the default empty schema for that section.
    """
    prompt_s3_key       = event["prompt_s3_key"]
    out_prefix          = event["out_prefix"]
    raw_text_key        = event["raw_text_key"]
    doc_type            = event.get("doc_type", "UNKNOWN")
    classification_json = event.get("classification_json", {})
    client_id           = event.get("client_id") or get_client_id_from_path(out_prefix)

    fname       = os.path.basename(prompt_s3_key)
    out_key     = f"{out_prefix}{get_clean_filename(prompt_s3_key)}"

    print(f"  [{client_id}] Running extraction prompt: {fname}")
    raw_text    = read_s3_text(raw_text_key)
    prompt_text = read_s3_text(prompt_s3_key)

    # v7: per-page text for page-window extraction (falls back to whole text
    # when raw_pages.json is missing, e.g. for runs rendered by v6).
    raw_pages = None
    try:
        raw_pages = read_s3_json(event.get("raw_pages_key") or f"{out_prefix}raw_pages.json")
        if not isinstance(raw_pages, list):
            raw_pages = None
    except Exception:
        print(f"  [{client_id}] raw_pages.json not found — page windows disabled for this run")

    try:
        # v7: run_prompt_extraction injects the needle and routes:
        #   non-list prompts → classify_smart (unchanged behaviour)
        #   list prompts     → page windows / needle batches / CIOMS lab parser
        # Caching still honors ENABLE_PROMPT_CACHE (default OFF).
        result = run_prompt_extraction(prompt_text, prompt_s3_key, classification_json,
                                       raw_text, raw_pages, fname, doc_type, out_prefix)
    except Exception as e:
        # CRITICAL for bulk consistency: a throttle/timeout failure here is
        # TRANSIENT, not "no data". If we swallowed it and wrote {"error":...},
        # merge would substitute an empty skeleton and the document would show
        # fewer fields than the same doc processed alone (the bulk vs single
        # discrepancy). Instead we RE-RAISE so the Step Functions Map state
        # retries this prompt (the ASL ExtractPrompt state has its own Retry
        # block). Only after Step Functions exhausts ITS retries does the
        # PromptFailed catch store an empty result.
        print(f"  [{client_id}] {fname} failed with transient error: {e} — re-raising for Step Functions retry")
        raise

    if not result:
        # Genuinely empty (model returned valid-but-empty JSON) — this is a
        # real "no data found" signal, safe to store. merge substitutes skeleton.
        print(f"  [{client_id}] {fname} returned empty result — using default schema at merge")
        result = {}

    # Write the successful result to S3
    upload_bytes(out_key,
                 json.dumps(result, ensure_ascii=False).encode("utf-8"),
                 "application/json")
    print(f"  [{client_id}] {fname} result written to S3: {out_key}")

    # Return minimal signal — merge_task reads from S3, not this payload
    return {"done": True, "prompt_s3_key": prompt_s3_key, "out_key": out_key, "client_id": client_id}


def _merge_genai_into_skeleton(skeleton_section, genai_section):
    """
    Merge GenAI output into the skeleton section field-by-field.

    PURPOSE: The prompts say "Output only non-empty keys" so GenAI omits
    fields it has no data for (e.g. preferred_term_code, MedDRA_LLT_code).
    A plain replacement of skeleton with GenAI output loses those fields.
    This function ensures every skeleton field is always present in the
    output — GenAI values fill in what they can, skeleton blanks cover the rest.

    Rules:
      - dict section  : merge key-by-key; skeleton keys absent from GenAI kept blank
      - list section  : zip GenAI items with skeleton items; extra GenAI items appended
                        as-is; each item merged recursively
      - scalar / other: GenAI value wins outright
    """
    import copy

    # Both are lists (e.g. event_details, drug_information)
    if isinstance(skeleton_section, list) and isinstance(genai_section, list):
        skeleton_item = skeleton_section[0] if skeleton_section else {}
        merged_list   = []
        for genai_item in genai_section:
            if isinstance(genai_item, dict) and isinstance(skeleton_item, dict):
                # Start from a fresh copy of the skeleton item, overlay GenAI values
                merged_item = copy.deepcopy(skeleton_item)
                for field_key, field_val in genai_item.items():
                    if isinstance(field_val, list) and isinstance(merged_item.get(field_key), list):
                        # Nested list (e.g. severity_of_the_illness, dosage_information)
                        merged_item[field_key] = _merge_genai_into_skeleton(
                            merged_item[field_key], field_val
                        )
                    else:
                        merged_item[field_key] = field_val
                merged_list.append(merged_item)
            else:
                merged_list.append(genai_item)
        return merged_list

    # Both are dicts (e.g. general_info, narrative_summary, patient_generic_info)
    if isinstance(skeleton_section, dict) and isinstance(genai_section, dict):
        merged = copy.deepcopy(skeleton_section)
        for field_key, field_val in genai_section.items():
            merged[field_key] = field_val
        return merged

    # Scalar or mismatched types — GenAI wins
    return genai_section


def merge_task(event: dict) -> dict:
    """
    STATE: Merge — runs AFTER all Map state iterations are complete.
    Receives: task, out_prefix, doc_type (from ASL Parameters)
    Everything else is read from S3 — this function is fully self-contained.

    Order of operations:
      1. Read layout.json → get pages, file_info, no_of_pages,
                             original_stem, timestamp
      2. Read Classification.json → prompt 1 output
      3. Read each prompt 2-8 output JSON → use default if missing/error
      4. recursive_enrich + attach_regions → build final_body
      5. Write final JSON to S3  ← complete before Archilles is called
      6. notify_archilles_api    ← only after full JSON confirmed on S3
      7. Cleanup intermediate files
    """
    out_prefix = event["out_prefix"]
    # layout.json (Step 1 below) carries client_id too, but we need it
    # before that read to build the failure message consistently, so
    # derive it here the same self-contained way the other tasks do.
    client_id  = event.get("client_id") or get_client_id_from_path(out_prefix)

    # ── Step 1: Read layout.json — source of truth for all metadata ───
    layout_key = f"{out_prefix}layout.json"
    try:
        layout_data = read_s3_json(layout_key)
    except Exception as e:
        raise RuntimeError(f"[{client_id}] Cannot read layout.json — cannot build final JSON: {e}")

    layout        = layout_data.get("pages",        [])
    file_info     = layout_data.get("file_info",    {})
    no_of_pages   = layout_data.get("no_of_pages",  0)
    original_stem = layout_data.get("original_stem","")
    timestamp     = layout_data.get("timestamp",    datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    doc_type      = layout_data.get("doc_type",     event.get("doc_type", "UNKNOWN"))
    # layout.json's own client_id (set by render_task) wins if present —
    # it's the most authoritative source since it was written at render time.
    client_id     = layout_data.get("client_id", client_id)

    client_config = get_client_config(client_id)
    extraction_prompt_sequence = client_config["extraction_prompt_sequence"]
    prompt_to_section_map      = client_config["prompt_to_section_map"]

    print(f"  [{client_id}] Merge starting: {original_stem}, {no_of_pages} pages, doc_type={doc_type}")

    # ── Step 2: Read Classification.json (prompt 1 output) ───────────
    try:
        classification_json = read_s3_json(f"{out_prefix}Classification.json")
        if not classification_json:
            classification_json = {}
    except Exception:
        print(f"  [{client_id}] Classification.json missing — using default schema for general_info")
        classification_json = {}

    first_s3_key  = extraction_prompt_sequence[0]
    first_base    = _prompt_base_name(first_s3_key)
    first_section = prompt_to_section_map.get(first_base, "general_info")

    # Default-first: parse skeleton from prompt, then overwrite with GenAI
    first_skeleton = get_skeleton_for_prompt(first_s3_key)
    body = first_skeleton if first_skeleton else {first_section: {}}

    classify_data = classification_json.get(first_section, classification_json)
    if classify_data and isinstance(classify_data, dict) and classify_data:
        # Merge into skeleton so any keys GenAI omitted stay as blank fields
        body[first_section] = _merge_genai_into_skeleton(body[first_section], classify_data)
        print(f"  Prompt 1: '{first_section}' merged from GenAI")
    else:
        print(f"  Prompt 1: '{first_section}' empty — skeleton retained")

    # ── Step 3: Build body for prompts 2-8 ───────────────────────────
    #
    # STRATEGY: default-first, overwrite-with-genai
    #   1. Always start by inserting the default empty skeleton for every
    #      section each prompt owns.
    #   2. If GenAI returned real data, overwrite ONLY the sections that
    #      have actual content — empty lists or empty dicts from GenAI
    #      are treated the same as missing data (skeleton is kept).
    #
    # This guarantees:
    #   - Every section ALWAYS appears in the final JSON
    #   - Every field is ALWAYS enriched with display stand metadata
    #   - End users can manually fill any section GenAI missed
    #   - Array sections (event_details, drug_information etc) always
    #     have at least one skeleton item — never an empty list []
    #
    # Sections each prompt owns (prompts returning multiple sections):
    PROMPT_SECTIONS = {
        "patient_demographics_prompt": ["patient_generic_info", "patient_medical_info"],
    }
    # All other prompts own exactly the section in PROMPT_TO_SECTION_MAP.

    for s3_key in extraction_prompt_sequence[1:]:
        base           = _prompt_base_name(s3_key)
        primary        = prompt_to_section_map.get(base, base)
        sections_owned = PROMPT_SECTIONS.get(base, [primary])

        # Step 3a — parse skeleton from the prompt file, insert all its sections
        prompt_skeleton = get_skeleton_for_prompt(s3_key)
        for k, v in prompt_skeleton.items():
            body[k] = v   # pre-populate with skeleton

        # Step 3b — try to load GenAI output from S3
        key  = f"{out_prefix}{get_clean_filename(s3_key)}"
        data = None
        try:
            raw = read_s3_json(key)
            # Reject: empty dict, error-only dict, or non-dict
            if (isinstance(raw, dict)
                    and raw
                    and not (len(raw) == 1 and "error" in raw)):
                data = raw
        except Exception:
            pass

        if data is None:
            print(f"  {base}: no valid GenAI data — skeleton retained")
            continue

        # Step 3c — merge GenAI into skeleton field-by-field
        # _merge_genai_into_skeleton ensures fields GenAI omitted
        # (prompt says "output only non-empty keys") stay as blank
        # skeleton entries so the UI always has every field.
        for k, v in data.items():
            if isinstance(v, list) and len(v) == 0:
                print(f"  {base}: '{k}' empty list from GenAI — skeleton retained")
                continue
            if isinstance(v, dict) and not v:
                print(f"  {base}: '{k}' empty dict from GenAI — skeleton retained")
                continue
            if k in body:
                body[k] = _merge_genai_into_skeleton(body[k], v)
            else:
                body[k] = v
            n = len(v) if isinstance(v, (list, dict)) else 1
            print(f"  {base}: '{k}' merged from GenAI ({n} item(s))")

    # ── Step 4: Load doc-type-specific display stand, then enrich ───
    print(f"  [{client_id}] Loading display stand for doc_type='{doc_type}'...")
    display_lookup, doc_form_id = get_display_lookup_for_doc_type(doc_type, client_id)

    # Use doc-type-specific form_id if available, otherwise fall back to global
    effective_form_id = doc_form_id if doc_form_id is not None else FORM_ID

    print("  Running recursive_enrich...")
    enriched_body = recursive_enrich(body, lookup=display_lookup)
    print("  Running attach_regions...")
    final_body    = attach_regions(enriched_body, layout)

    # ── Step 5: Build and write final JSON to S3 ──────────────────────
    final_json = {
        "file_name": f"{original_stem}.pdf",
        "form_id":   effective_form_id,
        "meta_info": {
            "form_identifier":                 "ISIForm",
            "original_file_name":              f"{original_stem}.pdf",
            "external_id":                     "",
            "document_id":                     "",
            "client_config_id":                "",
            "ocr_config_id":                   "",
            "client_fields":                   {},
            "batch_request_id":                "",
            "no_of_pages":                     no_of_pages,
            "file_info":                       file_info,
            "is_send_for_translation_enabled": "yes",
            "is_non_english":                  "no",
        },
        "gen_ai_response": {"body": final_body},
        "audit_trail":     {"bucket_name": BUCKET, "upload_file_path": out_prefix},
    }

    final_key      = f"{out_prefix}output_{original_stem}_{timestamp}.json"
    final_json_bytes = json.dumps(final_json, ensure_ascii=False, indent=2).encode("utf-8")
    upload_bytes(final_key, final_json_bytes, "application/json")
    print(f"  [{client_id}] Final JSON written to S3: {final_key} ({len(final_json_bytes):,} bytes)")

    # ── Step 6: Notify Archilles — ONLY after full JSON is on S3 ─────
    # This is the critical ordering fix: Archilles is notified last,
    # after the complete final JSON is confirmed written. Previously
    # this was called before the JSON was fully assembled in some
    # race scenarios via parallel processing.
    notify_archilles_api(client_id, original_stem, final_key, no_of_pages, effective_form_id)

    # ── Step 7: Cleanup intermediate files (gated) ───────────────────
    # While KEEP_INTERMEDIATE_JSON=true, per-prompt raw Bedrock JSONs are
    # preserved in S3 so you can validate the model responses. Set the env
    # var to "false" to re-enable cleanup once extraction is verified.
    if KEEP_INTERMEDIATE_JSON:
        print(f"  [{client_id}] KEEP_INTERMEDIATE_JSON=true — preserving per-prompt JSONs in {out_prefix}")
    else:
        delete_intermediate_json_files(out_prefix)

    return {
        "final_key":   final_key,
        "out_prefix":  out_prefix,
        "model_id":    MODEL_ID,
        "no_of_pages": no_of_pages,
        "doc_type":    doc_type,
        "client_id":   client_id,
    }
# ====================================================================
# ENTRY POINT — routes SQS / S3 / manual test → Step Functions
# ====================================================================
def _write_failure_marker(pdf_key: str, reason: str):
    """Write a .FAILED marker to S3 so no document disappears silently."""
    try:
        stem = os.path.splitext(os.path.basename(pdf_key))[0]
        try:
            client_id  = get_client_id_from_pdf_key(pdf_key)
            marker_key = f"{raw_json_folder(client_id)}_FAILED/{stem}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
        except ValueError:
            # Key didn't match clients/<id>/pdf/... at all — still record the
            # failure somewhere rather than losing it, under a bucket-level catch-all.
            marker_key = f"{CLIENTS_ROOT_FOLDER}_UNROUTABLE_FAILED/{stem}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
        upload_bytes(
            marker_key,
            f"pdf_key : {pdf_key}\nreason  : {reason}\ntime    : {datetime.utcnow().isoformat()}Z\n"
            .encode("utf-8"),
            "text/plain"
        )
        print(f"Failure marker written: {marker_key}")
    except Exception as e:
        print(f"Could not write failure marker: {e}")


def _start_step_functions(pdf_key: str) -> dict:
    """
    Start a Step Functions execution for one PDF.
    Returns immediately (<1s) so the SQS visibility timeout is never threatened.

    client_id is derived from pdf_key here (clients/<client_id>/pdf/<file>.pdf)
    and threaded through the whole payload, so render/classify/extract/merge
    all know which client's configuration.json + prompts to use.
    """
    client_id     = get_client_id_from_pdf_key(pdf_key)   # raises ValueError if key doesn't match layout
    client_config = get_client_config(client_id)

    original_stem = os.path.splitext(os.path.basename(pdf_key))[0]
    timestamp     = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_prefix    = f"{raw_json_folder(client_id)}{original_stem}_{timestamp}/"

    input_payload = {
        "task":          "render",
        "pdf_key":       pdf_key,
        "out_prefix":    out_prefix,
        "original_stem": original_stem,
        "timestamp":     timestamp,
        "client_id":     client_id,
        # Pass the full prompt sequence for the Map state to iterate
        "extraction_prompts": client_config["extraction_prompt_sequence"][1:],
    }

    if not STEP_FUNCTIONS_ARN:
        # No Step Functions configured — fall back to in-process execution
        print(f"  [{client_id}] STEP_FUNCTIONS_ARN not set — running in-process (single Lambda mode)")
        return _run_in_process(pdf_key, original_stem, timestamp, out_prefix, client_id)

    # Step Functions execution names allow only: letters, numbers, hyphens, underscores.
    # Replace spaces, dots, and any other invalid characters, then deduplicate underscores.
    safe_stem      = re.sub(r'[^a-zA-Z0-9\-_]', '_', original_stem)
    safe_stem      = re.sub(r'_+', '_', safe_stem).strip('_')
    # Append 6-char random suffix to prevent execution name collisions
    # when two docs produce the same sanitised stem after truncation
    rand_suffix    = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
    execution_name = f"{safe_stem[:50]}_{timestamp}_{rand_suffix}"
    try:
        resp = sfn.start_execution(
            stateMachineArn=STEP_FUNCTIONS_ARN,
            name=execution_name,
            input=json.dumps(input_payload),
        )
    except Exception as e:
        reason = f"sfn.start_execution failed: {e}"
        print(f"CRITICAL [{client_id}]: {reason} for {pdf_key}")
        _write_failure_marker(pdf_key, reason)
        raise   # re-raise so SQS routes this message to DLQ for retry

    print(f"  [{client_id}] Started Step Functions execution: {resp['executionArn']}")
    return {
        "mode":          "SFN_STARTED",
        "execution_arn": resp["executionArn"],
        "pdf_key":       pdf_key,
        "out_prefix":    out_prefix,
        "client_id":     client_id,
    }


def _run_in_process(pdf_key: str, original_stem: str, timestamp: str,
                     out_prefix: str, client_id: str) -> dict:
    """
    Fallback: run the full pipeline in a single Lambda invocation.
    Used when STEP_FUNCTIONS_ARN is not configured (e.g. local testing).
    Prompts 2-8 still run in parallel via ThreadPoolExecutor.
    """
    t0    = time.time()
    event = {"pdf_key": pdf_key, "out_prefix": out_prefix,
             "original_stem": original_stem, "timestamp": timestamp,
             "client_id": client_id}

    extraction_prompt_sequence = get_client_config(client_id)["extraction_prompt_sequence"]

    # Stage 1: Render
    event = render_task(event)

    # Stage 2: Classify
    event = classify_task(event)
    classification_json = event["classification_json"]
    raw_text_key        = event["raw_text_key"]
    doc_type            = event.get("doc_type", "UNKNOWN")

    # Stage 3: Extract prompts 2-8 in parallel
    prompt_events = [
        {**event, "prompt_s3_key": ps3, "classification_json": classification_json,
         "raw_text_key": raw_text_key, "doc_type": doc_type, "client_id": client_id}
        for ps3 in extraction_prompt_sequence[1:]
    ]
    with ThreadPoolExecutor(max_workers=min(len(prompt_events), 7)) as pool:
        futures = [pool.submit(extract_task, pe) for pe in prompt_events]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"  [{client_id}] Parallel prompt failed: {e}")

    # Stage 4: Merge
    result = merge_task(event)
    result["total_execution_time_seconds"] = round(time.time() - t0, 3)
    result["mode"] = "IN_PROCESS"
    return result


def lambda_handler(event, context):
    """
    ROUTING:
      event["task"] == "render"         → render_task   (called by Step Functions)
      event["task"] == "classify"       → classify_task (called by Step Functions)
      event["task"] == "extract_prompt" → extract_task  (called by Step Functions Map)
      event["task"] == "merge"          → merge_task    (called by Step Functions)
      event["pdf_key"]                  → manual test   (direct Lambda invocation)
      event["Records"][0].eventSource == "aws:sqs"  → SQS trigger
      event["Records"][0].eventSource == "aws:s3"   → direct S3 trigger (legacy)
    """

    # ── Step Functions task routing ───────────────────────────────────
    task = event.get("task")
    if task == "render":
        return render_task(event)
    if task == "classify":
        return classify_task(event)
    if task == "extract_prompt":
        return extract_task(event)
    if task == "merge":
        return merge_task(event)

    # ── Manual test invocation ────────────────────────────────────────
    if isinstance(event, dict) and "pdf_key" in event:
        t0 = time.time()
        result = _start_step_functions(event["pdf_key"])
        result["total_execution_time_seconds"] = round(time.time() - t0, 3)
        return result

    # ── SQS trigger (flood-safe) ──────────────────────────────────────
    if "Records" in event and len(event["Records"]) > 0:
        first_record = event["Records"][0]

        if first_record.get("eventSource") == "aws:sqs":
            # Process EVERY record in the batch — batch size > 1 means multiple
            # SQS messages arrived together. Previously only Records[0] was handled
            # and the rest were silently dropped. Now each gets its own SFN execution.
            results      = []
            failed_items = []   # partial batch failure support

            for record in event["Records"]:
                msg_id = record.get("messageId", "unknown")
                try:
                    body      = json.loads(record["body"])
                    s3_record = body["Records"][0]
                    key       = _unquote_key(s3_record["s3"]["object"]["key"])
                    print(f"SQS batch item {msg_id} — starting Step Functions for: {key}")
                    result = _start_step_functions(key)
                    results.append(result)
                except Exception as e:
                    print(f"SQS batch item {msg_id} failed: {e}")
                    # Record the failure (covers unroutable keys — e.g. missing
                    # clients/<id>/pdf/ prefix — as well as sfn start failures)
                    try:
                        _write_failure_marker(key, str(e))
                    except Exception:
                        pass
                    # Report partial batch failure so only THIS message goes to DLQ
                    # Other messages in the batch are not re-processed
                    failed_items.append({"itemIdentifier": msg_id})

            response = {"batchResults": results}
            if failed_items:
                # SQS partial batch failure response — only failed messages retry
                response["batchItemFailures"] = failed_items
            return response

        # ── Legacy direct S3 trigger ──────────────────────────────────
        if first_record.get("eventSource") == "aws:s3" or "s3" in first_record:
            key = _unquote_key(first_record["s3"]["object"]["key"])
            print(f"Direct S3 trigger — starting Step Functions for: {key}")
            try:
                return _start_step_functions(key)
            except Exception as e:
                print(f"CRITICAL: could not start pipeline for {key}: {e}")
                _write_failure_marker(key, str(e))
                raise

    # ── Fallback: process latest PDF across ALL onboarded clients ────
    # (manual test entry point only — SQS/S3 triggers above cover real uploads)
    latest = None
    paginator = s3.get_paginator("list_objects_v2")
    for client_id in list_client_ids():
        for page in paginator.paginate(Bucket=BUCKET, Prefix=input_pdf_folder(client_id)):
            for obj in page.get("Contents", []):
                if obj["Key"].lower().endswith(".pdf"):
                    if latest is None or obj["LastModified"] > latest["LastModified"]:
                        latest = obj
    if latest:
        return _start_step_functions(latest["Key"])

    return {"mode": "NO_VALID_EVENT", "message": "No PDF found in any client's pdf/ folder."}
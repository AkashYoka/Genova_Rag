"""Score new three-PDF PV battery vs prior single-session runs."""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = json.loads((HERE / "_e2e_three_pdfs_pv_battery.json").read_text(encoding="utf-8"))
OLD_PATHS = {
    "xolair": HERE / "_e2e_xolair_pv_extended_single_session.json",
    "lucentis": HERE / "_e2e_lucentis_pv_single_session.json",
    "tecentriq_hybreza": HERE / "_e2e_tecentriq_hybreza_pv_single_session.json",
}


def ans_of(row: dict) -> str:
    return row.get("answer") or row.get("ans_head") or ""


def flags_of(row: dict) -> dict:
    if isinstance(row.get("flags"), dict) and "insuff_prefix" in row["flags"]:
        base = dict(row["flags"])
    else:
        low = ans_of(row).lower()
        base = {
            "insuff_prefix": low.lstrip().startswith("insufficient table evidence"),
            "bad_banner": "authoritative verified table value" in low,
            "not_concordant": bool(
                re.search(
                    r"not label[- ]concordant|does not match|discrepan|inappropriate|"
                    r"medication error|under[- ]?dos|over[- ]?dos|incorrect",
                    low,
                )
            ),
            "off_label": bool(re.search(r"off[- ]?label|not indicated|not an approved", low)),
            "matches": bool(
                re.search(
                    r"label[- ]concordant|matches the label|matches the approved|recommended",
                    low,
                )
            )
            and not bool(re.search(r"not label[- ]concordant|does not match", low)),
        }
    low = ans_of(row).lower()
    base["clinical_voice"] = bool(
        re.search(r"looking at|i'd flag|from the label|clinical takeaway", low)
    )
    base["classifies_pv"] = bool(
        re.search(r"adverse event|special situation|off[- ]?label|medication error", low)
    )
    return base


def summarize(label: str, rows: list[dict]) -> None:
    fs = [flags_of(r) for r in rows]
    print(
        f"  {label}: n={len(rows)} "
        f"insuff={sum(f['insuff_prefix'] for f in fs)} "
        f"banner={sum(f['bad_banner'] for f in fs)} "
        f"clinical={sum(f['clinical_voice'] for f in fs)} "
        f"pv_terms={sum(f['classifies_pv'] for f in fs)} "
        f"issue={sum(f.get('not_concordant', False) for f in fs)} "
        f"off={sum(f.get('off_label', False) for f in fs)}"
    )


def main() -> None:
    for name, path in OLD_PATHS.items():
        old = json.loads(path.read_text(encoding="utf-8"))
        nr = NEW[name]["results"]
        orows = old["results"]
        print(f"==== {name}")
        print("  session", NEW[name]["session_id"])
        summarize("NEW", nr)
        summarize("OLD", orows)
        for r in nr:
            f = flags_of(r)
            tags = []
            if f["insuff_prefix"]:
                tags.append("INSUFF")
            if f["bad_banner"]:
                tags.append("BANNER")
            if not f["clinical_voice"]:
                tags.append("flat")
            tag = ("[" + ",".join(tags) + "] ") if tags else ""
            print(f"  Q{r['i']:02d} {tag}{ans_of(r)[:130].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()

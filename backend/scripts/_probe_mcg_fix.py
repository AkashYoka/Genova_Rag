from app.rag.llm.agent import _reconcile_answer_with_verified_cell, _harmonize_verified_units
import httpx

for ans in [
    "recommended dose is **300 mcg** every 2 weeks",
    "recommends a dose of **225 mcg** every 4 weeks",
    "labeled dose is **300 µg** every 4 weeks",
]:
    num = "300" if "300" in ans else "225"
    print("harm", _harmonize_verified_units(ans, num, "mg")[:90])
    sources = [
        {
            "verified_cell": {
                "value": num,
                "row_headers": [">300-400"],
                "col_headers": [">70-90 kg"],
                "page": 5,
                "status": "matched",
                "evidence": {"unit": "mg"},
            }
        }
    ]
    print("rec ", _reconcile_answer_with_verified_cell(ans, sources)[:120])
    print()

c = httpx.Client(base_url="http://127.0.0.1:7860", timeout=180)
tok = c.post(
    "/api/v1/auth/login",
    json={"email": "dev@genova.ai", "password": "localdev123"},
).json()["access_token"]
h = {"Authorization": f"Bearer {tok}"}
c.post("/api/v1/chat/cache/clear", headers=h)
q = (
    "According to Table 1 for patients 12 years and older with asthma, what XOLAIR "
    "dose and frequency apply for pretreatment IgE 350 IU/mL and body weight 78 kg?"
)
r = c.post(
    "/api/v1/chat/ask",
    headers=h,
    json={
        "question": q,
        "document_id": "43ae024f-3c0d-40f5-aa49-83e9f3682641",
        "llm_provider": "bedrock",
    },
)
print("LIVE", (r.json().get("answer") or "")[:350].replace("\n", " | "))

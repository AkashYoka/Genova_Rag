import json
from app.database import SessionLocal
from app.models import DocumentTable, DocumentChunk

DOC = "43ae024f-3c0d-40f5-aa49-83e9f3682641"
db = SessionLocal()
tables = db.query(DocumentTable).filter(DocumentTable.document_id == DOC).all()
print("tables", len(tables))
for i, t in enumerate(tables):
    headers = (
        t.headers_json
        if isinstance(t.headers_json, list)
        else json.loads(t.headers_json or "[]")
    )
    rows = (
        t.rows_json if isinstance(t.rows_json, list) else json.loads(t.rows_json or "[]")
    )
    header_rows = (
        t.header_rows_json
        if isinstance(t.header_rows_json, list)
        else json.loads(t.header_rows_json or "[]")
        if t.header_rows_json
        else []
    )
    print(
        f"\n=== TABLE {i} id={t.table_id} page={t.page_start}-{t.page_end} "
        f"rows={len(rows)} cols={len(headers)} caption={t.caption!r} ==="
    )
    print("HEADERS:")
    for h in headers:
        print(" -", repr(h))
    if header_rows:
        print("HEADER_ROWS:")
        for hr in header_rows:
            print(" ", hr)
    for ri, r in enumerate(rows):
        print(f"R{ri}", r)

print("\nTABLE-LIKE CHUNKS")
chunks = (
    db.query(DocumentChunk)
    .filter(DocumentChunk.document_id == DOC)
    .order_by(DocumentChunk.page_start.asc())
    .all()
)
for c in chunks:
    text = c.content or ""
    if "Every 4 weeks" in text or "Insufficient Data to Recommend" in text:
        print("---", getattr(c, "id", None), "page", c.page_start, getattr(c, "chunk_type", None))
        print(text[:3000])
        print()
db.close()

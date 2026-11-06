from pathlib import Path

p = Path(__file__).resolve().parents[1] / "tests" / "test_s3_storage.py"
text = p.read_text(encoding="utf-8")
# Restore NoteBase expectations if they were stripped.
if 'S3_BASE_PREFIX", "")' in text or '== "user-1/pdfs/' in text:
    text = text.replace(
        'monkeypatch.setattr(s3_storage.settings, "S3_BASE_PREFIX", "")',
        'monkeypatch.setattr(s3_storage.settings, "S3_BASE_PREFIX", "NoteBase")',
    )
    replacements = {
        '== "user-1/pdfs/abc.pdf"': '== "NoteBase/user-1/pdfs/abc.pdf"',
        '== "user-2/pdfs/abc.pdf"': '== "NoteBase/user-2/pdfs/abc.pdf"',
        '== "user-1/pdfs/secret.pdf"': '== "NoteBase/user-1/pdfs/secret.pdf"',
        '== "user-1/sticky_notes/notes.json"': '== "NoteBase/user-1/sticky_notes/notes.json"',
        '["user-1/sticky_notes/notes.json"]': '["NoteBase/user-1/sticky_notes/notes.json"]',
        '"user-1/sticky_notes/notes.json"': '"NoteBase/user-1/sticky_notes/notes.json"',
        '"user-2/sticky_notes/notes.json"': '"NoteBase/user-2/sticky_notes/notes.json"',
        '== "shared/exports/"': '== "NoteBase/shared/exports/"',
        '== "user-9/pdfs/report.pdf"': '== "NoteBase/user-9/pdfs/report.pdf"',
        '"user-9/pdfs/report.pdf"': '"NoteBase/user-9/pdfs/report.pdf"',
        's3_key="user/pdfs/stored.pdf"': 's3_key="NoteBase/user/pdfs/stored.pdf"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Fix double-prefix if any
    text = text.replace("NoteBase/NoteBase/", "NoteBase/")
    p.write_text(text, encoding="utf-8")
print("NoteBase count:", text.count("NoteBase"))

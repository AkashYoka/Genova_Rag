"""Tests for the S3-backed sticky-notes API (/api/v1/notes)."""
import pytest

from app.auth import create_access_token
from app.services import s3_storage


@pytest.fixture()
def fake_s3(monkeypatch):
    """In-memory stand-in for the S3 notes object."""
    state: dict = {
        "document": None,
        "updated_at": None,
        "load_user": None,
        "save_user": None,
        "saved": [],
        "deleted": [],
        "configured": True,
    }

    def fake_load(user_id):
        state["load_user"] = user_id
        return state["document"], state["updated_at"]

    def fake_save(user_id, document):
        state["save_user"] = user_id
        state["saved"].append(document)
        state["document"] = document
        state["updated_at"] = "2026-01-01T00:00:00+00:00"
        return state["updated_at"]

    def fake_delete(user_id):
        state["deleted"].append(user_id)
        state["document"] = None
        return True

    monkeypatch.setattr(s3_storage, "is_configured", lambda: state["configured"])
    monkeypatch.setattr(s3_storage, "load_notes", fake_load)
    monkeypatch.setattr(s3_storage, "save_notes", fake_save)
    monkeypatch.setattr(s3_storage, "delete_notes", fake_delete)
    return state


DOCUMENT = {
    "main": [
        {"name": "heading", "attributes": {"level": 2}, "children": [{"data": "Summary"}]},
        {
            "name": "paragraph",
            "children": [{"data": "Finding: "}, {"data": "critical", "bold": True}],
        },
        {
            "name": "list",
            "children": [
                {
                    "name": "listItem",
                    "children": [{"name": "paragraph", "children": [{"data": "Item"}]}],
                }
            ],
        },
    ]
}


# ── load ─────────────────────────────────────────────────────────────────────

def test_get_notes_requires_auth(client):
    assert client.get("/api/v1/notes").status_code in (401, 403)


def test_get_notes_returns_saved_json(client, auth_headers, user, fake_s3):
    fake_s3["document"] = DOCUMENT
    fake_s3["updated_at"] = "2026-01-01T00:00:00+00:00"

    response = client.get("/api/v1/notes", headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["notes"] == DOCUMENT
    assert payload["exists"] is True
    assert payload["storage"] == "s3"
    assert fake_s3["load_user"] == str(user.id)


def test_get_notes_without_file_returns_empty(client, auth_headers, user, fake_s3):
    response = client.get("/api/v1/notes", headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["notes"] is None
    assert payload["exists"] is False


def test_get_notes_when_storage_unconfigured(client, auth_headers, fake_s3):
    fake_s3["configured"] = False

    response = client.get("/api/v1/notes", headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["storage"] == "unavailable"
    assert payload["notes"] is None


# ── save ─────────────────────────────────────────────────────────────────────

def test_put_notes_persists_for_authenticated_user(client, auth_headers, user, fake_s3):
    response = client.put("/api/v1/notes", headers=auth_headers, json={"notes": DOCUMENT})

    assert response.status_code == 200, response.text
    assert response.json()["saved"] is True
    assert fake_s3["save_user"] == str(user.id)
    assert fake_s3["saved"] == [DOCUMENT]


def test_put_notes_isolated_per_user(client, other_user, fake_s3):
    headers = {"Authorization": f"Bearer {create_access_token(other_user.id)}"}

    response = client.put("/api/v1/notes", headers=headers, json={"notes": DOCUMENT})

    assert response.status_code == 200
    assert fake_s3["save_user"] == str(other_user.id)
    assert fake_s3["save_user"] != ""


def test_put_notes_accepts_empty_document(client, auth_headers, fake_s3):
    response = client.put("/api/v1/notes", headers=auth_headers, json={"notes": {"main": []}})

    assert response.status_code == 200
    assert fake_s3["saved"] == [{"main": []}]


def test_put_notes_accepts_structured_model_document(client, auth_headers, fake_s3):
    document = {
        "format": "ckeditor5-model",
        "version": 1,
        "main": DOCUMENT["main"],
    }

    response = client.put("/api/v1/notes", headers=auth_headers, json={"notes": document})

    assert response.status_code == 200, response.text
    assert fake_s3["saved"] == [document]


def test_put_notes_accepts_legacy_html_string(client, auth_headers, fake_s3):
    document = {"main": "<p>Carried over from the old editor</p>"}

    response = client.put("/api/v1/notes", headers=auth_headers, json={"notes": document})

    assert response.status_code == 200, response.text
    assert fake_s3["saved"] == [document]


def test_put_notes_rejects_non_collection_main(client, auth_headers, fake_s3):
    response = client.put("/api/v1/notes", headers=auth_headers, json={"notes": {"main": 42}})

    assert response.status_code == 400
    assert fake_s3["saved"] == []


def test_put_notes_rejects_non_list_content(client, auth_headers, fake_s3):
    response = client.put(
        "/api/v1/notes", headers=auth_headers, json={"notes": {"content": "nope"}}
    )

    assert response.status_code == 400
    assert fake_s3["saved"] == []


def test_put_notes_rejects_non_object(client, auth_headers, fake_s3):
    response = client.put("/api/v1/notes", headers=auth_headers, json={"notes": ["not", "an", "object"]})

    assert response.status_code == 422


def test_put_notes_rejects_invalid_blocks(client, auth_headers, fake_s3):
    response = client.put("/api/v1/notes", headers=auth_headers, json={"notes": {"blocks": "nope"}})

    assert response.status_code == 400
    assert fake_s3["saved"] == []


def test_put_notes_rejects_oversized_document(client, auth_headers, fake_s3):
    oversized = {"main": [{"name": "paragraph", "children": [{"data": "x" * (2 * 1024 * 1024)}]}]}

    response = client.put("/api/v1/notes", headers=auth_headers, json={"notes": oversized})

    assert response.status_code == 400
    assert "limit" in response.text.lower()
    assert fake_s3["saved"] == []


def test_put_notes_requires_auth(client, fake_s3):
    assert client.put("/api/v1/notes", json={"notes": DOCUMENT}).status_code in (401, 403)


def test_put_notes_when_storage_unconfigured(client, auth_headers, fake_s3):
    fake_s3["configured"] = False

    response = client.put("/api/v1/notes", headers=auth_headers, json={"notes": DOCUMENT})

    assert response.status_code == 503
    assert fake_s3["saved"] == []


def test_put_notes_surfaces_s3_failure(client, auth_headers, monkeypatch):
    monkeypatch.setattr(s3_storage, "is_configured", lambda: True)

    def boom(user_id, document):
        raise s3_storage.S3StorageError("access denied")

    monkeypatch.setattr(s3_storage, "save_notes", boom)

    response = client.put("/api/v1/notes", headers=auth_headers, json={"notes": DOCUMENT})

    assert response.status_code == 502


def test_get_notes_surfaces_s3_failure(client, auth_headers, monkeypatch):
    monkeypatch.setattr(s3_storage, "is_configured", lambda: True)

    def boom(user_id):
        raise s3_storage.S3StorageError("network down")

    monkeypatch.setattr(s3_storage, "load_notes", boom)

    response = client.get("/api/v1/notes", headers=auth_headers)

    assert response.status_code == 502


# ── clear ────────────────────────────────────────────────────────────────────

def test_delete_notes_clears_user_object(client, auth_headers, user, fake_s3):
    response = client.delete("/api/v1/notes", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert fake_s3["deleted"] == [str(user.id)]


# ── multiple notes (collection envelope) ─────────────────────────────────────

COLLECTION = {
    "format": "genova-sticky-notes",
    "version": 1,
    "activeId": "note-2",
    "notes": [
        {
            "id": "note-1",
            "document": {
                "format": "ckeditor5-model",
                "version": 1,
                "main": [{"name": "paragraph", "children": [{"data": "One"}]}],
            },
            "html": "<p>One</p>",
        },
        {
            "id": "note-2",
            "document": None,
            "html": "<p>Two</p>",
        },
    ],
}


def test_put_notes_accepts_multi_note_collection(client, auth_headers, fake_s3):
    response = client.put("/api/v1/notes", headers=auth_headers, json={"notes": COLLECTION})

    assert response.status_code == 200, response.text
    assert fake_s3["saved"] == [COLLECTION]


def test_put_notes_accepts_empty_collection(client, auth_headers, fake_s3):
    empty = {
        "format": "genova-sticky-notes",
        "version": 1,
        "activeId": None,
        "notes": [],
    }

    response = client.put("/api/v1/notes", headers=auth_headers, json={"notes": empty})

    assert response.status_code == 200, response.text
    assert fake_s3["saved"] == [empty]


def test_put_notes_rejects_collection_notes_not_list(client, auth_headers, fake_s3):
    invalid = {"format": "genova-sticky-notes", "notes": "nope"}

    response = client.put("/api/v1/notes", headers=auth_headers, json={"notes": invalid})

    assert response.status_code == 400
    assert fake_s3["saved"] == []


def test_put_notes_rejects_collection_note_not_object(client, auth_headers, fake_s3):
    invalid = {"format": "genova-sticky-notes", "notes": ["nope"]}

    response = client.put("/api/v1/notes", headers=auth_headers, json={"notes": invalid})

    assert response.status_code == 400
    assert fake_s3["saved"] == []


def test_put_notes_rejects_collection_document_not_object(client, auth_headers, fake_s3):
    invalid = {
        "format": "genova-sticky-notes",
        "notes": [{"id": "note-1", "document": "nope"}],
    }

    response = client.put("/api/v1/notes", headers=auth_headers, json={"notes": invalid})

    assert response.status_code == 400
    assert fake_s3["saved"] == []


def test_put_notes_rejects_collection_html_not_string(client, auth_headers, fake_s3):
    invalid = {
        "format": "genova-sticky-notes",
        "notes": [{"id": "note-1", "html": ["nope"]}],
    }

    response = client.put("/api/v1/notes", headers=auth_headers, json={"notes": invalid})

    assert response.status_code == 400
    assert fake_s3["saved"] == []


def test_put_notes_rejects_collection_active_id_not_string(client, auth_headers, fake_s3):
    invalid = {
        "format": "genova-sticky-notes",
        "activeId": 3,
        "notes": [],
    }

    response = client.put("/api/v1/notes", headers=auth_headers, json={"notes": invalid})

    assert response.status_code == 400
    assert fake_s3["saved"] == []


def test_put_notes_rejects_collection_note_id_not_string(client, auth_headers, fake_s3):
    invalid = {
        "format": "genova-sticky-notes",
        "notes": [{"id": 1, "html": "<p>One</p>"}],
    }

    response = client.put("/api/v1/notes", headers=auth_headers, json={"notes": invalid})

    assert response.status_code == 400
    assert fake_s3["saved"] == []


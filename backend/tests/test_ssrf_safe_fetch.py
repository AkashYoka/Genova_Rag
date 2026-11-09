"""Deterministic SSRF / DNS-rebinding regression tests (mocked resolver)."""
from __future__ import annotations

import ipaddress
from typing import List

import pytest

from app.security.ssrf import (
    ResolvedEndpoint,
    SafeFetchError,
    fetch_url_safe,
    validate_public_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost/secrets",
        "http://[::1]/",
        "http://10.0.0.5/x",
        "http://192.168.1.1/x",
        "http://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
        "http://user:pass@example.com/x",
    ],
)
def test_validate_public_url_blocks_dangerous_targets(url):
    with pytest.raises(SafeFetchError):
        validate_public_url(url)


def test_fetch_rebinding_rejected_via_resolver_injection():
    """Injected resolver returns a loopback IP → fetch must refuse."""
    import socket

    def resolver(hostname, port):
        return [
            ResolvedEndpoint(
                hostname=hostname, port=port, ip="127.0.0.1", family=socket.AF_INET
            )
        ]

    with pytest.raises(SafeFetchError):
        fetch_url_safe("http://evil.example/path", resolver=resolver)


def test_urlupload_endpoint_uses_safe_validation(client, auth_headers):
    response = client.post(
        "/api/v1/documents/urlupload",
        headers=auth_headers,
        json={"url": "http://127.0.0.1/admin"},
    )
    assert response.status_code in {400, 422}

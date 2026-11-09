"""SSRF-safe URL validation and IP-pinned HTTP(S) fetch.

Validates every A/AAAA address, connects to a chosen public IP while preserving
the original hostname for Host / SNI / TLS verification, and re-validates each
redirect hop. This closes classic resolve-then-fetch DNS-rebinding gaps.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
import ssl
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPSConnection
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_TIMEOUT = 15.0
DEFAULT_MAX_REDIRECTS = 5
ALLOWED_SCHEMES = {"http", "https"}
BLOCKED_SCHEMES = {"file", "ftp", "gopher", "dict", "smb", "ldap", "data", "javascript"}


class SafeFetchError(ValueError):
    """Raised when a URL fails SSRF policy or cannot be fetched safely."""


@dataclass(frozen=True)
class ResolvedEndpoint:
    hostname: str
    port: int
    ip: str
    family: int


def _is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or (isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped and _is_blocked_ip(ip.ipv4_mapped))
    )


def validate_public_url(url: str) -> Tuple[str, str, int, List[ResolvedEndpoint]]:
    """Parse + resolve URL; return (scheme, hostname, port, endpoints)."""
    parsed = urlparse((url or "").strip())
    scheme = (parsed.scheme or "").lower()
    if scheme in BLOCKED_SCHEMES:
        raise SafeFetchError(f"URL scheme '{scheme}' is not allowed")
    if scheme not in ALLOWED_SCHEMES:
        raise SafeFetchError("Only http and https URLs are allowed")
    if parsed.username or parsed.password:
        raise SafeFetchError("URLs with embedded credentials are not allowed")
    hostname = parsed.hostname
    if not hostname:
        raise SafeFetchError("Invalid URL host")
    # Literal IPs must themselves be public.
    try:
        literal = ipaddress.ip_address(hostname)
        if _is_blocked_ip(literal):
            raise SafeFetchError("Internal or private URLs are not allowed")
    except ValueError:
        pass

    port = parsed.port or (443 if scheme == "https" else 80)
    endpoints = resolve_public_endpoints(hostname, port)
    if not endpoints:
        raise SafeFetchError("Could not resolve URL host to a public address")
    return scheme, hostname, port, endpoints


def resolve_public_endpoints(hostname: str, port: int) -> List[ResolvedEndpoint]:
    try:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SafeFetchError("Could not resolve URL host") from exc

    endpoints: List[ResolvedEndpoint] = []
    seen = set()
    for family, _type, _proto, _canon, sockaddr in infos:
        ip = sockaddr[0]
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if _is_blocked_ip(addr):
            raise SafeFetchError("Internal or private URLs are not allowed")
        key = (family, ip)
        if key in seen:
            continue
        seen.add(key)
        endpoints.append(ResolvedEndpoint(hostname=hostname, port=port, ip=ip, family=family))
    return endpoints


class _PinnedHTTPSConnection(HTTPSConnection):
    """HTTPSConnection that dials a pre-selected IP with SNI for hostname."""

    def __init__(self, hostname: str, pinned_ip: str, port: int, timeout: float):
        context = ssl.create_default_context()
        super().__init__(hostname, port=port, timeout=timeout, context=context)
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        self.sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)
        server_hostname = self.host
        self.sock = self._context.wrap_socket(self.sock, server_hostname=server_hostname)


class _PinnedHTTPConnection(HTTPConnection):
    def __init__(self, hostname: str, pinned_ip: str, port: int, timeout: float):
        super().__init__(hostname, port=port, timeout=timeout)
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        self.sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)


def _request_once(
    *,
    scheme: str,
    hostname: str,
    port: int,
    pinned_ip: str,
    method: str,
    path: str,
    timeout: float,
    max_bytes: int,
) -> Tuple[int, dict, bytes, Optional[str]]:
    if scheme == "https":
        conn: HTTPConnection = _PinnedHTTPSConnection(hostname, pinned_ip, port, timeout)
    else:
        conn = _PinnedHTTPConnection(hostname, pinned_ip, port, timeout)
    try:
        conn.request(
            method,
            path or "/",
            headers={
                "Host": hostname if port in (80, 443) else f"{hostname}:{port}",
                "User-Agent": "GenovaSafeFetch/1.0",
                "Accept": "text/html,text/plain,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        resp = conn.getresponse()
        # Bound read to max_bytes + 1 to detect overflow.
        body = resp.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise SafeFetchError("Response exceeds maximum allowed size")
        headers = {k.lower(): v for k, v in resp.getheaders()}
        location = headers.get("location")
        return resp.status, headers, body, location
    finally:
        try:
            conn.close()
        except Exception:
            pass


def fetch_url_safe(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    resolver=None,
) -> Tuple[str, bytes, str]:
    """Fetch URL with IP pinning. Returns (final_url, body_bytes, content_type).

    ``resolver`` may be injected in tests: ``(hostname, port) -> list[ResolvedEndpoint]``.
    """
    current = url
    for _ in range(max_redirects + 1):
        scheme, hostname, port, endpoints = validate_public_url(current)
        if resolver is not None:
            endpoints = list(resolver(hostname, port))
            if not endpoints:
                raise SafeFetchError("Could not resolve URL host to a public address")
            for ep in endpoints:
                if _is_blocked_ip(ipaddress.ip_address(ep.ip)):
                    raise SafeFetchError("Internal or private URLs are not allowed")

        # Prefer IPv4 for broader compatibility, else first endpoint.
        pinned = next((e for e in endpoints if e.family == socket.AF_INET), endpoints[0])
        parsed = urlparse(current)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        status, headers, body, location = _request_once(
            scheme=scheme,
            hostname=hostname,
            port=port,
            pinned_ip=pinned.ip,
            method="GET",
            path=path,
            timeout=timeout,
            max_bytes=max_bytes,
        )

        if status in {301, 302, 303, 307, 308}:
            if not location:
                raise SafeFetchError("Redirect without Location header")
            current = urljoin(current, location)
            # Re-validate next hop (including DNS) on next loop iteration.
            continue

        if status >= 400:
            raise SafeFetchError(f"Upstream returned HTTP {status}")

        content_type = headers.get("content-type", "application/octet-stream").split(";")[0].strip()
        allowed_types = {
            "text/html",
            "text/plain",
            "application/xhtml+xml",
            "application/xml",
            "text/xml",
            "application/json",
        }
        if content_type and content_type not in allowed_types and not content_type.startswith("text/"):
            raise SafeFetchError(f"Content type '{content_type}' is not allowed")
        return current, body, content_type or "application/octet-stream"

    raise SafeFetchError("Too many redirects")

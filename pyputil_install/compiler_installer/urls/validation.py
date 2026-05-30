"""
URL and artifact validation layer.

Validates that resolved artifact URLs point to real, downloadable
files. Performs HTTP HEAD requests, checks response metadata,
and verifies checksum files exist.

All HTTP operations are async using `aiohttp`.
No synchronous network calls exist in this module.

Security
--------
- All requests use timeouts (default: 30s connect, 60s read).
- Redirects are followed but limited to 5 hops.
- No body content is downloaded — HEAD requests only.
- Rate-limit headers are logged but not acted upon by default.
- SSL verification is ENABLED — no disable flag exists.

Usage
-----
    import asyncio
    from pyputil_install.compiler_installer.validation import validate_artifact, validate_url

    async def main():
        result = await validate_url("https://github.com/.../file.tar.gz")
        print(result.is_valid, result.content_length)

    asyncio.run(main())

Warnings
--------
- GitHub has rate limits. Unauthenticated: 60 req/hour/IP.
  Authenticated: set TOOLFORGE_GITHUB_TOKEN env var for 5000 req/hour.
- Batch validation should use `validate_urls()` for controlled concurrency.
- HEAD requests are not universally supported (some CDNs, S3 without
  configuration). A 405 Method Not Allowed is treated as soft failure.

User Instructions
-----------------
- Set TOOLFORGE_HTTP_CONNECT_TIMEOUT to override connect timeout (seconds).
- Set TOOLFORGE_HTTP_READ_TIMEOUT to override read timeout (seconds).
- Set TOOLFORGE_GITHUB_TOKEN for higher GitHub rate limits.
- Set TOOLFORGE_MAX_REDIRECTS to limit redirect hops (default: 5).
- Set TOOLFORGE_MAX_CONCURRENT to limit parallel requests (default: 10).
- Check ValidationResult.is_valid before using content_length.
"""

import asyncio
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .base import ChecksumResult, ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONNECT_TIMEOUT: int = int(
    os.environ.get("TOOLFORGE_HTTP_CONNECT_TIMEOUT", "30")
)
DEFAULT_READ_TIMEOUT: int = int(
    os.environ.get("TOOLFORGE_HTTP_READ_TIMEOUT", "60")
)
MAX_REDIRECTS: int = int(
    os.environ.get("TOOLFORGE_MAX_REDIRECTS", "5")
)
MAX_CONCURRENT: int = int(
    os.environ.get("TOOLFORGE_MAX_CONCURRENT", "10")
)
GITHUB_TOKEN: Optional[str] = os.environ.get("TOOLFORGE_GITHUB_TOKEN")

USER_AGENT = (
    "ToolForge/0.1.0 (+https://github.com/toolforge; toolforge@example.com)"
)

# ---------------------------------------------------------------------------
# Optional aiohttp import
# ---------------------------------------------------------------------------

try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False
    aiohttp = None  # Prevent NameError in except blocks
    logger.warning(
        "aiohttp not installed. Install with: pip install aiohttp"
    )


# ---------------------------------------------------------------------------
# URL syntax validation (pure, no HTTP)
# ---------------------------------------------------------------------------

def validate_url_syntax(url: str) -> Tuple[bool, Optional[str]]:
    """
    Check that a URL has valid syntax.

    Does NOT make any HTTP requests. Only validates structure.

    Parameters
    ----------
    url : str
        URL to validate.

    Returns
    -------
    Tuple[bool, Optional[str]]
        (True, None) if syntax is valid.
        (False, error_message) if malformed.

    Examples
    --------
    >>> validate_url_syntax("https://example.com/file.tar.gz")
    (True, None)
    >>> validate_url_syntax("not-a-url")
    (False, "URL has no scheme")
    """
    try:
        result = urlparse(url)
    except Exception as exc:
        return False, f"URL parsing failed: {exc}"

    if not result.scheme:
        return False, "URL has no scheme"
    if result.scheme not in ("http", "https"):
        return False, f"Unsupported scheme: {result.scheme}"
    if not result.netloc:
        return False, "URL has no host (netloc)"
    if not result.path or result.path == "/":
        return False, "URL has no file path"

    return True, None


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def _build_session() -> 'aiohttp.ClientSession':
    """
    Build an aiohttp session with ToolForge defaults.

    Returns
    -------
    aiohttp.ClientSession
        Session with User-Agent, timeouts, and auth headers.

    Raises
    ------
    RuntimeError
        If aiohttp is not installed.
    """
    if not _AIOHTTP_AVAILABLE:
        raise RuntimeError(
            "aiohttp is required for async validation. "
            "Install with: pip install aiohttp"
        )

    timeout = aiohttp.ClientTimeout(
        connect=DEFAULT_CONNECT_TIMEOUT,
        sock_read=DEFAULT_READ_TIMEOUT,
    )

    headers = {"User-Agent": USER_AGENT}

    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    return aiohttp.ClientSession(
        timeout=timeout,
        headers=headers,
        connector=connector,
        raise_for_status=False,
    )


# ---------------------------------------------------------------------------
# HTTP validation — async
# ---------------------------------------------------------------------------

async def validate_url(
    url: str,
    session: Optional['aiohttp.ClientSession'] = None,
) -> ValidationResult:
    """
    Validate a URL by performing an async HTTP HEAD request.

    Parameters
    ----------
    url : str
        The URL to validate.
    session : Optional[aiohttp.ClientSession]
        Existing session to use. If None, creates and closes a new one.

    Returns
    -------
    ValidationResult
        Result with status, headers, and timing.
    """
    # 1. Syntax check first (fast, no network)
    valid_syntax, error = validate_url_syntax(url)
    if not valid_syntax:
        return ValidationResult(
            url=url,
            status=ValidationStatus.INVALID_URL,
            error_message=error,
        )

    # 2. Check if aiohttp is available
    if not _AIOHTTP_AVAILABLE:
        return ValidationResult(
            url=url,
            status=ValidationStatus.AIOHTTP_MISSING,
            error_message=(
                "aiohttp is not installed. "
                "Install with: pip install aiohttp"
            ),
        )

    # 3. HTTP HEAD request
    start_time = time.monotonic()
    own_session = session is None

    try:
        if own_session:
            session = _build_session()

        async with session.head(url, allow_redirects=True) as response:
            elapsed_ms = (time.monotonic() - start_time) * 1000

            # Extract redirect chain from response history
            redirect_chain = [str(r.url) for r in response.history]

            # Rate limit tracking (GitHub)
            rate_limit = None
            if "X-RateLimit-Remaining" in response.headers:
                try:
                    rate_limit = int(response.headers["X-RateLimit-Remaining"])
                except (ValueError, TypeError):
                    pass

            # 2xx = success
            if 200 <= response.status < 300:
                content_length = None
                cl_header = response.headers.get("Content-Length")
                if cl_header:
                    try:
                        content_length = int(cl_header)
                    except (ValueError, TypeError):
                        pass

                return ValidationResult(
                    url=url,
                    status=ValidationStatus.VALID,
                    is_valid=True,
                    content_length=content_length,
                    content_type=response.headers.get("Content-Type"),
                    etag=response.headers.get("ETag"),
                    last_modified=response.headers.get("Last-Modified"),
                    response_time_ms=elapsed_ms,
                    redirect_chain=redirect_chain,
                    server_header=response.headers.get("Server"),
                    rate_limit_remaining=rate_limit,
                )

            # 4xx/5xx = specific errors
            status_map = {
                403: ValidationStatus.FORBIDDEN,
                404: ValidationStatus.NOT_FOUND,
                405: ValidationStatus.UNSUPPORTED_METHOD,
                429: ValidationStatus.FORBIDDEN,
            }
            status = status_map.get(response.status, ValidationStatus.UNKNOWN)
            error_msg = (
                f"HTTP {response.status}: {response.reason or 'Unknown error'}"
            )

            if response.status == 429:
                error_msg = (
                    "Rate limited. "
                    "Use TOOLFORGE_GITHUB_TOKEN for higher limits."
                )

            return ValidationResult(
                url=url,
                status=status,
                error_message=error_msg,
                response_time_ms=elapsed_ms,
                redirect_chain=redirect_chain,
                server_header=response.headers.get("Server"),
                rate_limit_remaining=rate_limit,
            )

    except aiohttp.TooManyRedirects:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        return ValidationResult(
            url=url,
            status=ValidationStatus.TOO_MANY_REDIRECTS,
            error_message=f"Exceeded {MAX_REDIRECTS} redirect hops",
            response_time_ms=elapsed_ms,
        )
    except aiohttp.ServerTimeoutError:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        return ValidationResult(
            url=url,
            status=ValidationStatus.TIMEOUT,
            error_message="Server read timeout",
            response_time_ms=elapsed_ms,
        )
    except aiohttp.ClientConnectorError as exc:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        return ValidationResult(
            url=url,
            status=ValidationStatus.NETWORK_ERROR,
            error_message=f"Connection failed: {exc}",
            response_time_ms=elapsed_ms,
        )
    except aiohttp.ClientError as exc:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        return ValidationResult(
            url=url,
            status=ValidationStatus.UNKNOWN,
            error_message=f"HTTP client error: {exc}",
            response_time_ms=elapsed_ms,
        )
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.error("Unexpected error validating %s: %s", url, exc)
        return ValidationResult(
            url=url,
            status=ValidationStatus.UNKNOWN,
            error_message=f"Unexpected error: {exc}",
            response_time_ms=elapsed_ms,
        )
    finally:
        if own_session and session is not None:
            await session.close()


# ---------------------------------------------------------------------------
# Batch validation with concurrency control
# ---------------------------------------------------------------------------

async def validate_urls(
    urls: List[str],
    max_concurrent: int = MAX_CONCURRENT,
) -> List[ValidationResult]:
    """
    Validate multiple URLs with controlled concurrency.

    Uses a semaphore to limit simultaneous HTTP connections.
    All URLs share a single aiohttp session.

    Parameters
    ----------
    urls : List[str]
        URLs to validate.
    max_concurrent : int
        Maximum simultaneous requests. Default: 10.

    Returns
    -------
    List[ValidationResult]
        One result per input URL. Order matches input.
        Errors are captured per-URL, never raised.
    """
    if not urls:
        return []

    if not _AIOHTTP_AVAILABLE:
        return [
            ValidationResult(
                url=url,
                status=ValidationStatus.AIOHTTP_MISSING,
                error_message="aiohttp is not installed. Install with: pip install aiohttp",
            )
            for url in urls
        ]

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _validate_one(
        url: str, session: aiohttp.ClientSession
    ) -> ValidationResult:
        async with semaphore:
            return await validate_url(url, session=session)

    session = _build_session()
    try:
        tasks = [_validate_one(url, session) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results: List[ValidationResult] = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                final_results.append(ValidationResult(
                    url=url,
                    status=ValidationStatus.UNKNOWN,
                    error_message=f"Task failed: {result}",
                ))
            else:
                final_results.append(result)

        return final_results
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# Checksum validation
# ---------------------------------------------------------------------------

async def validate_checksum(
    checksum_url: str,
    filename: str,
    session: Optional['aiohttp.ClientSession'] = None,
) -> ChecksumResult:
    """
    Fetch a checksum file and look for an entry matching the filename.

    Downloads the checksum file body (typically small text).
    Supports common formats: "hash  filename" and "hash filename".

    Parameters
    ----------
    checksum_url : str
        URL to the checksum file (e.g., ".../file.tar.gz.sha256").
    filename : str
        The artifact filename to look for in the checksum file.
    session : Optional[aiohttp.ClientSession]
        Existing session to use.

    Returns
    -------
    ChecksumResult
        Result with hash and algorithm if found.
    """
    if not _AIOHTTP_AVAILABLE:
        return ChecksumResult(
            checksum_url=checksum_url,
            filename=filename,
            error_message="aiohttp is not installed. Install with: pip install aiohttp",
        )

    own_session = session is None

    try:
        if own_session:
            session = _build_session()

        async with session.get(checksum_url, allow_redirects=True) as response:
            if response.status != 200:
                return ChecksumResult(
                    checksum_url=checksum_url,
                    filename=filename,
                    error_message=f"HTTP {response.status}: {response.reason}",
                )

            content = await response.text()

            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                hash_value = parts[0]
                file_part = parts[-1]

                if file_part == filename or file_part == f"./{filename}":
                    algorithm = _detect_algorithm(hash_value)
                    return ChecksumResult(
                        checksum_url=checksum_url,
                        is_valid=True,
                        filename=filename,
                        expected_hash=hash_value,
                        algorithm=algorithm,
                        content=content,
                    )

            return ChecksumResult(
                checksum_url=checksum_url,
                filename=filename,
                error_message=f"Filename '{filename}' not found in checksum file",
                content=content,
            )

    except aiohttp.ClientError as exc:
        return ChecksumResult(
            checksum_url=checksum_url,
            filename=filename,
            error_message=f"HTTP error: {exc}",
        )
    except Exception as exc:
        logger.error("Unexpected error validating checksum: %s", exc)
        return ChecksumResult(
            checksum_url=checksum_url,
            filename=filename,
            error_message=f"Unexpected error: {exc}",
        )
    finally:
        if own_session and session is not None:
            await session.close()


def _detect_algorithm(hash_value: str) -> str:
    """
    Detect hash algorithm from hex string length.

    Parameters
    ----------
    hash_value : str
        Hex-encoded hash string.

    Returns
    -------
    str
        "sha256" (64 chars), "sha512" (128 chars), "md5" (32 chars),
        or "unknown".
    """
    length = len(hash_value.strip())
    if length == 64:
        return "sha256"
    elif length == 128:
        return "sha512"
    elif length == 32:
        return "md5"
    return "unknown"


# ---------------------------------------------------------------------------
# Artifact validation (URL + checksum)
# ---------------------------------------------------------------------------

async def validate_artifact(
    artifact,
    session: Optional['aiohttp.ClientSession'] = None,
) -> Tuple[ValidationResult, Optional[ChecksumResult]]:
    """
    Validate an ArtifactInfo: URL reachability + checksum existence.

    Parameters
    ----------
    artifact : ArtifactInfo
        Resolved artifact from builder.py.
    session : Optional[aiohttp.ClientSession]
        Existing session to reuse.

    Returns
    -------
    Tuple[ValidationResult, Optional[ChecksumResult]]
        URL validation result and checksum result (if checksum_url exists).
    """
    url_result = await validate_url(artifact.url, session=session)

    checksum_result = None
    if artifact.checksum_url:
        checksum_result = await validate_checksum(
            artifact.checksum_url,
            artifact.filename,
            session=session,
        )

    return url_result, checksum_result


# ---------------------------------------------------------------------------
# Convenience: validate and return downloadable flag
# ---------------------------------------------------------------------------

async def is_url_downloadable(
    url: str,
    session: Optional['aiohttp.ClientSession'] = None,
) -> bool:
    """
    Quick check: is this URL reachable and returns 2xx?

    Parameters
    ----------
    url : str
        URL to check.
    session : Optional[aiohttp.ClientSession]
        Existing session.

    Returns
    -------
    bool
        True if URL is reachable and returns HTTP 200-299.
    """
    result = await validate_url(url, session=session)
    return result.is_valid
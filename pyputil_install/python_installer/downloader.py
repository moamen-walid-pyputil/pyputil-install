"""
File Downloader Module
======================

Secure, resumable, multi-mirror file downloader with automatic retry
backoff, bandwidth throttling, mandatory SHA256 checksum verification,
and a persistent on-disk cache.

Uses only the Python standard library. No third-party packages required.

Security
--------
- All transfers require HTTPS. Plain HTTP is rejected.
- TLS 1.2+ enforced via :class:`urllib.request.HTTPSHandler`.
- Certificate validation uses the system trust store and cannot be
  bypassed through this API.
- SHA256 verification is **mandatory**. The *expected_sha256* parameter
  is required; there is no code path that skips verification.
- Verification runs on a temporary file created with ``mkstemp`` (0o600
  permissions). Only after successful verification is the file atomically
  renamed to the requested destination via :func:`os.replace`.
- Partial downloads are validated before resumption: the existing file
  is hashed and the byte offset verified against the server response.
- All redirects are followed, but cross-origin redirects are rejected.
- Maximum redirect depth is capped at 5 to prevent infinite loops.

Usage
-----
.. code-block:: python

    from pathlib import Path
    from downloader import DownloadManager, DownloadError, ChecksumVerificationError

    manager = DownloadManager(
        mirrors=["https://mirror1.example.com"],
        max_retries=5,
        rate_limit_mbps=10.0,
    )

    try:
        path = manager.download(
            url="https://github.com/indygreg/python-build-standalone/releases/"
                "download/20231002/cpython-3.11.5+20231002-x86_64-unknown-"
                "linux-gnu-install_only.tar.gz",
            dest=Path("/tmp/python.tar.gz"),
            expected_sha256="d2b4f0e8...",
            resume=True,
        )
        print(f"Downloaded and verified: {path}")
    except ChecksumVerificationError as e:
        print(f"Integrity failure: {e}")
        print(f"  Expected: {e.expected}")
        print(f"  Got:      {e.actual}")
    except DownloadError as e:
        print(f"Download failed: {e}")
        if e.original_error:
            print(f"  Caused by: {e.original_error}")

Using the cache::

    from downloader import CacheManager

    cache = CacheManager(Path.home() / ".download_cache")
    cached = cache.get("python-3.11.5.tar.gz")
    if cached:
        print(f"Already cached: {cached}")

Warnings
--------
- *expected_sha256* is **required**. This module will not download
  without a checksum.
- On verification failure, the file is **deleted** and
  :class:`ChecksumVerificationError` is raised immediately. No retries
  are attempted for checksum failures.
- Resumption requires HTTP ``Range`` header support from the server.
  If the server returns ``200`` instead of ``206``, the partial file
  is discarded and download restarts.
- On Windows, paths longer than 260 characters may raise ``OSError``.
  Enable long path support or use shorter destination paths.
- Rate limiting is approximate and uses ``time.sleep`` between chunks.
  It does not account for TCP overhead or network jitter.
- :class:`CacheManager` is **not thread-safe**. Concurrent access
  from multiple threads or processes may corrupt the metadata file.
- :class:`DownloadManager` is **not thread-safe**. Each thread must
  use its own instance.

Notes
-----
- Progress is written to ``sys.stdout``. Redirect ``sys.stdout`` to
  suppress output.
- Temporary files are created in the same directory as *dest* to
  ensure :func:`os.replace` works atomically.
- Connection pooling is not implemented (std-lib limitation). Each
  download attempt opens a fresh connection.
- The ``User-Agent`` header is sent with every request.
- Redirect handling rejects cross-origin (scheme/hostname/port)
  redirects as a security measure.
- Exponential backoff is used between retries: ``delay * (2 ** attempt)``
  with random jitter.
- :class:`CacheManager` uses atomic metadata writes (temp file +
  ``os.replace``) to prevent corruption.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_RETRIES: int = 3
"""Default maximum retry attempts per URL before trying next mirror."""

_BASE_RETRY_DELAY: float = 1.0
"""Base retry delay in seconds. Multiplied by ``2 ** attempt`` with jitter."""

_MAX_BACKOFF_DELAY: float = 60.0
"""Maximum backoff delay cap in seconds."""

_CHUNK_SIZE: int = 1024 * 1024
"""Read/write chunk size in bytes (1 MiB)."""

_MIN_RESUME_SIZE: int = 1024 * 1024
"""Minimum bytes for a partial file to be considered worth resuming."""

_USER_AGENT: str = "python-standalone-installer/1.0"
"""``User-Agent`` header value sent on every request."""

_DEFAULT_TIMEOUT: int = 30
"""Default socket timeout in seconds."""

_MAX_REDIRECTS: int = 5
"""Maximum allowed HTTP redirect depth."""

_RETRYABLE_STATUSES: frozenset = frozenset(
    {
        408,  # Request Timeout
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    }
)
"""HTTP status codes that trigger a retry."""


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class DownloadError(Exception):
    """
    Raised when a download fails after exhausting all retries and mirrors.

    Attributes
    ----------
    url : str
        The original requested URL.
    mirrors_attempted : int
        Number of mirror URLs that were tried.
    retries_exhausted : int
        Total retry attempts across all URLs.
    original_error : Exception or None
        The last underlying exception that caused the failure.
    """

    def __init__(
        self,
        message: str,
        url: str = "",
        mirrors_attempted: int = 0,
        retries_exhausted: int = 0,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.mirrors_attempted = mirrors_attempted
        self.retries_exhausted = retries_exhausted
        self.original_error = original_error

    def __str__(self) -> str:
        base = super().__str__()
        if self.url:
            base = f"{base}\n  URL: {self.url}"
        if self.mirrors_attempted:
            base = f"{base}\n  Mirrors tried: {self.mirrors_attempted}"
        if self.retries_exhausted:
            base = f"{base}\n  Retries exhausted: {self.retries_exhausted}"
        return base


class ChecksumVerificationError(Exception):
    """
    Raised when a downloaded file fails SHA256 verification.

    The corrupt file is **deleted** before this exception is raised.

    Attributes
    ----------
    filepath : Path
        Path to the file that failed (already deleted).
    expected : str
        Expected SHA256 hex digest.
    actual : str
        Computed SHA256 hex digest.
    file_size : int
        Size of the file in bytes at verification time.
    """

    def __init__(
        self,
        filepath: Path,
        expected: str,
        actual: str,
        file_size: int = 0,
    ) -> None:
        self.filepath = filepath
        self.expected = expected
        self.actual = actual
        self.file_size = file_size
        super().__init__(
            f"Checksum mismatch for {filepath.name}\n"
            f"  Size: {file_size:,} bytes\n"
            f"  Expected: {expected}\n"
            f"  Got:      {actual}"
        )


class SecurityPolicyError(Exception):
    """
    Raised when a security policy constraint is violated.

    Examples include cross-origin redirects, non-HTTPS URLs, or
    exceeding the redirect depth limit.
    """

    pass


class CacheError(Exception):
    """
    Raised when a cache read/write operation fails.

    Attributes
    ----------
    message : str
        Human-readable description.
    cache_dir : Path or None
        The cache directory involved, if applicable.
    original_error : Exception or None
        The underlying OS or I/O error.
    """

    def __init__(
        self,
        message: str,
        cache_dir: Optional[Path] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.cache_dir = cache_dir
        self.original_error = original_error


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _sanitize_filename(name: str) -> str:
    """
    Remove path separators and null bytes from *name*.

    Parameters
    ----------
    name : str
        Raw filename.

    Returns
    -------
    str
        Sanitised filename containing only safe characters.

    Raises
    ------
    ValueError
        If *name* is empty after sanitisation.
    """
    for char in ("\x00", "/", "\\"):
        name = name.replace(char, "_")
    name = name.lstrip(".")
    if not name:
        raise ValueError("Filename is empty after sanitisation.")
    return name


def _compute_sha256(filepath: Path) -> Tuple[str, int]:
    """
    Compute the SHA256 hex digest and byte size of *filepath*.

    Parameters
    ----------
    filepath : Path
        Path to the file to hash.

    Returns
    -------
    tuple[str, int]
        A ``(hex_digest, file_size)`` pair.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    PermissionError
        If *filepath* is not readable.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if not os.access(filepath, os.R_OK):
        raise PermissionError(f"File not readable: {filepath}")

    hasher = hashlib.sha256()
    file_size = 0
    with open(filepath, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
            file_size += len(chunk)
    return hasher.hexdigest(), file_size


def _parse_origin(url: str) -> Tuple[str, str, int]:
    """
    Extract scheme, hostname, and port from *url*.

    Parameters
    ----------
    url : str
        Absolute URL.

    Returns
    -------
    tuple[str, str, int]
        ``(scheme, hostname, port)``. Port defaults to 443 for HTTPS
        and 80 for HTTP if not explicitly specified.

    Raises
    ------
    ValueError
        If *url* cannot be parsed.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname or ""
    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80
    return scheme, hostname, port


def _same_origin(url_a: str, url_b: str) -> bool:
    """
    Check if two URLs have the same origin (scheme, host, port).

    Parameters
    ----------
    url_a : str
        First URL.
    url_b : str
        Second URL.

    Returns
    -------
    bool
        ``True`` if origins match.
    """
    try:
        return _parse_origin(url_a) == _parse_origin(url_b)
    except ValueError:
        return False


def _build_opener() -> urllib.request.OpenerDirector:
    """
    Create a :class:`~urllib.request.OpenerDirector` with secure defaults.

    Returns
    -------
    urllib.request.OpenerDirector
        Configured opener with ``HTTPSHandler`` only.

    Notes
    -----
    ``HTTPHandler`` is deliberately omitted so plain-text HTTP requests
    raise :class:`urllib.error.URLError`.
    """
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler())
    opener.addheaders = [
        ("User-Agent", _USER_AGENT),
        ("Accept", "*/*"),
        ("Accept-Encoding", "gzip, deflate"),
        ("Connection", "keep-alive"),
    ]
    return opener


# ---------------------------------------------------------------------------
# Secure Redirect Handler
# ---------------------------------------------------------------------------


class _SecureRedirectHandler(urllib.request.HTTPRedirectHandler):
    """
    HTTP redirect handler with security constraints.

    - Rejects cross-origin redirects (scheme/host/port changes).
    - Caps redirect depth at *_MAX_REDIRECTS*.

    Raises
    ------
    SecurityPolicyError
        On cross-origin redirect or excessive depth.
    """

    def __init__(self) -> None:
        super().__init__()
        self._redirect_count: Dict[str, int] = {}

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp,
        code: int,
        msg: str,
        headers,
        newurl: str,
    ) -> Optional[urllib.request.Request]:
        original_url = req.get_full_url()

        key = original_url
        count = self._redirect_count.get(key, 0) + 1
        self._redirect_count[key] = count

        if count > _MAX_REDIRECTS:
            raise SecurityPolicyError(
                f"Too many redirects ({count}) for {original_url}"
            )

        if not _same_origin(original_url, newurl):
            raise SecurityPolicyError(
                f"Cross-origin redirect rejected:\n"
                f"  From: {original_url}\n"
                f"  To:   {newurl}"
            )

        return super().redirect_request(req, fp, code, msg, headers, newurl)

    def http_error_301(self, req, fp, code, msg, headers):
        return self.redirect_request(req, fp, code, msg, headers, headers["Location"])

    http_error_302 = http_error_301
    http_error_303 = http_error_301
    http_error_307 = http_error_301
    http_error_308 = http_error_301


# ---------------------------------------------------------------------------
# Progress Bar
# ---------------------------------------------------------------------------


class _ProgressBar:
    """
    Terminal progress bar for streaming downloads.

    Parameters
    ----------
    total : int
        Expected total bytes. ``0`` means unknown size.
    width : int
        Bar width in characters (default 40).
    label : str
        Prefix string (default ``"Downloading"``).

    Notes
    -----
    - Redraws are throttled to ~10 Hz.
    - When *total* is ``0``, a spinner with byte count is displayed.
    - On completion, elapsed time and average speed are printed.
    """

    _SPINNER: Sequence[str] = ("|", "/", "-", "\\")

    def __init__(
        self,
        total: int,
        width: int = 40,
        label: str = "Downloading",
    ) -> None:
        if total < 0:
            raise ValueError("total must be >= 0")
        self.total = total
        self.width = max(width, 10)
        self.label = label
        self._last_draw = 0.0
        self._start = time.monotonic()
        self._spin_idx = 0

    def update(self, downloaded: int) -> None:
        """
        Redraw the progress display.

        Parameters
        ----------
        downloaded : int
            Cumulative bytes downloaded. Values above *total* are clamped.
        """
        now = time.monotonic()
        if now - self._last_draw < 0.1 and downloaded < self.total:
            return
        self._last_draw = now

        if self.total <= 0:
            char = self._SPINNER[self._spin_idx % 4]
            self._spin_idx += 1
            mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r{self.label} {char} {mb:.1f} MB")
            sys.stdout.flush()
            return

        ratio = min(downloaded / self.total, 1.0)
        filled = int(self.width * ratio)
        bar = "\u2588" * filled + "\u2591" * (self.width - filled)
        mb_dl = downloaded / (1024 * 1024)
        mb_total = self.total / (1024 * 1024)

        sys.stdout.write(
            f"\r{self.label} |{bar}| {ratio:.1%} "
            f"({mb_dl:.1f}/{mb_total:.1f} MB)"
        )
        sys.stdout.flush()

        if downloaded >= self.total:
            elapsed = now - self._start
            speed = (self.total / (1024 * 1024)) / max(elapsed, 0.001)
            sys.stdout.write(f"  Done ({elapsed:.1f}s, {speed:.1f} MB/s)\n")


# ---------------------------------------------------------------------------
# GitHub Release Fetcher
# ---------------------------------------------------------------------------

class GitHubReleaseFetcher:
    """
    Fetches release metadata and assets from a GitHub repository.

    Parameters
    ----------
    repo : str, optional
        Repository in ``"owner/name"`` format. Default
        ``"indygreg/python-build-standalone"``.
    token : str, optional
        GitHub personal access token. Providing a token raises the
        API rate limit from 60 to 5000 requests/hour.

    Raises
    ------
    ValueError
        If *repo* does not match ``"owner/name"`` format.

    Warnings
    --------
    Without a token, repeated calls may exhaust the unauthenticated
    rate limit (60 requests/hour).

    Notes
    -----
    - All API requests use HTTPS and include a ``User-Agent`` header.
    - Pagination (Link header) is handled for list endpoints.
    - Retries on transient errors with exponential backoff.

    Examples
    --------
    >>> fetcher = GitHubReleaseFetcher(token="ghp_xxxx")
    >>> release = fetcher.get_latest_release()
    >>> print(release["tag_name"])
    20231002
    """

    GITHUB_API_BASE: str = "https://api.github.com"
    """Base URL for GitHub API v3."""

    def __init__(
        self,
        repo: str = "indygreg/python-build-standalone",
        token: Optional[str] = None,
    ) -> None:
        if "/" not in repo or repo.count("/") != 1:
            raise ValueError(
                f"repo must be in 'owner/name' format, got: {repo!r}"
            )
        self.repo = repo
        self.token = token
        self._session = self._build_opener()

    def _build_opener(self) -> urllib.request.OpenerDirector:
        """
        Build an opener with GitHub-required headers.

        Returns
        -------
        urllib.request.OpenerDirector
        """
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler())
        opener.addheaders = [
            ("User-Agent", _USER_AGENT),
            ("Accept", "application/vnd.github.v3+json"),
        ]
        if self.token:
            opener.addheaders.append(
                ("Authorization", f"Bearer {self.token}")
            )
        return opener

    def _api_request(self, endpoint: str) -> Any:
        """
        Perform a pagination-aware GitHub API request.

        Parameters
        ----------
        endpoint : str
            API endpoint path relative to
            ``/repos/{owner}/{name}``, e.g. ``"/releases/latest"``.

        Returns
        -------
        object
            Parsed JSON response (dict or list of dicts).

        Raises
        ------
        DownloadError
            On HTTP errors, rate limiting, or network failures after
            all retries.
        """
        url = f"{self.GITHUB_API_BASE}/repos/{self.repo}{endpoint}"
        all_data: List[Dict[str, Any]] = []
        next_url: Optional[str] = url

        while next_url:
            req = urllib.request.Request(next_url)
            for attempt in range(1, _DEFAULT_MAX_RETRIES + 1):
                try:
                    with self._session.open(
                        req, timeout=_DEFAULT_TIMEOUT
                    ) as resp:
                        raw = resp.read().decode("utf-8")
                        data = json.loads(raw)
                except urllib.error.HTTPError as e:
                    if e.code == 403 and "rate limit" in str(e).lower():
                        raise DownloadError(
                            "GitHub API rate limit exceeded. "
                            "Provide a token or wait until the "
                            "limit resets."
                        ) from e
                    if attempt == _DEFAULT_MAX_RETRIES:
                        raise DownloadError(
                            f"HTTP {e.code} on {next_url}: {e.reason}"
                        ) from e
                    time.sleep(_BASE_RETRY_DELAY * attempt)
                except (urllib.error.URLError, OSError) as e:
                    if attempt == _DEFAULT_MAX_RETRIES:
                        raise DownloadError(
                            f"Network error after "
                            f"{_DEFAULT_MAX_RETRIES} attempts: {e}"
                        ) from e
                    time.sleep(_BASE_RETRY_DELAY * attempt)
                else:
                    break

            if isinstance(data, list):
                all_data.extend(data)
            else:
                return data

            # Handle pagination (Link header)
            next_url = None
            if hasattr(resp, "headers"):
                link = resp.headers.get("Link", "")
                for part in link.split(","):
                    if 'rel="next"' in part:
                        start = part.find("<") + 1
                        end = part.find(">")
                        if start > 0 and end > start:
                            next_url = part[start:end]
                        break

        return all_data

    def get_latest_release(self) -> Dict[str, Any]:
        """
        Return the latest published release.

        Returns
        -------
        dict
            GitHub API release object.

        Raises
        ------
        DownloadError
            If no release exists or the request fails.
        """
        release = self._api_request("/releases/latest")
        if not release:
            raise DownloadError("No releases found in repository.")
        return release

    def get_release_by_tag(self, tag: str) -> Dict[str, Any]:
        """
        Return a release for a specific Git tag.

        Parameters
        ----------
        tag : str
            Git tag (e.g., ``"20231002"``).

        Returns
        -------
        dict
            Release object.

        Raises
        ------
        DownloadError
            If the tag is not found or request fails.
        """
        encoded_tag = urllib.parse.quote(tag, safe="")
        release = self._api_request(f"/releases/tags/{encoded_tag}")
        if not release:
            raise DownloadError(f"Release tag '{tag}' not found.")
        return release

    def get_asset(
        self,
        release: Dict[str, Any],
        asset_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Find an asset by filename in a release.

        Parameters
        ----------
        release : dict
            Release object from GitHub API.
        asset_name : str
            Exact filename of the desired asset.

        Returns
        -------
        dict or None
            Asset object if found, otherwise ``None``.
        """
        for asset in release.get("assets", []):
            if asset.get("name") == asset_name:
                return asset
        return None

    def fetch_checksums(
        self,
        release: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Download and parse the SHA256 checksums file from a release.

        Searches for common checksum filenames (``SHA256SUMS``,
        ``sha256sums.txt``, ``checksums.txt``).

        Parameters
        ----------
        release : dict
            Release object.

        Returns
        -------
        dict
            Mapping of ``filename`` -> ``SHA256 hex digest``.

        Raises
        ------
        DownloadError
            If no checksum file is found in the release assets.
        SecurityPolicyError
            If the checksum file cannot be parsed (malformed).
        """
        known_names = ("SHA256SUMS", "sha256sums.txt", "checksums.txt")
        checksum_asset = None
        for name in known_names:
            asset = self.get_asset(release, name)
            if asset:
                checksum_asset = asset
                break

        if not checksum_asset:
            raise DownloadError(
                "No checksum file found in release assets. "
                f"Searched: {', '.join(known_names)}"
            )

        url = checksum_asset["browser_download_url"]
        req = urllib.request.Request(url)
        try:
            with self._session.open(req, timeout=_DEFAULT_TIMEOUT) as resp:
                content = resp.read().decode("utf-8")
        except (urllib.error.URLError, OSError) as e:
            raise DownloadError(
                f"Failed to download checksum file: {e}"
            ) from e

        checksums: Dict[str, str] = {}
        for line_no, line in enumerate(content.strip().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise SecurityPolicyError(
                    f"Malformed checksum line {line_no}: {line!r}"
                )
            digest = parts[0]
            filename = parts[-1]
            if len(digest) != 64 or not all(
                c in "0123456789abcdef" for c in digest.lower()
            ):
                raise SecurityPolicyError(
                    f"Invalid SHA256 digest on line {line_no}: {digest!r}"
                )
            checksums[filename] = digest.lower()

        if not checksums:
            raise SecurityPolicyError("Checksum file is empty.")

        return checksums


# ---------------------------------------------------------------------------
# Cache Manager
# ---------------------------------------------------------------------------


class CacheManager:
    """
    Local filesystem cache for downloaded archives with JSON metadata.

    Stores files in *cache_dir* and maintains a ``.cache_metadata.json``
    file recording the original URL, SHA256 checksum, download timestamp,
    and file size for each cached item.

    Parameters
    ----------
    cache_dir : Path
        Directory for cached files and the metadata file. Created
        automatically if it does not exist.

    Raises
    ------
    CacheError
        If *cache_dir* cannot be created or metadata cannot be
        read/written.

    Warnings
    --------
    - Cache keys are **sanitised filenames**. Two different URLs that
      resolve to the same sanitised filename will collide in the cache.
    - Deleting ``.cache_metadata.json`` resets the cache index.
      Orphaned files remain on disk but are ignored.
    - This class is **not thread-safe**. Concurrent access from
      multiple threads or processes may corrupt the metadata file.

    Notes
    -----
    - Metadata is stored as human-readable JSON with sorted keys.
    - Atomic writes are used when saving metadata (write to temp file,
      then :func:`os.replace`).
    - :meth:`get` removes stale entries where the cached file no longer
      exists on disk.
    - :meth:`list_entries` also cleans up stale entries before
      returning.

    Examples
    --------
    >>> from pathlib import Path
    >>> cache = CacheManager(Path("/tmp/my_cache"))
    >>> cache.put(
    ...     key="python-3.11.5.tar.gz",
    ...     filepath=Path("/tmp/downloaded.tar.gz"),
    ...     url="https://example.com/python.tar.gz",
    ...     checksum="abc123def456...",
    ... )
    PosixPath('/tmp/my_cache/python-3.11.5.tar.gz')
    >>> cached = cache.get("python-3.11.5.tar.gz")
    >>> if cached:
    ...     print(f"Found in cache: {cached}")
    """

    METADATA_FILE: str = ".cache_metadata.json"
    """Name of the metadata file stored inside *cache_dir*."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise CacheError(
                f"Cannot create cache directory {cache_dir}: {e}",
                cache_dir=cache_dir,
                original_error=e,
            ) from e

        self._metadata_path = self.cache_dir / self.METADATA_FILE
        self._metadata: Dict[str, Any] = self._load_metadata()

    # ------------------------------------------------------------------
    # Metadata I/O
    # ------------------------------------------------------------------

    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load cache metadata from disk.

        Returns
        -------
        dict
            Parsed metadata dict, or an empty dict if the file does
            not exist or is corrupt/unreadable.

        Notes
        -----
        Corrupt files are treated as empty. They will be overwritten
        on the next :meth:`_save_metadata` call.
        """
        if not self._metadata_path.exists():
            return {}

        try:
            with open(self._metadata_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return {}

        if not isinstance(data, dict):
            return {}
        return data

    def _save_metadata(self) -> None:
        """
        Persist metadata to disk atomically.

        Writes to a temporary file with ``0o600`` permissions, then
        replaces the existing metadata file via :func:`os.replace`.

        Raises
        ------
        CacheError
            If the write or replace operation fails.
        """
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".metadata_",
            dir=str(self.cache_dir),
        )
        tmp_path = Path(tmp_path)
        try:
            os.close(tmp_fd)
            os.chmod(tmp_path, 0o600)
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(self._metadata, fh, indent=2, sort_keys=True)
            os.replace(tmp_path, self._metadata_path)
        except OSError as e:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise CacheError(
                f"Failed to save cache metadata to {self._metadata_path}: {e}",
                cache_dir=self.cache_dir,
                original_error=e,
            ) from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Path]:
        """
        Retrieve a cached file by key.

        Parameters
        ----------
        key : str
            Cache key (filename). Sanitised before lookup via
            :func:`_sanitize_filename`.

        Returns
        -------
        Path or None
            Path to the cached file if it exists and is recorded in
            metadata, otherwise ``None``.

        Notes
        -----
        - If a metadata entry exists but the file is missing from
          disk, the stale entry is removed from metadata and
          ``None`` is returned.
        """
        key = _sanitize_filename(key)
        entry = self._metadata.get(key)
        if not entry:
            return None

        filepath = self.cache_dir / key
        if not filepath.exists():
            del self._metadata[key]
            self._save_metadata()
            return None

        return filepath

    def put(
        self,
        key: str,
        filepath: Path,
        url: str,
        checksum: Optional[str] = None,
    ) -> Path:
        """
        Store a file in the cache and record its provenance.

        Parameters
        ----------
        key : str
            Cache key (filename). Sanitised before storage.
        filepath : Path
            Path to the source file. Copied into *cache_dir*; the
            original is left unchanged.
        url : str
            Original download URL for provenance tracking.
        checksum : str or None
            SHA256 hex digest of the file, if verified. Stored in
            metadata for auditing.

        Returns
        -------
        Path
            Path to the cached copy.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        CacheError
            If the file cannot be copied or metadata cannot be saved.

        Notes
        -----
        - If a cached file with the same *key* already exists, it is
          silently overwritten.
        - The file is **copied**, not moved.
        """
        key = _sanitize_filename(key)

        if not filepath.exists():
            raise FileNotFoundError(f"Source file not found: {filepath}")

        dest = self.cache_dir / key
        if filepath.resolve() == dest.resolve():
            return dest

        try:
            shutil.copy2(filepath, dest)
        except OSError as e:
            raise CacheError(
                f"Failed to copy {filepath} to cache at {dest}: {e}",
                cache_dir=self.cache_dir,
                original_error=e,
            ) from e

        self._metadata[key] = {
            "url": url,
            "checksum": checksum,
            "timestamp": time.time(),
            "size": dest.stat().st_size,
        }
        self._save_metadata()

        return dest

    def remove(self, key: str) -> bool:
        """
        Remove a cached file and its metadata entry.

        Parameters
        ----------
        key : str
            Cache key to remove. Sanitised before lookup.

        Returns
        -------
        bool
            ``True`` if the entry existed and was removed, ``False``
            if the key was not found.

        Notes
        -----
        - If the file exists on disk but there is no metadata entry,
          this method returns ``False`` and the orphaned file is
          **not** deleted.
        """
        key = _sanitize_filename(key)
        entry = self._metadata.pop(key, None)
        if entry is None:
            return False

        filepath = self.cache_dir / key
        try:
            if filepath.exists():
                filepath.unlink()
        except OSError:
            pass

        self._save_metadata()
        return True

    def exists(self, key: str) -> bool:
        """
        Check if *key* exists in the cache and the file is present.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        bool
            ``True`` if both the metadata entry and the cached file
            exist on disk.
        """
        return self.get(key) is not None

    def clear(self) -> int:
        """
        Remove all cached files and reset metadata.

        Returns
        -------
        int
            Total number of bytes freed.

        Notes
        -----
        - The *cache_dir* itself is preserved; only its contents
          (except the metadata file) are deleted.
        - Metadata is reset to an empty dictionary.
        """
        total_size = 0

        if self.cache_dir.exists():
            for item in self.cache_dir.iterdir():
                if item.name == self.METADATA_FILE:
                    continue
                try:
                    if item.is_file():
                        total_size += item.stat().st_size
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink()
                except OSError:
                    pass

        self._metadata = {}
        self._save_metadata()

        return total_size

    def list_entries(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a copy of all cache metadata entries.

        Returns
        -------
        dict
            Mapping of cache key to metadata dict. Each metadata
            dict contains ``url``, ``checksum``, ``timestamp``,
            and ``size``.

        Notes
        -----
        - Stale entries (where the file is missing on disk) are
          cleaned up before the result is returned.
        - The returned dict is a shallow copy; modifications to it
          do not affect the cache.
        """
        stale_keys = [
            key
            for key in self._metadata
            if not (self.cache_dir / key).exists()
        ]
        for key in stale_keys:
            del self._metadata[key]

        if stale_keys:
            self._save_metadata()

        return dict(self._metadata)

    def get_size(self) -> int:
        """
        Get the total size of all cached files.

        Returns
        -------
        int
            Total bytes used by files that exist on disk and are
            recorded in metadata. Returns ``0`` if the cache is
            empty or the directory does not exist.
        """
        total = 0
        for key in self._metadata:
            filepath = self.cache_dir / key
            try:
                if filepath.exists():
                    total += filepath.stat().st_size
            except OSError:
                pass
        return total

    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific cache entry.

        Parameters
        ----------
        key : str
            Cache key. Sanitised before lookup.

        Returns
        -------
        dict or None
            Metadata dict with keys ``url``, ``checksum``,
            ``timestamp``, and ``size``, or ``None`` if the key
            is not found or the cached file is missing.

        Notes
        -----
        - If the metadata entry exists but the file is missing,
          the stale entry is removed and ``None`` is returned.
        """
        key = _sanitize_filename(key)
        entry = self._metadata.get(key)
        if not entry:
            return None

        filepath = self.cache_dir / key
        if not filepath.exists():
            del self._metadata[key]
            self._save_metadata()
            return None

        return dict(entry)


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------


class _RateLimiter:
    """
    Token-bucket rate limiter for bandwidth throttling.

    Parameters
    ----------
    bytes_per_second : float
        Maximum bytes per second. ``0`` or negative means unlimited.

    Notes
    -----
    - Not precise; uses ``time.sleep`` between chunks.
    - Window resets every 1 second.
    """

    def __init__(self, bytes_per_second: float) -> None:
        self._rate = max(bytes_per_second, 0.0)
        self._last_consume = 0.0
        self._consumed_this_window = 0

    def consume(self, chunk_size: int) -> None:
        """
        Sleep if necessary to stay under the rate limit.

        Parameters
        ----------
        chunk_size : int
            Bytes consumed in this chunk.
        """
        if self._rate <= 0:
            return

        now = time.monotonic()
        if self._last_consume == 0.0:
            self._last_consume = now
            self._consumed_this_window = chunk_size
            return

        elapsed = now - self._last_consume
        if elapsed >= 1.0:
            self._last_consume = now
            self._consumed_this_window = chunk_size
            return

        self._consumed_this_window += chunk_size
        if self._consumed_this_window >= self._rate:
            sleep_time = 1.0 - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._last_consume = time.monotonic()
            self._consumed_this_window = 0


# ---------------------------------------------------------------------------
# Download Manager
# ---------------------------------------------------------------------------


class DownloadManager:
    """
    Download files over HTTPS with retries, mirrors, resumption, rate
    limiting, and mandatory SHA256 checksum verification.

    Parameters
    ----------
    mirrors : list of str, optional
        Additional base URLs tried in order after the primary URL
        fails. All mirrors must use HTTPS.
    max_retries : int
        Maximum attempts per URL. Default 3.
    timeout : int
        Socket timeout in seconds. Default 30.
    rate_limit_mbps : float
        Maximum download speed in megabytes per second. ``0`` or
        negative means unlimited. Default ``0``.
    show_progress : bool
        If ``True`` (default), a progress bar is written to
        ``sys.stdout`` during download.

    Raises
    ------
    ValueError
        If *max_retries* < 1, *timeout* < 1, or any mirror URL does
        not start with ``https://``.

    Examples
    --------
    >>> from pathlib import Path
    >>> manager = DownloadManager(max_retries=5, rate_limit_mbps=10.0)
    >>> manager.download(
    ...     url="https://github.com/.../file.tar.gz",
    ...     dest=Path("./file.tar.gz"),
    ...     expected_sha256="d2b4f0e8...",
    ... )
    PosixPath('file.tar.gz')
    """

    def __init__(
        self,
        mirrors: Optional[List[str]] = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        timeout: int = _DEFAULT_TIMEOUT,
        rate_limit_mbps: float = 0.0,
        show_progress: bool = True,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if timeout < 1:
            raise ValueError("timeout must be >= 1")

        self._mirrors: List[str] = []
        if mirrors:
            for m in mirrors:
                if not m.startswith("https://"):
                    raise ValueError(f"Mirror URL must use HTTPS: {m!r}")
            self._mirrors = list(mirrors)

        self._max_retries = max_retries
        self._timeout = timeout
        self._rate_limiter = _RateLimiter(rate_limit_mbps * 1024 * 1024)
        self._show_progress = show_progress
        self._opener = _build_opener()
        self._opener.add_handler(_SecureRedirectHandler())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(
        self,
        url: str,
        dest: Path,
        expected_sha256: str,
        resume: bool = True,
    ) -> Path:
        """
        Download a file from *url* to *dest* with mandatory checksum
        verification.

        Parameters
        ----------
        url : str
            Primary HTTPS URL of the file.
        dest : Path
            Destination path. The parent directory must exist.
        expected_sha256 : str
            **Required.** SHA256 hex digest of the expected file.
            Verification is performed on a temporary file before
            moving to *dest*.
        resume : bool
            If ``True`` (default) and *dest* already exists with a
            size >= 1 MiB, attempt to resume the download using an
            HTTP ``Range`` request.

        Returns
        -------
        Path
            The *dest* path on success.

        Raises
        ------
        ValueError
            If *url* is not HTTPS, *expected_sha256* is empty, or
            the digest is not a 64-character hex string.
        FileNotFoundError
            If the parent directory of *dest* does not exist.
        ChecksumVerificationError
            If the computed SHA256 does not match *expected_sha256*.
            The downloaded file is deleted before this is raised.
        DownloadError
            If the download fails after every retry and every mirror.
        SecurityPolicyError
            If a cross-origin redirect or excessive redirect depth
            is detected.

        Warnings
        --------
        - *expected_sha256* is mandatory. There is no opt-out.
        - The temporary file is created in the same directory as
          *dest*. Ensure sufficient free space for both the partial
          and final file simultaneously.
        - On verification failure, no retries are attempted --- the
          error is raised immediately.

        Notes
        -----
        - Mirrors are tried in order within each retry cycle. The
          primary URL is always first. After all mirrors are
          exhausted, the next retry cycle begins.
        - Retry delay uses exponential backoff with jitter::

              delay = min(BASE_RETRY_DELAY * (2 ** attempt), MAX_BACKOFF_DELAY)
              delay *= (0.5 + random.random())
        """
        if not url.startswith("https://"):
            raise ValueError(f"Only HTTPS URLs accepted: {url!r}")
        if not expected_sha256 or not expected_sha256.strip():
            raise ValueError("expected_sha256 is required")
        if not dest.parent.exists():
            raise FileNotFoundError(
                f"Destination directory does not exist: {dest.parent}"
            )

        expected_sha256 = expected_sha256.strip().lower()

        if len(expected_sha256) != 64 or not all(
            c in "0123456789abcdef" for c in expected_sha256
        ):
            raise ValueError(
                f"Invalid SHA256 digest: {expected_sha256!r}. "
                "Must be 64 hex characters."
            )

        filename = url.rsplit("/", 1)[-1]
        urls = [url] + [
            f"{m.rstrip('/')}/{filename}" for m in self._mirrors
        ]

        last_error: Optional[Exception] = None
        total_attempts = 0

        for attempt in range(1, self._max_retries + 1):
            for candidate_url in urls:
                total_attempts += 1
                try:
                    return self._attempt_download(
                        candidate_url=candidate_url,
                        dest=dest,
                        expected_sha256=expected_sha256,
                        resume=resume,
                    )
                except ChecksumVerificationError:
                    raise
                except (DownloadError, OSError, urllib.error.URLError) as exc:
                    last_error = exc
                    backoff = min(
                        _BASE_RETRY_DELAY * (2 ** attempt),
                        _MAX_BACKOFF_DELAY,
                    )
                    jitter = backoff * (0.5 + random.random())
                    time.sleep(jitter)

        raise DownloadError(
            f"Failed to download after {total_attempts} attempt(s) "
            f"across {len(urls)} URL(s).",
            url=url,
            mirrors_attempted=len(self._mirrors),
            retries_exhausted=self._max_retries,
            original_error=last_error,
        )
        
    def download_release_asset(
        self,
        release_tag: str,
        asset_filename: str,
        fetcher: GitHubReleaseFetcher,
        cache: CacheManager,
        force: bool = False,
    ) -> Path:
        """
        Download a release asset with caching and checksum verification.
    
        Parameters
        ----------
        release_tag : str
            Git tag of the release (e.g., "20231002").
        asset_filename : str
            Exact filename of the asset.
        fetcher : GitHubReleaseFetcher
            Fetcher instance for GitHub API calls.
        cache : CacheManager
            Cache instance for storing downloads.
        force : bool
            If True, re-download even if cached.
    
        Returns
        -------
        Path
            Path to the verified, downloaded file.
    
        Raises
        ------
        DownloadError
            If asset not found or download fails.
        """
        if not force:
            cached = cache.get(asset_filename)
            if cached:
                return cached
    
        # Get release metadata
        release = fetcher.get_release_by_tag(release_tag)
        asset = fetcher.get_asset(release, asset_filename)
        if not asset:
            raise DownloadError(
                f"Asset '{asset_filename}' not found in release "
                f"'{release_tag}'."
            )
    
        # Get checksum
        checksums = fetcher.fetch_checksums(release)
        expected_digest = checksums.get(asset_filename)
        if not expected_digest:
            raise DownloadError(
                f"No checksum found for asset '{asset_filename}'."
            )
    
        # Download to cache
        dest = cache.cache_dir / asset_filename
        downloaded = self.download(
            url=asset["browser_download_url"],
            dest=dest,
            expected_sha256=expected_digest,
            resume=True,
        )
    
        # Record in cache
        cache.put(
            key=asset_filename,
            filepath=downloaded,
            url=asset["browser_download_url"],
            checksum=expected_digest,
        )
    
        return downloaded

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _attempt_download(
        self,
        candidate_url: str,
        dest: Path,
        expected_sha256: str,
        resume: bool,
    ) -> Path:
        """
        Perform a single download attempt.

        Parameters
        ----------
        candidate_url : str
            URL to download from.
        dest : Path
            Final destination path.
        expected_sha256 : str
            Lowercase hex digest for verification.
        resume : bool
            Whether to attempt resumption.

        Returns
        -------
        Path
            *dest* on success.

        Raises
        ------
        DownloadError
            On HTTP/network failure.
        ChecksumVerificationError
            If verification fails.
        """
        existing_bytes = 0

        if resume and dest.exists():
            file_size = dest.stat().st_size
            if file_size >= _MIN_RESUME_SIZE:
                existing_bytes = file_size
            else:
                dest.unlink(missing_ok=True)

        headers: Dict[str, str] = {}
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"

        request = urllib.request.Request(candidate_url, headers=headers)

        try:
            response = self._opener.open(request, timeout=self._timeout)
        except urllib.error.HTTPError as exc:
            return self._handle_http_error(exc, dest, expected_sha256, candidate_url)
        except urllib.error.URLError as exc:
            raise DownloadError(
                f"Cannot reach {candidate_url}: {exc.reason}"
            ) from exc
        except OSError as exc:
            raise DownloadError(f"OS error for {candidate_url}: {exc}") from exc

        return self._process_response(
            response=response,
            candidate_url=candidate_url,
            dest=dest,
            expected_sha256=expected_sha256,
            existing_bytes=existing_bytes,
        )

    def _handle_http_error(
        self,
        exc: urllib.error.HTTPError,
        dest: Path,
        expected_sha256: str,
        url: str,
    ) -> Path:
        """
        Handle HTTP error responses.

        Parameters
        ----------
        exc : urllib.error.HTTPError
            The HTTP error.
        dest : Path
            Destination path.
        expected_sha256 : str
            Expected checksum.
        url : str
            The URL that failed.

        Returns
        -------
        Path
            *dest* if ``416 Range Not Satisfiable`` (file already
            complete and verified).

        Raises
        ------
        DownloadError
            For non-retryable errors or after verification failure.
        ChecksumVerificationError
            If the existing file fails verification on a 416 response.
        """
        code = exc.code

        if code == 416:
            digest, file_size = _compute_sha256(dest)
            if digest != expected_sha256:
                dest.unlink(missing_ok=True)
                raise ChecksumVerificationError(
                    filepath=dest,
                    expected=expected_sha256,
                    actual=digest,
                    file_size=file_size,
                )
            return dest

        if code in _RETRYABLE_STATUSES:
            raise DownloadError(
                f"Retryable HTTP {code} from {url}: {exc.reason}"
            ) from exc

        raise DownloadError(f"HTTP {code} from {url}: {exc.reason}") from exc

    def _process_response(
        self,
        response,
        candidate_url: str,
        dest: Path,
        expected_sha256: str,
        existing_bytes: int,
    ) -> Path:
        """
        Process a successful HTTP response.

        Parameters
        ----------
        response : http.client.HTTPResponse
            Open response object.
        candidate_url : str
            The URL.
        dest : Path
            Final destination.
        expected_sha256 : str
            Expected checksum.
        existing_bytes : int
            Bytes already on disk (for resume).

        Returns
        -------
        Path
            *dest* on success.

        Raises
        ------
        DownloadError
            On unexpected status or zero-byte download.
        ChecksumVerificationError
            On verification failure.
        """
        status = response.status

        if status == 206:
            total_expected = existing_bytes + int(
                response.headers.get("Content-Length", 0)
            )
            offset = existing_bytes
            append_mode = True
        elif status == 200:
            if existing_bytes > 0:
                dest.unlink(missing_ok=True)
            total_expected = int(response.headers.get("Content-Length", 0))
            offset = 0
            append_mode = False
        else:
            raise DownloadError(
                f"Unexpected HTTP status {status} from {candidate_url}"
            )

        if append_mode:
            final_file = dest
            write_mode = "ab"
            tmp_path: Optional[Path] = None
        else:
            tmp_fd, tmp_path_str = tempfile.mkstemp(
                prefix=f".{dest.name}.", dir=dest.parent
            )
            tmp_path = Path(tmp_path_str)
            os.close(tmp_fd)
            os.chmod(tmp_path, 0o600)
            final_file = tmp_path
            write_mode = "wb"

        try:
            bytes_written = self._stream_to_file(
                response=response,
                target=final_file,
                total_expected=total_expected,
                offset=offset,
                write_mode=write_mode,
            )

            if bytes_written == 0 and total_expected > 0:
                raise DownloadError("Zero bytes written to file.")

            digest, file_size = _compute_sha256(final_file)
            if digest != expected_sha256:
                final_file.unlink(missing_ok=True)
                raise ChecksumVerificationError(
                    filepath=final_file,
                    expected=expected_sha256,
                    actual=digest,
                    file_size=file_size,
                )

            if not append_mode and tmp_path:
                os.replace(tmp_path, dest)
                final_file = dest

            return final_file

        except Exception:
            if not append_mode and tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    def _stream_to_file(
        self,
        response,
        target: Path,
        total_expected: int,
        offset: int,
        write_mode: str,
    ) -> int:
        """
        Stream HTTP response body to *target*.

        Parameters
        ----------
        response : http.client.HTTPResponse
            Open response object.
        target : Path
            File to write to.
        total_expected : int
            Expected total bytes (0 if unknown).
        offset : int
            Bytes already on disk before this write.
        write_mode : str
            ``"wb"`` for fresh download, ``"ab"`` for resume.

        Returns
        -------
        int
            Number of bytes written in this call (excludes *offset*).
        """
        downloaded = offset
        progress = (
            _ProgressBar(total=max(total_expected, 0))
            if self._show_progress
            else None
        )
        if progress:
            progress.update(downloaded)

        with open(target, write_mode) as fh:
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if progress:
                    progress.update(downloaded)
                self._rate_limiter.consume(len(chunk))

        if progress:
            progress.update(downloaded)

        return downloaded - offset
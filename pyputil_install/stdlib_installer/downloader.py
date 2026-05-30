"""
HTTP client and download management for stdlib_installer.

Handles all network communication with GitHub's API and raw content
endpoints. Provides:

- HTTP client with dual backend support (requests or urllib)
- Automatic retry with exponential backoff
- Rate limit detection and handling
- File-based caching with TTL expiration and size limits
- Recursive directory tree fetching and downloading
- Checksum verification for downloaded files

All GitHub API responses are parsed into dataclass instances rather
than raw dicts to provide type safety, autocompletion, and cleaner
code throughout the package.
"""

from __future__ import annotations

import hashlib
import json
import time
import logging
from pathlib import Path
from typing import Union, List, Dict, Optional, Any, Set, Iterator
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

from .exceptions import (
    NetworkError,
    GitHubAPIError,
    RateLimitError,
    PackageNotFoundError,
    ChecksumVerificationError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: requests
# ---------------------------------------------------------------------------

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    _HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    import urllib.parse

    _HAS_REQUESTS = False
    logger.debug("requests not available, falling back to urllib")


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class GitHubItem:
    """
    Parsed representation of a single file or directory entry from
    the GitHub Contents API.

    The GitHub Contents API returns JSON objects with keys such as
    ``name``, ``path``, ``type``, ``download_url``, ``url``, ``sha``,
    and ``size``. This class wraps that data with type-safe attribute
    access and computed properties for distinguishing files from
    directories and for building raw download paths.

    Parameters
    ----------
    name : str
        Base name of the file or directory (e.g., ``"__init__.py"``,
        ``"email"``).
    path : str
        Full repository-relative path (e.g.,
        ``"Lib/json/__init__.py"``, ``"Lib/email/mime"``).
    item_type : str
        Either ``"file"`` or ``"dir"``.
    download_url : str or None
        Direct raw download URL for files. Always ``None`` for
        directories because directories have no raw content.
    api_url : str or None
        API URL for fetching this item's metadata or (for directories)
        its child contents. Always present, but stored separately from
        download_url for clarity.
    sha : str
        Git blob SHA (files) or tree SHA (directories).
    size : int
        Content size in bytes. Always 0 for directories.

    Attributes
    ----------
    is_file : bool
        Computed property; ``True`` when ``item_type == "file"``.
    is_dir : bool
        Computed property; ``True`` when ``item_type == "dir"``.
    lib_path : str
        Computed property; the ``path`` with the leading ``"Lib/"``
        prefix removed, suitable for constructing raw download URLs
        (e.g., ``"Lib/json/__init__.py"`` becomes
        ``"json/__init__.py"``).

    Examples
    --------
    Parse a raw API dict:

    >>> raw = {
    ...     "name": "__init__.py",
    ...     "path": "Lib/json/__init__.py",
    ...     "type": "file",
    ...     "download_url": "https://raw.githubusercontent.com/...",
    ...     "url": "https://api.github.com/...",
    ...     "sha": "abc123",
    ...     "size": 14147,
    ... }
    >>> item = GitHubItem.from_api_response(raw)
    >>> item.name
    '__init__.py'
    >>> item.is_file
    True
    >>> item.lib_path
    'json/__init__.py'
    """

    name: str
    path: str
    item_type: str
    download_url: Optional[str]
    api_url: Optional[str]
    sha: str
    size: int

    @property
    def is_file(self) -> bool:
        """
        Return ``True`` if this item represents a regular file.

        Checks ``item_type`` against the string ``"file"``. All other
        types (``"dir"``, ``"symlink"``, ``"submodule"``) return
        ``False``.

        Returns
        -------
        bool
        """
        return self.item_type == "file"

    @property
    def is_dir(self) -> bool:
        """
        Return ``True`` if this item represents a directory.

        Checks ``item_type`` against the string ``"dir"``.

        Returns
        -------
        bool
        """
        return self.item_type == "dir"

    @property
    def lib_path(self) -> str:
        """
        Return the item's repository path with the leading ``"Lib/"``
        prefix stripped.

        The GitHub Contents API returns paths relative to the
        repository root (e.g., ``"Lib/json/__init__.py"``). Raw
        content URLs are also relative to the repository root but do
        not repeat the ``Lib/`` component because the raw URL base
        already includes it. This property removes the prefix so that
        callers can concatenate the result directly with the raw base
        URL.

        If the path does not start with ``"Lib/"``, it is returned
        unmodified.

        Returns
        -------
        str
            Path relative to the ``Lib/`` directory.
        """
        if self.path.startswith("Lib/"):
            return self.path[4:]
        return self.path

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "GitHubItem":
        """
        Construct a ``GitHubItem`` from a raw GitHub Contents API
        response dictionary.

        Extracts the standard keys (``name``, ``path``, ``type``,
        ``download_url``, ``url``, ``sha``, ``size``) and passes them
        to the constructor. Missing keys default to empty strings or
        zero where appropriate.

        Parameters
        ----------
        data : dict
            A single item dictionary from the GitHub Contents API.

        Returns
        -------
        GitHubItem
            A fully populated instance.
        """
        return cls(
            name=data.get("name", ""),
            path=data.get("path", ""),
            item_type=data.get("type", "file"),
            download_url=data.get("download_url"),
            api_url=data.get("url"),
            sha=data.get("sha", ""),
            size=data.get("size", 0),
        )


@dataclass
class FetchResult:
    """
    Unified container for the result of a ``fetch_tree`` operation.

    The GitHub Contents API returns a JSON object for single-file
    modules (e.g., ``pathlib.py``) and a JSON array for package
    directories (e.g., ``json/``). This class normalises both cases
    into a single type that can be iterated uniformly.

    Parameters
    ----------
    is_single_file : bool
        ``True`` when the result wraps a single file; ``False`` when
        it wraps a directory listing.
    single_item : GitHubItem or None
        The single file item. Only populated when ``is_single_file``
        is ``True``; ``None`` otherwise.
    items : list of GitHubItem
        The directory contents. Only populated when ``is_single_file``
        is ``False``; empty list otherwise.

    Examples
    --------
    Iterating over a ``FetchResult``:

    >>> result = downloader.fetch_tree("json", "3.13")
    >>> for item in result:
    ...     print(item.name)
    __init__.py
    decoder.py
    encoder.py
    scanner.py
    tool.py

    Checking length:

    >>> len(result)
    5
    """

    is_single_file: bool
    single_item: Optional[GitHubItem] = None
    items: List[GitHubItem] = field(default_factory=list)

    def __iter__(self) -> Iterator[GitHubItem]:
        """
        Yield each ``GitHubItem`` in the result.

        For single-file results, yields the single item once.
        For directory results, yields each child item in order.

        Yields
        ------
        GitHubItem
        """
        if self.is_single_file:
            if self.single_item is not None:
                yield self.single_item
        else:
            yield from self.items

    def __len__(self) -> int:
        """
        Return the number of items in this result.

        Returns 1 for single-file results, or the length of the
        ``items`` list for directories.

        Returns
        -------
        int
        """
        return 1 if self.is_single_file else len(self.items)

    def __bool__(self) -> bool:
        """
        Return ``True`` if the result contains at least one item.

        Returns
        -------
        bool
        """
        return self.is_single_file or bool(self.items)


# ============================================================================
# HTTP Client
# ============================================================================


class HttpClient:
    """
    HTTP client for communicating with the GitHub API and raw content
    CDN.

    Wraps either a ``requests.Session`` (if the ``requests`` library is
    installed) or ``urllib.request`` (standard library fallback). Both
    backends provide the same public interface: ``get()``, ``get_json()``,
    and ``head()``.

    Retry behaviour
    ---------------
    When using the ``requests`` backend, retries are handled by
    ``urllib3.Retry`` configured on a per-session adapter. Transient
    failures (HTTP 429, 500, 502, 503, 504) are retried up to
    ``max_retries`` times with exponential backoff.

    When using the ``urllib`` backend, retries are implemented manually
    in a loop. Only ``URLError`` exceptions trigger retries; HTTP error
    responses (including 429 and 5xx) are raised immediately after
    parsing.

    Rate limiting
    -------------
    HTTP 429 responses are parsed and re-raised as ``RateLimitError``.
    Callers are expected to implement their own waiting strategy; this
    client does not block automatically.

    Parameters
    ----------
    timeout : int
        Request timeout in seconds. Applied to both the connect and
        read phases. Must be a positive integer.
    max_retries : int
        Maximum number of retry attempts for transient failures. Set
        to 0 to disable retries entirely.
    backoff_factor : float
        Multiplier for exponential backoff when using the ``requests``
        backend. The delay before retry *n* is
        ``backoff_factor * (2 ** (n - 1))`` seconds.

    Attributes
    ----------
    session : requests.Session or None
        The underlying session object when ``requests`` is available;
        ``None`` when falling back to ``urllib``.
    timeout : int
        Configured timeout in seconds.
    max_retries : int
        Configured maximum retry count.
    backoff_factor : float
        Configured backoff multiplier.

    Examples
    --------
    >>> client = HttpClient(timeout=15, max_retries=5)
    >>> data = client.get_json("https://api.github.com/repos/python/cpython")
    >>> content = client.get("https://raw.githubusercontent.com/...")
    """

    API_BASE = "https://api.github.com/repos/python/cpython"
    RAW_BASE = "https://raw.githubusercontent.com/python/cpython"

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if backoff_factor < 0:
            raise ValueError("backoff_factor must be non-negative")

        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session: Optional[requests.Session] = None

        if _HAS_REQUESTS:
            self._setup_requests_session()

    def _setup_requests_session(self) -> None:
        """
        Create and configure a ``requests.Session`` with retry adapter.

        Sets default headers (User-Agent, Accept) and mounts an
        ``HTTPAdapter`` with a ``Retry`` strategy for both HTTP and
        HTTPS prefixes. Connection pooling is enabled with a pool size
        of 10 connections per host.
        """
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "StdlibInstaller/2.0 (Python)",
                "Accept": "application/vnd.github.v3+json",
            }
        )

        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # ------------------------------------------------------------------
    # Response status handling
    # ------------------------------------------------------------------

    def _handle_response_status(
        self,
        url: str,
        status_code: int,
        response_body: str,
    ) -> None:
        """
        Inspect an HTTP status code and raise the appropriate exception
        for non-success responses.

        Parameters
        ----------
        url : str
            The full URL that was requested.
        status_code : int
            HTTP status code from the response.
        response_body : str
            Raw response body text, used for extracting error details
            in some cases (e.g., rate limit information).

        Raises
        ------
        PackageNotFoundError
            When ``status_code == 404``. The package name is extracted
            heuristically from the last path segment of the URL.
        RateLimitError
            When ``status_code == 429``. The response body is parsed
            as JSON to attempt to extract remaining rate limit count.
        GitHubAPIError
            For all other non-2xx status codes.
        """
        if 200 <= status_code < 300:
            return

        if status_code == 404:
            package_name = url.rstrip("/").split("/")[-1]
            raise PackageNotFoundError(package_name=package_name)

        if status_code == 429:
            remaining = None
            try:
                body_data = json.loads(response_body)
                if "rate" in body_data:
                    remaining = body_data["rate"].get("remaining")
            except (json.JSONDecodeError, KeyError):
                pass

            raise RateLimitError(
                url=url,
                remaining=remaining,
            )

        raise GitHubAPIError(
            url=url,
            status_code=status_code,
            response_body=response_body,
        )

    # ------------------------------------------------------------------
    # Request backends
    # ------------------------------------------------------------------

    def _make_request_with_requests(
        self, url: str, stream: bool = False
    ) -> requests.Response:
        """
        Execute an HTTP GET using the ``requests`` library.

        Parameters
        ----------
        url : str
            Target URL.
        stream : bool
            If ``True``, response content is not immediately downloaded
            (used for large files).

        Returns
        -------
        requests.Response
            Successful HTTP response.

        Raises
        ------
        NetworkError
            Wraps any ``requests`` exception (timeout, connection,
            TLS, etc.) with a user-facing message.
        GitHubAPIError
            Propagated from ``_handle_response_status`` for non-2xx
            responses.
        """
        try:
            response = self.session.get(
                url,
                timeout=self.timeout,
                stream=stream,
            )
            self._handle_response_status(url, response.status_code, response.text)
            return response

        except requests.exceptions.Timeout as e:
            raise NetworkError(url, "Request timed out", original_error=e)
        except requests.exceptions.ConnectionError as e:
            raise NetworkError(
                url,
                f"Connection failed: {e}",
                original_error=e,
            )
        except requests.exceptions.SSLError as e:
            raise NetworkError(
                url,
                f"TLS/SSL error: {e}",
                original_error=e,
            )
        except requests.exceptions.RequestException as e:
            raise NetworkError(
                url,
                f"Request failed: {e}",
                original_error=e,
            )

    def _make_request_with_urllib(
        self, url: str
    ) -> "http.client.HTTPResponse":
        """
        Execute an HTTP GET using ``urllib.request`` from the standard
        library.

        Implements manual retry loop for ``URLError`` exceptions. HTTP
        error responses (``HTTPError``) are handled immediately without
        retry because they represent server-side rejections that are
        unlikely to change on retry.

        Parameters
        ----------
        url : str
            Target URL.

        Returns
        -------
        http.client.HTTPResponse
            Successful HTTP response object.

        Raises
        ------
        NetworkError
            After exhausting all retry attempts for ``URLError``, or
            immediately on ``TimeoutError``.
        GitHubAPIError
            Propagated from ``_handle_response_status`` for HTTP error
            responses.
        """
        import http.client

        headers = {
            "User-Agent": "StdlibInstaller/2.0 (Python)",
            "Accept": "application/vnd.github.v3+json",
        }

        req = urllib.request.Request(url, headers=headers)

        for attempt in range(self.max_retries + 1):
            try:
                response = urllib.request.urlopen(req, timeout=self.timeout)
                body = response.read().decode("utf-8", errors="replace")
                self._handle_response_status(url, response.getcode(), body)
                return response

            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace") if e.fp else ""
                self._handle_response_status(url, e.code, body)

            except urllib.error.URLError as e:
                if attempt < self.max_retries:
                    delay = self.backoff_factor * (2**attempt)
                    logger.debug(
                        f"Retrying {url} in {delay}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                    continue
                raise NetworkError(
                    url,
                    f"Connection failed after {self.max_retries + 1} attempts: {e.reason}",
                    original_error=e,
                )

            except TimeoutError as e:
                raise NetworkError(url, "Request timed out", original_error=e)

        raise NetworkError(url, "Request failed for unknown reason")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, url: str) -> bytes:
        """
        Execute an HTTP GET and return the raw response body as bytes.

        Dispatches to the ``requests`` or ``urllib`` backend depending
        on availability.

        Parameters
        ----------
        url : str
            The URL to fetch.

        Returns
        -------
        bytes
            Response body content.

        Raises
        ------
        NetworkError
            On connection failures, timeouts, or TLS errors.
        GitHubAPIError
            On non-2xx HTTP responses.
        """
        if self.session is not None:
            response = self._make_request_with_requests(url)
            return response.content
        else:
            response = self._make_request_with_urllib(url)
            return response.read()

    def get_json(self, url: str) -> Any:
        """
        Execute an HTTP GET and parse the response body as JSON.

        Parameters
        ----------
        url : str
            The URL to fetch JSON from.

        Returns
        -------
        Any
            Parsed JSON data (typically ``dict`` or ``list``).

        Raises
        ------
        NetworkError
            On connection failures, timeouts, or TLS errors.
        GitHubAPIError
            On non-2xx HTTP responses.
        ValueError
            If the response body is not valid JSON.
        """
        if self.session is not None:
            response = self._make_request_with_requests(url)
            try:
                return response.json()
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response from {url}: {e}")
        else:
            response = self._make_request_with_urllib(url)
            raw = response.read().decode("utf-8", errors="replace")
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response from {url}: {e}")

    def head(self, url: str) -> Dict[str, str]:
        """
        Send an HTTP HEAD request and return the response headers.

        Useful for checking whether a resource exists or for reading
        ``Content-Length`` without transferring the body.

        When using the ``urllib`` backend, a real HEAD request is
        attempted. If the server rejects HEAD (some CDNs do), the
        method falls back to a GET with a ``Range: bytes=0-0`` header
        to minimise data transfer.

        Parameters
        ----------
        url : str
            The URL to check.

        Returns
        -------
        dict
            Response headers as a dictionary (keys are lowercased).

        Raises
        ------
        NetworkError
            On connection failures.
        GitHubAPIError
            On HTTP error responses.
        """
        if self.session is not None:
            response = self.session.head(url, timeout=self.timeout)
            self._handle_response_status(url, response.status_code, response.text)
            return dict(response.headers)
        else:
            headers = {
                "User-Agent": "StdlibInstaller/2.0 (Python)",
                "Accept": "application/vnd.github.v3+json",
            }
            req = urllib.request.Request(url, headers=headers, method="HEAD")
            try:
                response = urllib.request.urlopen(req, timeout=self.timeout)
                return dict(response.headers)
            except urllib.error.HTTPError as e:
                body = (
                    e.read().decode("utf-8", errors="replace") if e.fp else ""
                )
                self._handle_response_status(url, e.code, body)
                return {}


# ============================================================================
# Cache Manager
# ============================================================================


class CacheManager:
    """
    File-based cache for downloaded content with TTL-based expiration
    and optional size limit.

    Each cached entry is stored as a binary file whose name is derived
    from a SHA-256 hash of the URL and version string. Metadata (URL,
    version, timestamp, size) is kept in a JSON index file alongside
    the cached blobs.

    Expiration
    ----------
    Entries older than ``ttl_seconds`` are considered stale.
    ``get()`` returns ``None`` for stale entries and removes them
    from the index automatically. Explicit cleanup is available via
    ``clear_expired()``.

    Size limit
    ----------
    When ``max_size_mb`` is set, the total cache size is checked after
    each ``put()``. If the limit is exceeded, the oldest entries (by
    timestamp) are evicted until the total size falls below the limit.

    Atomicity
    ---------
    The index is written atomically by writing to a temporary file and
    renaming it over the target path. Individual cache blobs are written
    directly; partial writes are possible on crash but do not corrupt
    the index.

    Parameters
    ----------
    cache_dir : Path or None
        Directory for cached files. Defaults to
        ``~/.cache/stdlib-installer``.
    ttl_seconds : int
        Time-to-live in seconds. Defaults to 86400 (24 hours).
    max_size_mb : int or None
        Maximum total cache size in megabytes. ``None`` disables the
        limit. Defaults to 500.

    Examples
    --------
    >>> cache = CacheManager(ttl_seconds=3600, max_size_mb=100)
    >>> cache.put("https://example.com/f.py", "3.13", b"print(1)")
    >>> cached = cache.get("https://example.com/f.py", "3.13")
    >>> cache.stats()
    {'entry_count': 1, 'total_size_bytes': 8, ...}
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl_seconds: int = 86400,
        max_size_mb: Optional[int] = 500,
    ) -> None:
        self.cache_dir = cache_dir or Path.home() / ".cache" / "stdlib-installer"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ttl = ttl_seconds
        self.max_size_bytes = max_size_mb * 1024 * 1024 if max_size_mb else None

        self._index_path = self.cache_dir / "index.json"
        self._index: Dict[str, Dict[str, Any]] = self._load_index()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the cache index from disk.

        If the index file does not exist, returns an empty dict.
        If the file contains invalid JSON, logs a warning, backs up
        the corrupted file by renaming it, and returns an empty dict.

        Returns
        -------
        dict
            Cache index mapping hex keys to metadata dicts with keys
            ``"timestamp"``, ``"size"``, ``"url"``, and ``"version"``.
        """
        if not self._index_path.exists():
            return {}

        try:
            raw = self._index_path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Cannot read cache index, starting fresh")
            return {}

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Cache index corrupted, backing up and rebuilding")
            self._backup_corrupted_index(raw)
            return {}

        if not isinstance(data, dict):
            logger.warning("Cache index root is not a dict, rebuilding")
            self._backup_corrupted_index(raw)
            return {}

        return data

    def _save_index(self) -> None:
        """
        Write the in-memory index to disk atomically.

        Writes to a ``.tmp`` file first, then renames it over the
        actual index path. If the rename fails, the temporary file
        is cleaned up.
        """
        temp_path = self._index_path.with_suffix(".tmp")
        try:
            temp_path.write_text(
                json.dumps(self._index, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            temp_path.replace(self._index_path)
        except OSError:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _backup_corrupted_index(self, raw_content: str) -> None:
        """
        Rename a corrupted index file for forensic inspection.

        The file is renamed with a ``.corrupted`` suffix. If that name
        already exists, a numeric counter is appended until a unique
        name is found.

        Parameters
        ----------
        raw_content : str
            The raw text of the corrupted index, preserved as-is.
        """
        backup = self._index_path.with_suffix(".corrupted")
        counter = 1
        while backup.exists():
            backup = self._index_path.with_suffix(f".corrupted.{counter}")
            counter += 1
        try:
            backup.write_text(raw_content, encoding="utf-8")
            logger.info(f"Corrupted index backed up to {backup}")
        except OSError as e:
            logger.error(f"Failed to backup corrupted index: {e}")

    # ------------------------------------------------------------------
    # Key generation and path resolution
    # ------------------------------------------------------------------

    def _cache_key(self, url: str, version: str) -> str:
        """
        Generate a deterministic hex key from a URL and version string.

        Uses SHA-256 truncated to 32 hex characters.

        Parameters
        ----------
        url : str
            Resource URL.
        version : str
            Version identifier (e.g., ``"3.13"``).

        Returns
        -------
        str
            32-character lowercase hex string.
        """
        combined = f"{url}#{version}".encode("utf-8")
        return hashlib.sha256(combined).hexdigest()[:32]

    def _entry_path(self, key: str) -> Path:
        """
        Return the filesystem path for a cache entry blob.

        Parameters
        ----------
        key : str
            Cache key from ``_cache_key``.

        Returns
        -------
        Path
        """
        return self.cache_dir / key

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def get(self, url: str, version: str) -> Optional[bytes]:
        """
        Retrieve a cached entry if it exists and has not expired.

        Parameters
        ----------
        url : str
            Resource URL.
        version : str
            Version identifier.

        Returns
        -------
        bytes or None
            Cached content, or ``None`` if not found, expired, or
            the blob file is missing.
        """
        key = self._cache_key(url, version)
        entry = self._index.get(key)
        if entry is None:
            return None

        age = time.time() - entry["timestamp"]
        if age > self.ttl:
            logger.debug(f"Cache entry expired: {key}")
            self._remove_entry(key)
            return None

        blob_path = self._entry_path(key)
        if not blob_path.exists():
            self._remove_entry(key)
            return None

        try:
            return blob_path.read_bytes()
        except OSError as e:
            logger.warning(f"Failed to read cache blob {key}: {e}")
            return None

    def put(self, url: str, version: str, content: bytes) -> None:
        """
        Store content in the cache.

        Parameters
        ----------
        url : str
            Resource URL.
        version : str
            Version identifier.
        content : bytes
            Raw bytes to cache.
        """
        key = self._cache_key(url, version)
        blob_path = self._entry_path(key)

        try:
            blob_path.write_bytes(content)
        except OSError as e:
            logger.warning(f"Failed to write cache blob {key}: {e}")
            return

        self._index[key] = {
            "timestamp": time.time(),
            "size": len(content),
            "url": url,
            "version": version,
        }
        self._save_index()

        if self.max_size_bytes is not None:
            self._enforce_size_limit()

    def _remove_entry(self, key: str) -> None:
        """
        Delete a cache entry's blob and index record.

        Parameters
        ----------
        key : str
            Cache key.
        """
        blob_path = self._entry_path(key)
        try:
            blob_path.unlink(missing_ok=True)
        except OSError:
            pass
        self._index.pop(key, None)
        self._save_index()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def _enforce_size_limit(self) -> None:
        """
        Evict oldest entries until total cache size is under the limit.

        Entries are sorted by timestamp ascending. Each entry is removed
        and the total size recalculated until the limit is satisfied or
        no entries remain.
        """
        if self.max_size_bytes is None:
            return

        total = sum(e.get("size", 0) for e in self._index.values())
        if total <= self.max_size_bytes:
            return

        sorted_keys = sorted(
            self._index.keys(),
            key=lambda k: self._index[k].get("timestamp", 0),
        )
        for key in sorted_keys:
            if total <= self.max_size_bytes:
                break
            entry_size = self._index[key].get("size", 0)
            self._remove_entry(key)
            total -= entry_size

    def clear(self) -> int:
        """
        Remove all cached entries.

        Returns
        -------
        int
            Number of entries removed.
        """
        count = len(self._index)
        for key in list(self._index.keys()):
            self._remove_entry(key)
        return count

    def clear_expired(self) -> int:
        """
        Remove only expired cache entries.

        Returns
        -------
        int
            Number of expired entries removed.
        """
        cutoff = time.time() - self.ttl
        expired = [
            key
            for key, entry in self._index.items()
            if entry.get("timestamp", 0) < cutoff
        ]
        for key in expired:
            self._remove_entry(key)
        return len(expired)

    def stats(self) -> Dict[str, Any]:
        """
        Return cache usage statistics.

        Returns
        -------
        dict
            Keys: ``"entry_count"``, ``"total_size_bytes"``,
            ``"total_size_mb"``, ``"cache_dir"``, ``"ttl_seconds"``.
        """
        total = sum(e.get("size", 0) for e in self._index.values())
        return {
            "entry_count": len(self._index),
            "total_size_bytes": total,
            "total_size_mb": round(total / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
            "ttl_seconds": self.ttl,
        }


# ============================================================================
# Downloader
# ============================================================================


class Downloader:
    """
    Coordinates HTTP requests, caching, and recursive file downloads
    from the CPython GitHub repository.

    Uses ``HttpClient`` for transport and ``CacheManager`` for local
    caching. Returns parsed ``GitHubItem`` and ``FetchResult`` objects
    rather than raw dicts or lists.

    Parameters
    ----------
    http_client : HttpClient or None
        Pre-configured HTTP client. One is created with defaults if
        ``None``.
    cache_manager : CacheManager or None
        Pre-configured cache manager. One is created with defaults if
        ``None``.

    Examples
    --------
    >>> dl = Downloader()
    >>> result = dl.fetch_tree("json", "3.13")
    >>> for item in result:
    ...     content = dl.download_item(item, "3.13")
    """

    def __init__(
        self,
        http_client: Optional[HttpClient] = None,
        cache_manager: Optional[CacheManager] = None,
    ) -> None:
        self.http = http_client or HttpClient()
        self.cache = cache_manager or CacheManager()

    # ------------------------------------------------------------------
    # Tree fetching
    # ------------------------------------------------------------------

    def fetch_tree(self, name: str, version: str) -> FetchResult:
        """
        Fetch the directory tree or single-file metadata for a standard
        library module from the GitHub Contents API.
    
        Constructs the appropriate API URL from the module name and
        version, issues a GET request, and wraps the parsed response in a
        ``FetchResult`` for uniform iteration regardless of whether the
        target is a single file or a directory.
    
        The GitHub Contents API returns two different shapes depending on
        what the path points to:
    
        * **Single file**: A JSON object with ``"type": "file"``. This
          occurs for modules like ``token.py``, ``pathlib.py``, or
          ``typing.py`` that live directly under ``Lib/`` as standalone
          ``.py`` files.
        * **Directory**: A JSON array of objects, each with ``"type"``
          of ``"file"`` or ``"dir"``. This occurs for packages like
          ``json/``, ``email/``, or ``xml/`` that are organised as
          directories containing multiple ``.py`` files and potentially
          nested subdirectories.
    
        This method normalises both cases into a ``FetchResult`` so that
        downstream code can iterate over items without branching on type.
    
        Parameters
        ----------
        name : str
            Dotted module name as used in Python imports. Dots are
            converted to forward slashes for the API path. Examples:
    
            * ``"json"`` → fetches ``Lib/json/``
            * ``"xml.etree"`` → fetches ``Lib/xml/etree/``
            * ``"token"`` → fetches ``Lib/token.py`` (single file)
    
            The name must not contain path traversal sequences (``..``)
            or start/end with a dot. Validation is performed upstream by
            the ``Installer``, not by this method.
        version : str
            CPython version tag, branch name, or commit SHA to fetch from.
            Appended as the ``?ref=`` query parameter. Examples:
    
            * ``"3.13"`` — a branch name
            * ``"v3.12.0"`` — a tag
            * ``"main"`` — the development branch
            * ``"abc123..."`` — a full commit SHA
    
        Returns
        -------
        FetchResult
            A normalised container that can be iterated over uniformly:
    
            * ``result.is_single_file`` is ``True`` for single-file modules,
              ``False`` for directories.
            * ``result.single_item`` holds the ``GitHubItem`` for single
              files; ``None`` for directories.
            * ``result.items`` holds the list of ``GitHubItem`` entries for
              directories; empty for single files.
            * ``len(result)`` returns the number of items.
            * ``for item in result`` works for both cases.
    
        Raises
        ------
        PackageNotFoundError
            Raised when the API returns HTTP 404, meaning the module name
            does not correspond to any file or directory under ``Lib/`` in
            the given version. This is raised after inspecting the response
            body for a ``"message": "Not Found"`` pattern.
        GitHubAPIError
            Raised when the API returns a non-2xx status code other than
            404, or when the response body is not a dict or list. This
            covers rate limiting (HTTP 429), server errors (HTTP 5xx), and
            unexpected response shapes.
        NetworkError
            Raised on connection failures, DNS resolution errors, TLS
            handshake failures, or request timeouts. The underlying
            exception is preserved in the ``original_error`` attribute
            for debugging.
    
        Notes
        -----
        The URL is constructed from scratch rather than following redirects
        or using URLs returned by previous API calls. This avoids the
        double-query-string bug that can occur when the API returns URLs
        that already contain ``?ref=...``.
    
        The method does **not** recurse into subdirectories. It only
        returns the immediate children of the requested path. Recursive
        descent is handled by ``download_recursive`` which calls
        ``fetch_subtree`` for each directory it encounters.
    
        Examples
        --------
        Fetching a directory (package):
    
        >>> dl = Downloader()
        >>> result = dl.fetch_tree("json", "3.13")
        >>> result.is_single_file
        False
        >>> len(result)
        5
        >>> for item in result:
        ...     print(item.name, item.item_type)
        __init__.py file
        decoder.py file
        encoder.py file
        scanner.py file
        tool.py file
    
        Fetching a single-file module:
    
        >>> result = dl.fetch_tree("token", "3.13")
        >>> result.is_single_file
        True
        >>> result.single_item.name
        'token.py'
        >>> result.single_item.is_file
        True
        >>> result.single_item.size
        24567
    
        Handling a non-existent module:
    
        >>> try:
        ...     dl.fetch_tree("nonexistent_module", "3.13")
        ... except PackageNotFoundError as e:
        ...     print(e.package_name)
        nonexistent_module
        """
        # Convert Python dotted name to filesystem path.
        # "xml.etree" → "xml/etree"
        package_path = name.replace(".", "/")
    
        # Build the API URL from scratch.
        # We do NOT use URLs from previous API responses because those
        # already contain ?ref=..., and appending another would produce
        # malformed double-query-string URLs like:
        #   .../contents/Lib/json?ref=3.13?ref=3.13
        url = (
            f"{HttpClient.API_BASE}/contents/Lib/{package_path}"
            f"?ref={version}"
        )
    
        logger.debug("Fetching tree from: %s", url)
    
        # Issue the request. get_json() handles HTTP errors by raising
        # NetworkError or GitHubAPIError as appropriate.
        data = self.http.get_json(url)
    
        # ------------------------------------------------------------------
        # Case 1: Single file
        # The API returns a dict with "type": "file" when the path points
        # directly to a .py file (e.g., Lib/token.py).
        # ------------------------------------------------------------------
        if isinstance(data, dict):
            # Check for the "Not Found" message that GitHub returns as a
            # 200-style response body for some edge cases (though normally
            # a 404 status code is used — we handle both).
            if data.get("message") == "Not Found":
                raise PackageNotFoundError(
                    package_name=name,
                    version=version,
                )
    
            # If the response is a dict but not a file, something is wrong
            # with our assumptions. Raise an explicit error rather than
            # failing mysteriously downstream.
            if data.get("type") != "file":
                raise GitHubAPIError(
                    url=url,
                    status_code=200,
                    response_body=json.dumps(data),
                )
    
            # Wrap the raw dict in a GitHubItem for type-safe access.
            item = GitHubItem.from_api_response(data)
            logger.debug(
                "Fetched single file: %s (%d bytes)", item.name, item.size
            )
            return FetchResult(is_single_file=True, single_item=item)
    
        # ------------------------------------------------------------------
        # Case 2: Directory
        # The API returns a list when the path points to a directory
        # (e.g., Lib/json/).
        # ------------------------------------------------------------------
        if isinstance(data, list):
            items = [GitHubItem.from_api_response(d) for d in data]
            logger.debug(
                "Fetched directory with %d items: %s",
                len(items),
                package_path,
            )
            return FetchResult(is_single_file=False, items=items)
    
        # ------------------------------------------------------------------
        # Case 3: Unexpected response shape
        # The API returned something that is neither a dict nor a list.
        # This should never happen with the GitHub API, but we guard
        # against it for defensive robustness.
        # ------------------------------------------------------------------
        raise GitHubAPIError(
            url=url,
            status_code=200,
            response_body=(
                f"Unexpected response type: {type(data).__name__}. "
                f"Raw data: {json.dumps(data)[:500]}"
            ),
        )

    def fetch_subtree(
        self, subtree_url: str, version: str
    ) -> List[GitHubItem]:
        """
        Fetch the contents of a subdirectory using its direct API URL.
    
        The GitHub Contents API returns an ``url`` field for each directory
        item that already includes a ``?ref=...`` query parameter. This
        method strips any existing query string and rebuilds the URL with
        the correct version to prevent double query parameters (which cause
        HTTP 404 errors like ``tests?ref=3.13?ref=3.13``).
    
        The method also validates that the API response is a list. If the
        response is a dict (which can happen when GitHub redirects or the
        URL is malformed), an explicit error is raised rather than allowing
        a confusing downstream failure.
    
        Parameters
        ----------
        subtree_url : str
            Full GitHub Contents API URL for the subdirectory. This is the
            value of the ``url`` key from a directory-type ``GitHubItem``.
            It typically has the form:
            ``https://api.github.com/repos/python/cpython/contents/Lib/json/tests?ref=3.13``
        version : str
            CPython version tag or branch to use in the ``?ref=`` parameter
            (e.g., ``"3.13"``, ``"v3.12.0"``, ``"main"``).
    
        Returns
        -------
        list of GitHubItem
            Parsed child items of the subdirectory. Each item represents
            either a file or a nested subdirectory. The list may be empty
            if the directory contains no entries.
    
        Raises
        ------
        GitHubAPIError
            If the API response is not a JSON array (expected for
            directories), or if the HTTP request fails with a non-2xx
            status code.
        NetworkError
            On connection failures, timeouts, or TLS errors.
    
        Notes
        -----
        The query string is stripped from ``subtree_url`` before rebuilding
        because the URL returned by the API already contains a ``ref``
        parameter from the original request. Using it verbatim and appending
        a second ``?ref=...`` would produce malformed URLs.
    
        Examples
        --------
        >>> item = GitHubItem(
        ...     name="tests",
        ...     path="Lib/json/tests",
        ...     item_type="dir",
        ...     download_url=None,
        ...     api_url="https://api.github.com/repos/python/cpython/contents/Lib/json/tests?ref=3.13",
        ...     sha="abc123",
        ...     size=0,
        ... )
        >>> children = downloader.fetch_subtree(item.api_url, "3.13")
        >>> for child in children:
        ...     print(child.name)
        test_decode.py
        test_encode.py
        ...
        """
        # Strip any existing query string to prevent doubling.
        # urllib.parse is available in the stdlib but we keep it simple
        # with a string split since we only need to remove ?ref=... once.
        base_url = subtree_url.split("?")[0]
    
        # Rebuild the URL with exactly one ?ref= parameter.
        # Using urllib.parse.urlencode would be more robust, but the only
        # query parameter we ever send is 'ref', and its value (a version
        # tag like "3.13" or "main") does not contain special characters
        # that require encoding.
        url = f"{base_url}?ref={version}"
    
        logger.debug("Fetching subtree from: %s", url)
    
        # Make the request and expect a JSON array.
        data = self.http.get_json(url)
    
        if not isinstance(data, list):
            raise GitHubAPIError(
                url=url,
                status_code=200,
                response_body=(
                    f"Expected a JSON array for directory contents, "
                    f"got {type(data).__name__}: {json.dumps(data)[:500]}"
                ),
            )
    
        # Convert raw dicts to GitHubItem instances for type safety
        # downstream.
        items = [GitHubItem.from_api_response(d) for d in data]
    
        logger.debug(
            "Fetched %d items from subtree %s", len(items), base_url
        )
    
        return items

    # ------------------------------------------------------------------
    # Raw file download
    # ------------------------------------------------------------------

    def download_raw_file(self, lib_path: str, version: str) -> bytes:
        """
        Download a single file from GitHub's raw content CDN.

        The ``lib_path`` must be relative to the ``Lib/`` directory
        (e.g., ``"json/__init__.py"``). It is concatenated with the
        raw base URL and version to form the full download URL.

        Results are cached using ``CacheManager``.

        Parameters
        ----------
        lib_path : str
            Path relative to the ``Lib/`` directory.
        version : str
            CPython version tag or branch.

        Returns
        -------
        bytes
            Raw file content.

        Raises
        ------
        NetworkError
            On download failures.
        PackageNotFoundError
            If the raw URL returns 404.
        """
        url = f"{HttpClient.RAW_BASE}/{version}/Lib/{lib_path}"

        cached = self.cache.get(url, version)
        if cached is not None:
            logger.debug(f"Cache hit: {lib_path}")
            return cached

        logger.debug(f"Downloading: {url}")
        content = self.http.get(url)
        self.cache.put(url, version, content)
        return content

    def download_item(self, item: GitHubItem, version: str) -> bytes:
        """
        Download the content of a single ``GitHubItem`` that represents
        a file.

        Uses the item's ``lib_path`` property to construct the raw URL.

        Parameters
        ----------
        item : GitHubItem
            A file item (``item.is_file`` must be ``True``).
        version : str
            CPython version tag or branch.

        Returns
        -------
        bytes
            Raw file content.

        Raises
        ------
        ValueError
            If ``item`` is not a file.
        NetworkError
            On download failures.
        """
        if not item.is_file:
            raise ValueError(f"Cannot download non-file item: {item.name}")
        return self.download_raw_file(item.lib_path, version)

    # ------------------------------------------------------------------
    # Recursive download
    # ------------------------------------------------------------------

    def download_recursive(
        self,
        result: FetchResult,
        target_dir: Path,
        version: str,
    ) -> List[Path]:
        """
        Recursively download all files described by a ``FetchResult``
        into a local directory.

        Creates subdirectories as needed for nested packages. Each file
        is written atomically by writing to a temporary name and then
        renaming.

        Parameters
        ----------
        result : FetchResult
            The result from ``fetch_tree`` or equivalent.
        target_dir : Path
            Local directory into which files are written. Created if
            it does not exist.
        version : str
            CPython version tag or branch for download URLs.

        Returns
        -------
        list of Path
            Absolute paths of all files successfully written.

        Raises
        ------
        NetworkError
            On any download failure.
        OSError
            On filesystem write errors.
        """
        downloaded: List[Path] = []

        # Single file
        if result.is_single_file and result.single_item is not None:
            target_dir.mkdir(parents=True, exist_ok=True)
            file_path = target_dir / result.single_item.name
            content = self.download_item(result.single_item, version)
            self._atomic_write(file_path, content)
            return [file_path]

        # Directory
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in result.items:
            if item.is_file:
                file_path = target_dir / item.name
                content = self.download_item(item, version)
                self._atomic_write(file_path, content)
                downloaded.append(file_path)

            elif item.is_dir:
                subdir = target_dir / item.name
                if item.api_url is None:
                    logger.warning(
                        f"Directory '{item.name}' has no API URL, skipping"
                    )
                    continue
                subtree_items = self.fetch_subtree(item.api_url, version)
                subtree_result = FetchResult(
                    is_single_file=False,
                    items=subtree_items,
                )
                sub_downloaded = self.download_recursive(
                    subtree_result, subdir, version
                )
                downloaded.extend(sub_downloaded)

        return downloaded

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        """
        Write bytes to a file atomically using a temporary file and rename.

        Parameters
        ----------
        path : Path
            Target file path.
        content : bytes
            Data to write.

        Raises
        ------
        OSError
            If the write or rename fails.
        """
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(content)
        tmp.replace(path)

    def verify_checksum(
        self,
        file_path: Path,
        expected_hash: str,
        algorithm: str = "sha256",
    ) -> bool:
        """
        Verify a downloaded file's integrity against an expected hash.

        Parameters
        ----------
        file_path : Path
            Path to the file to verify.
        expected_hash : str
            Expected hexadecimal digest string.
        algorithm : str
            Hash algorithm name accepted by ``hashlib.new()``.
            Defaults to ``"sha256"``.

        Returns
        -------
        bool
            ``True`` if the computed hash matches.

        Raises
        ------
        ChecksumVerificationError
            If the file does not exist, cannot be read, or the hash
            does not match.
        """
        if not file_path.exists():
            raise ChecksumVerificationError(
                file_name=file_path.name,
                expected_hash=expected_hash,
                actual_hash="FILE_NOT_FOUND",
            )

        try:
            hasher = hashlib.new(algorithm)
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            actual_hash = hasher.hexdigest()
        except OSError as e:
            raise ChecksumVerificationError(
                file_name=file_path.name,
                expected_hash=expected_hash,
                actual_hash=f"READ_ERROR: {e}",
            )

        if actual_hash != expected_hash:
            raise ChecksumVerificationError(
                file_name=file_path.name,
                expected_hash=expected_hash,
                actual_hash=actual_hash,
            )

        return True
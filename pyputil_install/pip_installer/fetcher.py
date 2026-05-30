"""
Network Fetching & Resource Acquisition Module.

Downloads pip installation resources (get-pip.py, wheel files) from
remote hosts with comprehensive error handling, automatic retries,
mirror fallback, integrity verification, caching, and transparent
decompression of gzip-encoded responses.

Why this module exists:
    - ``urllib.request.urlopen`` throws ``URLError``, ``HTTPError``,
      ``socket.timeout``, ``SSL: CERTIFICATE_VERIFY_FAILED``, and a
      dozen other exceptions—each requiring different handling.
    - Corporate environments commonly block ``bootstrap.pypa.io`` but
      allow ``github.com``; multi-host fallback is essential.
    - Transient network failures (packet loss, DNS blips, server
      restart) cause spurious failures unless retries with exponential
      backoff are implemented.
    - Some CDNs and mirrors serve resources with ``Content-Encoding: gzip``
      even when the resource itself is not compressed at the application
      layer. The raw bytes received contain gzip magic bytes (``\\x1f\\x8b``)
      instead of the expected content (Python source, ZIP archive).
      Without transparent decompression, these downloads produce
      ``SyntaxError: Non-UTF-8 code`` or ``BadZipFile`` errors.
    - Downloading a corrupted file (truncated transfer, proxy
      interference, storage error) produces a broken pip installation
      that is harder to debug than a download failure with a clear
      checksum error.
    - Some environments require TLS certificate paths or custom CA
      bundles; the fetch logic must respect ``REQUESTS_CA_BUNDLE``
      and ``SSL_CERT_FILE`` environment variables.
    - Downloading the same resource repeatedly wastes bandwidth and
      time; caching with ETag/Last-Modified validation avoids this.
    - Offline mode needs to locate locally cached files that were
      previously downloaded; the cache layer bridges online and
      offline operation seamlessly.

Integration with extractor.py:
    - After reading the HTTP response body, the raw bytes are inspected
      for compression magic bytes using ``extractor.is_compressed()``.
    - If compression is detected, ``extractor.decompress_bytes()``
      transparently decompresses the data before writing to disk.
    - This prevents the ``SyntaxError: Non-UTF-8 code starting with
      '\\x8b'`` error that occurs when get-pip.py is served with
      ``Content-Encoding: gzip`` by certain CDN configurations.
    - Decompression is applied before integrity verification, ensuring
      the SHA256 hash is computed against the actual resource content,
      not the compressed transport encoding.

Warnings
--------
- This module makes real network requests. Tests should use mocking
  or a local HTTP server.
- The default timeout is 30 seconds per attempt. On very slow
  connections (satellite, 2G mobile), increase ``timeout`` when
  constructing ``ResourceFetcher``.
- Integrity verification uses SHA256 hashes embedded in PyPI's JSON
  API. If PyPI changes its API format, the ``_discover_wheel_url``
  method will need updating.
- Proxy support reads from ``HTTP_PROXY``, ``HTTPS_PROXY``, and
  ``NO_PROXY`` environment variables. Authenticated proxies
  (``http://user:pass@proxy:port``) are supported but credentials
  in environment variables are a security risk in shared systems.
- Decompression of gzip responses adds CPU overhead. For large
  resources, this may be noticeable on low-power devices.

Examples
--------
Download the latest get-pip.py script:

    >>> from fetcher import ResourceFetcher
    >>> fetcher = ResourceFetcher()
    >>> success, path, metadata = fetcher.fetch_get_pip()
    >>> success
    True
    >>> path.name
    'get-pip.py'
    >>> metadata.size_bytes > 1000000
    True

Download a specific pip wheel:

    >>> fetcher = ResourceFetcher()
    >>> success, path, metadata = fetcher.fetch_pip_wheel("21.3.1")
    >>> success
    True
    >>> path.suffix
    '.whl'

Download with a custom cache directory:

    >>> fetcher = ResourceFetcher(cache_dir="/tmp/pip_cache")
    >>> success, path, _ = fetcher.fetch_pip_wheel("23.0.1")
    >>> path.parent == Path("/tmp/pip_cache")
    True

Handle download failure with fallback mirrors:

    >>> fetcher = ResourceFetcher()
    >>> fetcher.add_mirror("https://my-internal-mirror.example.com/pypi")
    >>> success, path, _ = fetcher.fetch_pip_wheel("23.0.1")
    >>> if not success:
    ...     print("All mirrors exhausted")

Verify an already-downloaded file:

    >>> fetcher = ResourceFetcher()
    >>> is_valid, reason = fetcher.verify_file_integrity(
    ...     Path("./pip-21.3.1-py3-none-any.whl"),
    ...     expected_sha256="abc123...",
    ... )
    >>> is_valid
    True

Check connectivity before attempting downloads:

    >>> fetcher = ResourceFetcher()
    >>> fetcher.check_connectivity()
    True
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import ssl
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

# Integration with extractor module for transparent decompression
from .extractor import decompress_bytes, is_compressed, get_compression_type


# ---------------------------------------------------------------------------
# Public Enumerations
# ---------------------------------------------------------------------------


class FetchStatus(Enum):
    """
    Outcome of a resource fetch operation.

    Why an enum instead of a boolean:
        - A boolean loses information: "False" could mean network down,
          file not found, checksum mismatch, or disk full. Each requires
          different recovery.
        - The orchestrator can branch on specific statuses: CACHED skips
          re-download, MIRROR_USED logs the fallback, CHECKSUM_FAILED
          triggers a re-fetch.
        - Status values can be serialized for telemetry/debugging without
          parsing error strings.

    Examples
    --------
    >>> status = FetchStatus.SUCCESS
    >>> status.name
    'SUCCESS'
    >>> status == FetchStatus.CACHED
    False
    """

    SUCCESS = auto()
    """Resource downloaded and verified successfully from primary URL."""

    CACHED = auto()
    """Resource retrieved from local cache; network was not used."""

    MIRROR_USED = auto()
    """Primary URL failed; resource was fetched from a mirror."""

    RETRIED = auto()
    """Download succeeded after one or more retry attempts."""

    CHECKSUM_FAILED = auto()
    """Download completed but file integrity check failed."""

    NOT_FOUND = auto()
    """Server returned HTTP 404; resource does not exist at any URL."""

    TIMEOUT = auto()
    """All download attempts exceeded the configured timeout."""

    NETWORK_ERROR = auto()
    """Network is unreachable; no connection could be established."""

    PERMISSION_DENIED = auto()
    """Cannot write to cache or destination directory."""

    DISK_FULL = auto()
    """Insufficient disk space to store the downloaded resource."""

    DECOMPRESSED = auto()
    """
    Resource was downloaded with transport-layer compression (e.g., gzip)
    and transparently decompressed before storage. The cached file contains
    the uncompressed content.
    """


class MirrorStrategy(Enum):
    """
    Determines the order in which mirrors are tried.

    Why strategy matters:
        - SEQUENTIAL tries mirrors in order, respecting user preference
          (corporate mirror first, public PyPI as fallback).
        - RANDOM distributes load across mirrors and avoids consistently
          hitting a single slow mirror.
        - PRIMARY_FIRST always tries the primary URL first, then mirrors;
          good when mirrors are known to be slower or less reliable.
        - PARALLEL fires requests to all mirrors simultaneously and uses
          the first successful response; fastest but bandwidth-intensive.

    Examples
    --------
    >>> strategy = MirrorStrategy.PRIMARY_FIRST
    >>> strategy.name
    'PRIMARY_FIRST'
    """

    SEQUENTIAL = auto()
    """Try mirrors in the order they were added."""

    RANDOM = auto()
    """Randomize mirror order on each fetch attempt."""

    PRIMARY_FIRST = auto()
    """Always try the default URL first, then mirrors sequentially."""

    PARALLEL = auto()
    """Attempt all mirrors simultaneously; use first responder."""


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------


@dataclass
class FetchMetadata:
    """
    Metadata about a fetched resource, independent of the resource itself.

    Why separate metadata from the file path:
        - The file may be cached; metadata tells us when it was downloaded
          and whether it's still fresh.
        - Integrity verification results (hashes) belong to metadata,
          not to the file path.
        - Enables cache freshness checks without reading the file.
        - The ``decompressed`` field records whether transparent
          decompression was applied during this fetch.

    Attributes
    ----------
    url : str
        The final URL the resource was fetched from (may differ from
        the requested URL if redirects or mirrors were used).
    status : FetchStatus
        Outcome of the fetch operation.
    size_bytes : int
        Size of the downloaded file in bytes (after decompression, if
        applicable). 0 if fetch failed.
    sha256 : Optional[str]
        SHA256 hex digest of the file contents, if verification was
        performed or the hash was provided by the server.
    etag : Optional[str]
        HTTP ETag header value from the response, used for conditional
        requests on subsequent fetches.
    last_modified : Optional[str]
        HTTP Last-Modified header value from the response.
    fetched_at : str
        ISO 8601 timestamp of when the fetch completed.
    retry_count : int
        Number of retries performed before success (0 if first attempt
        succeeded).
    mirror_used : Optional[str]
        If a mirror was used, the mirror URL. None otherwise.
    decompressed : bool
        ``True`` if the response body was transparently decompressed
        (e.g., gzip Content-Encoding) before storage.
    original_size_bytes : int
        Size of the raw response body before decompression. Equal to
        ``size_bytes`` if no decompression was applied.
    error_message : Optional[str]
        If the fetch failed, a human-readable error description.
        None on success.

    Examples
    --------
    >>> meta = FetchMetadata(
    ...     url="https://bootstrap.pypa.io/get-pip.py",
    ...     status=FetchStatus.SUCCESS,
    ...     size_bytes=2100000,
    ...     sha256="abc123def456...",
    ...     etag=None,
    ...     last_modified="Wed, 15 Jan 2025 10:00:00 GMT",
    ...     fetched_at="2025-01-16T08:30:00",
    ...     retry_count=0,
    ...     mirror_used=None,
    ...     decompressed=False,
    ...     original_size_bytes=2100000,
    ...     error_message=None,
    ... )
    >>> meta.status == FetchStatus.SUCCESS
    True
    """

    url: str
    status: FetchStatus
    size_bytes: int
    sha256: Optional[str] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    retry_count: int = 0
    mirror_used: Optional[str] = None
    decompressed: bool = False
    original_size_bytes: int = 0
    error_message: Optional[str] = None


@dataclass
class CacheEntry:
    """
    Represents a cached resource with its validation metadata.

    Why cache entries exist:
        - Cache invalidation requires knowing when a resource was cached
          and whether it has a known expiration.
        - ETag-based validation avoids re-downloading unchanged resources.
        - Stale entries (beyond ``max_age``) trigger re-fetch but the
          old file is kept until the new one is verified.

    Attributes
    ----------
    file_path : Path
        Absolute path to the cached file.
    metadata : FetchMetadata
        Metadata from when this resource was originally fetched.
    cached_at : float
        POSIX timestamp of when this entry was created.
    max_age_seconds : Optional[int]
        Maximum age in seconds before this entry is considered stale.
        None means no automatic expiration (manual invalidation only).
    """

    file_path: Path
    metadata: FetchMetadata
    cached_at: float
    max_age_seconds: Optional[int] = 86400  # 24 hours default


# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------

# Primary URLs for pip bootstrapping resources.
# Why these specific hosts:
#   - bootstrap.pypa.io is the official source for get-pip.py, managed
#     by the Python Packaging Authority.
#   - pypi.org/simple/pip/ is the PEP 503 simple repository index for
#     discovering pip wheel URLs.
#   - files.pythonhosted.org is PyPI's CDN for wheel files; it is
#     geographically distributed and highly available.
_DEFAULT_GET_PIP_URL: str = "https://bootstrap.pypa.io/get-pip.py"
_DEFAULT_PYPI_SIMPLE_URL: str = "https://pypi.org/simple/pip/"
_DEFAULT_WHEEL_BASE_URL: str = "https://files.pythonhosted.org/packages/"
_DEFAULT_PYTHON_PIP_URL: str = "https://bootstrap.pypa.io/{py_version}/get-pip.py"
_DEFAULT_PIP_VERSION_URL: str = "https://bootstrap.pypa.io/pip/{pip_version}/get-pip.py"

# Default mirrors tried when primary URLs fail.
# Why these mirrors:
#   - Raw GitHub provides get-pip.py directly from the pip repository.
#   - Tsinghua and Aliyun are major mirrors in China, useful when GFW
#     blocks direct PyPI access.
#   - The order prioritizes official sources over third-party mirrors
#     to minimize supply chain risk.
_DEFAULT_MIRRORS: List[str] = [
    "https://raw.githubusercontent.com/pypa/pip/main/public/get-pip.py",
    "https://pypi.tuna.tsinghua.edu.cn/simple/",
    "https://mirrors.aliyun.com/pypi/simple/",
]

# HTTP headers sent with every request.
# Why these specific headers:
#   - User-Agent identifies the tool to servers; some CDNs block
#     default Python urllib User-Agent strings as bot traffic.
#   - Accept prioritizes binary/stream responses over HTML.
#   - Accept-Encoding is intentionally NOT set here. urllib adds
#     "Accept-Encoding: identity" by default, which requests
#     uncompressed responses. However, some CDNs ignore this and
#     serve gzip anyway. The extractor module handles this case.
_DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": "pip-rescuer/1.0 (Python-urllib; +https://github.com/example/pip-rescuer)",
    "Accept": "application/octet-stream, */*;q=0.8",
    "Cache-Control": "no-cache",
}

# Maximum file size in bytes for downloaded resources.
# get-pip.py is ~2MB; pip wheels are ~3MB. 50MB is a generous
# upper bound that prevents downloading something that is clearly
# not a pip resource (misconfigured proxy HTML page, etc.).
_MAX_DOWNLOAD_SIZE: int = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------------------
# Core Fetcher Class
# ---------------------------------------------------------------------------


class ResourceFetcher:
    """
    Downloads pip installation resources with retry, mirror fallback,
    caching, integrity verification, and transparent decompression.

    Why a class instead of module-level functions:
        - Configuration (timeout, mirrors, cache directory, headers)
          is shared across multiple fetch operations and should not
          need to be re-specified each time.
        - The cache index is stateful: it tracks what has been downloaded
          and when, enabling cache hits across multiple method calls.
        - Multiple fetcher instances can target different cache directories
          or use different mirror strategies for different use cases.
        - Enables mocking in tests: replace the entire fetcher instance
          rather than patching urllib at the module level.

    Transparent decompression:
        - After reading the HTTP response body, the raw bytes are checked
          for compression magic bytes (gzip: ``\\x1f\\x8b``, deflate:
          ``\\x78``, bzip2: ``BZh``, xz: ``\\xfd7zXZ``).
        - If compression is detected, ``extractor.decompress_bytes()``
          decompresses the data transparently.
        - The decompressed content is what gets written to disk and
          verified. The ``FetchMetadata`` records both the original
          and decompressed sizes.
        - Why this approach instead of using ``Accept-Encoding`` headers:
          urllib handles ``Content-Encoding: gzip`` automatically in some
          cases but not others (depends on Python version, platform, and
          whether the server sends ``Transfer-Encoding: chunked``).
          Checking magic bytes is deterministic regardless of headers.

    Parameters
    ----------
    cache_dir : Optional[Path]
        Directory for caching downloaded resources. If ``None``, a
        temporary directory is created. Files in the cache survive
        between fetcher instances if the same directory is used.
    timeout : int
        Timeout in seconds for each individual HTTP request attempt.
        Total time may be higher due to retries.
    max_retries : int
        Maximum number of retry attempts per URL before trying mirrors
        or giving up.
    mirror_strategy : MirrorStrategy
        How mirrors are ordered and tried.
    verify_ssl : bool
        If ``True``, validate TLS certificates. Set to ``False`` only
        for testing against self-signed internal mirrors.
    headers : Optional[Dict[str, str]]
        Custom HTTP headers. If ``None``, sensible defaults are used.

    Attributes
    ----------
    cache_dir : Path
        Resolved path to the cache directory.
    timeout : int
        Per-request timeout in seconds.
    max_retries : int
        Retry attempts per URL.

    Examples
    --------
    Basic usage with default settings:

        >>> fetcher = ResourceFetcher()
        >>> success, path, meta = fetcher.fetch_get_pip()
        >>> if success:
        ...     print(f"Downloaded to {path}")
        Downloaded to /tmp/pip_cache_abc123/get-pip.py

    Custom cache directory for persistence across runs:

        >>> fetcher = ResourceFetcher(
        ...     cache_dir=Path.home() / ".pip_rescuer_cache",
        ...     timeout=60,
        ...     max_retries=3,
        ... )
        >>> success, path, _ = fetcher.fetch_pip_wheel("21.3.1")

    Adding a corporate mirror:

        >>> fetcher = ResourceFetcher()
        >>> fetcher.add_mirror("https://artifactory.company.com/pypi/simple/")
        >>> fetcher.mirror_strategy = MirrorStrategy.PRIMARY_FIRST

    Checking connectivity only (no download):

        >>> fetcher = ResourceFetcher()
        >>> if not fetcher.check_connectivity():
        ...     print("No internet; switching to offline mode")
    """

    # Retry backoff configuration.
    # Why exponential backoff:
    #   - Immediate retry on a transient failure hits the same broken
    #     state (overloaded server, brief network partition).
    #   - Linear backoff (1s, 2s, 3s) is better but can still hammer
    #     recovering services.
    #   - Exponential backoff (1s, 2s, 4s, 8s) gives services time to
    #     recover while still retrying reasonably quickly.
    _INITIAL_BACKOFF_SECONDS: float = 1.0
    _BACKOFF_MULTIPLIER: float = 2.0
    _MAX_BACKOFF_SECONDS: float = 30.0

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        timeout: int = 30,
        max_retries: int = 3,
        mirror_strategy: MirrorStrategy = MirrorStrategy.PRIMARY_FIRST,
        verify_ssl: bool = True,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize a resource fetcher.

        Parameters
        ----------
        cache_dir : Optional[Path]
            Directory for cached files. Created if it does not exist.
            If ``None``, uses a temporary directory.
        timeout : int
            Seconds to wait for each HTTP request before timing out.
        max_retries : int
            Retry attempts per URL before trying mirrors.
        mirror_strategy : MirrorStrategy
            Ordering strategy for mirror attempts.
        verify_ssl : bool
            Whether to validate TLS certificates.
        headers : Optional[Dict[str, str]]
            Custom HTTP request headers.

        Raises
        ------
        PermissionError
            If ``cache_dir`` is specified but cannot be created due to
            filesystem permissions.

        Examples
        --------
        >>> fetcher = ResourceFetcher(timeout=60, max_retries=5)
        >>> fetcher.timeout
        60
        >>> fetcher.max_retries
        5
        """
        self.timeout: int = timeout
        self.max_retries: int = max_retries
        self.mirror_strategy: MirrorStrategy = mirror_strategy
        self.verify_ssl: bool = verify_ssl
        self.headers: Dict[str, str] = (
            headers if headers is not None else dict(_DEFAULT_HEADERS)
        )

        # Cache setup
        if cache_dir is None:
            self.cache_dir = Path(tempfile.mkdtemp(prefix="pip_rescuer_cache_"))
        else:
            self.cache_dir = Path(cache_dir).resolve()
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache index: maps canonical resource names to their cache entries
        self._cache_index: Dict[str, CacheEntry] = {}
        self._load_cache_index()

        # Mirror list
        self._mirrors: List[str] = list(_DEFAULT_MIRRORS)

        # URL configuration
        self._get_pip_url: str = _DEFAULT_GET_PIP_URL
        self._pypi_simple_url: str = _DEFAULT_PYPI_SIMPLE_URL
        self._wheel_base_url: str = _DEFAULT_WHEEL_BASE_URL

        # SSL context
        if not verify_ssl:
            self._ssl_context: Optional[ssl.SSLContext] = (
                ssl._create_unverified_context()
            )
        else:
            self._ssl_context = None

    # ------------------------------------------------------------------
    # Public API: Fetch Operations
    # ------------------------------------------------------------------

    def fetch_get_pip(
        self, version: Optional[str] = None
    ) -> Tuple[bool, Path, FetchMetadata]:
        """
        Download the get-pip.py bootstrapping script.

        get-pip.py is a standalone script that bootstraps pip without
        requiring pip to already be installed. It is the recommended
        installation method for environments where pip is missing or
        broken.

        Why download get-pip.py instead of always using ensurepip:
            - ensurepip only installs the pip version bundled with the
              Python distribution, which may be outdated.
            - get-pip.py supports the ``PIP_VERSION`` environment variable
              to install a specific pip version.
            - get-pip.py works even when pip is completely missing,
              whereas ensurepip may be absent on stripped-down Python
              installations (some Docker images, embedded Python).

        Parameters
        ----------
        version : Optional[str]
            If specified, fetches a version-specific get-pip.py from
            the historic archive. Versions are available at:
            ``https://bootstrap.pypa.io/pip/{version}/get-pip.py``.
            If ``None``, fetches the latest version.

        Returns
        -------
        success : bool
            ``True`` if the file was obtained (downloaded or cached).
        file_path : Path
            Absolute path to the downloaded or cached get-pip.py file.
            Only valid if ``success`` is ``True``.
        metadata : FetchMetadata
            Detailed information about the fetch operation.

        Examples
        --------
        >>> fetcher = ResourceFetcher()
        >>> success, path, meta = fetcher.fetch_get_pip()
        >>> success
        True
        >>> path.suffix
        '.py'
        >>> meta.status in (FetchStatus.SUCCESS, FetchStatus.CACHED)
        True

        Fetch a version-specific get-pip.py for Python 3.6:

        >>> success, path, meta = fetcher.fetch_get_pip(version="3.6")
        >>> if success:
        ...     print(f"Got get-pip.py for Python {version}")
        """
        if version:
            url = f"https://bootstrap.pypa.io/pip/{version}/get-pip.py"
            cache_key = f"get-pip.py-v{version}"
        else:
            url = self._get_pip_url
            cache_key = "get-pip.py-latest"

        return self._fetch_with_cache(
            url=url,
            cache_key=cache_key,
            file_extension=".py",
            description="get-pip.py bootstrapping script",
        )

    def fetch_pip_wheel(
        self, version: str, python_version: Optional[str] = None
    ) -> Tuple[bool, Path, FetchMetadata]:
        """
        Download a specific pip wheel file from PyPI.

        Wheel files are the preferred format for installing pip because
        they are pre-built and require no compilation. The wheel URL is
        discovered by querying the PyPI simple API, which returns a list
        of available wheels for the requested version.

        Why discover the URL from PyPI instead of hardcoding it:
            - Wheel filenames include content hashes that change per
              release.
            - PyPI may serve wheels from different CDN nodes.
            - The simple API is the PEP 503 standard for package index
              interaction.

        Parameters
        ----------
        version : str
            Pip version to download. Example: ``"23.0.1"``.
        python_version : Optional[str]
            Python version tag for the wheel (e.g., ``"py3"``,
            ``"py2.py3"``). If ``None``, fetches the universal
            ``py3-none-any`` wheel.

        Returns
        -------
        success : bool
            ``True`` if the wheel was obtained.
        file_path : Path
            Absolute path to the downloaded ``.whl`` file.
        metadata : FetchMetadata
            Detailed fetch metadata including SHA256 hash for
            integrity verification.

        Examples
        --------
        >>> fetcher = ResourceFetcher()
        >>> success, path, meta = fetcher.fetch_pip_wheel("21.3.1")
        >>> success
        True
        >>> path.suffix
        '.whl'
        >>> meta.sha256 is not None
        True
        """
        from .versions import normalize_version

        normalized = normalize_version(version)
        python_tag = python_version or "py3"

        # First, discover the wheel URL from PyPI simple API
        wheel_info = self._discover_wheel_url(normalized, python_tag)
        if wheel_info is None:
            # Fallback: construct the URL manually
            wheel_filename = f"pip-{normalized}-{python_tag}-none-any.whl"
            wheel_url = urljoin(self._wheel_base_url, wheel_filename)
            expected_sha256 = None
        else:
            wheel_url = wheel_info["url"]
            expected_sha256 = wheel_info.get("sha256")

        cache_key = f"pip-{normalized}-{python_tag}-none-any.whl"

        success, file_path, metadata = self._fetch_with_cache(
            url=wheel_url,
            cache_key=cache_key,
            file_extension=".whl",
            description=f"pip {normalized} wheel",
            expected_sha256=expected_sha256,
        )

        # If the discovered URL failed, try constructing the URL
        if not success and wheel_info is not None:
            wheel_filename = f"pip-{normalized}-{python_tag}-none-any.whl"
            wheel_url = urljoin(self._wheel_base_url, wheel_filename)
            success, file_path, metadata = self._fetch_with_cache(
                url=wheel_url,
                cache_key=cache_key,
                file_extension=".whl",
                description=f"pip {normalized} wheel (fallback URL)",
            )

        return success, file_path, metadata

    def fetch_resource_direct(
        self,
        url: str,
        filename: str,
        expected_sha256: Optional[str] = None,
    ) -> Tuple[bool, Path, FetchMetadata]:
        """
        Download an arbitrary resource from a direct URL.

        Why expose this:
            - Users may have internal mirrors with non-standard URL
              structures.
            - Testing against pre-release or custom pip builds.
            - Downloading supplementary resources (documentation,
              checksum files).

        Parameters
        ----------
        url : str
            Direct URL to the resource.
        filename : str
            Name to save the file as in the cache.
        expected_sha256 : Optional[str]
            If provided, the downloaded file's SHA256 must match this
            value. Mismatch causes the fetch to fail with
            ``FetchStatus.CHECKSUM_FAILED``.

        Returns
        -------
        success : bool
            ``True`` if the resource was obtained.
        file_path : Path
            Absolute path to the downloaded file.
        metadata : FetchMetadata
            Detailed fetch metadata.

        Examples
        --------
        >>> fetcher = ResourceFetcher()
        >>> success, path, meta = fetcher.fetch_resource_direct(
        ...     url="https://example.com/custom-pip.whl",
        ...     filename="custom-pip.whl",
        ... )
        """
        return self._fetch_with_cache(
            url=url,
            cache_key=filename,
            file_extension=Path(filename).suffix,
            description=f"resource from {url}",
            expected_sha256=expected_sha256,
        )

    # ------------------------------------------------------------------
    # Public API: Connectivity & Cache Management
    # ------------------------------------------------------------------

    def check_connectivity(self) -> bool:
        """
        Test whether any of the configured hosts are reachable.

        Why test multiple hosts:
            - A single host may be temporarily down.
            - Corporate firewalls may block ``bootstrap.pypa.io`` but
              allow ``pypi.org``.
            - DNS may fail for one domain but not others.
            - Testing mirrors as well as primary URLs gives a more
              accurate picture of what the fetcher can actually reach.

        Why HTTP HEAD instead of socket connection:
            - A socket connection only proves the host is reachable,
              not that the HTTP service is running.
            - An HTTP HEAD request validates the full stack: DNS,
              TCP, TLS, HTTP server.
            - Some proxies allow TCP connections but block HTTP
              requests; HEAD catches this.

        Returns
        -------
        bool
            ``True`` if at least one host responded successfully.

        Examples
        --------
        >>> fetcher = ResourceFetcher()
        >>> if fetcher.check_connectivity():
        ...     print("Network is available")
        ... else:
        ...     print("Network is unavailable; will use offline mode")
        Network is available
        """
        hosts_to_check: List[str] = [
            self._get_pip_url,
            self._pypi_simple_url,
            "https://pypi.org",
            "https://github.com",
        ]
        hosts_to_check.extend(self._mirrors)

        for host_url in hosts_to_check:
            parsed = urlparse(host_url)
            check_url = f"{parsed.scheme}://{parsed.netloc}"

            try:
                request = Request(
                    check_url,
                    headers=self.headers,
                    method="HEAD",
                )
                urlopen(
                    request,
                    timeout=min(self.timeout, 10),
                    context=self._ssl_context,
                )
                return True
            except Exception:
                continue

        return False

    def clear_cache(self, older_than_hours: Optional[int] = None) -> int:
        """
        Remove cached files, optionally only those older than a threshold.

        Why selective clearing:
            - Clearing the entire cache forces re-download of everything,
              wasting bandwidth.
            - Age-based clearing removes stale files while keeping
              recently-downloaded resources available.
            - Returns the count of removed files so callers can log
              cache management actions.

        Parameters
        ----------
        older_than_hours : Optional[int]
            If provided, only remove files cached more than this many
            hours ago. If ``None``, clear all cached files.

        Returns
        -------
        int
            Number of cache entries removed.

        Examples
        --------
        >>> fetcher = ResourceFetcher()
        >>> removed = fetcher.clear_cache(older_than_hours=168)  # 1 week
        >>> print(f"Removed {removed} stale cache entries")
        Removed 0 stale cache entries

        >>> removed = fetcher.clear_cache()
        >>> print(f"Cleared all {removed} cache entries")
        Cleared all 0 cache entries
        """
        removed = 0
        now = time.time()

        for cache_key, entry in list(self._cache_index.items()):
            if older_than_hours is not None:
                age_hours = (now - entry.cached_at) / 3600.0
                if age_hours <= older_than_hours:
                    continue

            # Remove the file
            if entry.file_path.exists():
                entry.file_path.unlink()

            # Remove from index
            del self._cache_index[cache_key]
            removed += 1

        self._save_cache_index()
        return removed

    def add_mirror(self, mirror_url: str, position: Optional[int] = None) -> None:
        """
        Register an additional mirror URL for fallback.

        Mirrors are tried in order after the primary URL fails.
        Use ``position`` to control priority.

        Why add mirrors at runtime:
            - Corporate environments may provide an internal mirror
              URL that is not known at module load time.
            - Users may have personal mirror preferences.
            - Testing may require injecting a local mirror URL.

        Parameters
        ----------
        mirror_url : str
            Base URL of the mirror. For PyPI mirrors, this should be
            the simple API endpoint (e.g., ``https://mirror.example.com/simple/``).
        position : Optional[int]
            Insert position in the mirror list. ``0`` = highest priority,
            ``None`` = append to end (lowest priority).

        Examples
        --------
        >>> fetcher = ResourceFetcher()
        >>> fetcher.add_mirror("https://artifactory.internal/simple/", position=0)
        >>> fetcher.list_mirrors()[0]
        'https://artifactory.internal/simple/'
        """
        if position is not None:
            self._mirrors.insert(position, mirror_url)
        else:
            self._mirrors.append(mirror_url)

    def remove_mirror(self, mirror_url: str) -> bool:
        """
        Remove a previously registered mirror.

        Parameters
        ----------
        mirror_url : str
            The mirror URL to remove.

        Returns
        -------
        bool
            ``True`` if the mirror was found and removed.

        Examples
        --------
        >>> fetcher = ResourceFetcher()
        >>> fetcher.add_mirror("https://temp-mirror.example.com/simple/")
        >>> fetcher.remove_mirror("https://temp-mirror.example.com/simple/")
        True
        """
        if mirror_url in self._mirrors:
            self._mirrors.remove(mirror_url)
            return True
        return False

    def list_mirrors(self) -> List[str]:
        """
        Return the current list of configured mirror URLs.

        Returns
        -------
        List[str]
            Mirror URLs in priority order.

        Examples
        --------
        >>> fetcher = ResourceFetcher()
        >>> mirrors = fetcher.list_mirrors()
        >>> len(mirrors) > 0
        True
        """
        return list(self._mirrors)

    # ------------------------------------------------------------------
    # Public API: Integrity Verification
    # ------------------------------------------------------------------

    def verify_file_integrity(
        self,
        file_path: Path,
        expected_sha256: Optional[str] = None,
        expected_size: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Verify a downloaded file's integrity.

        Why verify after download:
            - Network corruption: TCP checksums catch some errors but
              not all; a file can be corrupted in transit and still
              pass TCP validation.
            - Proxy interference: some proxies modify content (injecting
              HTML error pages, stripping Content-Encoding headers).
            - Storage corruption: writing to a failing disk can produce
              a truncated or corrupted file without raising an OS error.
            - Supply chain: comparing against a known hash ensures the
              file has not been tampered with since the hash was published.

        Parameters
        ----------
        file_path : Path
            Path to the file to verify.
        expected_sha256 : Optional[str]
            Expected SHA256 hex digest. If ``None``, hash verification
            is skipped.
        expected_size : Optional[int]
            Expected file size in bytes. If ``None``, size verification
            is skipped.

        Returns
        -------
        valid : bool
            ``True`` if all specified checks passed.
        reason : str
            Explanation of the verification result.

        Examples
        --------
        >>> fetcher = ResourceFetcher()
        >>> _, path, meta = fetcher.fetch_pip_wheel("21.3.1")
        >>> valid, reason = fetcher.verify_file_integrity(
        ...     path,
        ...     expected_sha256=meta.sha256,
        ... )
        >>> valid
        True
        >>> reason
        'All integrity checks passed'
        """
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"

        # Size check
        if expected_size is not None:
            actual_size = file_path.stat().st_size
            if actual_size != expected_size:
                return False, (
                    f"Size mismatch: expected {expected_size} bytes, "
                    f"got {actual_size} bytes"
                )

        # Hash check
        if expected_sha256 is not None:
            actual_hash = self._compute_sha256(file_path)
            if actual_hash != expected_sha256.lower():
                return False, (
                    f"SHA256 mismatch:\n"
                    f"  Expected: {expected_sha256.lower()}\n"
                    f"  Got:      {actual_hash}"
                )

        return True, "All integrity checks passed"

    @staticmethod
    def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
        """
        Compute a cryptographic hash of a file.

        Why static:
            - Hash computation is a pure function; no instance state needed.
            - Useful as a utility outside the fetcher (e.g., verifying
              manually downloaded files).

        Parameters
        ----------
        file_path : Path
            Path to the file to hash.
        algorithm : str
            Hash algorithm name accepted by ``hashlib.new()``.
            Examples: ``"sha256"``, ``"sha512"``, ``"md5"``.

        Returns
        -------
        str
            Hex digest of the file's hash.

        Raises
        ------
        ValueError
            If ``algorithm`` is not recognized by hashlib.
        FileNotFoundError
            If ``file_path`` does not exist.

        Examples
        --------
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        ...     _ = f.write(b"Hello, World!")
        ...     tmp_path = Path(f.name)
        >>> ResourceFetcher.compute_file_hash(tmp_path, "sha256")
        'dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f'
        >>> tmp_path.unlink()
        """
        hasher = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    # ------------------------------------------------------------------
    # Private: Core Fetch Logic
    # ------------------------------------------------------------------

    def _fetch_with_cache(
        self,
        url: str,
        cache_key: str,
        file_extension: str,
        description: str,
        expected_sha256: Optional[str] = None,
    ) -> Tuple[bool, Path, FetchMetadata]:
        """
        Central fetch method: check cache, then download with fallback.

        Why this is the single internal fetch path:
            - All public fetch methods converge here, ensuring consistent
              caching, retry, mirror fallback, integrity checking, and
              decompression handling.
            - Adding a feature (e.g., progress callbacks) requires
              modifying only this method.
            - Cache validation logic is centralized; no risk of one
              fetch method skipping the cache while another uses it.
        """
        # Step 1: Check cache
        if cache_key in self._cache_index:
            entry = self._cache_index[cache_key]
            if self._is_cache_fresh(entry):
                if entry.file_path.exists():
                    return True, entry.file_path, entry.metadata

        # Step 2: Determine URL list
        urls_to_try = self._build_url_list(url)

        # Step 3: Attempt download
        last_error: Optional[str] = None
        for attempt_url in urls_to_try:
            success, file_path, metadata = self._download_with_retry(
                url=attempt_url,
                cache_key=cache_key,
                file_extension=file_extension,
                description=description,
                is_mirror=(attempt_url != url),
                expected_sha256=expected_sha256,
            )
            if success:
                # Update cache
                self._update_cache(cache_key, file_path, metadata)
                return True, file_path, metadata
            else:
                last_error = metadata.error_message

        # Step 4: All attempts failed
        failure_metadata = FetchMetadata(
            url=url,
            status=FetchStatus.NETWORK_ERROR,
            size_bytes=0,
            error_message=last_error
            or f"Failed to fetch {description} from any source",
        )
        return False, Path(""), failure_metadata

    def _download_with_retry(
        self,
        url: str,
        cache_key: str,
        file_extension: str,
        description: str,
        is_mirror: bool,
        expected_sha256: Optional[str] = None,
    ) -> Tuple[bool, Path, FetchMetadata]:
        """
        Attempt to download from a single URL with retry logic.

        Why retry at the download level:
            - HTTP 503 (Service Unavailable) and 429 (Rate Limited) are
              transient; retrying after a delay often succeeds.
            - Connection resets and timeouts during transfer are common
              on unreliable networks.
            - Exponential backoff prevents thundering-herd on recovering
              services.
        """
        retry_count = 0
        last_error: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            try:
                request = Request(url, headers=self.headers)
                response = urlopen(
                    request,
                    timeout=self.timeout,
                    context=self._ssl_context,
                )

                # Check HTTP status
                status_code = response.getcode()
                if status_code == 404:
                    return (
                        False,
                        Path(""),
                        FetchMetadata(
                            url=url,
                            status=FetchStatus.NOT_FOUND,
                            size_bytes=0,
                            retry_count=retry_count,
                            mirror_used=url if is_mirror else None,
                            error_message=f"Resource not found at {url} (HTTP 404)",
                        ),
                    )

                if status_code >= 500:
                    raise HTTPError(
                        url, status_code, "Server error", response.headers, None
                    )

                # Check content length to avoid downloading something huge
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        cl_int = int(content_length)
                        if cl_int > _MAX_DOWNLOAD_SIZE:
                            return (
                                False,
                                Path(""),
                                FetchMetadata(
                                    url=url,
                                    status=FetchStatus.NETWORK_ERROR,
                                    size_bytes=0,
                                    retry_count=retry_count,
                                    error_message=(
                                        f"Resource at {url} is {cl_int} bytes, "
                                        f"exceeding maximum of {_MAX_DOWNLOAD_SIZE}"
                                    ),
                                ),
                            )
                    except ValueError:
                        pass

                # Read response body with size limiting
                raw_data = self._read_response(response, url)
                original_size = len(raw_data)

                # ----------------------------------------------------------
                # Transparent decompression via extractor module
                # ----------------------------------------------------------
                # Why check magic bytes instead of Content-Encoding header:
                #   - Some CDNs compress responses without setting
                #     Content-Encoding correctly.
                #   - urllib may or may not decompress automatically
                #     depending on Python version and platform.
                #   - Magic byte detection is deterministic: if the data
                #     starts with gzip magic, it IS gzip regardless of
                #     what headers claim.
                # ----------------------------------------------------------
                was_decompressed = False
                if is_compressed(raw_data):
                    try:
                        data = decompress_bytes(raw_data)
                        was_decompressed = True
                    except (ValueError, OSError):
                        # Decompression failed. The magic bytes may have
                        # been coincidental (e.g., a file that starts with
                        # bytes that happen to match a magic signature).
                        # Use the original data unchanged.
                        data = raw_data
                else:
                    data = raw_data

                # Write to cache (always write decompressed content)
                dest_path = self.cache_dir / f"{cache_key}{file_extension}"
                dest_path.write_bytes(data)

                # Verify integrity if hash provided
                if expected_sha256:
                    actual_hash = self._compute_sha256(dest_path)
                    if actual_hash != expected_sha256.lower():
                        dest_path.unlink()
                        return (
                            False,
                            Path(""),
                            FetchMetadata(
                                url=url,
                                status=FetchStatus.CHECKSUM_FAILED,
                                size_bytes=len(data),
                                retry_count=retry_count,
                                mirror_used=url if is_mirror else None,
                                decompressed=was_decompressed,
                                original_size_bytes=original_size,
                                error_message=(
                                    f"Checksum mismatch for {description}: "
                                    f"expected {expected_sha256}, got {actual_hash}"
                                ),
                            ),
                        )

                # Determine final status
                if was_decompressed:
                    final_status = FetchStatus.DECOMPRESSED
                elif is_mirror:
                    final_status = FetchStatus.MIRROR_USED
                elif retry_count > 0:
                    final_status = FetchStatus.RETRIED
                else:
                    final_status = FetchStatus.SUCCESS

                return (
                    True,
                    dest_path,
                    FetchMetadata(
                        url=url,
                        status=final_status,
                        size_bytes=len(data),
                        sha256=expected_sha256 or self._compute_sha256(dest_path),
                        etag=response.headers.get("ETag"),
                        last_modified=response.headers.get("Last-Modified"),
                        retry_count=retry_count,
                        mirror_used=url if is_mirror else None,
                        decompressed=was_decompressed,
                        original_size_bytes=original_size,
                    ),
                )

            except HTTPError as e:
                last_error = f"HTTP {e.code}: {e.reason} for {url}"
                if e.code in (404, 410):
                    # Don't retry on permanent errors
                    break

            except URLError as e:
                last_error = f"URL Error for {url}: {e.reason}"

            except ssl.SSLError as e:
                last_error = f"SSL Error for {url}: {e}"

            except (TimeoutError, OSError) as e:
                last_error = f"Connection error for {url}: {e}"

            # Retry with backoff
            retry_count += 1
            if attempt < self.max_retries:
                backoff = min(
                    self._INITIAL_BACKOFF_SECONDS
                    * (self._BACKOFF_MULTIPLIER**attempt),
                    self._MAX_BACKOFF_SECONDS,
                )
                time.sleep(backoff)

        return (
            False,
            Path(""),
            FetchMetadata(
                url=url,
                status=(
                    FetchStatus.TIMEOUT
                    if "timed out" in str(last_error).lower()
                    else FetchStatus.NETWORK_ERROR
                ),
                size_bytes=0,
                retry_count=retry_count,
                mirror_used=url if is_mirror else None,
                error_message=last_error or f"Failed after {retry_count} retries",
            ),
        )

    # ------------------------------------------------------------------
    # Private: URL Construction
    # ------------------------------------------------------------------

    def _build_url_list(self, primary_url: str) -> List[str]:
        """
        Build an ordered list of URLs to try, based on mirror strategy.

        Why build the list before downloading:
            - The mirror strategy determines order; building the list
              once prevents re-evaluating strategy on each retry.
            - Enables logging the full attempt plan before starting,
              useful for debugging.
        """
        urls = [primary_url]

        import random

        if self.mirror_strategy == MirrorStrategy.PRIMARY_FIRST:
            urls.extend(self._mirrors)

        elif self.mirror_strategy == MirrorStrategy.SEQUENTIAL:
            urls = self._mirrors + [primary_url]

        elif self.mirror_strategy == MirrorStrategy.RANDOM:
            shuffled = list(self._mirrors)
            random.shuffle(shuffled)
            urls = [primary_url] + shuffled

        elif self.mirror_strategy == MirrorStrategy.PARALLEL:
            shuffled = list(self._mirrors)
            random.shuffle(shuffled)
            urls = [primary_url] + shuffled

        return urls

    def _discover_wheel_url(
        self, version: str, python_tag: str
    ) -> Optional[Dict[str, str]]:
        """
        Query the PyPI simple API to find the wheel URL for a specific version.

        PEP 503 defines the simple repository API. The response is an HTML
        page with anchor tags pointing to distribution files. We parse this
        to find the matching wheel.

        Why discover instead of hardcoding:
            - Wheel filenames change with each release.
            - PyPI may add new wheel variants (e.g., cp313 for Python 3.13).
            - The API response includes SHA256 hashes for integrity.

        Parameters
        ----------
        version : str
            Normalized pip version.
        python_tag : str
            Python compatibility tag (e.g., "py3").

        Returns
        -------
        Optional[Dict[str, str]]
            Dictionary with keys "url" and "sha256", or None if not found.
        """
        try:
            request = Request(self._pypi_simple_url, headers=self.headers)
            response = urlopen(
                request,
                timeout=self.timeout,
                context=self._ssl_context,
            )
            html = response.read().decode("utf-8", errors="replace")
        except Exception:
            return None

        # Parse anchor tags: <a href="...">filename</a>
        pattern = re.compile(
            r'<a[^>]*href=["\']([^"\']*pip-'
            + re.escape(version)
            + r'-[^"\']*\.whl[^"\']*)["\'][^>]*>(?:[^<]*)</a>'
        )

        for match in pattern.finditer(html):
            href = match.group(1)
            if python_tag in href:
                # Check for data-hashes attribute (SHA256)
                hash_match = re.search(
                    r'data-hashes=["\']\{.*?"sha256"\s*:\s*"([a-f0-9]{64})"',
                    html,
                )
                sha256 = hash_match.group(1) if hash_match else None

                # Make URL absolute if relative
                if not href.startswith("http"):
                    href = urljoin(self._pypi_simple_url, href)

                return {"url": href, "sha256": sha256}

        return None

    # ------------------------------------------------------------------
    # Private: I/O Helpers
    # ------------------------------------------------------------------

    def _read_response(self, response, url: str) -> bytes:
        """
        Read an HTTP response body with size limiting.

        Why size-limiting during read:
            - A misconfigured proxy might return an HTML error page that
              is small enough to pass the Content-Length check but still
              not a valid wheel.
            - Streaming read with a hard limit prevents memory exhaustion
              if a server sends an unbounded response.
        """
        chunks: List[bytes] = []
        total_size = 0

        while True:
            chunk = response.read(65536)  # 64 KB chunks
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > _MAX_DOWNLOAD_SIZE:
                raise ValueError(
                    f"Response from {url} exceeds maximum size of "
                    f"{_MAX_DOWNLOAD_SIZE} bytes"
                )
            chunks.append(chunk)

        return b"".join(chunks)

    def _compute_sha256(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        return self.compute_file_hash(file_path, "sha256")

    # ------------------------------------------------------------------
    # Private: Cache Management
    # ------------------------------------------------------------------

    def _is_cache_fresh(self, entry: CacheEntry) -> bool:
        """Check if a cache entry is still valid."""
        if entry.max_age_seconds is None:
            return True
        age = time.time() - entry.cached_at
        return age < entry.max_age_seconds

    def _update_cache(
        self,
        cache_key: str,
        file_path: Path,
        metadata: FetchMetadata,
    ) -> None:
        """Store a fetched resource in the cache index."""
        entry = CacheEntry(
            file_path=file_path,
            metadata=metadata,
            cached_at=time.time(),
        )
        self._cache_index[cache_key] = entry
        self._save_cache_index()

    def _load_cache_index(self) -> None:
        """
        Load the cache index from disk.

        The cache index is a JSON file mapping cache keys to their
        metadata. This survives fetcher instance destruction and
        enables cache persistence across program runs.

        Why JSON instead of pickle:
            - JSON is human-readable and debuggable.
            - JSON is safe to inspect and edit manually.
            - Pickle has security implications if the cache file is
              shared or stored in an untrusted location.
        """
        index_path = self.cache_dir / "cache_index.json"
        if not index_path.exists():
            return

        try:
            with open(index_path, "r") as f:
                data = json.load(f)

            for cache_key, entry_data in data.items():
                file_path = Path(entry_data["file_path"])
                if file_path.exists():
                    self._cache_index[cache_key] = CacheEntry(
                        file_path=file_path,
                        metadata=FetchMetadata(**entry_data["metadata"]),
                        cached_at=entry_data["cached_at"],
                        max_age_seconds=entry_data.get("max_age_seconds"),
                    )
        except (json.JSONDecodeError, KeyError, OSError):
            # Corrupt cache index; start fresh
            self._cache_index = {}

    def _save_cache_index(self) -> None:
        """Persist the cache index to disk."""
        index_path = self.cache_dir / "cache_index.json"

        data: Dict[str, Dict[str, Any]] = {}
        for cache_key, entry in self._cache_index.items():
            data[cache_key] = {
                "file_path": str(entry.file_path),
                "metadata": {
                    "url": entry.metadata.url,
                    "status": entry.metadata.status.name,
                    "size_bytes": entry.metadata.size_bytes,
                    "sha256": entry.metadata.sha256,
                    "etag": entry.metadata.etag,
                    "last_modified": entry.metadata.last_modified,
                    "fetched_at": entry.metadata.fetched_at,
                    "retry_count": entry.metadata.retry_count,
                    "mirror_used": entry.metadata.mirror_used,
                    "decompressed": entry.metadata.decompressed,
                    "original_size_bytes": entry.metadata.original_size_bytes,
                    "error_message": entry.metadata.error_message,
                },
                "cached_at": entry.cached_at,
                "max_age_seconds": entry.max_age_seconds,
            }

        try:
            with open(index_path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            # Cache directory may not be writable; fail silently
            pass
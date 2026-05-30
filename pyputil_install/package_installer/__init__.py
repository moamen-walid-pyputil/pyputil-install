#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Package Installer - Advanced Python Package Manager.

A comprehensive Python package management library that provides
installation, uninstallation, upgrading, and querying of Python
packages with built-in caching, security verification, retry logic,
dependency resolution, and virtual environment support.

This library wraps common pip operations with enhanced error handling,
network resilience, and security features while maintaining a simple,
Pythonic API.

Subpackages
-----------
network
    HTTP session management and PyPI JSON API client.
security
    Cryptographic hash verification and trust management.
environment
    Virtual environment creation and management.
cli
    Command-line interface with argument parsing and formatting.

Submodules
----------
exceptions
    Custom exception hierarchy for granular error handling.
retry
    Retry mechanism with exponential backoff and jitter.
cache
    Persistent file-based caching with TTL and compression.
installer
    Core package installer with dependency resolution and rollback.

Classes
-------
PackageInstaller
    Main class for managing individual Python packages.
InstallConfig
    Configuration for installation behavior.
InstallResult
    Result of an installation operation with detailed metadata.
PackageSession
    Thread-safe HTTP session with connection pooling and retry logic.
PyPIClient
    Client for the PyPI JSON API with caching awareness.
HashVerifier
    Verifies file integrity using cryptographic hashes.
TrustVerifier
    Verifies package source trustworthiness and authenticity.
VirtualEnvironment
    Manages a Python virtual environment lifecycle.
PackageCache
    Persistent file-based cache for package metadata and API responses.
RetryConfig
    Configuration for retry behavior.
RetryManager
    Context manager for retry logic with detailed statistics.
CacheConfig
    Configuration for cache behavior.
SessionConfig
    Configuration for HTTP session behavior.
TrustConfig
    Configuration for trust verification behavior.
HashAlgorithm
    Enumeration of supported hash algorithms.
EnvironmentInfo
    Information about a Python environment.

Functions
---------
retry
    Decorator that adds retry behavior to a function.
compute_hash
    Compute the cryptographic hash of data.
verify_hash
    Verify that data matches an expected hash value.
multi_hash
    Compute multiple hash algorithms in a single pass.
compare_hashes
    Constant-time hash comparison to prevent timing attacks.
is_trusted_host
    Convenience function to check if a host is trusted.
validate_url
    Convenience function to validate a URL's security.
clear_system_cache
    Remove all cached data from the default cache location.
get_cache_dir
    Determine the platform-appropriate cache directory.

Exceptions
----------
PackageInstallerError
    Base exception for all package installer errors.
PackageNotFoundError
    Raised when a package cannot be found locally or remotely.
PackageInstallError
    Raised when installation fails.
PackageUninstallError
    Raised when uninstallation fails.
PackageUpgradeError
    Raised when upgrade fails.
PackageVersionError
    Raised when version parsing or comparison fails.
NetworkError
    Raised when network operations fail.
TimeoutError
    Raised when operations exceed their time limit.
PermissionError
    Raised when file system permissions prevent an operation.
CacheError
    Raised when cache operations fail.
ValidationError
    Raised when package validation fails.
SecurityError
    Raised when security checks fail.

Examples
--------
Basic installation:

>>> from package_installer import PackageInstaller
>>> installer = PackageInstaller("requests")
>>> installer.is_installed()
False
>>> installer.install()
>>> installer.get_version()
'2.31.0'

With custom configuration:

>>> from package_installer import PackageInstaller, InstallConfig
>>> config = InstallConfig(
...     require_hashes=True,
...     max_retries=5,
...     use_cache=True,
... )
>>> installer = PackageInstaller("django", config=config)
>>> installer.install(version="4.2.0")

Using retry decorator:

>>> from package_installer import retry
>>> from package_installer.exceptions import NetworkError
>>> @retry(max_attempts=3, retryable_exceptions=(NetworkError,))
... def fetch_data(url):
...     pass

Using cache directly:

>>> from package_installer import PackageCache
>>> cache = PackageCache()
>>> cache.set("my_key", {"version": "1.0.0"})
>>> data = cache.get("my_key")

Computing hashes:

>>> from package_installer import compute_hash, HashAlgorithm
>>> digest = compute_hash(b"hello world", HashAlgorithm.SHA256)

Working with virtual environments:

>>> from package_installer import VirtualEnvironment
>>> venv = VirtualEnvironment("./my-env")
>>> venv.create()
>>> installer = PackageInstaller("flask", venv=venv)
>>> installer.install()

Using PyPI client directly:

>>> from package_installer import PyPIClient
>>> client = PyPIClient()
>>> metadata = client.get_package_metadata("numpy")
>>> versions = client.get_package_versions("django")

Checking for upgrades:

>>> installer = PackageInstaller("pip")
>>> if installer.check_upgrade():
...     result = installer.upgrade()
...     print(f"Upgraded to {result.version_installed}")

Getting package information:

>>> installer = PackageInstaller("numpy")
>>> info = installer.get_package_info()
>>> print(f"{info['name']} {info['latest_version']}: {info['summary']}")

Verifying trust:

>>> from package_installer import TrustVerifier, is_trusted_host
>>> is_trusted_host("pypi.org")
True

Command-line usage::

    $ package-installer install requests
    $ package-installer search "web framework"
    $ package-installer info flask --deps
    $ package-installer venv create ./project-env
    $ package-installer freeze --output requirements.txt

Notes
-----
This package is designed with a defense-in-depth security model:

1. **Transport Security**: TLS with certificate validation for all
   PyPI communications.
2. **Trust Verification**: Hostname allowlisting and certificate
   pinning support.
3. **Integrity Verification**: Cryptographic hash checking of
   downloaded packages.
4. **Recovery**: Automatic rollback on installation failure.

Package names are normalized per PEP 503: case-insensitive, with
``-``, ``_``, and ``.`` treated as equivalent separators.

All network operations use retry logic with exponential backoff and
full jitter to prevent thundering herd problems in distributed
environments.

The caching layer supports TTL-based expiration, automatic garbage
collection, zlib compression for large entries, and thread-safe
access via reentrant locks.

Warnings
--------
- Installing packages from untrusted sources can execute arbitrary
  code during setup. Always verify package provenance.
- System-wide installation may require elevated privileges. Use
  ``user_install=True`` or virtual environments to avoid permission
  issues.
- Disabling SSL verification (``verify_ssl=False``) exposes package
  downloads to man-in-the-middle attacks.
- MD5 hash verification is supported for legacy compatibility but
  is not cryptographically secure.
- The ``VirtualEnvironment.destroy()`` method irreversibly deletes
  all packages and configuration in the environment.
- Calling ``PackageSession.close()`` or ``PackageInstaller.close()``
  makes the instance unusable. Create a new instance if needed.
"""

# ── Core Installer ──────────────────────────────────────────────
from .installer import (
    PackageInstaller,
    InstallConfig,
    InstallResult,
)

# ── Exceptions ──────────────────────────────────────────────────
from .exceptions import (
    PackageInstallerError,
    PackageNotFoundError,
    PackageInstallError,
    PackageUninstallError,
    PackageUpgradeError,
    PackageVersionError,
    NetworkError,
    TimeoutError,
    PermissionError,
    CacheError,
    ValidationError,
    SecurityError,
)

# ── Retry System ────────────────────────────────────────────────
from .retry import (
    retry,
    RetryConfig,
    RetryManager,
    is_retryable_exception,
    combine_retry_configs,
)

# ── Cache System ────────────────────────────────────────────────
from .cache import (
    PackageCache,
    CacheConfig,
    CacheEntry,
    get_cache_dir,
    clear_system_cache,
)

# ── Network Layer ───────────────────────────────────────────────
from .network.session import (
    PackageSession,
    SessionConfig,
)

from .network.pypi import (
    PyPIClient,
)

# ── Security Layer ──────────────────────────────────────────────
from .security.hashes import (
    HashVerifier,
    HashAlgorithm,
    compute_hash,
    verify_hash,
    multi_hash,
    compare_hashes,
    compute_hashes_from_pypi_digests,
    is_hash_string,
)

from .security.verify import (
    TrustVerifier,
    TrustConfig,
    is_trusted_host,
    validate_url,
    parse_requirements_hash,
)

# ── Environment Layer ───────────────────────────────────────────
from .environment.venv import (
    VirtualEnvironment,
    EnvironmentInfo,
)

# ── Public API ──────────────────────────────────────────────────
__all__ = [
    # Core installer
    "PackageInstaller",
    "InstallConfig",
    "InstallResult",

    # Exceptions
    "PackageInstallerError",
    "PackageNotFoundError",
    "PackageInstallError",
    "PackageUninstallError",
    "PackageUpgradeError",
    "PackageVersionError",
    "NetworkError",
    "TimeoutError",
    "PermissionError",
    "CacheError",
    "ValidationError",
    "SecurityError",

    # Retry system
    "retry",
    "RetryConfig",
    "RetryManager",
    "is_retryable_exception",
    "combine_retry_configs",

    # Cache system
    "PackageCache",
    "CacheConfig",
    "CacheEntry",
    "get_cache_dir",
    "clear_system_cache",

    # Network layer
    "PackageSession",
    "SessionConfig",
    "PyPIClient",

    # Security - Hashes
    "HashVerifier",
    "HashAlgorithm",
    "compute_hash",
    "verify_hash",
    "multi_hash",
    "compare_hashes",
    "compute_hashes_from_pypi_digests",
    "is_hash_string",

    # Security - Trust
    "TrustVerifier",
    "TrustConfig",
    "is_trusted_host",
    "validate_url",
    "parse_requirements_hash",

    # Environment
    "VirtualEnvironment",
    "EnvironmentInfo",
]
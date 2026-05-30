"""
Standard Library Module Installer.

A tool for downloading and installing pure-Python standard library
modules directly from the CPython repository on GitHub. Useful when
the system Python installation is minimal or missing certain standard
library components.

Public API — Classes
--------------------
Installer
    Main class for installing, removing, and managing packages.
Downloader
    Handles network requests, caching, and recursive file downloads.
HttpClient
    Low-level HTTP client with retry logic and rate limit handling.
CacheManager
    File-based cache with TTL expiration and size management.
Manifest
    JSON-backed record of installed packages and their metadata.
DependencyResolver
    AST-based detection of module dependencies.
FetchResult
    Container for directory tree or single-file fetch results.
GitHubItem
    Parsed file or directory entry from GitHub Contents API.

Public API — High-Level Functions
----------------------------------
install_stdlib(name, version=None, force=False, ...)
    Install a single stdlib module or package.
install_many_stdlib(names, version=None, force=False, ...)
    Install multiple packages at once.
remove_stdlib(name)
    Remove an installed package.
remove_many_stdlib(names, ignore_missing=True)
    Remove multiple packages.
is_installed_stdlib(name)
    Check if a package is installed.
list_installed_stdlib()
    List all installed packages with metadata.
get_stdlib_info(name)
    Get detailed info for an installed package.
update_stdlib(name, version=None)
    Update a package to a newer version.
update_all_stdlib()
    Update all outdated packages.
check_updates_stdlib()
    Check for available updates.
register_stdlib(source_dir)
    Register a local directory as stdlib source.
add_to_sys_path(storage_dir=None, position=0)
    Add storage directory to sys.path.
remove_from_sys_path(storage_dir=None)
    Remove storage directory from sys.path.
clear_cache(expired_only=True)
    Clear the download cache.
get_cache_stats()
    Return cache usage statistics.

Exceptions
----------
StdlibInstallerError
    Base exception for all installer errors.
NetworkError
    Network-level failures (timeout, DNS, TLS).
GitHubAPIError
    GitHub API returned an error status.
RateLimitError
    GitHub API rate limit exceeded.
PackageNotFoundError
    Requested package not found in repository.
ModuleNotInstalledError
    Package not installed locally.
AlreadyInstalledError
    Package already installed.
CompiledModuleError
    Package is a C extension and cannot be installed.
ChecksumVerificationError
    Downloaded file failed integrity check.
CircularDependencyError
    Circular dependency chain detected.
InstallationError
    Generic installation failure.

Examples
--------
Quick install:

    >>> from pyputil_install.stdlib_installer import install_stdlib, is_installed_stdlib
    >>> install_stdlib("json")
    PosixPath('/home/user/.stdlib-packages/json')
    >>> is_installed_stdlib("json")
    True

Bulk install:

    >>> from pyputil_install.stdlib_installer import install_many_stdlib
    >>> results = install_many_stdlib(["json", "csv", "datetime"])
    >>> for name, result in results.items():
    ...     if isinstance(result, Path):
    ...         print(f"{name}: ok")
    ...     else:
    ...         print(f"{name}: failed - {result}")

Use with sys.path:

    >>> from pyputil_install.stdlib_installer import add_to_sys_path, install_stdlib
    >>> add_to_sys_path()
    >>> install_stdlib("tomllib", version="3.13")
    >>> import tomllib  # works even on older Python

CLI Usage
---------
     $ stdlib_installer install json
     $ stdlib_installer list
     $ stdlib_installer remove json
     $ stdlib_installer check-updates
"""

# ---------------------------------------------------------------------------
# Core classes
# ---------------------------------------------------------------------------

from .installer import Installer, DependencyResolver
from .downloader import (
    Downloader,
    HttpClient,
    CacheManager,
    FetchResult,
    GitHubItem,
)
from .manifest import Manifest

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

from .exceptions import (
    StdlibInstallerError,
    NetworkError,
    GitHubAPIError,
    RateLimitError,
    PackageNotFoundError,
    ModuleNotInstalledError,
    AlreadyInstalledError,
    CompiledModuleError,
    ChecksumVerificationError,
    CircularDependencyError,
    InstallationError,
)

# ---------------------------------------------------------------------------
# High-level convenience functions
# ---------------------------------------------------------------------------

from .utils import (
    install_stdlib,
    install_many_stdlib,
    remove_stdlib,
    remove_many_stdlib,
    is_installed_stdlib,
    list_installed_stdlib,
    get_stdlib_info,
    update_stdlib,
    update_all_stdlib,
    check_updates_stdlib,
    register_stdlib,
    add_to_sys_path,
    remove_from_sys_path,
    clear_cache,
    get_cache_stats,
)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # Core classes
    "Installer",
    "Downloader",
    "HttpClient",
    "CacheManager",
    "Manifest",
    "DependencyResolver",
    # Data classes
    "FetchResult",
    "GitHubItem",
    # High-level functions — install
    "install_stdlib",
    "install_many_stdlib",
    # High-level functions — remove
    "remove_stdlib",
    "remove_many_stdlib",
    # High-level functions — query
    "is_installed_stdlib",
    "list_installed_stdlib",
    "get_stdlib_info",
    # High-level functions — update
    "update_stdlib",
    "update_all_stdlib",
    "check_updates_stdlib",
    # High-level functions — sys.path
    "register_stdlib",
    "add_to_sys_path",
    "remove_from_sys_path",
    # High-level functions — cache
    "clear_cache",
    "get_cache_stats",
    # Exceptions
    "StdlibInstallerError",
    "NetworkError",
    "GitHubAPIError",
    "RateLimitError",
    "PackageNotFoundError",
    "ModuleNotInstalledError",
    "AlreadyInstalledError",
    "CompiledModuleError",
    "ChecksumVerificationError",
    "CircularDependencyError",
    "InstallationError",
]


from typing import List
def __dir__() -> List[str]:
	"""Show ONLY the public API."""
	return __all__
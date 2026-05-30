"""
Custom exceptions for the stdlib_installer package.

This module defines a hierarchy of exceptions used throughout the package
to provide clear, actionable error messages for different failure modes.
"""

from typing import Optional, List


class StdlibInstallerError(Exception):
    """
    Base exception for all errors raised by stdlib_installer.

    All custom exceptions in this package inherit from this class,
    allowing callers to catch all installer-related errors with a single
    except clause.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    original_error : Exception or None, optional
        The underlying exception that caused this error, if any.
        Used for exception chaining.

    Examples
    --------
    >>> try:
    ...     installer.install("nonexistent")
    ... except StdlibInstallerError as e:
    ...     print(f"Installation failed: {e}")
    """

    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error

    def __str__(self) -> str:
        if self.original_error:
            return f"{self.message} (caused by: {self.original_error})"
        return self.message


class NetworkError(StdlibInstallerError):
    """
    Raised when a network operation fails.

    This covers connection timeouts, DNS resolution failures,
    TLS/SSL errors, and other transport-level issues.

    Parameters
    ----------
    url : str
        The URL that was being accessed when the error occurred.
    message : str
        Description of the network failure.
    original_error : Exception or None, optional
        The underlying network exception.

    Examples
    --------
    >>> raise NetworkError(
    ...     url="https://api.github.com/repos/python/cpython",
    ...     message="Connection timed out",
    ...     original_error=TimeoutError()
    ... )
    """

    def __init__(
        self,
        url: str,
        message: str,
        original_error: Optional[Exception] = None
    ) -> None:
        self.url = url
        full_message = f"Network error accessing {url}: {message}"
        super().__init__(full_message, original_error)


class GitHubAPIError(StdlibInstallerError):
    """
    Raised when the GitHub API returns an error response.

    This covers rate limiting (HTTP 429), authentication failures (HTTP 401),
    not found (HTTP 404), and other API-level errors.

    Parameters
    ----------
    url : str
        The API endpoint that returned the error.
    status_code : int
        The HTTP status code.
    response_body : str
        The raw response body, useful for debugging.
    retry_after : int or None, optional
        If rate-limited, the number of seconds to wait before retrying.
        Extracted from the X-RateLimit-Reset or Retry-After header.

    Examples
    --------
    >>> raise GitHubAPIError(
    ...     url="https://api.github.com/repos/python/cpython/contents/Lib/json",
    ...     status_code=404,
    ...     response_body='{"message": "Not Found"}'
    ... )
    """

    def __init__(
        self,
        url: str,
        status_code: int,
        response_body: str,
        retry_after: Optional[int] = None
    ) -> None:
        self.url = url
        self.status_code = status_code
        self.response_body = response_body
        self.retry_after = retry_after

        message = f"GitHub API returned {status_code} for {url}"
        if retry_after:
            message += f" (retry after {retry_after}s)"
        super().__init__(message)


class RateLimitError(GitHubAPIError):
    """
    Raised specifically when the GitHub API rate limit is exceeded.

    This is a specialized form of GitHubAPIError for HTTP 429 responses
    with optional rate limit reset information.

    Parameters
    ----------
    url : str
        The API endpoint that was rate-limited.
    reset_time : str or None, optional
        ISO-formatted timestamp when the rate limit resets.
        Extracted from X-RateLimit-Reset header.
    remaining : int or None, optional
        Number of remaining requests (typically 0 when this is raised).

    Examples
    --------
    >>> raise RateLimitError(
    ...     url="https://api.github.com/repos/python/cpython/contents/Lib",
    ...     reset_time="2024-01-01T12:00:00Z",
    ...     remaining=0
    ... )
    """

    def __init__(
        self,
        url: str,
        reset_time: Optional[str] = None,
        remaining: Optional[int] = None
    ) -> None:
        self.reset_time = reset_time
        self.remaining = remaining

        message = f"GitHub API rate limit exceeded for {url}"
        if reset_time:
            message += f" (resets at {reset_time})"
        if remaining is not None:
            message += f" ({remaining} requests remaining)"

        super().__init__(
            url=url,
            status_code=429,
            response_body="Rate limit exceeded",
        )
        # Override the generic message with our detailed one
        self.message = message


class PackageNotFoundError(StdlibInstallerError):
    """
    Raised when a requested package or module cannot be found.

    This is raised when the GitHub API returns a 404 for the package path,
    or when a package is not found in the manifest during removal.

    Parameters
    ----------
    package_name : str
        The name of the package that was not found.
    version : str or None, optional
        The Python version that was searched, if applicable.
    available_alternatives : list of str or None, optional
        List of similarly-named packages that might be what the user intended.
        Suggested via Levenshtein distance or similar fuzzy matching.

    Examples
    --------
    >>> raise PackageNotFoundError(
    ...     package_name="jsn",
    ...     version="3.11",
    ...     available_alternatives=["json"]
    ... )
    """

    def __init__(
        self,
        package_name: str,
        version: Optional[str] = None,
        available_alternatives: Optional[List[str]] = None
    ) -> None:
        self.package_name = package_name
        self.version = version
        self.available_alternatives = available_alternatives or []

        parts = [f"Package '{package_name}' not found"]
        if version:
            parts.append(f"in Python {version}")
        if self.available_alternatives:
            alternatives = ", ".join(f"'{a}'" for a in self.available_alternatives)
            parts.append(f"(did you mean: {alternatives}?)")

        super().__init__(" ".join(parts))


class ModuleNotInstalledError(StdlibInstallerError):
    """
    Raised when attempting to operate on a module that is not installed.

    This differs from PackageNotFoundError in that the package exists
    in the repository but has not been installed locally yet.

    Parameters
    ----------
    module_name : str
        The name of the module that is not installed.

    Examples
    --------
    >>> raise ModuleNotInstalledError("xml")
    """

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        super().__init__(
            f"Module '{module_name}' is not installed. "
            f"Use 'stdlib install {module_name}' to install it."
        )


class AlreadyInstalledError(StdlibInstallerError):
    """
    Raised when attempting to install a package that is already installed.

    Parameters
    ----------
    package_name : str
        The package that is already installed.
    installed_path : str
        The filesystem path where it is installed.

    Examples
    --------
    >>> raise AlreadyInstalledError("json", installed_path="/home/user/.stdlib-packages/json")
    """

    def __init__(self, package_name: str, installed_path: str) -> None:
        self.package_name = package_name
        self.installed_path = installed_path
        super().__init__(
            f"Package '{package_name}' is already installed at {installed_path}. "
            f"Use --force to reinstall."
        )


class CompiledModuleError(StdlibInstallerError):
    """
    Raised when attempting to install a C extension module that cannot
    be distributed as pure Python source code.

    Compiled modules require platform-specific build tools and shared
    libraries that are not available through this installer. Users are
    directed to system package managers instead.

    Parameters
    ----------
    module_name : str
        The name of the compiled module.
    system_packages : dict or None, optional
        Mapping of system package managers to the corresponding package name.
        For example: {'apt': 'python3-tk', 'dnf': 'python3-tkinter'}.

    Examples
    --------
    >>> raise CompiledModuleError(
    ...     module_name="_ssl",
    ...     system_packages={"apt": "python3-openssl", "brew": "openssl"}
    ... )
    """

    def __init__(
        self,
        module_name: str,
        system_packages: Optional[dict] = None
    ) -> None:
        self.module_name = module_name
        self.system_packages = system_packages or {}

        message = (
            f"'{module_name}' is a compiled C extension and cannot be "
            f"installed from source. Install it via your system package manager."
        )
        if self.system_packages:
            suggestions = "; ".join(
                f"{mgr}: {pkg}" for mgr, pkg in self.system_packages.items()
            )
            message += f" Suggestions: {suggestions}"

        super().__init__(message)


class ChecksumVerificationError(StdlibInstallerError):
    """
    Raised when a downloaded file fails checksum verification.

    Parameters
    ----------
    file_name : str
        The file that failed verification.
    expected_hash : str
        The expected SHA-256 hash.
    actual_hash : str
        The actual computed hash of the downloaded file.

    Examples
    --------
    >>> raise ChecksumVerificationError(
    ...     file_name="json/__init__.py",
    ...     expected_hash="abc123...",
    ...     actual_hash="def456..."
    ... )
    """

    def __init__(
        self,
        file_name: str,
        expected_hash: str,
        actual_hash: str
    ) -> None:
        self.file_name = file_name
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        super().__init__(
            f"Checksum mismatch for '{file_name}': "
            f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
        )


class CircularDependencyError(StdlibInstallerError):
    """
    Raised when circular dependencies are detected during resolution.

    Parameters
    ----------
    chain : list of str
        The dependency chain showing the circular path.
        For example: ['a', 'b', 'c', 'a'].

    Examples
    --------
    >>> raise CircularDependencyError(chain=["pkg_a", "pkg_b", "pkg_a"])
    """

    def __init__(self, chain: List[str]) -> None:
        self.chain = chain
        chain_str = " -> ".join(chain)
        super().__init__(
            f"Circular dependency detected: {chain_str}"
        )


class RollbackError(StdlibInstallerError):
    """
    Raised when an installation rollback fails, leaving the system
    in a potentially inconsistent state.

    This is a critical error indicating that manual cleanup may be required.

    Parameters
    ----------
    package_name : str
        The package whose rollback failed.
    details : str
        Description of what went wrong during rollback.

    Examples
    --------
    >>> raise RollbackError(
    ...     package_name="json",
    ...     details="Failed to restore backup: permission denied"
    ... )
    """

    def __init__(self, package_name: str, details: str) -> None:
        self.package_name = package_name
        self.details = details
        super().__init__(
            f"Rollback failed for '{package_name}': {details}. "
            f"Manual cleanup may be required."
        )


class InstallationError(StdlibInstallerError):
    """
    Raised when a package installation fails for any reason.

    This is a generic exception that wraps lower-level errors during
    the installation process, including download failures, filesystem
    errors, and manifest write failures. It preserves the original
    exception for debugging while providing a user-facing message.

    Parameters
    ----------
    message : str
        Human-readable description of what failed.
    package_name : str or None, optional
        The name of the package being installed when the error occurred.
    original_error : Exception or None, optional
        The underlying exception that triggered this error.

    Attributes
    ----------
    package_name : str or None
        The package that failed to install.
    original_error : Exception or None
        The original exception for debugging.

    Examples
    --------
    >>> raise InstallationError(
    ...     message="Failed to download file",
    ...     package_name="json",
    ...     original_error=NetworkError("timeout")
    ... )
    """

    def __init__(
        self,
        message: str,
        package_name: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        self.package_name = package_name
        self.original_error = original_error

        parts = []
        if package_name:
            parts.append(f"Installation of '{package_name}' failed")
        else:
            parts.append("Installation failed")
        parts.append(message)

        full_message = ": ".join(parts)
        super().__init__(full_message, original_error)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom exceptions for the PackMan package manager.

This module defines a hierarchy of exceptions used throughout the PackMan
package to provide granular error handling for package management operations.

Exceptions
----------
PackageInstallerError
    Base exception for all packman errors.
PackageNotFoundError
    Raised when a package cannot be found locally or remotely.
PackageInstallError
    Raised when package installation fails.
PackageUninstallError
    Raised when package uninstallation fails.
PackageUpgradeError
    Raised when package upgrade fails.
PackageVersionError
    Raised when version parsing or comparison fails.
NetworkError
    Raised when network operations fail.
TimeoutError
    Raised when operations exceed their time limit.
PermissionError
    Raised when insufficient permissions prevent an operation.
CacheError
    Raised when cache operations fail.
ValidationError
    Raised when package validation fails.
SecurityError
    Raised when security checks fail.
"""

from typing import Optional, List, Dict, Any


class PackageInstallerError(Exception):
    """
    Base exception for all PackMan errors.

    All custom exceptions in the packman package inherit from this class.
    This allows catching all packman-related errors with a single except
    clause while still enabling specific error handling when needed.

    Parameters
    ----------
    message : str
        Human-readable error description.
    package_name : str, optional
        Name of the package involved in the error, if applicable.
    details : dict, optional
        Additional error context as key-value pairs.

    Examples
    --------
    >>> try:
    ...     raise PackageInstallerError("Operation failed", package_name="numpy")
    ... except PackageInstallerError as e:
    ...     print(e.package_name)
    numpy
    """

    def __init__(
        self,
        message: str,
        package_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.package_name = package_name
        self.details = details or {}
        full_message = message
        if package_name:
            full_message = f"[{package_name}] {message}"
        super().__init__(full_message)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception data to a dictionary representation.

        Returns
        -------
        dict
            Dictionary containing error type, message, package name, and
            additional details.

        Examples
        --------
        >>> err = PackageInstallerError("test", package_name="x", details={"code": 1})
        >>> d = err.to_dict()
        >>> d["type"]
        'PackageInstallerError'
        """
        return {
            "type": self.__class__.__name__,
            "message": str(self),
            "package_name": self.package_name,
            "details": self.details,
        }


class PackageNotFoundError(PackageInstallerError):
    """
    Raised when a package is not found in the local environment or remote index.

    This exception indicates that the requested package does not exist
    in the specified location or could not be resolved.

    Parameters
    ----------
    package_name : str
        Name of the package that was not found.
    message : str, optional
        Custom error message. If not provided, a default message is generated.
    location : str, optional
        Where the package was searched ('local', 'pypi', 'custom-index').
    details : dict, optional
        Additional error context.

    Examples
    --------
    >>> raise PackageNotFoundError("nonexistent-package", location="pypi")
    Traceback (most recent call last):
        ...
    packman.exceptions.PackageNotFoundError: [nonexistent-package] Package not found in pypi
    """

    def __init__(
        self,
        package_name: str,
        message: Optional[str] = None,
        location: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if message is None:
            location_str = f" in {location}" if location else ""
            message = f"Package not found{location_str}"
        self.location = location
        super().__init__(message, package_name=package_name, details=details)


class PackageInstallError(PackageInstallerError):
    """
    Raised when package installation fails.

    This exception wraps installation failures from pip or direct
    package management operations, providing context about what failed.

    Parameters
    ----------
    package_name : str
        Name of the package being installed.
    message : str, optional
        Custom error message describing the failure.
    exit_code : int, optional
        Exit code from the installation subprocess, if applicable.
    stderr : str, optional
        Standard error output from the failed installation.
    version : str, optional
        The version that was being installed, if specified.
    details : dict, optional
        Additional error context.

    Notes
    -----
    The `stderr` attribute often contains pip's error output, which can
    be useful for debugging dependency conflicts, network issues, or
    compilation errors.

    Examples
    --------
    >>> raise PackageInstallError(
    ...     "requests",
    ...     message="Failed to build wheel",
    ...     exit_code=1,
    ...     stderr="error: command 'gcc' failed"
    ... )
    Traceback (most recent call last):
        ...
    packman.exceptions.PackageInstallError: [requests] Failed to build wheel
    """

    def __init__(
        self,
        package_name: str,
        message: Optional[str] = None,
        exit_code: Optional[int] = None,
        stderr: Optional[str] = None,
        version: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if message is None:
            message = "Package installation failed"
        self.exit_code = exit_code
        self.stderr = stderr
        self.version = version
        if details is None:
            details = {}
        if exit_code is not None:
            details["exit_code"] = exit_code
        if version is not None:
            details["version"] = version
        super().__init__(message, package_name=package_name, details=details)


class PackageUninstallError(PackageInstallerError):
    """
    Raised when package uninstallation fails.

    Parameters
    ----------
    package_name : str
        Name of the package being uninstalled.
    message : str, optional
        Custom error message.
    exit_code : int, optional
        Exit code from the uninstallation subprocess.
    stderr : str, optional
        Standard error output from the failed operation.
    details : dict, optional
        Additional error context.

    Examples
    --------
    >>> raise PackageUninstallError("numpy", exit_code=1, stderr="Permission denied")
    Traceback (most recent call last):
        ...
    packman.exceptions.PackageUninstallError: [numpy] Package uninstallation failed
    """

    def __init__(
        self,
        package_name: str,
        message: Optional[str] = None,
        exit_code: Optional[int] = None,
        stderr: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if message is None:
            message = "Package uninstallation failed"
        self.exit_code = exit_code
        self.stderr = stderr
        if details is None:
            details = {}
        if exit_code is not None:
            details["exit_code"] = exit_code
        super().__init__(message, package_name=package_name, details=details)


class PackageUpgradeError(PackageInstallerError):
    """
    Raised when a package upgrade operation fails.

    This exception provides information about both the current and
    target versions to aid in debugging upgrade failures.

    Parameters
    ----------
    package_name : str
        Name of the package being upgraded.
    message : str, optional
        Custom error message.
    current_version : str, optional
        The version currently installed.
    target_version : str, optional
        The version being upgraded to.
    details : dict, optional
        Additional error context.

    Examples
    --------
    >>> raise PackageUpgradeError(
    ...     "django",
    ...     current_version="3.2.0",
    ...     target_version="4.0.0",
    ...     message="Incompatible dependencies"
    ... )
    Traceback (most recent call last):
        ...
    packman.exceptions.PackageUpgradeError: [django] Incompatible dependencies
    """

    def __init__(
        self,
        package_name: str,
        message: Optional[str] = None,
        current_version: Optional[str] = None,
        target_version: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if message is None:
            message = "Package upgrade failed"
        self.current_version = current_version
        self.target_version = target_version
        if details is None:
            details = {}
        if current_version is not None:
            details["current_version"] = current_version
        if target_version is not None:
            details["target_version"] = target_version
        super().__init__(message, package_name=package_name, details=details)


class PackageVersionError(PackageInstallerError):
    """
    Raised when version parsing, comparison, or resolution fails.

    This exception covers invalid version strings, incompatible version
    requirements, and failures in semantic version matching.

    Parameters
    ----------
    package_name : str
        Name of the package.
    message : str, optional
        Custom error message.
    version_string : str, optional
        The version string that caused the error.
    constraint : str, optional
        The version constraint being applied, if any.
    details : dict, optional
        Additional error context.

    Examples
    --------
    >>> raise PackageVersionError(
    ...     "flask",
    ...     version_string="not-a-version",
    ...     message="Invalid version format"
    ... )
    Traceback (most recent call last):
        ...
    packman.exceptions.PackageVersionError: [flask] Invalid version format
    """

    def __init__(
        self,
        package_name: str,
        message: Optional[str] = None,
        version_string: Optional[str] = None,
        constraint: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if message is None:
            message = "Version error"
        self.version_string = version_string
        self.constraint = constraint
        if details is None:
            details = {}
        if version_string is not None:
            details["version_string"] = version_string
        if constraint is not None:
            details["constraint"] = constraint
        super().__init__(message, package_name=package_name, details=details)


class NetworkError(PackageInstallerError):
    """
    Raised when a network operation fails.

    This exception covers DNS resolution failures, connection timeouts,
    HTTP errors, and other network-related issues when communicating
    with package indices.

    Parameters
    ----------
    message : str
        Description of the network failure.
    url : str, optional
        The URL being accessed when the error occurred.
    status_code : int, optional
        HTTP status code, if an HTTP response was received.
    package_name : str, optional
        Name of the package being fetched.
    details : dict, optional
        Additional error context.

    Notes
    -----
    This exception is often retryable. The `packman.retry` module
    can be used to automatically retry operations that raise this
    exception with exponential backoff.

    Examples
    --------
    >>> raise NetworkError(
    ...     "Connection refused",
    ...     url="https://pypi.org/simple/requests/",
    ...     package_name="requests"
    ... )
    Traceback (most recent call last):
        ...
    packman.exceptions.NetworkError: [requests] Connection refused
    """

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        package_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.url = url
        self.status_code = status_code
        if details is None:
            details = {}
        if url is not None:
            details["url"] = url
        if status_code is not None:
            details["status_code"] = status_code
        super().__init__(message, package_name=package_name, details=details)


class TimeoutError(PackageInstallerError):
    """
    Raised when an operation exceeds its configured time limit.

    Parameters
    ----------
    message : str
        Description of the timeout.
    timeout_seconds : int or float
        The timeout duration that was exceeded.
    operation : str, optional
        Description of the operation that timed out.
    package_name : str, optional
        Name of the package involved.
    details : dict, optional
        Additional error context.

    Examples
    --------
    >>> raise TimeoutError(
    ...     "Download timed out",
    ...     timeout_seconds=30,
    ...     operation="download",
    ...     package_name="tensorflow"
    ... )
    Traceback (most recent call last):
        ...
    packman.exceptions.TimeoutError: [tensorflow] Download timed out
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        operation: Optional[str] = None,
        package_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        if details is None:
            details = {}
        details["timeout_seconds"] = timeout_seconds
        if operation is not None:
            details["operation"] = operation
        super().__init__(message, package_name=package_name, details=details)


class PermissionError(PackageInstallerError):
    """
    Raised when file system permissions prevent an operation.

    This typically occurs when attempting to install packages system-wide
    without sufficient privileges, or when package files are read-only.

    Parameters
    ----------
    message : str
        Description of the permission issue.
    path : str, optional
        The file system path where permission was denied.
    operation : str, optional
        The operation being attempted ('write', 'read', 'execute', 'delete').
    package_name : str, optional
        Name of the package involved.
    details : dict, optional
        Additional error context.

    Notes
    -----
    Consider using the ``--user`` flag for user-level installations
    or creating a virtual environment to avoid permission issues.

    Examples
    --------
    >>> raise PermissionError(
    ...     "Cannot write to site-packages",
    ...     path="/usr/lib/python3.10/site-packages",
    ...     operation="write"
    ... )
    Traceback (most recent call last):
        ...
    packman.exceptions.PermissionError: Cannot write to site-packages
    """

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        operation: Optional[str] = None,
        package_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.path = path
        self.operation = operation
        if details is None:
            details = {}
        if path is not None:
            details["path"] = path
        if operation is not None:
            details["operation"] = operation
        super().__init__(message, package_name=package_name, details=details)


class CacheError(PackageInstallerError):
    """
    Raised when cache operations fail.

    This exception covers cache read/write errors, corruption detection,
    and cache directory permission issues.

    Parameters
    ----------
    message : str
        Description of the cache error.
    cache_path : str, optional
        Path to the cache file or directory.
    operation : str, optional
        The cache operation that failed ('read', 'write', 'clear', 'verify').
    package_name : str, optional
        Name of the package associated with the cached data.
    details : dict, optional
        Additional error context.

    Examples
    --------
    >>> raise CacheError(
    ...     "Cache file corrupted",
    ...     cache_path="/home/user/.cache/packman/requests.json",
    ...     operation="read",
    ...     package_name="requests"
    ... )
    Traceback (most recent call last):
        ...
    packman.exceptions.CacheError: [requests] Cache file corrupted
    """

    def __init__(
        self,
        message: str,
        cache_path: Optional[str] = None,
        operation: Optional[str] = None,
        package_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cache_path = cache_path
        self.operation = operation
        if details is None:
            details = {}
        if cache_path is not None:
            details["cache_path"] = cache_path
        if operation is not None:
            details["operation"] = operation
        super().__init__(message, package_name=package_name, details=details)


class ValidationError(PackageInstallerError):
    """
    Raised when package validation fails.

    This covers hash mismatches, signature verification failures,
    and other integrity checks.

    Parameters
    ----------
    message : str
        Description of the validation failure.
    package_name : str
        Name of the package being validated.
    expected_hash : str, optional
        The expected hash value.
    actual_hash : str, optional
        The computed hash value.
    algorithm : str, optional
        The hash algorithm used ('sha256', 'sha512', 'md5').
    details : dict, optional
        Additional error context.

    Examples
    --------
    >>> raise ValidationError(
    ...     "Hash mismatch",
    ...     package_name="requests",
    ...     expected_hash="abc123...",
    ...     actual_hash="def456...",
    ...     algorithm="sha256"
    ... )
    Traceback (most recent call last):
        ...
    packman.exceptions.ValidationError: [requests] Hash mismatch
    """

    def __init__(
        self,
        message: str,
        package_name: str,
        expected_hash: Optional[str] = None,
        actual_hash: Optional[str] = None,
        algorithm: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        self.algorithm = algorithm
        if details is None:
            details = {}
        if expected_hash is not None:
            details["expected_hash"] = expected_hash
        if actual_hash is not None:
            details["actual_hash"] = actual_hash
        if algorithm is not None:
            details["algorithm"] = algorithm
        super().__init__(message, package_name=package_name, details=details)


class SecurityError(PackageInstallerError):
    """
    Raised when security checks fail.

    This exception covers certificate errors, untrusted sources,
    and other security-related issues.

    Parameters
    ----------
    message : str
        Description of the security issue.
    package_name : str, optional
        Name of the package involved.
    source : str, optional
        The source that triggered the security error (URL, hostname).
    reason : str, optional
        Specific security reason ('untrusted-host', 'expired-cert', 'revoked-key').
    details : dict, optional
        Additional error context.

    Warnings
    --------
    Never bypass security checks in production environments. If you
    encounter this error, verify the package source and consider using
    ``--trusted-host`` only for known, internal mirrors.

    Examples
    --------
    >>> raise SecurityError(
    ...     "Untrusted host",
    ...     package_name="private-package",
    ...     source="internal-pypi.example.com",
    ...     reason="untrusted-host"
    ... )
    Traceback (most recent call last):
        ...
    packman.exceptions.SecurityError: [private-package] Untrusted host
    """

    def __init__(
        self,
        message: str,
        package_name: Optional[str] = None,
        source: Optional[str] = None,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.source = source
        self.reason = reason
        if details is None:
            details = {}
        if source is not None:
            details["source"] = source
        if reason is not None:
            details["reason"] = reason
        super().__init__(message, package_name=package_name, details=details)
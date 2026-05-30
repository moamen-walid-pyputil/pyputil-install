#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom Exception Hierarchy for Python Headers Installer.

This module defines a comprehensive exception hierarchy for handling
all error scenarios that may arise during Python header installation.
Each exception class is designed to carry contextual information
about the failure, enabling precise error handling and debugging.

Exception Hierarchy:
    HeaderInstallError (Base)
    ├── ConfigurationError
    ├── PlatformNotSupportedError
    ├── NetworkError
    │   ├── DownloadError
    │   └── VerificationError
    ├── ExtractionError
    ├── FileSystemError
    │   ├── BackupError
    │   └── InstallationError
    └── ValidationError
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union
import sys
import traceback


class HeaderInstallError(Exception):
    """
    Base exception for all header installation errors.
    
    This is the root of the exception hierarchy. All custom exceptions
    inherit from this class to allow broad exception catching when
    specific error types are not required.
    
    Parameters
    ----------
    message : str
        Human-readable error description
    details : Optional[Dict[str, Any]]
        Additional contextual information about the error
    cause : Optional[Exception]
        The original exception that caused this error, if any
    
    Attributes
    ----------
    timestamp : float
        Unix timestamp when the exception was created
    traceback_str : str
        String representation of the traceback at exception creation
    
    Examples
    --------
    >>> try:
    ...     raise HeaderInstallError("Installation failed")
    ... except HeaderInstallError as e:
    ...     print(str(e))
    Installation failed
    
    >>> try:
    ...     raise HeaderInstallError(
    ...         "Download failed",
    ...         details={"url": "https://example.com", "attempt": 3},
    ...         cause=ConnectionError("Network unreachable")
    ...     )
    ... except HeaderInstallError as e:
    ...     print(e.details["url"])
    https://example.com
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause
        self.timestamp = __import__('time').time()
        self.traceback_str = self._capture_traceback()
    
    def _capture_traceback(self) -> str:
        """
        Capture the current traceback as a string.
        
        Returns
        -------
        str
            Formatted traceback string
        """
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_tb is not None:
            return ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        return traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the exception to a dictionary representation.
        
        Useful for logging, serialization, or API responses.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all exception information
        """
        result = {
            "exception_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }
        
        if self.cause:
            result["cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause)
            }
        
        return result
    
    def get_error_chain(self) -> str:
        """
        Get the complete chain of error messages from root cause to this exception.
        
        Returns
        -------
        str
            Formatted error chain
        """
        chain = [f"{self.__class__.__name__}: {self.message}"]
        current = self.cause
        
        while current is not None:
            chain.append(f"  Caused by: {type(current).__name__}: {str(current)}")
            current = getattr(current, 'cause', None)
        
        return "\n".join(chain)


class ConfigurationError(HeaderInstallError):
    """
    Raised when there is an invalid configuration.
    
    This exception indicates that the provided configuration parameters
    are invalid, missing, or contradictory.
    
    Parameters
    ----------
    parameter : str
        The configuration parameter that caused the error
    value : Any
        The invalid value that was provided
    expected : Optional[str]
        Description of expected valid values
    
    Examples
    --------
    >>> raise ConfigurationError(
    ...     "Invalid version format",
    ...     parameter="version",
    ...     value="3.x",
    ...     expected="Version in format 'major.minor.patch'"
    ... )
    """
    
    def __init__(
        self,
        message: str,
        parameter: str,
        value: Any,
        expected: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        details = {
            "parameter": parameter,
            "invalid_value": value,
            "expected": expected
        }
        details.update(kwargs.get('details', {}))
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class PlatformNotSupportedError(HeaderInstallError):
    """
    Raised when the current platform is not supported.
    
    This exception indicates that the installer cannot operate on
    the current operating system or architecture.
    
    Parameters
    ----------
    platform : str
        The unsupported platform identifier
    supported_platforms : Optional[List[str]]
        List of supported platforms
    
    Examples
    --------
    >>> raise PlatformNotSupportedError(
    ...     "Platform not supported",
    ...     platform="Solaris",
    ...     supported_platforms=["Linux", "Windows", "macOS", "Termux"]
    ... )
    """
    
    def __init__(
        self,
        message: str,
        platform: str,
        supported_platforms: Optional[list] = None,
        **kwargs: Any
    ) -> None:
        details = {
            "current_platform": platform,
            "supported_platforms": supported_platforms or []
        }
        details.update(kwargs.get('details', {}))
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class NetworkError(HeaderInstallError):
    """
    Base exception for network-related errors.
    
    This is the parent class for all errors related to network operations
    including downloads, connection issues, and timeouts.
    
    Parameters
    ----------
    url : str
        The URL involved in the network operation
    status_code : Optional[int]
        HTTP status code if applicable
    response_headers : Optional[Dict[str, str]]
        Response headers if available
    """
    
    def __init__(
        self,
        message: str,
        url: str,
        status_code: Optional[int] = None,
        response_headers: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> None:
        details = {
            "url": url,
            "status_code": status_code,
            "response_headers": response_headers or {}
        }
        details.update(kwargs.get('details', {}))
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class DownloadError(NetworkError):
    """
    Raised when a file download fails.
    
    This exception provides detailed information about the failed download
    including the URL, number of attempts, and the specific failure reason.
    
    Parameters
    ----------
    attempt : int
        The attempt number when the failure occurred
    total_attempts : int
        The total number of attempts configured
    file_size_expected : Optional[int]
        Expected file size in bytes
    file_size_received : Optional[int]
        Actually received file size in bytes
    
    Examples
    --------
    >>> raise DownloadError(
    ...     "Download failed after multiple attempts",
    ...     url="https://github.com/python/cpython/archive/v3.9.5.zip",
    ...     attempt=3,
    ...     total_attempts=3,
    ...     cause=ConnectionError("Timeout")
    ... )
    """
    
    def __init__(
        self,
        message: str,
        url: str,
        attempt: int = 1,
        total_attempts: int = 1,
        file_size_expected: Optional[int] = None,
        file_size_received: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        details = {
            "attempt": attempt,
            "total_attempts": total_attempts,
            "file_size_expected": file_size_expected,
            "file_size_received": file_size_received,
            "download_percentage": (
                (file_size_received / file_size_expected * 100)
                if file_size_expected and file_size_received
                else None
            )
        }
        details.update(kwargs.get('details', {}))
        kwargs['details'] = details
        super().__init__(message, url=url, **kwargs)


class VerificationError(NetworkError):
    """
    Raised when file verification fails after download.
    
    This exception indicates that the downloaded file does not match
    the expected checksum, suggesting corruption or tampering.
    
    Parameters
    ----------
    expected_hash : Optional[str]
        The expected hash value
    actual_hash : Optional[str]
        The computed hash value
    algorithm : str
        The hash algorithm used (e.g., 'sha256', 'md5')
    
    Examples
    --------
    >>> raise VerificationError(
    ...     "SHA256 mismatch",
    ...     url="https://github.com/python/cpython/archive/v3.9.5.zip",
    ...     expected_hash="abc123...",
    ...     actual_hash="def456...",
    ...     algorithm="sha256"
    ... )
    """
    
    def __init__(
        self,
        message: str,
        url: str,
        expected_hash: Optional[str] = None,
        actual_hash: Optional[str] = None,
        algorithm: str = "sha256",
        **kwargs: Any
    ) -> None:
        details = {
            "hash_algorithm": algorithm,
            "expected_hash": expected_hash,
            "actual_hash": actual_hash,
            "hash_mismatch": expected_hash != actual_hash
        }
        details.update(kwargs.get('details', {}))
        kwargs['details'] = details
        super().__init__(message, url=url, **kwargs)


class ExtractionError(HeaderInstallError):
    """
    Raised when archive extraction fails.
    
    This exception provides information about the archive that failed
    to extract, including its format and the target directory.
    
    Parameters
    ----------
    archive_path : Union[str, Path]
        Path to the archive file
    target_path : Union[str, Path]
        Directory where extraction was attempted
    archive_format : Optional[str]
        Format of the archive ('zip', 'tar.xz', etc.)
    
    Examples
    --------
    >>> raise ExtractionError(
    ...     "Corrupted archive",
    ...     archive_path="/tmp/python_source.zip",
    ...     target_path="/tmp/extracted",
    ...     archive_format="zip"
    ... )
    """
    
    def __init__(
        self,
        message: str,
        archive_path: Union[str, Path],
        target_path: Union[str, Path],
        archive_format: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        details = {
            "archive_path": str(archive_path),
            "target_path": str(target_path),
            "archive_format": archive_format
        }
        details.update(kwargs.get('details', {}))
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class FileSystemError(HeaderInstallError):
    """
    Base exception for file system operations.
    
    This is the parent class for errors related to file system operations
    such as copying, moving, deleting files and directories.
    
    Parameters
    ----------
    path : Union[str, Path]
        The file system path involved in the operation
    operation : str
        Description of the operation being performed
    """
    
    def __init__(
        self,
        message: str,
        path: Union[str, Path],
        operation: str,
        **kwargs: Any
    ) -> None:
        details = {
            "path": str(path),
            "operation": operation,
            "path_exists": Path(path).exists() if path else False,
            "path_type": (
                "directory" if Path(path).is_dir()
                else "file" if Path(path).is_file()
                else "nonexistent"
            ) if path and Path(path).exists() else "unknown"
        }
        details.update(kwargs.get('details', {}))
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class BackupError(FileSystemError):
    """
    Raised when a backup operation fails.
    
    This exception indicates failure in creating, restoring, or managing
    backups of the target directory.
    
    Parameters
    ----------
    source_path : Union[str, Path]
        The original directory being backed up
    backup_path : Union[str, Path]
        The intended backup location
    
    Examples
    --------
    >>> raise BackupError(
    ...     "Backup directory creation failed",
    ...     path="/usr/include/python3.9",
    ...     operation="backup",
    ...     source_path="/usr/include/python3.9",
    ...     backup_path="/usr/include/python3.9.backup"
    ... )
    """
    
    def __init__(
        self,
        message: str,
        path: Union[str, Path],
        operation: str,
        source_path: Union[str, Path],
        backup_path: Union[str, Path],
        **kwargs: Any
    ) -> None:
        details = {
            "source_path": str(source_path),
            "backup_path": str(backup_path)
        }
        details.update(kwargs.get('details', {}))
        kwargs['details'] = details
        super().__init__(message, path=path, operation=operation, **kwargs)


class InstallationError(FileSystemError):
    """
    Raised when header installation fails.
    
    This exception provides details about which headers failed to install
    and the reason for failure.
    
    Parameters
    ----------
    source_path : Union[str, Path]
        The source directory containing the headers
    files_copied : int
        Number of files successfully copied
    files_failed : int
        Number of files that failed to copy
    failed_files : Optional[List[str]]
        List of file names that failed to copy
    
    Examples
    --------
    >>> raise InstallationError(
    ...     "Failed to copy some header files",
    ...     path="/usr/include/python3.9",
    ...     operation="install",
    ...     source_path="/tmp/extracted/Include",
    ...     files_copied=42,
    ...     files_failed=3,
    ...     failed_files=["pyconfig.h", "abstract.h"]
    ... )
    """
    
    def __init__(
        self,
        message: str,
        path: Union[str, Path],
        operation: str,
        source_path: Union[str, Path],
        files_copied: int = 0,
        files_failed: int = 0,
        failed_files: Optional[list] = None,
        **kwargs: Any
    ) -> None:
        details = {
            "source_path": str(source_path),
            "files_copied": files_copied,
            "files_failed": files_failed,
            "failed_files": failed_files or [],
            "success_rate": (
                files_copied / (files_copied + files_failed) * 100
                if (files_copied + files_failed) > 0
                else 0
            )
        }
        details.update(kwargs.get('details', {}))
        kwargs['details'] = details
        super().__init__(message, path=path, operation=operation, **kwargs)


class ValidationError(HeaderInstallError):
    """
    Raised when input validation fails.
    
    This exception is raised when user-provided inputs do not meet
    the required format or constraints.
    
    Parameters
    ----------
    field : str
        The field or parameter that failed validation
    constraint : str
        Description of the validation constraint
    value : Any
        The value that failed validation
    
    Examples
    --------
    >>> raise ValidationError(
    ...     "Invalid Python version format",
    ...     field="version",
    ...     constraint="Must match pattern 'X.Y.Z'",
    ...     value="python3.9"
    ... )
    """
    
    def __init__(
        self,
        message: str,
        field: str,
        constraint: str,
        value: Any,
        **kwargs: Any
    ) -> None:
        details = {
            "field": field,
            "constraint": constraint,
            "invalid_value": value,
            "value_type": type(value).__name__
        }
        details.update(kwargs.get('details', {}))
        kwargs['details'] = details
        super().__init__(message, **kwargs)
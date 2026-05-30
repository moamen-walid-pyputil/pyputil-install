#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python Headers Installer
========================

Downloads, extracts, and installs CPython C development header files
into the target include directory. Useful in environments where Python
development headers are not pre-installed, such as minimal containers,
CI/CD runners, or embedded systems.

The package provides both a programmatic API and a command-line interface.
The main entry point for installation is install_python_headers().

Package Structure
-----------------

pyheaders_installer/
    __init__.py          This file. Public API surface.
    exceptions.py        Custom exception hierarchy.
    config.py            Configuration dataclasses and enums.
    downloader.py        HTTP download manager with retry/resume.
    extractor.py         ZIP and tar.xz archive extraction.
    installer.py         Header file copy and verification.
    cli.py               Command-line argument parsing and execution.

Public API
----------

Functions:
    install_python_headers    Main installation function.
    get_python_version        Return current Python version string.
    get_target_include_dir    Return the system include path.

Exceptions:
    HeaderInstallError        Base exception.
    DownloadError             Download failure.
    ExtractionError           Archive extraction failure.
    InstallationError         Header copy failure.
    VerificationError         Integrity check failure.
    ConfigurationError        Invalid configuration.
    PlatformNotSupportedError Unsupported platform.

Configuration:
    InstallConfig             Master configuration dataclass.
    NetworkConfig             Download/retry settings.
    VerificationConfig        Hash verification settings.
    BackupConfig              Backup behavior settings.
    PathConfig                File system path settings.
    ExtractionConfig          Archive extraction settings.

Enums:
    DownloadSource            github, python.org, custom.
    HashAlgorithm             sha256, sha384, sha512, md5.
    ArchiveFormat             zip, tar.xz, auto.
    OverwritePolicy           skip, overwrite, backup.

Usage Examples
--------------

Programmatic:

    >>> from pyheaders_installer import install_python_headers
    >>> path = install_python_headers(version='3.11.0')
    >>> print(path)
    /usr/include/python3.11

    >>> from pyheaders_installer import install_python_headers
    >>> path = install_python_headers(
    ...     version='3.9.5',
    ...     target_dir='./my_headers',
    ...     source='python.org',
    ...     verbose=True,
    ... )
    >>> print(path)
    ./my_headers

    >>> from pyheaders_installer.config import InstallConfig, DownloadSource
    >>> config = InstallConfig(
    ...     version='3.10.0',
    ...     source=DownloadSource.PYTHON_ORG,
    ...     verbose=True,
    ... )
    >>> from pyheaders_installer import install_python_headers
    >>> path = install_python_headers(
    ...     version=config.version,
    ...     source=config.source.value,
    ...     verbose=config.verbose,
    ... )

Command-line:

    python -m pyheaders_installer --version 3.11 --verbose
    python -m pyheaders_installer --target-dir ./headers --clean-existing
    python -m pyheaders_installer --source python.org --backup
    python -m pyheaders_installer --info
    python -m pyheaders_installer --clean

Warnings
--------
- Installing to system directories (e.g., /usr/include) may require
  elevated privileges (root or sudo).
- The package downloads source archives from the internet. A working
  network connection is required unless the archive is provided locally.
- On minimal systems without CA certificates, HTTPS downloads will fail.
  Install ca-certificates or use a custom HTTP URL.
- Temporary files are created in the system temp directory. Ensure
  sufficient disk space (approximately 100 MiB for full CPython source).
- Checkpoint files accumulate in ~/.cache/pyheaders_installer/.
  Run with --clean or call the clean function periodically.
"""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Public exception re-exports
# ---------------------------------------------------------------------------

from .exceptions import (
    HeaderInstallError,
    DownloadError,
    ExtractionError,
    InstallationError,
    VerificationError,
    ConfigurationError,
    PlatformNotSupportedError,
    NetworkError,
    BackupError,
    FileSystemError,
    ValidationError,
)

# ---------------------------------------------------------------------------
# Public configuration re-exports
# ---------------------------------------------------------------------------

from .config import (
    InstallConfig,
    NetworkConfig,
    VerificationConfig,
    BackupConfig,
    PathConfig,
    ExtractionConfig,
    DownloadSource,
    HashAlgorithm,
    PlatformPreset,
    LogLevel,
    SystemInfo,
)

# ---------------------------------------------------------------------------
# Public installer re-exports
# ---------------------------------------------------------------------------

from .installer import (
    HeaderInstaller,
    InstallResult,
    VerificationResult,
    OverwritePolicy,
    AtomicMode,
    BackupManager,
    HeaderVerifier,
    find_include_directory,
    copy_python_headers,
)

# ---------------------------------------------------------------------------
# Public downloader re-exports
# ---------------------------------------------------------------------------

from .downloader import (
    DownloadManager,
    DownloadProgress,
    DownloadStatus,
    ChunkScheduler,
    SpeedTracker,
    CheckpointManager,
    IntegrityVerifier,
    ConnectionPool,
    format_bytes,
    format_duration,
    calculate_jitter,
)

# ---------------------------------------------------------------------------
# Public extractor re-exports
# ---------------------------------------------------------------------------

from .extractor import (
    ArchiveExtractor,
    ExtractionResult,
    ArchiveFormat,
    ZipExtractor,
    TarXzExtractor,
    PathValidator,
    EntryFilter,
    extract_python_headers_archive,
)

# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def get_python_version() -> str:
    """
    Return the current Python interpreter version.

    Returns
    -------
    str
        Version string in 'major.minor.micro' format.

    Examples
    --------
    >>> get_python_version()
    '3.11.0'
    """
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


def get_target_include_dir() -> Path:
    """
    Return the system include directory for the current Python.

    Uses sysconfig.get_paths() to locate the include path where
    Python header files are expected.

    Returns
    -------
    Path
        Path to the include directory.

    Examples
    --------
    >>> path = get_target_include_dir()
    >>> path.name.startswith('python')
    True
    """
    include_path = sysconfig.get_paths().get("include")
    if include_path:
        return Path(include_path)
    return Path(sys.prefix) / "include"


def install_python_headers(
    version: Optional[str] = None,
    target_dir: Optional[Union[str, Path]] = None,
    retries: int = 3,
    retry_delay: float = 2.0,
    clean_existing: bool = False,
    backup_existing: bool = True,
    verbose: bool = False,
    include_subdirs: bool = True,
    source: str = "github",
    custom_url: Optional[str] = None,
    verify: bool = True,
    verify_algorithm: str = "sha256",
    atomic: bool = True,
    timeout: float = 30.0,
) -> Optional[str]:
    """
    Install Python C development headers from source.

    Downloads the CPython source archive for the specified version,
    extracts the Include directory, and copies header files to the
    target include directory.

    Parameters
    ----------
    version : Optional[str]
        Python version string (e.g., '3.11.0'). If None, uses the
        current interpreter version.
    target_dir : Optional[Union[str, Path]]
        Target directory for header installation. If None, uses the
        system include directory from sysconfig.
    retries : int
        Maximum number of download retry attempts (default: 3).
    retry_delay : float
        Base delay between retries in seconds. Actual delay uses
        exponential backoff with jitter (default: 2.0).
    clean_existing : bool
        If True, remove the target directory before installation
        (default: False).
    backup_existing : bool
        If True, create a backup of existing headers before
        overwriting (default: True).
    verbose : bool
        Enable detailed progress logging (default: False).
    include_subdirs : bool
        If True, copy subdirectories like cpython/ and internal/
        (default: True).
    source : str
        Download source: 'github' or 'python.org' (default: 'github').
    custom_url : Optional[str]
        Custom download URL template with {version} placeholder.
        Overrides the source parameter.
    verify : bool
        If True, verify downloaded file integrity using the specified
        hash algorithm (default: True).
    verify_algorithm : str
        Hash algorithm for verification: 'sha256', 'sha384', 'sha512',
        or 'md5' (default: 'sha256').
    atomic : bool
        If True, use atomic installation via a staging directory to
        prevent partial installations (default: True).
    timeout : float
        Connection timeout in seconds (default: 30.0).

    Returns
    -------
    Optional[str]
        String path to the installed headers directory if successful,
        None if the installation failed.

    Raises
    ------
    DownloadError
        If the source archive cannot be downloaded after all retries.
    ExtractionError
        If the downloaded archive is corrupt or unreadable.
    InstallationError
        If header files cannot be copied to the target directory.
    VerificationError
        If integrity verification is enabled and the downloaded file
        does not match the expected hash.
    ConfigurationError
        If the provided configuration is invalid.
    PlatformNotSupportedError
        If the current platform is not supported.

    Examples
    --------
    >>> # Install headers for the current Python version
    >>> path = install_python_headers()
    >>> print(path)
    /usr/include/python3.11

    >>> # Install headers for a specific version with backup
    >>> path = install_python_headers(
    ...     version='3.9.5',
    ...     backup_existing=True,
    ...     verbose=True,
    ... )

    >>> # Install from python.org to a custom directory
    >>> path = install_python_headers(
    ...     version='3.10.0',
    ...     target_dir='./local_headers',
    ...     source='python.org',
    ...     clean_existing=True,
    ... )

    >>> # Use a custom download URL
    >>> path = install_python_headers(
    ...     version='3.11.0',
    ...     custom_url='https://mirror.example.com/Python-{version}.tar.xz',
    ... )

    Notes
    -----
    - This function is the primary entry point for the package.
    - Temporary files are created in the system temp directory and
      cleaned up automatically.
    - On Unix systems, installing to /usr/include typically requires
      root privileges.
    - The function is synchronous and will block until the installation
      completes or fails.
    """
    import logging
    import tempfile
    import shutil
    from urllib.parse import urlparse

    from .downloader import DownloadManager
    from .extractor import ArchiveExtractor, ArchiveFormat, find_directory_with_file
    from .installer import (
        HeaderInstaller,
        OverwritePolicy,
        AtomicMode,
        BackupConfig,
        VerificationConfig as InstallVerificationConfig,
    )

    logger = logging.getLogger(__name__)

    # -----------------------------------------------------------------------
    # Determine version and target directory
    # -----------------------------------------------------------------------

    if version is None:
        version = get_python_version()

    if target_dir is None:
        target_dir = get_target_include_dir()
    else:
        target_dir = Path(target_dir)

    if verbose:
        logger.info(f"Installing Python {version} headers")
        logger.info(f"Target directory: {target_dir}")

    # -----------------------------------------------------------------------
    # Build download URL
    # -----------------------------------------------------------------------

    if custom_url:
        url = custom_url.format(version=version)
        archive_format = ArchiveFormat.AUTO
    elif source == "github":
        url = f"https://github.com/python/cpython/archive/refs/tags/v{version}.zip"
        archive_format = ArchiveFormat.ZIP
    elif source == "python.org":
        url = f"https://www.python.org/ftp/python/{version}/Python-{version}.tar.xz"
        archive_format = ArchiveFormat.TAR_XZ
    else:
        raise ConfigurationError(
            f"Unsupported source: {source}",
            parameter="source",
            value=source,
            expected="'github' or 'python.org'",
        )

    if verbose:
        logger.info(f"Download URL: {url}")

    # -----------------------------------------------------------------------
    # Set up temporary directory
    # -----------------------------------------------------------------------

    temp_dir = Path(tempfile.mkdtemp(prefix="pyheaders_"))
    archive_path = temp_dir / f"python_source{_get_archive_suffix(source, custom_url)}"
    extract_dir = temp_dir / "extracted"

    if verbose:
        logger.info(f"Temporary directory: {temp_dir}")

    try:
        # -------------------------------------------------------------------
        # Download the archive
        # -------------------------------------------------------------------

        if verbose:
            logger.info("Downloading source archive...")

        download_manager = DownloadManager(
            network_config=NetworkConfig(
                retries=retries,
                retry_delay=retry_delay,
                timeout=timeout,
            ),
            verification_config=VerificationConfig(
                enabled=verify,
                algorithm=HashAlgorithm(verify_algorithm),
            ),
        )

        if verbose:
            download_manager.on_progress = lambda p: logger.debug(p.format_human())
            download_manager.on_status_change = lambda s, m: logger.debug(f"{s.name}: {m}")

        download_success = download_manager.download(url, archive_path)
        if not download_success:
            logger.error("Download failed")
            return None

        if verbose:
            logger.info(f"Downloaded: {archive_path} ({archive_path.stat().st_size} bytes)")

        # -------------------------------------------------------------------
        # Extract the archive
        # -------------------------------------------------------------------

        if verbose:
            logger.info("Extracting archive...")

        extractor = ArchiveExtractor(
            target_dir=extract_dir,
            max_file_size=10 * 1024 * 1024,  # 10 MiB per file
        )

        result = extractor.extract(archive_path, format=archive_format)

        if verbose:
            logger.info(result.summary())

        if not result:
            logger.error("Extraction produced no files")
            return None

        # -------------------------------------------------------------------
        # Find the Include directory
        # -------------------------------------------------------------------

        include_source = find_directory_with_file(
            extract_dir,
            target_dir_name="Include",
            target_file_name="Python.h",
        )

        if include_source is None:
            logger.error(
                f"Could not find Include directory with Python.h in {extract_dir}"
            )
            return None

        if verbose:
            logger.info(f"Found Include directory: {include_source}")

        # -------------------------------------------------------------------
        # Install headers
        # -------------------------------------------------------------------

        if verbose:
            logger.info("Installing headers...")

        overwrite_policy = OverwritePolicy.BACKUP if backup_existing else OverwritePolicy.OVERWRITE
        if clean_existing:
            overwrite_policy = OverwritePolicy.OVERWRITE

        atomic_mode = AtomicMode.ENABLED if atomic else AtomicMode.DISABLED

        installer = HeaderInstaller(
            source_dir=include_source,
            target_dir=target_dir,
            include_subdirs=include_subdirs,
            overwrite_policy=overwrite_policy,
            backup_config=BackupConfig() if backup_existing else None,
            atomic_mode=atomic_mode,
        )

        install_result = installer.install()

        if verbose:
            logger.info(install_result.summary())

        if not install_result:
            logger.error(f"Installation failed: {install_result.failed_files}")
            return None

        if verbose:
            logger.info(f"Headers installed to: {target_dir}")

        return str(target_dir)

    finally:
        # -------------------------------------------------------------------
        # Clean up temporary files
        # -------------------------------------------------------------------

        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                if verbose:
                    logger.info(f"Cleaned up: {temp_dir}")
            except OSError as e:
                logger.warning(f"Failed to clean up {temp_dir}: {e}")


def _get_archive_suffix(source: str, custom_url: Optional[str] = None) -> str:
    """
    Return the expected archive file extension for the given source.

    Parameters
    ----------
    source : str
        Download source identifier.
    custom_url : Optional[str]
        Custom URL that may contain a recognizable extension.

    Returns
    -------
    str
        File extension including the dot (e.g., '.zip', '.tar.xz').
    """
    if custom_url:
        custom_lower = custom_url.lower()
        if '.tar.xz' in custom_lower or '.txz' in custom_lower:
            return '.tar.xz'
        if '.tar.gz' in custom_lower or '.tgz' in custom_lower:
            return '.tar.gz'
        if '.zip' in custom_lower:
            return '.zip'

    if source == "python.org":
        return '.tar.xz'
    return '.zip'


def clean_cache() -> int:
    """
    Remove all cached checkpoint and temporary files.

    Searches default cache directories for files created by the
    installer and removes them.

    Returns
    -------
    int
        Number of directories removed.

    Examples
    --------
    >>> count = clean_cache()
    >>> isinstance(count, int)
    True
    """
    import shutil

    cache_dirs = [
        Path.home() / '.cache' / 'pyheaders_installer',
        Path.home() / '.local' / 'share' / 'pyheaders_installer',
    ]

    removed = 0
    for directory in cache_dirs:
        if directory.exists():
            try:
                shutil.rmtree(directory)
                removed += 1
            except OSError:
                pass

    return removed


# ---------------------------------------------------------------------------
# Module-level metadata
# ---------------------------------------------------------------------------

__all__ = [
    # Functions
    'install_python_headers',
    'get_python_version',
    'get_target_include_dir',
    'clean_cache',
    'format_bytes',
    'format_duration',
    'calculate_jitter',
    'find_include_directory',
    'find_directory_with_file',
    'copy_python_headers',
    'extract_python_headers_archive',
    # Exceptions
    'HeaderInstallError',
    'DownloadError',
    'ExtractionError',
    'InstallationError',
    'VerificationError',
    'ConfigurationError',
    'PlatformNotSupportedError',
    'NetworkError',
    'BackupError',
    'FileSystemError',
    'ValidationError',
    # Configuration
    'InstallConfig',
    'NetworkConfig',
    'VerificationConfig',
    'BackupConfig',
    'PathConfig',
    'ExtractionConfig',
    'SystemInfo',
    # Enums
    'DownloadSource',
    'HashAlgorithm',
    'PlatformPreset',
    'LogLevel',
    'ArchiveFormat',
    'OverwritePolicy',
    'AtomicMode',
    'DownloadStatus',
    # Core classes
    'DownloadManager',
    'DownloadProgress',
    'SpeedTracker',
    'CheckpointManager',
    'IntegrityVerifier',
    'ConnectionPool',
    'ChunkScheduler',
    'ArchiveExtractor',
    'ExtractionResult',
    'ZipExtractor',
    'TarXzExtractor',
    'PathValidator',
    'EntryFilter',
    'HeaderInstaller',
    'InstallResult',
    'VerificationResult',
    'BackupManager',
    'HeaderVerifier',
]


# ---------------------------------------------------------------------------
# Entry point for python -m
# ---------------------------------------------------------------------------

def _main() -> None:
    """
    Entry point when the package is executed as a module.

    Delegates to the CLI main function.
    """
    from .cli import main as cli_main
    sys.exit(cli_main())


if __name__ == '__main__':
    _main()
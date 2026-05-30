"""
Python Standalone Installer
============================

Unified high-level API for downloading, installing, managing, and running
portable Python builds from ``indygreg/python-build-standalone``.

This module integrates :class:`DownloadManager`, :class:`ArchiveExtractor`,
:class:`PythonVersionManager`, :class:`PythonRunner`, and
:class:`PlatformDetector` into a single, coherent interface that handles
the complete lifecycle of standalone Python installations.

Security
--------
- All downloads use HTTPS with mandatory SHA256 checksum verification.
- Archives are validated for path traversal, archive bombs, and disk
  space before extraction.
- Extracted files are sanitised (setuid/setgid stripped, permissions
  normalised).
- Python installations are fully isolated; system Python is never
  modified.
- Shell configuration updates are additive and wrapped in marker
  comments for clean removal.
- Process switching (:meth:`set_current`) uses :func:`os.execve` with
  a carefully filtered environment to prevent variable leakage.
- Temporary files are created with ``0o600`` permissions and
  atomically renamed.
- All user-supplied paths and version strings are validated against
  allowlists before use in commands or filesystem operations.

Usage
-----
.. code-block:: python

    from installer import PythonInstaller

    # One-line install
    installer = PythonInstaller()
    python_path = installer.install("3.11.5")

    # Install to custom location
    installer = PythonInstaller(
        install_root="/opt/pythons",
        cache_dir="/var/cache/python",
    )
    installer.install("3.12.0")

    # Install and set as current process
    installer.install("3.11.5")
    installer.set_current("3.11.5")  # replaces running process

    # Install and set as system default
    installer.install("3.11.5")
    installer.set_default("3.11.5")

    # Run code with an installed version
    result = installer.run_code("3.11.5", "print('Hello')")
    print(result.stdout)

    # List installed versions
    for version, path in installer.list_installed().items():
        print(f"Python {version}: {path}")

    # Remove a version
    installer.uninstall("3.9.18")

Warnings
--------
- :meth:`set_current` calls :func:`os.execve` and **does not return**.
  The current process is replaced entirely. Unsaved data is lost.
- :meth:`set_default` modifies shell configuration files (``.bashrc``,
  ``.zshrc``, ``.profile``, ``.bash_profile``). Back up these files
  before first use.
- Installation requires disk space for: downloaded archive (~40 MB),
  temporary extraction (~120 MB), and final installation (~120 MB).
  Ensure at least 300 MB free.
- On macOS, ``platform.machine()`` reports ``x86_64`` under Rosetta 2
  even on Apple Silicon. Set the environment variable
  ``PYTHON_STANDALONE_TARGET=aarch64-apple-darwin`` to override.
- On Windows, ``set_default`` creates batch-file shims instead of
  symlinks due to privilege requirements. These shims only work in
  ``cmd.exe`` and PowerShell, not in WSL or MSYS2.
- Android/Termux detection relies on ``ANDROID_ROOT`` or
  ``TERMUX_VERSION`` environment variables. On custom Android
  environments, detection may fail.
- Rate-limited GitHub API (60 req/h unauthenticated). Set
  ``GITHUB_TOKEN`` environment variable for 5000 req/h.
- The ``install_only`` variant is used by default (no debug symbols,
  no headers). Use ``variant="full"`` for a complete installation
  including development files.

Notes
-----
- All installed Pythons are fully self-contained. Deleting
  *install_root* removes everything managed by this module.
- The state file (``state.json``) is human-readable JSON. Deleting it
  resets the manager's view but does not remove installed Pythons.
- Shell ``PATH`` modifications are wrapped in comments:
  ``# >>> python-standalone-manager >>>`` /
  ``# <<< python-standalone-manager <<<``.
  Removing these blocks reverts the change.
- The download cache is separate from installations. Use
  ``clear_cache()`` to free space without affecting installed versions.
- All network operations honour ``HTTP_PROXY``, ``HTTPS_PROXY``, and
  ``NO_PROXY`` environment variables via the standard library.
- Thread safety: this class is **not thread-safe**. Concurrent
  installations must use separate :class:`PythonInstaller` instances
  or external locking.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Internal imports — all modules built previously
from .downloader import (
    CacheManager,
    ChecksumVerificationError,
    DownloadError,
    DownloadManager,
    GitHubReleaseFetcher,
)
from .extractor import (
    ArchiveBombError,
    ArchiveExtractor,
    DiskSpaceError,
    ExtractionError,
    SecurityError as ExtractionSecurityError,
)
from .platforms import (
    PlatformDetector,
    TargetTriple,
    build_asset_filename,
    detect_target,
    is_supported,
)
from .manager import (
    PythonVersionManager,
    ShellConfigError,
    VersionActiveError,
    VersionAlreadyInstalledError,
    VersionManagerError,
    VersionNotFoundError,
)
from .runner import (
    PipError,
    PythonRunner,
    RunnerError,
    RunnerResult,
    RunnerTimeoutError,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default directory for all Python installations.
_DEFAULT_INSTALL_ROOT: Path = Path.home() / ".python_standalone"

#: Default directory for the download cache.
_DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "python_standalone"

#: Default download mirrors for resilience.
_DEFAULT_MIRRORS: Tuple[str, ...] = (
    "https://github.com/indygreg/python-build-standalone/releases/download",
)

#: Maximum number of concurrent installations (for future use).
_MAX_CONCURRENT_INSTALLS: int = 1

#: Default request timeout in seconds.
_DEFAULT_REQUEST_TIMEOUT: int = 30

#: Default download rate limit (0 = unlimited).
_DEFAULT_RATE_LIMIT_MBPS: float = 0.0

#: Maximum allowed archive compression ratio.
_DEFAULT_MAX_COMPRESSION_RATIO: float = 100.0

#: Default Python variant to download.
_DEFAULT_VARIANT: str = "install_only"

#: Known release dates for Python versions.
#: Maps version prefix to release tag date.
#: Updated as new releases are published.
_KNOWN_RELEASES: Dict[str, str] = {
    "3.8.18": "20230825",
    "3.8.19": "20240320",
    "3.9.18": "20230825",
    "3.9.19": "20240320",
    "3.10.13": "20230825",
    "3.10.14": "20240320",
    "3.11.5": "20231002",
    "3.11.6": "20231002",
    "3.11.7": "20231205",
    "3.11.8": "20240208",
    "3.11.9": "20240402",
    "3.12.0": "20231002",
    "3.12.1": "20231205",
    "3.12.2": "20240208",
    "3.12.3": "20240402",
}


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class InstallerError(Exception):
    """
    Base exception for all installer errors.

    Parameters
    ----------
    message : str
        Human-readable description.
    version : str or None
        Python version involved, if any.
    original_error : Exception or None
        Underlying exception that caused this error.
    """

    def __init__(
        self,
        message: str,
        version: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.version = version
        self.original_error = original_error

    def __str__(self) -> str:
        base = super().__str__()
        if self.version:
            base = f"{base}\n  Version: {self.version}"
        if self.original_error:
            base = f"{base}\n  Caused by: {self.original_error}"
        return base


class InstallError(InstallerError):
    """Raised when installation fails for any reason."""

    pass


class UnsupportedVersionError(InstallerError):
    """
    Raised when a Python version is not available as a standalone build.

    Parameters
    ----------
    version : str
        The unsupported version.
    available : list of str
        Known available versions (may be partial).
    """

    def __init__(
        self,
        version: str,
        available: Optional[List[str]] = None,
    ) -> None:
        self.available = available or []
        msg = f"Python {version} is not available as a standalone build."
        if self.available:
            msg += f" Available (partial list): {', '.join(self.available[:10])}"
        super().__init__(msg, version=version)


class NetworkError(InstallerError):
    """Raised when network operations fail after all retries."""

    pass


class IntegrityError(InstallerError):
    """Raised when downloaded files fail integrity checks."""

    pass


class ConfigurationError(InstallerError):
    """
    Raised when the installer configuration is invalid.

    For example, when *install_root* and *cache_dir* are the same.
    """

    pass


# ---------------------------------------------------------------------------
# Release Database
# ---------------------------------------------------------------------------


class _ReleaseDatabase:
    """
    Maps Python versions to release dates for
    ``python-build-standalone``.

    Parameters
    ----------
    fetcher : GitHubReleaseFetcher or None
        If provided, fetches release metadata from GitHub to discover
        new versions not in the static mapping.

    Notes
    -----
    - The static mapping :data:`_KNOWN_RELEASES` is consulted first.
    - If a version is not found, and *fetcher* is available, GitHub
      is queried.
    - Results are cached in-memory for the lifetime of the instance.
    """

    def __init__(
        self, fetcher: Optional[GitHubReleaseFetcher] = None
    ) -> None:
        self._static = dict(_KNOWN_RELEASES)
        self._fetcher = fetcher
        self._cache: Dict[str, str] = {}

    def get_release_date(self, version: str) -> str:
        """
        Get the release tag date for *version*.

        Parameters
        ----------
        version : str
            Python version, e.g. ``"3.11.5"``.

        Returns
        -------
        str
            Release date tag, e.g. ``"20231002"``.

        Raises
        ------
        UnsupportedVersionError
            If the version cannot be found in static data or via
            GitHub API.
        """
        # Check cache
        if version in self._cache:
            return self._cache[version]

        # Check static map
        if version in self._static:
            self._cache[version] = self._static[version]
            return self._static[version]

        # Try GitHub API
        if self._fetcher:
            date = self._fetch_from_github(version)
            if date:
                self._cache[version] = date
                return date

        raise UnsupportedVersionError(
            version=version,
            available=sorted(self._static.keys()),
        )

    def _fetch_from_github(self, version: str) -> Optional[str]:
        """
        Query GitHub releases for *version*.

        Parameters
        ----------
        version : str
            Python version.

        Returns
        -------
        str or None
            Release date tag, or ``None`` if not found.
        """
        try:
            release = self._fetcher.get_latest_release()
        except DownloadError:
            return None

        tag = release.get("tag_name", "")
        # Tags are date-based: "20231002"
        if not tag or not tag.isdigit() or len(tag) != 8:
            return None

        # Check if the version exists in this release
        for asset in release.get("assets", []):
            name = asset.get("name", "")
            if f"cpython-{version}+" in name:
                return tag

        return None

    def add_known_release(self, version: str, date: str) -> None:
        """
        Register a known version-date mapping.

        Parameters
        ----------
        version : str
            Python version.
        date : str
            Release date tag (8 digits, e.g. ``"20231002"``).

        Raises
        ------
        ValueError
            If *date* is not an 8-digit string.
        """
        if not date.isdigit() or len(date) != 8:
            raise ValueError(
                f"Release date must be 8 digits, got: {date!r}"
            )
        self._static[version] = date
        self._cache[version] = date

    def list_known_versions(self) -> List[str]:
        """
        Return all known Python versions.

        Returns
        -------
        list of str
            Sorted list of version strings.
        """
        return sorted(self._static.keys())


# ---------------------------------------------------------------------------
# Python Installer
# ---------------------------------------------------------------------------


class PythonInstaller:
    """
    Unified installer for portable Python builds.

    Handles the complete lifecycle: discovery, download, extraction,
    installation, execution, and removal.

    Parameters
    ----------
    install_root : Path, optional
        Directory for installed Python versions. Defaults to
        ``~/.python_standalone``.
    cache_dir : Path, optional
        Directory for cached downloads. Defaults to
        ``~/.cache/python_standalone``.
    mirrors : list of str, optional
        Additional download mirror URLs.
    github_token : str, optional
        GitHub personal access token. If ``None``, reads the
        ``GITHUB_TOKEN`` environment variable.
    max_retries : int
        Maximum download retry attempts per URL. Default 3.
    timeout : int
        Network timeout in seconds. Default 30.
    rate_limit_mbps : float
        Maximum download speed in MB/s. ``0`` = unlimited.
    show_progress : bool
        If ``True`` (default), show progress bars during download
        and extraction.
    variant : str
        Python build variant. ``"install_only"`` (default) for
        minimal, ``"full"`` for development files.
    max_compression_ratio : float
        Maximum allowed archive compression ratio (anti-bomb).
        Default 100.

    Raises
    ------
    ConfigurationError
        If *install_root* equals *cache_dir*, or if directories
        cannot be created.

    Examples
    --------
    >>> installer = PythonInstaller()
    >>> python_path = installer.install("3.11.5")
    >>> result = installer.run_code("3.11.5", "print('Hello')")
    >>> print(result.stdout)
    Hello

    With custom paths::

    >>> installer = PythonInstaller(
    ...     install_root=Path("/opt/pythons"),
    ...     cache_dir=Path("/var/cache/py_standalone"),
    ...     github_token="ghp_xxxx",
    ...     max_retries=5,
    ... )
    """

    def __init__(
        self,
        install_root: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        mirrors: Optional[List[str]] = None,
        github_token: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = _DEFAULT_REQUEST_TIMEOUT,
        rate_limit_mbps: float = _DEFAULT_RATE_LIMIT_MBPS,
        show_progress: bool = True,
        variant: str = _DEFAULT_VARIANT,
        max_compression_ratio: float = _DEFAULT_MAX_COMPRESSION_RATIO,
    ) -> None:
        # Resolve paths
        self._install_root = (install_root or _DEFAULT_INSTALL_ROOT).resolve()
        self._cache_dir = (cache_dir or _DEFAULT_CACHE_DIR).resolve()

        # Validate configuration
        if self._install_root == self._cache_dir:
            raise ConfigurationError(
                "install_root and cache_dir must be different.\n"
                f"  install_root: {self._install_root}\n"
                f"  cache_dir:    {self._cache_dir}"
            )

        # Create directories
        try:
            self._install_root.mkdir(parents=True, exist_ok=True)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ConfigurationError(
                f"Cannot create directories: {e}"
            ) from e

        # Resolve GitHub token
        self._github_token = github_token or os.environ.get("GITHUB_TOKEN")

        # Store configuration
        self._variant = variant
        self._show_progress = show_progress

        # Initialise sub-components
        self._platform_detector = PlatformDetector(allow_override=True)
        self._cache_manager = CacheManager(self._cache_dir)
        self._fetcher = GitHubReleaseFetcher(
            token=self._github_token,
        )
        self._downloader = DownloadManager(
            mirrors=mirrors,
            max_retries=max_retries,
            timeout=timeout,
            rate_limit_mbps=rate_limit_mbps,
            show_progress=show_progress,
        )
        self._extractor = ArchiveExtractor(
            max_compression_ratio=max_compression_ratio,
        )
        self._version_manager = PythonVersionManager(
            install_root=self._install_root,
            downloader=self._downloader,
            extractor=self._extractor,
            platform_detector=self._platform_detector,
        )
        self._release_db = _ReleaseDatabase(fetcher=self._fetcher)

        # Runner cache (one per version)
        self._runners: Dict[str, PythonRunner] = {}

    # ------------------------------------------------------------------
    # Public API — Installation
    # ------------------------------------------------------------------

    def install(
        self,
        version: str,
        release_date: Optional[str] = None,
        force: bool = False,
        target_triple: Optional[str] = None,
    ) -> Path:
        """
        Download and install a standalone Python build.

        Parameters
        ----------
        version : str
            Python version, e.g. ``"3.11.5"``.
        release_date : str, optional
            Release tag date (8 digits, e.g. ``"20231002"``).
            If ``None``, looked up from the release database.
        force : bool
            If ``True``, reinstall even if already present.
        target_triple : str, optional
            Platform target triple. If ``None``, auto-detected.

        Returns
        -------
        Path
            Absolute path to the installed Python executable.

        Raises
        ------
        InstallError
            If any step fails (download, extraction, verification).
        UnsupportedVersionError
            If the version is not available and *release_date* is
            not provided.
        VersionAlreadyInstalledError
            If the version is already installed and *force* is
            ``False``.

        Notes
        -----
        - The installation is fully self-contained in a subdirectory
          of *install_root*.
        - Downloaded archives are cached in *cache_dir* for future
          use.
        - Progress bars are shown during download if *show_progress*
          was ``True`` at construction.
        """
        # Resolve release date
        if release_date is None:
            release_date = self._release_db.get_release_date(version)

        # Determine target triple
        if target_triple is None:
            target = self._platform_detector.detect()
            target_triple = target.raw
        else:
            # Validate
            target = TargetTriple.from_string(target_triple)
            if not is_supported(target):
                raise InstallError(
                    f"Unsupported target triple: {target_triple}",
                    version=version,
                )

        # Build asset filename
        asset_filename = build_asset_filename(
            python_version=version,
            release_date=release_date,
            target=target,
            variant=self._variant,
        )

        try:
            # Download
            archive_path = self._downloader.download_release_asset(
                release_tag=release_date,
                asset_filename=asset_filename,
                fetcher=self._fetcher,
                cache=self._cache_manager,
            )

            # Install via version manager
            python_bin = self._version_manager.install(
                version=version,
                release_date=release_date,
                target_triple=target_triple,
                archive_path=archive_path,
                force=force,
            )

            # Cache the runner
            if version not in self._runners:
                self._runners[version] = PythonRunner(
                    python_bin=python_bin,
                )

            return python_bin

        except DownloadError as e:
            raise NetworkError(
                f"Failed to download Python {version}: {e}",
                version=version,
                original_error=e,
            ) from e
        except ChecksumVerificationError as e:
            raise IntegrityError(
                f"Checksum verification failed for Python {version}",
                version=version,
                original_error=e,
            ) from e
        except (ExtractionError, ExtractionSecurityError) as e:
            raise InstallError(
                f"Extraction failed for Python {version}: {e}",
                version=version,
                original_error=e,
            ) from e
        except VersionManagerError as e:
            raise InstallError(
                f"Installation failed for Python {version}: {e}",
                version=version,
                original_error=e,
            ) from e

    def install_latest(self, prefix: str = "3.11") -> Path:
        """
        Install the latest available Python version matching *prefix*.

        Parameters
        ----------
        prefix : str
            Version prefix, e.g. ``"3.11"`` or ``"3"``.

        Returns
        -------
        Path
            Path to the installed Python executable.

        Raises
        ------
        InstallError
            If no matching version is found.

        Notes
        -----
        - Only consults the static release database. Use
          :meth:`fetch_available_versions` to query GitHub first.
        """
        versions = self._release_db.list_known_versions()
        matching = [v for v in versions if v.startswith(prefix)]
        if not matching:
            raise InstallError(
                f"No known version matches prefix {prefix!r}."
            )

        latest = matching[-1]  # Sorted, last is newest
        return self.install(latest)

    # ------------------------------------------------------------------
    # Public API — Execution
    # ------------------------------------------------------------------

    def run_script(
        self,
        version: str,
        script: Path,
        args: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        cwd: Optional[Path] = None,
        extra_env: Optional[Dict[str, str]] = None,
        check: bool = True,
    ) -> RunnerResult:
        """
        Execute a Python script with the specified version.

        Parameters
        ----------
        version : str
            Python version to use.
        script : Path
            Path to the ``.py`` script.
        args : list of str, optional
            Command-line arguments.
        timeout : float, optional
            Subprocess timeout in seconds.
        cwd : Path, optional
            Working directory.
        extra_env : dict[str, str], optional
            Additional environment variables.
        check : bool
            If ``True`` (default), raise on non-zero exit.

        Returns
        -------
        RunnerResult

        Raises
        ------
        VersionNotFoundError
            If *version* is not installed.
        RunnerError
            If execution fails and *check* is ``True``.
        """
        runner = self._get_runner(version)
        return runner.run_script(
            script=script,
            args=args,
            timeout=timeout,
            cwd=cwd,
            extra_env=extra_env,
            check=check,
        )

    def run_code(
        self,
        version: str,
        code: str,
        timeout: Optional[float] = None,
        cwd: Optional[Path] = None,
        extra_env: Optional[Dict[str, str]] = None,
        check: bool = True,
    ) -> RunnerResult:
        """
        Execute Python code with the specified version.

        Parameters
        ----------
        version : str
            Python version.
        code : str
            Python source code.
        timeout : float, optional
            Timeout in seconds.
        cwd : Path, optional
            Working directory.
        extra_env : dict[str, str], optional
            Additional environment variables.
        check : bool
            If ``True`` (default), raise on non-zero exit.

        Returns
        -------
        RunnerResult
        """
        runner = self._get_runner(version)
        return runner.run_code(
            code=code,
            timeout=timeout,
            cwd=cwd,
            extra_env=extra_env,
            check=check,
        )

    def run_module(
        self,
        version: str,
        module: str,
        args: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        cwd: Optional[Path] = None,
        extra_env: Optional[Dict[str, str]] = None,
        check: bool = True,
    ) -> RunnerResult:
        """
        Execute a Python module with the specified version.

        Parameters
        ----------
        version : str
            Python version.
        module : str
            Module name (e.g., ``"pip"``).
        args : list of str, optional
            Module arguments.
        timeout : float, optional
            Timeout in seconds.
        cwd : Path, optional
            Working directory.
        extra_env : dict[str, str], optional
            Additional environment variables.
        check : bool
            If ``True`` (default), raise on non-zero exit.

        Returns
        -------
        RunnerResult
        """
        runner = self._get_runner(version)
        return runner.run_module(
            module=module,
            args=args,
            timeout=timeout,
            cwd=cwd,
            extra_env=extra_env,
            check=check,
        )

    def pip_install(
        self,
        version: str,
        packages: List[str],
        upgrade: bool = False,
        timeout: Optional[float] = None,
        check: bool = True,
    ) -> RunnerResult:
        """
        Install pip packages for the specified Python version.

        Parameters
        ----------
        version : str
            Python version.
        packages : list of str
            Package specifications.
        upgrade : bool
            If ``True``, upgrade existing packages.
        timeout : float, optional
            Timeout in seconds.
        check : bool
            If ``True`` (default), raise on failure.

        Returns
        -------
        RunnerResult
        """
        runner = self._get_runner(version)
        return runner.pip_install(
            packages=packages,
            upgrade=upgrade,
            timeout=timeout,
            check=check,
        )

    # ------------------------------------------------------------------
    # Public API — Version Management
    # ------------------------------------------------------------------

    def set_current(
        self,
        version: str,
        args: Optional[List[str]] = None,
    ) -> None:
        """
        Replace the running process with the specified Python version.

        **This method does not return.** It uses :func:`os.execve` to
        replace the current process.

        Parameters
        ----------
        version : str
            Python version to switch to.
        args : list of str, optional
            Arguments for the new process. Defaults to ``sys.argv``.

        Raises
        ------
        VersionNotFoundError
            If *version* is not installed.

        Warnings
        --------
        - The current process is **replaced**. All unsaved state,
          open file handles not marked ``FD_CLOEXEC``, and pending
          I/O are lost.
        - Child processes are inherited by the new interpreter.
        - Environment variables are filtered; Python-specific
          variables (``PYTHONHOME``, ``PYTHONPATH``, etc.) are
          removed.
        """
        self._version_manager.set_current(version, args=args)

    def set_default(self, version: str, fallback: bool = False) -> Path:
        """
        Set a Python version as the system default.

        Creates symlinks (or shims on Windows) and updates shell
        configuration files.

        Parameters
        ----------
        version : str
            Python version to set as default.

        Returns
        -------
        Path
            Path to the default Python executable.

        Raises
        ------
        VersionNotFoundError
            If *version* is not installed.
        ShellConfigError
            If shell configuration files cannot be updated.

        Notes
        -----
        - Modifies ``~/.bashrc``, ``~/.zshrc``, ``~/.profile``, and
          ``~/.bash_profile`` if they exist.
        - Changes are wrapped in marker comments for clean removal.
        - Run ``source ~/.bashrc`` (or equivalent) after calling this
          method to apply changes to the current shell.
        """
        bashrc = Path.home() / ".bashrc"
        if not bashrc.exists() and fallback:
            bashrc.touch()

        return self._version_manager.set_default(version)

    def uninstall(self, version: str, force: bool = False) -> None:
        """
        Remove an installed Python version.

        Parameters
        ----------
        version : str
            Python version to remove.
        force : bool
            If ``True``, remove even if this version is the default.

        Raises
        ------
        VersionNotFoundError
            If *version* is not installed.
        VersionActiveError
            If *version* is the default and *force* is ``False``.
        """
        self._version_manager.uninstall(version, force=force)
        self._runners.pop(version, None)

    def switch(self, version: str) -> None:
        """
        Alias for :meth:`set_current`.

        Parameters
        ----------
        version : str
            Python version to switch to.
        """
        self.set_current(version)

    # ------------------------------------------------------------------
    # Public API — Query
    # ------------------------------------------------------------------

    def list_installed(self) -> Dict[str, Path]:
        """
        List all installed Python versions.

        Returns
        -------
        dict[str, Path]
            Mapping of version string to Python executable path.
        """
        return self._version_manager.list_installed()

    def list_available(self) -> List[str]:
        """
        List all known Python versions (from static database).

        Returns
        -------
        list of str
            Sorted version strings.

        Notes
        -----
        - This is the static database only. Use
          :meth:`fetch_available_versions` to query GitHub for the
          latest releases.
        """
        return self._release_db.list_known_versions()

    def fetch_available_versions(self) -> List[str]:
        """
        Query GitHub for the latest available Python versions.

        Returns
        -------
        list of str
            Versions found in the latest GitHub release.

        Raises
        ------
        NetworkError
            If the GitHub API is unreachable.

        Notes
        -----
        - This updates the internal release database.
        - Requires network access.
        - Results are cached for the lifetime of the instance.
        """
        try:
            release = self._fetcher.get_latest_release()
        except DownloadError as e:
            raise NetworkError(
                "Failed to fetch available versions from GitHub.",
                original_error=e,
            ) from e

        tag = release.get("tag_name", "")
        versions: List[str] = []
        seen: set = set()

        for asset in release.get("assets", []):
            name = asset.get("name", "")
            # Extract version from "cpython-X.Y.Z+..."
            if name.startswith("cpython-") and "+" in name:
                ver = name.split("+")[0].replace("cpython-", "")
                if ver not in seen:
                    seen.add(ver)
                    versions.append(ver)
                    # Register in database
                    self._release_db.add_known_release(ver, tag)

        return sorted(versions)

    def get_default(self) -> Optional[str]:
        """
        Return the currently-set default Python version.

        Returns
        -------
        str or None
        """
        return self._version_manager.get_default()

    def is_installed(self, version: str) -> bool:
        """
        Check if a Python version is installed.

        Parameters
        ----------
        version : str
            Python version.

        Returns
        -------
        bool
        """
        return self._version_manager.is_installed(version)

    def get_python_path(self, version: str) -> Path:
        """
        Get the path to a Python executable.

        Parameters
        ----------
        version : str
            Python version.

        Returns
        -------
        Path

        Raises
        ------
        VersionNotFoundError
            If *version* is not installed.
        """
        return self._version_manager.get_python_path(version)

    def get_version_info(self, version: str) -> Dict[str, str]:
        """
        Get detailed information about an installed Python version.

        Parameters
        ----------
        version : str
            Python version.

        Returns
        -------
        dict[str, str]
            Keys: ``version``, ``build``, ``compiler``, ``platform``,
            ``implementation``, ``architecture``, ``executable``.
        """
        runner = self._get_runner(version)
        return runner.get_version_info()

    # ------------------------------------------------------------------
    # Public API — Maintenance
    # ------------------------------------------------------------------

    def clear_cache(self) -> int:
        """
        Clear the download cache.

        Returns
        -------
        int
            Number of bytes freed.

        Notes
        -----
        - Only removes cached downloads, not installed Pythons.
        - Subsequent installations will re-download archives.
        """
        total_size = 0
        if self._cache_dir.exists():
            for item in self._cache_dir.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
            shutil.rmtree(self._cache_dir, ignore_errors=True)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            # Re-initialise cache manager
            self._cache_manager = CacheManager(self._cache_dir)
        return total_size

    def get_cache_size(self) -> int:
        """
        Get the total size of the download cache.

        Returns
        -------
        int
            Bytes used by cached files.
        """
        total = 0
        if self._cache_dir.exists():
            for item in self._cache_dir.rglob("*"):
                if item.is_file():
                    total += item.stat().st_size
        return total

    def get_install_size(self, version: Optional[str] = None) -> int:
        """
        Get the disk usage of installed Python versions.

        Parameters
        ----------
        version : str, optional
            If provided, only measure this version. Otherwise, all.

        Returns
        -------
        int
            Bytes used.
        """
        total = 0
        if version:
            version_dir = self._install_root / f"python-{version}"
            if version_dir.exists():
                for item in version_dir.rglob("*"):
                    if item.is_file():
                        total += item.stat().st_size
        else:
            if self._install_root.exists():
                for item in self._install_root.rglob("*"):
                    if item.is_file():
                        total += item.stat().st_size
        return total

    def repair(self, version: str) -> Path:
        """
        Reinstall a Python version if it is damaged.

        Parameters
        ----------
        version : str
            Python version to repair.

        Returns
        -------
        Path
            Path to the repaired Python executable.

        Notes
        -----
        - If the version directory is missing, a full reinstall is
          performed.
        - If the Python executable is missing but the directory
          exists, re-extraction is attempted from the cached archive.
        """
        return self.install(version, force=True)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_runner(self, version: str) -> PythonRunner:
        """
        Get or create a :class:`PythonRunner` for *version*.

        Parameters
        ----------
        version : str
            Python version.

        Returns
        -------
        PythonRunner

        Raises
        ------
        VersionNotFoundError
            If *version* is not installed.
        """
        if version not in self._runners:
            python_bin = self._version_manager.get_python_path(version)
            self._runners[version] = PythonRunner(python_bin=python_bin)
        return self._runners[version]

    def __repr__(self) -> str:
        installed = len(self._version_manager.list_installed())
        default = self._version_manager.get_default() or "none"
        return (
            f"PythonInstaller("
            f"install_root={self._install_root}, "
            f"installed={installed} version(s), "
            f"default={default})"
        )
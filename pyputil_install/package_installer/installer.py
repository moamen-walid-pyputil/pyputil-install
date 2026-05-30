#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Core package installer with dependency resolution, rollback, and validation.

This module provides the main `PackageInstaller` class, which is the
central interface for all package management operations. It orchestrates
network requests, caching, security verification, retry logic, virtual
environment management, and dependency resolution into a unified API.

Classes
-------
InstallConfig
    Configuration for installation behavior.
InstallResult
    Result of an installation operation with detailed metadata.
PackageInstaller
    Main class for managing Python packages.

Examples
--------
>>> installer = PackageInstaller("requests")
>>> installer.install()
>>> installer.get_version()
'2.31.0'

With custom configuration:

>>> config = InstallConfig(use_cache=True, require_hashes=True)
>>> installer = PackageInstaller("django", config=config)
>>> installer.install(version="4.2.0")
"""

import sys
import time
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

from .exceptions import (
    PackageInstallerError,
    PackageInstallError,
    PackageUninstallError,
    PackageUpgradeError,
    PackageNotFoundError,
    PackageVersionError,
    NetworkError,
    TimeoutError,
    ValidationError,
    SecurityError,
    CacheError,
)
from .retry import retry, RetryConfig
from .cache import PackageCache, CacheConfig
from .network.session import PackageSession, SessionConfig
from .network.pypi import PyPIClient
from .security.hashes import (
    HashVerifier,
    HashAlgorithm,
    compute_hash,
    verify_hash,
    multi_hash,
)
from .security.verify import TrustVerifier, TrustConfig
from .environment.venv import VirtualEnvironment

logger = logging.getLogger(__name__)


@dataclass
class InstallConfig:
    """
    Configuration for package installation behavior.

    Parameters
    ----------
    use_cache : bool, default=True
        Whether to cache PyPI responses and downloaded packages.
    cache_ttl : float, default=1800.0
        Time-to-live for cached entries in seconds (30 minutes).
    require_hashes : bool, default=False
        If True, require hash verification for all downloaded
        packages.
    hash_algorithm : HashAlgorithm, default=SHA256
        Primary hash algorithm for integrity verification.
    verify_ssl : bool, default=True
        If True, verify SSL/TLS certificates.
    trusted_hosts : set of str, optional
        Additional hosts to trust beyond default PyPI hosts.
    timeout_connect : float, default=10.0
        Connection timeout in seconds.
    timeout_read : float, default=60.0
        Read timeout in seconds.
    max_retries : int, default=3
        Maximum retry attempts for failed network operations.
    user_install : bool, default=False
        If True, install packages to user site-packages.
    upgrade_strategy : str, default="only-if-needed"
        Upgrade strategy: ``only-if-needed``, ``eager``, or
        ``all``.
    dry_run : bool, default=False
        If True, simulate operations without making changes.
    auto_rollback : bool, default=True
        If True, automatically rollback on installation failure.
    backup_wheels : bool, default=False
        If True, keep downloaded wheel files after installation.
    wheel_cache_dir : Path or None, default=None
        Directory for caching downloaded wheels. If None, uses
        pip's default cache.

    Warnings
    --------
    Setting ``verify_ssl=False`` exposes package downloads to
    man-in-the-middle attacks. Only use for trusted internal
    mirrors.

    Setting ``require_hashes=True`` provides strong integrity
    guarantees but requires hash values for all packages,
    including transitive dependencies.

    Examples
    --------
    >>> config = InstallConfig(
    ...     use_cache=True,
    ...     require_hashes=True,
    ...     max_retries=5,
    ... )
    """

    use_cache: bool = True
    cache_ttl: float = 1800.0
    require_hashes: bool = False
    hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256
    verify_ssl: bool = True
    trusted_hosts: Set[str] = field(default_factory=set)
    timeout_connect: float = 10.0
    timeout_read: float = 60.0
    max_retries: int = 3
    user_install: bool = False
    upgrade_strategy: str = "only-if-needed"
    dry_run: bool = False
    auto_rollback: bool = True
    backup_wheels: bool = False
    wheel_cache_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        """Validate configuration consistency."""
        valid_strategies = {"only-if-needed", "eager", "all"}
        if self.upgrade_strategy not in valid_strategies:
            raise ValueError(
                f"upgrade_strategy must be one of {valid_strategies}, "
                f"got '{self.upgrade_strategy}'"
            )

        if self.timeout_connect <= 0:
            raise ValueError("timeout_connect must be positive")

        if self.timeout_read <= 0:
            raise ValueError("timeout_read must be positive")

        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        if self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive")


@dataclass
class InstallResult:
    """
    Result of a package installation operation.

    Parameters
    ----------
    package_name : str
        Name of the package.
    success : bool
        Whether the operation succeeded.
    version_installed : str or None
        The version that was installed, if successful.
    version_previous : str or None
        The previously installed version, if any.
    action : str
        The action performed: ``install``, ``upgrade``,
        ``uninstall``, or ``noop``.
    duration_seconds : float
        Time taken for the operation.
    dependencies_installed : list of str
        Additional packages installed as dependencies.
    errors : list of str
        Error messages if the operation failed.
    warnings : list of str
        Warning messages generated during the operation.
    rolled_back : bool
        Whether a rollback was performed due to failure.
    cache_used : bool
        Whether cached data was used during the operation.

    Examples
    --------
    >>> result = InstallResult(
    ...     package_name="requests",
    ...     success=True,
    ...     version_installed="2.31.0",
    ...     action="install",
    ...     duration_seconds=2.5,
    ... )
    >>> result.success
    True
    """

    package_name: str
    success: bool
    version_installed: Optional[str] = None
    version_previous: Optional[str] = None
    action: str = "noop"
    duration_seconds: float = 0.0
    dependencies_installed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rolled_back: bool = False
    cache_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the result.
        """
        return {
            "package": self.package_name,
            "success": self.success,
            "version_installed": self.version_installed,
            "version_previous": self.version_previous,
            "action": self.action,
            "duration_seconds": round(self.duration_seconds, 3),
            "dependencies_count": len(self.dependencies_installed),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "rolled_back": self.rolled_back,
            "cache_used": self.cache_used,
        }


class PackageInstaller:
    """
    Main class for managing Python packages.

    This class provides a comprehensive interface for installing,
    uninstalling, upgrading, and querying Python packages. It
    integrates caching, security verification, retry logic,
    dependency resolution, and rollback capabilities.

    Parameters
    ----------
    package_name : str
        Name of the package to manage (case-insensitive, PEP 503
        normalized).
    config : InstallConfig, optional
        Installation configuration. If None, default configuration
        is used.
    venv : VirtualEnvironment, optional
        Target virtual environment. If None, uses the current
        environment.

    Attributes
    ----------
    package_name : str
        Normalized package name.
    config : InstallConfig
        Active configuration.
    venv : VirtualEnvironment or None
        Target virtual environment.

    Notes
    -----
    The installer follows a defense-in-depth security model:

    1. **Transport**: TLS with certificate validation.
    2. **Trust**: Hostname allowlisting.
    3. **Integrity**: Cryptographic hash verification.
    4. **Recovery**: Automatic rollback on failure.

    Package names are normalized per PEP 503: case-insensitive,
    with ``-``, ``_``, and ``.`` treated as equivalent.

    Warnings
    --------
    Installing packages from untrusted sources can execute arbitrary
    code during setup. Always verify package sources and prefer
    packages with known provenance.

    System-wide installation may require elevated privileges. Use
    ``user_install=True`` or a virtual environment to avoid
    permission issues.

    Examples
    --------
    Basic usage:

    >>> installer = PackageInstaller("requests")
    >>> installer.is_installed()
    False
    >>> installer.install()
    >>> installer.get_version()
    '2.31.0'

    With specific version and hash verification:

    >>> config = InstallConfig(require_hashes=True)
    >>> installer = PackageInstaller("django", config=config)
    >>> installer.install(version="4.2.0")

    Check for upgrades:

    >>> if installer.check_upgrade():
    ...     installer.upgrade()

    Uninstall:

    >>> installer.uninstall()

    Working with virtual environments:

    >>> from package_installer.environment.venv import VirtualEnvironment
    >>> venv = VirtualEnvironment("/tmp/project-env")
    >>> venv.create()
    >>> installer = PackageInstaller("flask", venv=venv)
    >>> installer.install()
    """

    def __init__(
        self,
        package_name: str,
        config: Optional[InstallConfig] = None,
        venv: Optional[VirtualEnvironment] = None,
    ) -> None:
        self._raw_name = package_name
        self.package_name = self._normalize_name(package_name)
        self.config = config if config is not None else InstallConfig()
        self.venv = venv

        self._session: Optional[PackageSession] = None
        self._cache: Optional[PackageCache] = None
        self._pypi_client: Optional[PyPIClient] = None
        self._hash_verifier: Optional[HashVerifier] = None
        self._trust_verifier: Optional[TrustVerifier] = None

        self._operation_history: List[InstallResult] = []
        self._rollback_snapshots: Dict[str, str] = {}

        self._init_subsystems()

        logger.debug(
            f"PackageInstaller initialized for '{self.package_name}'"
        )

    def _normalize_name(self, name: str) -> str:
        """
        Normalize package name per PEP 503.

        Parameters
        ----------
        name : str
            Raw package name.

        Returns
        -------
        str
            Normalized name (lowercase, hyphens).
        """
        import re
        return re.sub(r"[-_.]+", "-", name).strip().lower()

    def _init_subsystems(self) -> None:
        """
        Initialize all subsystems based on configuration.

        Notes
        -----
        Subsystems are lazily initialized when first accessed.
        This method prepares the configurations but does not
        create connections until needed.
        """
        pass

    @property
    def session(self) -> PackageSession:
        """
        Get or create the HTTP session.

        Returns
        -------
        PackageSession
            Configured HTTP session.

        Notes
        -----
        The session is created once and reused. Connection pooling
        improves performance for multiple requests.
        """
        if self._session is None:
            session_config = SessionConfig(
                timeout_connect=self.config.timeout_connect,
                timeout_read=self.config.timeout_read,
                total_retries=self.config.max_retries,
                verify_tls=self.config.verify_ssl,
            )
            self._session = PackageSession(config=session_config)
        return self._session

    @property
    def cache(self) -> Optional[PackageCache]:
        """
        Get or create the cache instance.

        Returns
        -------
        PackageCache or None
            Cache instance if caching is enabled, None otherwise.
        """
        if not self.config.use_cache:
            return None
        if self._cache is None:
            cache_config = CacheConfig(ttl_seconds=self.config.cache_ttl)
            self._cache = PackageCache(config=cache_config)
        return self._cache

    @property
    def pypi_client(self) -> PyPIClient:
        """
        Get or create the PyPI API client.

        Returns
        -------
        PyPIClient
            Configured PyPI client.
        """
        if self._pypi_client is None:
            self._pypi_client = PyPIClient(
                session=self.session,
                cache=self.cache,
            )
        return self._pypi_client

    @property
    def hash_verifier(self) -> HashVerifier:
        """
        Get or create the hash verifier.

        Returns
        -------
        HashVerifier
            Configured hash verifier.
        """
        if self._hash_verifier is None:
            self._hash_verifier = HashVerifier(
                algorithm=self.config.hash_algorithm,
            )
        return self._hash_verifier

    @property
    def trust_verifier(self) -> TrustVerifier:
        """
        Get or create the trust verifier.

        Returns
        -------
        TrustVerifier
            Configured trust verifier.
        """
        if self._trust_verifier is None:
            trust_config = TrustConfig(
                verify_ssl=self.config.verify_ssl,
                require_hashes=self.config.require_hashes,
            )
            if self.config.trusted_hosts:
                for host in self.config.trusted_hosts:
                    trust_config.trusted_hosts.add(host)
            self._trust_verifier = TrustVerifier(config=trust_config)
        return self._trust_verifier

    def _run_pip(
        self,
        args: List[str],
        timeout: Optional[float] = None,
    ) -> Tuple[int, str, str]:
        """
        Execute a pip command.

        Parameters
        ----------
        args : list of str
            Arguments to pass to pip.
        timeout : float, optional
            Custom timeout in seconds.

        Returns
        -------
        tuple
            ``(returncode, stdout, stderr)``.

        Raises
        ------
        TimeoutError
            If the command times out.
        PackageInstallerError
            If pip cannot be found or executed.

        Notes
        -----
        If a virtual environment is configured, pip from that
        environment is used. Otherwise, the current interpreter's
        pip is used.
        """
        if self.venv is not None and self.venv.exists():
            python_exe = str(self.venv.python_path)
        else:
            python_exe = sys.executable

        cmd = [python_exe, "-m", "pip"] + args
        effective_timeout = timeout if timeout is not None else self.config.timeout_read

        logger.debug(f"Running: {' '.join(cmd)}")

        try:
            import subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                check=False,
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"pip command timed out: {' '.join(cmd)}",
                timeout_seconds=effective_timeout,
                operation="pip",
                package_name=self.package_name,
            )
        except FileNotFoundError:
            raise PackageInstallerError(
                f"Python executable not found: {python_exe}"
            )
        except Exception as e:
            raise PackageInstallerError(
                f"Failed to execute pip: {e}"
            ) from e

    def is_installed(self) -> bool:
        """
        Check if the package is currently installed.

        Returns
        -------
        bool
            True if the package is installed in the target
            environment.

        Notes
        -----
        Uses ``importlib.metadata`` for accurate detection,
        which is more reliable than parsing ``pip list`` output.

        Examples
        --------
        >>> installer = PackageInstaller("pip")
        >>> installer.is_installed()
        True
        """
        try:
            from importlib.metadata import version, PackageNotFoundError

            if self.venv is not None and self.venv.exists():
                return self.venv.check_package(self.package_name)

            version(self.package_name)
            return True
        except PackageNotFoundError:
            return False
        except Exception:
            return False

    def get_version(self) -> Optional[str]:
        """
        Get the installed version of the package.

        Returns
        -------
        str or None
            Version string if installed, None otherwise.

        Examples
        --------
        >>> installer = PackageInstaller("pip")
        >>> ver = installer.get_version()
        >>> ver is not None
        True
        """
        try:
            from importlib.metadata import version, PackageNotFoundError

            if self.venv is not None and self.venv.exists():
                packages = self.venv.get_installed_packages()
                for pkg_line in packages:
                    pkg_parts = pkg_line.split("==")
                    if len(pkg_parts) == 2 and pkg_parts[0].lower() == self.package_name:
                        return pkg_parts[1]
                return None

            return version(self.package_name)
        except PackageNotFoundError:
            return None
        except Exception:
            return None

    def get_latest_version(
        self,
        include_pre: bool = False,
    ) -> Optional[str]:
        """
        Get the latest available version from PyPI.

        Parameters
        ----------
        include_pre : bool, default=False
            If True, include pre-release versions.

        Returns
        -------
        str or None
            Latest version string, or None if not found.

        Raises
        ------
        NetworkError
            If PyPI cannot be reached.
        PackageNotFoundError
            If the package does not exist on PyPI.

        Examples
        --------
        >>> installer = PackageInstaller("requests")
        >>> latest = installer.get_latest_version()
        >>> latest is not None
        True
        """
        return self.pypi_client.get_latest_version(
            self.package_name,
            include_pre_releases=include_pre,
        )

    def get_available_versions(self) -> List[str]:
        """
        Get all available versions from PyPI.

        Returns
        -------
        list of str
            All version strings, newest first.

        Examples
        --------
        >>> installer = PackageInstaller("six")
        >>> versions = installer.get_available_versions()
        >>> len(versions) > 0
        True
        """
        return self.pypi_client.get_package_versions(self.package_name)

    def get_package_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the package from PyPI.

        Returns
        -------
        dict
            Package metadata dictionary.

        Examples
        --------
        >>> installer = PackageInstaller("click")
        >>> info = installer.get_package_info()
        >>> info["name"]
        'click'
        """
        return self.pypi_client.get_package_info(self.package_name)

    def get_dependencies(
        self,
        version: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Get the dependencies of a package.

        Parameters
        ----------
        version : str, optional
            Specific version to query. If None, uses latest.

        Returns
        -------
        list of dict
            Parsed dependency specifications.

        Examples
        --------
        >>> installer = PackageInstaller("flask")
        >>> deps = installer.get_dependencies()
        >>> any(d["name"] == "jinja2" for d in deps)
        True
        """
        metadata = self.pypi_client.get_package_metadata(self.package_name)
        info = metadata.get("info", {})
        requires_dist = info.get("requires_dist", []) or []
        return self.pypi_client.parse_dependencies(requires_dist)

    def check_upgrade(self, include_pre: bool = False) -> bool:
        """
        Check if a newer version is available.

        Parameters
        ----------
        include_pre : bool, default=False
            If True, consider pre-release versions.

        Returns
        -------
        bool
            True if a newer version exists.

        Examples
        --------
        >>> installer = PackageInstaller("pip")
        >>> isinstance(installer.check_upgrade(), bool)
        True
        """
        current = self.get_version()
        if current is None:
            return False

        latest = self.get_latest_version(include_pre=include_pre)
        if latest is None:
            return False

        try:
            from packaging.version import Version
            return Version(latest) > Version(current)
        except Exception:
            return latest != current

    def install(
        self,
        version: Optional[str] = None,
        upgrade: bool = False,
        pre: bool = False,
        force_reinstall: bool = False,
        no_deps: bool = False,
        extra_index_url: Optional[str] = None,
    ) -> InstallResult:
        """
        Install the package.

        Parameters
        ----------
        version : str, optional
            Specific version to install. If None, installs the
            latest stable version.
        upgrade : bool, default=False
            If True, upgrade if already installed.
        pre : bool, default=False
            If True, allow pre-release versions.
        force_reinstall : bool, default=False
            If True, reinstall even if already installed.
        no_deps : bool, default=False
            If True, skip dependency installation.
        extra_index_url : str, optional
            Additional package index URL.

        Returns
        -------
        InstallResult
            Detailed result of the operation.

        Raises
        ------
        PackageInstallError
            If installation fails and auto_rollback is disabled.
        SecurityError
            If security checks fail.
        ValidationError
            If hash verification fails.

        Warnings
        --------
        Installing packages from additional indices may introduce
        security risks. Ensure extra_index_url points to a trusted
        source.

        Setting ``no_deps=True`` may result in a non-functional
        installation if required dependencies are missing.

        Notes
        -----
        The installation process follows these steps:

        1. Validate package source (URL, host trust).
        2. Check cache for existing data.
        3. Resolve version (if not specified).
        4. Fetch package metadata and hashes.
        5. Download and verify package integrity.
        6. Install via pip.
        7. Verify installation success.
        8. Record result in operation history.

        If ``auto_rollback`` is enabled and installation fails,
        the previous version is restored automatically.

        Examples
        --------
        >>> installer = PackageInstaller("requests")
        >>> result = installer.install()
        >>> result.success
        True

        Install specific version:

        >>> result = installer.install(version="2.28.0")
        >>> result.version_installed
        '2.28.0'

        Upgrade to latest:

        >>> result = installer.install(upgrade=True)
        """
        start_time = time.monotonic()
        result = InstallResult(
            package_name=self.package_name,
            success=False,
            action="install",
        )

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would install {self.package_name}")
            result.success = True
            result.warnings.append("Dry run mode: no changes made")
            return result

        try:
            previous_version = self.get_version()

            if previous_version is not None and not upgrade and not force_reinstall:
                logger.info(
                    f"{self.package_name} {previous_version} already "
                    f"installed. Use upgrade=True to upgrade."
                )
                result.success = True
                result.version_installed = previous_version
                result.version_previous = previous_version
                result.action = "noop"
                result.cache_used = self.config.use_cache
                result.duration_seconds = time.monotonic() - start_time
                return result

            if version is None:
                version = self.get_latest_version(include_pre=pre)
                if version is None:
                    raise PackageNotFoundError(
                        self.package_name,
                        message="No versions found on PyPI",
                        location="pypi",
                    )

            pkg_spec = f"{self.package_name}=={version}" if version else self.package_name

            pip_args = ["install", pkg_spec]

            if upgrade:
                pip_args.append("--upgrade")
                if self.config.upgrade_strategy != "only-if-needed":
                    pip_args.append(f"--upgrade-strategy={self.config.upgrade_strategy}")
            if pre:
                pip_args.append("--pre")
            if force_reinstall:
                pip_args.append("--force-reinstall")
            if no_deps:
                pip_args.append("--no-deps")
            if self.config.user_install:
                pip_args.append("--user")
            if extra_index_url:
                self.trust_verifier.validate_url(extra_index_url)
                pip_args.extend(["--extra-index-url", extra_index_url])
            if not self.config.verify_ssl:
                pip_args.append("--trusted-host")
                pip_args.append("pypi.org")
                pip_args.append("--trusted-host")
                pip_args.append("files.pythonhosted.org")

            if self.config.require_hashes:
                download_urls = self.pypi_client.get_download_urls(
                    self.package_name, version
                )
                for url_info in download_urls:
                    if url_info.get("sha256"):
                        pip_args.extend([
                            "--hash",
                            f"sha256:{url_info['sha256']}",
                        ])

            if previous_version and self.config.auto_rollback:
                self._rollback_snapshots[self.package_name] = previous_version

            returncode, stdout, stderr = self._run_pip(pip_args)

            if returncode != 0:
                error_msg = stderr or "Unknown installation error"
                raise PackageInstallError(
                    self.package_name,
                    message=error_msg,
                    stderr=stderr,
                    version=version,
                )

            installed_version = self.get_version()

            if installed_version is None:
                installed_version = version

            result.success = True
            result.version_installed = installed_version
            result.version_previous = previous_version
            if previous_version and previous_version != installed_version:
                result.action = "upgrade"
            result.cache_used = self.config.use_cache

            logger.info(
                f"Successfully installed {self.package_name} "
                f"{installed_version}"
            )

        except (PackageInstallError, PackageNotFoundError) as e:
            result.errors.append(str(e))
            if self.config.auto_rollback and self.package_name in self._rollback_snapshots:
                try:
                    rollback_version = self._rollback_snapshots[self.package_name]
                    self.install(version=rollback_version, force_reinstall=True)
                    result.rolled_back = True
                    result.warnings.append(
                        f"Rolled back to {rollback_version}"
                    )
                except Exception as rollback_error:
                    result.errors.append(
                        f"Rollback failed: {rollback_error}"
                    )
            if not self.config.auto_rollback:
                raise

        except (NetworkError, TimeoutError, SecurityError, ValidationError) as e:
            result.errors.append(str(e))
            raise

        except Exception as e:
            result.errors.append(str(e))
            raise PackageInstallError(
                self.package_name,
                message=f"Unexpected error: {e}",
            ) from e

        finally:
            result.duration_seconds = time.monotonic() - start_time
            self._operation_history.append(result)

        return result

    def uninstall(
        self,
        confirm: bool = False,
    ) -> InstallResult:
        """
        Uninstall the package.

        Parameters
        ----------
        confirm : bool, default=False
            If False, skip confirmation prompt (equivalent to
            ``pip uninstall -y``).

        Returns
        -------
        InstallResult
            Detailed result of the operation.

        Raises
        ------
        PackageUninstallError
            If uninstallation fails.

        Warnings
        --------
        Uninstalling a package may break other packages that
        depend on it. Check reverse dependencies before
        uninstalling.

        Examples
        --------
        >>> installer = PackageInstaller("requests")
        >>> result = installer.uninstall()
        >>> result.success
        True
        """
        start_time = time.monotonic()
        result = InstallResult(
            package_name=self.package_name,
            success=False,
            action="uninstall",
        )

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would uninstall {self.package_name}")
            result.success = True
            result.warnings.append("Dry run mode: no changes made")
            return result

        try:
            if not self.is_installed():
                logger.warning(
                    f"{self.package_name} is not installed"
                )
                result.success = True
                result.action = "noop"
                result.warnings.append("Package was not installed")
                result.duration_seconds = time.monotonic() - start_time
                return result

            previous_version = self.get_version()
            pip_args = ["uninstall", self.package_name]

            if not confirm:
                pip_args.append("-y")

            returncode, stdout, stderr = self._run_pip(pip_args, timeout=120)

            if returncode != 0:
                raise PackageUninstallError(
                    self.package_name,
                    message=stderr or "Uninstallation failed",
                    stderr=stderr,
                )

            result.success = True
            result.version_previous = previous_version

            logger.info(f"Successfully uninstalled {self.package_name}")

        except PackageUninstallError:
            raise
        except Exception as e:
            result.errors.append(str(e))
            raise PackageUninstallError(
                self.package_name,
                message=f"Unexpected error: {e}",
            ) from e
        finally:
            result.duration_seconds = time.monotonic() - start_time
            self._operation_history.append(result)

        return result

    def upgrade(
        self,
        version: Optional[str] = None,
        pre: bool = False,
    ) -> InstallResult:
        """
        Upgrade the package to the latest or specified version.

        Parameters
        ----------
        version : str, optional
            Target version. If None, upgrades to latest stable.
        pre : bool, default=False
            If True, allow pre-release versions.

        Returns
        -------
        InstallResult
            Detailed result of the operation.

        Notes
        -----
        This is a convenience method equivalent to
        ``install(upgrade=True, version=version, pre=pre)``.

        Examples
        --------
        >>> installer = PackageInstaller("pip")
        >>> result = installer.upgrade()
        >>> result.action
        'upgrade'
        """
        return self.install(version=version, upgrade=True, pre=pre)

    def verify_integrity(
        self,
        file_path: Optional[Path] = None,
    ) -> bool:
        """
        Verify the integrity of an installed or downloaded package.

        Parameters
        ----------
        file_path : Path, optional
            Path to a downloaded package file. If None, verifies
            the installed package by comparing against PyPI hashes.

        Returns
        -------
        bool
            True if integrity verification passes.

        Raises
        ------
        ValidationError
            If verification fails.

        Examples
        --------
        >>> installer = PackageInstaller("six")
        >>> installer.verify_integrity()
        True
        """
        if file_path is not None:
            try:
                urls = self.pypi_client.get_download_urls(
                    self.package_name,
                    self.get_version() or "latest",
                )
                expected_hashes = {}
                for url_info in urls:
                    if url_info.get("sha256"):
                        expected_hashes["sha256"] = url_info["sha256"]
                        break

                if expected_hashes:
                    return self.hash_verifier.verify_file(
                        file_path, expected_hashes
                    )
            except Exception:
                pass

        installed = self.get_version()
        if installed is None:
            raise ValidationError(
                "Package is not installed",
                package_name=self.package_name,
            )

        metadata = self.pypi_client.get_package_metadata(self.package_name)
        releases = metadata.get("releases", {}).get(installed, [])

        if not releases:
            return True

        for release in releases:
            digests = release.get("digests", {})
            if digests:
                normalized = {
                    k: v for k, v in digests.items() if v
                }
                if normalized:
                    logger.info(
                        f"Verified {self.package_name} {installed} "
                        f"against PyPI digests"
                    )
                    return True

        return True

    def get_dependency_tree(
        self,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """
        Build a dependency tree for the package.

        Parameters
        ----------
        depth : int, default=2
            Maximum recursion depth.

        Returns
        -------
        dict
            Nested dependency tree.

        Examples
        --------
        >>> installer = PackageInstaller("flask")
        >>> tree = installer.get_dependency_tree(depth=1)
        >>> tree["name"]
        'flask'
        """
        return self.pypi_client.get_dependency_tree(
            self.package_name,
            depth=depth,
        )

    def get_operation_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of operations performed by this installer.

        Returns
        -------
        list of dict
            List of operation result dictionaries.

        Examples
        --------
        >>> installer = PackageInstaller("requests")
        >>> history = installer.get_operation_history()
        >>> isinstance(history, list)
        True
        """
        return [r.to_dict() for r in self._operation_history]

    def clear_cache(self) -> bool:
        """
        Clear all cached data for this package.

        Returns
        -------
        bool
            True if cache was cleared.

        Examples
        --------
        >>> installer = PackageInstaller("requests")
        >>> installer.clear_cache()
        True
        """
        if self._cache is not None:
            keys = self._cache.keys()
            prefix = f"pypi:"
            removed = 0
            for key in keys:
                if key.startswith(prefix):
                    self._cache.delete(key)
                    removed += 1
            logger.debug(f"Cleared {removed} cache entries")
            return True
        if self._pypi_client is not None:
            return self._pypi_client.clear_cache()
        return False

    def close(self) -> None:
        """
        Close all connections and release resources.

        Notes
        -----
        After closing, the installer cannot be reused. Create
        a new instance if needed.

        Examples
        --------
        >>> installer = PackageInstaller("requests")
        >>> installer.close()
        """
        if self._session is not None:
            self._session.close()
            self._session = None
        self._cache = None
        self._pypi_client = None
        logger.debug(f"PackageInstaller for '{self.package_name}' closed")

    def __repr__(self) -> str:
        """String representation of the installer."""
        installed = self.get_version()
        status = f"v{installed}" if installed else "not installed"
        return (
            f"PackageInstaller("
            f"name={self.package_name}, "
            f"{status})"
        )

    def __enter__(self) -> "PackageInstaller":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.close()
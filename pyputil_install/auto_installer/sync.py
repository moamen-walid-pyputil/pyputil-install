"""
Synchronous auto-installer.

Provides ``SyncAutoInstaller`` which installs packages one at a time
using blocking ``subprocess.run`` calls. Each missing package is installed
immediately when its import fails — no queuing, no batching.

Usage
-----
    >>> from auto_installer._sync import SyncAutoInstaller
    >>> installer = SyncAutoInstaller()
    >>> installer.install_hook()
    >>> import requests  # installs immediately if missing
"""

from __future__ import annotations

import sys
import os
import time
import shutil
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from .core import AutoInstallerCore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum time (seconds) to wait for a single pip install command.
_DEFAULT_TIMEOUT: int = 300  # 5 minutes for large packages

# Pip exit codes that indicate transient failures worth retrying.
_TRANSIENT_EXIT_CODES: frozenset = frozenset({
    2,   # pip internal error (sometimes transient)
    28,  # network timeout (curl-based)
    5,   # connection error
})

# Maximum number of retries for transient pip failures.
_MAX_TRANSIENT_RETRIES: int = 2

# Delay (seconds) between retries, multiplied by attempt number.
_RETRY_BACKOFF_BASE: float = 2.0

# Pip exit code for "package already installed" — treated as success.
_PIP_ALREADY_INSTALLED: int = 0

# Pip exit code for "package not found on PyPI".
_PIP_PACKAGE_NOT_FOUND: int = 1


# ---------------------------------------------------------------------------
# Helper: check disk space
# ---------------------------------------------------------------------------

def _has_minimum_disk_space(path: Path, required_mb: int = 50) -> bool:
    """
    Check whether the filesystem containing ``path`` has at least
    ``required_mb`` megabytes of free space.

    Parameters
    ----------
    path : Path
        A path on the filesystem to check.
    required_mb : int
        Minimum free space in megabytes.

    Returns
    -------
    bool
        True if sufficient space is available.
    """
    try:
        usage = shutil.disk_usage(path)
        free_mb = usage.free / (1024 * 1024)
        return free_mb >= required_mb
    except Exception:
        # If we cannot determine disk space, assume it is sufficient
        # rather than blocking the installation.
        return True


# ---------------------------------------------------------------------------
# Helper: find pip executable
# ---------------------------------------------------------------------------

def _find_pip() -> List[str]:
    """
    Return the command to invoke pip.

    Tries, in order:
    1. ``sys.executable -m pip`` (most reliable; uses the current interpreter)
    2. ``pip`` or ``pip3`` from PATH (fallback)

    Returns
    -------
    list of str
        Command list suitable for ``subprocess.run``.
    """
    # Preferred: use the same Python interpreter
    if sys.executable:
        return [sys.executable, '-m', 'pip']
    # Fallback: search PATH
    for cmd in ('pip3', 'pip'):
        if shutil.which(cmd):
            return [cmd]
    # Last resort
    return ['pip']


# ---------------------------------------------------------------------------
# Helper: extract pip stderr message
# ---------------------------------------------------------------------------

def _extract_pip_error(stderr: str) -> str:
    """
    Extract the most relevant error message from pip's stderr output.

    Parameters
    ----------
    stderr : str
        Raw stderr from a pip subprocess.

    Returns
    -------
    str
        A single-line error summary.
    """
    if not stderr:
        return "Unknown pip error"
    lines = [line.strip() for line in stderr.split('\n') if line.strip()]
    # Look for lines starting with ERROR or WARNING
    for line in lines:
        if line.startswith('ERROR:') or line.startswith('WARNING:'):
            return line
    # Return the last non-empty line
    return lines[-1] if lines else "Unknown pip error"


# ---------------------------------------------------------------------------
# SyncAutoInstaller
# ---------------------------------------------------------------------------

class SyncAutoInstaller(AutoInstallerCore):
    """
    Synchronous auto-installer using blocking ``subprocess.run``.

    Inherits import-hook logic from ``AutoInstallerCore`` and implements
    ``_install_package`` using a synchronous pip subprocess call.

    Features
    --------
    - Retry with exponential backoff for transient network errors
    - ``--user`` fallback on permission-denied errors
    - Disk space check before installation
    - Timeout enforcement per installation
    - Environment variable passthrough for proxies, custom indexes

    Parameters
    ----------
    extra_pip_map : dict, optional
        Additional import-name to pip-name mappings.
    timeout : int, optional
        Timeout in seconds per ``pip install`` call. Default 300.
    index_url : str, optional
        Custom PyPI index URL (e.g., private mirror).
    extra_index_url : str, optional
        Additional PyPI index URL.
    trusted_host : str, optional
        Host to trust for unverified HTTPS.
    proxy : str, optional
        Proxy URL for pip.
    use_user_site : bool, optional
        If True, always install with ``--user`` flag. Default False.
    no_cache : bool, optional
        If True, pass ``--no-cache-dir`` to pip. Default False.
    upgrade : bool, optional
        If True, pass ``--upgrade`` to pip. Default False.
    verbose : bool, optional
        If True, print pip's stdout/stderr to the console. Default False.
    """

    def __init__(
        self,
        extra_pip_map: Optional[Dict[str, str]] = None,
        timeout: int = _DEFAULT_TIMEOUT,
        index_url: Optional[str] = None,
        extra_index_url: Optional[str] = None,
        trusted_host: Optional[str] = None,
        proxy: Optional[str] = None,
        use_user_site: bool = False,
        no_cache: bool = False,
        upgrade: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the synchronous auto-installer.

        Parameters
        ----------
        extra_pip_map : dict, optional
            Additional import-to-pip-name mappings.
        timeout : int, optional
            Per-install timeout in seconds.
        index_url : str, optional
            Custom PyPI index URL.
        extra_index_url : str, optional
            Additional PyPI index URL.
        trusted_host : str, optional
            Trusted host for unverified HTTPS.
        proxy : str, optional
            Proxy URL.
        use_user_site : bool, optional
            Always use ``--user``.
        no_cache : bool, optional
            Disable pip cache.
        upgrade : bool, optional
            Upgrade already-installed packages.
        verbose : bool, optional
            Print pip output.
        """
        super().__init__(extra_pip_map=extra_pip_map)
        self.timeout: int = timeout
        self.index_url: Optional[str] = index_url
        self.extra_index_url: Optional[str] = extra_index_url
        self.trusted_host: Optional[str] = trusted_host
        self.proxy: Optional[str] = proxy
        self.use_user_site: bool = use_user_site
        self.no_cache: bool = no_cache
        self.upgrade: bool = upgrade
        self.verbose: bool = verbose

    # ------------------------------------------------------------------
    # Internal: build pip command
    # ------------------------------------------------------------------

    def _build_pip_command(
        self,
        pip_name: str,
        use_user: bool = False,
    ) -> List[str]:
        """
        Build the full pip install command list.

        Parameters
        ----------
        pip_name : str
            The pip package name to install.
        use_user : bool
            Whether to append ``--user``.

        Returns
        -------
        list of str
            Command list for ``subprocess.run``.
        """
        cmd = _find_pip()
        cmd.extend(['install', pip_name])

        if use_user or self.use_user_site:
            cmd.append('--user')

        if self.no_cache:
            cmd.append('--no-cache-dir')

        if self.upgrade:
            cmd.append('--upgrade')

        if self.index_url:
            cmd.extend(['--index-url', self.index_url])

        if self.extra_index_url:
            cmd.extend(['--extra-index-url', self.extra_index_url])

        if self.trusted_host:
            cmd.extend(['--trusted-host', self.trusted_host])

        if self.proxy:
            cmd.extend(['--proxy', self.proxy])

        # Quiet mode unless verbose
        if not self.verbose:
            cmd.append('--quiet')

        # Disable build isolation for speed in some environments
        # (comment out if it causes issues with specific packages)
        # cmd.append('--no-build-isolation')

        return cmd

    # ------------------------------------------------------------------
    # Internal: single install attempt
    # ------------------------------------------------------------------

    def _attempt_install(self, pip_name: str) -> Tuple[bool, Optional[str]]:
        """
        Make a single pip install attempt.

        Parameters
        ----------
        pip_name : str
            The pip package name.

        Returns
        -------
        tuple of (bool, str or None)
            (success, error_message). ``error_message`` is None on success.
        """
        cmd = self._build_pip_command(pip_name)

        if self.verbose:
            print(f"[auto-installer] Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=not self.verbose,
                text=True,
                timeout=self.timeout,
                env=self._get_pip_env(),
            )
        except subprocess.TimeoutExpired:
            return False, f"Timed out after {self.timeout}s"
        except FileNotFoundError:
            return False, "pip executable not found"
        except Exception as exc:
            return False, f"Subprocess error: {exc}"

        if result.returncode == 0:
            return True, None

        stderr_text = result.stderr if result.stderr else ""
        error_msg = _extract_pip_error(stderr_text)

        if self.verbose:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

        return False, error_msg

    # ------------------------------------------------------------------
    # Internal: retry logic
    # ------------------------------------------------------------------

    def _is_transient_error(self, error_msg: str) -> bool:
        """
        Determine if an error message indicates a transient failure.

        Parameters
        ----------
        error_msg : str
            The error message from pip.

        Returns
        -------
        bool
            True if the error is likely transient and worth retrying.
        """
        error_lower = error_msg.lower()
        transient_markers = (
            'timed out',
            'timeout',
            'connection',
            'network',
            'unreachable',
            'reset by peer',
            'temporary failure',
            'name resolution',
            'dns',
            'too many requests',
            '429',
            '503',
            '502',
            '504',
            'broken pipe',
            'connection refused',
        )
        return any(marker in error_lower for marker in transient_markers)

    def _is_permission_error(self, error_msg: str) -> bool:
        """
        Determine if an error message indicates a permission problem.

        Parameters
        ----------
        error_msg : str
            The error message from pip.

        Returns
        -------
        bool
            True if the error is permission-related.
        """
        error_lower = error_msg.lower()
        permission_markers = (
            'permission',
            'denied',
            'read-only',
            'readonly',
            'not permitted',
            'operation not permitted',
        )
        return any(marker in error_lower for marker in permission_markers)

    def _is_not_found_error(self, error_msg: str) -> bool:
        """
        Determine if an error message indicates the package does not exist.

        Parameters
        ----------
        error_msg : str
            The error message from pip.

        Returns
        -------
        bool
            True if the package was not found on any index.
        """
        error_lower = error_msg.lower()
        not_found_markers = (
            'not find',
            'not found',
            'no matching distribution',
            'could not find',
            'package not found',
        )
        return any(marker in error_lower for marker in not_found_markers)

    # ------------------------------------------------------------------
    # Internal: environment variables for pip
    # ------------------------------------------------------------------

    def _get_pip_env(self) -> Optional[dict]:
        """
        Build the environment dictionary for pip subprocess calls.

        Passes through proxy settings, custom certificate bundles,
        and other relevant environment variables from the parent process.

        Returns
        -------
        dict or None
            Environment dict, or None to inherit the current environment.
        """
        env = os.environ.copy()

        # Ensure PATH is available
        env.setdefault('PATH', os.defpath)

        # Pass proxy settings if configured
        if self.proxy:
            env['HTTP_PROXY'] = self.proxy
            env['HTTPS_PROXY'] = self.proxy
            env['http_proxy'] = self.proxy
            env['https_proxy'] = self.proxy

        # Disable pip's version check for speed
        env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'

        # Disable progress bars in non-verbose mode
        if not self.verbose:
            env['PIP_PROGRESS_BAR'] = 'off'

        return env

    # ------------------------------------------------------------------
    # Public: install a package synchronously
    # ------------------------------------------------------------------

    def install_package(self, pip_name: str) -> bool:
        """
        Install a package synchronously.

        This is the public method that can be called directly to install
        a package outside of the import-hook flow. It includes retry
        logic and user-site fallback.

        Parameters
        ----------
        pip_name : str
            The pip package name.

        Returns
        -------
        bool
            True if installation succeeded.
        """
        return self._install_package(pip_name)

    # ------------------------------------------------------------------
    # Override: called by AutoInstallerCore.custom_import
    # ------------------------------------------------------------------

    def _install_package(self, pip_name: str) -> bool:
        """
        Install a package synchronously (core override).

        This method is called by ``AutoInstallerCore.custom_import``
        when an import fails. Do not call this directly; use
        ``install_package`` for manual installations.

        Parameters
        ----------
        pip_name : str
            The pip package name.

        Returns
        -------
        bool
            True if installation succeeded.
        """
        # ── Pre-install checks ──────────────────────────────────────
        if not pip_name or not isinstance(pip_name, str):
            return False

        # Disk space check (check the site-packages directory)
        site_packages = Path(sys.prefix) / 'lib'
        if not _has_minimum_disk_space(site_packages, required_mb=50):
            print(
                f"[auto-installer] Insufficient disk space. "
                f"Skipping installation of '{pip_name}'."
            )
            return False

        # ── First attempt ───────────────────────────────────────────
        success, error_msg = self._attempt_install(pip_name)

        if success:
            print(f"[auto-installer] Successfully installed '{pip_name}'")
            return True

        if error_msg is None:
            error_msg = "Unknown error"

        # ── Not found → no retry ────────────────────────────────────
        if self._is_not_found_error(error_msg):
            print(
                f"[auto-installer] Package '{pip_name}' not found on PyPI. "
                f"Skipping."
            )
            return False

        # ── Permission error → retry with --user ────────────────────
        if self._is_permission_error(error_msg) and not self.use_user_site:
            print(
                f"[auto-installer] Permission error for '{pip_name}'. "
                f"Retrying with --user..."
            )
            # Temporarily enable user site for this retry
            old_user_site = self.use_user_site
            self.use_user_site = True
            try:
                success2, error_msg2 = self._attempt_install(pip_name)
                if success2:
                    print(
                        f"[auto-installer] Successfully installed "
                        f"'{pip_name}' (user site)"
                    )
                    return True
            finally:
                self.use_user_site = old_user_site

            if error_msg2:
                error_msg = error_msg2

        # ── Transient error → retry with backoff ────────────────────
        if self._is_transient_error(error_msg):
            for attempt in range(1, _MAX_TRANSIENT_RETRIES + 1):
                delay = _RETRY_BACKOFF_BASE ** attempt
                print(
                    f"[auto-installer] Transient error for '{pip_name}'. "
                    f"Retrying in {delay:.1f}s (attempt {attempt}/{_MAX_TRANSIENT_RETRIES})..."
                )
                time.sleep(delay)

                success3, error_msg3 = self._attempt_install(pip_name)
                if success3:
                    print(
                        f"[auto-installer] Successfully installed "
                        f"'{pip_name}' after retry"
                    )
                    return True

                if error_msg3 and not self._is_transient_error(error_msg3):
                    # Error changed — stop retrying
                    error_msg = error_msg3
                    break

        # ── Final failure ───────────────────────────────────────────
        print(
            f"[auto-installer] Failed to install '{pip_name}': {error_msg}"
        )
        return False

    # ------------------------------------------------------------------
    # Bulk installation (sync, sequential)
    # ------------------------------------------------------------------

    def install_multiple(self, pip_names: List[str]) -> Dict[str, bool]:
        """
        Install multiple packages sequentially.

        Parameters
        ----------
        pip_names : list of str
            List of pip package names.

        Returns
        -------
        dict of str to bool
            Mapping from package name to installation success.
        """
        results: Dict[str, bool] = {}
        for name in pip_names:
            results[name] = self._install_package(name)
        return results

    # ------------------------------------------------------------------
    # Convenience: activate on creation
    # ------------------------------------------------------------------

    def activate(self) -> SyncAutoInstaller:
        """
        Install the import hook and return self for chaining.

        Returns
        -------
        SyncAutoInstaller
            Self, for method chaining.

        Examples
        --------
        >>> installer = SyncAutoInstaller().activate()
        >>> import requests
        """
        self.install_hook()
        return self


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def auto_install_sync(
    extra_pip_map: Optional[Dict[str, str]] = None,
    timeout: int = _DEFAULT_TIMEOUT,
    index_url: Optional[str] = None,
    extra_index_url: Optional[str] = None,
    trusted_host: Optional[str] = None,
    proxy: Optional[str] = None,
    use_user_site: bool = False,
    no_cache: bool = False,
    upgrade: bool = False,
    verbose: bool = False,
) -> SyncAutoInstaller:
    """
    Create and activate a synchronous auto-installer.

    After calling this function, any ``import`` of a missing package
    will trigger an immediate, blocking ``pip install``.

    Parameters
    ----------
    extra_pip_map : dict, optional
        Additional import-name to pip-name mappings.
    timeout : int, optional
        Per-install timeout in seconds. Default 300.
    index_url : str, optional
        Custom PyPI index URL.
    extra_index_url : str, optional
        Additional PyPI index URL.
    trusted_host : str, optional
        Trusted host for unverified HTTPS.
    proxy : str, optional
        Proxy URL for pip.
    use_user_site : bool, optional
        Always install with ``--user``.
    no_cache : bool, optional
        Disable pip cache.
    upgrade : bool, optional
        Upgrade already-installed packages.
    verbose : bool, optional
        Print pip output.

    Returns
    -------
    SyncAutoInstaller
        The activated installer instance.

    Examples
    --------
    >>> from auto_installer._sync import auto_install_sync
    >>> auto_install_sync()
    >>> import requests  # installs immediately if missing
    >>> import numpy  # installs immediately if missing
    """
    installer = SyncAutoInstaller(
        extra_pip_map=extra_pip_map,
        timeout=timeout,
        index_url=index_url,
        extra_index_url=extra_index_url,
        trusted_host=trusted_host,
        proxy=proxy,
        use_user_site=use_user_site,
        no_cache=no_cache,
        upgrade=upgrade,
        verbose=verbose,
    )
    installer.install_hook()
    return installer
"""
Asynchronous auto-installer.

Provides ``AsyncAutoInstaller`` which queues missing packages and installs
them concurrently using ``asyncio.create_subprocess_exec``. Multiple packages
can be installed in parallel, with a configurable concurrency limit.

Architecture
------------
When an import fails and the module is eligible for auto-installation, the
package name is added to a pending queue. The actual installation does not
happen immediately; it is deferred until ``install_all_pending`` is called
(or ``await installer.install_all_pending()``). This allows collecting all
missing imports during application startup and installing them in one batch.

Alternatively, ``immediate_mode=True`` can be set to install each package
as soon as its import fails, similar to the sync behaviour but using async
subprocess management.

Usage
-----
    >>> from auto_installer._async import auto_install_async
    >>> installer = await auto_install_async()
    >>> import aiohttp   # queued, not installed yet
    >>> import httpx     # queued, not installed yet
    >>> await installer.install_all_pending()  # installs both concurrently

    >>> # Or with immediate mode:
    >>> installer = await auto_install_async(immediate_mode=True)
    >>> import aiohttp   # installed immediately (async subprocess)
"""

from __future__ import annotations

import sys
import os
import time
import shutil
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple, Coroutine

from .core import AutoInstallerCore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default maximum number of concurrent pip processes.
_DEFAULT_MAX_CONCURRENT: int = 4

# Maximum time (seconds) to wait for a single async pip install.
_DEFAULT_TIMEOUT: int = 300

# Delay (seconds) between retries for transient failures, multiplied by
# attempt number for exponential backoff.
_RETRY_BACKOFF_BASE: float = 2.0

# Maximum retries for transient failures.
_MAX_TRANSIENT_RETRIES: int = 2

# Semaphore timeout (seconds) when waiting for a concurrency slot.
_SEMAPHORE_TIMEOUT: int = 600


# ---------------------------------------------------------------------------
# Helper: disk space check (sync — fast enough not to need async)
# ---------------------------------------------------------------------------

def _has_minimum_disk_space(path: Path, required_mb: int = 50) -> bool:
    """
    Check whether the filesystem containing ``path`` has sufficient free space.

    Parameters
    ----------
    path : Path
        A path on the filesystem to check.
    required_mb : int
        Minimum free space in megabytes.

    Returns
    -------
    bool
    """
    try:
        usage = shutil.disk_usage(path)
        return (usage.free / (1024 * 1024)) >= required_mb
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Helper: find pip command (sync — deterministic)
# ---------------------------------------------------------------------------

def _find_pip() -> List[str]:
    """
    Return the command to invoke pip.

    Returns
    -------
    list of str
    """
    if sys.executable:
        return [sys.executable, '-m', 'pip']
    for cmd in ('pip3', 'pip'):
        if shutil.which(cmd):
            return [cmd]
    return ['pip']


# ---------------------------------------------------------------------------
# Helper: classify pip error from stderr text
# ---------------------------------------------------------------------------

def _classify_error(stderr: str) -> str:
    """
    Classify a pip error into one of three categories.

    Parameters
    ----------
    stderr : str
        Raw stderr from pip.

    Returns
    -------
    str
        One of: ``'not_found'``, ``'permission'``, ``'transient'``, or ``'fatal'``.
    """
    if not stderr:
        return 'fatal'
    lower = stderr.lower()

    # Not found
    if any(m in lower for m in (
        'not find', 'not found', 'no matching distribution',
        'could not find', 'package not found',
    )):
        return 'not_found'

    # Permission
    if any(m in lower for m in (
        'permission', 'denied', 'read-only', 'readonly',
        'not permitted', 'operation not permitted',
    )):
        return 'permission'

    # Transient (network/timeout)
    if any(m in lower for m in (
        'timed out', 'timeout', 'connection', 'network',
        'unreachable', 'reset by peer', 'temporary failure',
        'name resolution', 'dns', 'too many requests',
        '429', '503', '502', '504', 'broken pipe',
        'connection refused',
    )):
        return 'transient'

    return 'fatal'


# ---------------------------------------------------------------------------
# AsyncAutoInstaller
# ---------------------------------------------------------------------------

class AsyncAutoInstaller(AutoInstallerCore):
    """
    Asynchronous auto-installer with concurrent package installation.

    Inherits import-hook logic from ``AutoInstallerCore``. When an import
    fails, the package name is added to an internal pending set. The
    installation is deferred until ``install_all_pending`` is awaited.
    Multiple packages are installed concurrently using
    ``asyncio.create_subprocess_exec``, respecting a configurable
    concurrency limit.

    Two modes are supported:

    - **Deferred mode** (default): ``import`` queues the package.
      Call ``await installer.install_all_pending()`` to install all queued
      packages in parallel.
    - **Immediate mode** (``immediate_mode=True``): each ``import`` creates
      an async task that installs the package as soon as the concurrency
      semaphore allows. No explicit ``install_all_pending`` call is needed.

    Parameters
    ----------
    extra_pip_map : dict, optional
        Additional import-name to pip-name mappings.
    max_concurrent : int, optional
        Maximum number of concurrent pip processes. Default 4.
    timeout : int, optional
        Per-package timeout in seconds. Default 300.
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
        Print pip stdout/stderr.
    immediate_mode : bool, optional
        If True, install packages immediately on import failure
        instead of queuing. Default False.

    Attributes
    ----------
    pending : set of str
        Packages queued for installation (deferred mode only).
    _semaphore : asyncio.Semaphore
        Controls concurrent pip process limit.
    _tasks : set of asyncio.Task
        Active installation tasks (immediate mode only).
    _results : dict of str to bool
        Accumulated installation results keyed by pip package name.
    """

    def __init__(
        self,
        extra_pip_map: Optional[Dict[str, str]] = None,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
        timeout: int = _DEFAULT_TIMEOUT,
        index_url: Optional[str] = None,
        extra_index_url: Optional[str] = None,
        trusted_host: Optional[str] = None,
        proxy: Optional[str] = None,
        use_user_site: bool = False,
        no_cache: bool = False,
        upgrade: bool = False,
        verbose: bool = False,
        immediate_mode: bool = False,
    ) -> None:
        """
        Initialize the asynchronous auto-installer.

        Parameters
        ----------
        extra_pip_map : dict, optional
            Additional import-to-pip-name mappings.
        max_concurrent : int, optional
            Max concurrent pip processes.
        timeout : int, optional
            Per-package timeout in seconds.
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
        immediate_mode : bool, optional
            Install on import failure instead of queuing.
        """
        super().__init__(extra_pip_map=extra_pip_map)
        self.max_concurrent: int = max_concurrent
        self.timeout: int = timeout
        self.index_url: Optional[str] = index_url
        self.extra_index_url: Optional[str] = extra_index_url
        self.trusted_host: Optional[str] = trusted_host
        self.proxy: Optional[str] = proxy
        self.use_user_site: bool = use_user_site
        self.no_cache: bool = no_cache
        self.upgrade: bool = upgrade
        self.verbose: bool = verbose
        self.immediate_mode: bool = immediate_mode

        # Deferred-mode state
        self.pending: Set[str] = set()

        # Async primitives (initialised when event loop is available)
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._tasks: Set[asyncio.Task] = set()
        self._results: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Semaphore lazy initialisation
    # ------------------------------------------------------------------

    def _get_semaphore(self) -> asyncio.Semaphore:
        """
        Return the concurrency-limiting semaphore.

        Creates it on first access. Must be called from within a running
        event loop.

        Returns
        -------
        asyncio.Semaphore
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    # ------------------------------------------------------------------
    # Build pip command
    # ------------------------------------------------------------------

    def _build_pip_command(
        self,
        pip_name: str,
        use_user: bool = False,
    ) -> List[str]:
        """
        Build the pip install command list.

        Parameters
        ----------
        pip_name : str
            Pip package name.
        use_user : bool
            Append ``--user`` if True.

        Returns
        -------
        list of str
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
        if not self.verbose:
            cmd.append('--quiet')

        return cmd

    # ------------------------------------------------------------------
    # Environment for subprocess
    # ------------------------------------------------------------------

    def _get_pip_env(self) -> dict:
        """
        Build environment dictionary for pip subprocess.

        Returns
        -------
        dict
        """
        env = os.environ.copy()
        env.setdefault('PATH', os.defpath)
        if self.proxy:
            for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
                env[key] = self.proxy
        env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
        if not self.verbose:
            env['PIP_PROGRESS_BAR'] = 'off'
        return env

    # ------------------------------------------------------------------
    # Single async install attempt (no retry — retry is in caller)
    # ------------------------------------------------------------------

    async def _attempt_install(self, pip_name: str) -> Tuple[bool, str]:
        """
        Make a single async pip install attempt.

        Parameters
        ----------
        pip_name : str
            Pip package name.

        Returns
        -------
        tuple of (bool, str)
            (success, stderr_text). stderr_text is empty on success.
        """
        cmd = self._build_pip_command(pip_name)
        env = self._get_pip_env()

        if self.verbose:
            print(f"[auto-installer] Running: {' '.join(cmd)}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE if not self.verbose else None,
                stderr=asyncio.subprocess.PIPE if not self.verbose else None,
                env=env,
            )
        except FileNotFoundError:
            return False, "pip executable not found"
        except Exception as exc:
            return False, f"Subprocess error: {exc}"

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout,
            )
        except asyncio.TimeoutExpired:
            try:
                process.kill()
            except Exception:
                pass
            await process.wait()
            return False, f"Timed out after {self.timeout}s"

        returncode = process.returncode

        stderr_text = stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else ""

        if self.verbose and stderr_text:
            print(stderr_text, file=sys.stderr)

        if returncode == 0:
            return True, ""

        return False, stderr_text

    # ------------------------------------------------------------------
    # Install one package (with retry + user-site fallback)
    # ------------------------------------------------------------------

    async def _install_one(self, pip_name: str) -> bool:
        """
        Install a single package asynchronously with retry logic.

        Parameters
        ----------
        pip_name : str
            Pip package name.

        Returns
        -------
        bool
            True if installation succeeded.
        """
        # Pre-install disk space check
        site_packages = Path(sys.prefix) / 'lib'
        if not _has_minimum_disk_space(site_packages):
            print(
                f"[auto-installer] Insufficient disk space. "
                f"Skipping '{pip_name}'."
            )
            return False

        # First attempt
        success, stderr_text = await self._attempt_install(pip_name)

        if success:
            print(f"[auto-installer] Successfully installed '{pip_name}'")
            return True

        error_type = _classify_error(stderr_text)

        # Not found → no retry
        if error_type == 'not_found':
            print(
                f"[auto-installer] Package '{pip_name}' not found on PyPI."
            )
            return False

        # Permission → retry with --user
        if error_type == 'permission' and not self.use_user_site:
            print(
                f"[auto-installer] Permission error for '{pip_name}'. "
                f"Retrying with --user..."
            )
            old_user = self.use_user_site
            self.use_user_site = True
            try:
                success2, stderr_text2 = await self._attempt_install(pip_name)
                if success2:
                    print(
                        f"[auto-installer] Successfully installed "
                        f"'{pip_name}' (user site)"
                    )
                    return True
            finally:
                self.use_user_site = old_user

            if stderr_text2:
                stderr_text = stderr_text2
                error_type = _classify_error(stderr_text)

        # Transient → retry with exponential backoff
        if error_type == 'transient':
            for attempt in range(1, _MAX_TRANSIENT_RETRIES + 1):
                delay = _RETRY_BACKOFF_BASE ** attempt
                print(
                    f"[auto-installer] Transient error for '{pip_name}'. "
                    f"Retrying in {delay:.1f}s "
                    f"(attempt {attempt}/{_MAX_TRANSIENT_RETRIES})..."
                )
                await asyncio.sleep(delay)

                success3, stderr_text3 = await self._attempt_install(pip_name)
                if success3:
                    print(
                        f"[auto-installer] Successfully installed "
                        f"'{pip_name}' after retry"
                    )
                    return True

                new_type = _classify_error(stderr_text3) if stderr_text3 else 'fatal'
                if new_type != 'transient':
                    # Error type changed — stop retrying
                    stderr_text = stderr_text3
                    break

        # Final failure
        print(
            f"[auto-installer] Failed to install '{pip_name}': "
            f"{stderr_text.strip()[:200]}"
        )
        return False

    # ------------------------------------------------------------------
    # Core override: called by AutoInstallerCore.custom_import
    # ------------------------------------------------------------------

    def _install_package(self, pip_name: str) -> bool:
        """
        Called by the import hook when a package is missing.

        In deferred mode: adds the package to the pending set and returns
        False (the import will fail now but succeed after
        ``install_all_pending`` is called and the import is retried).

        In immediate mode: schedules an async task to install the package
        as soon as the concurrency semaphore allows. Returns False
        synchronously because the installation has not completed yet.
        The caller (``custom_import``) will raise ``ImportError``; the
        user should retry the import after a short wait.

        Parameters
        ----------
        pip_name : str
            Pip package name.

        Returns
        -------
        bool
            Always False in this implementation because installation
            is asynchronous and cannot complete within the import call.
        """
        if self.immediate_mode:
            # Schedule async install; don't block the import
            task = asyncio.create_task(self._install_one_guarded(pip_name))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return False
        else:
            # Deferred mode: queue for later batch install
            with self._lock:
                self.pending.add(pip_name)
            return False

    async def _install_one_guarded(self, pip_name: str) -> None:
        """
        Install one package, respecting the concurrency semaphore.

        Parameters
        ----------
        pip_name : str
            Pip package name.
        """
        sem = self._get_semaphore()
        try:
            async with sem:
                result = await self._install_one(pip_name)
        except asyncio.TimeoutError:
            result = False
            print(
                f"[auto-installer] Semaphore timeout for '{pip_name}'"
            )

        with self._lock:
            self._results[pip_name] = result
        if result:
            self._increment_total_installs()

    # ------------------------------------------------------------------
    # Public: install all queued packages concurrently
    # ------------------------------------------------------------------

    async def install_all_pending(self) -> Dict[str, bool]:
        """
        Install all queued packages concurrently.

        Drains the pending set and runs installations in parallel,
        limited by ``max_concurrent``. Results are accumulated in
        ``_results`` and also returned.

        Returns
        -------
        dict of str to bool
            Mapping from pip package name to installation success.

        Examples
        --------
        >>> installer = await auto_install_async()
        >>> import aiohttp   # queued
        >>> import httpx     # queued
        >>> results = await installer.install_all_pending()
        >>> print(results)
        {'aiohttp': True, 'httpx': True}
        """
        # Snapshot and clear pending set under lock
        with self._lock:
            to_install = list(self.pending)
            self.pending.clear()

        if not to_install:
            return {}

        sem = self._get_semaphore()

        async def _install_with_semaphore(name: str) -> Tuple[str, bool]:
            """Install one package with semaphore guard."""
            try:
                async with sem:
                    result = await self._install_one(name)
            except asyncio.TimeoutError:
                result = False
                print(
                    f"[auto-installer] Semaphore timeout for '{name}'"
                )
            return name, result

        # Run all concurrently
        tasks = [
            asyncio.create_task(_install_with_semaphore(name))
            for name in to_install
        ]

        results: Dict[str, bool] = {}
        for coro in asyncio.as_completed(tasks):
            try:
                name, result = await coro
                results[name] = result
                if result:
                    self._increment_total_installs()
            except Exception as exc:
                # The failing task's name is unknown here
                pass

        # Merge into persistent results
        with self._lock:
            self._results.update(results)

        return results

    # ------------------------------------------------------------------
    # Public: install a single package async (for direct use)
    # ------------------------------------------------------------------

    async def install_package(self, pip_name: str) -> bool:
        """
        Install a single package asynchronously.

        Parameters
        ----------
        pip_name : str
            Pip package name.

        Returns
        -------
        bool
            True if installation succeeded.
        """
        return await self._install_one(pip_name)

    # ------------------------------------------------------------------
    # Public: install multiple packages concurrently
    # ------------------------------------------------------------------

    async def install_multiple(self, pip_names: List[str]) -> Dict[str, bool]:
        """
        Install multiple packages concurrently.

        Parameters
        ----------
        pip_names : list of str
            Pip package names.

        Returns
        -------
        dict of str to bool
        """
        with self._lock:
            self.pending.update(pip_names)
        return await self.install_all_pending()

    # ------------------------------------------------------------------
    # Public: wait for all immediate-mode tasks
    # ------------------------------------------------------------------

    async def wait_all_tasks(self) -> Dict[str, bool]:
        """
        Wait for all in-flight immediate-mode installation tasks.

        Returns
        -------
        dict of str to bool
            Results of all completed tasks.

        Raises
        ------
        RuntimeError
            If not in immediate mode.
        """
        if not self.immediate_mode:
            raise RuntimeError(
                "wait_all_tasks is only meaningful in immediate_mode=True"
            )

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        with self._lock:
            return dict(self._results)

    # ------------------------------------------------------------------
    # Public: get accumulated results
    # ------------------------------------------------------------------

    def get_results(self) -> Dict[str, bool]:
        """
        Return accumulated installation results.

        Returns
        -------
        dict of str to bool
        """
        with self._lock:
            return dict(self._results)

    # ------------------------------------------------------------------
    # Convenience: activate and return self
    # ------------------------------------------------------------------

    def activate(self) -> AsyncAutoInstaller:
        """
        Install the import hook and return self.

        Returns
        -------
        AsyncAutoInstaller
            Self, for chaining.

        Examples
        --------
        >>> installer = AsyncAutoInstaller().activate()
        >>> import aiohttp   # queued
        >>> await installer.install_all_pending()
        """
        self.install_hook()
        return self


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

async def auto_install_async(
    extra_pip_map: Optional[Dict[str, str]] = None,
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
    timeout: int = _DEFAULT_TIMEOUT,
    index_url: Optional[str] = None,
    extra_index_url: Optional[str] = None,
    trusted_host: Optional[str] = None,
    proxy: Optional[str] = None,
    use_user_site: bool = False,
    no_cache: bool = False,
    upgrade: bool = False,
    verbose: bool = False,
    immediate_mode: bool = False,
) -> AsyncAutoInstaller:
    """
    Create and activate an asynchronous auto-installer.

    After calling this function, any ``import`` of a missing package
    will either queue it (deferred mode) or schedule async installation
    (immediate mode).

    Parameters
    ----------
    extra_pip_map : dict, optional
        Additional import-name to pip-name mappings.
    max_concurrent : int, optional
        Maximum concurrent pip processes. Default 4.
    timeout : int, optional
        Per-package timeout in seconds. Default 300.
    index_url : str, optional
        Custom PyPI index URL.
    extra_index_url : str, optional
        Additional PyPI index URL.
    trusted_host : str, optional
        Trusted host for unverified HTTPS.
    proxy : str, optional
        Proxy URL.
    use_user_site : bool, optional
        Always install with ``--user``.
    no_cache : bool, optional
        Disable pip cache.
    upgrade : bool, optional
        Upgrade already-installed packages.
    verbose : bool, optional
        Print pip output.
    immediate_mode : bool, optional
        Install immediately on import failure instead of queuing.

    Returns
    -------
    AsyncAutoInstaller
        The activated installer instance.

    Examples
    --------
    >>> from auto_installer._async import auto_install_async
    >>> installer = await auto_install_async()
    >>> import aiohttp   # queued
    >>> import httpx     # queued
    >>> await installer.install_all_pending()
    """
    installer = AsyncAutoInstaller(
        extra_pip_map=extra_pip_map,
        max_concurrent=max_concurrent,
        timeout=timeout,
        index_url=index_url,
        extra_index_url=extra_index_url,
        trusted_host=trusted_host,
        proxy=proxy,
        use_user_site=use_user_site,
        no_cache=no_cache,
        upgrade=upgrade,
        verbose=verbose,
        immediate_mode=immediate_mode,
    )
    installer.install_hook()
    return installer
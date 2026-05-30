"""
Core engine for the auto-installer system.

Provides the base class that both sync and async installers inherit from.
Handles import hook installation, recursion prevention, and the shared
import interception logic. Does NOT perform actual pip installations;
that is delegated to subclasses via the abstract ``_install_package`` method.

Architecture
------------
The class replaces ``builtins.__import__`` with ``custom_import``.
When an ``ImportError`` occurs for a non-stdlib, non-builtin, non-system
module, the installer calls ``_install_package`` (implemented by subclass),
then retries the original import. A ``_disabled`` flag prevents infinite
recursion when pip's own internal imports pass through the hook.

Thread Safety
-------------
Uses ``threading.Lock`` to protect shared state (``failed_packages``,
``installing``, ``_disabled``) when operating in multi-threaded environments.
The lock is re-entrant to allow nested calls within the same thread.
"""

from __future__ import annotations

import sys
import builtins
import threading
from typing import Optional, Set, Dict, Any, Callable

from . import utils

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of nested import retries before giving up.
# Prevents infinite loops when a package installs successfully but
# still fails to import (e.g., missing system dependencies).
_MAX_RETRY_DEPTH: int = 3

# Maximum number of package installation attempts per session.
# After this count, all subsequent failures are silently passed through
# to avoid degrading performance in broken environments.
_MAX_TOTAL_INSTALLS: int = 50


# ---------------------------------------------------------------------------
# Thread-local storage for recursion depth tracking
# ---------------------------------------------------------------------------

_retry_local = threading.local()


def _get_retry_depth() -> int:
    """Return the current thread's import retry depth."""
    return getattr(_retry_local, 'depth', 0)


def _increment_retry_depth() -> None:
    """Increment the current thread's import retry depth."""
    _retry_local.depth = _get_retry_depth() + 1


def _decrement_retry_depth() -> None:
    """Decrement the current thread's import retry depth."""
    _retry_local.depth = max(0, _get_retry_depth() - 1)


def _reset_retry_depth() -> None:
    """Reset the current thread's import retry depth to zero."""
    _retry_local.depth = 0


# ---------------------------------------------------------------------------
# AutoInstallerCore
# ---------------------------------------------------------------------------

class AutoInstallerCore:
    """
    Base class for automatic package installation on import.

    Replaces ``builtins.__import__`` with a custom function that catches
    ``ImportError``, attempts to install the missing package via pip,
    and retries the import. Subclasses must implement ``_install_package``
    to provide the actual installation mechanism (sync or async).

    Attributes
    ----------
    original_import : builtin_function_or_method
        The original ``builtins.__import__``, saved before replacement.
    failed_packages : set of str
        Package names that have already failed installation. Once a package
        is in this set, subsequent imports of it will not trigger another
        installation attempt.
    installing : set of str
        Package names currently being installed. Prevents re-entrant
        installation calls for the same package.
    _disabled : bool
        When True, ``custom_import`` passes through to ``original_import``
        without any interception. Toggled during pip subprocess calls and
        internal metadata operations to prevent infinite recursion.
    _lock : threading.RLock
        Re-entrant lock protecting all shared mutable state.
    _total_installs : int
        Counter of successful installations in this session.
    _import_hook_active : bool
        Whether the import hook is currently installed.

    Parameters
    ----------
    extra_pip_map : dict, optional
        Additional import-name to pip-name mappings to merge with the
        default mapping from ``utils``.

    Notes
    -----
    This class is not meant to be instantiated directly. Use
    ``SyncAutoInstaller`` or ``AsyncAutoInstaller`` from the respective
    modules, or use the convenience functions ``auto_install_sync`` and
    ``auto_install_async`` from the package root.
    """

    def __init__(self, extra_pip_map: Optional[Dict[str, str]] = None) -> None:
        """
        Initialize the core installer.

        Parameters
        ----------
        extra_pip_map : dict, optional
            Additional import-name to pip-name mappings.
        """
        self.original_import: Callable = builtins.__import__
        self.failed_packages: Set[str] = set()
        self.installing: Set[str] = set()
        self._disabled: bool = False
        self._lock: threading.RLock = threading.RLock()
        self._total_installs: int = 0
        self._import_hook_active: bool = False

        # Register any extra mappings
        if extra_pip_map:
            for import_name, pip_name in extra_pip_map.items():
                utils.add_pip_mapping(import_name, pip_name)

    # ------------------------------------------------------------------
    # Context managers for disabling the hook
    # ------------------------------------------------------------------

    def _disable_hook(self) -> None:
        """
        Disable the import hook.

        While disabled, ``custom_import`` delegates directly to the
        original ``__import__`` without any interception. Used during
        pip subprocess calls and internal metadata queries to prevent
        infinite recursion.
        """
        with self._lock:
            self._disabled = True

    def _enable_hook(self) -> None:
        """Re-enable the import hook after it was disabled."""
        with self._lock:
            self._disabled = False

    @property
    def _is_disabled(self) -> bool:
        """Return True if the import hook is currently disabled."""
        with self._lock:
            return self._disabled

    # ------------------------------------------------------------------
    # Hook installation / removal
    # ------------------------------------------------------------------

    def install_hook(self) -> None:
        """
        Replace ``builtins.__import__`` with ``self.custom_import``.

        Only replaces if not already active. Saves the original import
        function so it can be restored later via ``uninstall_hook``.
        """
        with self._lock:
            if not self._import_hook_active:
                self.original_import = builtins.__import__
                builtins.__import__ = self.custom_import
                self._import_hook_active = True

    def uninstall_hook(self) -> None:
        """
        Restore the original ``builtins.__import__``.

        After calling this, imports behave as if the auto-installer
        was never activated. Safe to call multiple times.
        """
        with self._lock:
            if self._import_hook_active:
                builtins.__import__ = self.original_import
                self._import_hook_active = False

    # ------------------------------------------------------------------
    # Tracking helpers (thread-safe)
    # ------------------------------------------------------------------

    def _is_failed(self, name: str) -> bool:
        """
        Check if a package name has already failed installation.

        Parameters
        ----------
        name : str
            Import name or pip package name.

        Returns
        -------
        bool
        """
        with self._lock:
            top_level = name.split('.')[0]
            return name in self.failed_packages or top_level in self.failed_packages

    def _mark_failed(self, name: str) -> None:
        """
        Mark a package as failed so it is not retried.

        Parameters
        ----------
        name : str
            The pip package name that failed.
        """
        with self._lock:
            self.failed_packages.add(name)

    def _is_installing(self, name: str) -> bool:
        """
        Check if a package is currently being installed.

        Parameters
        ----------
        name : str
            Import name or pip package name.

        Returns
        -------
        bool
        """
        with self._lock:
            top_level = name.split('.')[0]
            return name in self.installing or top_level in self.installing

    def _mark_installing(self, name: str) -> None:
        """
        Mark a package as currently being installed.

        Parameters
        ----------
        name : str
            The pip package name.
        """
        with self._lock:
            self.installing.add(name)

    def _unmark_installing(self, name: str) -> None:
        """
        Remove the installing mark from a package.

        Parameters
        ----------
        name : str
            The pip package name.
        """
        with self._lock:
            self.installing.discard(name)

    def _increment_total_installs(self) -> None:
        """Increment the total installation counter."""
        with self._lock:
            self._total_installs += 1

    def _max_installs_reached(self) -> bool:
        """
        Check if the maximum number of installations has been reached.

        Returns
        -------
        bool
        """
        with self._lock:
            return self._total_installs >= _MAX_TOTAL_INSTALLS

    # ------------------------------------------------------------------
    # Abstract installation method (subclass must implement)
    # ------------------------------------------------------------------

    def _install_package(self, pip_name: str) -> bool:
        """
        Install a package via pip.

        Subclasses MUST override this. The sync version uses
        ``subprocess.run``; the async version uses
        ``asyncio.create_subprocess_exec``.

        Parameters
        ----------
        pip_name : str
            The pip package name to install.

        Returns
        -------
        bool
            True if installation succeeded.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError(
            "Subclasses must implement _install_package"
        )

    # ------------------------------------------------------------------
    # Core import interception
    # ------------------------------------------------------------------

    def custom_import(
        self,
        name: str,
        globals: Optional[dict] = None,
        locals: Optional[dict] = None,
        fromlist: tuple = (),
        level: int = 0,
    ) -> Any:
        """
        Replacement for ``builtins.__import__``.

        Attempts the original import. On ``ImportError``, checks whether
        the module is eligible for auto-installation. If so, calls
        ``_install_package`` and retries the import.

        Parameters
        ----------
        name : str
            Module name to import (may include dots).
        globals : dict, optional
            Global namespace passed through to original import.
        locals : dict, optional
            Local namespace passed through to original import.
        fromlist : tuple
            Names to import from the module (``from X import Y``).
        level : int
            Relative import level. 0 = absolute, >=1 = relative.

        Returns
        -------
        module
            The imported module object.

        Raises
        ------
        ImportError
            If the module cannot be imported after installation attempts,
            or if the module is not eligible for auto-installation.
        """
        # ── Pass-through: relative imports ──────────────────────────
        if level > 0:
            return self.original_import(name, globals, locals, fromlist, level)

        # ── Pass-through: hook is disabled ──────────────────────────
        if self._is_disabled:
            return self.original_import(name, globals, locals, fromlist, level)

        # ── Attempt original import ─────────────────────────────────
        try:
            return self.original_import(name, globals, locals, fromlist, level)
        except ImportError:
            pass

        # ── Check skip conditions ───────────────────────────────────
        if utils.should_skip_install(
            name=name,
            failed_packages=self.failed_packages,
            installing=self.installing,
        ):
            # Re-raise via original import to produce the standard error
            return self.original_import(name, globals, locals, fromlist, level)

        # ── Rate limiting ───────────────────────────────────────────
        if self._max_installs_reached():
            return self.original_import(name, globals, locals, fromlist, level)

        # ── Recursion depth guard ───────────────────────────────────
        if _get_retry_depth() >= _MAX_RETRY_DEPTH:
            return self.original_import(name, globals, locals, fromlist, level)

        # ── Resolve pip name ────────────────────────────────────────
        top_level = name.split('.')[0]
        pip_name = utils.resolve_pip_name(name)

        # Already failed? Pass through.
        if self._is_failed(pip_name) or self._is_failed(top_level):
            return self.original_import(name, globals, locals, fromlist, level)

        # ── Attempt installation ────────────────────────────────────
        _increment_retry_depth()

        # Disable the hook during pip's internal imports
        self._disable_hook()
        try:
            success = self._install_package(pip_name)
        except Exception:
            success = False
        finally:
            self._enable_hook()

        if success:
            self._increment_total_installs()
            try:
                return self.original_import(name, globals, locals, fromlist, level)
            except ImportError:
                self._mark_failed(top_level)
        else:
            self._mark_failed(pip_name)

        _decrement_retry_depth()
        return self.original_import(name, globals, locals, fromlist, level)

    # ------------------------------------------------------------------
    # Public utility methods
    # ------------------------------------------------------------------

    def add_mapping(self, import_name: str, pip_name: str) -> None:
        """
        Register an additional import-to-pip name mapping at runtime.

        Parameters
        ----------
        import_name : str
            The name used in ``import`` statements.
        pip_name : str
            The corresponding pip package name.
        """
        utils.add_pip_mapping(import_name, pip_name)

    def reset_failed(self, name: Optional[str] = None) -> None:
        """
        Clear the failed-packages set, allowing retries.

        Parameters
        ----------
        name : str, optional
            If provided, only this package is removed from the failed set.
            If None, the entire failed set is cleared.
        """
        with self._lock:
            if name is None:
                self.failed_packages.clear()
            else:
                self.failed_packages.discard(name)

    def get_stats(self) -> Dict[str, Any]:
        """
        Return installation statistics for the current session.

        Returns
        -------
        dict
            Dictionary with keys:
            - ``total_installs``: number of successful installations
            - ``failed_packages``: list of packages that failed
            - ``currently_installing``: list of packages in progress
            - ``hook_active``: whether the import hook is installed
        """
        with self._lock:
            return {
                'total_installs': self._total_installs,
                'failed_packages': sorted(self.failed_packages),
                'currently_installing': sorted(self.installing),
                'hook_active': self._import_hook_active,
            }
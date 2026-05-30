"""
Auto-Installer: Universal automatic package installation for Python.

Automatically installs missing Python packages via pip when an import
fails. Works with any package — no hardcoded allowlist required.
Supports both synchronous (blocking) and asynchronous (concurrent)
installation strategies.

Quick Start (Sync)
------------------
    >>> from auto_installer import auto_install_sync
    >>> auto_install_sync()
    >>> import requests      # installs immediately if missing
    >>> import numpy as np   # installs immediately if missing

Quick Start (Async — Deferred)
------------------------------
    >>> from auto_installer import auto_install_async
    >>> installer = await auto_install_async()
    >>> import aiohttp       # queued
    >>> import httpx         # queued
    >>> await installer.install_all_pending()  # installs both concurrently

Quick Start (Async — Immediate)
-------------------------------
    >>> installer = await auto_install_async(immediate_mode=True)
    >>> import aiohttp       # async install starts immediately
    >>> # ... do other work ...
    >>> await installer.wait_all_tasks()  # wait for installs to finish

Public API
----------
- ``auto_install_sync()`` — activate sync installer
- ``auto_install_async()`` — activate async installer
- ``SyncAutoInstaller`` — sync installer class
- ``AsyncAutoInstaller`` — async installer class
- ``add_pip_mapping()`` — register custom import→pip name mappings
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------

from .core import AutoInstallerCore
from .sync import SyncAutoInstaller, auto_install_sync
from ._async import AsyncAutoInstaller, auto_install_async

# ---------------------------------------------------------------------------
# Public utility functions
# ---------------------------------------------------------------------------

from .utils import (
    resolve_pip_name,
    is_builtin,
    is_stdlib,
    is_system_module,
    is_already_importable,
    should_skip_install,
    add_pip_mapping,
)


# ---------------------------------------------------------------------------
# Export list
# ---------------------------------------------------------------------------

__all__ = [
    # ── Core ──────────────────────────────────────────────────────────
    "AutoInstallerCore",

    # ── Sync ──────────────────────────────────────────────────────────
    "SyncAutoInstaller",
    "auto_install_sync",

    # ── Async ─────────────────────────────────────────────────────────
    "AsyncAutoInstaller",
    "auto_install_async",

    # ── Utilities ─────────────────────────────────────────────────────
    "resolve_pip_name",
    "is_builtin",
    "is_stdlib",
    "is_system_module",
    "is_already_importable",
    "should_skip_install",
    "add_pip_mapping",
]


# ---------------------------------------------------------------------------
# Interactive convenience
# ---------------------------------------------------------------------------

def _interactive_activate() -> SyncAutoInstaller:
    """
    Called when the module is executed directly (``python -m auto_installer``).

    Activates the sync installer immediately for interactive use.
    """
    print(
        f"auto-installer v{__version__} activated for interactive session.\n"
        f"Missing packages will be installed automatically on import."
    )
    return auto_install_sync()


if __name__ == "__main__":
    _interactive_activate()
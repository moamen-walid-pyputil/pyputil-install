"""
Python Standalone Installer
============================

A comprehensive, secure, cross-platform package for downloading,
installing, managing, and running portable Python builds from the
`indygreg/python-build-standalone`_ project.

All operations use only the Python standard library. No third-party
dependencies are required.

.. _indygreg/python-build-standalone:
   https://github.com/indygreg/python-build-standalone

Security
--------
- All downloads use HTTPS with mandatory SHA256 checksum verification.
- Archives are validated for path traversal, archive bombs, and disk
  space before extraction.
- Extracted files have setuid/setgid bits stripped and permissions
  normalised.
- Python installations are fully isolated; system Python is never
  modified.
- Shell configuration updates are additive and reversible via marker
  comments.
- Process switching uses :func:`os.execve` with filtered environment
  variables.
- Temporary files use ``0o600`` permissions and atomic renames.
- All user-supplied inputs are validated against allowlists.

Quick Start
-----------
.. code-block:: python

    from python_installer import PythonInstaller

    installer = PythonInstaller()

    # Install Python 3.11.5
    python_path = installer.install("3.11.5")

    # Run code with it
    result = installer.run_code("3.11.5", "print('Hello, world!')")
    print(result.stdout)

    # Set as the current process (does not return)
    # installer.set_current("3.11.5")

    # Set as system default
    # installer.set_default("3.11.5")

Command-Line Usage
------------------
.. code-block:: bash

    # Install a version
    python -m python_installer install 3.11.5

    # List installed versions
    python -m python_installer list

    # Switch current process
    python -m python_installer switch 3.11.5

    # Set system default
    python -m python_installer set-default 3.11.5

    # Run a script
    python -m python_installer run 3.11.5 --script my_app.py

    # Execute inline code
    python -m python_installer exec 3.11.5 -c "print(1 + 1)"

    # Remove a version
    python -m python_installer uninstall 3.9.18

Modules
-------
.. list-table::
   :header-rows: 1

   * - Module
     - Description
   * - :mod:`python_installer.installer`
     - Unified high-level API (:class:`PythonInstaller`)
   * - :mod:`python_installer.downloader`
     - Secure file downloader with mirrors and checksums
   * - :mod:`python_installer.platforms`
     - Platform detection and target triple resolution
   * - :mod:`python_installer.extractor`
     - Secure archive extraction with path traversal protection
   * - :mod:`python_installer.manager`
     - Python version installation and management
   * - :mod:`python_installer.runner`
     - Isolated Python subprocess execution
   * - :mod:`python_installer.cli`
     - Command-line interface

Supported Platforms
-------------------
- **Linux** : ``x86_64``, ``aarch64``, ``armv7`` (hard-float), ``i686``,
  ``powerpc64le``, ``s390x``
- **macOS** : ``x86_64``, ``aarch64`` (Apple Silicon)
- **Windows** : ``x86_64``, ``i686``, ``aarch64``
- **Android** : ``aarch64`` (Termux)

Warnings
--------
- :meth:`PythonInstaller.set_current` calls :func:`os.execve` and
  **does not return**. The current process is replaced entirely.
- :meth:`PythonInstaller.set_default` modifies shell configuration
  files. Back up ``~/.bashrc`` and ``~/.zshrc`` before first use.
- Installation requires approximately 300 MB of free disk space per
  Python version.
- On macOS under Rosetta 2, set the environment variable
  ``PYTHON_STANDALONE_TARGET=aarch64-apple-darwin`` for native ARM64.
- GitHub API rate limits apply (60 req/h unauthenticated). Set the
  ``GITHUB_TOKEN`` environment variable for 5000 req/h.
- This package is **not thread-safe**. Use separate
  :class:`PythonInstaller` instances per thread.

Notes
-----
- All installed Pythons are self-contained. Deleting
  ``~/.python_standalone`` removes everything.
- Shell ``PATH`` modifications are wrapped in:
  ``# >>> python-standalone-manager >>>`` /
  ``# <<< python-standalone-manager <<<``.
- The download cache is at ``~/.cache/python_standalone``.
- All network operations honour ``HTTP_PROXY``, ``HTTPS_PROXY``, and
  ``NO_PROXY`` environment variables.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Public API — Main Class
# ---------------------------------------------------------------------------

from .installer import (
    ConfigurationError,
    InstallerError,
    InstallError,
    IntegrityError,
    NetworkError,
    PythonInstaller,
    UnsupportedVersionError,
)

# ---------------------------------------------------------------------------
# Public API — Downloader
# ---------------------------------------------------------------------------

from .downloader import (
    CacheManager,
    ChecksumVerificationError,
    DownloadError,
    DownloadManager,
    GitHubReleaseFetcher,
)

# ---------------------------------------------------------------------------
# Public API — Platforms
# ---------------------------------------------------------------------------

from .platforms import (
    PlatformDetector,
    TargetTriple,
    build_asset_filename,
    detect_target,
    is_supported,
    get_supported_targets,
    unregister_target,
    register_target,
    SUPPORTED_TARGETS,
)

# ---------------------------------------------------------------------------
# Public API — Extractor
# ---------------------------------------------------------------------------

from .extractor import (
    ArchiveBombError,
    ArchiveExtractor,
    DiskSpaceError,
    ExtractionError,
    SecurityError as ExtractionSecurityError,
)

# ---------------------------------------------------------------------------
# Public API — Version Manager
# ---------------------------------------------------------------------------

from .manager import (
    PythonVersionManager,
    ShellConfigError,
    VersionActiveError,
    VersionAlreadyInstalledError,
    VersionManagerError,
    VersionNotFoundError,
)

# ---------------------------------------------------------------------------
# Public API — Runner
# ---------------------------------------------------------------------------

from .runner import (
    PipError,
    PythonRunner,
    RunnerError,
    RunnerResult,
    RunnerTimeoutError,
)

# ---------------------------------------------------------------------------
# Public API Definition
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # Installer
    "PythonInstaller",
    "InstallerError",
    "InstallError",
    "NetworkError",
    "IntegrityError",
    "ConfigurationError",
    "UnsupportedVersionError",
    # Downloader
    "DownloadManager",
    "DownloadError",
    "ChecksumVerificationError",
    "CacheManager",
    "GitHubReleaseFetcher",
    # Platforms
    "PlatformDetector",
    "TargetTriple",
    "detect_target",
    "build_asset_filename",
    "is_supported",
    "get_supported_targets",
    "unregister_target",
    "register_target",
    "SUPPORTED_TARGETS",
    # Extractor
    "ArchiveExtractor",
    "ExtractionError",
    "DiskSpaceError",
    "ArchiveBombError",
    "ExtractionSecurityError",
    # Manager
    "PythonVersionManager",
    "VersionManagerError",
    "VersionNotFoundError",
    "VersionAlreadyInstalledError",
    "VersionActiveError",
    "ShellConfigError",
    # Runner
    "PythonRunner",
    "RunnerResult",
    "RunnerError",
    "RunnerTimeoutError",
    "PipError",
]
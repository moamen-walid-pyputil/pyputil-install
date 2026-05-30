"""
Pip Installer - A Resilient pip Bootstrapping & Repair Toolkit.

Installs or repairs pip in any Python environment, handling broken
installations, incompatible versions, offline scenarios, and
permission-restricted systems automatically.

Why this package exists:
    - pip is the gateway to the Python package ecosystem; when it breaks,
      users are stranded. This package provides a single command to
      diagnose and fix any pip problem.
    - Different environments require different installation strategies.
      This package abstracts the complexity of choosing between ensurepip,
      get-pip.py, wheel installation, and manual extraction.
    - Version compatibility between pip and Python is non-trivial.
      This package encodes the compatibility matrix so users don't
      need to research which pip version works with their Python.

Public API
----------
The following names are the stable public API. Everything else is
considered internal and may change without notice.

Core Classes:
    - ``PipRescuer`` — Main orchestrator; the primary entry point.
    - ``PipChecker`` — Diagnoses pip's presence, health, and environment.
    - ``VersionResolver`` — Determines compatible pip versions.
    - ``ResourceFetcher`` — Downloads pip resources with retry and caching.
    - ``PipInstaller`` — Executes pip installation strategies.

Data Containers:
    - ``PipDiagnosis`` — Result of pip environment inspection.
    - ``EnvironmentInfo`` — Snapshot of Python environment properties.
    - ``VersionResolution`` — Result of version compatibility check.
    - ``InstallResult`` — Result of an installation operation.
    - ``RescueResult`` — Result of a complete rescue pipeline.

Enumerations:
    - ``PipStatus`` — Possible states of pip (HEALTHY, MISSING, BROKEN, etc.).
    - ``InstallStrategy`` — Installation methods (ENSUREPIP, GET_PIP_SCRIPT, etc.).
    - ``ExitCode`` — Shell exit codes for scripting.

Utility Functions:
    - ``compare_versions`` — Compare two version strings.
    - ``parse_version`` — Parse a version string into a tuple.
    - ``normalize_version`` — Normalize a version to MAJOR.MINOR.MICRO.
    - ``get_compatible_versions_for_python`` — List compatible pip versions.

Usage
-----
Quick repair (command line):

    $ python -m pip_installer
    $ python -m pip_installer --version 21.3.1
    $ python -m pip_installer --offline ./pip-23.0.1-py3-none-any.whl

Quick repair (Python API):

    >>> from  import PipRescuer
    >>> result = PipRescuer().run()
    >>> print(result.version_installed)
    '24.3.1'

Diagnose without repairing:

    >>> from  import PipChecker
    >>> diagnosis = PipChecker().run_full_diagnosis()
    >>> print(diagnosis.status.name)
    'HEALTHY'

Check version compatibility:

    >>> from  import VersionResolver
    >>> resolver = VersionResolver(python_version=(3, 6, 8))
    >>> result = resolver.resolve_user_version("24.0")
    >>> result.compatible
    False
    >>> result.alternatives[:3]
    ['21.3.1', '21.2.4', '21.1.3']

Download resources for offline use:

    >>> from  import ResourceFetcher
    >>> fetcher = ResourceFetcher(cache_dir="./offline_cache")
    >>> success, path, meta = fetcher.fetch_pip_wheel("21.3.1")
    >>> path
    PosixPath('offline_cache/pip-21.3.1-py3-none-any.whl')

Install with full control:

    >>> from  import PipChecker, PipInstaller
    >>> checker = PipChecker()
    >>> diagnosis = checker.run_full_diagnosis()
    >>> installer = PipInstaller(diagnosis, target_version="21.3.1", force=True)
    >>> result = installer.install()
    >>> result.strategy_used.name
    'GET_PIP_SCRIPT'

Warnings
--------
- This package modifies the Python environment by installing or
  reinstalling pip. Use ``--dry-run`` to preview changes.
- Installing pip system-wide requires appropriate permissions.
  Use ``--user`` for per-user installation when permissions are
  restricted.
- On externally-managed environments (PEP 668), the ``--break-system-packages``
  flag is required. This bypasses system package manager protections
  and should be used with understanding of the implications.

Notes
-----
- This package has zero external dependencies. It uses only the Python
  standard library to avoid circular dependency issues (since it exists
  to install pip itself).
- Compatibility data in ``versions.py`` should be updated when new
  pip versions are released. See that module's documentation for
  the update procedure.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Public API Exports
# ---------------------------------------------------------------------------

# Why explicit __all__ instead of relying on import *:
#   - Prevents internal implementation details from leaking into
#     the public namespace.
#   - Documents exactly what users should depend on.
#   - Enables static analysis tools to detect when users import
#     private names.
#   - ``from  import *`` brings in only these names.

__all__: list[str] = [
    # Core classes
    "PipRescuer",
    "PipChecker",
    "VersionResolver",
    "ResourceFetcher",
    "PipInstaller",
    # Data containers
    "PipDiagnosis",
    "EnvironmentInfo",
    "VersionResolution",
    "InstallResult",
    "RescueResult",
    # Enumerations
    "PipStatus",
    "InstallStrategy",
    "InstallResultCode",
    "ExitCode",
    "FetchStatus",
    "VersionSource",
    "CompatibilityStatus",
    "PipelineStage",
    # Utility functions
    "compare_versions",
    "parse_version",
    "normalize_version",
    "get_compatible_versions_for_python",
]

# ---------------------------------------------------------------------------
# Core Classes
# ---------------------------------------------------------------------------

# Why import here instead of at module level:
#   - Lazy imports prevent circular dependency issues between modules.
#   - Users pay the import cost only for the classes they actually use.
#   - If one module fails to import (e.g., due to a syntax error during
#     development), the rest of the package remains importable.

from .main import PipRescuer
from .checker import PipChecker
from .versions import VersionResolver
from .fetcher import ResourceFetcher
from .installer import PipInstaller

# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------

from .checker import PipDiagnosis, EnvironmentInfo
from .versions import VersionResolution
from .installer import InstallResult
from .main import RescueResult

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

from .checker import PipStatus
from .installer import InstallStrategy, InstallResultCode
from .main import ExitCode, PipelineStage
from .fetcher import FetchStatus
from .versions import VersionSource, CompatibilityStatus

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

from .versions import (
    compare_versions,
    parse_version,
    normalize_version,
    get_compatible_versions_for_python,
)


# ---------------------------------------------------------------------------
# Package-Level Convenience Functions
# ---------------------------------------------------------------------------


def repair(
    target_version: Optional[str] = None,
    python_executable: Optional[Path] = None,
    wheel_path: Optional[Path] = None,
    get_pip_path: Optional[Path] = None,
    user_site: bool = False,
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> RescueResult:
    """
    Convenience function for a one-line pip repair operation.

    Why this function exists:
        - Most users just want to run the pipeline with minimal
          configuration. This function provides sensible defaults
          while still allowing overrides.
        - Avoids the need to instantiate ``PipRescuer`` for simple
          use cases.
        - The function signature serves as documentation for the
          most commonly used parameters.

    Parameters
    ----------
    target_version : Optional[str]
        Specific pip version to install. ``None`` for latest compatible.
    python_executable : Optional[Path]
        Target Python interpreter. ``None`` for current Python.
    wheel_path : Optional[Path]
        Local pip wheel for offline installation.
    get_pip_path : Optional[Path]
        Local get-pip.py script.
    user_site : bool
        Install to user site-packages.
    force : bool
        Reinstall even if pip is healthy.
    dry_run : bool
        Preview actions without making changes.
    verbose : bool
        Print detailed progress.

    Returns
    -------
    RescueResult
        Complete result of the repair operation.

    Examples
    --------
    >>> from  import repair
    >>> result = repair()
    >>> result.success
    True

    >>> result = repair(target_version="21.3.1", force=True)
    >>> result.version_installed
    '21.3.1'

    >>> result = repair(
    ...     wheel_path=Path("./pip-23.0.1-py3-none-any.whl"),
    ...     verbose=True,
    ... )
    """
    rescuer = PipRescuer(
        target_version=target_version,
        python_executable=python_executable,
        wheel_path=wheel_path,
        get_pip_path=get_pip_path,
        user_site=user_site,
        force=force,
        dry_run=dry_run,
        verbose=verbose,
    )
    return rescuer.run()


def diagnose(
    python_executable: Optional[Path] = None,
) -> PipDiagnosis:
    """
    Convenience function for a one-line pip diagnosis.

    Why this function exists:
        - Users often want to check pip's state before deciding
          whether to repair it.
        - Provides a simpler interface than instantiating and
          configuring ``PipChecker`` manually.

    Parameters
    ----------
    python_executable : Optional[Path]
        Target Python interpreter. ``None`` for current Python.

    Returns
    -------
    PipDiagnosis
        Complete diagnosis of pip's state and environment.

    Examples
    --------
    >>> from  import diagnose
    >>> diagnosis = diagnose()
    >>> diagnosis.status.name
    'HEALTHY'
    >>> diagnosis.current_version
    '24.3.1'
    >>> diagnosis.environment.is_virtualenv
    True

    >>> diagnosis = diagnose(Path("/usr/bin/python3.8"))
    >>> if diagnosis.status == PipStatus.MISSING:
    ...     print("pip is not installed for Python 3.8")
    """
    checker = PipChecker(python_executable=python_executable)
    return checker.run_full_diagnosis()


def check_compatibility(
    pip_version: str,
    python_version: Optional[Tuple[int, ...]] = None,
) -> Tuple[bool, str]:
    """
    Convenience function to check if a pip version is compatible.

    Why this function exists:
        - Quick compatibility check without instantiating the full
          resolver.
        - Common use case: "Will pip X work on Python Y?"

    Parameters
    ----------
    pip_version : str
        Pip version to check (e.g., "24.0").
    python_version : Optional[Tuple[int, ...]]
        Python version tuple. ``None`` for current Python.

    Returns
    -------
    compatible : bool
        ``True`` if the pip version supports the Python version.
    reason : str
        Explanation of the compatibility determination.

    Examples
    --------
    >>> from  import check_compatibility
    >>> compatible, reason = check_compatibility("24.0", (3, 6, 8))
    >>> compatible
    False
    >>> print(reason)
    pip>=24.0 requires Python>=3.8; current Python is 3.6

    >>> compatible, reason = check_compatibility("21.3.1", (3, 9, 18))
    >>> compatible
    True
    """
    resolver = VersionResolver(python_version=python_version)
    return resolver.is_compatible(pip_version)

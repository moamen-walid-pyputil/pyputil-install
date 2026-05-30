"""
Pip State Inspection & Diagnosis Module.

Examines the Python environment to determine pip's presence, integrity,
version, and the characteristics of the host Python installation. The
diagnosis produced by this module directly determines which installation
strategy the orchestrator selects.

Why this module exists:
    - A broken pip requires force-reinstall, not a fresh install.
    - A missing pip requires bootstrapping from scratch.
    - An outdated pip on an old Python requires a specific compatible version.
    - A pip that exists but fails to import needs zipimport-based recovery.
    - Permission constraints require user-site installation paths.
    - Each diagnosis maps to a different strategy; guessing wastes time
      and risks leaving the environment in an inconsistent state.

Warnings
--------
- This module executes subprocess calls to inspect pip. If pip is severely
  corrupted, these calls may hang. All subprocess calls use a 30-second
  timeout to prevent indefinite blocking.
- On some corporate Windows environments, antivirus software may intercept
  Python subprocess calls and cause false negatives. Run with ``--verbose``
  to inspect raw outputs when results seem inconsistent.
- Modifying ``sys.path`` or ``os.environ`` while using this checker may
  produce misleading diagnoses. Take an environment snapshot before making
  changes elsewhere.

Examples
--------
Quick diagnosis of the current environment:

    >>> from checker import PipChecker
    >>> checker = PipChecker()
    >>> diagnosis = checker.run_full_diagnosis()
    >>> print(diagnosis.status.name)
    'BROKEN'
    >>> print(diagnosis.import_error)
    'No module named pip._vendor.requests'

Diagnose a specific Python installation:

    >>> checker = PipChecker(python_executable=Path("/usr/bin/python3.8"))
    >>> diagnosis = checker.run_full_diagnosis()

Check only whether pip exists without full environment analysis:

    >>> checker = PipChecker()
    >>> exists, path = checker.locate_pip_module()
    >>> if exists:
    ...     print(f"pip found at {path}")

Get environment information without pip checks:

    >>> checker = PipChecker()
    >>> env = checker.inspect_environment()
    >>> print(env.python_version)
    '3.9.18'
    >>> print(env.is_virtualenv)
    True
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Public Enumerations
# ---------------------------------------------------------------------------


class PipStatus(Enum):
    """
    Enumerates all possible pip states that influence repair strategy selection.

    Each status maps to a different installation approach. The order of
    declaration does not imply severity; each is handled distinctly by the
    orchestrator in ``main.py``.

    Why enum instead of strings:
        - Prevents typos in status comparison across modules (e.g., 'brokne'
          would silently fail with strings).
        - Enables exhaustive pattern matching in Python 3.10+.
        - Each variant carries explicit, searchable meaning in the codebase.

    Attributes
    ----------
    HEALTHY : PipStatus
        pip is installed, importable, and fully functional.
    MISSING : PipStatus
        No pip installation detected at any expected path on the system.
    BROKEN : PipStatus
        pip files exist on disk but fail to import or execute. This covers
        corrupted wheels, missing vendored dependencies, incompatible
        bytecode caches, and partial uninstalls left by system package managers.
    OUTDATED : PipStatus
        pip exists and functions but its version is older than what the user
        requires or what the current Python version optimally supports.
    BLOCKED : PipStatus
        pip exists but the operating system prevents its execution due to
        permission errors, mandatory access controls (SELinux/AppArmor),
        or antivirus file locks.
    UNKNOWN : PipStatus
        Diagnosis could not complete; the environment could not be reliably
        assessed. This status forces the orchestrator into fallback mode.

    Examples
    --------
    >>> status = PipStatus.BROKEN
    >>> status.name
    'BROKEN'
    >>> status.value
    3
    >>> PipStatus(3)
    <PipStatus.BROKEN: 3>
    """

    HEALTHY = auto()
    """pip is installed, importable, and fully functional."""

    MISSING = auto()
    """No pip installation detected at any expected path."""

    BROKEN = auto()
    """
    pip files exist but fail to import or execute.

    Common causes:
        - Corrupted ``pip/_vendor/`` directory
        - Missing ``pip/_vendor/requests`` after system package manager removal
        - Stale ``.pyc`` bytecode from a different Python version
        - Partial uninstall where ``pip/__init__.py`` exists but
          ``pip/_internal/`` does not
    """

    OUTDATED = auto()
    """
    pip exists and works but its version predates what the user needs
    or what the current Python version optimally supports.
    """

    BLOCKED = auto()
    """
    pip exists but cannot execute due to external restrictions.

    Common causes:
        - ``PermissionError`` when accessing pip's script directory
        - SELinux ``denied`` audit entries for Python process
        - Windows Defender Controlled Folder Access blocking writes
        - Filesystem mounted with ``noexec`` flag
    """

    UNKNOWN = auto()
    """Diagnosis failed entirely; environment could not be assessed."""


class EnvironmentOrigin(Enum):
    """
    Identifies the type of Python environment in use.

    Why distinguish environment types:
        - Conda environments should use ``conda install pip``, not get-pip.py.
        - stdlib venvs have ``ensurepip`` always available.
        - System Python on Debian/Ubuntu may require ``--break-system-packages``.
        - virtualenv (pre-3.3) may store pip in a non-standard location.

    Examples
    --------
    >>> origin = EnvironmentOrigin.VENV
    >>> origin.name
    'VENV'
    """

    SYSTEM = auto()
    """Operating-system-managed Python (e.g., /usr/bin/python3 on Linux)."""

    VENV = auto()
    """Python standard library venv (created with ``python -m venv``)."""

    VIRTUALENV = auto()
    """Third-party virtualenv (created with ``virtualenv`` command)."""

    CONDA = auto()
    """Conda-managed environment (base or named)."""

    PYENV = auto()
    """pyenv-managed Python installation."""

    DOCKER = auto()
    """Running inside a Docker container (typically system Python)."""

    UNKNOWN = auto()
    """Environment origin could not be determined."""


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------


@dataclass
class EnvironmentInfo:
    """
    Structured snapshot of Python environment properties that affect
    pip installation decisions.

    Why a dataclass instead of a dictionary:
        - Type hints enable IDE autocompletion for all fields, reducing
          the need to consult documentation during development.
        - Fields are explicitly declared; consumers never need to check
          ``"is_virtualenv" in env_dict``.
        - Enables pattern matching on specific field combinations in
          Python 3.10+ match statements.
        - Dataclass fields have a defined order for serialization.

    Attributes
    ----------
    python_version : str
        Full Python version string as reported by ``sys.version``.
        Example: ``'3.9.18 (main, Aug 24 2023, 15:18:16) [GCC 11.4.0]'``.
    python_version_tuple : Tuple[int, int, int]
        Parsed version components as ``(major, minor, micro)``.
        Example: ``(3, 9, 18)``.
    python_executable : Path
        Absolute path to the Python interpreter binary being used.
        Example: ``Path('/usr/bin/python3')``.
    python_implementation : str
        Which Python implementation is running.
        Example: ``'CPython'``, ``'PyPy'``, ``'Jython'``.
    python_implementation_version : str
        Version string specific to the implementation.
        Example: ``'7.3.13'`` for PyPy.
    is_virtualenv : bool
        ``True`` if running inside any kind of virtual environment
        (venv, virtualenv, conda env, or pyenv).
    environment_origin : EnvironmentOrigin
        The specific type of environment detected.
    is_conda : bool
        ``True`` if this Python is managed by Conda (derived from
        ``environment_origin`` for convenience).
    is_user_site_enabled : bool
        ``True`` if ``pip install --user`` can write to a user site-packages
        directory. ``False`` inside some containerized or restricted setups.
    site_packages_paths : List[Path]
        All directories where Python searches for installed packages.
        Obtained from ``site.getsitepackages()``.
    user_site_packages : Optional[Path]
        The user-specific site-packages directory if it exists.
        Obtained from ``site.getusersitepackages()``.
    has_internet : bool
        ``True`` if a basic internet connectivity check succeeded.
        Uses multiple fallback hosts to avoid single-point-of-failure.
    os_name : str
        Operating system identifier. One of ``'linux'``, ``'darwin'``,
        ``'windows'``, or ``sys.platform`` if unrecognized.
    os_version : str
        Human-readable OS version string when available.
    is_admin : bool
        ``True`` if the current process has elevated privileges
        (root on Unix, Administrator on Windows).
    env_vars : Dict[str, str]
        Snapshot of environment variables relevant to pip behavior.
        Includes ``PIP_REQUIRE_VIRTUALENV``, ``PIP_USER``,
        ``PIP_BREAK_SYSTEM_PACKAGES``, ``PIP_TARGET``, and ``PIP_PREFIX``.
    disk_free_mb : float
        Free disk space in megabytes on the filesystem containing
        ``site_packages_paths[0]``. Installation requires approximately
        10-20 MB for pip and its dependencies.

    Examples
    --------
    >>> info = EnvironmentInfo(
    ...     python_version="3.9.18",
    ...     python_version_tuple=(3, 9, 18),
    ...     python_executable=Path("/usr/bin/python3"),
    ...     python_implementation="CPython",
    ...     python_implementation_version="3.9.18",
    ...     is_virtualenv=False,
    ...     environment_origin=EnvironmentOrigin.SYSTEM,
    ...     is_conda=False,
    ...     is_user_site_enabled=True,
    ...     site_packages_paths=[Path("/usr/lib/python3.9/site-packages")],
    ...     user_site_packages=Path("/home/user/.local/lib/python3.9/site-packages"),
    ...     has_internet=True,
    ...     os_name="linux",
    ...     os_version="Ubuntu 22.04.3 LTS",
    ...     is_admin=False,
    ...     env_vars={"PIP_REQUIRE_VIRTUALENV": "false"},
    ...     disk_free_mb=4520.5,
    ... )
    >>> info.is_virtualenv
    False
    >>> info.environment_origin.name
    'SYSTEM'
    """

    python_version: str
    python_version_tuple: Tuple[int, int, int]
    python_executable: Path
    python_implementation: str
    python_implementation_version: str
    is_virtualenv: bool
    environment_origin: EnvironmentOrigin
    is_conda: bool
    is_user_site_enabled: bool
    site_packages_paths: List[Path]
    user_site_packages: Optional[Path]
    has_internet: bool
    os_name: str
    os_version: str
    is_admin: bool
    env_vars: Dict[str, str]
    disk_free_mb: float


@dataclass
class PipDiagnosis:
    """
    Complete diagnosis result produced after inspecting pip and the environment.

    Why a dedicated result type instead of returning multiple values:
        - Separates data collection from decision-making. The diagnosis can
          be logged, serialized, or transmitted before any action is taken.
        - Allows the orchestrator to make conditional decisions based on
          combinations of fields. For example: ``BROKEN + CONDA`` requires
          a different approach than ``BROKEN + VENV``.
        - ``raw_output`` preserves diagnostic subprocess outputs for debugging
          without cluttering the structured fields.

    Attributes
    ----------
    status : PipStatus
        The determined state of pip after all checks complete.
    current_version : Optional[str]
        Currently installed pip version string if determinable.
        Example: ``'23.0.1'``. ``None`` if pip is missing or version
        could not be parsed.
    install_paths : List[Path]
        Paths where pip files were actually found on disk. May contain
        multiple entries if pip is partially installed across directories.
    executable_path : Optional[Path]
        Path to the ``pip`` (or ``pip.exe`` on Windows) executable script.
    module_path : Optional[Path]
        Path to the ``pip/`` Python package directory (containing
        ``__init__.py``).
    import_error : Optional[str]
        The full traceback or error message if ``import pip`` failed.
        ``None`` if pip imported successfully.
    environment : EnvironmentInfo
        Full environment snapshot captured at diagnosis time.
    raw_output : Dict[str, str]
        Raw stdout/stderr from diagnostic subprocess commands. Preserved
        for verbose logging and debugging. Keys include:
        ``'pip_version'``, ``'pip_list'``, ``'pip_check'``,
        ``'pip_debug'``, ``'connectivity_test'``.
    recommended_action : str
        Human-readable suggestion set by the orchestrator after analyzing
        this diagnosis. Not populated by the checker itself.
    install_constraints : List[str]
        CLI flags the orchestrator must pass to the installer.
        Examples: ``['--user']``, ``['--break-system-packages']``.

    Examples
    --------
    >>> diagnosis = PipDiagnosis(
    ...     status=PipStatus.BROKEN,
    ...     current_version="21.0.1",
    ...     install_paths=[Path("/usr/lib/python3/dist-packages/pip")],
    ...     executable_path=Path("/usr/bin/pip"),
    ...     module_path=Path("/usr/lib/python3/dist-packages/pip"),
    ...     import_error="ModuleNotFoundError: No module named 'pip._vendor.requests'",
    ...     environment=some_environment_info,
    ...     raw_output={"pip_version": "pip 21.0.1\\n...", "pip_check": "ERROR..."},
    ... )
    >>> diagnosis.status == PipStatus.BROKEN
    True
    >>> len(diagnosis.install_constraints)
    0
    """

    status: PipStatus
    current_version: Optional[str]
    install_paths: List[Path]
    executable_path: Optional[Path]
    module_path: Optional[Path]
    import_error: Optional[str]
    environment: EnvironmentInfo
    raw_output: Dict[str, str] = field(default_factory=dict)
    recommended_action: str = ""
    install_constraints: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core Checker Class
# ---------------------------------------------------------------------------


class PipChecker:
    """
    Diagnoses pip's presence, health, and the Python environment.

    Why a class instead of standalone functions:
        - Caches environment information; querying ``sysconfig`` and running
          subprocesses is expensive and should not be repeated per check.
        - Holds configuration (timeouts, target executable) that affects
          all diagnostic methods consistently.
        - Enables complete mocking in unit tests by replacing the checker
          instance rather than patching multiple module-level functions.
        - The diagnosis builds incrementally: environment inspection happens
          once, then multiple pip checks consume the cached result.

    .. warning::
       This class executes subprocess calls to the target Python interpreter.
       If the target pip is severely corrupted (e.g., segfaults on import),
       subprocess calls may hang. All calls use a configurable timeout
       (default 30 seconds) to prevent indefinite blocking.

    .. warning::
       On Windows, antivirus real-time scanning can intercept Python
       subprocess creation and cause ``ACCESS_DENIED`` errors. If diagnosis
       returns ``UNKNOWN`` unexpectedly, temporarily disabling on-access
       scanning for the Python directory may help isolate the issue.

    Parameters
    ----------
    python_executable : Optional[Path]
        Path to the Python interpreter to diagnose. If ``None``, uses
        ``sys.executable`` (the currently running Python).
        Why: enables diagnosing a different Python installation than the
        one executing this script (e.g., ``/usr/bin/python3.8`` while
        running under ``/usr/bin/python3.12``).

    Attributes
    ----------
    python_executable : Path
        The Python interpreter being diagnosed.
    timeout : int
        Maximum seconds to wait for any subprocess command.
    _cached_environment : Optional[EnvironmentInfo]
        Lazily populated environment snapshot. Access via
        ``inspect_environment()``.

    Examples
    --------
    Diagnose the current Python environment:

        >>> checker = PipChecker()
        >>> result = checker.run_full_diagnosis()
        >>> print(result.status.name)
        'HEALTHY'
        >>> print(result.current_version)
        '23.2.1'

    Diagnose a specific Python installation:

        >>> checker = PipChecker(python_executable=Path("/opt/python3.10/bin/python"))
        >>> result = checker.run_full_diagnosis()
        >>> if result.status == PipStatus.MISSING:
        ...     print("pip not found, will bootstrap")

    Check only whether pip's module can be imported without full diagnosis:

        >>> checker = PipChecker()
        >>> can_import, error_msg = checker.try_import_pip()
        >>> if not can_import:
        ...     print(f"Cannot import pip: {error_msg}")

    Inspect environment only, skipping pip checks entirely:

        >>> checker = PipChecker()
        >>> env_info = checker.inspect_environment()
        >>> print(f"Python {env_info.python_version} on {env_info.os_name}")
        Python 3.11.5 on linux
    """

    def __init__(self, python_executable: Optional[Path] = None) -> None:
        """
        Initialize a checker targeting a specific Python interpreter.

        Parameters
        ----------
        python_executable : Optional[Path]
            Absolute or relative path to the Python interpreter to diagnose.
            If ``None``, uses ``sys.executable``.

        Raises
        ------
        FileNotFoundError
            If ``python_executable`` is provided but does not exist on disk.

        Examples
        --------
        Default (current interpreter):

            >>> checker = PipChecker()

        Target a specific Python version:

            >>> checker = PipChecker(Path("/usr/bin/python3.9"))
        """
        if python_executable is None:
            self.python_executable = Path(sys.executable).resolve()
        else:
            candidate = Path(python_executable).resolve()
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Python executable not found: {candidate}"
                )
            self.python_executable = candidate

        self.timeout: int = 30
        self._cached_environment: Optional[EnvironmentInfo] = None

    # ------------------------------------------------------------------
    # Public API: Full Diagnosis
    # ------------------------------------------------------------------

    def run_full_diagnosis(self) -> PipDiagnosis:
        """
        Execute all diagnostic checks and return a complete diagnosis.

        This is the primary entry point. It calls each inspection method
        in a deliberate order:

        1. ``inspect_environment()`` — snapshot the Python environment.
        2. ``locate_pip_module()`` — find pip's files on disk.
        3. ``locate_pip_executable()`` — find the pip script.
        4. ``detect_pip_version()`` — determine the installed version.
        5. ``try_import_pip()`` — attempt to import pip.
        6. ``run_pip_check()`` — run ``pip check`` if pip executes.

        The order matters: if pip cannot even be found on disk, there is
        no point attempting to import or execute it. Similarly, if import
        fails, execution attempts are skipped.

        Returns
        -------
        PipDiagnosis
            Complete diagnosis with status, version, paths, errors, and
            environment snapshot.

        Examples
        --------
        >>> checker = PipChecker()
        >>> diagnosis = checker.run_full_diagnosis()
        >>> print(diagnosis.status)
        PipStatus.HEALTHY
        >>> if diagnosis.status == PipStatus.HEALTHY:
        ...     print(f"pip {diagnosis.current_version} is healthy")
        pip 24.0 is healthy
        """
        env_info = self.inspect_environment()
        raw_output: Dict[str, str] = {}

        # Step 1: Check disk presence
        module_exists, module_path, module_paths = self.locate_pip_module()

        # Step 2: Check executable presence
        exe_exists, exe_path = self.locate_pip_executable()

        # Step 3: Try version detection (best-effort, may fail)
        version: Optional[str] = None
        if exe_exists and exe_path is not None:
            version, version_raw = self._detect_pip_version_via_subprocess(exe_path)
            raw_output["pip_version"] = version_raw

        # Step 4: Try import
        import_error: Optional[str] = None
        if module_exists:
            can_import, import_error = self.try_import_pip()

        # Step 5: Try pip check
        check_raw: str = ""
        if exe_exists and import_error is None:
            _, check_raw = self._run_pip_check(exe_path)
            raw_output["pip_check"] = check_raw

        # Determine status
        status = self._classify_status(
            module_exists=module_exists,
            exe_exists=exe_exists,
            import_error=import_error,
            version=version,
            env_info=env_info,
        )

        # Collect install paths
        install_paths: List[Path] = list(module_paths) if module_paths else []
        if module_path:
            install_paths.append(module_path)

        return PipDiagnosis(
            status=status,
            current_version=version,
            install_paths=install_paths,
            executable_path=exe_path,
            module_path=module_path,
            import_error=import_error,
            environment=env_info,
            raw_output=raw_output,
        )

    # ------------------------------------------------------------------
    # Public API: Individual Checks
    # ------------------------------------------------------------------

    def inspect_environment(self) -> EnvironmentInfo:
        """
        Collect comprehensive information about the Python environment.

        This method is called once and its result is cached. Subsequent
        calls return the cached ``EnvironmentInfo`` without re-running
        any subprocesses or filesystem scans.

        Why caching:
            - ``sysconfig.get_paths()`` and ``shutil.disk_usage()`` involve
              filesystem I/O.
            - Connectivity checks make HTTP requests.
            - The environment does not change during a single diagnosis session.
            - Caching prevents redundant work when multiple diagnosis methods
              need environment information.

        Returns
        -------
        EnvironmentInfo
            Complete snapshot of the Python environment.

        Examples
        --------
        >>> checker = PipChecker()
        >>> env = checker.inspect_environment()
        >>> env.python_version
        '3.11.5'
        >>> env.is_conda
        False
        >>> env.has_internet
        True
        >>> env.site_packages_paths
        [Path('/usr/lib/python3.11/site-packages')]
        """
        if self._cached_environment is not None:
            return self._cached_environment

        python_version = sys.version.split()[0] if self._is_current_python() else self._get_remote_python_version()

        version_tuple = self._parse_version_tuple(python_version)

        implementation = platform.python_implementation()
        implementation_version = (
            platform.python_version()
            if self._is_current_python()
            else python_version
        )

        environment_origin = self._detect_environment_origin()
        is_virtualenv = environment_origin not in (
            EnvironmentOrigin.SYSTEM,
            EnvironmentOrigin.DOCKER,
            EnvironmentOrigin.UNKNOWN,
        )
        is_conda = environment_origin == EnvironmentOrigin.CONDA

        site_packages_paths = self._get_site_packages_paths()
        user_site = self._get_user_site_packages()
        is_user_site_enabled = user_site is not None and user_site.exists()

        has_internet = self._check_connectivity()

        os_name = sys.platform
        if os_name.startswith("linux"):
            os_name = "linux"
        elif os_name == "darwin":
            os_name = "darwin"
        elif os_name in ("win32", "cygwin"):
            os_name = "windows"

        os_version = self._get_os_version()

        is_admin = self._check_admin()

        env_vars = self._capture_relevant_env_vars()

        disk_free_mb = self._get_disk_free_mb(site_packages_paths)

        self._cached_environment = EnvironmentInfo(
            python_version=python_version,
            python_version_tuple=version_tuple,
            python_executable=self.python_executable,
            python_implementation=implementation,
            python_implementation_version=implementation_version,
            is_virtualenv=is_virtualenv,
            environment_origin=environment_origin,
            is_conda=is_conda,
            is_user_site_enabled=is_user_site_enabled,
            site_packages_paths=site_packages_paths,
            user_site_packages=user_site,
            has_internet=has_internet,
            os_name=os_name,
            os_version=os_version,
            is_admin=is_admin,
            env_vars=env_vars,
            disk_free_mb=disk_free_mb,
        )

        return self._cached_environment

    def locate_pip_module(self) -> Tuple[bool, Optional[Path], List[Path]]:
        """
        Search for pip's Python module directory on disk.

        Searches all directories in ``sys.path`` (or the target Python's
        equivalent) for a directory named ``pip`` containing an
        ``__init__.py`` file.

        Why search all paths instead of only site-packages:
            - ``pip`` could be installed in a custom ``--target`` directory.
            - ``PYTHONPATH`` may include pip locations unknown to site.
            - System Python on Debian/Ubuntu uses ``dist-packages``, not
              ``site-packages``.
            - A partial uninstall may leave pip in one path but not another.

        Returns
        -------
        exists : bool
            ``True`` if at least one ``pip/__init__.py`` was found.
        primary_path : Optional[Path]
            The first valid ``pip`` module directory found (highest priority
            in ``sys.path`` order).
        all_paths : List[Path]
            All directories named ``pip`` that contain ``__init__.py``,
            including those that may be partial or orphaned installations.

        Examples
        --------
        >>> checker = PipChecker()
        >>> exists, primary, all_paths = checker.locate_pip_module()
        >>> exists
        True
        >>> primary
        Path('/usr/lib/python3/dist-packages/pip')
        >>> len(all_paths)
        1
        """
        search_paths = self._get_search_paths()
        all_found: List[Path] = []

        for search_dir in search_paths:
            candidate = search_dir / "pip"
            init_file = candidate / "__init__.py"
            if init_file.exists() and init_file.is_file():
                all_found.append(candidate)

        exists = len(all_found) > 0
        primary = all_found[0] if exists else None

        return exists, primary, all_found

    def locate_pip_executable(self) -> Tuple[bool, Optional[Path]]:
        """
        Find the pip executable script on the filesystem.

        Searches in this order:
        1. The ``bin`` directory adjacent to the Python executable (or
           ``Scripts`` on Windows).
        2. The directories listed in the ``PATH`` environment variable.
        3. Well-known system paths (``/usr/local/bin``, etc.).

        Why not rely solely on ``shutil.which``:
            - ``shutil.which`` respects the current ``PATH``, which may not
              include the Python's own ``bin`` directory (common in pyenv
              and virtualenv setups where activation was skipped).
            - On Windows, ``pip`` may exist as ``pip.exe`` in ``Scripts``
              but that directory may not be on ``PATH``.

        Returns
        -------
        exists : bool
            ``True`` if a pip executable file was located.
        path : Optional[Path]
            Absolute path to the pip executable, or ``None`` if not found.

        Examples
        --------
        >>> checker = PipChecker()
        >>> exists, path = checker.locate_pip_executable()
        >>> exists
        True
        >>> path.name
        'pip'
        """
        # Priority 1: Adjacent to the Python executable
        python_dir = self.python_executable.parent
        script_dir_name = "Scripts" if sys.platform == "win32" else "bin"
        adjacent_bin = python_dir / script_dir_name
        pip_name = "pip.exe" if sys.platform == "win32" else "pip"

        candidate = adjacent_bin / pip_name
        if candidate.exists() and candidate.is_file():
            return True, candidate

        # Priority 2: shutil.which using the PATH from the target environment
        found = shutil.which(pip_name)
        if found:
            return True, Path(found)

        # Priority 3: Brute-force search in common locations
        common_locations: List[Path] = [
            Path("/usr/local/bin"),
            Path("/usr/bin"),
            Path("/opt/homebrew/bin"),
            Path.home() / ".local" / "bin",
        ]
        for loc in common_locations:
            candidate = loc / pip_name
            if candidate.exists() and candidate.is_file():
                return True, candidate

        return False, None

    def detect_pip_version(self) -> Tuple[Optional[str], str]:
        """
        Determine the installed pip version using the best available method.

        Tries in order:
        1. ``pip --version`` via subprocess (fastest, most reliable).
        2. ``python -m pip --version`` via subprocess.
        3. Parsing ``pip/__init__.py`` for ``__version__`` (fallback if
           pip exists on disk but cannot execute).

        Why multiple methods:
            - ``pip --version`` fails if the script is broken but the
              module is intact.
            - ``python -m pip`` may work even when the script is missing.
            - Parsing ``__init__.py`` works even if pip's dependencies
              are corrupted, as long as the version string is readable.

        Returns
        -------
        version : Optional[str]
            Version string like ``'23.0.1'``, or ``None`` if undetermined.
        raw_output : str
            Complete stdout/stderr from the version detection attempt,
            preserved for debugging.

        Examples
        --------
        >>> checker = PipChecker()
        >>> version, raw = checker.detect_pip_version()
        >>> version
        '24.0'
        >>> print(raw)
        pip 24.0 from /usr/lib/python3.12/site-packages/pip (python 3.12)
        """
        # Method 1: pip --version
        exe_exists, exe_path = self.locate_pip_executable()
        if exe_exists and exe_path is not None:
            version, raw = self._detect_pip_version_via_subprocess(exe_path)
            if version:
                return version, raw

        # Method 2: python -m pip --version
        try:
            result = subprocess.run(
                [str(self.python_executable), "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            output = result.stdout + result.stderr
            version = self._parse_version_from_output(output)
            if version:
                return version, output
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        # Method 3: Parse pip/__init__.py directly
        _, module_path, _ = self.locate_pip_module()
        if module_path:
            init_file = module_path / "__init__.py"
            if init_file.exists():
                version = self._parse_version_from_init_file(init_file)
                if version:
                    return version, f"Parsed from {init_file}"

        return None, ""

    def try_import_pip(self) -> Tuple[bool, Optional[str]]:
        """
        Attempt to import pip and capture any resulting error.

        Why attempt import even when files exist on disk:
            - A directory named ``pip`` with ``__init__.py`` may still fail
              to import due to missing vendored dependencies, syntax errors
              in the code, or incompatible bytecode.
            - The specific error message (``ModuleNotFoundError`` vs
              ``ImportError`` vs ``SyntaxError``) guides the repair strategy.
            - A successful import confirms pip is usable at the Python level,
              which is necessary for ``python -m pip`` operations.

        Returns
        -------
        success : bool
            ``True`` if ``import pip`` completed without raising an exception.
        error_message : Optional[str]
            The full traceback as a string if import failed, or ``None``
            if import succeeded.

        Examples
        --------
        >>> checker = PipChecker()
        >>> ok, error = checker.try_import_pip()
        >>> ok
        False
        >>> print(error)
        ModuleNotFoundError: No module named 'pip._vendor.urllib3'
        """
        # Use a subprocess to avoid polluting the current process's sys.modules
        import_script = """
import sys
try:
    import pip
    print("OK:" + pip.__version__)
except Exception as e:
    print("ERROR:" + repr(e))
    print("TRACEBACK_START")
    import traceback
    traceback.print_exc()
    print("TRACEBACK_END")
"""
        try:
            result = subprocess.run(
                [str(self.python_executable), "-c", import_script],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            combined = result.stdout + result.stderr

            if combined.startswith("OK:"):
                return True, None
            else:
                # Extract the error portion
                if "TRACEBACK_START" in combined and "TRACEBACK_END" in combined:
                    start = combined.index("TRACEBACK_START") + len("TRACEBACK_START")
                    end = combined.index("TRACEBACK_END")
                    error_msg = combined[start:end].strip()
                else:
                    error_msg = combined.strip()
                return False, error_msg if error_msg else "Unknown import error"

        except subprocess.TimeoutExpired:
            return False, "Import timed out (pip may be deadlocked or corrupted)"
        except Exception as e:
            return False, f"Failed to run import check: {e}"

    def run_pip_check(self) -> Tuple[bool, str]:
        """
        Execute ``pip check`` to verify installed package integrity.

        ``pip check`` validates that all installed packages have their
        dependencies satisfied. A failure here indicates broken
        dependencies, which may affect pip itself if its vendored
        packages are externally installed.

        Why run this:
            - Some distributions (Debian, Ubuntu) unbundle pip's vendored
              dependencies and install them as system packages. If those
              are removed, ``pip check`` reveals the missing dependencies.
            - A successful ``pip check`` is a strong signal that pip is
              fully functional.

        Returns
        -------
        success : bool
            ``True`` if ``pip check`` exited with code 0.
        raw_output : str
            Complete stdout and stderr from the check command.

        Examples
        --------
        >>> checker = PipChecker()
        >>> ok, output = checker.run_pip_check()
        >>> ok
        True
        >>> print(output)
        No broken requirements found.
        """
        exe_exists, exe_path = self.locate_pip_executable()
        if not exe_exists or exe_path is None:
            return False, "pip executable not found"

        return self._run_pip_check(exe_path)

    # ------------------------------------------------------------------
    # Private: Status Classification
    # ------------------------------------------------------------------

    def _classify_status(
        self,
        module_exists: bool,
        exe_exists: bool,
        import_error: Optional[str],
        version: Optional[str],
        env_info: EnvironmentInfo,
    ) -> PipStatus:
        """Determine PipStatus from collected signals."""
        # Neither module nor executable found
        if not module_exists and not exe_exists:
            return PipStatus.MISSING

        # Files exist but import fails
        if module_exists and import_error is not None:
            return PipStatus.BROKEN

        # Executable exists but module does not (orphaned script)
        if exe_exists and not module_exists:
            return PipStatus.BROKEN

        # Both exist and import works
        if module_exists and import_error is None and version is not None:
            return PipStatus.HEALTHY

        # Could not determine — assume the worst but don't claim MISSING
        return PipStatus.UNKNOWN

    # ------------------------------------------------------------------
    # Private: Subprocess Helpers
    # ------------------------------------------------------------------

    def _run_pip_check(self, exe_path: Path) -> Tuple[bool, str]:
        """Run pip check via subprocess. Returns (success, output)."""
        try:
            result = subprocess.run(
                [str(exe_path), "check"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            output = result.stdout + result.stderr
            return result.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, "pip check timed out"
        except Exception as e:
            return False, str(e)

    def _detect_pip_version_via_subprocess(
        self, exe_path: Path
    ) -> Tuple[Optional[str], str]:
        """Run pip --version and parse output. Returns (version, raw)."""
        try:
            result = subprocess.run(
                [str(exe_path), "--version"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            output = result.stdout + result.stderr
            return self._parse_version_from_output(output), output
        except subprocess.TimeoutExpired:
            return None, "pip --version timed out"
        except Exception as e:
            return None, str(e)

    @staticmethod
    def _parse_version_from_output(output: str) -> Optional[str]:
        """
        Parse version string from pip --version output.

        Expected format: "pip X.Y.Z from /path (python A.B.C)"
        """
        import re

        match = re.match(r"pip\s+(\d+\.\d+(?:\.\d+)?)", output)
        return match.group(1) if match else None

    @staticmethod
    def _parse_version_from_init_file(init_path: Path) -> Optional[str]:
        """Extract __version__ string from pip/__init__.py."""
        try:
            content = init_path.read_text(encoding="utf-8", errors="replace")
            for line in content.splitlines():
                if line.strip().startswith("__version__"):
                    import re

                    match = re.search(r"['\"]([^'\"]+)['\"]", line)
                    if match:
                        return match.group(1)
        except (OSError, UnicodeDecodeError):
            pass
        return None

    @staticmethod
    def _parse_version_tuple(version_str: str) -> Tuple[int, int, int]:
        """Parse '3.9.18' into (3, 9, 18)."""
        parts = version_str.split(".")
        try:
            return (
                int(parts[0]),
                int(parts[1]) if len(parts) > 1 else 0,
                int(parts[2]) if len(parts) > 2 else 0,
            )
        except (ValueError, IndexError):
            return (0, 0, 0)

    # ------------------------------------------------------------------
    # Private: Environment Detection
    # ------------------------------------------------------------------

    def _is_current_python(self) -> bool:
        """Check if we're diagnosing the currently running Python."""
        return self.python_executable.resolve() == Path(sys.executable).resolve()

    def _get_remote_python_version(self) -> str:
        """Get version string from a different Python executable."""
        try:
            result = subprocess.run(
                [str(self.python_executable), "-c", "import sys; print(sys.version.split()[0])"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return result.stdout.strip() or "0.0.0"
        except Exception:
            return "0.0.0"

    def _detect_environment_origin(self) -> EnvironmentOrigin:
        """Determine the specific type of Python environment."""
        # Check Conda first (most distinctive)
        if "CONDA_PREFIX" in os.environ or "CONDA_DEFAULT_ENV" in os.environ:
            return EnvironmentOrigin.CONDA
        if (Path(sys.prefix) / "conda-meta").exists():
            return EnvironmentOrigin.CONDA

        # Check stdlib venv
        if sys.prefix != sys.base_prefix:
            if (Path(sys.prefix) / "pyvenv.cfg").exists():
                return EnvironmentOrigin.VENV

        # Check virtualenv (no pyvenv.cfg but different prefix)
        if hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        ):
            return EnvironmentOrigin.VIRTUALENV

        # Check pyenv
        if "PYENV_ROOT" in os.environ or "PYENV_VERSION" in os.environ:
            return EnvironmentOrigin.PYENV

        # Check Docker
        if Path("/.dockerenv").exists() or "DOCKER" in os.environ.get("container", ""):
            return EnvironmentOrigin.DOCKER

        return EnvironmentOrigin.SYSTEM

    def _get_site_packages_paths(self) -> List[Path]:
        """Get site-packages paths for the target Python."""
        try:
            import site

            return [Path(p) for p in site.getsitepackages()]
        except Exception:
            # Fallback: sysconfig
            try:
                purelib = sysconfig.get_path("purelib")
                platlib = sysconfig.get_path("platlib")
                paths = [Path(purelib)]
                if platlib != purelib:
                    paths.append(Path(platlib))
                return paths
            except Exception:
                return [Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"]

    def _get_user_site_packages(self) -> Optional[Path]:
        """Get user site-packages path."""
        try:
            import site

            return Path(site.getusersitepackages())
        except Exception:
            return None

    def _get_search_paths(self) -> List[Path]:
        """Get all paths where pip module could reside."""
        paths: List[Path] = []
        try:
            # Add sys.path locations
            for p in sys.path:
                if p:
                    paths.append(Path(p))
        except Exception:
            pass

        # Add site-packages explicitly
        paths.extend(self._get_site_packages_paths())

        # Add user site
        user_site = self._get_user_site_packages()
        if user_site:
            paths.append(user_site)

        # Deduplicate while preserving order
        seen: set = set()
        unique: List[Path] = []
        for p in paths:
            resolved = p.resolve() if p.exists() else p
            if resolved not in seen:
                seen.add(resolved)
                unique.append(p)

        return unique

    # ------------------------------------------------------------------
    # Private: System Checks
    # ------------------------------------------------------------------

    def _check_connectivity(self) -> bool:
        """
        Test internet connectivity by attempting to reach multiple hosts.

        Why multiple hosts:
            - A single host may be blocked by corporate firewall.
            - PyPI may be down while GitHub is up (or vice versa).
            - Some networks block raw IP connections but allow DNS.
            - Testing multiple hosts reduces false negatives.

        Returns
        -------
        bool
            ``True`` if at least one host could be reached.
        """
        hosts = [
            ("pypi.org", 443),
            ("bootstrap.pypa.io", 443),
            ("github.com", 443),
            ("google.com", 443),
        ]

        import socket

        for host, port in hosts:
            try:
                sock = socket.create_connection(
                    (host, port), timeout=5
                )
                sock.close()
                return True
            except (socket.timeout, socket.error, OSError):
                continue

        return False

    def _check_admin(self) -> bool:
        """Detect elevated privileges."""
        if sys.platform == "win32":
            try:
                import ctypes

                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            except Exception:
                return False
        else:
            # Unix: check if effective UID is 0 (root)
            return os.geteuid() == 0

    def _capture_relevant_env_vars(self) -> Dict[str, str]:
        """Capture environment variables that affect pip behavior."""
        relevant_keys = [
            "PIP_REQUIRE_VIRTUALENV",
            "PIP_USER",
            "PIP_BREAK_SYSTEM_PACKAGES",
            "PIP_TARGET",
            "PIP_PREFIX",
            "PIP_INDEX_URL",
            "PIP_NO_INDEX",
            "PIP_TRUSTED_HOST",
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "PYTHONPATH",
        ]
        return {k: os.environ[k] for k in relevant_keys if k in os.environ}

    def _get_os_version(self) -> str:
        """Get human-readable OS version."""
        try:
            return f"{platform.system()} {platform.release()} {platform.version()}".strip()
        except Exception:
            return "Unknown"

    def _get_disk_free_mb(self, site_paths: List[Path]) -> float:
        """
        Check free disk space in MB on the filesystem of site-packages.

        Why this matters:
            - pip installation requires writing ~10-20 MB of files.
            - A full disk causes cryptic write errors mid-installation.
            - Warning the user early prevents partial installations.
        """
        if not site_paths:
            return 0.0

        target = site_paths[0]
        if not target.exists():
            target = target.parent
            while not target.exists() and target != target.root:
                target = target.parent

        try:
            usage = shutil.disk_usage(target)
            return usage.free / (1024 * 1024)
        except OSError:
            return 0.0
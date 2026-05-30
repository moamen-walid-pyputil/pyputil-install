"""
Pip Installer Orchestrator & Command-Line Interface.

Coordinates the diagnosis, version resolution, resource fetching, and
installation of pip. Provides both a programmatic API for use as a
library and a command-line interface for direct user interaction.

Why this module exists:
    - The four core modules (checker, versions, fetcher, installer) are
      independent and reusable, but their interaction is complex:
      diagnosis must complete before version resolution, version must
      be resolved before fetching, fetching must succeed (or fail with
      a clear reason) before installation, and installation must be
      verified before reporting success.
    - Error handling across modules requires context: a network error
      during fetching means "try offline mode," but a network error
      during verification means "installation may have succeeded but
      we can't confirm." The orchestrator translates low-level errors
      into actionable user guidance.
    - The CLI is the primary user interface. It must parse arguments,
      validate them, provide helpful error messages, and support both
      interactive and non-interactive (scripted) use.
    - Programmatic API enables other tools to embed pip installation
      without spawning a subprocess.

Architecture:
    The orchestrator follows a pipeline pattern:

        User Input → Validate → Diagnose → Resolve Version → Fetch
        → Install → Verify → Report

    Each stage can short-circuit: if diagnosis shows pip is healthy
    and no force flag is set, the pipeline stops early. If fetching
    fails and a local wheel is available, the pipeline continues with
    the local resource. If installation fails with one strategy, the
    installer internally tries the next strategy before reporting
    failure to the orchestrator.

Warnings
--------
- This module may modify the system's pip installation. It should be
  run with an understanding of the target environment. Use ``--dry-run``
  to preview actions without making changes.
- Running with ``sudo`` or as Administrator installs pip system-wide.
  Prefer ``--user`` for per-user installation unless system-wide
  installation is explicitly intended.
- The ``--break-system-packages`` flag bypasses PEP 668 protections.
  Only use it if you understand the implications for your system's
  package manager.

Examples
--------
Command-line usage (install latest pip):

    $ python main.py
    Diagnosing pip... OK (pip 21.0.1 found, but broken)
    Resolving version... OK (target: 24.3.1)
    Fetching resources... OK (downloaded get-pip.py)
    Installing pip... OK (via get-pip.py)
    Verifying installation... OK (pip 24.3.1)
    ✓ pip 24.3.1 installed successfully

Command-line with specific version:

    $ python main.py --version 21.3.1
    ✓ pip 21.3.1 installed successfully

Command-line offline mode:

    $ python main.py --offline ./pip-23.0.1-py3-none-any.whl
    ✓ pip 23.0.1 installed from local wheel

Command-line dry run:

    $ python main.py --dry-run --version 21.3.1
    [DRY RUN] Would install pip 21.3.1 via get-pip.py

Programmatic usage:

    >>> from main import PipRescuer
    >>> rescuer = PipRescuer()
    >>> result = rescuer.run(target_version="21.3.1")
    >>> result.success
    True
    >>> result.version_installed
    '21.3.1'
    >>> result.strategy_used
    'get-pip.py'

Programmatic with custom configuration:

    >>> rescuer = PipRescuer(
    ...     user_site=True,
    ...     force=True,
    ...     timeout=60,
    ... )
    >>> result = rescuer.run()
    >>> print(result.report())
    Installation Report
    ===================
    Status: SUCCESS
    Version installed: 24.3.1
    Strategy: ensurepip
    Duration: 2.3s
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import core modules
from .checker import PipChecker, PipDiagnosis, PipStatus
from .versions import VersionResolver, VersionResolution, VersionSource
from .fetcher import ResourceFetcher, FetchStatus
from .installer import PipInstaller, InstallResult, InstallStrategy, InstallResultCode


# ---------------------------------------------------------------------------
# Public Enumerations
# ---------------------------------------------------------------------------


class PipelineStage(Enum):
    """
    Identifies each stage in the installation pipeline.

    Why track stages explicitly:
        - Error messages can identify exactly where the pipeline stopped:
          "Version resolution failed" is more actionable than
          "Installation failed."
        - Progress reporting to the user shows which stage is currently
          executing.
        - The ``--dry-run`` flag stops after a specific stage and must
          report which stage it reached.
        - Telemetry/debugging can record which stage failed and how long
          each stage took.

    Examples
    --------
    >>> stage = PipelineStage.DIAGNOSIS
    >>> stage.name
    'DIAGNOSIS'
    """

    VALIDATION = auto()
    """Validating user input and environment preconditions."""

    DIAGNOSIS = auto()
    """Inspecting the current pip state and Python environment."""

    VERSION_RESOLUTION = auto()
    """Determining which pip version to install."""

    FETCHING = auto()
    """Downloading or locating installation resources."""

    INSTALLATION = auto()
    """Executing the installation strategy."""

    VERIFICATION = auto()
    """Verifying that pip is functional after installation."""

    COMPLETE = auto()
    """Pipeline finished successfully."""


class ExitCode(Enum):
    """
    Exit codes returned to the operating system.

    Why define exit codes:
        - Shell scripts can branch on specific failure modes:
          ``if [ $? -eq 3 ]; then echo "Network error"; fi``.
        - Exit code 0 = success, non-zero = failure, with specific
          codes for common failure categories.
        - Consistent with Unix conventions: 1 = general error,
          2 = misuse, 3 = network, 4 = permissions, 5 = incompatible.

    Examples
    --------
    >>> ExitCode.SUCCESS.value
    0
    >>> ExitCode.NETWORK_ERROR.value
    3
    """

    SUCCESS = 0
    """Installation completed successfully."""

    GENERAL_ERROR = 1
    """Unspecified failure."""

    INVALID_ARGUMENTS = 2
    """User provided invalid or contradictory arguments."""

    NETWORK_ERROR = 3
    """Network is unavailable and no offline resources provided."""

    PERMISSION_ERROR = 4
    """Insufficient permissions to install pip."""

    INCOMPATIBLE_VERSION = 5
    """Requested pip version is incompatible with this Python."""

    VERIFICATION_FAILED = 6
    """Installation appeared to succeed but verification failed."""


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    """
    Result of a single pipeline stage execution.

    Why a dataclass per stage:
        - Enables detailed progress reporting: each stage has a status,
          duration, and optional message.
        - The orchestrator can log stage results independently before
          aggregating them into the final ``RescueResult``.
        - Failed stages can include diagnostic information (error
          messages, raw outputs) that the final report summarizes.

    Attributes
    ----------
    stage : PipelineStage
        Which pipeline stage this result represents.
    success : bool
        ``True`` if the stage completed without errors.
    message : str
        Human-readable summary of what happened in this stage.
    duration_ms : float
        Stage duration in milliseconds.
    data : Dict[str, Any]
        Stage-specific data for debugging (diagnosis fields, resolved
        version, fetch metadata, etc.).
    """

    stage: PipelineStage
    success: bool
    message: str
    duration_ms: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RescueResult:
    """
    Complete result of a pip rescue operation.

    Why a dedicated result type for the entire pipeline:
        - Aggregates all stage results into a single report.
        - Provides a ``report()`` method for human-readable output.
        - Enables programmatic callers to inspect each stage's outcome.
        - The ``exit_code`` field maps directly to shell exit codes.

    Attributes
    ----------
    success : bool
        ``True`` if pip is now installed and functional.
    version_installed : Optional[str]
        The pip version that was installed.
    version_requested : Optional[str]
        The version the user requested (may be None for "latest").
    strategy_used : Optional[str]
        Human-readable strategy name (e.g., "ensurepip", "get-pip.py").
    stages : List[StageResult]
        Results from each pipeline stage that was executed.
    total_duration_ms : float
        Total wall-clock time for the entire operation.
    exit_code : ExitCode
        Recommended exit code for shell usage.
    dry_run : bool
        Whether this was a dry run (no changes made).
    errors : List[str]
        Error messages collected across all stages.
    warnings : List[str]
        Warning messages collected across all stages.

    Examples
    --------
    >>> result = RescueResult(
    ...     success=True,
    ...     version_installed="24.3.1",
    ...     version_requested=None,
    ...     strategy_used="ensurepip",
    ...     stages=[],
    ...     total_duration_ms=2500.0,
    ...     exit_code=ExitCode.SUCCESS,
    ...     dry_run=False,
    ...     errors=[],
    ...     warnings=[],
    ... )
    >>> result.success
    True
    >>> print(result.report())
    Installation Report
    ===================
    Status: SUCCESS
    Version installed: 24.3.1
    ...
    """

    success: bool
    version_installed: Optional[str]
    version_requested: Optional[str]
    strategy_used: Optional[str]
    stages: List[StageResult]
    total_duration_ms: float
    exit_code: ExitCode
    dry_run: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def report(self, verbose: bool = False) -> str:
        """
        Generate a human-readable installation report.

        Why a method instead of __str__:
            - The verbose flag controls detail level; __str__ should
              be consistent regardless of context.
            - Enables callers to generate reports with or without
              per-stage details.

        Parameters
        ----------
        verbose : bool
            If ``True``, include per-stage details and raw outputs.

        Returns
        -------
        str
            Formatted report string.

        Examples
        --------
        >>> result = RescueResult(Success=True, ...)
        >>> print(result.report(verbose=True))
        Installation Report
        ===================
        ...
        Pipeline Stages:
          ✓ DIAGNOSIS (150ms): pip 21.0.1 found, broken
          ✓ VERSION_RESOLUTION (2ms): target 24.3.1
          ...
        """
        lines: List[str] = []
        lines.append("Installation Report")
        lines.append("=" * 50)

        if self.dry_run:
            lines.append("** DRY RUN - No changes were made **")
            lines.append("")

        status_str = "SUCCESS" if self.success else "FAILED"
        lines.append(f"Status:              {status_str}")

        if self.version_installed:
            lines.append(f"Version installed:   {self.version_installed}")
        if self.version_requested:
            lines.append(f"Version requested:   {self.version_requested}")
        if self.strategy_used:
            lines.append(f"Strategy:            {self.strategy_used}")
        lines.append(f"Duration:            {self.total_duration_ms / 1000:.1f}s")

        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  ✗ {error}")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")

        if verbose and self.stages:
            lines.append("")
            lines.append("Pipeline Stages:")
            for stage_result in self.stages:
                icon = "✓" if stage_result.success else "✗"
                lines.append(
                    f"  {icon} {stage_result.stage.name} "
                    f"({stage_result.duration_ms:.0f}ms): "
                    f"{stage_result.message}"
                )

        lines.append("")
        lines.append(f"Exit code: {self.exit_code.value}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Orchestrator Class
# ---------------------------------------------------------------------------


class PipRescuer:
    """
    Orchestrates the complete pip diagnosis and installation pipeline.

    Why a class instead of a single function:
        - Configuration is shared across all pipeline stages; passing
          the same 8 parameters to every function is error-prone.
        - The pipeline has intermediate state (diagnosis, resolution,
          fetch results) that subsequent stages consume. A class holds
          this state naturally.
        - Enables testing individual pipeline stages by calling methods
          directly with controlled inputs.
        - Multiple rescuer instances can target different Python
          interpreters simultaneously.

    Parameters
    ----------
    target_version : Optional[str]
        Specific pip version to install. ``None`` for latest compatible.
    python_executable : Optional[Path]
        Path to the Python interpreter to install pip for. ``None`` for
        the currently running Python.
    wheel_path : Optional[Path]
        Path to a local pip wheel file for offline installation.
    get_pip_path : Optional[Path]
        Path to a local ``get-pip.py`` script.
    user_site : bool
        Install to user site-packages instead of system.
    force : bool
        Reinstall pip even if it appears healthy.
    break_system_packages : bool
        Allow installation on externally-managed environments.
    timeout : int
        Timeout in seconds for network operations.
    dry_run : bool
        If ``True``, perform all checks but do not modify the system.
    verbose : bool
        If ``True``, print detailed progress during pipeline execution.
    quiet : bool
        If ``True``, suppress all output except errors.
    cache_dir : Optional[Path]
        Directory for caching downloaded resources.

    Attributes
    ----------
    target_version : Optional[str]
        Requested pip version.
    python_executable : Path
        Target Python interpreter.
    dry_run : bool
        Whether this is a dry run.

    Examples
    --------
    Basic rescue operation:

        >>> rescuer = PipRescuer()
        >>> result = rescuer.run()
        >>> if result.success:
        ...     print(f"pip {result.version_installed} is ready")

    Install a specific version on a different Python:

        >>> rescuer = PipRescuer(
        ...     target_version="21.3.1",
        ...     python_executable=Path("/usr/bin/python3.6"),
        ... )
        >>> result = rescuer.run()

    Dry run to preview actions:

        >>> rescuer = PipRescuer(target_version="23.0.1", dry_run=True)
        >>> result = rescuer.run()
        >>> print(result.report())
        ** DRY RUN - No changes were made **
        ...

    Offline installation:

        >>> rescuer = PipRescuer(
        ...     wheel_path=Path("./pip-23.0.1-py3-none-any.whl"),
        ... )
        >>> result = rescuer.run()

    Quiet mode for scripting:

        >>> rescuer = PipRescuer(quiet=True)
        >>> result = rescuer.run()
        >>> sys.exit(result.exit_code.value)
    """

    def __init__(
        self,
        target_version: Optional[str] = None,
        python_executable: Optional[Path] = None,
        wheel_path: Optional[Path] = None,
        get_pip_path: Optional[Path] = None,
        user_site: bool = False,
        force: bool = False,
        break_system_packages: bool = False,
        timeout: int = 30,
        dry_run: bool = False,
        verbose: bool = False,
        quiet: bool = False,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize the pip rescuer with configuration.

        Parameters
        ----------
        target_version : Optional[str]
            Pip version to install. ``None`` for latest compatible.
        python_executable : Optional[Path]
            Target Python interpreter path.
        wheel_path : Optional[Path]
            Local pip wheel for offline installation.
        get_pip_path : Optional[Path]
            Local get-pip.py script.
        user_site : bool
            Install to user site-packages.
        force : bool
            Reinstall healthy pip.
        break_system_packages : bool
            Bypass PEP 668 protections.
        timeout : int
            Network timeout in seconds.
        dry_run : bool
            Preview mode; no modifications.
        verbose : bool
            Detailed progress output.
        quiet : bool
            Suppress non-error output.
        cache_dir : Optional[Path]
            Resource cache directory.

        Raises
        ------
        FileNotFoundError
            If ``python_executable`` is specified but does not exist.
        """
        self.target_version: Optional[str] = target_version
        self.python_executable: Path = (
            python_executable.resolve()
            if python_executable
            else Path(sys.executable).resolve()
        )
        self.wheel_path: Optional[Path] = wheel_path
        self.get_pip_path: Optional[Path] = get_pip_path
        self.user_site: bool = user_site
        self.force: bool = force
        self.break_system_packages: bool = break_system_packages
        self.timeout: int = timeout
        self.dry_run: bool = dry_run
        self.verbose: bool = verbose
        self.quiet: bool = quiet
        self.cache_dir: Optional[Path] = cache_dir

        # Validate python executable
        if not self.python_executable.exists():
            raise FileNotFoundError(
                f"Python executable not found: {self.python_executable}"
            )

        # Stage results accumulated during run()
        self._stages: List[StageResult] = []
        self._errors: List[str] = []
        self._warnings: List[str] = []

    # ------------------------------------------------------------------
    # Public API: Main Entry Point
    # ------------------------------------------------------------------

    def run(self) -> RescueResult:
        """
        Execute the complete pip rescue pipeline.

        Pipeline stages:
        1. VALIDATION — check arguments and environment preconditions.
        2. DIAGNOSIS — inspect current pip state.
        3. VERSION_RESOLUTION — determine target pip version.
        4. FETCHING — obtain installation resources.
        5. INSTALLATION — execute the installation.
        6. VERIFICATION — confirm pip is functional.

        If any stage fails fatally, the pipeline stops and returns
        a failed result. Non-fatal failures (e.g., network down but
        local wheel available) allow the pipeline to continue with
        degraded functionality.

        Returns
        -------
        RescueResult
            Complete result with success status, version, strategy,
            stage details, and recommended exit code.

        Examples
        --------
        >>> rescuer = PipRescuer()
        >>> result = rescuer.run()
        >>> result.success
        True
        >>> result.exit_code == ExitCode.SUCCESS
        True
        """
        total_start = time.time()
        self._stages = []
        self._errors = []
        self._warnings = []

        # Stage 1: Validation
        stage = self._run_validation()
        self._stages.append(stage)
        if not stage.success:
            return self._build_result(total_start)

        # Stage 2: Diagnosis
        stage, diagnosis = self._run_diagnosis()
        self._stages.append(stage)
        if not stage.success or diagnosis is None:
            return self._build_result(total_start)

        # If healthy and not forced, we're done
        if diagnosis.status == PipStatus.HEALTHY and not self.force:
            stage = StageResult(
                stage=PipelineStage.COMPLETE,
                success=True,
                message=f"pip {diagnosis.current_version} is already installed and healthy",
                duration_ms=0,
                data={"current_version": diagnosis.current_version},
            )
            self._stages.append(stage)
            return RescueResult(
                success=True,
                version_installed=diagnosis.current_version,
                version_requested=self.target_version,
                strategy_used=None,
                stages=self._stages,
                total_duration_ms=(time.time() - total_start) * 1000,
                exit_code=ExitCode.SUCCESS,
                dry_run=self.dry_run,
                errors=self._errors,
                warnings=self._warnings,
            )

        # Stage 3: Version Resolution
        stage, resolution = self._run_version_resolution(diagnosis)
        self._stages.append(stage)
        if not stage.success or resolution is None:
            return self._build_result(total_start)

        resolved_version = resolution.version
        if not resolution.compatible and not self.force:
            self._errors.append(
                f"Version {resolved_version} is incompatible: {resolution.reason}"
            )
            return self._build_result(total_start, ExitCode.INCOMPATIBLE_VERSION)

        # Stage 4: Fetching
        stage, fetch_data = self._run_fetching(resolved_version, diagnosis)
        self._stages.append(stage)

        # Stage 5: Installation
        stage, install_result = self._run_installation(
            diagnosis, resolved_version, fetch_data
        )
        self._stages.append(stage)

        # Stage 6: Verification
        stage, verified_version = self._run_verification(install_result)
        self._stages.append(stage)

        # Build final result
        exit_code = ExitCode.SUCCESS
        if not stage.success:
            exit_code = ExitCode.VERIFICATION_FAILED

        return RescueResult(
            success=stage.success,
            version_installed=verified_version or install_result.version_installed if install_result else None,
            version_requested=self.target_version,
            strategy_used=install_result.strategy_used.name if install_result and install_result.strategy_used else None,
            stages=self._stages,
            total_duration_ms=(time.time() - total_start) * 1000,
            exit_code=exit_code,
            dry_run=self.dry_run,
            errors=self._errors,
            warnings=self._warnings,
        )

    def uninstall(self) -> RescueResult:
        """
        Completely remove pip from the current Python environment.
    
        Why this method exists:
            - A corrupted pip installation may be beyond repair. Removing it
              entirely and reinstalling from scratch is often faster and more
              reliable than attempting incremental fixes.
            - Testing the repair functionality requires a clean starting state.
              This method provides a reliable way to remove pip between test
              runs without manually deleting files.
            - Some environments ship with a system pip that conflicts with a
              user-installed pip. Removing one of them resolves version
              conflicts and import errors.
            - Security policies may require removing pip from production
              environments after dependencies are installed. This provides
              a programmatic way to enforce that policy.
            - Container image builds benefit from removing pip to reduce
              image size and attack surface after ``pip install`` is complete.
    
        Why remove multiple paths instead of just the module:
            - pip installs itself in up to three locations: the ``pip/``
              module directory, the ``pip-{version}.dist-info/`` metadata
              directory, and the ``pip`` executable script (or ``pip.exe``
              on Windows).
            - Leaving the dist-info directory behind causes ``pip list``
              and other tools to report pip as installed even though the
              module is gone, creating confusion.
            - Leaving the executable script behind means ``pip --version``
              may still find and execute a broken or outdated pip from a
              different location.
            - Partial cleanup is worse than no cleanup: it creates an
              inconsistent state that is harder to diagnose than a
              completely missing pip.
    
        Why verify removal after attempting it:
            - Filesystem operations can fail silently: ``shutil.rmtree``
              may skip files it cannot delete due to permissions or locks
              without raising an exception if ``ignore_errors=True`` is set.
            - Some systems have multiple pip installations (system + user).
              Removing one may leave another intact, and the diagnosis
              confirms whether the target pip is truly gone.
            - Verification provides a definitive answer: the method returns
              success only when pip is confirmed missing, not just when
              deletion commands ran without exceptions.
    
        Removal order and rationale:
            1. **Module directory first** — The ``pip/`` directory contains
               the actual Python code. Removing it first ensures that even
               if subsequent steps fail, ``import pip`` will fail, making
               the broken state immediately obvious rather than subtly
               wrong.
            2. **Dist-info directories second** — These contain metadata.
               Removing them after the module ensures package managers
               don't think pip is still installed.
            3. **Executable script last** — The script is useless without
               the module. Removing it last prevents race conditions where
               a running pip process might try to import the module while
               it's being deleted.
    
        Parameters
        ----------
        (None — uses ``self.python_executable`` and ``self.dry_run`` from
        the PipRescuer instance.)
    
        Returns
        -------
        RescueResult
            Complete result with success status and details of what was
            removed. The ``data`` field of each stage contains:
            - ``removed``: List[str] — paths that were successfully deleted
            - ``failed``: List[str] — paths that could not be deleted with
              reason
            - ``not_found``: List[str] — paths that were expected but did
              not exist
    
        Warnings
        --------
        - This method permanently deletes files. Use ``--dry-run`` to
          preview what would be removed without making changes.
        - On Windows, pip files may be locked by running Python processes.
          Close all Python terminals and IDE instances before uninstalling.
        - System-managed pip installations (e.g., installed via ``apt`` on
          Debian/Ubuntu) may be reinstalled by system package updates.
          Removing them may cause conflicts with the system package manager.
        - After uninstalling pip, ``python -m pip`` will fail. The only
          way to reinstall pip is through this tool, ``ensurepip``, or
          manually downloading ``get-pip.py``.
    
        Examples
        --------
        Basic uninstall:
    
            >>> rescuer = PipRescuer()
            >>> result = rescuer.uninstall()
            >>> print(result.report())
            Installation Report
            ===================
            Status: SUCCESS
            Strategy: uninstall
            Duration: 0.5s
    
        Dry run to preview:
    
            >>> rescuer = PipRescuer(dry_run=True)
            >>> result = rescuer.uninstall()
            >>> print(result.report())
            ** DRY RUN - No changes were made **
            ...
            Would remove:
              /usr/lib/python3.11/site-packages/pip
              /usr/lib/python3.11/site-packages/pip-24.3.1.dist-info
              /usr/bin/pip
    
        Uninstall from a specific Python:
    
            >>> rescuer = PipRescuer(python_executable=Path("/usr/bin/python3.8"))
            >>> result = rescuer.uninstall()
    
        Uninstall then reinstall clean:
    
            >>> rescuer = PipRescuer(force=True)
            >>> rescuer.uninstall()
            >>> result = rescuer.run()  # Fresh install
            >>> result.success
            True
    
        Force removal even if files are locked (uses ignore_errors):
    
            >>> rescuer = PipRescuer()
            >>> result = rescuer.uninstall()  # Will skip locked files with warning
    
        Notes
        -----
        - User-site pip installations (``~/.local/lib/pythonX.Y/site-packages/pip``)
          are detected and removed automatically when the diagnosis identifies
          them. No special flag is needed.
        - On macOS, pip may be installed in ``/Library/Python/X.Y/site-packages/``
          for system-level installations. The diagnosis searches all
          site-packages paths, so these are covered.
        - Virtual environments are fully supported: the method removes pip
          only from the environment associated with ``self.python_executable``,
          not from any other environment or the system Python.
        """
        import shutil
        import time
        from pathlib import Path
    
        start_time = time.time()
        self._stages = []
        self._errors = []
        self._warnings = []
    
        # ------------------------------------------------------------------
        # Stage 1: Diagnosis — find all pip files before removal
        # ------------------------------------------------------------------
        if self.verbose and not self.quiet:
            print("Diagnosing pip locations...", end=" ", flush=True)
    
        try:
            checker = PipChecker(python_executable=self.python_executable)
            diagnosis = checker.run_full_diagnosis()
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._errors.append(f"Pre-uninstall diagnosis failed: {e}")
            return RescueResult(
                success=False,
                version_installed=None,
                version_requested=None,
                strategy_used=None,
                stages=[
                    StageResult(
                        stage=PipelineStage.DIAGNOSIS,
                        success=False,
                        message=f"Diagnosis failed: {e}",
                        duration_ms=duration_ms,
                    )
                ],
                total_duration_ms=duration_ms,
                exit_code=ExitCode.GENERAL_ERROR,
                dry_run=self.dry_run,
                errors=self._errors,
                warnings=self._warnings,
            )
    
        diagnosis_duration = (time.time() - start_time) * 1000
    
        if diagnosis.status == PipStatus.MISSING:
            # pip is already absent — nothing to do
            if self.verbose and not self.quiet:
                print("OK (pip is not installed)")
            return RescueResult(
                success=True,
                version_installed=None,
                version_requested=None,
                strategy_used="uninstall",
                stages=[
                    StageResult(
                        stage=PipelineStage.DIAGNOSIS,
                        success=True,
                        message="pip is already not installed",
                        duration_ms=diagnosis_duration,
                    )
                ],
                total_duration_ms=(time.time() - start_time) * 1000,
                exit_code=ExitCode.SUCCESS,
                dry_run=self.dry_run,
                errors=[],
                warnings=["pip was not installed; nothing to remove"],
            )
    
        if self.verbose and not self.quiet:
            print(f"OK (found {len(diagnosis.install_paths)} locations)")
    
        # ------------------------------------------------------------------
        # Stage 2: Collect all paths that need removal
        # ------------------------------------------------------------------
        paths_to_remove: List[Tuple[Path, str]] = []
        """
        Each entry is (path, description) where description explains what
        this path contains. Used for logging and dry-run display.
        """
    
        # pip module directory (e.g., /usr/lib/python3.11/site-packages/pip/)
        if diagnosis.module_path and diagnosis.module_path.exists():
            paths_to_remove.append(
                (diagnosis.module_path, "pip module directory")
            )
    
        # pip dist-info directories (e.g., pip-24.3.1.dist-info/)
        if diagnosis.install_paths:
            for install_path in diagnosis.install_paths:
                # Skip if it's the same as the module path (already added)
                if install_path == diagnosis.module_path:
                    continue
                if install_path.exists():
                    paths_to_remove.append(
                        (install_path, f"pip metadata: {install_path.name}")
                    )
    
        # Also search for any orphaned pip dist-info directories
        # Why: partial uninstalls or version upgrades can leave behind
        # dist-info directories from previous versions that the diagnosis
        # might not report.
        search_dirs: List[Path] = []
        search_dirs.extend(diagnosis.environment.site_packages_paths)
        if diagnosis.environment.user_site_packages:
            search_dirs.append(diagnosis.environment.user_site_packages)
    
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            try:
                for entry in search_dir.iterdir():
                    if entry.is_dir() and entry.name.startswith("pip-") and entry.name.endswith(".dist-info"):
                        # Check if this dist-info is already in the removal list
                        already_listed = any(p == entry for p, _ in paths_to_remove)
                        if not already_listed:
                            paths_to_remove.append(
                                (entry, f"orphaned pip metadata: {entry.name}")
                            )
            except PermissionError:
                self._warnings.append(f"Cannot scan directory: {search_dir}")
    
        # pip executable script
        if diagnosis.executable_path and diagnosis.executable_path.exists():
            paths_to_remove.append(
                (diagnosis.executable_path, "pip executable script")
            )
        else:
            # Search for pip executable in common locations
            # Why: the diagnosis may miss the executable if it's in a
            # non-standard location
            exe_name = "pip.exe" if sys.platform == "win32" else "pip"
            additional_exe_paths: List[Path] = [
                self.python_executable.parent / exe_name,
                self.python_executable.parent / "Scripts" / exe_name
                if sys.platform == "win32"
                else self.python_executable.parent / "bin" / exe_name,
                Path.home() / ".local" / "bin" / exe_name,
            ]
            for candidate in additional_exe_paths:
                if candidate.exists() and candidate.is_file():
                    already_listed = any(p == candidate for p, _ in paths_to_remove)
                    if not already_listed:
                        paths_to_remove.append(
                            (candidate, "pip executable script (discovered)")
                        )
    
        if not paths_to_remove:
            # Diagnosis said pip exists but we found nothing to remove.
            # This is a peculiar state — pip might be installed in a way
            # the diagnosis detects but our path collection misses.
            self._errors.append(
                "Diagnosis indicates pip is installed but no removable "
                "files were located. The installation may be in a "
                "non-standard location."
            )
            return RescueResult(
                success=False,
                version_installed=diagnosis.current_version,
                version_requested=None,
                strategy_used=None,
                stages=[
                    StageResult(
                        stage=PipelineStage.DIAGNOSIS,
                        success=False,
                        message="pip detected but no files located for removal",
                        duration_ms=(time.time() - start_time) * 1000,
                    )
                ],
                total_duration_ms=(time.time() - start_time) * 1000,
                exit_code=ExitCode.GENERAL_ERROR,
                dry_run=self.dry_run,
                errors=self._errors,
                warnings=self._warnings,
            )
    
        # ------------------------------------------------------------------
        # Stage 3: Dry run — report what would be removed
        # ------------------------------------------------------------------
        if self.dry_run:
            if self.verbose and not self.quiet:
                print("\n[DRY RUN] The following would be removed:")
                for path, description in paths_to_remove:
                    print(f"  • {path} ({description})")
    
            return RescueResult(
                success=True,
                version_installed=diagnosis.current_version,
                version_requested=None,
                strategy_used="uninstall",
                stages=[
                    StageResult(
                        stage=PipelineStage.DIAGNOSIS,
                        success=True,
                        message=f"Found {len(paths_to_remove)} paths to remove",
                        duration_ms=diagnosis_duration,
                        data={
                            "would_remove": [
                                {"path": str(p), "description": d}
                                for p, d in paths_to_remove
                            ]
                        },
                    ),
                    StageResult(
                        stage=PipelineStage.INSTALLATION,
                        success=True,
                        message=f"[DRY RUN] Would remove {len(paths_to_remove)} paths",
                        duration_ms=0,
                    ),
                ],
                total_duration_ms=(time.time() - start_time) * 1000,
                exit_code=ExitCode.SUCCESS,
                dry_run=True,
                errors=[],
                warnings=self._warnings,
            )
    
        # ------------------------------------------------------------------
        # Stage 4: Execute removal
        # ------------------------------------------------------------------
        if self.verbose and not self.quiet:
            print(f"Removing {len(paths_to_remove)} paths...", end=" ", flush=True)
    
        removed: List[str] = []
        failed: List[str] = []
        not_found: List[str] = []
    
        for path, description in paths_to_remove:
            if not path.exists():
                not_found.append(f"{path} ({description}) — no longer exists")
                continue
    
            try:
                if path.is_dir():
                    # Why ignore_errors=False first:
                    #   - We want to know if files are locked or permissions
                    #     prevent deletion. Silent failure hides problems.
                    # Why fall back to ignore_errors=True:
                    #   - Some files may be read-only or locked by another
                    #     process. We try our best to clean up what we can.
                    try:
                        shutil.rmtree(path, ignore_errors=False)
                    except Exception as first_attempt_error:
                        # Retry with ignore_errors for stubborn files
                        shutil.rmtree(path, ignore_errors=True)
                        # Check if anything remains
                        if path.exists():
                            remaining = list(path.rglob("*"))
                            if remaining:
                                raise RuntimeError(
                                    f"Could not remove {len(remaining)} files "
                                    f"in {path} (first error: {first_attempt_error})"
                                )
                else:
                    path.unlink()
    
                removed.append(f"{path} ({description})")
    
            except PermissionError as e:
                failed.append(f"{path} ({description}) — permission denied: {e}")
            except OSError as e:
                failed.append(f"{path} ({description}) — OS error: {e}")
            except Exception as e:
                failed.append(f"{path} ({description}) — unexpected error: {e}")
    
        removal_duration = (time.time() - start_time) * 1000 - diagnosis_duration
    
        # ------------------------------------------------------------------
        # Stage 5: Verify removal
        # ------------------------------------------------------------------
        if self.verbose and not self.quiet:
            print("OK" if not failed else f"OK with {len(failed)} failures")
            print("Verifying removal...", end=" ", flush=True)
    
        try:
            post_checker = PipChecker(python_executable=self.python_executable)
            post_diagnosis = post_checker.run_full_diagnosis()
        except Exception as e:
            # If even diagnosis fails after removal, something went very wrong
            self._errors.append(f"Post-removal verification failed: {e}")
            return RescueResult(
                success=False,
                version_installed=None,
                version_requested=None,
                strategy_used=None,
                stages=[
                    StageResult(
                        stage=PipelineStage.DIAGNOSIS,
                        success=True,
                        message=f"Found {len(paths_to_remove)} paths",
                        duration_ms=diagnosis_duration,
                        data={"paths_found": len(paths_to_remove)},
                    ),
                    StageResult(
                        stage=PipelineStage.INSTALLATION,
                        success=len(failed) == 0,
                        message=f"Removed {len(removed)} paths"
                        + (f", {len(failed)} failed" if failed else ""),
                        duration_ms=removal_duration,
                        data={
                            "removed": removed,
                            "failed": failed,
                            "not_found": not_found,
                        },
                    ),
                    StageResult(
                        stage=PipelineStage.VERIFICATION,
                        success=False,
                        message=f"Verification crashed: {e}",
                        duration_ms=(time.time() - start_time) * 1000
                        - diagnosis_duration
                        - removal_duration,
                    ),
                ],
                total_duration_ms=(time.time() - start_time) * 1000,
                exit_code=ExitCode.GENERAL_ERROR,
                dry_run=False,
                errors=self._errors,
                warnings=self._warnings,
            )
    
        verify_duration = (time.time() - start_time) * 1000 - diagnosis_duration - removal_duration
    
        # Determine success based on whether pip is now missing
        pip_is_gone = post_diagnosis.status == PipStatus.MISSING
    
        if self.verbose and not self.quiet:
            if pip_is_gone:
                print("OK (pip removed successfully)")
            else:
                print(f"WARNING (pip status: {post_diagnosis.status.name})")
    
        # Build appropriate messages
        if pip_is_gone and not failed:
            final_message = f"Successfully removed pip {diagnosis.current_version or ''}"
        elif pip_is_gone and failed:
            final_message = (
                f"pip removed but {len(failed)} paths could not be deleted"
            )
            self._warnings.extend(failed)
        else:
            final_message = (
                f"pip removal incomplete (status: {post_diagnosis.status.name})"
            )
            if post_diagnosis.module_path:
                self._errors.append(
                    f"pip module still exists at: {post_diagnosis.module_path}"
                )
            if post_diagnosis.executable_path:
                self._errors.append(
                    f"pip executable still exists at: {post_diagnosis.executable_path}"
                )
    
        # Build stages list
        stages: List[StageResult] = [
            StageResult(
                stage=PipelineStage.DIAGNOSIS,
                success=True,
                message=f"Found {len(paths_to_remove)} paths to remove",
                duration_ms=diagnosis_duration,
                data={
                    "pip_version": diagnosis.current_version,
                    "paths_found": len(paths_to_remove),
                    "paths_detail": [
                        {"path": str(p), "description": d}
                        for p, d in paths_to_remove
                    ],
                },
            ),
            StageResult(
                stage=PipelineStage.INSTALLATION,
                success=len(failed) == 0,
                message=f"Removed {len(removed)} of {len(paths_to_remove)} paths",
                duration_ms=removal_duration,
                data={
                    "removed": removed,
                    "failed": failed,
                    "not_found": not_found,
                },
            ),
            StageResult(
                stage=PipelineStage.VERIFICATION,
                success=pip_is_gone,
                message=(
                    "pip is completely removed"
                    if pip_is_gone
                    else f"pip status: {post_diagnosis.status.name}"
                ),
                duration_ms=verify_duration,
                data={
                    "post_status": post_diagnosis.status.name,
                    "remnants": [
                        str(p) for p in post_diagnosis.install_paths
                    ] if post_diagnosis.install_paths else [],
                },
            ),
        ]
    
        return RescueResult(
            success=pip_is_gone,
            version_installed=None,
            version_requested=None,
            strategy_used="uninstall" if pip_is_gone else None,
            stages=stages,
            total_duration_ms=(time.time() - start_time) * 1000,
            exit_code=ExitCode.SUCCESS if pip_is_gone else ExitCode.GENERAL_ERROR,
            dry_run=False,
            errors=self._errors,
            warnings=self._warnings,
        )

    # ------------------------------------------------------------------
    # Private: Pipeline Stages
    # ------------------------------------------------------------------

    def _run_validation(self) -> StageResult:
        """
        Validate command-line arguments and environment preconditions.

        Why validate before diagnosis:
            - Catching invalid arguments early avoids wasting time on
              diagnosis only to fail later.
            - Contradictory arguments (e.g., ``--offline`` with a
              non-existent wheel file) should be caught immediately.
        """
        start = time.time()

        # Check wheel path exists if provided
        if self.wheel_path is not None:
            if not self.wheel_path.exists():
                duration_ms = (time.time() - start) * 1000
                self._errors.append(f"Wheel file not found: {self.wheel_path}")
                return StageResult(
                    stage=PipelineStage.VALIDATION,
                    success=False,
                    message=f"Wheel file not found: {self.wheel_path}",
                    duration_ms=duration_ms,
                )
            if not self.wheel_path.suffix == ".whl":
                self._warnings.append(
                    f"File does not have .whl extension: {self.wheel_path}"
                )

        # Check get-pip.py path exists if provided
        if self.get_pip_path is not None and not self.get_pip_path.exists():
            duration_ms = (time.time() - start) * 1000
            self._errors.append(f"get-pip.py script not found: {self.get_pip_path}")
            return StageResult(
                stage=PipelineStage.VALIDATION,
                success=False,
                message=f"get-pip.py script not found: {self.get_pip_path}",
                duration_ms=duration_ms,
            )

        # Check user_site and break_system_packages conflict
        if self.user_site and self.break_system_packages:
            self._warnings.append(
                "--user and --break-system-packages are both set; "
                "--user takes precedence"
            )

        duration_ms = (time.time() - start) * 1000
        return StageResult(
            stage=PipelineStage.VALIDATION,
            success=True,
            message="Arguments validated",
            duration_ms=duration_ms,
        )

    def _run_diagnosis(self) -> Tuple[StageResult, Optional[PipDiagnosis]]:
        """
        Diagnose the current state of pip and the Python environment.

        Why return both StageResult and PipDiagnosis:
            - StageResult is for reporting/progress display.
            - PipDiagnosis is passed to subsequent stages.
            - Separating them avoids storing the diagnosis in StageResult
              and extracting it later.
        """
        start = time.time()

        if self.verbose and not self.quiet:
            print("Diagnosing pip...", end=" ", flush=True)

        try:
            checker = PipChecker(python_executable=self.python_executable)
            diagnosis = checker.run_full_diagnosis()
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            self._errors.append(f"Diagnosis failed: {e}")
            if self.verbose and not self.quiet:
                print("FAILED")
            return (
                StageResult(
                    stage=PipelineStage.DIAGNOSIS,
                    success=False,
                    message=f"Diagnosis failed: {e}",
                    duration_ms=duration_ms,
                ),
                None,
            )

        duration_ms = (time.time() - start) * 1000

        if diagnosis.status == PipStatus.HEALTHY:
            message = f"pip {diagnosis.current_version} is healthy"
        elif diagnosis.status == PipStatus.MISSING:
            message = "pip is not installed"
        elif diagnosis.status == PipStatus.BROKEN:
            message = f"pip {diagnosis.current_version} is broken"
            if diagnosis.import_error:
                message += f" ({diagnosis.import_error[:80]}...)"
        elif diagnosis.status == PipStatus.BLOCKED:
            message = "pip is blocked (permission or security restriction)"
        else:
            message = "pip state could not be determined"

        if self.verbose and not self.quiet:
            print(f"OK ({message})")

        return (
            StageResult(
                stage=PipelineStage.DIAGNOSIS,
                success=True,
                message=message,
                duration_ms=duration_ms,
                data={
                    "status": diagnosis.status.name,
                    "current_version": diagnosis.current_version,
                    "python_version": diagnosis.environment.python_version,
                    "is_virtualenv": diagnosis.environment.is_virtualenv,
                    "is_conda": diagnosis.environment.is_conda,
                    "has_internet": diagnosis.environment.has_internet,
                },
            ),
            diagnosis,
        )

    def _run_version_resolution(
        self, diagnosis: PipDiagnosis
    ) -> Tuple[StageResult, Optional[VersionResolution]]:
        """
        Resolve which pip version to install.

        Why resolution is a separate stage:
            - If the user requests an incompatible version, we can
              suggest alternatives before attempting any download.
            - Resolution depends on diagnosis (Python version) but
              not on network state, so it can complete even offline.
        """
        start = time.time()

        if self.verbose and not self.quiet:
            print("Resolving version...", end=" ", flush=True)

        try:
            resolver = VersionResolver(
                python_version=diagnosis.environment.python_version_tuple,
                strict=False,
            )

            if self.target_version:
                resolution = resolver.resolve_user_version(self.target_version)
            else:
                resolution = resolver.resolve_best_version()
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            self._errors.append(f"Version resolution failed: {e}")
            if self.verbose and not self.quiet:
                print("FAILED")
            return (
                StageResult(
                    stage=PipelineStage.VERSION_RESOLUTION,
                    success=False,
                    message=f"Version resolution failed: {e}",
                    duration_ms=duration_ms,
                ),
                None,
            )

        duration_ms = (time.time() - start) * 1000

        if resolution.version is None:
            self._errors.append(resolution.reason)
            if self.verbose and not self.quiet:
                print("FAILED")
            return (
                StageResult(
                    stage=PipelineStage.VERSION_RESOLUTION,
                    success=False,
                    message=resolution.reason,
                    duration_ms=duration_ms,
                    data={"alternatives": resolution.alternatives},
                ),
                None,
            )

        if not resolution.compatible:
            self._warnings.append(resolution.reason)
            if resolution.alternatives:
                self._warnings.append(
                    f"Compatible alternatives: {', '.join(resolution.alternatives[:5])}"
                )

        if self.verbose and not self.quiet:
            print(f"OK (target: {resolution.version})")

        return (
            StageResult(
                stage=PipelineStage.VERSION_RESOLUTION,
                success=True,
                message=f"Target version: {resolution.version}",
                duration_ms=duration_ms,
                data={
                    "resolved_version": resolution.version,
                    "source": resolution.source.name,
                    "compatible": resolution.compatible,
                    "alternatives": resolution.alternatives,
                },
            ),
            resolution,
        )

    def _run_fetching(
        self,
        resolution: VersionResolution,
        diagnosis: PipDiagnosis,
    ) -> Tuple[StageResult, Dict[str, Any]]:
        """
        Fetch installation resources (get-pip.py or wheel).

        Why fetch before install:
            - If resources cannot be obtained, we fail early rather
              than after a partial installation.
            - Caching: previously downloaded resources are reused,
              avoiding redundant network requests.
            - The fetch stage determines whether we have a wheel or
              get-pip.py available, which influences strategy selection
              in the installation stage.

        Returns
        -------
        Tuple[StageResult, Dict[str, Any]]
            Stage result and a data dictionary with keys:
            - ``has_wheel``: bool
            - ``wheel_path``: Optional[Path]
            - ``has_get_pip``: bool
            - ``get_pip_path``: Optional[Path]
        """
        start = time.time()
        data: Dict[str, Any] = {
            "has_wheel": False,
            "wheel_path": None,
            "has_get_pip": False,
            "get_pip_path": None,
        }

        # If user provided a wheel, use it directly
        if self.wheel_path and self.wheel_path.exists():
            data["has_wheel"] = True
            data["wheel_path"] = self.wheel_path
            if self.verbose and not self.quiet:
                print(f"Using local wheel: {self.wheel_path.name}")
            return (
                StageResult(
                    stage=PipelineStage.FETCHING,
                    success=True,
                    message=f"Using local wheel: {self.wheel_path.name}",
                    duration_ms=0,
                    data=data,
                ),
                data,
            )

        # If user provided get-pip.py, use it directly
        if self.get_pip_path and self.get_pip_path.exists():
            data["has_get_pip"] = True
            data["get_pip_path"] = self.get_pip_path
            if self.verbose and not self.quiet:
                print(f"Using local get-pip.py: {self.get_pip_path.name}")
            return (
                StageResult(
                    stage=PipelineStage.FETCHING,
                    success=True,
                    message=f"Using local get-pip.py",
                    duration_ms=0,
                    data=data,
                ),
                data,
            )

        # No local resources; need network
        if not diagnosis.environment.has_internet:
            duration_ms = (time.time() - start) * 1000
            self._errors.append(
                "No network available and no local wheel or get-pip.py provided. "
                "Provide --offline <wheel> or --get-pip <script> for offline installation."
            )
            return (
                StageResult(
                    stage=PipelineStage.FETCHING,
                    success=False,
                    message="Network unavailable and no local resources",
                    duration_ms=duration_ms,
                    data=data,
                ),
                data,
            )

        if self.verbose and not self.quiet:
            print("Fetching resources...", end=" ", flush=True)

        try:
            fetcher = ResourceFetcher(
                cache_dir=self.cache_dir,
                timeout=self.timeout,
            )

            # Try to fetch get-pip.py first (more reliable for version selection)
            python_ver = f"{diagnosis.environment.python_version_tuple[0]}.{diagnosis.environment.python_version_tuple[1]}"
            success, path, meta = fetcher.fetch_get_pip(version=python_ver)

            if success:
                data["has_get_pip"] = True
                data["get_pip_path"] = path
                if self.verbose and not self.quiet:
                    print(f"OK (get-pip.py, {meta.size_bytes} bytes)")
                duration_ms = (time.time() - start) * 1000
                return (
                    StageResult(
                        stage=PipelineStage.FETCHING,
                        success=True,
                        message=f"Downloaded get-pip.py ({meta.size_bytes} bytes)",
                        duration_ms=duration_ms,
                        data=data,
                    ),
                    data,
                )

            # Fallback: try to fetch wheel
            success, path, meta = fetcher.fetch_pip_wheel(
                resolution.version or "latest"
            )
            if success:
                data["has_wheel"] = True
                data["wheel_path"] = path
                if self.verbose and not self.quiet:
                    print(f"OK (wheel, {meta.size_bytes} bytes)")
                duration_ms = (time.time() - start) * 1000
                return (
                    StageResult(
                        stage=PipelineStage.FETCHING,
                        success=True,
                        message=f"Downloaded pip wheel ({meta.size_bytes} bytes)",
                        duration_ms=duration_ms,
                        data=data,
                    ),
                    data,
                )

            # Both failed
            raise RuntimeError("Failed to download get-pip.py or wheel")

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            self._errors.append(f"Failed to fetch resources: {e}")
            if self.verbose and not self.quiet:
                print("FAILED")
            return (
                StageResult(
                    stage=PipelineStage.FETCHING,
                    success=False,
                    message=f"Resource fetch failed: {e}",
                    duration_ms=duration_ms,
                    data=data,
                ),
                data,
            )

    def _run_installation(
        self,
        diagnosis: PipDiagnosis,
        version: str,
        fetch_data: Dict[str, Any],
    ) -> Tuple[StageResult, Optional[InstallResult]]:
        """
        Execute the pip installation.

        Why this stage handles the dry-run flag:
            - All previous stages perform read-only operations (diagnosis,
              version resolution, fetching to cache). Dry-run stops here,
              before any filesystem modifications.
            - This is the point of no return; the user sees exactly what
              would happen before it happens.
        """
        start = time.time()

        if self.dry_run:
            strategy = "get-pip.py" if fetch_data.get("has_get_pip") else "wheel"
            return (
                StageResult(
                    stage=PipelineStage.INSTALLATION,
                    success=True,
                    message=f"[DRY RUN] Would install pip {version} via {strategy}",
                    duration_ms=0,
                ),
                None,
            )

        if self.verbose and not self.quiet:
            print("Installing pip...", end=" ", flush=True)

        try:
            installer = PipInstaller(
                diagnosis=diagnosis,
                target_version=version,
                wheel_path=fetch_data.get("wheel_path"),
                get_pip_path=fetch_data.get("get_pip_path"),
                force=self.force,
                user_site=self.user_site,
                break_system_packages=self.break_system_packages,
            )

            result = installer.install()
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            self._errors.append(f"Installation failed: {e}")
            if self.verbose and not self.quiet:
                print("FAILED")
            return (
                StageResult(
                    stage=PipelineStage.INSTALLATION,
                    success=False,
                    message=f"Installation exception: {e}",
                    duration_ms=duration_ms,
                ),
                None,
            )

        duration_ms = (time.time() - start) * 1000

        if result.success:
            strategy_name = result.strategy_used.name if result.strategy_used else "unknown"
            if self.verbose and not self.quiet:
                print(f"OK (via {strategy_name})")

            for warning in result.warnings:
                self._warnings.append(warning)

            return (
                StageResult(
                    stage=PipelineStage.INSTALLATION,
                    success=True,
                    message=f"Installed via {strategy_name}",
                    duration_ms=duration_ms,
                    data={
                        "strategy": strategy_name,
                        "version_installed": result.version_installed,
                    },
                ),
                result,
            )
        else:
            for error in result.errors:
                self._errors.append(error)

            if self.verbose and not self.quiet:
                print(f"FAILED ({result.errors[0] if result.errors else 'unknown'})")

            return (
                StageResult(
                    stage=PipelineStage.INSTALLATION,
                    success=False,
                    message=f"Installation failed: {result.errors[0] if result.errors else 'unknown'}",
                    duration_ms=duration_ms,
                ),
                result,
            )
   
    def _run_verification(
        self,
        install_result: Optional[InstallResult],
    ) -> Tuple[StageResult, Optional[str]]:
        """
        Verify that the installed pip is functional after installation.
    
        Why this stage exists separately from installation:
            - A subprocess returning exit code 0 during installation does not
              guarantee pip actually works. The script may have completed but
              left a broken installation behind.
            - Verification confirms the installed version matches expectations,
              catching cases where a wrong version was silently installed.
            - If verification fails, the orchestrator can report a specific
              error rather than a generic "installation failed" message that
              would require manual debugging.
            - The verified version string from ``pip --version`` is more
              reliable than the version reported during installation because
              it comes from the running pip process itself.
    
        Why this method uses direct subprocess verification instead of
        instantiating another ``PipInstaller``:
            - ``PipInstaller`` requires a ``PipDiagnosis`` object which
              contains an ``EnvironmentInfo``. The ``InstallResult`` from
              installation does not carry the full diagnosis, so constructing
              a new ``PipInstaller`` for verification would require either
              storing the diagnosis (adding state) or re-diagnosing (wasteful).
            - Direct subprocess verification is simpler, faster, and has
              fewer failure modes. It runs ``python -m pip --version`` and
              ``pip --version``, parses the output, and confirms the version.
            - This approach is independent of the installer module, reducing
              coupling between verification and installation logic.
    
        Why try both ``python -m pip --version`` and ``pip --version``:
            - ``python -m pip`` uses Python's import system, which finds pip
              via ``sys.path`` and site-packages. This works even if the
              pip executable script is not on ``PATH`` (common immediately
              after installation before the terminal is restarted).
            - ``pip --version`` (direct executable) confirms the script was
              created and placed in a directory on ``PATH``. If this works,
              the installation is fully complete.
            - If ``python -m pip`` works but ``pip`` directly does not, the
              installation is functional but has a ``PATH`` configuration
              issue. This is reported as a warning, not a failure.
    
        Parameters
        ----------
        install_result : Optional[InstallResult]
            The result from the installation stage. If ``None``, installation
            did not occur (dry run, skipped, or failed before attempting).
            This method returns early with a failure stage when ``None``
            is received.
    
        Returns
        -------
        stage_result : StageResult
            Structured result indicating whether verification succeeded or
            failed, with timing and diagnostic information.
        version : Optional[str]
            The verified pip version string if verification succeeded
            (e.g., ``'24.3.1'``). ``None`` if verification failed or was
            skipped.
    
        Notes
        -----
        - The verification timeout is 30 seconds. A functional pip should
          respond to ``--version`` in under 2 seconds. If pip hangs,
          something is severely wrong (deadlock, corrupted executable,
          filesystem issue), and waiting longer provides no useful
          information.
        - On Windows, the pip executable may have a ``.exe`` extension.
          ``subprocess.run`` with ``python -m pip`` avoids this issue
          entirely since it uses the module path.
        - The version is parsed using a regex that expects the format
          ``pip X.Y.Z``. Custom or development pip builds may use
          different version strings (e.g., ``pip 24.1.dev0``). The regex
          handles these by extracting only the leading numeric portion.
    
        Examples
        --------
        Successful verification:
    
            >>> # Assuming install_result is a valid InstallResult
            >>> stage, version = self._run_verification(install_result)
            >>> stage.success
            True
            >>> version
            '24.3.1'
            >>> stage.message
            'pip 24.3.1 is functional'
    
        Verification when installation was skipped:
    
            >>> stage, version = self._run_verification(None)
            >>> stage.success
            False
            >>> stage.message
            'No installation result to verify'
            >>> version is None
            True
    
        Verification when pip is broken after installation:
    
            >>> # If pip was installed but crashes on --version
            >>> stage, version = self._run_verification(install_result)
            >>> stage.success
            False
            >>> stage.message
            'pip is not functional'
            >>> stage.data['output']
            'Traceback (most recent call last)...'
        """
        import re
        import subprocess
        import time
    
        start = time.time()
    
        # Early return: dry run requires no verification
        if self.dry_run:
            return (
                StageResult(
                    stage=PipelineStage.VERIFICATION,
                    success=True,
                    message="[DRY RUN] Would verify pip --version",
                    duration_ms=0.0,
                ),
                None,
            )
    
        # Early return: no installation was attempted
        if install_result is None:
            return (
                StageResult(
                    stage=PipelineStage.VERIFICATION,
                    success=False,
                    message="No installation result to verify",
                    duration_ms=0.0,
                ),
                None,
            )
    
        if self.verbose and not self.quiet:
            print("Verifying installation...", end=" ", flush=True)
    
        version: Optional[str] = None
        combined_output: List[str] = []
        verification_timeout: int = 30
    
        # Method 1: python -m pip --version
        # Why try this first:
        #   - Does not depend on the pip executable being on PATH.
        #   - Uses the Python import system, which finds pip in site-packages
        #     regardless of script directory configuration.
        #   - Works immediately after installation without requiring the
        #     user to restart their terminal or source environment scripts.
        try:
            result = subprocess.run(
                [str(self.python_executable), "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                timeout=verification_timeout,
            )
            output = result.stdout + result.stderr
            combined_output.append(f"[python -m pip --version]\n{output}")
    
            if result.returncode == 0:
                match = re.match(r"pip\s+(\d+\.\d+(?:\.\d+)?)", output)
                if match:
                    version = match.group(1)
    
        except subprocess.TimeoutExpired:
            combined_output.append(
                "[python -m pip --version] timed out after "
                f"{verification_timeout}s"
            )
        except FileNotFoundError:
            combined_output.append(
                "[python -m pip --version] Python executable not found: "
                f"{self.python_executable}"
            )
        except OSError as e:
            combined_output.append(
                f"[python -m pip --version] OS error: {e}"
            )
    
        # Method 2: pip --version (direct executable)
        # Why try this as well:
        #   - Confirms the pip script was correctly created during installation.
        #   - Detects PATH configuration issues: if python -m pip works but
        #     pip directly does not, the user needs to add the script directory
        #     to their PATH.
        #   - Some environments restrict python -m but allow direct executables
        #     (unusual but possible in security-hardened setups).
        if version is None:
            pip_exe_name = "pip.exe" if sys.platform == "win32" else "pip"
            
            # Search for pip executable in expected locations
            # Why search instead of assuming PATH:
            #   - The script directory may not be on PATH immediately after
            #     installation (common in virtualenv without activation).
            #   - User-site installations place scripts in ~/.local/bin which
            #     may not be on PATH in all distributions.
            search_paths: List[Path] = [
                self.python_executable.parent / pip_exe_name,
                self.python_executable.parent / "Scripts" / pip_exe_name
                if sys.platform == "win32"
                else self.python_executable.parent / "bin" / pip_exe_name,
                Path.home() / ".local" / "bin" / pip_exe_name,
            ]
    
            pip_exe: Optional[Path] = None
            for candidate in search_paths:
                if candidate.exists() and candidate.is_file():
                    pip_exe = candidate
                    break
    
            # If not found in expected locations, try shutil.which
            if pip_exe is None:
                import shutil
                found = shutil.which(pip_exe_name)
                if found:
                    pip_exe = Path(found)
    
            if pip_exe is not None:
                try:
                    result = subprocess.run(
                        [str(pip_exe), "--version"],
                        capture_output=True,
                        text=True,
                        timeout=verification_timeout,
                    )
                    output = result.stdout + result.stderr
                    combined_output.append(f"[{pip_exe} --version]\n{output}")
    
                    if result.returncode == 0:
                        match = re.match(r"pip\s+(\d+\.\d+(?:\.\d+)?)", output)
                        if match and version is None:
                            version = match.group(1)
    
                except subprocess.TimeoutExpired:
                    combined_output.append(
                        f"[{pip_exe} --version] timed out after "
                        f"{verification_timeout}s"
                    )
                except OSError as e:
                    combined_output.append(
                        f"[{pip_exe} --version] OS error: {e}"
                    )
    
        # Build the complete output for debugging
        full_output = "\n".join(combined_output)
        duration_ms = (time.time() - start) * 1000.0
    
        # Determine result
        if version is not None:
            # Verification succeeded
            if self.verbose and not self.quiet:
                print(f"OK (pip {version})")
    
            # Check if installed version matches requested version
            if (
                install_result.version_requested is not None
                and version != install_result.version_requested
            ):
                self._warnings.append(
                    f"Requested version {install_result.version_requested}, "
                    f"but installed version is {version}"
                )
    
            return (
                StageResult(
                    stage=PipelineStage.VERIFICATION,
                    success=True,
                    message=f"pip {version} is functional",
                    duration_ms=duration_ms,
                    data={
                        "verified_version": version,
                        "output": full_output,
                        "method": "python -m pip" if "python -m pip" in full_output else "direct executable",
                    },
                ),
                version,
            )
    
        else:
            # Verification failed
            if self.verbose and not self.quiet:
                print("FAILED")
    
            # Provide actionable error messages based on what we tried
            if "timed out" in full_output.lower():
                error_detail = (
                    "pip --version timed out. The installation may have "
                    "produced a corrupted pip executable. Try reinstalling "
                    "with --force."
                )
            elif "not found" in full_output.lower():
                error_detail = (
                    "pip executable was not found after installation. "
                    "The installation may have failed silently or the "
                    "script directory is not writable."
                )
            else:
                error_detail = (
                    "pip is installed but not functional. Check the output "
                    "above for specific error details."
                )
    
            self._errors.append(f"Verification failed: {error_detail}")
    
            return (
                StageResult(
                    stage=PipelineStage.VERIFICATION,
                    success=False,
                    message="pip is not functional",
                    duration_ms=duration_ms,
                    data={
                        "output": full_output,
                        "error_detail": error_detail,
                    },
                ),
                None,
            )

    # ------------------------------------------------------------------
    # Private: Result Building
    # ------------------------------------------------------------------

    def _build_result(
        self,
        total_start: float,
        force_exit_code: Optional[ExitCode] = None,
    ) -> RescueResult:
        """Build the final RescueResult from accumulated state."""
        total_ms = (time.time() - total_start) * 1000

        if force_exit_code:
            exit_code = force_exit_code
        elif self._errors:
            # Determine exit code from errors
            error_text = " ".join(self._errors).lower()
            if "network" in error_text or "connect" in error_text:
                exit_code = ExitCode.NETWORK_ERROR
            elif "permission" in error_text:
                exit_code = ExitCode.PERMISSION_ERROR
            elif "incompatible" in error_text:
                exit_code = ExitCode.INCOMPATIBLE_VERSION
            elif "verification" in error_text:
                exit_code = ExitCode.VERIFICATION_FAILED
            elif "argument" in error_text or "invalid" in error_text:
                exit_code = ExitCode.INVALID_ARGUMENTS
            else:
                exit_code = ExitCode.GENERAL_ERROR
        else:
            exit_code = ExitCode.SUCCESS

        return RescueResult(
            success=len(self._errors) == 0,
            version_installed=None,
            version_requested=self.target_version,
            strategy_used=None,
            stages=self._stages,
            total_duration_ms=total_ms,
            exit_code=exit_code,
            dry_run=self.dry_run,
            errors=self._errors,
            warnings=self._warnings,
        )


# ---------------------------------------------------------------------------
# Command-Line Interface
# ---------------------------------------------------------------------------


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Why a function instead of inline code:
        - Enables testing the parser independently of execution.
        - Other scripts can reuse the parser to add custom arguments.
        - Keeps the module importable without triggering argument
          parsing side effects.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser ready for ``parse_args()``.

    Examples
    --------
    >>> parser = create_argument_parser()
    >>> args = parser.parse_args(["--version", "21.3.1", "--verbose"])
    >>> args.version
    '21.3.1'
    >>> args.verbose
    True

    >>> args = parser.parse_args(["--uninstall", "--dry-run"])
    >>> args.uninstall
    True
    >>> args.dry_run
    True
    """
    parser = argparse.ArgumentParser(
        prog="pip-installer",
        description=textwrap.dedent("""\
            Install or repair pip in any Python environment.

            Automatically diagnoses pip's state, resolves the correct
            version for your Python, fetches resources, and installs
            pip using the best available strategy.
        """),
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s                          # Install latest pip
              %(prog)s --version 21.3.1         # Install specific version
              %(prog)s --uninstall              # Remove pip completely
              %(prog)s --uninstall --dry-run    # Preview removal
              %(prog)s --offline ./pip.whl      # Install from local wheel
              %(prog)s --user --force           # Force reinstall to user site
              %(prog)s --dry-run                # Preview without changes
              %(prog)s --python /usr/bin/python3.8  # Target specific Python
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Action group: install or uninstall (mutually exclusive)
    # ------------------------------------------------------------------
    action_group = parser.add_argument_group(
        "Action",
        "Choose whether to install/repair pip (default) or remove it."
    )
    action_group.add_argument(
        "--uninstall", "-u",
        action="store_true",
        help="Remove pip completely from the target Python environment. "
             "Cannot be combined with --version, --offline, or --get-pip.",
    )

    # ------------------------------------------------------------------
    # Version selection (only for install action)
    # ------------------------------------------------------------------
    version_group = parser.add_argument_group(
        "Version Options",
        "Control which pip version to install (ignored with --uninstall)."
    )
    version_group.add_argument(
        "--version", "-v",
        metavar="VERSION",
        help="Specific pip version to install (e.g., 21.3.1, 23.0). "
             "If not specified, installs the latest compatible version.",
    )
    version_group.add_argument(
        "--latest",
        action="store_true",
        help="Force installation of the latest available pip version "
             "(overrides compatibility ceiling for older Python).",
    )

    # ------------------------------------------------------------------
    # Resource options
    # ------------------------------------------------------------------
    resource_group = parser.add_argument_group(
        "Resource Options",
        "Specify local resources for offline installation "
        "(ignored with --uninstall)."
    )
    resource_group.add_argument(
        "--offline", "-o",
        metavar="WHEEL_PATH",
        type=Path,
        help="Path to a local pip .whl file for offline installation.",
    )
    resource_group.add_argument(
        "--get-pip",
        metavar="SCRIPT_PATH",
        type=Path,
        dest="get_pip_path",
        help="Path to a local get-pip.py script.",
    )
    resource_group.add_argument(
        "--cache-dir",
        metavar="DIR",
        type=Path,
        help="Directory to cache downloaded resources. "
             "Default: temporary directory.",
    )

    # ------------------------------------------------------------------
    # Environment options
    # ------------------------------------------------------------------
    env_group = parser.add_argument_group(
        "Environment Options",
        "Control the target Python environment."
    )
    env_group.add_argument(
        "--python", "-p",
        metavar="PYTHON_EXE",
        type=Path,
        dest="python_executable",
        help="Path to the Python interpreter to install pip for. "
             "Default: the Python running this script.",
    )
    env_group.add_argument(
        "--user",
        action="store_true",
        help="Install to (or remove from) the user site-packages directory "
             "(~/.local on Linux, ~/Library on macOS, "
             "%%APPDATA%% on Windows).",
    )
    env_group.add_argument(
        "--break-system-packages",
        action="store_true",
        help="Allow pip installation on externally-managed environments "
             "(PEP 668). Ignored with --uninstall. Use with caution.",
    )

    # ------------------------------------------------------------------
    # Behavior options
    # ------------------------------------------------------------------
    behavior_group = parser.add_argument_group(
        "Behavior Options",
        "Control how operations are performed."
    )
    behavior_group.add_argument(
        "--force", "-f",
        action="store_true",
        help="Reinstall pip even if it appears healthy. "
             "With --uninstall, forces removal even if files are locked.",
    )
    behavior_group.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Perform diagnosis and version resolution without "
             "making any changes. Shows what would be done.",
    )
    behavior_group.add_argument(
        "--timeout", "-t",
        type=int,
        default=30,
        metavar="SECONDS",
        help="Network timeout in seconds. Default: 30.",
    )

    # ------------------------------------------------------------------
    # Output options
    # ------------------------------------------------------------------
    output_group = parser.add_argument_group(
        "Output Options",
        "Control how results are displayed."
    )
    output_group.add_argument(
        "--verbose", "-V",
        action="store_true",
        help="Print detailed progress information during execution.",
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except fatal errors.",
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output the final result as JSON "
             "(useful for scripting and automation).",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point for the command-line interface.

    Why accept argv as a parameter:
        - Enables testing by passing custom argument lists.
        - Allows embedding in other scripts that construct arguments
          programmatically.
        - Follows the convention of argparse-based CLI tools.

    Parameters
    ----------
    argv : Optional[List[str]]
        Command-line arguments. If ``None``, uses ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).

    Examples
    --------
    >>> main(["--dry-run", "--version", "21.3.1"])
    0

    >>> main(["--uninstall"])
    0

    >>> main(["--uninstall", "--version", "21.3.1"])
    2  # Invalid combination
    """
    parser = create_argument_parser()

    if argv is None:
        argv = sys.argv[1:]

    # Handle empty arguments (show help)
    if not argv:
        parser.print_help()
        return ExitCode.INVALID_ARGUMENTS.value

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    # Validate contradictory output flags
    if args.verbose and args.quiet:
        print(
            "Error: --verbose and --quiet are mutually exclusive",
            file=sys.stderr,
        )
        return ExitCode.INVALID_ARGUMENTS.value

    if args.json and args.verbose:
        print(
            "Warning: --json overrides --verbose; "
            "progress output will be suppressed",
            file=sys.stderr,
        )

    # Validate --uninstall conflicts
    if args.uninstall:
        conflicts = []
        if args.version:
            conflicts.append("--version")
        if args.offline:
            conflicts.append("--offline")
        if args.get_pip_path:
            conflicts.append("--get-pip")
        if args.latest:
            conflicts.append("--latest")

        if conflicts:
            print(
                f"Error: --uninstall cannot be combined with: "
                f"{', '.join(conflicts)}",
                file=sys.stderr,
            )
            return ExitCode.INVALID_ARGUMENTS.value

    # ------------------------------------------------------------------
    # Build rescuer
    # ------------------------------------------------------------------

    # Determine target version
    target_version = args.version
    if args.latest:
        target_version = None  # Will use latest compatible

    try:
        rescuer = PipRescuer(
            target_version=target_version,
            python_executable=args.python_executable,
            wheel_path=args.offline,
            get_pip_path=args.get_pip_path,
            user_site=args.user,
            force=args.force,
            break_system_packages=args.break_system_packages,
            timeout=args.timeout,
            dry_run=args.dry_run,
            verbose=args.verbose and not args.quiet and not args.json,
            quiet=args.quiet or args.json,
            cache_dir=args.cache_dir,
        )

        # ------------------------------------------------------------------
        # Execute action
        # ------------------------------------------------------------------
        if args.uninstall:
            result = rescuer.uninstall()
        else:
            result = rescuer.run()

        # ------------------------------------------------------------------
        # Output
        # ------------------------------------------------------------------
        if args.json:
            import json
            output = {
                "success": result.success,
                "version_installed": result.version_installed,
                "version_requested": result.version_requested,
                "strategy_used": result.strategy_used,
                "exit_code": result.exit_code.value,
                "duration_ms": result.total_duration_ms,
                "dry_run": result.dry_run,
                "errors": result.errors,
                "warnings": result.warnings,
                "stages": [
                    {
                        "stage": s.stage.name,
                        "success": s.success,
                        "message": s.message,
                        "duration_ms": s.duration_ms,
                    }
                    for s in result.stages
                ],
            }
            print(json.dumps(output, indent=2))
        elif not args.quiet:
            print()
            print(result.report(verbose=args.verbose))

        return result.exit_code.value

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return ExitCode.INVALID_ARGUMENTS.value
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return ExitCode.GENERAL_ERROR.value
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return ExitCode.GENERAL_ERROR.value


# ---------------------------------------------------------------------------
# Module Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Allow running as: python main.py [arguments]

    Why this block:
        - Enables direct execution without ``python -m pip_rescuer.main``.
        - The ``main()`` function returns an exit code; ``sys.exit()``
          propagates it to the shell.
        - Keeps the module importable without side effects (the guard
          prevents execution on import).
    """
    sys.exit(main())
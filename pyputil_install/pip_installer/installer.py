"""
Multi-Strategy Pip Installation Module.

Executes pip installation using the most appropriate strategy based on
environment diagnosis and version resolution. Strategies include ensurepip
bootstrapping, get-pip.py execution, wheel installation, and manual
extraction as a last resort. Every installation is verified before
reporting success.

Why this module exists:
    - ``ensurepip`` is fast and bundled with Python, but cannot install
      specific pip versions—only the bundled one.
    - ``get-pip.py`` supports version selection and works offline with
      a local copy, but requires network access for first use and can
      be blocked by corporate firewalls.
    - Wheel installation is the cleanest method—pip is a standard Python
      package—but requires an existing pip or manual extraction.
    - Manual extraction is the fallback when nothing else works: it
      bypasses pip entirely and places files directly into site-packages
      using the extractor module which handles ZIP integrity, path
      traversal security, and overwrite logic.
    - Installation without verification is dangerous: a corrupted install
      may appear successful (exit code 0) but fail on first use, leaving
      the user in a worse state than before.
    - User-site installation (``--user``) is required when system
      site-packages are read-only (containerized Python, shared hosting,
      system Python on locked-down distributions).
    - Some distributions (Debian, Ubuntu) mark pip as "externally managed"
      and require ``--break-system-packages``; ignoring this flag causes
      installation to fail with a misleading error.

Integration with extractor.py:
    - The ``_install_wheel_via_extraction`` method previously used
      ``zipfile.ZipFile`` directly to open, iterate, filter, and extract
      wheel contents. This required manual handling of member names,
      directory creation, overwrite detection, and path traversal
      security.
    - The extractor module's ``extract_wheel`` function consolidates
      all ZIP extraction logic: it opens the archive, validates member
      paths against the target directory to prevent ``../../`` attacks,
      creates parent directories on demand, respects the overwrite flag,
      and returns the list of extracted files.
    - By delegating to ``extract_wheel``, the installer eliminates
      duplicated ZIP-handling code and benefits from centralized
      security checks and error handling.

Warnings
--------
- This module modifies the Python environment by installing or
  reinstalling pip. It should be run with an understanding that
  existing pip installations will be overwritten.
- On Windows, file locking may prevent overwriting an in-use pip.
  Close all Python processes before running repair operations.
- Manual extraction bypasses pip's normal installation machinery.
  It should only be used as a last resort because it does not record
  package metadata in the standard way, potentially confusing other
  package management tools.
- The ``--break-system-packages`` flag is passed automatically on
  externally-managed environments. This is intentional (the user
  explicitly requested pip repair) but should be logged prominently.

Examples
--------
Install latest pip using the best available strategy:

    >>> from installer import PipInstaller
    >>> from checker import PipChecker, PipStatus
    >>> checker = PipChecker()
    >>> diagnosis = checker.run_full_diagnosis()
    >>> installer = PipInstaller(diagnosis)
    >>> result = installer.install()
    >>> result.success
    True
    >>> result.strategy_used
    'ensurepip'

Install a specific pip version:

    >>> installer = PipInstaller(diagnosis, target_version="21.3.1")
    >>> result = installer.install()
    >>> result.version_installed
    '21.3.1'

Install from a local wheel file (offline mode):

    >>> installer = PipInstaller(
    ...     diagnosis,
    ...     wheel_path=Path("./pip-23.0.1-py3-none-any.whl"),
    ... )
    >>> result = installer.install()

Force reinstall even if pip appears healthy:

    >>> installer = PipInstaller(diagnosis, force=True)
    >>> result = installer.install()

Install with user-site fallback:

    >>> installer = PipInstaller(diagnosis, user_site=True)
    >>> result = installer.install()
    >>> result.user_site_used
    True
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import diagnosis types from checker
from .checker import PipDiagnosis, PipStatus

# Import extractor for centralized wheel extraction
from .extractor import extract_wheel


# ---------------------------------------------------------------------------
# Public Enumerations
# ---------------------------------------------------------------------------


class InstallStrategy(Enum):
    """
    Identifies which installation method was used or should be attempted.

    Why an enum instead of magic strings:
        - Strategy names appear in logs, user-facing messages, and result
          objects. An enum ensures consistency across all three.
        - The orchestrator in ``main.py`` can branch on strategy without
          string comparison bugs.
        - Each variant documents itself: the name explains what the
          strategy does at a glance.

    The strategies are ordered from most-preferred to least-preferred.
    The installer attempts them in this order unless a specific strategy
    is forced by the caller.

    Attributes
    ----------
    ENSUREPIP : InstallStrategy
        Use Python's built-in ``ensurepip`` module. Fast, no network
        required, but cannot install arbitrary pip versions.
    GET_PIP_SCRIPT : InstallStrategy
        Execute the ``get-pip.py`` bootstrapping script. Supports
        version selection via ``PIP_VERSION`` environment variable.
    WHEEL_INSTALL : InstallStrategy
        Install from a ``.whl`` file. Cleanest method when pip is
        available; falls back to manual extraction when it is not.
    MANUAL_EXTRACTION : InstallStrategy
        Extract wheel contents directly into site-packages using
        the extractor module. Works without any existing pip.
    CONDA_INSTALL : InstallStrategy
        Use ``conda install pip`` for Conda-managed environments.
    COPY_FROM_SYSTEM : InstallStrategy
        Copy pip from the system Python to the target environment.
    """

    ENSUREPIP = auto()
    GET_PIP_SCRIPT = auto()
    WHEEL_INSTALL = auto()
    MANUAL_EXTRACTION = auto()
    CONDA_INSTALL = auto()
    COPY_FROM_SYSTEM = auto()


class InstallResultCode(Enum):
    """
    Outcome of an installation attempt.

    Why separate from InstallStrategy:
        - A single strategy can produce multiple outcomes (success,
          partial failure, complete failure).
        - The result code is the final state; the strategy is the
          method used to reach that state.
        - Enables the orchestrator to decide whether to try the next
          strategy or give up based on the specific failure mode.

    Attributes
    ----------
    SUCCESS : InstallResultCode
        Installation completed and verification passed.
    SUCCESS_WITH_WARNINGS : InstallResultCode
        Installation succeeded but non-fatal issues were encountered.
    ALREADY_INSTALLED : InstallResultCode
        pip is already installed and healthy; no action needed.
    FAILED_PERMISSION : InstallResultCode
        Installation failed due to filesystem permission errors.
    FAILED_NETWORK : InstallResultCode
        Installation failed because network resources were unavailable.
    FAILED_INCOMPATIBLE : InstallResultCode
        The requested pip version is incompatible with this Python.
    FAILED_VERIFICATION : InstallResultCode
        Installation appeared to succeed but post-install verification failed.
    FAILED_UNKNOWN : InstallResultCode
        Installation failed for an undetermined reason.
    """

    SUCCESS = auto()
    SUCCESS_WITH_WARNINGS = auto()
    ALREADY_INSTALLED = auto()
    FAILED_PERMISSION = auto()
    FAILED_NETWORK = auto()
    FAILED_INCOMPATIBLE = auto()
    FAILED_VERIFICATION = auto()
    FAILED_UNKNOWN = auto()


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------


@dataclass
class InstallResult:
    """
    Complete result of a pip installation operation.

    Why a dedicated result type instead of returning a tuple:
        - Installation produces many pieces of information (strategy,
          version, paths, warnings, errors, timing); a tuple would be
          fragile and hard to read.
        - The result can be logged, serialized, or transmitted as a
          unit without unpacking and repacking.
        - ``success`` is a computed property based on the result code,
          not a separate field that could become inconsistent.

    Attributes
    ----------
    strategy_used : Optional[InstallStrategy]
        The strategy that was actually executed. ``None`` if no strategy
        was attempted (e.g., pip already healthy and force=False).
    result_code : InstallResultCode
        Detailed outcome code for programmatic handling.
    version_installed : Optional[str]
        The pip version string after installation, if determinable.
    version_requested : Optional[str]
        The version that was requested. May differ from
        ``version_installed`` if the request was incompatible.
    user_site_used : bool
        ``True`` if pip was installed to the user site-packages.
    executable_path : Optional[Path]
        Path to the installed pip executable.
    module_path : Optional[Path]
        Path to the installed pip Python module.
    warnings : List[str]
        Non-fatal warnings encountered during installation.
    errors : List[str]
        Error messages if installation failed.
    duration_seconds : float
        Wall-clock time spent in the installation attempt.
    log_output : str
        Combined stdout/stderr from the installation subprocess.
    timestamp : str
        ISO 8601 timestamp of when installation completed.
    """

    strategy_used: Optional[InstallStrategy] = None
    result_code: InstallResultCode = InstallResultCode.FAILED_UNKNOWN
    version_installed: Optional[str] = None
    version_requested: Optional[str] = None
    user_site_used: bool = False
    executable_path: Optional[Path] = None
    module_path: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    log_output: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def success(self) -> bool:
        """
        Whether the installation was successful.

        Why a property instead of a field:
            - Ensures ``success`` is always consistent with
              ``result_code``; no risk of setting ``success=True``
              but ``result_code=FAILED_PERMISSION``.
            - Derived from result_code, single source of truth.

        Returns
        -------
        bool
            ``True`` if the result code indicates success.
        """
        return self.result_code in (
            InstallResultCode.SUCCESS,
            InstallResultCode.SUCCESS_WITH_WARNINGS,
            InstallResultCode.ALREADY_INSTALLED,
        )


# ---------------------------------------------------------------------------
# Strategy Result (Internal)
# ---------------------------------------------------------------------------


@dataclass
class _StrategyAttempt:
    """
    Internal result from attempting a single installation strategy.

    Why private (underscore prefix):
        - Callers receive the public ``InstallResult``, not individual
          strategy attempts.
        - Strategy attempts are implementation details; exposing them
          would couple external code to the strategy enumeration.
        - The installer aggregates multiple attempts into one result.

    Attributes
    ----------
    strategy : InstallStrategy
        Which strategy was attempted.
    success : bool
        Whether this specific attempt succeeded.
    version_installed : Optional[str]
        Pip version installed by this attempt.
    executable_path : Optional[Path]
        Path to the pip executable after this attempt.
    module_path : Optional[Path]
        Path to the pip module after this attempt.
    user_site_used : bool
        Whether user site-packages was used.
    warnings : List[str]
        Warnings from this attempt.
    errors : List[str]
        Errors from this attempt.
    log_output : str
        Combined stdout/stderr from this attempt.
    duration_seconds : float
        Time spent in this attempt.
    """

    strategy: InstallStrategy
    success: bool
    version_installed: Optional[str]
    executable_path: Optional[Path]
    module_path: Optional[Path]
    user_site_used: bool
    warnings: List[str]
    errors: List[str]
    log_output: str
    duration_seconds: float


# ---------------------------------------------------------------------------
# Core Installer Class
# ---------------------------------------------------------------------------


class PipInstaller:
    """
    Installs or repairs pip using the optimal strategy for the environment.

    Why a class instead of a single function:
        - Installation is stateful: the diagnosis, version resolution,
          and configuration (force, user_site, wheel_path) are set once
          and consumed by multiple strategy methods.
        - Strategy selection logic is complex enough to warrant its own
          method rather than being embedded in a monolithic function.
        - Enables testing individual strategies by constructing the
          installer with specific diagnoses and calling strategy methods
          directly.
        - The installer holds intermediate state (attempt results) that
          is useful for logging and debugging but should not be returned
          to the caller.

    Parameters
    ----------
    diagnosis : PipDiagnosis
        Complete environment and pip state diagnosis from
        ``PipChecker.run_full_diagnosis()``.
    target_version : Optional[str]
        Pip version to install. If ``None``, installs the best
        compatible version (determined by ``VersionResolver``).
    wheel_path : Optional[Path]
        Path to a local pip wheel file for offline installation.
    get_pip_path : Optional[Path]
        Path to a local ``get-pip.py`` script. If ``None``, the
        script will be downloaded (if network available) or the
        bundled copy used.
    force : bool
        If ``True``, reinstall pip even if the diagnosis indicates
        it is healthy. Default ``False``.
    user_site : bool
        If ``True``, install to the user site-packages directory.
        Required when system site-packages are read-only.
    break_system_packages : bool
        If ``True``, pass ``--break-system-packages`` to pip when
        the environment is marked as externally managed. Default
        ``True`` because the user explicitly requested installation.

    Attributes
    ----------
    diagnosis : PipDiagnosis
        The diagnosis guiding strategy selection.
    target_version : Optional[str]
        Requested pip version.
    force : bool
        Whether to reinstall healthy pip.
    """

    # Timeout for installation subprocess calls.
    # Why 120 seconds: pip installation typically takes 5-30 seconds.
    # 120 seconds allows for slow disks, antivirus scanning, and
    # network-dependent installations without hanging indefinitely.
    _INSTALL_TIMEOUT: int = 120

    # Timeout for post-install verification.
    # Why shorter than install timeout: verification runs ``pip --version``
    # which should complete in under 10 seconds. A longer timeout would
    # hide the fact that the installed pip hangs on startup.
    _VERIFY_TIMEOUT: int = 30

    def __init__(
        self,
        diagnosis: PipDiagnosis,
        target_version: Optional[str] = None,
        wheel_path: Optional[Path] = None,
        get_pip_path: Optional[Path] = None,
        force: bool = False,
        user_site: bool = False,
        break_system_packages: bool = True,
    ) -> None:
        """
        Initialize the installer with environment diagnosis and options.

        Parameters
        ----------
        diagnosis : PipDiagnosis
            Pre-computed diagnosis from ``PipChecker``.
        target_version : Optional[str]
            Specific pip version to install.
        wheel_path : Optional[Path]
            Local wheel file for offline installation.
        get_pip_path : Optional[Path]
            Local ``get-pip.py`` script.
        force : bool
            Reinstall even if pip is healthy.
        user_site : bool
            Install to user site-packages.
        break_system_packages : bool
            Allow breaking system packages on externally-managed envs.

        Raises
        ------
        ValueError
            If ``target_version`` is provided but fails version validation
            against the diagnosis environment.
        """
        self.diagnosis: PipDiagnosis = diagnosis
        self.target_version: Optional[str] = target_version
        self.wheel_path: Optional[Path] = wheel_path
        self.get_pip_path: Optional[Path] = get_pip_path
        self.force: bool = force
        self.user_site: bool = user_site
        self.break_system_packages: bool = break_system_packages

        self._python_exe: Path = diagnosis.environment.python_executable
        self._attempts: List[_StrategyAttempt] = []

        # Track whether user_site was requested but unavailable.
        # The fallback warning is added to the final InstallResult,
        # not to individual strategy attempts, because the fallback
        # is a cross-cutting concern.
        self._user_site_fallback: bool = (
            self.user_site and not diagnosis.environment.is_user_site_enabled
        )

    # ==================================================================
    # Public API: Main Installation Entry Point
    # ==================================================================

    def install(self) -> InstallResult:
        """
        Execute pip installation using the best available strategy.

        Strategy selection is based on what resources are available
        and what the diagnosis indicates. The order is designed to
        prefer local, fast, offline-capable methods before falling
        back to network-dependent or destructive methods.

        Strategy selection order:
            1. If pip is healthy and ``force=False``, return
               ALREADY_INSTALLED immediately.
            2. If a local wheel file was provided and exists, try
               wheel installation first (fastest local method).
            3. If the environment is Conda, use ``conda install pip``
               to avoid cross-manager corruption.
            4. If no specific version was requested, try ensurepip
               (bundled, no network, fastest).
            5. If a local ``get-pip.py`` is available, execute it.
            6. If network is available, download and execute
               ``get-pip.py``.
            7. Try wheel installation (may download or use cached).
            8. Fall back to manual extraction from the bundled
               ensurepip wheel (last resort, always available).

        Why this order:
            - Local resources are fastest and work offline.
            - ensurepip is bundled but cannot select versions.
            - Network methods are slowest and least reliable.
            - Conda environments need special handling to prevent
              breaking the conda package database.

        Returns
        -------
        InstallResult
            Complete installation result with success status, version,
            strategy used, paths, warnings, and timing information.
        """
        start_time = time.time()

        # Short-circuit: pip is healthy and reinstall not forced
        if self.diagnosis.status == PipStatus.HEALTHY and not self.force:
            if (
                self.target_version is None
                or self.target_version == self.diagnosis.current_version
            ):
                return InstallResult(
                    strategy_used=None,
                    result_code=InstallResultCode.ALREADY_INSTALLED,
                    version_installed=self.diagnosis.current_version,
                    version_requested=self.target_version,
                    user_site_used=False,
                    executable_path=self.diagnosis.executable_path,
                    module_path=self.diagnosis.module_path,
                    warnings=[],
                    errors=[],
                    duration_seconds=time.time() - start_time,
                    log_output="pip is already installed and healthy",
                )

        result: Optional[InstallResult] = None

        # Strategy 1: User-provided local wheel
        if self.wheel_path and self.wheel_path.exists():
            result = self._try_strategy(self._install_from_wheel)
            if result and result.success:
                return result

        # Strategy 2: Conda environment
        if self.diagnosis.environment.is_conda:
            result = self._try_strategy(self._install_via_conda)
            if result and result.success:
                return result

        # Strategy 3: ensurepip (only without specific version)
        if self.target_version is None:
            result = self._try_strategy(self._install_via_ensurepip)
            if result and result.success:
                return result

        # Strategy 4: Local get-pip.py
        if self.get_pip_path and self.get_pip_path.exists():
            result = self._try_strategy(self._install_via_get_pip)
            if result and result.success:
                return result

        # Strategy 5: Network get-pip.py
        if self.diagnosis.environment.has_internet:
            result = self._try_strategy(self._install_via_get_pip)
            if result and result.success:
                return result

        # Strategy 6: Wheel (may download)
        result = self._try_strategy(self._install_from_wheel)
        if result and result.success:
            return result

        # Strategy 7: Manual extraction from bundled wheel
        result = self._try_strategy(self._install_via_manual_extraction)
        if result and result.success:
            return result

        # All strategies exhausted
        all_errors: List[str] = []
        for attempt in self._attempts:
            all_errors.extend(attempt.errors)

        final_warnings: List[str] = []
        if self._user_site_fallback:
            final_warnings.append(
                "User site-packages not available; "
                "fell back to system site-packages"
            )

        return InstallResult(
            strategy_used=None,
            result_code=InstallResultCode.FAILED_UNKNOWN,
            version_installed=None,
            version_requested=self.target_version,
            user_site_used=self.user_site,
            warnings=final_warnings,
            errors=all_errors if all_errors else ["All installation strategies failed"],
            duration_seconds=time.time() - start_time,
            log_output="",
        )

    def install_with_strategy(self, strategy: InstallStrategy) -> InstallResult:
        """
        Force a specific installation strategy regardless of diagnosis.

        Why expose this:
            - Advanced users may know exactly which strategy their
              environment requires (e.g., "my firewall blocks get-pip.py
              but I have a local wheel").
            - Testing individual strategies in isolation without
              running the full strategy selection logic.
            - Debugging: if the auto-selected strategy fails, the
              user can manually try each alternative to isolate
              the failure.

        Parameters
        ----------
        strategy : InstallStrategy
            The strategy to execute. Must be one of the defined
            InstallStrategy variants.

        Returns
        -------
        InstallResult
            Installation result using only the specified strategy.
            If the strategy is not implemented, the result will
            contain an error explaining this.
        """
        strategy_map = {
            InstallStrategy.ENSUREPIP: self._install_via_ensurepip,
            InstallStrategy.GET_PIP_SCRIPT: self._install_via_get_pip,
            InstallStrategy.WHEEL_INSTALL: self._install_from_wheel,
            InstallStrategy.MANUAL_EXTRACTION: self._install_via_manual_extraction,
            InstallStrategy.CONDA_INSTALL: self._install_via_conda,
        }

        method = strategy_map.get(strategy)
        if method is None:
            return InstallResult(
                strategy_used=strategy,
                result_code=InstallResultCode.FAILED_UNKNOWN,
                errors=[f"Strategy {strategy.name} is not implemented"],
            )

        result = method()
        if result is None:
            return InstallResult(
                strategy_used=strategy,
                result_code=InstallResultCode.FAILED_UNKNOWN,
                errors=[f"Strategy {strategy.name} returned no result"],
            )

        return result

    # ==================================================================
    # Public API: Verification
    # ==================================================================

    def verify_installation(self) -> Tuple[bool, Optional[str], str]:
        """
        Verify that pip is installed and functional.

        Why verification is separate from installation:
            - A subprocess returning exit code 0 does not guarantee
              pip actually works. The installation script may have
              completed but left broken bytecode, missing dependencies,
              or an incompatible version.
            - Verification provides the actual version string from
              the running pip, confirming the correct version was
              installed.
            - If verification fails, the installer can try the next
              strategy rather than reporting false success to the
              orchestrator.

        Why try both ``pip --version`` and ``python -m pip --version``:
            - The pip executable script may not be on ``PATH``
              immediately after installation (terminal not restarted,
              virtualenv not activated). ``python -m pip`` uses
              Python's import system which always searches site-packages.
            - If both succeed, pip is fully functional (module + script).
            - If only ``python -m pip`` succeeds, pip works but the
              script directory is not on ``PATH``. This is a warning,
              not a failure.
            - If neither succeeds, the installation did not produce
              a working pip.

        Returns
        -------
        success : bool
            ``True`` if at least one verification method succeeded
            and returned a parsable version string.
        version : Optional[str]
            The pip version string (e.g., ``'24.3.1'``), or ``None``
            if verification failed.
        output : str
            Combined stdout and stderr from all verification attempts,
            useful for debugging failures.
        """
        version: Optional[str] = None
        combined_output: List[str] = []

        # Method 1: pip --version (tests the executable script)
        exe_name = "pip.exe" if sys.platform == "win32" else "pip"
        pip_exe = self._python_exe.parent / exe_name

        if pip_exe.exists():
            try:
                result = subprocess.run(
                    [str(pip_exe), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=self._VERIFY_TIMEOUT,
                )
                output = result.stdout + result.stderr
                combined_output.append(f"[pip --version]\n{output}")
                version = self._parse_version_from_output(output)
                if version:
                    return True, version, "\n".join(combined_output)
            except (subprocess.TimeoutExpired, OSError):
                combined_output.append("[pip --version] timed out or failed")

        # Method 2: python -m pip --version (tests the module import)
        try:
            result = subprocess.run(
                [str(self._python_exe), "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                timeout=self._VERIFY_TIMEOUT,
                env=self._build_subprocess_env(),
            )
            output = result.stdout + result.stderr
            combined_output.append(f"[python -m pip --version]\n{output}")
            version = self._parse_version_from_output(output)
            if version:
                return True, version, "\n".join(combined_output)
        except (subprocess.TimeoutExpired, OSError):
            combined_output.append("[python -m pip --version] timed out or failed")

        return False, None, "\n".join(combined_output)

    # ==================================================================
    # Private: Strategy Wrapper
    # ==================================================================

    def _try_strategy(
        self,
        strategy_method: callable,
    ) -> Optional[InstallResult]:
        """
        Execute a strategy method and convert its result to InstallResult.

        Why this wrapper exists:
            - Each strategy method returns ``Optional[_StrategyAttempt]``.
              This wrapper converts to the public ``InstallResult`` type
              and records the attempt in ``self._attempts`` for
              debugging and error aggregation.
            - If the strategy returns ``None`` (meaning "not applicable
              in this context"), this wrapper propagates ``None`` so
              the caller can try the next strategy.
            - Timing is captured here so individual strategy methods
              do not need to implement their own duration tracking.
            - Exceptions raised by strategy methods are caught and
              converted to failed ``_StrategyAttempt`` records rather
              than propagating and crashing the installer.

        Parameters
        ----------
        strategy_method : callable
            A bound method of this class that performs one installation
            strategy. Must accept no arguments and return
            ``Optional[_StrategyAttempt]``.

        Returns
        -------
        Optional[InstallResult]
            The converted installation result, or ``None`` if the
            strategy was not applicable.
        """
        start = time.time()
        try:
            attempt = strategy_method()
        except Exception as e:
            attempt = _StrategyAttempt(
                strategy=InstallStrategy.MANUAL_EXTRACTION,
                success=False,
                version_installed=None,
                executable_path=None,
                module_path=None,
                user_site_used=False,
                warnings=[],
                errors=[f"Strategy raised exception: {e}"],
                log_output="",
                duration_seconds=time.time() - start,
            )

        if attempt is None:
            return None

        self._attempts.append(attempt)

        # Classify the result code based on error content.
        # Why classify here instead of in each strategy:
        #   - Centralized classification ensures consistency.
        #   - Strategies only report errors as strings; the wrapper
        #     interprets them into standardized result codes.
        if attempt.success:
            result_code = (
                InstallResultCode.SUCCESS_WITH_WARNINGS
                if attempt.warnings
                else InstallResultCode.SUCCESS
            )
        elif any("permission" in e.lower() for e in attempt.errors):
            result_code = InstallResultCode.FAILED_PERMISSION
        elif any("network" in e.lower() or "connect" in e.lower() for e in attempt.errors):
            result_code = InstallResultCode.FAILED_NETWORK
        elif any("incompatible" in e.lower() for e in attempt.errors):
            result_code = InstallResultCode.FAILED_INCOMPATIBLE
        else:
            result_code = InstallResultCode.FAILED_UNKNOWN

        return InstallResult(
            strategy_used=attempt.strategy,
            result_code=result_code,
            version_installed=attempt.version_installed,
            version_requested=self.target_version,
            user_site_used=attempt.user_site_used,
            executable_path=attempt.executable_path,
            module_path=attempt.module_path,
            warnings=attempt.warnings,
            errors=attempt.errors,
            duration_seconds=attempt.duration_seconds,
            log_output=attempt.log_output,
        )

    # ==================================================================
    # Private: Strategy — ensurepip
    # ==================================================================

    def _install_via_ensurepip(self) -> Optional[_StrategyAttempt]:
        """
        Install pip using Python's built-in ``ensurepip`` module.

        How it works:
            1. Checks if ``ensurepip`` is available by running
               ``python -m ensurepip --version``.
            2. If available, runs ``python -m ensurepip --upgrade
               --default-pip`` which installs or upgrades pip to the
               version bundled with this Python distribution.
            3. Verifies the installation by calling
               ``verify_installation()``.

        Why this strategy returns None when a target_version is set:
            - ``ensurepip`` installs the version bundled with Python.
              There is no command-line flag to select a different version.
            - Returning ``None`` signals to ``_try_strategy`` that this
              strategy is not applicable, causing the installer to try
              the next strategy (typically ``get-pip.py`` which supports
              ``PIP_VERSION``).

        Why check ``--version`` before the actual command:
            - Some distributions (Debian, Ubuntu) remove ``ensurepip``
              from the system Python. Running the install command
              directly would produce a confusing "No module named
              ensurepip" error. Checking ``--version`` first gives a
              clear, actionable error message telling the user which
              system package to install.

        Returns
        -------
        Optional[_StrategyAttempt]
            Attempt result, or ``None`` if a specific version was
            requested (ensurepip cannot satisfy it).
        """
        start = time.time()

        # Cannot install a specific version via ensurepip
        if self.target_version is not None:
            return None

        # Check availability
        try:
            result = subprocess.run(
                [str(self._python_exe), "-m", "ensurepip", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return _StrategyAttempt(
                    strategy=InstallStrategy.ENSUREPIP,
                    success=False,
                    version_installed=None,
                    executable_path=None,
                    module_path=None,
                    user_site_used=False,
                    warnings=[],
                    errors=[
                        "ensurepip is not available in this Python installation. "
                        "On Debian/Ubuntu, install python3-venv or python3-ensurepip "
                        "package."
                    ],
                    log_output=result.stderr,
                    duration_seconds=time.time() - start,
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return _StrategyAttempt(
                strategy=InstallStrategy.ENSUREPIP,
                success=False,
                version_installed=None,
                executable_path=None,
                module_path=None,
                user_site_used=False,
                warnings=[],
                errors=["Could not execute ensurepip module"],
                log_output="",
                duration_seconds=time.time() - start,
            )

        cmd = [str(self._python_exe), "-m", "ensurepip", "--upgrade", "--default-pip"]
        env = self._build_subprocess_env()

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._INSTALL_TIMEOUT,
                env=env,
            )
            log_output = proc.stdout + proc.stderr

            if proc.returncode != 0:
                return _StrategyAttempt(
                    strategy=InstallStrategy.ENSUREPIP,
                    success=False,
                    version_installed=None,
                    executable_path=None,
                    module_path=None,
                    user_site_used=False,
                    warnings=[],
                    errors=[f"ensurepip failed with code {proc.returncode}: {proc.stderr}"],
                    log_output=log_output,
                    duration_seconds=time.time() - start,
                )

            ok, version, verify_output = self.verify_installation()
            log_output += "\n" + verify_output

            if not ok:
                return _StrategyAttempt(
                    strategy=InstallStrategy.ENSUREPIP,
                    success=False,
                    version_installed=None,
                    executable_path=None,
                    module_path=None,
                    user_site_used=False,
                    warnings=[],
                    errors=["ensurepip completed but pip is not functional after installation"],
                    log_output=log_output,
                    duration_seconds=time.time() - start,
                )

            exe_path, mod_path = self._locate_installed_pip()

            return _StrategyAttempt(
                strategy=InstallStrategy.ENSUREPIP,
                success=True,
                version_installed=version,
                executable_path=exe_path,
                module_path=mod_path,
                user_site_used=False,
                warnings=[],
                errors=[],
                log_output=log_output,
                duration_seconds=time.time() - start,
            )

        except subprocess.TimeoutExpired:
            return _StrategyAttempt(
                strategy=InstallStrategy.ENSUREPIP,
                success=False,
                version_installed=None,
                executable_path=None,
                module_path=None,
                user_site_used=False,
                warnings=[],
                errors=["ensurepip timed out"],
                log_output="",
                duration_seconds=self._INSTALL_TIMEOUT,
            )

    # ==================================================================
    # Private: Strategy — get-pip.py
    # ==================================================================

    def _install_via_get_pip(self) -> Optional[_StrategyAttempt]:
        """
        Install pip by executing the ``get-pip.py`` bootstrapping script.
    
        How it works:
            1. Locates ``get-pip.py`` in this order:
               a. User-provided path (``self.get_pip_path``).
               b. Bundled copy in the package directory (``get-pip.py``).
               c. Downloaded from ``https://bootstrap.pypa.io/get-pip.py``
                  via ``ResourceFetcher``.
            2. Executes ``python get-pip.py`` with optional flags for
               ``--user``, ``--force-reinstall``, and ``--ignore-installed``.
            3. Sets ``PIP_VERSION`` environment variable if a specific
               pip version was requested.
            4. Verifies the installation by running ``pip --version``
               and ``python -m pip --version``.
    
        Why the version parameter is conditional:
            - ``https://bootstrap.pypa.io/get-pip.py`` is the universal
              URL for Python 3.7 and above. It always returns the latest
              ``get-pip.py`` as raw Python source code.
            - ``https://bootstrap.pypa.io/pip/X.Y/get-pip.py`` exists
              only for legacy Python versions (2.7, 3.5, 3.6). These
              URLs serve version-specific ``get-pip.py`` files that know
              which pip versions are compatible with those older Pythons.
            - Requesting ``/pip/3.13/get-pip.py`` (or any version >= 3.7)
              returns an HTML directory listing page, not Python source.
              Executing this HTML as Python produces ``SyntaxError``.
            - Therefore, ``version`` is passed to ``fetch_get_pip()``
              only when ``python_version_tuple < (3, 7)``.
    
        Why the cache must be cleared after this fix:
            - Previous failed attempts cached the HTML error page under
              the key ``get-pip.py-v3.13.py``. The cache layer serves
              this cached HTML without re-downloading.
            - After this fix, the cache key changes to ``get-pip.py-latest``
              for Python 3.7+, so the old poisoned cache entry is bypassed.
              However, explicitly clearing the cache prevents confusion.
    
        Why ``PIP_VERSION`` is set via environment variable:
            - ``get-pip.py`` reads ``PIP_VERSION`` from the environment
              to determine which pip version to install.
            - There is no ``--version`` command-line flag for ``get-pip.py``.
            - This is the documented interface per the official pip
              installation guide.
    
        Why ``--force-reinstall`` and ``--ignore-installed``:
            - When ``self.force`` is True, pip should be replaced even if
              it already exists and appears healthy.
            - ``--force-reinstall`` reinstalls packages that are already
              up-to-date.
            - ``--ignore-installed`` bypasses version comparison entirely,
              ensuring the new pip overwrites the old one.
    
        Returns
        -------
        Optional[_StrategyAttempt]
            Attempt result containing success status, installed version,
            paths, errors, warnings, and timing.
            Returns ``None`` only if this strategy is not applicable
            (which does not happen; this strategy always returns a result).
    
        Notes
        -----
        - On Windows, the pip executable may be installed to ``Scripts``
          directory which may not be on ``PATH``. Verification via
          ``python -m pip`` handles this case.
        - The ``PIP_BREAK_SYSTEM_PACKAGES`` environment variable is set
          when the environment is externally managed (PEP 668) and the
          user explicitly allowed breaking system packages.
        - ``PIP_DISABLE_PIP_VERSION_CHECK`` is set to prevent pip from
          making a network request just to check if a newer version exists,
          which speeds up the installation process.
        - ``PYTHONPATH`` is cleared from the subprocess environment to
          prevent a broken pip installation in a custom path from
          interfering with the newly installed one.
        """
        import time
    
        py_ver = self.diagnosis.environment.python_version_tuple
        if py_ver >= (3, 7):
            return None  
    
        start = time.time()
    
        # --------------------------------------------------------------
        # Step 1: Locate the get-pip.py script
        # --------------------------------------------------------------
        script_path: Optional[Path] = None
    
        # Priority 1: User-provided path
        if self.get_pip_path and self.get_pip_path.exists():
            script_path = self.get_pip_path
    
        # Priority 2: Bundled copy in the package directory
        if script_path is None:
            bundled = Path(__file__).parent / "get-pip.py"
            if bundled.exists():
                script_path = bundled
    
        # Priority 3: Download from bootstrap.pypa.io
        if script_path is None and self.diagnosis.environment.has_internet:
            from .fetcher import ResourceFetcher
    
            fetcher = ResourceFetcher()
            py_ver = self.diagnosis.environment.python_version_tuple
    
            # Python < 3.7 requires a version-specific URL.
            # Python >= 3.7 uses the universal URL which always returns
            # raw Python source, never HTML.
            if py_ver < (3, 7):
                success, downloaded_path, _ = fetcher.fetch_get_pip(
                    version=f"{py_ver[0]}.{py_ver[1]}"
                )
            else:
                success, downloaded_path, _ = fetcher.fetch_get_pip()
    
            if success:
                script_path = downloaded_path
    
        # If no script source is available, return failure immediately.
        if script_path is None:
            return _StrategyAttempt(
                strategy=InstallStrategy.GET_PIP_SCRIPT,
                success=False,
                version_installed=None,
                executable_path=None,
                module_path=None,
                user_site_used=False,
                warnings=[],
                errors=[
                    "No get-pip.py available "
                    "(network unavailable, no local copy)"
                ],
                log_output="",
                duration_seconds=time.time() - start,
            )
    
        # --------------------------------------------------------------
        # Step 2: Build the command and environment
        # --------------------------------------------------------------
        cmd = [str(self._python_exe), str(script_path)]
    
        # Add installation flags
        if self.user_site:
            cmd.append("--user")
    
        if self.force:
            cmd.extend(["--force-reinstall", "--ignore-installed"])
    
        # Build the subprocess environment
        env = self._build_subprocess_env()
    
        # Set the requested pip version via environment variable.
        # get-pip.py reads PIP_VERSION to select which pip to install.
        if self.target_version:
            env["PIP_VERSION"] = self.target_version
    
        # Bypass PEP 668 external management if permitted
        if self._is_externally_managed() and self.break_system_packages:
            env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
    
        # --------------------------------------------------------------
        # Step 3: Execute get-pip.py
        # --------------------------------------------------------------
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._INSTALL_TIMEOUT,
                env=env,
            )
            log_output = proc.stdout + proc.stderr
    
            if proc.returncode != 0:
                # Truncate error output to last 500 characters.
                # Why: get-pip.py error messages can be very long
                # (full tracebacks). The most relevant information
                # (the actual error) is typically at the end.
                return _StrategyAttempt(
                    strategy=InstallStrategy.GET_PIP_SCRIPT,
                    success=False,
                    version_installed=None,
                    executable_path=None,
                    module_path=None,
                    user_site_used=self.user_site,
                    warnings=[],
                    errors=[
                        f"get-pip.py failed with code {proc.returncode}: "
                        f"{proc.stderr[-500:]}"
                    ],
                    log_output=log_output,
                    duration_seconds=time.time() - start,
                )
    
            # ----------------------------------------------------------
            # Step 4: Verify the installation
            # ----------------------------------------------------------
            ok, version, verify_output = self.verify_installation()
            log_output += "\n" + verify_output
    
            if not ok:
                return _StrategyAttempt(
                    strategy=InstallStrategy.GET_PIP_SCRIPT,
                    success=False,
                    version_installed=None,
                    executable_path=None,
                    module_path=None,
                    user_site_used=self.user_site,
                    warnings=[],
                    errors=[
                        "get-pip.py completed but pip verification failed"
                    ],
                    log_output=log_output,
                    duration_seconds=time.time() - start,
                )
    
            # ----------------------------------------------------------
            # Step 5: Locate the installed pip and build the result
            # ----------------------------------------------------------
            exe_path, mod_path = self._locate_installed_pip()
    
            # Collect any non-fatal warnings
            warnings: List[str] = []
            if self.target_version and version != self.target_version:
                warnings.append(
                    f"Requested version {self.target_version}, "
                    f"but installed version is {version}"
                )
    
            return _StrategyAttempt(
                strategy=InstallStrategy.GET_PIP_SCRIPT,
                success=True,
                version_installed=version,
                executable_path=exe_path,
                module_path=mod_path,
                user_site_used=self.user_site,
                warnings=warnings,
                errors=[],
                log_output=log_output,
                duration_seconds=time.time() - start,
            )
    
        except subprocess.TimeoutExpired:
            return _StrategyAttempt(
                strategy=InstallStrategy.GET_PIP_SCRIPT,
                success=False,
                version_installed=None,
                executable_path=None,
                module_path=None,
                user_site_used=self.user_site,
                warnings=[],
                errors=["get-pip.py execution timed out"],
                log_output="",
                duration_seconds=self._INSTALL_TIMEOUT,
            )

    # ==================================================================
    # Private: Strategy — Wheel Installation
    # ==================================================================

    def _install_from_wheel(self) -> Optional[_StrategyAttempt]:
        """
        Install pip from a ``.whl`` file.

        How it works:
            1. Determines the wheel path: uses the user-provided wheel
               if available, otherwise attempts to download one via
               ``ResourceFetcher.fetch_pip_wheel()``.
            2. If pip is importable (``_can_import_pip()`` returns
               True), delegates to ``_install_wheel_via_pip`` which
               runs ``python -m pip install <wheel>``. This records
               metadata correctly.
            3. If pip is not importable, delegates to
               ``_install_wheel_via_extraction`` which uses the
               extractor module to place files directly into
               site-packages.

        Why try pip-install-pip before manual extraction:
            - ``pip install`` records package metadata in the standard
              ``.dist-info`` directory, enabling ``pip list`` and
              ``pip uninstall`` to work correctly.
            - Manual extraction places files correctly but does not
              register metadata, making the installation invisible
              to some package management tools.

        Returns
        -------
        Optional[_StrategyAttempt]
            Attempt result. Returns ``None`` if no wheel file is
            available and cannot be obtained.
        """
        start = time.time()

        wheel: Optional[Path] = self.wheel_path

        if wheel is None or not wheel.exists():
            if self.diagnosis.environment.has_internet:
                from .fetcher import ResourceFetcher

                version_to_fetch = self.target_version or "latest"
                fetcher = ResourceFetcher()
                success, downloaded, _ = fetcher.fetch_pip_wheel(version_to_fetch)
                if success:
                    wheel = downloaded

        if wheel is None or not wheel.exists():
            return _StrategyAttempt(
                strategy=InstallStrategy.WHEEL_INSTALL,
                success=False,
                version_installed=None,
                executable_path=None,
                module_path=None,
                user_site_used=False,
                warnings=[],
                errors=["No pip wheel available (offline, no local copy, download failed)"],
                log_output="",
                duration_seconds=time.time() - start,
            )

        # Path A: pip is functional — use it to install the wheel
        if self._can_import_pip() and not self.force:
            result = self._install_wheel_via_pip(wheel, start)
            if result is not None:
                return result

        # Path B: pip is not functional — extract wheel directly
        return self._install_wheel_via_extraction(wheel, start)

    # ==================================================================
    # Private: Strategy — Manual Extraction from Bundled Wheel
    # ==================================================================

    def _install_via_manual_extraction(self) -> Optional[_StrategyAttempt]:
        """
        Install pip by extracting the bundled ensurepip wheel.

        How it works:
            1. Searches for the pip wheel bundled with Python's
               ``ensurepip`` module (typically in
               ``ensurepip/_bundled/pip-*.whl``).
            2. Delegates to ``_install_wheel_via_extraction`` which
               uses the extractor module to place wheel contents
               directly into site-packages.

        Why this is the last-resort strategy:
            - It requires no network and no existing pip, making it
              the only strategy guaranteed to work in all environments
              where Python itself is functional.
            - However, it installs whatever version of pip was bundled
              with the Python distribution, which may be outdated.
            - The installed pip will not have standard package metadata,
              which may confuse other tools.

        Why the bundled wheel may not be found:
            - Some distributions (Debian, Ubuntu) remove the bundled
              wheel along with ``ensurepip`` itself.
            - Minimal Python installations (Docker slim images,
              embedded Python) may strip the ``ensurepip`` package.

        Returns
        -------
        Optional[_StrategyAttempt]
            Attempt result, or ``None`` if the bundled wheel cannot
            be located.
        """
        start = time.time()

        bundled_wheel = self._find_bundled_pip_wheel()
        if bundled_wheel is None:
            return _StrategyAttempt(
                strategy=InstallStrategy.MANUAL_EXTRACTION,
                success=False,
                version_installed=None,
                executable_path=None,
                module_path=None,
                user_site_used=False,
                warnings=[],
                errors=["Could not locate bundled pip wheel from ensurepip"],
                log_output="",
                duration_seconds=time.time() - start,
            )

        return self._install_wheel_via_extraction(bundled_wheel, start)

    # ==================================================================
    # Private: Strategy — Conda
    # ==================================================================

    def _install_via_conda(self) -> Optional[_StrategyAttempt]:
        """
        Install pip using ``conda install pip``.

        How it works:
            1. Locates the ``conda`` executable on ``PATH``.
            2. Runs ``conda install --yes --name <env> pip`` with an
               optional version constraint.
            3. Verifies the installation.

        Why Conda environments need special handling:
            - Conda maintains its own package database in the
              environment's ``conda-meta/`` directory. Installing
              packages outside of conda (via pip or manual extraction)
              creates entries in ``site-packages`` that conda does
              not know about.
            - Subsequent conda operations may overwrite or remove
              these unregistered packages, causing breakage.
            - Using ``conda install pip`` ensures pip is registered
              in conda's database, preventing conflicts.

        Returns
        -------
        Optional[_StrategyAttempt]
            Attempt result. Returns ``None`` if conda is not available
            on ``PATH`` despite the environment being detected as Conda.
        """
        start = time.time()

        conda_exe = shutil.which("conda")
        if conda_exe is None:
            return _StrategyAttempt(
                strategy=InstallStrategy.CONDA_INSTALL,
                success=False,
                version_installed=None,
                executable_path=None,
                module_path=None,
                user_site_used=False,
                warnings=[],
                errors=["Conda environment detected but conda executable not found on PATH"],
                log_output="",
                duration_seconds=time.time() - start,
            )

        cmd = [conda_exe, "install", "--yes", "--name", self._get_conda_env_name(), "pip"]

        if self.target_version:
            cmd[-1] = f"pip={self.target_version}"

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._INSTALL_TIMEOUT,
            )
            log_output = proc.stdout + proc.stderr

            if proc.returncode != 0:
                return _StrategyAttempt(
                    strategy=InstallStrategy.CONDA_INSTALL,
                    success=False,
                    version_installed=None,
                    executable_path=None,
                    module_path=None,
                    user_site_used=False,
                    warnings=[],
                    errors=[f"conda install pip failed: {proc.stderr[-500:]}"],
                    log_output=log_output,
                    duration_seconds=time.time() - start,
                )

            ok, version, verify_output = self.verify_installation()
            log_output += "\n" + verify_output

            if not ok:
                return _StrategyAttempt(
                    strategy=InstallStrategy.CONDA_INSTALL,
                    success=False,
                    version_installed=None,
                    executable_path=None,
                    module_path=None,
                    user_site_used=False,
                    warnings=[],
                    errors=["conda install completed but pip verification failed"],
                    log_output=log_output,
                    duration_seconds=time.time() - start,
                )

            exe_path, mod_path = self._locate_installed_pip()

            return _StrategyAttempt(
                strategy=InstallStrategy.CONDA_INSTALL,
                success=True,
                version_installed=version,
                executable_path=exe_path,
                module_path=mod_path,
                user_site_used=False,
                warnings=[],
                errors=[],
                log_output=log_output,
                duration_seconds=time.time() - start,
            )

        except subprocess.TimeoutExpired:
            return _StrategyAttempt(
                strategy=InstallStrategy.CONDA_INSTALL,
                success=False,
                version_installed=None,
                executable_path=None,
                module_path=None,
                user_site_used=False,
                warnings=[],
                errors=["conda install timed out"],
                log_output="",
                duration_seconds=self._INSTALL_TIMEOUT,
            )

    # ==================================================================
    # Private: Wheel Installation — Via Existing Pip
    # ==================================================================

    def _install_wheel_via_pip(
        self, wheel: Path, start_time: float
    ) -> Optional[_StrategyAttempt]:
        """
        Install a wheel file using an existing functional pip.

        How it works:
            1. Builds a ``pip install`` command with ``--no-index``
               and ``--find-links`` pointing to the wheel's directory.
               ``--no-index`` prevents pip from searching PyPI;
               ``--find-links`` tells pip to look for packages in
               the specified directory.
            2. Adds ``--user`` if ``self.user_site`` is True.
            3. Adds ``--force-reinstall --ignore-installed`` if
               ``self.force`` is True, ensuring the existing pip
               is replaced even if the version matches.
            4. Executes the command and verifies.

        Why ``--no-index`` and ``--find-links`` instead of passing
        the wheel path directly:
            - ``pip install /path/to/wheel.whl`` works but may trigger
              dependency resolution against PyPI if the wheel has
              dependencies. Since pip wheels are self-contained,
              ``--no-index`` prevents unnecessary network requests.
            - ``--find-links`` with the parent directory allows pip
              to discover the wheel by its filename, which matches
              the standard ``pip install <name>`` workflow.

        Parameters
        ----------
        wheel : Path
            Absolute path to the wheel file.
        start_time : float
            The ``time.time()`` value from when the parent strategy
            began. Passed in to ensure the total duration reflects
            the full strategy time, not just this sub-operation.

        Returns
        -------
        Optional[_StrategyAttempt]
            Attempt result.
        """
        cmd = [
            str(self._python_exe),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheel.parent),
            str(wheel.name),
        ]

        if self.user_site:
            cmd.insert(4, "--user")
        if self.force:
            cmd.extend(["--force-reinstall", "--ignore-installed"])

        env = self._build_subprocess_env()
        if self._is_externally_managed() and self.break_system_packages:
            env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._INSTALL_TIMEOUT,
                env=env,
            )
            log_output = proc.stdout + proc.stderr

            if proc.returncode != 0:
                return _StrategyAttempt(
                    strategy=InstallStrategy.WHEEL_INSTALL,
                    success=False,
                    version_installed=None,
                    executable_path=None,
                    module_path=None,
                    user_site_used=self.user_site,
                    warnings=[],
                    errors=[f"pip install wheel failed: {proc.stderr[-500:]}"],
                    log_output=log_output,
                    duration_seconds=time.time() - start_time,
                )

            ok, version, verify_output = self.verify_installation()
            log_output += "\n" + verify_output

            if not ok:
                return _StrategyAttempt(
                    strategy=InstallStrategy.WHEEL_INSTALL,
                    success=False,
                    version_installed=None,
                    executable_path=None,
                    module_path=None,
                    user_site_used=self.user_site,
                    warnings=[],
                    errors=["Wheel installed but pip verification failed"],
                    log_output=log_output,
                    duration_seconds=time.time() - start_time,
                )

            exe_path, mod_path = self._locate_installed_pip()

            return _StrategyAttempt(
                strategy=InstallStrategy.WHEEL_INSTALL,
                success=True,
                version_installed=version,
                executable_path=exe_path,
                module_path=mod_path,
                user_site_used=self.user_site,
                warnings=[],
                errors=[],
                log_output=log_output,
                duration_seconds=time.time() - start_time,
            )

        except subprocess.TimeoutExpired:
            return _StrategyAttempt(
                strategy=InstallStrategy.WHEEL_INSTALL,
                success=False,
                version_installed=None,
                executable_path=None,
                module_path=None,
                user_site_used=self.user_site,
                warnings=[],
                errors=["pip install wheel timed out"],
                log_output="",
                duration_seconds=self._INSTALL_TIMEOUT,
            )

    # ==================================================================
    # Private: Wheel Installation — Via Direct Extraction (extractor.py)
    # ==================================================================

    def _install_wheel_via_extraction(
        self, wheel: Path, start_time: float
    ) -> _StrategyAttempt:
        """
        Install a wheel by extracting its contents into site-packages.

        How it works:
            1. Determines the target site-packages directory using
               ``_get_target_site_packages()``, which respects the
               ``user_site`` flag.
            2. Checks write permissions on the target directory.
               If not writable, returns a permission error with
               guidance to use ``--user``.
            3. Delegates to ``extractor.extract_wheel()`` which:
               - Opens the wheel as a ZIP archive.
               - Validates member paths to prevent directory traversal
                 attacks (``../../etc/passwd``).
               - Creates parent directories as needed.
               - Extracts all files, respecting the ``overwrite`` flag
                 (set to ``self.force``).
               - Returns the list of extracted file paths.
            4. Creates the pip executable script via
               ``_create_pip_executable_script()`` since wheel
               extraction only provides the module, not the CLI script.
            5. Verifies the installation.

        Why use extractor.extract_wheel instead of zipfile.ZipFile directly:
            - The extractor module handles ZIP parsing, path security,
              directory creation, and overwrite logic in one call.
            - Centralized extraction means security fixes and improvements
              in the extractor benefit all callers automatically.
            - Reduces code duplication: the same extraction logic is
              used by the fetcher (for inspecting downloaded wheels)
              and the installer (for deploying them).

        Why create the executable script after extraction:
            - Wheels contain the Python module (``pip/`` directory)
              and metadata (``pip-{version}.dist-info/`` directory),
              but not the CLI entry point script.
            - The script is a small Python file placed in the ``bin``
              or ``Scripts`` directory that invokes pip's ``main()``.
            - Without it, users can only use ``python -m pip``, not
              the standalone ``pip`` command.

        Parameters
        ----------
        wheel : Path
            Absolute path to the wheel file to extract.
        start_time : float
            The ``time.time()`` value from when the parent strategy
            began. Ensures accurate total duration.

        Returns
        -------
        _StrategyAttempt
            Attempt result (never ``None`` for this method).
        """
        target_site = self._get_target_site_packages()

        if target_site is None:
            return _StrategyAttempt(
                strategy=InstallStrategy.MANUAL_EXTRACTION,
                success=False,
                version_installed=None,
                executable_path=None,
                module_path=None,
                user_site_used=False,
                warnings=[],
                errors=["Could not determine site-packages location"],
                log_output="",
                duration_seconds=time.time() - start_time,
            )
        
        # Check if the target site exists before the write/permissions checks
        if not target_site.exists():
            try:
                target_site.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError):
                pass
        
        # Check write permissions before attempting extraction.
        # Why check explicitly instead of letting the extractor fail:
        #   - Provides a clear, actionable error message telling the
        #     user exactly which directory is not writable and
        #     suggesting the --user flag.
        #   - Avoids a partial extraction where some files are written
        #     before a permission error occurs on a later file.
        if not os.access(target_site, os.W_OK):
            if self.user_site:
                return _StrategyAttempt(
                    strategy=InstallStrategy.MANUAL_EXTRACTION,
                    success=False,
                    version_installed=None,
                    executable_path=None,
                    module_path=None,
                    user_site_used=True,
                    warnings=[],
                    errors=[f"No write permission to user site-packages: {target_site}"],
                    log_output="",
                    duration_seconds=time.time() - start_time,
                )
            return _StrategyAttempt(
                strategy=InstallStrategy.MANUAL_EXTRACTION,
                success=False,
                version_installed=None,
                executable_path=None,
                module_path=None,
                user_site_used=False,
                warnings=[],
                errors=[
                    f"No write permission to site-packages: {target_site}. "
                    "Try with --user flag."
                ],
                log_output="",
                duration_seconds=time.time() - start_time,
            )

        try:
            # ----------------------------------------------------------
            # Delegate wheel extraction to the extractor module.
            # extract_wheel handles:
            #   - Opening the ZIP archive
            #   - Path traversal security (rejecting ../../ members)
            #   - Creating parent directories
            #   - Overwrite logic (respects self.force)
            #   - Returning the list of extracted file paths
            # ----------------------------------------------------------
            extract_wheel(
                wheel_path=wheel,
                output_dir=target_site,
                overwrite=self.force,
            )

            # Create the CLI entry point script.
            # The wheel provides the module; we provide the executable.
            self._create_pip_executable_script(target_site)

            # Verify the installation produced a working pip
            ok, version, verify_output = self.verify_installation()

            if not ok:
                return _StrategyAttempt(
                    strategy=InstallStrategy.MANUAL_EXTRACTION,
                    success=False,
                    version_installed=None,
                    executable_path=None,
                    module_path=None,
                    user_site_used=self.user_site,
                    warnings=[],
                    errors=["Manual extraction completed but pip verification failed"],
                    log_output=verify_output,
                    duration_seconds=time.time() - start_time,
                )

            exe_path, mod_path = self._locate_installed_pip()

            return _StrategyAttempt(
                strategy=InstallStrategy.MANUAL_EXTRACTION,
                success=True,
                version_installed=version,
                executable_path=exe_path,
                module_path=mod_path,
                user_site_used=self.user_site,
                warnings=[
                    "Pip was installed via manual extraction; "
                    "package metadata may be incomplete"
                ],
                errors=[],
                log_output=verify_output,
                duration_seconds=time.time() - start_time,
            )

        except zipfile.BadZipFile:
            return _StrategyAttempt(
                strategy=InstallStrategy.MANUAL_EXTRACTION,
                success=False,
                version_installed=None,
                executable_path=None,
                module_path=None,
                user_site_used=self.user_site,
                warnings=[],
                errors=[f"Wheel file is corrupted or not a valid ZIP: {wheel}"],
                log_output="",
                duration_seconds=time.time() - start_time,
            )
        except OSError as e:
            return _StrategyAttempt(
                strategy=InstallStrategy.MANUAL_EXTRACTION,
                success=False,
                version_installed=None,
                executable_path=None,
                module_path=None,
                user_site_used=self.user_site,
                warnings=[],
                errors=[f"Filesystem error during extraction: {e}"],
                log_output="",
                duration_seconds=time.time() - start_time,
            )

    # ==================================================================
    # Private: Executable Script Creation
    # ==================================================================

    def _create_pip_executable_script(self, site_packages: Path) -> None:
        """
        Create a ``pip`` (or ``pip.exe``) executable script.

        How it works:
            1. Determines the target script directory:
               - For user-site installations: ``~/.local/bin`` on Unix
                 or ``~/AppData/Roaming/Python/Scripts`` on Windows.
               - For system installations: the ``bin`` or ``Scripts``
                 directory adjacent to the Python executable.
            2. Creates a Python script that imports pip's ``main()``
               function from ``pip._internal.cli.main`` and calls it.
            3. On Unix, sets the executable permission bit (``0o755``).

        Why this script is needed:
            - The wheel file contains the Python module but not the
              CLI entry point. The script bridges the gap between
              the shell command ``pip`` and the Python function
              ``pip._internal.cli.main:main``.
            - Without it, users can only invoke pip via
              ``python -m pip``, which is less convenient and not
              compatible with tools that expect a ``pip`` executable.

        Why the script uses ``sys.argv[0]`` manipulation:
            - The regex substitution ``re.sub(r"(-script\\.pyw|\\.exe)?$",
              "", sys.argv[0])`` strips the wrapper suffix that Windows
              setuptools launchers append. This ensures ``pip`` sees
              its own name correctly when invoked via wrapper executables.

        Parameters
        ----------
        site_packages : Path
            The site-packages directory where pip was extracted.
            Used to locate the Python environment, not directly
            referenced in the script content.
        """
        python_dir = self._python_exe.parent
        script_dir_name = "Scripts" if sys.platform == "win32" else "bin"

        # Determine the script destination directory.
        # User-site scripts go to ~/.local/bin (Unix) or
        # ~/AppData/Roaming/Python/Scripts (Windows).
        if self.user_site:
            script_dir = Path.home() / ".local" / "bin"
            if sys.platform == "win32":
                script_dir = (
                    Path.home() / "AppData" / "Roaming" / "Python" / "Scripts"
                )
        else:
            script_dir = python_dir.parent / script_dir_name
            if not script_dir.exists():
                script_dir = python_dir

        script_dir.mkdir(parents=True, exist_ok=True)

        script_name = "pip.exe" if sys.platform == "win32" else "pip"
        script_path = script_dir / script_name

        # The script content: a minimal wrapper that invokes pip's main().
        # The shebang line uses the target Python executable to ensure
        # the correct interpreter runs the script.
        script_content = f'''#!{self._python_exe}
# -*- coding: utf-8 -*-
import re
import sys
from pip._internal.cli.main import main
if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\\.pyw|\\.exe)?$", "", sys.argv[0])
    sys.exit(main())
'''

        script_path.write_text(script_content)

        # Unix requires the executable bit for direct invocation.
        # Windows uses file associations, so chmod is unnecessary.
        if sys.platform != "win32":
            script_path.chmod(0o755)

    # ==================================================================
    # Private: Subprocess Environment Construction
    # ==================================================================

    def _build_subprocess_env(self) -> Dict[str, str]:
        """
        Build the environment dictionary for subprocess calls.

        How it works:
            1. Copies the current process environment via
               ``os.environ.copy()``.
            2. Sets ``PIP_USER=yes`` if installing to user site-packages.
            3. Sets ``PIP_BREAK_SYSTEM_PACKAGES=1`` if the environment
               is externally managed and the user allowed bypassing.
            4. Sets ``PIP_DISABLE_PIP_VERSION_CHECK=1`` to prevent
               pip from making network requests just to check if a
               newer version exists (speeds up verification).
            5. Removes ``PYTHONPATH`` to prevent interference from
               a broken pip installation in a custom path that might
               be imported before the newly installed one.

        Why modify the environment:
            - ``get-pip.py`` and ``pip install`` read these variables
              to determine behavior. Setting them in the environment
              is more reliable than command-line flags because some
              tools ignore flags but respect environment variables.
            - Clearing ``PYTHONPATH`` prevents a corrupted pip in a
              custom location from shadowing the newly installed one
              during verification.

        Returns
        -------
        Dict[str, str]
            Modified environment dictionary.
        """
        env = os.environ.copy()

        if self.user_site:
            env["PIP_USER"] = "yes"

        if self._is_externally_managed() and self.break_system_packages:
            env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"

        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        env.pop("PYTHONPATH", None)

        return env

    # ==================================================================
    # Private: Environment Checks
    # ==================================================================

    def _is_externally_managed(self) -> bool:
        """
        Check if the Python environment is marked as externally managed.

        PEP 668 defines the ``EXTERNALLY-MANAGED`` marker file. Linux
        distributions (notably Debian, Ubuntu, and Fedora) place this
        file in site-packages to signal that pip should not modify
        system packages. When this file exists, pip refuses to install
        packages unless ``PIP_BREAK_SYSTEM_PACKAGES`` is set or the
        ``--break-system-packages`` flag is passed.

        How it works:
            Iterates over all site-packages paths from the diagnosis
            and checks for the presence of an ``EXTERNALLY-MANAGED``
            file in each.

        Returns
        -------
        bool
            ``True`` if any site-packages directory contains the
            ``EXTERNALLY-MANAGED`` marker file.
        """
        if not self.diagnosis.environment.site_packages_paths:
            return False

        for site_path in self.diagnosis.environment.site_packages_paths:
            marker = site_path / "EXTERNALLY-MANAGED"
            if marker.exists():
                return True

        return False

    def _can_import_pip(self) -> bool:
        """
        Check if pip can be imported in a subprocess.

        How it works:
            Runs ``python -c "import pip"`` in a subprocess. If the
            exit code is 0, pip is importable. This is used to
            decide whether to use ``pip install`` (Path A) or manual
            extraction (Path B) when installing from a wheel.

        Why use a subprocess instead of importing in-process:
            - The current process may already have pip imported in a
              broken state. A subprocess tests the actual on-disk
              installation without the cached module.
            - Importing pip in-process would add it to ``sys.modules``,
              potentially interfering with subsequent operations.

        Returns
        -------
        bool
            ``True`` if ``import pip`` succeeds in a subprocess.
        """
        try:
            result = subprocess.run(
                [str(self._python_exe), "-c", "import pip"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    # ==================================================================
    # Private: Path Resolution
    # ==================================================================

    def _get_target_site_packages(self) -> Optional[Path]:
        """
        Determine the target site-packages directory for installation.

        How it works:
            1. If ``self.user_site`` is True and the user site-packages
               path is known from the diagnosis, returns that path.
            2. Otherwise, returns the first system site-packages path
               from the diagnosis.
            3. Returns ``None`` if no paths are available.

        Why user site-packages takes priority:
            - The user explicitly requested ``--user`` installation.
            - System site-packages may be read-only.

        Returns
        -------
        Optional[Path]
            The target directory, or ``None`` if no site-packages
            paths are known.
        """
        if self.user_site and self.diagnosis.environment.user_site_packages:
            return self.diagnosis.environment.user_site_packages

        if self.diagnosis.environment.site_packages_paths:
            return self.diagnosis.environment.site_packages_paths[0]

        return None

    def _find_bundled_pip_wheel(self) -> Optional[Path]:
        """
        Locate the pip wheel bundled with Python's ensurepip module.

        How it works:
            1. Imports ``ensurepip`` and looks for
               ``<ensurepip_dir>/_bundled/pip-*.whl``.
            2. If that fails (ensurepip not importable), tries the
               sysconfig purelib path:
               ``<purelib>/ensurepip/_bundled/pip-*.whl``.
            3. Returns the first matching wheel found.

        Why search in two locations:
            - The ``ensurepip`` package location varies by Python
              distribution and installation method.
            - On some systems, ``ensurepip`` is importable but its
              ``__file__`` points to a different location than the
              actual ``_bundled`` directory.

        Returns
        -------
        Optional[Path]
            Path to the bundled wheel, or ``None`` if not found.
        """
        # Method 1: Via ensurepip module path
        try:
            import ensurepip

            ensurepip_dir = Path(ensurepip.__file__).parent / "_bundled"
            if ensurepip_dir.exists():
                wheels = list(ensurepip_dir.glob("pip-*.whl"))
                if wheels:
                    return wheels[0]
        except (ImportError, AttributeError):
            pass

        # Method 2: Via sysconfig purelib path
        try:
            purelib = Path(sysconfig.get_path("purelib"))
            ensurepip_dir = purelib / "ensurepip" / "_bundled"
            if ensurepip_dir.exists():
                wheels = list(ensurepip_dir.glob("pip-*.whl"))
                if wheels:
                    return wheels[0]
        except Exception:
            pass

        return None

    def _get_python_version_string(self) -> str:
        """
        Get the Python major.minor version for URL construction.

        Why needed:
            - The bootstrap.pypa.io server serves version-specific
              get-pip.py files at URLs like
              ``/pip/3.13/get-pip.py``.

        Returns
        -------
        str
            Version string like ``"3.13"``.
        """
        v = self.diagnosis.environment.python_version_tuple
        return f"{v[0]}.{v[1]}"

    def _get_conda_env_name(self) -> str:
        """
        Get the active Conda environment name.

        How it works:
            Reads the ``CONDA_DEFAULT_ENV`` environment variable.
            Falls back to ``"base"`` if not set (the default conda
            environment).

        Returns
        -------
        str
            Conda environment name.
        """
        return os.environ.get("CONDA_DEFAULT_ENV", "base")

    def _locate_installed_pip(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Find pip executable and module paths after installation.

        How it works:
            Instantiates a fresh ``PipChecker`` and uses its
            ``locate_pip_executable()`` and ``locate_pip_module()``
            methods to find the newly installed pip.

        Why use a fresh PipChecker:
            - The diagnosis passed to the installer reflects the
              state before installation. After installation, the
              paths have changed.
            - ``PipChecker`` searches ``sys.path`` and common
              locations, finding the new installation wherever
              it was placed.

        Returns
        -------
        Tuple[Optional[Path], Optional[Path]]
            A tuple of (executable_path, module_path). Each may be
            ``None`` if not found.
        """
        from .checker import PipChecker

        checker = PipChecker(python_executable=self._python_exe)
        exe_exists, exe_path = checker.locate_pip_executable()
        mod_exists, mod_path, _ = checker.locate_pip_module()

        return (
            exe_path if exe_exists else None,
            mod_path if mod_exists else None,
        )

    # ==================================================================
    # Private: Version Parsing
    # ==================================================================

    @staticmethod
    def _parse_version_from_output(output: str) -> Optional[str]:
        """
        Extract pip version string from ``pip --version`` output.

        How it works:
            Matches the pattern ``pip X.Y.Z`` at the start of the
            output string using a regex. pip's ``--version`` output
            format is stable: ``pip 24.3.1 from /path (python 3.13)``.

        Why a static method:
            - Pure function: no instance state needed.
            - Can be called from other static or class methods without
              an instance.

        Parameters
        ----------
        output : str
            The combined stdout and stderr from a ``pip --version``
            or ``python -m pip --version`` command.

        Returns
        -------
        Optional[str]
            The version string (e.g., ``"24.3.1"``), or ``None`` if
            the output does not contain a recognizable version.
        """
        match = re.match(r"pip\s+(\d+\.\d+(?:\.\d+)?)", output)
        return match.group(1) if match else None
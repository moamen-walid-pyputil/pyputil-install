"""
Python Runner Module
=====================

Executes Python scripts and commands using specific Python installations
managed by :class:`PythonVersionManager`. Provides subprocess execution,
environment isolation, pip integration, and cross-version script running.

All operations use :mod:`subprocess` with strict security controls.
No third-party dependencies.

Security
--------
- All subprocess calls use absolute paths to the target Python
  executable. The system ``PATH`` is never searched implicitly.
- Environment variables are explicitly constructed for each invocation.
  The calling process's environment is not leaked unless explicitly
  requested via *inherit_env*.
- Command arguments are passed as lists to prevent shell injection.
  ``shell=True`` is **never** used.
- Standard streams are captured by default; the caller must opt-in
  to inheriting the parent's terminal.
- Subprocess timeout is enforced on every call. Default 300 seconds
  prevents runaway processes.
- On timeout, the child process is terminated (``SIGTERM``) followed
  by forceful kill (``SIGKILL``) after a grace period.
- Return code checking is strict: non-zero exits raise
  :class:`RunnerError` by default unless explicitly allowed.

Usage
-----
.. code-block:: python

    from pathlib import Path
    from runner import PythonRunner

    runner = PythonRunner(python_bin=Path("/opt/python/bin/python3"))

    # Run a script
    result = runner.run_script(
        script=Path("my_script.py"),
        args=["--verbose", "--output", "result.json"],
    )
    print(result.stdout)

    # Run inline code
    result = runner.run_code("print('Hello from isolated Python')")

    # Install a package
    runner.pip_install(["requests", "numpy==1.24.0"])

    # Get version info
    info = runner.get_version_info()
    print(f"Python {info['version']} on {info['platform']}")

Warnings
--------
- :meth:`PythonRunner.run_script` and :meth:`PythonRunner.run_code`
  block until the subprocess completes or times out.
- The *timeout* parameter applies to the entire subprocess lifetime,
  not to individual I/O operations.
- ``pip`` operations modify the target Python's ``site-packages``.
  Ensure the installation directory is writable.
- On Windows, the ``python`` executable must be a real ``.exe`` file.
  Batch-file shims are not supported.
- Environment variable values containing newlines or null bytes are
  rejected.

Notes
-----
- All subprocess I/O uses ``PIPE`` by default. Large outputs are
  buffered in memory; use file redirection for multi-gigabyte outputs.
- ``pip`` is invoked as ``[python, "-m", "pip"]`` to ensure the
  correct pip is used for the target interpreter.
- Virtual environment creation uses ``venv`` from the standard
  library, which is always available in Python >= 3.3.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default subprocess timeout in seconds (5 minutes).
_DEFAULT_TIMEOUT: int = 300

#: Grace period in seconds between SIGTERM and SIGKILL on timeout.
_KILL_GRACE_PERIOD: float = 5.0

#: Maximum bytes to read from stdout/stderr before truncating.
_MAX_OUTPUT_BYTES: int = 10 * 1024 * 1024  # 10 MiB

#: Environment variable names that are always stripped from the
#: subprocess environment to prevent interference.
_STRIP_ENV_VARS: Tuple[str, ...] = (
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONCASEOK",
    "PYTHONIOENCODING",
    "PYTHONFAULTHANDLER",
    "PYTHONHASHSEED",
    "PYTHONBREAKPOINT",
    "PYTHONDEBUG",
    "PYTHONINSPECT",
    "PYTHONNOUSERSITE",
    "PYTHONOPTIMIZE",
    "PYTHONUNBUFFERED",
    "PYTHONVERBOSE",
    "PYTHONWARNINGS",
    "PYTHONASYNCIODEBUG",
)

#: Allowed characters in environment variable **values**.
#: Rejects newlines, null bytes, and other control characters.
_VALID_ENV_VALUE_CHARS: frozenset = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "_-./:+=@%^&*()!~`\"'<>?,;[]{}|\\ "
)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class RunnerError(Exception):
    """
    Raised when a Python subprocess fails.

    Parameters
    ----------
    message : str
        Human-readable description.
    returncode : int or None
        Process return code, or ``None`` if terminated by signal.
    stdout : str
        Captured stdout (may be truncated).
    stderr : str
        Captured stderr (may be truncated).
    command : list of str
        The command that was executed.
    """

    def __init__(
        self,
        message: str,
        returncode: Optional[int] = None,
        stdout: str = "",
        stderr: str = "",
        command: Optional[List[str]] = None,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.command = command or []

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.command:
            parts.append(f"  Command: {' '.join(shlex.quote(c) for c in self.command)}")
        if self.returncode is not None:
            parts.append(f"  Return code: {self.returncode}")
        if self.stderr:
            stderr_short = self.stderr[:500]
            if len(self.stderr) > 500:
                stderr_short += f"\n  ... ({len(self.stderr) - 500} more bytes)"
            parts.append(f"  Stderr:\n    {stderr_short.strip()}")
        return "\n".join(parts)


class RunnerTimeoutError(RunnerError):
    """
    Raised when a subprocess exceeds its time limit.

    Parameters
    ----------
    timeout : float
        The timeout that was exceeded.
    """

    def __init__(
        self,
        timeout: float,
        command: Optional[List[str]] = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.timeout = timeout
        super().__init__(
            f"Process timed out after {timeout:.1f} seconds",
            returncode=None,
            stdout=stdout,
            stderr=stderr,
            command=command,
        )


class PipError(RunnerError):
    """Raised when a pip operation fails."""

    pass


class EnvironmentError(Exception):
    """Raised when environment variable validation fails."""

    pass


# ---------------------------------------------------------------------------
# Subprocess Result
# ---------------------------------------------------------------------------


class RunnerResult:
    """
    Holds the result of a completed subprocess execution.

    Parameters
    ----------
    returncode : int
        Process exit code.
    stdout : str
        Captured standard output.
    stderr : str
        Captured standard error.
    elapsed : float
        Wall-clock execution time in seconds.
    command : list of str
        The command that was executed.
    timed_out : bool
        Whether the process was terminated due to timeout.

    Attributes
    ----------
    returncode : int
    stdout : str
    stderr : str
    elapsed : float
    command : list of str
    timed_out : bool
    success : bool
        ``True`` if returncode is 0 and not timed out.
    """

    def __init__(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
        elapsed: float,
        command: List[str],
        timed_out: bool = False,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.elapsed = elapsed
        self.command = command
        self.timed_out = timed_out

    @property
    def success(self) -> bool:
        """``True`` if the process completed successfully."""
        return self.returncode == 0 and not self.timed_out

    def __repr__(self) -> str:
        status = "success" if self.success else f"failed (rc={self.returncode})"
        return (
            f"RunnerResult({status}, "
            f"elapsed={self.elapsed:.2f}s, "
            f"stdout={len(self.stdout)} bytes, "
            f"stderr={len(self.stderr)} bytes)"
        )

    def json(self) -> Any:
        """
        Parse stdout as JSON.

        Returns
        -------
        Any
            Parsed JSON value.

        Raises
        ------
        json.JSONDecodeError
            If stdout is not valid JSON.
        """
        return json.loads(self.stdout)


# ---------------------------------------------------------------------------
# Environment Validator
# ---------------------------------------------------------------------------


class _EnvironmentValidator:
    """
    Validates and sanitises environment variables for subprocesses.

    Notes
    -----
    - Rejects variable names containing ``=`` or null bytes.
    - Rejects variable values containing newlines or null bytes.
    - Truncates values longer than 32 KiB.
    """

    MAX_VALUE_LENGTH: int = 32 * 1024  # 32 KiB

    @classmethod
    def validate_name(cls, name: str) -> str:
        """
        Validate an environment variable name.

        Parameters
        ----------
        name : str
            Variable name.

        Returns
        -------
        str
            The validated name.

        Raises
        ------
        EnvironmentError
            If the name is invalid.
        """
        if not name:
            raise EnvironmentError("Environment variable name is empty.")
        if "=" in name:
            raise EnvironmentError(
                f"Environment variable name contains '=': {name!r}"
            )
        if "\x00" in name:
            raise EnvironmentError(
                f"Environment variable name contains null byte: {name!r}"
            )
        return name.strip()

    @classmethod
    def validate_value(cls, value: str) -> str:
        """
        Validate an environment variable value.

        Parameters
        ----------
        value : str
            Variable value.

        Returns
        -------
        str
            The validated (and possibly truncated) value.

        Raises
        ------
        EnvironmentError
            If the value contains forbidden characters.
        """
        if "\x00" in value:
            raise EnvironmentError(
                "Environment variable value contains null byte."
            )
        if "\n" in value or "\r" in value:
            raise EnvironmentError(
                "Environment variable value contains newline."
            )
        if len(value) > cls.MAX_VALUE_LENGTH:
            value = value[: cls.MAX_VALUE_LENGTH]
        return value

    @classmethod
    def validate_dict(cls, env: Dict[str, str]) -> Dict[str, str]:
        """
        Validate all entries in an environment dict.

        Parameters
        ----------
        env : dict[str, str]
            Environment variables.

        Returns
        -------
        dict[str, str]
            Validated copy.

        Raises
        ------
        EnvironmentError
            If any entry is invalid.
        """
        result: Dict[str, str] = {}
        for key, value in env.items():
            valid_key = cls.validate_name(key)
            valid_value = cls.validate_value(value)
            result[valid_key] = valid_value
        return result


# ---------------------------------------------------------------------------
# Python Runner
# ---------------------------------------------------------------------------


class PythonRunner:
    """
    Execute Python scripts and commands using a specific Python
    installation.

    Parameters
    ----------
    python_bin : Path
        Absolute path to the Python executable.
    default_timeout : float
        Default subprocess timeout in seconds. Set to ``0`` or
        ``None`` to disable. Default 300.
    inherit_env : bool
        If ``True``, inherit the current process environment (minus
        stripped vars). If ``False``, start with a minimal
        environment. Default ``False``.

    Raises
    ------
    FileNotFoundError
        If *python_bin* does not exist or is not executable.
    ValueError
        If *python_bin* is not an absolute path.

    Examples
    --------
    >>> from pathlib import Path
    >>> runner = PythonRunner(Path("/opt/python/bin/python3"))
    >>> result = runner.run_code("import sys; print(sys.version)")
    >>> print(result.stdout)
    3.11.5 (main, Oct  2 2023, 13:45:12) [GCC 12.2.0]

    With inherited environment::

    >>> runner = PythonRunner(
    ...     python_bin=Path("/opt/python/bin/python3"),
    ...     inherit_env=True,
    ... )
    """

    def __init__(
        self,
        python_bin: Path,
        default_timeout: float = _DEFAULT_TIMEOUT,
        inherit_env: bool = False,
    ) -> None:
        if not python_bin.is_absolute():
            raise ValueError(
                f"python_bin must be an absolute path, got: {python_bin}"
            )
        if not python_bin.exists():
            raise FileNotFoundError(
                f"Python executable not found: {python_bin}"
            )
        if not os.access(python_bin, os.X_OK):
            raise PermissionError(
                f"Python executable is not executable: {python_bin}"
            )

        self._python_bin = python_bin
        self._default_timeout = default_timeout
        self._inherit_env = inherit_env
        self._env_validator = _EnvironmentValidator()

    # ------------------------------------------------------------------
    # Public API — Script Execution
    # ------------------------------------------------------------------

    def run_script(
        self,
        script: Path,
        args: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        cwd: Optional[Path] = None,
        extra_env: Optional[Dict[str, str]] = None,
        check: bool = True,
    ) -> RunnerResult:
        """
        Execute a Python script.

        Parameters
        ----------
        script : Path
            Path to the ``.py`` script. Must exist.
        args : list of str, optional
            Command-line arguments for the script.
        timeout : float, optional
            Timeout in seconds. Overrides *default_timeout*.
            ``0`` or ``None`` disables the timeout.
        cwd : Path, optional
            Working directory for the subprocess. Defaults to
            *script*'s parent directory.
        extra_env : dict[str, str], optional
            Additional environment variables. Validated for safety.
        check : bool
            If ``True`` (default), raise :class:`RunnerError` on
            non-zero exit or timeout.

        Returns
        -------
        RunnerResult
            Execution result.

        Raises
        ------
        FileNotFoundError
            If *script* does not exist.
        RunnerError
            If *check* is ``True`` and the process fails.
        RunnerTimeoutError
            If the process times out and *check* is ``True``.
        EnvironmentError
            If *extra_env* contains invalid values.
        """
        if not script.exists():
            raise FileNotFoundError(f"Script not found: {script}")

        command = [str(self._python_bin), str(script.resolve())]
        if args:
            command.extend(args)

        return self._run(
            command=command,
            timeout=timeout,
            cwd=cwd or script.parent,
            extra_env=extra_env,
            check=check,
        )

    def run_code(
        self,
        code: str,
        timeout: Optional[float] = None,
        cwd: Optional[Path] = None,
        extra_env: Optional[Dict[str, str]] = None,
        check: bool = True,
    ) -> RunnerResult:
        """
        Execute Python code passed as a string.

        Parameters
        ----------
        code : str
            Python source code.
        timeout : float, optional
            Timeout in seconds.
        cwd : Path, optional
            Working directory.
        extra_env : dict[str, str], optional
            Additional environment variables.
        check : bool
            If ``True`` (default), raise on failure.

        Returns
        -------
        RunnerResult

        Raises
        ------
        RunnerError
            If execution fails and *check* is ``True``.
        """
        command = [str(self._python_bin), "-c", code]
        return self._run(
            command=command,
            timeout=timeout,
            cwd=cwd,
            extra_env=extra_env,
            check=check,
        )

    def run_module(
        self,
        module: str,
        args: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        cwd: Optional[Path] = None,
        extra_env: Optional[Dict[str, str]] = None,
        check: bool = True,
    ) -> RunnerResult:
        """
        Execute a Python module as ``python -m <module>``.

        Parameters
        ----------
        module : str
            Module name (e.g., ``"pip"``, ``"http.server"``).
        args : list of str, optional
            Arguments for the module.
        timeout : float, optional
            Timeout in seconds.
        cwd : Path, optional
            Working directory.
        extra_env : dict[str, str], optional
            Additional environment variables.
        check : bool
            If ``True`` (default), raise on failure.

        Returns
        -------
        RunnerResult
        """
        command = [str(self._python_bin), "-m", module]
        if args:
            command.extend(args)
        return self._run(
            command=command,
            timeout=timeout,
            cwd=cwd,
            extra_env=extra_env,
            check=check,
        )

    # ------------------------------------------------------------------
    # Public API — Pip Operations
    # ------------------------------------------------------------------

    def pip_install(
        self,
        packages: List[str],
        upgrade: bool = False,
        index_url: Optional[str] = None,
        extra_index_url: Optional[str] = None,
        trusted_host: Optional[str] = None,
        timeout: Optional[float] = None,
        check: bool = True,
    ) -> RunnerResult:
        """
        Install Python packages using pip.

        Parameters
        ----------
        packages : list of str
            Package specifications (e.g., ``["requests", "numpy==1.24.0"]``).
        upgrade : bool
            If ``True``, pass ``--upgrade`` to pip.
        index_url : str, optional
            Custom PyPI index URL (``--index-url``).
        extra_index_url : str, optional
            Additional index URL (``--extra-index-url``).
        trusted_host : str, optional
            Trusted host for insecure indexes (``--trusted-host``).
        timeout : float, optional
            Timeout in seconds. Pip operations may take longer;
            consider a generous timeout.
        check : bool
            If ``True`` (default), raise :class:`PipError` on failure.

        Returns
        -------
        RunnerResult

        Raises
        ------
        PipError
            If pip fails and *check* is ``True``.
        ValueError
            If *packages* is empty.
        """
        if not packages:
            raise ValueError("packages list must not be empty.")

        args = [
            str(self._python_bin),
            "-m",
            "pip",
            "install",
            "--no-input",
            "--no-color",
        ]

        if upgrade:
            args.append("--upgrade")
        if index_url:
            args.extend(["--index-url", index_url])
        if extra_index_url:
            args.extend(["--extra-index-url", extra_index_url])
        if trusted_host:
            args.extend(["--trusted-host", trusted_host])

        args.extend(packages)

        try:
            return self._run(
                command=args,
                timeout=timeout,
                cwd=None,
                extra_env=None,
                check=check,
            )
        except RunnerError as e:
            raise PipError(
                str(e),
                returncode=e.returncode,
                stdout=e.stdout,
                stderr=e.stderr,
                command=e.command,
            ) from e

    def pip_uninstall(
        self,
        packages: List[str],
        yes: bool = True,
        timeout: Optional[float] = None,
        check: bool = True,
    ) -> RunnerResult:
        """
        Uninstall Python packages using pip.

        Parameters
        ----------
        packages : list of str
            Package names to uninstall.
        yes : bool
            If ``True`` (default), pass ``--yes`` to skip confirmation.
        timeout : float, optional
            Timeout in seconds.
        check : bool
            If ``True`` (default), raise on failure.

        Returns
        -------
        RunnerResult

        Raises
        ------
        PipError
            If pip fails and *check* is ``True``.
        """
        args = [
            str(self._python_bin),
            "-m",
            "pip",
            "uninstall",
            "--no-input",
            "--no-color",
        ]
        if yes:
            args.append("--yes")
        args.extend(packages)

        try:
            return self._run(
                command=args,
                timeout=timeout,
                cwd=None,
                extra_env=None,
                check=check,
            )
        except RunnerError as e:
            raise PipError(
                str(e),
                returncode=e.returncode,
                stdout=e.stdout,
                stderr=e.stderr,
                command=e.command,
            ) from e

    def pip_list(
        self,
        outdated: bool = False,
        format: str = "json",
        timeout: Optional[float] = None,
        check: bool = True,
    ) -> Union[RunnerResult, List[Dict[str, str]]]:
        """
        List installed packages.

        Parameters
        ----------
        outdated : bool
            If ``True``, list only outdated packages.
        format : str
            Output format: ``"json"`` (default) or ``"columns"``.
        timeout : float, optional
            Timeout in seconds.
        check : bool
            If ``True`` (default), raise on failure.

        Returns
        -------
        RunnerResult or list of dict
            If *format* is ``"json"`` and the command succeeds,
            returns the parsed package list. Otherwise returns
            :class:`RunnerResult`.
        """
        args = [
            str(self._python_bin),
            "-m",
            "pip",
            "list",
            "--no-color",
            f"--format={format}",
        ]
        if outdated:
            args.append("--outdated")

        result = self._run(
            command=args,
            timeout=timeout,
            cwd=None,
            extra_env=None,
            check=check,
        )

        if format == "json" and result.success:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                pass

        return result

    # ------------------------------------------------------------------
    # Public API — Information
    # ------------------------------------------------------------------

    def get_version_info(self) -> Dict[str, str]:
        """
        Get detailed version information from the Python interpreter.

        Returns
        -------
        dict[str, str]
            Keys: ``version``, ``build``, ``compiler``, ``platform``,
            ``implementation``, ``architecture``, ``executable``.

        Raises
        ------
        RunnerError
            If the version query fails.
        """
        code = """
import json, sys, platform
info = {
    "version": sys.version.split()[0],
    "full_version": sys.version,
    "build": platform.python_build()[1],
    "compiler": platform.python_compiler(),
    "platform": platform.platform(),
    "implementation": platform.python_implementation(),
    "architecture": platform.architecture()[0],
    "executable": sys.executable,
}
print(json.dumps(info))
"""
        result = self.run_code(code.strip(), check=True)
        return json.loads(result.stdout)

    def get_pip_version(self) -> str:
        """
        Get the installed pip version.

        Returns
        -------
        str
            Pip version string (e.g., ``"23.2.1"``).

        Raises
        ------
        RunnerError
            If the query fails.
        """
        result = self.run_module(
            "pip", args=["--version"], check=True
        )
        # Output format: "pip X.Y.Z from /path (python 3.11)"
        return result.stdout.split()[1]

    def is_package_installed(self, package: str) -> bool:
        """
        Check if a package is installed.

        Parameters
        ----------
        package : str
            Package name.

        Returns
        -------
        bool
            ``True`` if the package is importable.
        """
        code = f"""
try:
    import {package}
    print("INSTALLED")
except ImportError:
    print("NOT_INSTALLED")
"""
        result = self.run_code(code.strip(), check=False)
        return "INSTALLED" in result.stdout

    # ------------------------------------------------------------------
    # Public API — Virtual Environments
    # ------------------------------------------------------------------

    def create_venv(
        self,
        venv_dir: Path,
        system_site_packages: bool = False,
        clear: bool = False,
        timeout: Optional[float] = None,
        check: bool = True,
    ) -> RunnerResult:
        """
        Create a virtual environment.

        Parameters
        ----------
        venv_dir : Path
            Directory for the virtual environment. Created if it
            does not exist.
        system_site_packages : bool
            If ``True``, give the venv access to system site-packages.
        clear : bool
            If ``True``, clear existing venv directory before creation.
        timeout : float, optional
            Timeout in seconds.
        check : bool
            If ``True`` (default), raise on failure.

        Returns
        -------
        RunnerResult
        """
        args = [
            str(self._python_bin),
            "-m",
            "venv",
            str(venv_dir),
        ]
        if system_site_packages:
            args.append("--system-site-packages")
        if clear:
            args.append("--clear")

        return self._run(
            command=args,
            timeout=timeout,
            cwd=None,
            extra_env=None,
            check=check,
        )

    # ------------------------------------------------------------------
    # Internal Execution
    # ------------------------------------------------------------------

    def _run(
        self,
        command: List[str],
        timeout: Optional[float] = None,
        cwd: Optional[Path] = None,
        extra_env: Optional[Dict[str, str]] = None,
        check: bool = True,
    ) -> RunnerResult:
        """
        Execute a command via :mod:`subprocess`.

        Parameters
        ----------
        command : list of str
            Command and arguments (no shell).
        timeout : float, optional
            Timeout override.
        cwd : Path, optional
            Working directory.
        extra_env : dict[str, str], optional
            Additional environment variables.
        check : bool
            Raise on failure.

        Returns
        -------
        RunnerResult

        Raises
        ------
        RunnerError
            On non-zero exit (if *check* is ``True``).
        RunnerTimeoutError
            On timeout (if *check* is ``True``).
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout is not None and effective_timeout <= 0:
            effective_timeout = None

        # Build environment
        env = self._build_env(extra_env)

        start_time = time.monotonic()
        timed_out = False

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(cwd) if cwd else None,
                env=env,
                universal_newlines=False,  # read as bytes
            )

            try:
                stdout_bytes, stderr_bytes = process.communicate(
                    timeout=effective_timeout
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                # Terminate gracefully
                process.terminate()
                try:
                    stdout_bytes, stderr_bytes = process.communicate(
                        timeout=_KILL_GRACE_PERIOD
                    )
                except subprocess.TimeoutExpired:
                    # Force kill
                    process.kill()
                    stdout_bytes, stderr_bytes = process.communicate()

            elapsed = time.monotonic() - start_time

            # Decode output (truncate if needed)
            stdout = self._decode_and_truncate(stdout_bytes)
            stderr = self._decode_and_truncate(stderr_bytes)

            result = RunnerResult(
                returncode=process.returncode or -1,
                stdout=stdout,
                stderr=stderr,
                elapsed=elapsed,
                command=command,
                timed_out=timed_out,
            )

            if check:
                if timed_out:
                    raise RunnerTimeoutError(
                        timeout=effective_timeout or 0,
                        command=command,
                        stdout=stdout,
                        stderr=stderr,
                    )
                if result.returncode != 0:
                    raise RunnerError(
                        f"Process exited with code {result.returncode}",
                        returncode=result.returncode,
                        stdout=stdout,
                        stderr=stderr,
                        command=command,
                    )

            return result

        except (RunnerError, RunnerTimeoutError):
            raise
        except OSError as e:
            raise RunnerError(
                f"Failed to start process: {e}",
                returncode=None,
                command=command,
            ) from e

    def _build_env(
        self,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Build the subprocess environment.

        Parameters
        ----------
        extra_env : dict[str, str], optional
            Additional variables.

        Returns
        -------
        dict[str, str]
            Sanitised environment.

        Notes
        -----
        - If *inherit_env* is ``True``, starts with ``os.environ``
          minus stripped variables.
        - If *inherit_env* is ``False``, starts with a minimal
          environment containing only ``PATH`` and ``HOME``.
        """
        if self._inherit_env:
            env = os.environ.copy()
            # Strip interfering variables
            for var in _STRIP_ENV_VARS:
                env.pop(var, None)
        else:
            env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": os.environ.get("HOME", str(Path.home())),
                "USER": os.environ.get("USER", ""),
                "TMPDIR": os.environ.get("TMPDIR", tempfile.gettempdir()),
                "TEMP": os.environ.get("TEMP", tempfile.gettempdir()),
                "TMP": os.environ.get("TMP", tempfile.gettempdir()),
            }

        # Add Python binary directory to PATH
        bin_dir = str(self._python_bin.parent)
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"

        # Mark as managed
        env["PYTHON_STANDALONE_RUNNER"] = "1"

        # Merge extra env
        if extra_env:
            validated = self._env_validator.validate_dict(extra_env)
            env.update(validated)

        return env

    @staticmethod
    def _decode_and_truncate(data: bytes) -> str:
        """
        Decode bytes to str, truncating if necessary.

        Parameters
        ----------
        data : bytes
            Raw stdout/stderr bytes.

        Returns
        -------
        str
            Decoded string, possibly truncated.
        """
        if len(data) > _MAX_OUTPUT_BYTES:
            data = data[:_MAX_OUTPUT_BYTES]
        try:
            return data.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="replace")
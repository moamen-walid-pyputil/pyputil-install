"""
Command-Line Interface for Python Standalone Installer
========================================================

Provides a comprehensive command-line interface for managing portable
Python installations. Supports installation, removal, listing, switching,
execution, and configuration of standalone Python builds.

Uses only the Python standard library (:mod:`argparse`, :mod:`sys`,
:mod:`pathlib`). No third-party dependencies.

Security
--------
- All user-supplied version strings are validated against known good
  patterns before being passed to internal APIs.
- Shell injection is prevented by using :func:`argparse` for all
  argument parsing; no manual shell command construction.
- File paths from command-line arguments are resolved and validated
  before use.
- Environment variable manipulation via ``set-default`` is additive
  and reversible.
- The ``run`` and ``exec`` commands pass arguments as lists to
  subprocesses; ``shell=True`` is never used.

Usage
-----
.. code-block:: bash

    # Install a Python version
    python -m python_installer install 3.11.5

    # Install to a custom location
    python -m python_installer install 3.11.5 --install-root /opt/pythons

    # List installed versions
    python -m python_installer list

    # List available versions
    python -m python_installer list --available

    # Set a version as the current process
    python -m python_installer switch 3.11.5

    # Set a version as the system default
    python -m python_installer set-default 3.11.5

    # Run a script with a specific version
    python -m python_installer run 3.11.5 --script my_script.py -- --flag value

    # Execute inline code
    python -m python_installer exec 3.11.5 -c "print('Hello')"

    # Install a pip package
    python -m python_installer pip 3.11.5 install requests numpy

    # Uninstall a version
    python -m python_installer uninstall 3.9.18

    # Show version info
    python -m python_installer info 3.11.5

    # Clear the download cache
    python -m python_installer clear-cache

    # Show the current default version
    python -m python_installer default

Warnings
--------
- The ``switch`` command replaces the current process via
  :func:`os.execve` and **does not return**.
- The ``set-default`` command modifies shell configuration files.
  Back up ``~/.bashrc`` and ``~/.zshrc`` before first use.
- The ``uninstall`` command permanently deletes the Python
  installation directory. This cannot be undone.
- The ``clear-cache`` command deletes all cached downloads.
  Subsequent installations will re-download archives.
- Commands that require network access (``install``, ``list
  --available``) will fail if offline.
- On Windows, the ``set-default`` command creates batch shims that
  only work in ``cmd.exe`` and PowerShell.

Notes
-----
- All commands support ``--install-root`` and ``--cache-dir``
  overrides.
- The ``--github-token`` option accepts a token via command line or
  the ``GITHUB_TOKEN`` environment variable.
- Progress bars are shown by default. Use ``--no-progress`` to
  suppress them.
- Exit codes: ``0`` on success, ``1`` on expected error (version not
  found, etc.), ``2`` on unexpected error.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Program name displayed in help text.
_PROG: str = "python-installer"

#: Version string for the CLI itself.
_CLI_VERSION: str = "1.0.0"

#: Exit codes.
_EXIT_SUCCESS: int = 0
_EXIT_FAILURE: int = 1
_EXIT_INTERNAL_ERROR: int = 2

#: Maximum length for inline code strings.
_MAX_CODE_LENGTH: int = 10_000

#: Valid Python version pattern (e.g., "3.11.5").
_VERSION_PATTERN: str = r"^\d+\.\d+\.\d+$"


# ---------------------------------------------------------------------------
# Argument Parser Construction
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the complete argument parser hierarchy.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with subcommands.

    Notes
    -----
    - Uses ``formatter_class=argparse.RawDescriptionHelpFormatter``
      to preserve formatting in help text.
    - Global options are added to the root parser and inherited by
      all subcommands via ``parents``.
    """
    # Global options
    global_options = argparse.ArgumentParser(add_help=False)
    global_options.add_argument(
        "--install-root",
        type=Path,
        default=None,
        help=(
            "Root directory for Python installations "
            "(default: ~/.python_standalone)"
        ),
    )
    global_options.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached downloads "
            "(default: ~/.cache/python_standalone)"
        ),
    )
    global_options.add_argument(
        "--github-token",
        type=str,
        default=None,
        help=(
            "GitHub personal access token for higher API rate limits "
            "(env: GITHUB_TOKEN)"
        ),
    )
    global_options.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Suppress progress bars",
    )
    global_options.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Network timeout in seconds (default: 30)",
    )
    global_options.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum download retry attempts (default: 3)",
    )

    # Root parser
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description=(
            "Install and manage portable Python builds from "
            "indygreg/python-build-standalone."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s install 3.11.5\n"
            "  %(prog)s list\n"
            "  %(prog)s switch 3.11.5\n"
            "  %(prog)s set-default 3.11.5\n"
            "  %(prog)s run 3.11.5 --script app.py\n"
            "  %(prog)s exec 3.11.5 -c 'print(1+1)'\n"
            "  %(prog)s uninstall 3.9.18"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_CLI_VERSION}",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        metavar="COMMAND",
    )
    subparsers.required = True

    # ------------------------------------------------------------------
    # install
    # ------------------------------------------------------------------
    install_parser = subparsers.add_parser(
        "install",
        parents=[global_options],
        help="Download and install a Python version",
        description="Download and install a standalone Python build.",
    )
    install_parser.add_argument(
        "version",
        type=str,
        help='Python version, e.g. "3.11.5"',
    )
    install_parser.add_argument(
        "--release-date",
        type=str,
        default=None,
        help='Release date tag (8 digits, e.g. "20231002")',
    )
    install_parser.add_argument(
        "--target",
        type=str,
        default=None,
        help='Platform target triple (e.g. "x86_64-unknown-linux-gnu")',
    )
    install_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Reinstall even if already present",
    )
    install_parser.add_argument(
        "--variant",
        type=str,
        default="install_only",
        choices=["install_only", "full"],
        help="Build variant (default: install_only)",
    )
    install_parser.add_argument(
        "--set-default",
        action="store_true",
        default=False,
        dest="set_default_flag",
        help="Set as system default after installation",
    )

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------
    list_parser = subparsers.add_parser(
        "list",
        parents=[global_options],
        help="List installed or available Python versions",
        description="List Python versions.",
    )
    list_parser.add_argument(
        "--available",
        action="store_true",
        default=False,
        help="List versions available for download (requires network)",
    )
    list_parser.add_argument(
        "--paths",
        action="store_true",
        default=False,
        help="Show installation paths",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output as JSON",
    )

    # ------------------------------------------------------------------
    # switch
    # ------------------------------------------------------------------
    switch_parser = subparsers.add_parser(
        "switch",
        parents=[global_options],
        help="Replace current process with a Python version",
        description=(
            "Replace the running process with the specified Python "
            "version. This command does not return."
        ),
    )
    switch_parser.add_argument(
        "version",
        type=str,
        help="Python version to switch to",
    )
    switch_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the new Python process",
    )

    # ------------------------------------------------------------------
    # set-default
    # ------------------------------------------------------------------
    set_default_parser = subparsers.add_parser(
        "set-default",
        parents=[global_options],
        help="Set a Python version as the system default",
        description="Create symlinks and update shell configuration.",
    )
    set_default_parser.add_argument(
        "version",
        type=str,
        help="Python version to set as default",
    )

    # ------------------------------------------------------------------
    # default (show)
    # ------------------------------------------------------------------
    default_parser = subparsers.add_parser(
        "default",
        parents=[global_options],
        help="Show the current default Python version",
        description="Print the currently-set default version.",
    )

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        parents=[global_options],
        help="Run a Python script with a specific version",
        description="Execute a Python script.",
    )
    run_parser.add_argument(
        "version",
        type=str,
        help="Python version to use",
    )
    run_parser.add_argument(
        "--script",
        type=Path,
        required=True,
        help="Path to the Python script",
    )
    run_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments for the script (use -- to separate)",
    )

    # ------------------------------------------------------------------
    # exec
    # ------------------------------------------------------------------
    exec_parser = subparsers.add_parser(
        "exec",
        parents=[global_options],
        help="Execute inline Python code",
        description="Execute Python code passed as a string.",
    )
    exec_parser.add_argument(
        "version",
        type=str,
        help="Python version to use",
    )
    exec_parser.add_argument(
        "-c",
        "--code",
        type=str,
        required=True,
        dest="code",
        help="Python code to execute",
    )

    # ------------------------------------------------------------------
    # pip
    # ------------------------------------------------------------------
    pip_parser = subparsers.add_parser(
        "pip",
        parents=[global_options],
        help="Run pip commands for a specific Python version",
        description="Install or uninstall pip packages.",
    )
    pip_parser.add_argument(
        "version",
        type=str,
        help="Python version to use",
    )
    pip_sub = pip_parser.add_subparsers(
        title="pip-command",
        dest="pip_command",
        metavar="PIP_COMMAND",
    )
    pip_sub.required = True

    # pip install
    pip_install = pip_sub.add_parser("install", help="Install packages")
    pip_install.add_argument(
        "packages",
        nargs="+",
        type=str,
        help="Package specifications",
    )
    pip_install.add_argument(
        "--upgrade",
        action="store_true",
        default=False,
        help="Upgrade existing packages",
    )

    # pip uninstall
    pip_uninstall = pip_sub.add_parser("uninstall", help="Uninstall packages")
    pip_uninstall.add_argument(
        "packages",
        nargs="+",
        type=str,
        help="Package names",
    )

    # pip list
    pip_list = pip_sub.add_parser("list", help="List installed packages")
    pip_list.add_argument(
        "--outdated",
        action="store_true",
        default=False,
        help="Show only outdated packages",
    )

    # ------------------------------------------------------------------
    # info
    # ------------------------------------------------------------------
    info_parser = subparsers.add_parser(
        "info",
        parents=[global_options],
        help="Show information about an installed Python version",
        description="Display version details.",
    )
    info_parser.add_argument(
        "version",
        type=str,
        help="Python version",
    )

    # ------------------------------------------------------------------
    # uninstall
    # ------------------------------------------------------------------
    uninstall_parser = subparsers.add_parser(
        "uninstall",
        parents=[global_options],
        help="Remove an installed Python version",
        description="Delete a Python installation.",
    )
    uninstall_parser.add_argument(
        "version",
        type=str,
        help="Python version to remove",
    )
    uninstall_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Remove even if this is the default version",
    )

    # ------------------------------------------------------------------
    # clear-cache
    # ------------------------------------------------------------------
    clear_cache_parser = subparsers.add_parser(
        "clear-cache",
        parents=[global_options],
        help="Clear the download cache",
        description="Delete all cached downloads.",
    )

    # ------------------------------------------------------------------
    # repair
    # ------------------------------------------------------------------
    repair_parser = subparsers.add_parser(
        "repair",
        parents=[global_options],
        help="Reinstall a Python version",
        description="Force-reinstall a version to repair it.",
    )
    repair_parser.add_argument(
        "version",
        type=str,
        help="Python version to repair",
    )

    return parser


# ---------------------------------------------------------------------------
# Version Validation
# ---------------------------------------------------------------------------


def _validate_version(version: str) -> str:
    """
    Validate a Python version string.

    Parameters
    ----------
    version : str
        Raw version string from user input.

    Returns
    -------
    str
        The validated version.

    Raises
    ------
    SystemExit
        If the version is invalid.
    """
    import re

    if not re.match(_VERSION_PATTERN, version):
        print(
            f"Error: Invalid version format: {version!r}\n"
            f"Expected format: X.Y.Z (e.g., 3.11.5)",
            file=sys.stderr,
        )
        sys.exit(_EXIT_FAILURE)

    return version


# ---------------------------------------------------------------------------
# Command Handlers
# ---------------------------------------------------------------------------


def _cmd_install(
    installer: Any, args: argparse.Namespace
) -> int:
    """
    Handle the ``install`` command.

    Parameters
    ----------
    installer : PythonInstaller
        The installer instance.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    version = _validate_version(args.version)

    try:
        python_path = installer.install(
            version=version,
            release_date=args.release_date,
            target_triple=args.target,
            force=args.force,
        )
        print(f"Successfully installed Python {version}")
        print(f"  Executable: {python_path}")

        if args.set_default_flag:
            installer.set_default(version)
            print(f"  Set as system default.")

        return _EXIT_SUCCESS

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return _EXIT_FAILURE


def _cmd_list(
    installer: Any, args: argparse.Namespace
) -> int:
    """
    Handle the ``list`` command.

    Parameters
    ----------
    installer : PythonInstaller
        The installer instance.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    import json

    try:
        if args.available:
            print("Fetching available versions from GitHub...")
            versions = installer.fetch_available_versions()
            if args.json:
                print(json.dumps({"available": versions}, indent=2))
            else:
                print("Available versions:")
                for v in versions:
                    print(f"  - {v}")
        else:
            installed = installer.list_installed()
            if args.json:
                data = {
                    version: str(path)
                    for version, path in installed.items()
                }
                print(json.dumps(data, indent=2))
            elif not installed:
                print("No Python versions installed.")
                print(f"Install root: {installer._install_root}")
            else:
                default = installer.get_default()
                print(f"Installed versions ({len(installed)}):")
                for version, path in sorted(installed.items()):
                    marker = " (default)" if version == default else ""
                    if args.paths:
                        print(f"  Python {version}{marker}: {path}")
                    else:
                        print(f"  Python {version}{marker}")

        return _EXIT_SUCCESS

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return _EXIT_FAILURE


def _cmd_switch(
    installer: Any, args: argparse.Namespace
) -> int:
    """
    Handle the ``switch`` command.

    Parameters
    ----------
    installer : PythonInstaller
        The installer instance.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code (but usually does not return).
    """
    version = _validate_version(args.version)
    switch_args = args.args if args.args else None

    try:
        print(f"Switching to Python {version}...")
        installer.set_current(version, args=switch_args)
        # This line is never reached
        return _EXIT_SUCCESS
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return _EXIT_FAILURE


def _cmd_set_default(
    installer: Any, args: argparse.Namespace
) -> int:
    """
    Handle the ``set-default`` command.

    Parameters
    ----------
    installer : PythonInstaller
        The installer instance.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    version = _validate_version(args.version)

    try:
        python_path = installer.set_default(version)
        print(f"Python {version} is now the system default.")
        print(f"  Executable: {python_path}")
        print()
        print("To apply changes to the current shell, run:")
        print("  source ~/.bashrc  # or ~/.zshrc, ~/.profile")
        return _EXIT_SUCCESS
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return _EXIT_FAILURE


def _cmd_default(
    installer: Any, args: argparse.Namespace
) -> int:
    """
    Handle the ``default`` command (show current default).

    Parameters
    ----------
    installer : PythonInstaller
        The installer instance.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    default = installer.get_default()
    if default:
        path = installer.get_python_path(default)
        print(f"Default Python version: {default}")
        print(f"  Executable: {path}")
    else:
        print("No default Python version is set.")
        print("Use 'set-default <version>' to set one.")
    return _EXIT_SUCCESS


def _cmd_run(
    installer: Any, args: argparse.Namespace
) -> int:
    """
    Handle the ``run`` command.

    Parameters
    ----------
    installer : PythonInstaller
        The installer instance.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    version = _validate_version(args.version)
    script = args.script

    if not script.exists():
        print(f"Error: Script not found: {script}", file=sys.stderr)
        return _EXIT_FAILURE

    # Parse remaining args (after --)
    script_args = args.args if args.args else None

    try:
        result = installer.run_script(
            version=version,
            script=script,
            args=script_args,
            check=False,
        )
        if result.stdout:
            sys.stdout.write(result.stdout)
            if not result.stdout.endswith("\n"):
                sys.stdout.write("\n")
        if result.stderr:
            sys.stderr.write(result.stderr)
        return result.returncode
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return _EXIT_FAILURE


def _cmd_exec(
    installer: Any, args: argparse.Namespace
) -> int:
    """
    Handle the ``exec`` command.

    Parameters
    ----------
    installer : PythonInstaller
        The installer instance.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    version = _validate_version(args.version)
    code = args.code

    if len(code) > _MAX_CODE_LENGTH:
        print(
            f"Error: Code exceeds maximum length of "
            f"{_MAX_CODE_LENGTH} characters.",
            file=sys.stderr,
        )
        return _EXIT_FAILURE

    try:
        result = installer.run_code(
            version=version,
            code=code,
            check=False,
        )
        if result.stdout:
            sys.stdout.write(result.stdout)
            if not result.stdout.endswith("\n"):
                sys.stdout.write("\n")
        if result.stderr:
            sys.stderr.write(result.stderr)
        return result.returncode
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return _EXIT_FAILURE


def _cmd_pip(
    installer: Any, args: argparse.Namespace
) -> int:
    """
    Handle the ``pip`` command.

    Parameters
    ----------
    installer : PythonInstaller
        The installer instance.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    version = _validate_version(args.version)

    try:
        if args.pip_command == "install":
            result = installer.pip_install(
                version=version,
                packages=args.packages,
                upgrade=args.upgrade,
                check=False,
            )
        elif args.pip_command == "uninstall":
            runner = installer._get_runner(version)
            result = runner.pip_uninstall(
                packages=args.packages,
                check=False,
            )
        elif args.pip_command == "list":
            runner = installer._get_runner(version)
            result = runner.pip_list(
                outdated=args.outdated,
                format="columns",
                check=False,
            )
        else:
            print(
                f"Error: Unknown pip command: {args.pip_command}",
                file=sys.stderr,
            )
            return _EXIT_FAILURE

        if result.stdout:
            sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
        return result.returncode

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return _EXIT_FAILURE


def _cmd_info(
    installer: Any, args: argparse.Namespace
) -> int:
    """
    Handle the ``info`` command.

    Parameters
    ----------
    installer : PythonInstaller
        The installer instance.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    import json

    version = _validate_version(args.version)

    try:
        info = installer.get_version_info(version)
        print(json.dumps(info, indent=2))
        return _EXIT_SUCCESS
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return _EXIT_FAILURE


def _cmd_uninstall(
    installer: Any, args: argparse.Namespace
) -> int:
    """
    Handle the ``uninstall`` command.

    Parameters
    ----------
    installer : PythonInstaller
        The installer instance.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    version = _validate_version(args.version)

    try:
        installer.uninstall(version=version, force=args.force)
        print(f"Successfully removed Python {version}.")
        return _EXIT_SUCCESS
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return _EXIT_FAILURE


def _cmd_clear_cache(
    installer: Any, args: argparse.Namespace
) -> int:
    """
    Handle the ``clear-cache`` command.

    Parameters
    ----------
    installer : PythonInstaller
        The installer instance.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    try:
        freed = installer.clear_cache()
        if freed > 0:
            mb = freed / (1024 * 1024)
            print(f"Cache cleared: {mb:.1f} MB freed.")
        else:
            print("Cache is already empty.")
        return _EXIT_SUCCESS
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return _EXIT_FAILURE


def _cmd_repair(
    installer: Any, args: argparse.Namespace
) -> int:
    """
    Handle the ``repair`` command.

    Parameters
    ----------
    installer : PythonInstaller
        The installer instance.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    version = _validate_version(args.version)

    try:
        python_path = installer.repair(version)
        print(f"Successfully repaired Python {version}.")
        print(f"  Executable: {python_path}")
        return _EXIT_SUCCESS
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return _EXIT_FAILURE


# ---------------------------------------------------------------------------
# Command Dispatch Table
# ---------------------------------------------------------------------------

_COMMAND_HANDLERS = {
    "install": _cmd_install,
    "list": _cmd_list,
    "switch": _cmd_switch,
    "set-default": _cmd_set_default,
    "default": _cmd_default,
    "run": _cmd_run,
    "exec": _cmd_exec,
    "pip": _cmd_pip,
    "info": _cmd_info,
    "uninstall": _cmd_uninstall,
    "clear-cache": _cmd_clear_cache,
    "repair": _cmd_repair,
}


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on expected error, 2 on internal
        error.
    """
    parser = _build_parser()

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        # argparse calls sys.exit on --help or parse errors
        return e.code if isinstance(e.code, int) else _EXIT_FAILURE

    # Late import to avoid slow startup for --help
    try:
        from .installer import PythonInstaller
    except ImportError as e:
        print(
            f"Internal error: Cannot import PythonInstaller: {e}",
            file=sys.stderr,
        )
        return _EXIT_INTERNAL_ERROR

    # Build installer instance
    try:
        installer = PythonInstaller(
            install_root=args.install_root,
            cache_dir=args.cache_dir,
            github_token=args.github_token,
            max_retries=args.max_retries,
            timeout=args.timeout,
            show_progress=not args.no_progress,
            variant=getattr(args, "variant", "install_only"),
        )
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return _EXIT_FAILURE

    # Dispatch command
    handler = _COMMAND_HANDLERS.get(args.command)
    if handler is None:
        print(
            f"Internal error: No handler for command {args.command!r}",
            file=sys.stderr,
        )
        return _EXIT_INTERNAL_ERROR

    try:
        return handler(installer, args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        return _EXIT_FAILURE
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return _EXIT_INTERNAL_ERROR


# ---------------------------------------------------------------------------
# Package Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
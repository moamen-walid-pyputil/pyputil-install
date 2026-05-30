#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command-line interface entry point.

This module provides the main entry point for the package_installer
CLI. It uses ``argparse`` to define subcommands for all major
operations: install, uninstall, upgrade, search, info, freeze,
venv, and check.

Functions
--------
main
    Main CLI entry point. Parses arguments and dispatches to
    appropriate handlers.

Examples
--------
From the command line:

.. code-block:: bash

    package-installer install requests
    package-installer install django==4.2.0 --require-hashes
    package-installer search "web framework"
    package-installer info flask
    package-installer upgrade --all
    package-installer freeze --output requirements.txt
    package-installer venv create /tmp/my-env
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..installer import PackageInstaller, InstallConfig
from ..network.pypi import PyPIClient
from ..environment.venv import VirtualEnvironment
from ..exceptions import (
    PackageInstallerError,
    PackageNotFoundError,
    NetworkError,
)
from .formatters import (
    OutputFormatter,
    TableFormatter,
    JsonFormatter,
    SimpleFormatter,
)

logger = logging.getLogger(__name__)


def _create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for the CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="package-installer",
        description="Advanced Python package manager with caching, "
                    "security verification, and virtual environment support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  package-installer install requests
  package-installer install django==4.2.0
  package-installer uninstall flask
  package-installer upgrade requests --to 3.0.0
  package-installer search "machine learning"
  package-installer info numpy
  package-installer freeze
  package-installer venv create ./my-env
  package-installer check --upgrades
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="package-installer 0.1.0",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv, -vvv)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching",
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL certificate verification (not recommended)",
    )
    parser.add_argument(
        "--require-hashes",
        action="store_true",
        help="Require hash verification for all packages",
    )
    parser.add_argument(
        "--user",
        action="store_true",
        help="Install to user site-packages",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Network timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Maximum retry attempts (default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate operations without making changes",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        metavar="COMMAND",
    )

    _add_install_parser(subparsers)
    _add_uninstall_parser(subparsers)
    _add_upgrade_parser(subparsers)
    _add_search_parser(subparsers)
    _add_info_parser(subparsers)
    _add_freeze_parser(subparsers)
    _add_list_parser(subparsers)
    _add_venv_parser(subparsers)
    _add_check_parser(subparsers)

    return parser


def _add_install_parser(subparsers) -> None:
    """Add the 'install' subcommand."""
    install_parser = subparsers.add_parser(
        "install",
        help="Install a package",
        aliases=["i"],
    )
    install_parser.add_argument(
        "package",
        help="Package name and optional version (e.g., 'requests' or 'requests==2.28.0')",
    )
    install_parser.add_argument(
        "--upgrade", "-U",
        action="store_true",
        help="Upgrade package if already installed",
    )
    install_parser.add_argument(
        "--pre",
        action="store_true",
        help="Include pre-release versions",
    )
    install_parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="Reinstall even if already installed",
    )
    install_parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Skip dependency installation",
    )
    install_parser.add_argument(
        "--extra-index-url",
        help="Additional package index URL",
    )


def _add_uninstall_parser(subparsers) -> None:
    """Add the 'uninstall' subcommand."""
    uninstall_parser = subparsers.add_parser(
        "uninstall",
        help="Uninstall a package",
        aliases=["remove", "rm"],
    )
    uninstall_parser.add_argument(
        "package",
        help="Package name to uninstall",
    )
    uninstall_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )


def _add_upgrade_parser(subparsers) -> None:
    """Add the 'upgrade' subcommand."""
    upgrade_parser = subparsers.add_parser(
        "upgrade",
        help="Upgrade packages",
        aliases=["up"],
    )
    upgrade_parser.add_argument(
        "package",
        nargs="?",
        help="Package to upgrade (omit for --all)",
    )
    upgrade_parser.add_argument(
        "--all",
        action="store_true",
        help="Upgrade all installed packages",
    )
    upgrade_parser.add_argument(
        "--to",
        dest="version",
        help="Upgrade to a specific version",
    )
    upgrade_parser.add_argument(
        "--pre",
        action="store_true",
        help="Include pre-release versions",
    )


def _add_search_parser(subparsers) -> None:
    """Add the 'search' subcommand."""
    search_parser = subparsers.add_parser(
        "search",
        help="Search PyPI for packages",
        aliases=["find"],
    )
    search_parser.add_argument(
        "query",
        help="Search query",
    )
    search_parser.add_argument(
        "--max-results", "-n",
        type=int,
        default=20,
        help="Maximum results to display (default: 20)",
    )


def _add_info_parser(subparsers) -> None:
    """Add the 'info' subcommand."""
    info_parser = subparsers.add_parser(
        "info",
        help="Show package information",
        aliases=["show"],
    )
    info_parser.add_argument(
        "package",
        help="Package name",
    )
    info_parser.add_argument(
        "--versions",
        action="store_true",
        help="Show all available versions",
    )
    info_parser.add_argument(
        "--deps",
        action="store_true",
        help="Show dependency tree",
    )
    info_parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Dependency tree depth (default: 2)",
    )


def _add_freeze_parser(subparsers) -> None:
    """Add the 'freeze' subcommand."""
    freeze_parser = subparsers.add_parser(
        "freeze",
        help="Export installed packages",
    )
    freeze_parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path",
    )


def _add_list_parser(subparsers) -> None:
    """Add the 'list' subcommand."""
    list_parser = subparsers.add_parser(
        "list",
        help="List installed packages",
        aliases=["ls"],
    )
    list_parser.add_argument(
        "--outdated",
        action="store_true",
        help="Show only outdated packages",
    )


def _add_venv_parser(subparsers) -> None:
    """Add the 'venv' subcommand."""
    venv_parser = subparsers.add_parser(
        "venv",
        help="Manage virtual environments",
    )
    venv_subparsers = venv_parser.add_subparsers(
        dest="venv_command",
        title="venv commands",
    )

    create_parser = venv_subparsers.add_parser(
        "create",
        help="Create a virtual environment",
    )
    create_parser.add_argument(
        "path",
        type=Path,
        help="Path for the virtual environment",
    )
    create_parser.add_argument(
        "--python",
        help="Python interpreter to use",
    )
    create_parser.add_argument(
        "--system-site-packages",
        action="store_true",
        help="Give access to system site-packages",
    )
    create_parser.add_argument(
        "--upgrade-pip",
        action="store_true",
        help="Upgrade pip after creation",
    )

    destroy_parser = venv_subparsers.add_parser(
        "destroy",
        help="Remove a virtual environment",
    )
    destroy_parser.add_argument(
        "path",
        type=Path,
        help="Path of the environment to destroy",
    )


def _add_check_parser(subparsers) -> None:
    """Add the 'check' subcommand."""
    check_parser = subparsers.add_parser(
        "check",
        help="Check for issues",
    )
    check_parser.add_argument(
        "--upgrades",
        action="store_true",
        help="Check for available upgrades",
    )
    check_parser.add_argument(
        "--integrity",
        action="store_true",
        help="Verify package integrity",
    )
    check_parser.add_argument(
        "package",
        nargs="?",
        help="Package to check (omit for all)",
    )


def _get_formatter(args: argparse.Namespace) -> OutputFormatter:
    """
    Create the appropriate formatter based on CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    OutputFormatter
        Configured formatter instance.
    """
    if args.json:
        return JsonFormatter()
    elif args.quiet:
        return SimpleFormatter()
    else:
        return TableFormatter()


def _get_config(args: argparse.Namespace) -> InstallConfig:
    """
    Build InstallConfig from CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    InstallConfig
        Configuration instance.
    """
    return InstallConfig(
        use_cache=not args.no_cache,
        require_hashes=args.require_hashes,
        verify_ssl=not args.no_verify_ssl,
        timeout_read=args.timeout,
        max_retries=args.retries,
        user_install=args.user,
        dry_run=args.dry_run,
    )


def _parse_package_spec(spec: str) -> tuple:
    """
    Parse a package specification string.

    Parameters
    ----------
    spec : str
        Package spec like ``requests`` or ``requests==2.28.0``.

    Returns
    -------
    tuple
        ``(package_name, version_or_None)``.
    """
    if "==" in spec:
        name, version = spec.split("==", 1)
        return name.strip(), version.strip()
    return spec.strip(), None


def _handle_install(args: argparse.Namespace, config: InstallConfig, fmt: OutputFormatter) -> int:
    """
    Handle the 'install' command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    config : InstallConfig
        Install configuration.
    fmt : OutputFormatter
        Output formatter.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    package_name, version = _parse_package_spec(args.package)

    try:
        installer = PackageInstaller(package_name, config=config)
        result = installer.install(
            version=version,
            upgrade=args.upgrade,
            pre=args.pre,
            force_reinstall=args.force_reinstall,
            no_deps=args.no_deps,
            extra_index_url=args.extra_index_url,
        )

        fmt.write(fmt.format_result(result.to_dict()))
        return 0 if result.success else 1

    except PackageInstallerError as e:
        fmt.write(fmt.format_error(str(e), details=getattr(e, "details", None)))
        return 1


def _handle_uninstall(args: argparse.Namespace, config: InstallConfig, fmt: OutputFormatter) -> int:
    """
    Handle the 'uninstall' command.
    """
    try:
        installer = PackageInstaller(args.package, config=config)
        result = installer.uninstall(confirm=args.yes)
        fmt.write(fmt.format_result(result.to_dict()))
        return 0 if result.success else 1
    except PackageInstallerError as e:
        fmt.write(fmt.format_error(str(e)))
        return 1


def _handle_upgrade(args: argparse.Namespace, config: InstallConfig, fmt: OutputFormatter) -> int:
    """
    Handle the 'upgrade' command.
    """
    if args.all:
        fmt.write(fmt.format_error(
            "Upgrading all packages is not yet implemented.",
        ))
        return 1

    if not args.package:
        fmt.write(fmt.format_error(
            "Specify a package name or use --all.",
        ))
        return 1

    try:
        installer = PackageInstaller(args.package, config=config)
        result = installer.upgrade(version=args.version, pre=args.pre)
        fmt.write(fmt.format_result(result.to_dict()))
        return 0 if result.success else 1
    except PackageInstallerError as e:
        fmt.write(fmt.format_error(str(e)))
        return 1


def _handle_search(args: argparse.Namespace, config: InstallConfig, fmt: OutputFormatter) -> int:
    """
    Handle the 'search' command.
    """
    try:
        client = PyPIClient()
        results = client.search_packages(args.query, max_results=args.max_results)

        if not results:
            fmt.write(fmt.format_info({"message": f"No results for '{args.query}'"}))
            return 0

        formatted = []
        for r in results:
            formatted.append([
                r.get("name", ""),
                r.get("version", ""),
                (r.get("summary", "") or "")[:60],
            ])

        table_fmt = TableFormatter() if not args.json else fmt
        if hasattr(table_fmt, '_format_table'):
            output = table_fmt._format_table(
                ["Name", "Version", "Summary"],
                formatted,
            )
            fmt.write(output)
        else:
            fmt.write(fmt.format_info({"results": results}))

        return 0

    except NetworkError as e:
        fmt.write(fmt.format_error(str(e)))
        return 1


def _handle_info(args: argparse.Namespace, config: InstallConfig, fmt: OutputFormatter) -> int:
    """
    Handle the 'info' command.
    """
    try:
        client = PyPIClient()
        info = client.get_package_info(args.package)

        if args.versions:
            versions = client.get_package_versions(args.package)
            info["versions"] = versions[:20]

        if args.deps:
            tree = client.get_dependency_tree(args.package, depth=args.depth)
            info["dependency_tree"] = tree

        fmt.write(fmt.format_info(info))
        return 0

    except PackageNotFoundError:
        fmt.write(fmt.format_error(f"Package '{args.package}' not found"))
        return 1
    except NetworkError as e:
        fmt.write(fmt.format_error(str(e)))
        return 1


def _handle_freeze(args: argparse.Namespace, config: InstallConfig, fmt: OutputFormatter) -> int:
    """
    Handle the 'freeze' command.
    """
    try:
        venv = VirtualEnvironment(Path(sys.prefix)) if args.output else None

        if venv and venv.exists():
            output = venv.freeze(output_file=args.output)
        else:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            output = result.stdout.strip()

            if args.output:
                args.output.write_text(output + "\n", encoding="utf-8")

        fmt.write(output)
        return 0

    except Exception as e:
        fmt.write(fmt.format_error(str(e)))
        return 1


def _handle_list(args: argparse.Namespace, config: InstallConfig, fmt: OutputFormatter) -> int:
    """
    Handle the 'list' command.
    """
    try:
        import subprocess
        import json

        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )

        packages = json.loads(result.stdout)

        if args.outdated:
            outdated_result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            packages = json.loads(outdated_result.stdout)

        formatted = []
        for pkg in packages:
            formatted.append([
                pkg.get("name", ""),
                pkg.get("version", ""),
            ])

        if hasattr(fmt, '_format_table'):
            output = fmt._format_table(
                ["Package", "Version"],
                formatted,
            )
            fmt.write(output)
        else:
            fmt.write(fmt.format_info({"packages": packages}))

        return 0

    except Exception as e:
        fmt.write(fmt.format_error(str(e)))
        return 1


def _handle_venv(args: argparse.Namespace, config: InstallConfig, fmt: OutputFormatter) -> int:
    """
    Handle the 'venv' command.
    """
    if not args.venv_command:
        fmt.write(fmt.format_error("Specify a venv command: create, destroy"))
        return 1

    try:
        if args.venv_command == "create":
            venv = VirtualEnvironment(
                args.path,
                python_executable=args.python,
                system_site_packages=args.system_site_packages,
                upgrade=args.upgrade_pip,
            )
            venv.create()
            fmt.write(fmt.format_info({
                "message": f"Virtual environment created at {args.path}",
                "path": str(args.path),
                "python": str(venv.python_path),
                "pip": str(venv.pip_path),
            }))
            return 0

        elif args.venv_command == "destroy":
            venv = VirtualEnvironment(args.path)
            if venv.exists():
                venv.destroy()
                fmt.write(fmt.format_info({
                    "message": f"Virtual environment at {args.path} destroyed",
                }))
            else:
                fmt.write(fmt.format_error(f"Environment not found: {args.path}"))
                return 1
            return 0

    except PackageInstallerError as e:
        fmt.write(fmt.format_error(str(e)))
        return 1


def _handle_check(args: argparse.Namespace, config: InstallConfig, fmt: OutputFormatter) -> int:
    """
    Handle the 'check' command.
    """
    if args.upgrades:
        try:
            import subprocess
            import json

            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )

            packages = json.loads(result.stdout)

            if not packages:
                fmt.write(fmt.format_info({"message": "All packages are up to date."}))
                return 0

            formatted = []
            for pkg in packages:
                formatted.append([
                    pkg.get("name", ""),
                    pkg.get("version", ""),
                    pkg.get("latest_version", ""),
                ])

            if hasattr(fmt, '_format_table'):
                output = fmt._format_table(
                    ["Package", "Current", "Latest"],
                    formatted,
                )
                fmt.write(output)
            else:
                fmt.write(fmt.format_info({"outdated": packages}))

            return 0

        except Exception as e:
            fmt.write(fmt.format_error(str(e)))
            return 1

    if args.integrity:
        if args.package:
            try:
                installer = PackageInstaller(args.package, config=config)
                installer.verify_integrity()
                fmt.write(fmt.format_info({
                    "message": f"Integrity verified for {args.package}",
                }))
                return 0
            except Exception as e:
                fmt.write(fmt.format_error(str(e)))
                return 1
        else:
            fmt.write(fmt.format_error("Specify a package for integrity check"))
            return 1

    fmt.write(fmt.format_error("Specify --upgrades or --integrity"))
    return 1


def main(args: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.

    Parameters
    ----------
    args : list of str, optional
        Command-line arguments. If None, uses ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for error).

    Notes
    -----
    This function parses command-line arguments, configures logging,
    and dispatches to the appropriate command handler.

    Examples
    --------
    >>> main(["install", "requests"])
    0
    """
    parser = _create_parser()

    if args is None:
        args = sys.argv[1:]

    parsed_args = parser.parse_args(args)

    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    verbosity = min(parsed_args.verbose, len(log_levels) - 1)
    logging.basicConfig(
        level=log_levels[verbosity],
        format="%(levelname)s: %(message)s",
    )

    if not parsed_args.command:
        parser.print_help()
        return 1

    fmt = _get_formatter(parsed_args)
    config = _get_config(parsed_args)

    handlers = {
        "install": _handle_install,
        "i": _handle_install,
        "uninstall": _handle_uninstall,
        "remove": _handle_uninstall,
        "rm": _handle_uninstall,
        "upgrade": _handle_upgrade,
        "up": _handle_upgrade,
        "search": _handle_search,
        "find": _handle_search,
        "info": _handle_info,
        "show": _handle_info,
        "freeze": _handle_freeze,
        "list": _handle_list,
        "ls": _handle_list,
        "venv": _handle_venv,
        "check": _handle_check,
    }

    handler = handlers.get(parsed_args.command)
    if handler is None:
        fmt.write(fmt.format_error(f"Unknown command: {parsed_args.command}"))
        return 1

    try:
        return handler(parsed_args, config, fmt)
    except KeyboardInterrupt:
        fmt.write(fmt.format_error("Operation cancelled by user"))
        return 130
    except Exception as e:
        fmt.write(fmt.format_error(f"Unexpected error: {e}"))
        logger.exception("Unhandled exception in CLI")
        return 1


if __name__ == "__main__":
    sys.exit(main())
"""
Command-line interface for stdlib_installer.

Provides a user-facing CLI for installing, removing, listing, and
managing standard library modules. Uses argparse for argument parsing
with no external dependencies required.
"""

import sys
import argparse
import logging
import textwrap
from pathlib import Path
from typing import Optional

from .installer import Installer
from .exceptions import (
    StdlibInstallerError,
    AlreadyInstalledError,
    ModuleNotInstalledError,
    CompiledModuleError,
    PackageNotFoundError,
    NetworkError,
)

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """
    Configure logging level and format based on verbosity flags.

    Parameters
    ----------
    verbose : bool
        If True, set log level to DEBUG.
    quiet : bool
        If True, suppress all log output except errors.
    """
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(levelname)s: %(message)s")
    )

    root_logger = logging.getLogger("stdlib_installer")
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def _create_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser with subcommands.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with all subcommands and options.
    """
    parser = argparse.ArgumentParser(
        prog="stdlib",
        description=textwrap.dedent(
            """\
            Install standard library modules from the CPython repository.

            Downloads pure-Python standard library modules directly from
            GitHub and installs them locally for use when the system Python
            installation is missing certain modules.
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              stdlib install json
              stdlib install xml --version 3.9
              stdlib install csv datetime collections
              stdlib remove json
              stdlib list
              stdlib info json
              stdlib check-updates
              stdlib install json --dry-run
            """
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version="stdlib-installer 2.0.0",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=None,
        help="Custom storage directory for installed packages",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        help="Available commands",
    )
    subparsers.required = True

    # install
    _add_install_parser(subparsers)

    # remove
    _add_remove_parser(subparsers)

    # list
    _add_list_parser(subparsers)

    # info
    _add_info_parser(subparsers)

    # check-updates
    _add_check_updates_parser(subparsers)

    # update
    _add_update_parser(subparsers)

    # clean-cache
    _add_clean_cache_parser(subparsers)

    return parser


def _add_install_parser(subparsers) -> None:
    """Add the 'install' subcommand parser."""
    parser = subparsers.add_parser(
        "install",
        help="Install one or more packages",
        description="Download and install standard library modules from CPython repository.",
    )
    parser.add_argument(
        "packages",
        nargs="+",
        help="Package name(s) to install (e.g., json xml.etree)",
    )
    parser.add_argument(
        "--version",
        "-V",
        help="CPython version to install from (default: current interpreter version)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force reinstallation even if already installed",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Skip automatic dependency resolution and installation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be installed without making changes",
    )


def _add_remove_parser(subparsers) -> None:
    """Add the 'remove' subcommand parser."""
    parser = subparsers.add_parser(
        "remove",
        help="Remove installed packages",
        description="Remove one or more installed standard library packages.",
        aliases=["rm", "uninstall"],
    )
    parser.add_argument(
        "packages",
        nargs="+",
        help="Package name(s) to remove",
    )


def _add_list_parser(subparsers) -> None:
    """Add the 'list' subcommand parser."""
    parser = subparsers.add_parser(
        "list",
        help="List installed packages",
        description="Display all installed standard library packages.",
        aliases=["ls"],
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--paths",
        action="store_true",
        help="Show installation paths",
    )


def _add_info_parser(subparsers) -> None:
    """Add the 'info' subcommand parser."""
    parser = subparsers.add_parser(
        "info",
        help="Show detailed package information",
        description="Display metadata for an installed package.",
    )
    parser.add_argument(
        "package",
        help="Package name",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )


def _add_check_updates_parser(subparsers) -> None:
    """Add the 'check-updates' subcommand parser."""
    subparsers.add_parser(
        "check-updates",
        help="Check for available updates",
        description="Check if installed packages have newer versions available.",
        aliases=["outdated"],
    )


def _add_update_parser(subparsers) -> None:
    """Add the 'update' subcommand parser."""
    parser = subparsers.add_parser(
        "update",
        help="Update installed packages",
        description="Update one or all installed packages to the current Python version.",
    )
    parser.add_argument(
        "packages",
        nargs="*",
        help="Package name(s) to update (omit to update all)",
    )
    parser.add_argument(
        "--version",
        "-V",
        help="Target CPython version (default: current interpreter version)",
    )


def _add_clean_cache_parser(subparsers) -> None:
    """Add the 'clean-cache' subcommand parser."""
    parser = subparsers.add_parser(
        "clean-cache",
        help="Clean the download cache",
        description="Remove cached download files to free disk space.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Remove all cached entries, not just expired ones",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show cache statistics without cleaning",
    )


def _get_installer(storage_dir: Optional[Path] = None) -> Installer:
    """
    Create an Installer instance, optionally with a custom storage directory.

    Parameters
    ----------
    storage_dir : Path or None
        Custom storage directory path.

    Returns
    -------
    Installer
        Configured installer instance.
    """
    kwargs = {}
    if storage_dir:
        kwargs["storage_dir"] = storage_dir
    return Installer(**kwargs)


def _handle_install(args: argparse.Namespace) -> int:
    """
    Execute the 'install' command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    installer = _get_installer(args.storage_dir)
    exit_code = 0

    for package in args.packages:
        try:
            path = installer.install(
                name=package,
                version=args.version,
                force=args.force,
                with_dependencies=not args.no_deps,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                print(f"[DRY RUN] Would install: {package}")
            else:
                print(f"Installed: {package} -> {path}")

        except AlreadyInstalledError as e:
            print(f"Skipped: {package} (already installed, use --force to reinstall)")
            logger.debug(str(e))

        except CompiledModuleError as e:
            print(f"Error: {e}", file=sys.stderr)
            exit_code = 1

        except PackageNotFoundError as e:
            print(f"Error: {package} not found in repository", file=sys.stderr)
            logger.debug(str(e))
            exit_code = 1

        except NetworkError as e:
            print(f"Network error: {e}", file=sys.stderr)
            exit_code = 1

        except StdlibInstallerError as e:
            print(f"Error installing {package}: {e}", file=sys.stderr)
            exit_code = 1

    return exit_code


def _handle_remove(args: argparse.Namespace) -> int:
    """
    Execute the 'remove' command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    installer = _get_installer(args.storage_dir)
    exit_code = 0

    for package in args.packages:
        try:
            installer.remove(package)
            print(f"Removed: {package}")
        except ModuleNotInstalledError as e:
            print(f"Skipped: {package} (not installed)")
            logger.debug(str(e))
        except StdlibInstallerError as e:
            print(f"Error removing {package}: {e}", file=sys.stderr)
            exit_code = 1

    return exit_code


def _handle_list(args: argparse.Namespace) -> int:
    """
    Execute the 'list' command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command arguments.

    Returns
    -------
    int
        Exit code.
    """
    import json

    installer = _get_installer(args.storage_dir)
    packages = installer.list_installed()

    if not packages:
        if args.json:
            print(json.dumps([]))
        else:
            print("No packages installed.")
        return 0

    if args.json:
        print(json.dumps(packages, indent=2))
        return 0

    # Find max name length for alignment
    max_name_len = max(len(p["name"]) for p in packages)

    print(f"{'Package':<{max_name_len}}  {'Version':<8}  {'Files':>5}  Installed")
    print("-" * (max_name_len + 38))

    for pkg in packages:
        installed_at = pkg.get("installed_at", "unknown")
        if installed_at != "unknown":
            try:
                # Truncate timestamp for readability
                installed_at = installed_at[:19].replace("T", " ")
            except (IndexError, TypeError):
                pass

        print(
            f"{pkg['name']:<{max_name_len}}  "
            f"{pkg['version']:<8}  "
            f"{pkg['file_count']:>5}  "
            f"{installed_at}"
        )

    print(f"\n{len(packages)} package(s) installed")
    return 0


def _handle_info(args: argparse.Namespace) -> int:
    """
    Execute the 'info' command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command arguments.

    Returns
    -------
    int
        Exit code.
    """
    import json

    installer = _get_installer(args.storage_dir)
    info = installer.get_package_info(args.package)

    if info is None:
        print(f"Package '{args.package}' is not installed.", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(info, indent=2, default=str))
        return 0

    print(f"Name:        {info.get('name', args.package)}")
    print(f"Version:     {info.get('version', 'unknown')}")
    print(f"Installed:   {info.get('installed_at', 'unknown')}")
    print(f"Files:       {len(info.get('files', []))}")

    deps = info.get("dependencies", [])
    if deps:
        print(f"Dependencies: {', '.join(deps)}")
    else:
        print("Dependencies: none")

    files = info.get("files", [])
    if files:
        print("\nInstalled files:")
        for f in sorted(files):
            print(f"  {f}")

    return 0


def _handle_check_updates(args: argparse.Namespace) -> int:
    """
    Execute the 'check-updates' command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command arguments.

    Returns
    -------
    int
        Exit code.
    """
    installer = _get_installer(args.storage_dir)
    updates = installer.check_updates()

    if not updates:
        print("All packages are up to date.")
        return 0

    max_name_len = max(len(u["name"]) for u in updates)

    print(f"{'Package':<{max_name_len}}  {'Installed':<10}  {'Available':<10}")
    print("-" * (max_name_len + 32))

    for update in updates:
        print(
            f"{update['name']:<{max_name_len}}  "
            f"{update['installed_version']:<10}  "
            f"{update['available_version']:<10}"
        )

    print(f"\n{len(updates)} package(s) can be updated. Use 'stdlib update' to update.")
    return 0


def _handle_update(args: argparse.Namespace) -> int:
    """
    Execute the 'update' command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command arguments.

    Returns
    -------
    int
        Exit code.
    """
    installer = _get_installer(args.storage_dir)
    exit_code = 0

    # If no packages specified, update all outdated
    if not args.packages:
        updates = installer.check_updates()
        args.packages = [u["name"] for u in updates]

        if not args.packages:
            print("All packages are up to date.")
            return 0

    for package in args.packages:
        try:
            path = installer.update(package, version=args.version)
            print(f"Updated: {package} -> {path}")
        except ModuleNotInstalledError:
            print(f"Skipped: {package} (not installed)")
        except StdlibInstallerError as e:
            print(f"Error updating {package}: {e}", file=sys.stderr)
            exit_code = 1

    return exit_code


def _handle_clean_cache(args: argparse.Namespace) -> int:
    """
    Execute the 'clean-cache' command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command arguments.

    Returns
    -------
    int
        Exit code.
    """
    from .downloader import CacheManager

    cache = CacheManager()
    stats = cache.stats()

    if args.stats:
        print(f"Cache directory: {stats['cache_dir']}")
        print(f"Entries:        {stats['entry_count']}")
        print(f"Size:           {stats['total_size_mb']} MB")
        print(f"TTL:            {stats['ttl_seconds']}s")
        return 0

    if args.all:
        removed = cache.clear()
        print(f"Cleared all {removed} cached entries ({stats['total_size_mb']} MB freed)")
    else:
        removed = cache.clear_expired()
        print(f"Cleared {removed} expired cached entries")

    return 0


def main(argv: Optional[list] = None) -> int:
    """
    Entry point for the stdlib CLI.

    Parses arguments, configures logging, and dispatches to the
    appropriate command handler.

    Parameters
    ----------
    argv : list or None, optional
        Command-line arguments (defaults to sys.argv[1:]).

    Returns
    -------
    int
        Exit code (0 for success, non-zero for errors).
    """
    parser = _create_parser()

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1

    _setup_logging(verbose=args.verbose, quiet=args.quiet)

    # Dispatch table mapping commands to handlers
    handlers = {
        "install": _handle_install,
        "remove": _handle_remove,
        "rm": _handle_remove,
        "uninstall": _handle_remove,
        "list": _handle_list,
        "ls": _handle_list,
        "info": _handle_info,
        "check-updates": _handle_check_updates,
        "outdated": _handle_check_updates,
        "update": _handle_update,
        "clean-cache": _handle_clean_cache,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        return 130
    except Exception as e:
        logger.debug(f"Unexpected error: {e}", exc_info=True)
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
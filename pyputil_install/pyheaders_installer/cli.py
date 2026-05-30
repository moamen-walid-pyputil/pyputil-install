#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command-Line Interface for Python Headers Installer.

Provides a terminal-based interface for installing Python C development
headers. Accepts arguments for version selection, download source,
target directory, and operational flags. Translates parsed arguments
into configuration objects and executes the installation pipeline.

The CLI supports three output modes:
    standard  - Human-readable progress and summary.
    quiet     - Only errors and the final result.
    json      - Machine-parseable JSON on stdout, logs on stderr.

Exit Codes
----------
0   SUCCESS              Installation completed successfully.
1   GENERAL_ERROR        Unspecified runtime error.
2   ARGUMENT_ERROR       Invalid or contradictory command-line arguments.
3   DOWNLOAD_ERROR       Failed to download Python source archive.
4   EXTRACTION_ERROR     Failed to extract the downloaded archive.
5   INSTALLATION_ERROR   Failed to copy header files to target.
6   VERIFICATION_ERROR   Post-installation header verification failed.

Argument Groups
---------------
Required (mutually exclusive with --info and --clean):
    (none, install is the default action)

Optional:
    --version VERSION     Python version (default: current interpreter version).
    --target-dir PATH     Target include directory (default: system include).
    --source SOURCE       Download source: github, python.org, or auto.
    --custom-url URL      Custom download URL with {version} placeholder.
    --retries N           Maximum download retries (default: 3).
    --timeout SECONDS     Connection timeout (default: 30).

Flags:
    --verbose, -v         Detailed progress output.
    --quiet, -q           Suppress non-error output.
    --json                Output final result as JSON.
    --clean-existing      Remove existing headers before installation.
    --no-backup           Do not back up existing headers.
    --no-subdirs          Do not copy subdirectories (cpython/, internal/).
    --no-verify           Skip post-installation header verification.
    --atomic              Use atomic installation (default).
    --no-atomic           Disable atomic installation.

Actions (mutually exclusive):
    --info                Display system and Python information, then exit.
    --clean               Remove temporary and checkpoint files, then exit.

Examples
--------
    # Install headers for the current Python version
    python -m pyheaders_installer

    # Install headers for a specific version with verbose output
    python -m pyheaders_installer --version 3.11.0 --verbose

    # Install from python.org to a custom directory
    python -m pyheaders_installer --source python.org --target-dir ./headers

    # Install with backup disabled and existing headers cleaned
    python -m pyheaders_installer --clean-existing --no-backup

    # Display system information
    python -m pyheaders_installer --info

    # Clean temporary files
    python -m pyheaders_installer --clean

    # Machine-readable JSON output
    python -m pyheaders_installer --json --quiet

Warnings
--------
- Installing to system directories (e.g., /usr/include) may require
  elevated privileges. The CLI does not escalate permissions.
- The --clean-existing flag irreversibly removes the target directory
  before installation. Use --backup (the default) to preserve data.
- Custom URLs must contain the {version} placeholder. The placeholder
  is replaced with the requested Python version string.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
from pathlib import Path
from typing import Optional, Sequence, List

from .config import (
    InstallConfig,
    NetworkConfig,
    VerificationConfig,
    BackupConfig,
    PathConfig,
    ExtractionConfig,
    DownloadSource,
    HashAlgorithm,
    LogLevel,
)
from .exceptions import (
    HeaderInstallError,
    DownloadError,
    ExtractionError,
    InstallationError,
    VerificationError,
)

logger = logging.getLogger(__name__)


class ExitCode:
    """Exit codes returned to the shell."""

    SUCCESS: int = 0
    GENERAL_ERROR: int = 1
    ARGUMENT_ERROR: int = 2
    DOWNLOAD_ERROR: int = 3
    EXTRACTION_ERROR: int = 4
    INSTALLATION_ERROR: int = 5
    VERIFICATION_ERROR: int = 6


class OutputFormatter:
    """
    Formats and writes CLI output according to the selected mode.

    Parameters
    ----------
    mode : str
        Output mode: 'standard', 'quiet', or 'json'.
    use_colors : bool
        Whether to emit ANSI color codes (auto-detected if None).

    Examples
    --------
    >>> formatter = OutputFormatter(mode='standard')
    >>> formatter.print_info("Starting installation")
    Starting installation
    """

    def __init__(self, mode: str = 'standard', use_colors: Optional[bool] = None) -> None:
        self.mode = mode
        self.use_colors = use_colors if use_colors is not None else self._detect_color()
        self._json_data: dict = {}

    @staticmethod
    def _detect_color() -> bool:
        """
        Detect whether the terminal supports ANSI color codes.

        Returns False when output is redirected to a file or pipe.
        Returns False on Windows unless colorama is installed.

        Returns
        -------
        bool
            True if color output should be emitted.
        """
        if not sys.stdout.isatty():
            return False
        if os.name == 'nt':
            try:
                import colorama
                colorama.init()
                return True
            except ImportError:
                return False
        return True

    def _color(self, text: str, code: str) -> str:
        """
        Wrap text in ANSI color codes if colors are enabled.

        Parameters
        ----------
        text : str
            Text to colorize.
        code : str
            ANSI escape code (e.g., '91' for red).

        Returns
        -------
        str
            Colorized text or plain text.
        """
        if self.use_colors:
            return f"\033[{code}m{text}\033[0m"
        return text

    def print_info(self, message: str) -> None:
        """
        Print an informational message.

        Suppressed in quiet and json modes.

        Parameters
        ----------
        message : str
            Message to print.
        """
        if self.mode == 'standard':
            print(message)

    def print_success(self, message: str) -> None:
        """
        Print a success message in green.

        Parameters
        ----------
        message : str
            Message to print.
        """
        if self.mode == 'standard':
            print(self._color(message, '92'))
        elif self.mode == 'json':
            self._json_data['status'] = 'success'
            self._json_data['message'] = message

    def print_warning(self, message: str) -> None:
        """
        Print a warning message in yellow.

        Parameters
        ----------
        message : str
            Message to print.
        """
        if self.mode in ('standard', 'quiet'):
            print(self._color(f"Warning: {message}", '93'), file=sys.stderr)
        if self.mode == 'json':
            self._json_data.setdefault('warnings', []).append(message)

    def print_error(self, message: str) -> None:
        """
        Print an error message in red.

        Parameters
        ----------
        message : str
            Message to print.
        """
        if self.mode in ('standard', 'quiet'):
            print(self._color(f"Error: {message}", '91'), file=sys.stderr)
        if self.mode == 'json':
            self._json_data['status'] = 'error'
            self._json_data['error'] = message

    def print_progress(self, message: str) -> None:
        """
        Print a progress update.

        In standard mode, replaces the current line.
        In other modes, this is suppressed.

        Parameters
        ----------
        message : str
            Progress message.
        """
        if self.mode == 'standard':
            print(f"\r{message}", end='', flush=True)

    def print_json_result(self) -> None:
        """
        Print accumulated JSON data to stdout.

        Used at the end of a json-mode run.
        """
        if self.mode == 'json':
            json.dump(self._json_data, sys.stdout, indent=2)
            sys.stdout.write('\n')
            sys.stdout.flush()

    def add_json_field(self, key: str, value) -> None:
        """
        Add a field to the JSON output.

        Parameters
        ----------
        key : str
            Field name.
        value : any
            Field value. Must be JSON-serializable.
        """
        if self.mode == 'json':
            self._json_data[key] = value


class ArgumentParser:
    """
    Builds and validates the command-line argument parser.

    Constructs an argparse.ArgumentParser with all supported
    arguments, mutually exclusive groups, and validation logic.

    Examples
    --------
    >>> parser = ArgumentParser()
    >>> args = parser.parse_args(['--version', '3.11.0', '--verbose'])
    >>> args.version
    '3.11.0'
    >>> args.verbose
    True
    """

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            prog='python-headers-installer',
            description=(
                'Install Python C development headers from source. '
                'Downloads CPython source, extracts header files, '
                'and installs them to the target include directory.'
            ),
            epilog=(
                'Exit codes: 0=success, 1=general error, 2=argument error, '
                '3=download error, 4=extraction error, 5=installation error, '
                '6=verification error.'
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self._build_arguments()

    def _build_arguments(self) -> None:
        """Register all argument groups and flags."""
        self._add_optional_arguments()
        self._add_flag_arguments()
        self._add_output_arguments()
        self._add_action_arguments()
        self._add_advanced_arguments()

    def _add_optional_arguments(self) -> None:
        """Register optional string/numeric arguments."""
        group = self.parser.add_argument_group('optional arguments')
        group.add_argument(
            '--version',
            type=self._validate_version,
            default=None,
            help=(
                'Python version to install headers for '
                '(e.g., 3.11.0). Defaults to the current '
                'interpreter version.'
            ),
            metavar='VERSION',
        )
        group.add_argument(
            '--target-dir',
            type=Path,
            default=None,
            help=(
                'Target include directory. Defaults to the system '
                'include path for the current Python installation.'
            ),
            metavar='PATH',
        )
        group.add_argument(
            '--source',
            type=str,
            choices=['github', 'python.org', 'auto'],
            default='github',
            help=(
                'Download source repository. github fetches from '
                'github.com/python/cpython as ZIP. python.org fetches '
                'from www.python.org as tar.xz. Default: github.'
            ),
        )
        group.add_argument(
            '--custom-url',
            type=str,
            default=None,
            help=(
                'Custom download URL template. Must contain {version} '
                'placeholder. Overrides --source.'
            ),
            metavar='URL',
        )

    def _add_flag_arguments(self) -> None:
        """Register boolean flag arguments."""
        group = self.parser.add_argument_group('flags')
        group.add_argument(
            '--verbose', '-v',
            action='store_true',
            default=False,
            help='Enable detailed progress output.',
        )
        group.add_argument(
            '--quiet', '-q',
            action='store_true',
            default=False,
            help='Suppress non-error output.',
        )
        group.add_argument(
            '--json',
            action='store_true',
            default=False,
            help='Output final result as JSON on stdout.',
        )
        group.add_argument(
            '--clean-existing',
            action='store_true',
            default=False,
            help='Remove existing headers before installation.',
        )
        group.add_argument(
            '--no-backup',
            action='store_true',
            default=False,
            help='Do not back up existing headers.',
        )
        group.add_argument(
            '--no-subdirs',
            action='store_true',
            default=False,
            help='Do not copy subdirectories (cpython/, internal/).',
        )
        group.add_argument(
            '--no-verify',
            action='store_true',
            default=False,
            help='Skip post-installation header verification.',
        )
        group.add_argument(
            '--atomic',
            action='store_true',
            default=True,
            help='Use atomic installation (default).',
        )
        group.add_argument(
            '--no-atomic',
            action='store_true',
            default=False,
            help='Disable atomic installation.',
        )

    def _add_output_arguments(self) -> None:
        """Register output and logging arguments."""
        group = self.parser.add_argument_group('output options')
        group.add_argument(
            '--log-level',
            type=str,
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            default='INFO',
            help='Set logging level (default: INFO).',
        )
        group.add_argument(
            '--log-file',
            type=Path,
            default=None,
            help='Write log output to a file.',
            metavar='PATH',
        )
        group.add_argument(
            '--no-color',
            action='store_true',
            default=False,
            help='Disable ANSI color output.',
        )

    def _add_action_arguments(self) -> None:
        """Register mutually exclusive action arguments."""
        action_group = self.parser.add_mutually_exclusive_group()
        action_group.add_argument(
            '--info',
            action='store_true',
            default=False,
            help='Display system and Python information, then exit.',
        )
        action_group.add_argument(
            '--clean',
            action='store_true',
            default=False,
            help='Remove temporary and checkpoint files, then exit.',
        )

    def _add_advanced_arguments(self) -> None:
        """Register advanced configuration arguments."""
        group = self.parser.add_argument_group('advanced options')
        group.add_argument(
            '--retries',
            type=int,
            default=3,
            help='Maximum download retry attempts (default: 3).',
            metavar='N',
        )
        group.add_argument(
            '--timeout',
            type=float,
            default=30.0,
            help='Connection timeout in seconds (default: 30).',
            metavar='SECONDS',
        )
        group.add_argument(
            '--chunk-size',
            type=int,
            default=8192,
            help='Download chunk size in bytes (default: 8192).',
            metavar='BYTES',
        )
        group.add_argument(
            '--verify-algorithm',
            type=str,
            choices=['sha256', 'sha384', 'sha512', 'md5'],
            default='sha256',
            help='Hash algorithm for file verification (default: sha256).',
        )
        group.add_argument(
            '--state-dir',
            type=Path,
            default=None,
            help='Directory for checkpoint files.',
            metavar='PATH',
        )

    @staticmethod
    def _validate_version(value: str) -> str:
        """
        Validate a Python version string.

        Accepts formats: '3.11', '3.11.0', '3.11.0rc1'.
        Rejects: 'python3.11', '3.x', 'latest'.

        Parameters
        ----------
        value : str
            Version string from command line.

        Returns
        -------
        str
            Validated version string.

        Raises
        ------
        argparse.ArgumentTypeError
            If the format is invalid.
        """
        import re
        pattern = r'^\d+\.\d+(\.\d+)?([a-z]+\d*)?$'
        if not re.match(pattern, value):
            raise argparse.ArgumentTypeError(
                f"Invalid version format: '{value}'. "
                f"Expected format: major.minor[.patch] (e.g., 3.11.0)"
            )
        return value

    def parse_args(self, args: Optional[Sequence[str]] = None) -> argparse.Namespace:
        """
        Parse command-line arguments.

        Parameters
        ----------
        args : Optional[Sequence[str]]
            Argument list. Uses sys.argv[1:] if None.

        Returns
        -------
        argparse.Namespace
            Parsed and validated arguments.

        Raises
        ------
        SystemExit
            If arguments are invalid.
        """
        parsed = self.parser.parse_args(args)

        # Additional cross-argument validation
        if parsed.quiet and parsed.verbose:
            self.parser.error("Cannot use both --quiet and --verbose")

        if parsed.no_atomic and parsed.atomic:
            self.parser.error("Cannot use both --atomic and --no-atomic")

        if parsed.json and parsed.verbose:
            self.parser.error("Cannot use both --json and --verbose")

        return parsed


class CliRunner:
    """
    Executes CLI commands based on parsed arguments.

    Translates argparse.Namespace into InstallConfig objects,
    invokes the appropriate command handler, and formats output.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    formatter : OutputFormatter
        Output formatting strategy.

    Examples
    --------
    >>> parser = ArgumentParser()
    >>> args = parser.parse_args(['--info'])
    >>> runner = CliRunner(args)
    >>> exit_code = runner.run()
    System Information:
      ...
    >>> exit_code
    0
    """

    def __init__(
        self,
        args: argparse.Namespace,
        formatter: Optional[OutputFormatter] = None,
    ) -> None:
        self.args = args
        self.formatter = formatter or self._create_formatter()

    def _create_formatter(self) -> OutputFormatter:
        """
        Create an OutputFormatter from parsed arguments.

        Returns
        -------
        OutputFormatter
            Configured formatter.
        """
        if self.args.json:
            mode = 'json'
        elif self.args.quiet:
            mode = 'quiet'
        else:
            mode = 'standard'

        return OutputFormatter(mode=mode, use_colors=not self.args.no_color)

    def _configure_logging(self) -> None:
        """
        Set up logging based on CLI arguments.

        Configures log level, format, and optional file output.
        """
        level_name = self.args.log_level.upper()
        level = getattr(logging, level_name, logging.INFO)

        handlers: List[logging.Handler] = []

        if self.args.log_file:
            file_handler = logging.FileHandler(self.args.log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
            )
            file_handler.setLevel(level)
            handlers.append(file_handler)

        if not self.args.quiet or self.args.log_file is None:
            console_handler = logging.StreamHandler(sys.stderr)
            if self.args.verbose:
                fmt = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
            else:
                fmt = '%(levelname)s: %(message)s'
            console_handler.setFormatter(logging.Formatter(fmt))
            console_handler.setLevel(level)
            handlers.append(console_handler)

        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        for handler in handlers:
            root_logger.addHandler(handler)

    def run(self) -> int:
        """
        Dispatch to the appropriate command handler.

        Returns
        -------
        int
            Exit code (0 on success).
        """
        self._configure_logging()

        try:
            if self.args.info:
                return self._cmd_info()
            elif self.args.clean:
                return self._cmd_clean()
            else:
                return self._cmd_install()
        except HeaderInstallError as e:
            self.formatter.print_error(str(e))
            exit_code = self._map_exception_to_exit_code(e)
            self.formatter.add_json_field('exit_code', exit_code)
            self.formatter.print_json_result()
            return exit_code
        except KeyboardInterrupt:
            self.formatter.print_warning("Interrupted by user")
            self.formatter.print_json_result()
            return ExitCode.GENERAL_ERROR
        except Exception as e:
            self.formatter.print_error(f"Unexpected error: {e}")
            logger.debug("Unexpected error traceback:", exc_info=True)
            self.formatter.print_json_result()
            return ExitCode.GENERAL_ERROR

    def _cmd_info(self) -> int:
        """
        Display system and Python information.

        Returns
        -------
        int
            ExitCode.SUCCESS.

        Examples
        --------
        >>> parser = ArgumentParser()
        >>> args = parser.parse_args(['--info'])
        >>> CliRunner(args).run()
        System Information:
          ...
        0
        """
        import platform
        import sysconfig

        info = {
            'os': platform.system(),
            'os_version': platform.release(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'python_executable': sys.executable,
            'python_prefix': sys.prefix,
            'include_directory': sysconfig.get_paths().get('include', 'Not found'),
            'stdlib_directory': sysconfig.get_paths().get('stdlib', 'Not found'),
        }

        if self.args.json:
            self.formatter.add_json_field('system_info', info)
            self.formatter.print_json_result()
        else:
            self.formatter.print_info("System Information:")
            self.formatter.print_info(f"  OS:              {info['os']} {info['os_version']}")
            self.formatter.print_info(f"  Architecture:    {info['architecture']}")
            self.formatter.print_info(f"  Python:          {info['python_implementation']} {info['python_version']}")
            self.formatter.print_info(f"  Executable:      {info['python_executable']}")
            self.formatter.print_info(f"  Prefix:          {info['python_prefix']}")
            self.formatter.print_info(f"  Include Dir:     {info['include_directory']}")
            self.formatter.print_info(f"  Stdlib Dir:      {info['stdlib_directory']}")

        return ExitCode.SUCCESS

    def _cmd_clean(self) -> int:
        """
        Remove temporary and checkpoint files.

        Searches default cache directories for files matching
        the installer's naming patterns and removes them.

        Returns
        -------
        int
            ExitCode.SUCCESS if cleaning completed.

        Examples
        --------
        >>> parser = ArgumentParser()
        >>> args = parser.parse_args(['--clean'])
        >>> CliRunner(args).run()
        Cleaned up temporary files.
        0
        """
        import shutil

        dirs_to_check = [
            Path.home() / '.cache' / 'pyheaders_installer',
            Path.home() / '.local' / 'share' / 'pyheaders_installer',
        ]

        cleaned_count = 0
        for directory in dirs_to_check:
            if directory.exists():
                try:
                    shutil.rmtree(directory)
                    cleaned_count += 1
                    logger.info(f"Removed: {directory}")
                except OSError as e:
                    logger.warning(f"Could not remove {directory}: {e}")

        if cleaned_count > 0:
            self.formatter.print_success(f"Cleaned {cleaned_count} cache directories.")
        else:
            self.formatter.print_info("No temporary files found.")

        if self.args.json:
            self.formatter.add_json_field('cleaned_directories', cleaned_count)
            self.formatter.print_json_result()

        return ExitCode.SUCCESS

    def _cmd_install(self) -> int:
        """
        Execute the header installation pipeline.

        Builds an InstallConfig from CLI arguments, runs the
        installation, and reports results.

        Returns
        -------
        int
            Exit code based on installation outcome.
        """
        config = self._build_install_config()

        self.formatter.print_info(f"Installing Python {config.version or '(current)'} headers...")
        self.formatter.print_info(f"Source: {config.source.value}")
        self.formatter.print_info(f"Target: {config.paths.resolve_target_dir()}")

        if self.args.verbose:
            self.formatter.print_info(f"Retries: {config.network.retries}")
            self.formatter.print_info(f"Timeout: {config.network.timeout}s")
            self.formatter.print_info(f"Verification: {config.verification.algorithm.value}")
            self.formatter.print_info(f"Atomic: {not self.args.no_atomic}")

        try:
            from .__init__ import install_python_headers

            result_path = install_python_headers(
                version=config.version,
                target_dir=config.paths.resolve_target_dir(),
                retries=config.network.retries,
                retry_delay=config.network.retry_delay,
                clean_existing=config.clean_existing,
                backup_existing=config.backup.enabled,
                verbose=config.verbose,
                include_subdirs=config.include_subdirs,
                source=config.source.value,
                custom_url=config.custom_url,
            )

            if result_path:
                self.formatter.print_success(
                    f"Headers installed successfully to: {result_path}"
                )
                self.formatter.add_json_field('installed_to', str(result_path))
                self.formatter.print_json_result()
                return ExitCode.SUCCESS
            else:
                self.formatter.print_error("Installation returned no path")
                self.formatter.print_json_result()
                return ExitCode.INSTALLATION_ERROR

        except DownloadError as e:
            self.formatter.print_error(f"Download failed: {e}")
            self.formatter.add_json_field('error_details', e.to_dict() if hasattr(e, 'to_dict') else str(e))
            self.formatter.print_json_result()
            return ExitCode.DOWNLOAD_ERROR

        except ExtractionError as e:
            self.formatter.print_error(f"Extraction failed: {e}")
            self.formatter.print_json_result()
            return ExitCode.EXTRACTION_ERROR

        except InstallationError as e:
            self.formatter.print_error(f"Installation failed: {e}")
            self.formatter.print_json_result()
            return ExitCode.INSTALLATION_ERROR

        except VerificationError as e:
            self.formatter.print_error(f"Verification failed: {e}")
            self.formatter.print_json_result()
            return ExitCode.VERIFICATION_ERROR

    def _build_install_config(self) -> InstallConfig:
        """
        Build an InstallConfig from parsed CLI arguments.

        Returns
        -------
        InstallConfig
            Configuration object ready for the installer.

        Examples
        --------
        >>> parser = ArgumentParser()
        >>> args = parser.parse_args(['--version', '3.11.0'])
        >>> config = CliRunner(args)._build_install_config()
        >>> config.version
        '3.11.0'
        """
        # Determine download source
        source = DownloadSource(self.args.source)
        custom_url = self.args.custom_url

        if custom_url:
            source = DownloadSource.CUSTOM

        # Build network config
        network = NetworkConfig(
            retries=self.args.retries,
            timeout=self.args.timeout,
            chunk_size=self.args.chunk_size,
        )

        # Build verification config
        verification = VerificationConfig(
            enabled=not self.args.no_verify,
            algorithm=HashAlgorithm(self.args.verify_algorithm),
        )

        # Build backup config
        backup = BackupConfig(
            enabled=not self.args.no_backup,
        )

        # Build path config
        paths = PathConfig(
            target_include_dir=self.args.target_dir,
        )

        # Build extraction config
        extraction = ExtractionConfig()

        # Determine log level
        log_level_name = self.args.log_level.upper()
        log_level = LogLevel(log_level_name)

        return InstallConfig(
            version=self.args.version,
            source=source,
            custom_url=custom_url,
            clean_existing=self.args.clean_existing,
            include_subdirs=not self.args.no_subdirs,
            verbose=self.args.verbose,
            network=network,
            verification=verification,
            backup=backup,
            paths=paths,
            extraction=extraction,
            log_level=log_level,
        )

    @staticmethod
    def _map_exception_to_exit_code(exc: Exception) -> int:
        """
        Map an exception type to the appropriate exit code.

        Parameters
        ----------
        exc : Exception
            The raised exception.

        Returns
        -------
        int
            Corresponding exit code from ExitCode.
        """
        mapping = {
            DownloadError: ExitCode.DOWNLOAD_ERROR,
            ExtractionError: ExitCode.EXTRACTION_ERROR,
            InstallationError: ExitCode.INSTALLATION_ERROR,
            VerificationError: ExitCode.VERIFICATION_ERROR,
        }
        for exc_type, code in mapping.items():
            if isinstance(exc, exc_type):
                return code
        return ExitCode.GENERAL_ERROR


def main(args: Optional[Sequence[str]] = None) -> int:
    """
    Entry point for the command-line interface.

    Parses arguments, creates a runner, and executes the command.
    Handles ArgumentError and SystemExit for invalid input.

    Parameters
    ----------
    args : Optional[Sequence[str]]
        Command-line arguments. Uses sys.argv[1:] if None.

    Returns
    -------
    int
        Exit code (0 on success, non-zero on error).

    Examples
    --------
    >>> main(['--info'])
    System Information:
      ...
    0

    >>> main(['--invalid-flag'])
    2
    """
    parser = ArgumentParser()

    try:
        parsed = parser.parse_args(args)
    except SystemExit as e:
        if e.code == 0:
            return ExitCode.SUCCESS
        return ExitCode.ARGUMENT_ERROR

    formatter = OutputFormatter(
        mode=(
            'json' if parsed.json
            else 'quiet' if parsed.quiet
            else 'standard'
        ),
        use_colors=not parsed.no_color,
    )

    runner = CliRunner(parsed, formatter)
    return runner.run()


if __name__ == '__main__':
    sys.exit(main())
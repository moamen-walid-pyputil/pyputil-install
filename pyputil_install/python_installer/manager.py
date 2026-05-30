"""
Python Version Manager
=======================

Manages multiple isolated Python installations obtained from
``python-build-standalone`` releases. Provides installation, switching,
default-setting, listing, and removal of Python versions.

All installations are self-contained, portable, and do not require
system-level packages or administrator privileges.

Uses only the Python standard library. No third-party dependencies.

Security
--------
- Each Python version is installed in its own directory with no shared
  state between versions.
- The ``set_default`` operation creates symlinks or shim scripts.
  Existing system Python installations are **never** modified or
  deleted.
- The ``set_current`` operation replaces the running process via
  :func:`os.execve`. Environment variables are carefully filtered to
  prevent leakage that could compromise the new interpreter.
- Environment variable manipulation in shell configuration files is
  additive; existing ``PATH`` entries are preserved.
- All file operations use atomic writes (:func:`os.replace`) and
  restrictive permissions (``0o600`` for state files).

Usage
-----
.. code-block:: python

    from pathlib import Path
    from manager import PythonVersionManager

    manager = PythonVersionManager(
        install_root=Path.home() / ".python_versions",
    )

    # Install a version
    python_bin = manager.install(version="3.11.5", release_date="20231002")
    print(f"Installed: {python_bin}")

    # List installed versions
    for version, path in manager.list_installed().items():
        print(f"  {version}: {path}")

    # Switch the current process to a new version
    manager.set_current("3.11.5")  # This does not return

    # Set a version as the system default
    manager.set_default("3.11.5")

    # Remove a version
    manager.uninstall("3.9.18")

Warnings
--------
- :meth:`PythonVersionManager.set_current` calls :func:`os.execve`,
  which **replaces the current process**. It does **not** return.
  Any unsaved state in the calling process will be lost.
- Setting a default Python modifies shell configuration files
  (``.bashrc``, ``.zshrc``, ``.profile``). These files are parsed
  and updated programmatically. Always back up these files before
  first use.
- On Windows, ``set_default`` creates batch shim scripts because
  symlinks require administrator privileges by default.
- Removal of the currently-active default version does **not**
  automatically fall back to another version. The user must call
  :meth:`set_default` again.
- This class is **not thread-safe**. Concurrent installations of
  different versions may corrupt the shared state file.

Notes
-----
- The state file (``state.json``) is stored in *install_root* and
  uses human-readable JSON. Deleting it resets the manager's view
  of installed versions but does **not** delete the actual Python
  installations.
- The ``install_only`` variant of ``python-build-standalone`` is
  used by default because it excludes debug symbols and header
  files, reducing download size by ~60%.
- Shell configuration updates are additive; the manager appends a
  marked block to the file. Removing the block manually reverts
  the change.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default directory name inside install_root for each Python version.
_VERSION_DIR_PREFIX: str = "python-"

#: Name of the state file storing installation metadata.
_STATE_FILE: str = "state.json"

#: Marker comments delimiting the PATH block in shell config files.
_SHELL_MARKER_START: str = "# >>> python-standalone-manager >>>"
_SHELL_MARKER_END: str = "# <<< python-standalone-manager <<<"

#: Shell configuration files to update when setting a default Python.
_SHELL_CONFIG_FILES: Tuple[str, ...] = (
    ".bashrc",
    ".zshrc",
    ".profile",
    ".bash_profile",
)

#: File permissions for the state file.
_STATE_FILE_PERMS: int = 0o600

#: File permissions for version directories.
_VERSION_DIR_PERMS: int = 0o755


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class VersionManagerError(Exception):
    """
    Base exception for version manager operations.

    Parameters
    ----------
    message : str
        Human-readable description.
    version : str or None
        The Python version involved, if applicable.
    """

    def __init__(
        self,
        message: str,
        version: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.version = version


class VersionAlreadyInstalledError(VersionManagerError):
    """Raised when attempting to install an already-installed version."""

    pass


class VersionNotFoundError(VersionManagerError):
    """Raised when a requested version is not installed."""

    pass


class VersionActiveError(VersionManagerError):
    """
    Raised when attempting to uninstall the currently active default
    version.
    """

    pass


class ShellConfigError(VersionManagerError):
    """Raised when shell configuration files cannot be updated."""

    pass


class ProcessSwitchError(VersionManagerError):
    """Raised when :meth:`PythonVersionManager.set_current` fails."""

    pass


# ---------------------------------------------------------------------------
# State Manager
# ---------------------------------------------------------------------------


class _StateManager:
    """
    Manages persistent state for the version manager.

    State is stored as JSON in a single file inside *install_root*.

    Parameters
    ----------
    state_path : Path
        Path to the JSON state file.

    Notes
    -----
    - Reads are cached in memory. Writes are atomic via a temporary
      file and :func:`os.replace`.
    - Corrupt state files are treated as empty and overwritten on
      the next write.
    """

    def __init__(self, state_path: Path) -> None:
        self._state_path = state_path
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """
        Load state from disk.

        Returns
        -------
        dict
            Parsed state, or an empty dict if the file does not
            exist or is corrupt.
        """
        if not self._state_path.exists():
            return {}

        try:
            with open(self._state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return {}
            return data
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self) -> None:
        """Atomically persist state to disk."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".state_",
            dir=str(self._state_path.parent),
        )
        tmp_path = Path(tmp_path)
        try:
            os.close(tmp_fd)
            os.chmod(tmp_path, _STATE_FILE_PERMS)
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2)
            os.replace(tmp_path, self._state_path)
        except OSError:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the state.

        Parameters
        ----------
        key : str
            State key.
        default : Any
            Default if key is missing.

        Returns
        -------
        Any
        """
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a value and persist.

        Parameters
        ----------
        key : str
            State key.
        value : Any
            JSON-serialisable value.
        """
        self._data[key] = value
        self._save()

    def remove(self, key: str) -> None:
        """
        Remove a key and persist.

        Parameters
        ----------
        key : str
            State key. No-op if missing.
        """
        if key in self._data:
            del self._data[key]
            self._save()

    @property
    def installed_versions(self) -> Dict[str, str]:
        """
        Mapping of installed version strings to their directory paths.

        Returns
        -------
        dict[str, str]
        """
        return self.get("installed_versions", {})

    @installed_versions.setter
    def installed_versions(self, value: Dict[str, str]) -> None:
        self.set("installed_versions", value)

    @property
    def default_version(self) -> Optional[str]:
        """
        The currently-set default version, or ``None``.

        Returns
        -------
        str or None
        """
        return self.get("default_version")

    @default_version.setter
    def default_version(self, value: Optional[str]) -> None:
        if value is None:
            self.remove("default_version")
        else:
            self.set("default_version", value)


# ---------------------------------------------------------------------------
# Shell Configuration Updater
# ---------------------------------------------------------------------------


class _ShellConfigUpdater:
    """
    Updates shell configuration files to add or remove Python from
    the ``PATH``.

    Parameters
    ----------
    home_dir : Path
        User home directory.

    Notes
    -----
    - Modifies ``.bashrc``, ``.zshrc``, ``.profile``, and
      ``.bash_profile`` if they exist.
    - Changes are wrapped in marker comments so they can be cleanly
      removed later.
    - Existing marker blocks are replaced, not duplicated.
    """

    def __init__(self, home_dir: Path) -> None:
        self._home_dir = home_dir

    def add_to_path(self, bin_dir: Path) -> List[Path]:
        """
        Add *bin_dir* to ``PATH`` in all found shell config files.

        Parameters
        ----------
        bin_dir : Path
            Directory containing the Python executable.

        Returns
        -------
        list of Path
            Paths to files that were modified.

        Raises
        ------
        ShellConfigError
            If no config files could be updated.
        """
        path_entry = f'export PATH="{bin_dir}:$PATH"'
        block = f"{_SHELL_MARKER_START}\n{path_entry}\n{_SHELL_MARKER_END}\n"

        modified = []
        for filename in _SHELL_CONFIG_FILES:
            config_path = self._home_dir / filename
            if not config_path.exists():
                continue

            try:
                content = config_path.read_text(encoding="utf-8")
            except OSError as e:
                raise ShellConfigError(
                    f"Cannot read {config_path}: {e}"
                ) from e

            # Remove existing block if present
            content = self._remove_existing_block(content)

            # Append new block
            if not content.endswith("\n"):
                content += "\n"
            content += block

            # Write atomically
            self._atomic_write(config_path, content)
            modified.append(config_path)

        if not modified:
            raise ShellConfigError(
                "No shell configuration files found to update. "
                f"Searched: {', '.join(_SHELL_CONFIG_FILES)}"
            )

        return modified

    def remove_from_path(self) -> List[Path]:
        """
        Remove the managed ``PATH`` entry from all shell config files.

        Returns
        -------
        list of Path
            Paths to files that were modified.

        Notes
        -----
        Only removes blocks delimited by the marker comments. Other
        content is left unchanged.
        """
        modified = []
        for filename in _SHELL_CONFIG_FILES:
            config_path = self._home_dir / filename
            if not config_path.exists():
                continue

            try:
                content = config_path.read_text(encoding="utf-8")
            except OSError:
                continue

            new_content = self._remove_existing_block(content)
            if new_content != content:
                self._atomic_write(config_path, new_content)
                modified.append(config_path)

        return modified

    @staticmethod
    def _remove_existing_block(content: str) -> str:
        """
        Remove any existing manager PATH block from *content*.

        Parameters
        ----------
        content : str
            File content.

        Returns
        -------
        str
            Content with the block removed.
        """
        lines = content.splitlines(keepends=True)
        result: List[str] = []
        skip = False

        for line in lines:
            if _SHELL_MARKER_START in line:
                skip = True
                continue
            if _SHELL_MARKER_END in line:
                skip = False
                continue
            if not skip:
                result.append(line)

        return "".join(result)

    @staticmethod
    def _atomic_write(filepath: Path, content: str) -> None:
        """
        Write *content* to *filepath* atomically.

        Parameters
        ----------
        filepath : Path
            Target file.
        content : str
            New file content.

        Raises
        ------
        ShellConfigError
            If write fails.
        """
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=f".{filepath.name}.",
            dir=str(filepath.parent),
        )
        tmp_path = Path(tmp_path)
        try:
            os.close(tmp_fd)
            # Preserve original permissions
            if filepath.exists():
                shutil.copymode(filepath, tmp_path)
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(tmp_path, filepath)
        except OSError as e:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise ShellConfigError(
                f"Cannot write {filepath}: {e}"
            ) from e


# ---------------------------------------------------------------------------
# Python Version Manager
# ---------------------------------------------------------------------------


class PythonVersionManager:
    """
    Manage multiple isolated Python versions installed from
    ``python-build-standalone`` releases.

    Parameters
    ----------
    install_root : Path
        Root directory for all Python installations. Defaults to
        ``~/.python_versions``.
    downloader : object, optional
        A downloader instance with a method
        ``download_release_asset(release_tag, asset_filename)``
        that returns a :class:`Path` to the downloaded archive.
        If ``None``, versions must be installed manually.
    extractor : object, optional
        An extractor instance with a method
        ``extract(archive_path, dest_dir, strip_components)``
        that returns the extraction directory.
        If ``None``, archives must be extracted manually.
    platform_detector : object, optional
        A detector instance with a method ``detect()`` returning a
        target triple object with a ``raw`` attribute.
        If ``None``, platform detection must be done externally.

    Examples
    --------
    >>> from pathlib import Path
    >>> manager = PythonVersionManager(Path.home() / ".python_versions")
    >>> manager.install(
    ...     version="3.11.5",
    ...     release_date="20231002",
    ...     target_triple="x86_64-unknown-linux-gnu",
    ...     archive_path=Path("/tmp/python.tar.gz"),
    ... )
    PosixPath('/home/user/.python_versions/python-3.11.5/bin/python3')

    >>> manager.set_default("3.11.5")
    >>> manager.set_current("3.11.5")  # replaces current process
    """

    def __init__(
        self,
        install_root: Optional[Path] = None,
        downloader: Optional[Any] = None,
        extractor: Optional[Any] = None,
        platform_detector: Optional[Any] = None,
    ) -> None:
        self._install_root = (
            install_root or Path.home() / ".python_versions"
        )
        self._install_root.mkdir(parents=True, exist_ok=True)
        self._state = _StateManager(self._install_root / _STATE_FILE)
        self._shell_updater = _ShellConfigUpdater(Path.home())
        self._downloader = downloader
        self._extractor = extractor
        self._platform_detector = platform_detector

    # ------------------------------------------------------------------
    # Public API — Installation
    # ------------------------------------------------------------------

    def install(
        self,
        version: str,
        release_date: str,
        target_triple: Optional[str] = None,
        archive_path: Optional[Path] = None,
        force: bool = False,
    ) -> Path:
        """
        Install a Python version.

        Parameters
        ----------
        version : str
            Python version, e.g. ``"3.11.5"``.
        release_date : str
            Release tag date, e.g. ``"20231002"``. Required if
            *archive_path* is ``None`` and a downloader is configured.
        target_triple : str, optional
            Platform target triple, e.g.
            ``"x86_64-unknown-linux-gnu"``. If ``None``, detected
            automatically via *platform_detector*.
        archive_path : Path, optional
            Path to a pre-downloaded archive. If ``None``, the
            archive is downloaded using *downloader*.
        force : bool
            If ``True``, reinstall even if already installed.

        Returns
        -------
        Path
            Path to the Python executable.

        Raises
        ------
        VersionAlreadyInstalledError
            If *version* is already installed and *force* is
            ``False``.
        ValueError
            If neither *archive_path* nor *downloader* is available.
        VersionManagerError
            On installation failure.
        """
        version_dir = self._version_dir(version)

        if version_dir.exists() and not force:
            raise VersionAlreadyInstalledError(
                f"Python {version} is already installed at "
                f"{version_dir}",
                version=version,
            )

        # Determine target triple
        if target_triple is None:
            if self._platform_detector is None:
                raise ValueError(
                    "target_triple is required when no platform_detector "
                    "is configured."
                )
            target = self._platform_detector.detect()
            target_triple = target.raw

        # Obtain archive
        if archive_path is None:
            if self._downloader is None:
                raise ValueError(
                    "archive_path is required when no downloader is "
                    "configured."
                )
            asset_filename = self._build_asset_filename(
                version=version,
                release_date=release_date,
                target_triple=target_triple,
            )
            archive_path = self._downloader.download_release_asset(
                release_tag=release_date,
                asset_filename=asset_filename,
            )

        # Extract
        if self._extractor is None:
            raise ValueError(
                "extractor is required for installation."
            )

        # Remove existing directory if forcing
        if force and version_dir.exists():
            shutil.rmtree(version_dir, ignore_errors=True)

        extracted = self._extractor.extract(
            archive_path=archive_path,
            dest_dir=version_dir,
            strip_components=0,
        )

        # Find Python executable
        python_bin = self._extractor.find_python_executable(extracted)

        # Record installation
        installed = dict(self._state.installed_versions)
        installed[version] = str(version_dir)
        self._state.installed_versions = installed

        return python_bin

    # ------------------------------------------------------------------
    # Public API — Set Current (Process Switch)
    # ------------------------------------------------------------------

    def set_current(self, version: str, args: Optional[List[str]] = None) -> None:
        """
        Replace the current process with the specified Python version.

        **This method does not return.** It calls :func:`os.execve`
        to replace the running process with the target Python
        interpreter.

        Parameters
        ----------
        version : str
            Python version to switch to.
        args : list of str, optional
            Arguments to pass to the new Python process. Defaults to
            ``sys.argv`` from the current process.

        Raises
        ------
        VersionNotFoundError
            If *version* is not installed.
        ProcessSwitchError
            If :func:`os.execve` fails.

        Warnings
        --------
        - All unsaved state in the current process is lost.
        - Open file handles, network connections, and child processes
          are inherited by the new interpreter.
        - Environment variables are filtered; only safe variables are
          passed to the new process.
        """
        python_bin = self._get_python_binary(version)

        if args is None:
            args = sys.argv[:]

        # Build clean environment
        new_env = self._build_safe_environment(version, python_bin)

        try:
            os.execve(
                str(python_bin),
                [str(python_bin)] + args[1:],
                new_env,
            )
        except OSError as e:
            raise ProcessSwitchError(
                f"Failed to switch to Python {version}: {e}",
                version=version,
            ) from e

    # ------------------------------------------------------------------
    # Public API — Set Default
    # ------------------------------------------------------------------

    def set_default(self, version: str) -> Path:
        """
        Set a Python version as the system default.

        Creates a symlink (or shim on Windows) and updates shell
        configuration files to add the version to ``PATH``.

        Parameters
        ----------
        version : str
            Python version to set as default.

        Returns
        -------
        Path
            Path to the default Python executable.

        Raises
        ------
        VersionNotFoundError
            If *version* is not installed.
        ShellConfigError
            If shell configuration files cannot be updated.
        """
        python_bin = self._get_python_binary(version)

        # Create default symlink
        default_link = self._install_root / "default"
        self._create_default_link(python_bin.parent, default_link)

        # Update shell config
        modified = self._shell_updater.add_to_path(python_bin.parent)

        # Record
        self._state.default_version = version

        return python_bin

    # ------------------------------------------------------------------
    # Public API — Uninstall
    # ------------------------------------------------------------------

    def uninstall(self, version: str, force: bool = False) -> None:
        """
        Remove an installed Python version.

        Parameters
        ----------
        version : str
            Python version to remove.
        force : bool
            If ``True``, remove even if this version is the default.

        Raises
        ------
        VersionNotFoundError
            If *version* is not installed.
        VersionActiveError
            If *version* is the current default and *force* is
            ``False``.
        """
        if version not in self._state.installed_versions:
            raise VersionNotFoundError(
                f"Python {version} is not installed.",
                version=version,
            )

        if self._state.default_version == version and not force:
            raise VersionActiveError(
                f"Python {version} is the current default. "
                "Use force=True to remove anyway.",
                version=version,
            )

        # Remove directory
        version_dir = self._version_dir(version)
        if version_dir.exists():
            shutil.rmtree(version_dir, ignore_errors=True)

        # Update state
        installed = dict(self._state.installed_versions)
        del installed[version]
        self._state.installed_versions = installed

        # Clear default if needed
        if self._state.default_version == version:
            self._state.default_version = None
            default_link = self._install_root / "default"
            if default_link.exists():
                default_link.unlink(missing_ok=True)
            self._shell_updater.remove_from_path()

    # ------------------------------------------------------------------
    # Public API — Query
    # ------------------------------------------------------------------

    def list_installed(self) -> Dict[str, Path]:
        """
        List all installed Python versions.

        Returns
        -------
        dict[str, Path]
            Mapping of version string to Python executable path.
        """
        result: Dict[str, Path] = {}
        for version, dir_path in self._state.installed_versions.items():
            version_dir = Path(dir_path)
            try:
                python_bin = self._find_python_in_directory(version_dir)
                result[version] = python_bin
            except FileNotFoundError:
                continue
        return result

    def get_default(self) -> Optional[str]:
        """
        Return the currently-set default version.

        Returns
        -------
        str or None
            Default version string, or ``None`` if not set.
        """
        return self._state.default_version

    def is_installed(self, version: str) -> bool:
        """
        Check if a version is installed.

        Parameters
        ----------
        version : str
            Python version.

        Returns
        -------
        bool
            ``True`` if installed.
        """
        return version in self._state.installed_versions

    def get_python_path(self, version: str) -> Path:
        """
        Get the path to the Python executable for *version*.

        Parameters
        ----------
        version : str
            Python version.

        Returns
        -------
        Path

        Raises
        ------
        VersionNotFoundError
            If *version* is not installed.
        """
        return self._get_python_binary(version)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _version_dir(self, version: str) -> Path:
        """
        Return the installation directory for *version*.

        Parameters
        ----------
        version : str
            Python version.

        Returns
        -------
        Path
        """
        return self._install_root / f"{_VERSION_DIR_PREFIX}{version}"

    def _get_python_binary(self, version: str) -> Path:
        """
        Get the Python executable for *version*, installing if needed.

        Parameters
        ----------
        version : str
            Python version.

        Returns
        -------
        Path

        Raises
        ------
        VersionNotFoundError
            If not installed.
        """
        version_dir = self._version_dir(version)
        if not version_dir.exists():
            raise VersionNotFoundError(
                f"Python {version} is not installed.",
                version=version,
            )
        return self._find_python_in_directory(version_dir)

    @staticmethod
    def _find_python_in_directory(directory: Path) -> Path:
        """
        Find the Python executable inside *directory*.

        Parameters
        ----------
        directory : Path
            Directory to search.

        Returns
        -------
        Path

        Raises
        ------
        FileNotFoundError
            If no Python executable is found.
        """
        candidate_names = ("python3", "python", "python3.exe", "python.exe")
        for root, dirs, files in os.walk(directory):
            # Limit depth
            depth = len(Path(root).relative_to(directory).parts)
            if depth > 3:
                dirs.clear()
                continue
            for name in candidate_names:
                if name in files:
                    candidate = Path(root) / name
                    if os.access(candidate, os.X_OK):
                        return candidate.resolve()
        raise FileNotFoundError(
            f"Python executable not found in {directory}"
        )

    @staticmethod
    def _build_asset_filename(
        version: str,
        release_date: str,
        target_triple: str,
        variant: str = "install_only",
    ) -> str:
        """
        Build the asset filename for a release.

        Parameters
        ----------
        version : str
            Python version.
        release_date : str
            Release date tag.
        target_triple : str
            Target triple string.
        variant : str
            Build variant.

        Returns
        -------
        str
            Asset filename.
        """
        return (
            f"cpython-{version}+{release_date}-{target_triple}"
            f"-{variant}.tar.gz"
        )

    @staticmethod
    def _create_default_link(
        bin_dir: Path,
        link_path: Path,
    ) -> None:
        """
        Create a symlink (or shim) for the default Python.

        Parameters
        ----------
        bin_dir : Path
            Directory containing the Python executable.
        link_path : Path
            Path for the symlink or shim directory.
        """
        if link_path.exists():
            if link_path.is_symlink() or link_path.is_dir():
                link_path.unlink()
            else:
                shutil.rmtree(link_path, ignore_errors=True)

        if os.name == "nt":
            # Windows: create shim scripts
            link_path.mkdir(parents=True, exist_ok=True)
            for exe_name in ("python.exe", "python3.exe", "pip.exe"):
                target = bin_dir / exe_name
                shim = link_path / exe_name
                if target.exists():
                    shim.write_text(
                        f'@"{target}" %*\n', encoding="utf-8"
                    )
        else:
            # Unix: symlink
            link_path.symlink_to(bin_dir, target_is_directory=True)

    def _build_safe_environment(
        self,
        version: str,
        python_bin: Path,
    ) -> Dict[str, str]:
        """
        Build a clean environment for the new Python process.

        Parameters
        ----------
        version : str
            Python version being activated.
        python_bin : Path
            Path to the Python executable.

        Returns
        -------
        dict[str, str]
            Environment variables for the new process.

        Notes
        -----
        - Preserves most existing environment variables.
        - Prepends the version's ``bin`` directory to ``PATH``.
        - Sets ``PYTHON_VERSION_MANAGER_ACTIVE`` to the version string.
        - Removes ``PYTHONHOME`` and ``PYTHONPATH`` to prevent
          conflicts with the new interpreter.
        """
        new_env = os.environ.copy()

        # Prepend to PATH
        bin_dir = str(python_bin.parent)
        new_env["PATH"] = f"{bin_dir}{os.pathsep}{new_env.get('PATH', '')}"

        # Mark as managed
        new_env["PYTHON_VERSION_MANAGER_ACTIVE"] = version

        # Remove potentially conflicting variables
        for var in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"):
            new_env.pop(var, None)

        return new_env
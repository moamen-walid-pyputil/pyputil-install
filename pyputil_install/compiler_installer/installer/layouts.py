"""
Toolchain installation layout management.

Defines the on-disk structure for installed compiler toolchains
and provides utilities for path construction, listing, cleanup,
and default version selection via symlinks.

Design
------
All toolchains live under a single configurable root directory.
Each compiler family gets its own subdirectory, and within that,
each installed version is a self-contained directory.

    {root}/
        gcc/
            14.2.0-2/
                bin/
                lib/
                include/
                ...
            13.3.0/
                ...
            default -> 14.2.0-2/    (symlink, optional)
        clang/
            18.1.0/
                ...
        .manifests/                  (installation metadata)
        .tmp/                        (staging area for downloads)

This two-level hierarchy (compiler/version) avoids naming collisions
and allows per-compiler default version symlinks.

Environment
-----------
TOOLFORGE_HOME
    Overrides the installation root. Example: /opt/toolforge.
    If set, all toolchains are installed under this directory.
    Default: platform-specific (see get_install_root).

TOOLFORGE_TEMP_DIR
    Overrides the temporary staging directory used during installation.
    If not set, uses {root}/.tmp.

Usage
-----
    from pyputil_install.compiler_installer.layouts import (
        get_install_root,
        get_toolchain_path,
        list_installed,
        set_default,
        remove_toolchain_layout,
    )

    root = get_install_root()
    gcc_path = get_toolchain_path("gcc", "14.2.0-2")
    installed = list_installed()

Warnings
--------
- On Windows, paths may exceed MAX_PATH (260 characters) if the
  installation root is deeply nested. Use TOOLFORGE_HOME to set
  a shorter path if needed.
- Symlinks are used for default version selection on Unix. Windows
  requires administrator privileges or developer mode for symlinks.
  On Windows without symlink support, a `.default_version` file
  is written instead.
- The .tmp directory is NOT cleaned automatically. Call
  cleanup_temp() after successful installations.
- Do NOT manually delete directories under the root. Use the
  uninstall module to properly clean up symlinks and manifests.

User Instructions
-----------------
- Set TOOLFORGE_HOME before first use to control installation location.
- Use list_installed() to see all available toolchains.
- Use set_default() to create a version-agnostic symlink (e.g., "gcc"
  points to the latest installed version).
"""

import os
import shutil
import stat
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


# ============================================================================
# Root directory resolution
# ============================================================================

def get_install_root() -> Path:
    """
    Return the absolute path to the toolchain installation root.

    Resolution order:
        1. `TOOLFORGE_HOME` environment variable (if set and non-empty).
        2. Platform default:
           - Unix (Linux, macOS):
             `$XDG_DATA_HOME/toolforge/toolchains` or
             `~/.local/share/toolforge/toolchains` if XDG_DATA_HOME
             is not set.
           - Windows:
             `%LOCALAPPDATA%/toolforge/toolchains` or
             `~/AppData/Local/toolforge/toolchains` if LOCALAPPDATA
             is not set.

    Returns
    -------
    Path
        Absolute, expanded path to the root directory.
        The directory may not exist yet.

    Examples
    --------
    >>> get_install_root()  # Linux, XDG_DATA_HOME unset
    PosixPath('/home/user/.local/share/toolforge/toolchains')

    >>> # With TOOLFORGE_HOME set
    >>> os.environ['TOOLFORGE_HOME'] = '/opt/my-tools'
    >>> get_install_root()
    PosixPath('/opt/my-tools')
    """
    env_root = os.environ.get("TOOLFORGE_HOME")
    if env_root:
        return Path(env_root).expanduser().resolve()

    if os.name == "nt":  # Windows
        local_app_data = os.environ.get(
            "LOCALAPPDATA",
            str(Path.home() / "AppData" / "Local"),
        )
        return Path(local_app_data) / "toolforge" / "toolchains"

    # Unix (Linux, macOS, etc.)
    xdg_data = os.environ.get("XDG_DATA_HOME", "")
    if xdg_data:
        return Path(xdg_data) / "toolforge" / "toolchains"

    return Path.home() / ".local" / "share" / "toolforge" / "toolchains"


# ============================================================================
# Temporary directory
# ============================================================================

def get_temp_dir(install_root: Optional[Path] = None) -> Path:
    """
    Return the path to the temporary staging directory for downloads.

    Parameters
    ----------
    install_root : Optional[Path]
        Root directory. If None, `get_install_root()` is called.

    Returns
    -------
    Path
        Path to the .tmp directory.

    Environment
    -----------
    TOOLFORGE_TEMP_DIR
        If set, overrides the default .tmp location entirely.
        This directory is used as-is, not appended to the root.
    """
    env_tmp = os.environ.get("TOOLFORGE_TEMP_DIR")
    if env_tmp:
        return Path(env_tmp).expanduser().resolve()

    root = install_root or get_install_root()
    return root / ".tmp"


# ============================================================================
# Path construction — per-compiler, per-version
# ============================================================================

def get_toolchain_path(
    compiler: str,
    version: str,
    install_root: Optional[Path] = None,
) -> Path:
    """
    Build the path where a specific compiler+version is installed.

    Structure: `{root}/{compiler}/{version}/`

    Parameters
    ----------
    compiler : str
        Compiler name, e.g., "gcc", "clang", "zig".
        Must not contain path separators.
    version : str
        Version string, e.g., "14.2.0-2", "18.1.0".
    install_root : Optional[Path]
        Root directory. If None, `get_install_root()` is called.

    Returns
    -------
    Path
        Absolute path to the toolchain version directory.
        Does NOT check if the directory exists.

    Raises
    ------
    ValueError
        If `compiler` or `version` contains '/' or '\\'.

    Examples
    --------
    >>> get_toolchain_path("gcc", "14.2.0-2")
    PosixPath('/home/user/.local/share/toolforge/toolchains/gcc/14.2.0-2')
    """
    if "/" in compiler or "\\" in compiler:
        raise ValueError(f"Compiler name must not contain path separators: {compiler!r}")
    if "/" in version or "\\" in version:
        raise ValueError(f"Version string must not contain path separators: {version!r}")

    root = install_root or get_install_root()
    return root / compiler / version


def get_bin_dir(
    compiler: str,
    version: str,
    install_root: Optional[Path] = None,
) -> Path:
    """
    Return the `bin` subdirectory inside a toolchain installation.

    Parameters
    ----------
    compiler : str
        Compiler name.
    version : str
        Version string.
    install_root : Optional[Path]
        Root directory. If None, `get_install_root()` is called.

    Returns
    -------
    Path
        Path to the `bin` directory.
    """
    return get_toolchain_path(compiler, version, install_root) / "bin"


def get_default_symlink_path(
    compiler: str,
    install_root: Optional[Path] = None,
) -> Path:
    """
    Return the path to the per-compiler default version symlink.

    Structure: `{root}/{compiler}/default`

    This symlink points to the currently active version directory
    and is managed by `set_default()`.

    Parameters
    ----------
    compiler : str
        Compiler name.
    install_root : Optional[Path]
        Root directory. If None, `get_install_root()` is called.

    Returns
    -------
    Path
        Path to the symlink (or marker file on Windows).
    """
    root = install_root or get_install_root()
    return root / compiler / "default"


def get_manifests_dir(install_root: Optional[Path] = None) -> Path:
    """
    Return the path to the manifests directory.

    Manifests store per-installation metadata (symlinks created,
    checksums, installation timestamps, etc.).

    Parameters
    ----------
    install_root : Optional[Path]
        Root directory. If None, `get_install_root()` is called.

    Returns
    -------
    Path
        Path to the `.manifests` directory.
    """
    root = install_root or get_install_root()
    return root / ".manifests"


# ============================================================================
# Directory creation
# ============================================================================

def create_root(install_root: Optional[Path] = None) -> Path:
    """
    Create the installation root and required subdirectories.

    Creates:
        {root}/
        {root}/.manifests/
        {root}/.tmp/

    Safe to call multiple times; existing directories are not modified.

    Parameters
    ----------
    install_root : Optional[Path]
        Root directory to create. If None, `get_install_root()` is used.

    Returns
    -------
    Path
        The created root directory path.

    Raises
    ------
    OSError
        If directory creation fails (e.g., permission denied on a
        parent directory).
    """
    root = install_root or get_install_root()
    root.mkdir(parents=True, exist_ok=True)

    # Create essential subdirectories
    get_manifests_dir(root).mkdir(parents=True, exist_ok=True)
    get_temp_dir(root).mkdir(parents=True, exist_ok=True)

    return root


# ============================================================================
# ToolchainLayout — object representing one installed version
# ============================================================================

class ToolchainLayout:
    """
    Represents the on-disk layout of a single installed toolchain version.

    Parameters
    ----------
    compiler : str
        Compiler name.
    version : str
        Version string.
    install_root : Optional[Path]
        Root directory. If None, `get_install_root()` is called.

    Attributes
    ----------
    path : Path
        Absolute path to the version directory.
    bin_dir : Path
        Path to the `bin` subdirectory.
    compiler : str
        Compiler name.
    version : str
        Version string.
    """

    def __init__(
        self,
        compiler: str,
        version: str,
        install_root: Optional[Path] = None,
    ) -> None:
        self._compiler = compiler
        self._version = version
        self._root = install_root or get_install_root()

    @property
    def compiler(self) -> str:
        """Return the compiler name."""
        return self._compiler

    @property
    def version(self) -> str:
        """Return the version string."""
        return self._version

    @property
    def path(self) -> Path:
        """Return the absolute path to the version directory."""
        return get_toolchain_path(self._compiler, self._version, self._root)

    @property
    def bin_dir(self) -> Path:
        """Return the path to the `bin` subdirectory."""
        return self.path / "bin"

    @property
    def lib_dir(self) -> Path:
        """Return the path to the `lib` subdirectory, if standard layout."""
        return self.path / "lib"

    @property
    def include_dir(self) -> Path:
        """Return the path to the `include` subdirectory."""
        return self.path / "include"

    def exists(self) -> bool:
        """
        Check whether the toolchain directory exists on disk.

        Returns
        -------
        bool
            True if the version directory exists.
        """
        return self.path.exists() and self.path.is_dir()

    def find_executable(self, name: str) -> Optional[Path]:
        """
        Find a named executable inside the toolchain's bin directory.

        Parameters
        ----------
        name : str
            Executable name, e.g., "gcc", "clang".
            On Windows, ".exe" is appended automatically if not present.

        Returns
        -------
        Optional[Path]
            Path to the executable if found, None otherwise.
        """
        if os.name == "nt" and not name.endswith(".exe"):
            name = f"{name}.exe"

        candidate = self.bin_dir / name
        if candidate.is_file():
            return candidate
        return None

    def __repr__(self) -> str:
        return f"ToolchainLayout({self._compiler!r}, {self._version!r})"


# ============================================================================
# ToolchainSet — all versions of a compiler family
# ============================================================================

class ToolchainSet:
    """
    Represents all installed versions of a single compiler family.

    Parameters
    ----------
    compiler : str
        Compiler name.
    install_root : Optional[Path]
        Root directory. If None, `get_install_root()` is called.
    """

    def __init__(
        self,
        compiler: str,
        install_root: Optional[Path] = None,
    ) -> None:
        self._compiler = compiler
        self._root = install_root or get_install_root()

    @property
    def compiler(self) -> str:
        """Return the compiler name."""
        return self._compiler

    @property
    def path(self) -> Path:
        """Return the path to the compiler family directory."""
        return self._root / self._compiler

    def list_versions(self) -> List[str]:
        """
        List all installed versions for this compiler.

        Returns
        -------
        List[str]
            Sorted list of version strings.
        """
        if not self.path.exists() or not self.path.is_dir():
            return []

        versions = []
        for entry in self.path.iterdir():
            if entry.is_dir() and entry.name not in (".manifests", ".tmp", "default"):
                versions.append(entry.name)

        # Simple version-aware sort (lexicographic, fine for numeric versions)
        versions.sort(reverse=True)
        return versions

    def latest_version(self) -> Optional[str]:
        """
        Return the latest installed version string.

        Returns
        -------
        Optional[str]
            The latest version string, or None if no versions installed.
        """
        versions = self.list_versions()
        return versions[0] if versions else None

    def get_layout(self, version: str) -> ToolchainLayout:
        """
        Return a ToolchainLayout for a specific version.

        Parameters
        ----------
        version : str
            Version string.

        Returns
        -------
        ToolchainLayout
            Layout object for that version.
        """
        return ToolchainLayout(self._compiler, version, self._root)

    def __repr__(self) -> str:
        return f"ToolchainSet({self._compiler!r})"


# ============================================================================
# Listing installed toolchains
# ============================================================================

def list_installed(
    install_root: Optional[Path] = None,
) -> List[Tuple[str, str]]:
    """
    List all installed toolchains under the root.

    Scans for directories matching `{root}/{compiler}/{version}`.

    Parameters
    ----------
    install_root : Optional[Path]
        Root directory. If None, `get_install_root()` is called.

    Returns
    -------
    List[Tuple[str, str]]
        List of (compiler, version) tuples. Sorted by compiler, then
        version descending. May be empty if root does not exist.

    Examples
    --------
    >>> list_installed()
    [('gcc', '14.2.0-2'), ('gcc', '13.3.0'), ('zig', '0.11.0')]
    """
    root = install_root or get_install_root()
    if not root.exists():
        return []

    results: List[Tuple[str, str]] = []
    for compiler_dir in root.iterdir():
        if not compiler_dir.is_dir():
            continue
        compiler = compiler_dir.name
        if compiler.startswith("."):
            continue

        for version_dir in compiler_dir.iterdir():
            if not version_dir.is_dir():
                continue
            version = version_dir.name
            if version in ("default", ".tmp"):
                continue
            results.append((compiler, version))

    results.sort(key=lambda x: (x[0], _version_sort_key(x[1])))
    return results


def list_compiler_families(install_root: Optional[Path] = None) -> List[str]:
    """
    List all compiler families that have at least one installed version.

    Parameters
    ----------
    install_root : Optional[Path]
        Root directory. If None, `get_install_root()` is called.

    Returns
    -------
    List[str]
        Sorted list of compiler names.
    """
    root = install_root or get_install_root()
    if not root.exists():
        return []

    families = []
    for compiler_dir in root.iterdir():
        if compiler_dir.is_dir() and not compiler_dir.name.startswith("."):
            families.append(compiler_dir.name)

    families.sort()
    return families


# ============================================================================
# Default version management (symlinks)
# ============================================================================

def set_default(
    compiler: str,
    version: str,
    install_root: Optional[Path] = None,
) -> bool:
    """
    Mark a specific version as the default for its compiler family.

    On Unix, creates a symlink: `{root}/{compiler}/default -> {version}/`
    On Windows without symlink capability, writes a `.default_version`
    marker file containing the version string.

    Parameters
    ----------
    compiler : str
        Compiler name.
    version : str
        Version string to set as default.
    install_root : Optional[Path]
        Root directory. If None, `get_install_root()` is called.

    Returns
    -------
    bool
        True if the default was set successfully, False if the
        target version directory does not exist.

    Warnings
    --------
    - This function does NOT validate that the version is installed.
      Check with `ToolchainLayout(compiler, version).exists()` first.
    - On Windows without developer mode, symlink creation requires
      administrator privileges. The fallback marker file avoids this.
    """
    layout = ToolchainLayout(compiler, version, install_root)
    if not layout.exists():
        return False

    symlink_path = get_default_symlink_path(compiler, install_root)

    # Remove existing default
    if symlink_path.is_symlink():
        symlink_path.unlink()
    elif symlink_path.exists():
        # It's a regular file or directory — remove it
        if symlink_path.is_dir():
            shutil.rmtree(symlink_path)
        else:
            symlink_path.unlink()

    # Create symlink (Unix) or marker file (Windows fallback)
    try:
        symlink_path.symlink_to(layout.path.name, target_is_directory=True)
    except OSError:
        # Windows without symlink support — write marker file
        marker_path = symlink_path.parent / ".default_version"
        marker_path.write_text(f"{compiler}:{version}\n")

    return True


def get_default_version(
    compiler: str,
    install_root: Optional[Path] = None,
) -> Optional[str]:
    """
    Get the currently active default version for a compiler.

    Checks the `default` symlink first, then falls back to the
    `.default_version` marker file.

    Parameters
    ----------
    compiler : str
        Compiler name.
    install_root : Optional[Path]
        Root directory. If None, `get_install_root()` is called.

    Returns
    -------
    Optional[str]
        The default version string, or None if no default is set.
    """
    symlink_path = get_default_symlink_path(compiler, install_root)

    if symlink_path.is_symlink():
        target = os.readlink(str(symlink_path))
        return Path(target).name

    # Check marker file (Windows fallback)
    marker_path = symlink_path.parent / ".default_version"
    if marker_path.exists():
        content = marker_path.read_text().strip()
        if ":" in content:
            _, version = content.split(":", 1)
            return version.strip()

    return None


# ============================================================================
# Cleanup
# ============================================================================

def cleanup_temp(install_root: Optional[Path] = None) -> int:
    """
    Remove all files in the temporary staging directory.

    Parameters
    ----------
    install_root : Optional[Path]
        Root directory. If None, `get_install_root()` is called.

    Returns
    -------
    int
        Number of files/directories removed.
    """
    temp_dir = get_temp_dir(install_root)
    if not temp_dir.exists():
        return 0

    count = 0
    for entry in temp_dir.iterdir():
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            count += 1
        except OSError:
            pass

    return count


def remove_toolchain_layout(
    compiler: str,
    version: str,
    install_root: Optional[Path] = None,
) -> bool:
    """
    Remove a specific toolchain version directory.

    Does NOT clean up symlinks or manifests. Use uninstall.py
    for a full uninstall cycle. This function only removes the
    core toolchain files.

    If the compiler family directory becomes empty after removal,
    it is also deleted.

    Parameters
    ----------
    compiler : str
        Compiler name.
    version : str
        Version string.
    install_root : Optional[Path]
        Root directory. If None, `get_install_root()` is called.

    Returns
    -------
    bool
        True if the directory was found and removed, False if it
        did not exist.
    """
    layout = ToolchainLayout(compiler, version, install_root)
    if not layout.exists():
        return False

    # Remove the version directory with retry for permission issues
    _rmtree_robust(layout.path)

    # Remove compiler family directory if empty
    compiler_dir = layout.path.parent
    if compiler_dir.exists() and not any(compiler_dir.iterdir()):
        try:
            compiler_dir.rmdir()
        except OSError:
            pass

    return True


def _rmtree_robust(path: Path, max_attempts: int = 3) -> None:
    """
    Robustly remove a directory tree with retries.

    On some platforms (notably Windows), files may be locked by
    antivirus or indexing services. Retries handle transient locks.

    Parameters
    ----------
    path : Path
        Directory to remove.
    max_attempts : int
        Number of removal attempts before raising.

    Raises
    ------
    OSError
        If removal fails after all attempts.
    """
    import time
    for attempt in range(max_attempts):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            if attempt < max_attempts - 1:
                time.sleep(0.5 * (attempt + 1))
            else:
                raise


# ============================================================================
# Internal helpers
# ============================================================================

def _version_sort_key(version: str) -> tuple:
    """
    Generate a sort key for a version string.

    Tries to split by '.' and convert to integers for proper numeric
    sorting. Falls back to lexicographic for non-numeric segments.

    Parameters
    ----------
    version : str
        Version string.

    Returns
    -------
    tuple
        Sort key usable with sorted().
    """
    parts = []
    for segment in version.split("."):
        try:
            parts.append((0, int(segment)))
        except ValueError:
            parts.append((1, segment))
    return tuple(parts)
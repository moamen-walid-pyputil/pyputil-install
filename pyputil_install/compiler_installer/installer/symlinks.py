"""
Symlink manager for compiler toolchain executables.

Creates and manages symbolic links from installed toolchain `bin`
directories to a centralized `bin` directory that the user can
add to their PATH once, rather than per-toolchain.

Design
------
    {root}/
        gcc/
            14.2.0-2/
                bin/
                    gcc
                    g++
                    ...
            13.3.0/
                bin/
                    ...
        clang/
            18.1.0/
                bin/
                    ...

    {link_root}/                           # e.g., ~/.local/bin or {root}/bin
        gcc -> {root}/gcc/default/bin/gcc
        g++ -> {root}/gcc/default/bin/g++
        clang -> {root}/clang/default/bin/clang
        gcc@14 -> {root}/gcc/14.2.0-2/bin/gcc
        gcc@13 -> {root}/gcc/13.3.0/bin/gcc

Two types of links are created:
    1. Short name (e.g., `gcc`) — points to the default version.
    2. Versioned name (e.g., `gcc@14`) — points to a specific version,
       using the major version number as the alias.

On Windows, where symlinks require administrator privileges or
developer mode, `.bat` wrapper scripts are created instead.

Environment
-----------
TOOLFORGE_BIN_DIR
    Overrides the centralized bin directory path. Default:
    `{root}/bin` where root is the installation root.

TOOLFORGE_LINK_STYLE
    Controls which links are created:
    - "both" (default): create short + versioned links
    - "short": only `gcc`
    - "versioned": only `gcc@14`

TOOLFORGE_LINK_FORCE
    If set to "1", existing links are overwritten.
    Default: skip if a link/binary already exists at the target.

Usage
-----
    from pyputil_install.compiler_installer.symlinks import SymlinkManager

    manager = SymlinkManager()
    manager.create_links("gcc", "14.2.0-2")
    manager.set_default("gcc", "14.2.0-2")
    manager.remove_links("gcc", "13.3.0")

Warnings
--------
- On Windows without Developer Mode enabled, symlink creation fails
  with OSError. This module falls back to `.bat` wrappers.
- Short links (`gcc`) point to the default version. If no default
  is set, short links are NOT created.
- Removing a version that is the current default leaves dangling
  short links. Call `set_default()` to reassign before removal.
- This module never modifies PATH. The user must add the centralized
  bin directory to their PATH manually, or use `activation.py` for
  temporary shell activation.

User Instructions
-----------------
- Add TOOLFORGE_BIN_DIR to your shell PATH permanently:
    export PATH="$HOME/.local/share/toolforge/toolchains/bin:$PATH"
- Use versioned links (`gcc@14`) to pin a specific version in scripts.
- Use short links (`gcc`) for interactive use that follows defaults.
- Call `manager.rescan()` after manual changes to detect stale links.
"""

import logging
import os
import re
import shutil
import stat
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .layouts import (
    get_install_root,
    get_default_symlink_path,
    get_default_version,
    ToolchainLayout,
    ToolchainSet,
    _rmtree_robust,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

def _get_link_root() -> Path:
    """
    Return the centralized bin directory for symlinks.

    Resolution order:
        1. `TOOLFORGE_BIN_DIR` environment variable.
        2. `{install_root}/bin` (default).

    Returns
    -------
    Path
        Absolute path to the bin directory for links.
    """
    env_bin = os.environ.get("TOOLFORGE_BIN_DIR")
    if env_bin:
        return Path(env_bin).expanduser().resolve()
    return get_install_root() / "bin"


def _get_link_style() -> str:
    """
    Return the link creation style.

    Returns
    -------
    str
        "both", "short", or "versioned".
    """
    return os.environ.get("TOOLFORGE_LINK_STYLE", "both").lower()


def _get_link_force() -> bool:
    """
    Return whether to force-overwrite existing links.

    Returns
    -------
    bool
        True if TOOLFORGE_LINK_FORCE=1.
    """
    return os.environ.get("TOOLFORGE_LINK_FORCE", "") == "1"


def _supports_symlinks() -> bool:
    """
    Check if the platform supports symlink creation without elevation.

    Returns
    -------
    bool
        True if symlinks can be created by the current user.
    """
    if os.name == "nt":
        # On Windows, symlinks require Developer Mode or admin.
        # Test by trying to create and remove a temp symlink.
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            test_target = tmp_dir / "target"
            test_link = tmp_dir / "link"
            test_target.touch()
            test_link.symlink_to(test_target)
            test_link.unlink()
            return True
        except OSError:
            return False
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return True  # Unix-like always supports symlinks


# ============================================================================
# Executable discovery
# ============================================================================

def _find_executables(bin_dir: Path) -> List[str]:
    """
    List all executable files in a toolchain's bin directory.

    Parameters
    ----------
    bin_dir : Path
        Path to a toolchain `bin` directory.

    Returns
    -------
    List[str]
        Filenames of executable files (without path).
    """
    if not bin_dir.exists() or not bin_dir.is_dir():
        return []

    executables = []
    for entry in bin_dir.iterdir():
        if entry.is_file() and os.access(str(entry), os.X_OK):
            executables.append(entry.name)

    return sorted(executables)


# ============================================================================
# Link naming
# ============================================================================

def _extract_major_version(version: str) -> str:
    """
    Extract the major version number from a version string.

    Parameters
    ----------
    version : str
        Version string, e.g., "14.2.0-2", "18.1.0".

    Returns
    -------
    str
        Major version, e.g., "14", "18".
    """
    match = re.match(r"(\d+)", version)
    return match.group(1) if match else version


def _short_link_name(executable_name: str) -> str:
    """
    Return the short link name for an executable.

    This is simply the executable name without version suffix.

    Parameters
    ----------
    executable_name : str
        The original executable filename.

    Returns
    -------
    str
        Short link name.
    """
    return executable_name


def _versioned_link_name(executable_name: str, version: str) -> str:
    """
    Return the versioned link name for an executable.

    Format: `{name}@{major}`, e.g., "gcc@14".

    Parameters
    ----------
    executable_name : str
        The original executable filename.
    version : str
        Version string.

    Returns
    -------
    str
        Versioned link name.
    """
    major = _extract_major_version(version)
    return f"{executable_name}@{major}"


# ============================================================================
# SymlinkManager
# ============================================================================

class SymlinkManager:
    """
    Manages symlinks from toolchain bin directories to a central bin.

    Parameters
    ----------
    install_root : Optional[Path]
        Root directory for toolchain installations.
        If None, `get_install_root()` is called.
    link_root : Optional[Path]
        Centralized bin directory for symlinks.
        If None, uses `TOOLFORGE_BIN_DIR` or `{root}/bin`.

    Attributes
    ----------
    install_root : Path
        The toolchain installation root.
    link_root : Path
        The centralized bin directory.
    """

    def __init__(
        self,
        install_root: Optional[Path] = None,
        link_root: Optional[Path] = None,
    ) -> None:
        self.install_root = install_root or get_install_root()
        self.link_root = link_root or _get_link_root()
        self._use_symlinks = _supports_symlinks()

        # Ensure the link root exists
        self.link_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_links(
        self,
        compiler: str,
        version: str,
    ) -> int:
        """
        Create symlinks (or wrappers) for all executables in a toolchain.

        Creates short links if a default version is set, and versioned
        links (`@major`) for this specific version.

        Parameters
        ----------
        compiler : str
            Compiler name.
        version : str
            Version string.

        Returns
        -------
        int
            Number of links created.

        Warnings
        --------
        - Existing links are skipped unless `TOOLFORGE_LINK_FORCE=1`.
        - If the toolchain bin directory does not exist, no links
          are created and 0 is returned.
        """
        layout = ToolchainLayout(compiler, version, self.install_root)
        if not layout.exists():
            logger.warning("Toolchain not found: %s", layout.path)
            return 0

        executables = _find_executables(layout.bin_dir)
        if not executables:
            logger.warning("No executables found in %s", layout.bin_dir)
            return 0

        style = _get_link_style()
        force = _get_link_force()
        default_version = get_default_version(compiler, self.install_root)
        count = 0

        for exe_name in executables:
            exe_path = layout.bin_dir / exe_name

            # Versioned links (always created if style allows)
            if style in ("both", "versioned"):
                versioned_name = _versioned_link_name(exe_name, version)
                if self._create_single_link(exe_path, versioned_name, force):
                    count += 1

            # Short links (only if this version is the default)
            if style in ("both", "short") and default_version == version:
                short_name = _short_link_name(exe_name)
                if self._create_single_link(exe_path, short_name, force):
                    count += 1

        logger.info(
            "Created %d links for %s@%s", count, compiler, version
        )
        return count

    def remove_links(
        self,
        compiler: str,
        version: str,
    ) -> int:
        """
        Remove all versioned links for a specific version.

        Short links (without version) are NOT removed. Use
        `remove_short_links()` to clean those up.

        Parameters
        ----------
        compiler : str
            Compiler name.
        version : str
            Version string.

        Returns
        -------
        int
            Number of links removed.
        """
        layout = ToolchainLayout(compiler, version, self.install_root)
        executables = _find_executables(layout.bin_dir) if layout.exists() else []

        if not executables:
            # Try to discover from existing links in the link root
            executables = self._find_linked_executables(compiler, version)

        count = 0
        for exe_name in executables:
            versioned_name = _versioned_link_name(exe_name, version)
            link_path = self.link_root / versioned_name

            if self._remove_single_link(link_path):
                count += 1

        if count > 0:
            logger.info("Removed %d links for %s@%s", count, compiler, version)
        return count

    def set_default(
        self,
        compiler: str,
        version: str,
    ) -> int:
        """
        Change the default version and update short links.

        Removes old short links, updates the default symlink,
        and creates new short links pointing to the new version.

        Parameters
        ----------
        compiler : str
            Compiler name.
        version : str
            Version string.

        Returns
        -------
        int
            Number of short links updated.

        Warnings
        --------
        - This overwrites existing short links.
        - If the new version is not installed, no links are created.
        """
        from .layouts import set_default as layout_set_default

        # Get old default to clean up its short links
        old_default = get_default_version(compiler, self.install_root)

        # Update the default symlink
        if not layout_set_default(compiler, version, self.install_root):
            return 0

        # Remove old short links
        if old_default and old_default != version:
            self._remove_short_links_for_version(compiler, old_default)

        # Create new short links
        return self._create_short_links(compiler, version)

    def remove_short_links(self, compiler: str) -> int:
        """
        Remove all short (non-versioned) links for a compiler family.

        Parameters
        ----------
        compiler : str
            Compiler name.

        Returns
        -------
        int
            Number of short links removed.
        """
        count = 0
        for entry in self.link_root.iterdir():
            if not entry.is_symlink() and not entry.is_file():
                continue
            name = entry.name
            if "@" in name:
                continue  # Skip versioned links

            # Check if this link points to the given compiler's directory
            try:
                target = self._read_link_target(entry)
                if target and f"/{compiler}/" in str(target):
                    if self._remove_single_link(entry):
                        count += 1
            except OSError:
                pass

        if count > 0:
            logger.info("Removed %d short links for %s", count, compiler)
        return count

    def rescan(self) -> Tuple[int, int]:
        """
        Scan for and clean up stale links.

        A link is stale if its target does not exist.

        Returns
        -------
        Tuple[int, int]
            (stale_count, removed_count) — number of stale links
            found and number successfully removed.
        """
        stale = []
        for entry in self.link_root.iterdir():
            if entry.is_symlink():
                try:
                    target = Path(os.readlink(str(entry)))
                    if not target.exists():
                        stale.append(entry)
                except OSError:
                    stale.append(entry)
            elif entry.is_file() and entry.suffix in (".bat", ".cmd"):
                # Check wrapper scripts
                content = entry.read_text()
                if not self._wrapper_target_exists(content):
                    stale.append(entry)

        removed = 0
        for link in stale:
            if self._remove_single_link(link):
                removed += 1

        if stale:
            logger.info(
                "Found %d stale links, removed %d", len(stale), removed
            )

        return len(stale), removed

    def list_links(
        self,
        compiler: Optional[str] = None,
    ) -> Dict[str, List[Tuple[str, Path]]]:
        """
        List all managed links, grouped by compiler.

        Parameters
        ----------
        compiler : Optional[str]
            Filter by compiler name. If None, all are listed.

        Returns
        -------
        Dict[str, List[Tuple[str, Path]]]
            Map of compiler → list of (link_name, target_path).
        """
        result: Dict[str, List[Tuple[str, Path]]] = {}

        for entry in sorted(self.link_root.iterdir()):
            try:
                target = self._read_link_target(entry)
            except OSError:
                continue

            if target is None:
                continue

            # Determine compiler from target path
            detected_compiler = self._compiler_from_path(target)
            if detected_compiler is None:
                continue

            if compiler is not None and detected_compiler != compiler:
                continue

            if detected_compiler not in result:
                result[detected_compiler] = []

            result[detected_compiler].append((entry.name, target))

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_single_link(
        self,
        target: Path,
        link_name: str,
        force: bool = False,
    ) -> bool:
        """
        Create a single symlink or wrapper script.

        Parameters
        ----------
        target : Path
            Absolute path to the executable.
        link_name : str
            Name of the link in the link root.
        force : bool
            If True, overwrite existing files.

        Returns
        -------
        bool
            True if the link was created.
        """
        link_path = self.link_root / link_name

        # Check if already exists
        if link_path.exists():
            if force:
                self._remove_single_link(link_path)
            else:
                logger.debug("Link already exists: %s", link_path)
                return False

        if self._use_symlinks:
            return self._create_symlink(target, link_path)
        else:
            return self._create_wrapper(target, link_path)

    @staticmethod
    def _create_symlink(target: Path, link_path: Path) -> bool:
        """
        Create a symbolic link.

        Parameters
        ----------
        target : Path
            Target file.
        link_path : Path
            Link location.

        Returns
        -------
        bool
            True on success.
        """
        try:
            link_path.symlink_to(target)
            return True
        except OSError as exc:
            logger.error("Failed to create symlink %s -> %s: %s", link_path, target, exc)
            return False

    @staticmethod
    def _create_wrapper(target: Path, link_path: Path) -> bool:
        """
        Create a .bat wrapper script (Windows fallback).

        Parameters
        ----------
        target : Path
            Target executable.
        link_path : Path
            Wrapper location (forced to .bat extension).

        Returns
        -------
        bool
            True on success.
        """
        if link_path.suffix not in (".bat", ".cmd"):
            link_path = link_path.with_suffix(".bat")

        content = (
            f'@echo off\r\n'
            f'"{target}" %*\r\n'
        )

        try:
            link_path.write_text(content)
            link_path.chmod(0o755)
            return True
        except OSError as exc:
            logger.error("Failed to create wrapper %s: %s", link_path, exc)
            return False

    @staticmethod
    def _remove_single_link(link_path: Path) -> bool:
        """
        Remove a single symlink or wrapper.

        Parameters
        ----------
        link_path : Path
            Path to remove.

        Returns
        -------
        bool
            True if removed.
        """
        try:
            if link_path.is_symlink() or link_path.is_file():
                link_path.unlink()
                return True
        except OSError as exc:
            logger.warning("Failed to remove %s: %s", link_path, exc)
        return False

    @staticmethod
    def _read_link_target(entry: Path) -> Optional[Path]:
        """
        Read the target of a symlink or wrapper.

        Parameters
        ----------
        entry : Path
            Symlink or wrapper file.

        Returns
        -------
        Optional[Path]
            Target path, or None if it cannot be determined.
        """
        if entry.is_symlink():
            try:
                return Path(os.readlink(str(entry)))
            except OSError:
                return None
        elif entry.is_file() and entry.suffix in (".bat", ".cmd"):
            content = entry.read_text()
            # Extract path from: @"C:\path\to\exe" %*
            match = re.search(r'"([^"]+)"', content)
            if match:
                return Path(match.group(1))
        return None

    @staticmethod
    def _wrapper_target_exists(content: str) -> bool:
        """
        Check if the target referenced in a wrapper script exists.

        Parameters
        ----------
        content : str
            Contents of a .bat wrapper.

        Returns
        -------
        bool
            True if the referenced executable exists.
        """
        match = re.search(r'"([^"]+)"', content)
        if match:
            return Path(match.group(1)).exists()
        return False

    @staticmethod
    def _compiler_from_path(target: Path) -> Optional[str]:
        """
        Determine the compiler name from a target path.

        Expects the path structure: `{root}/{compiler}/{version}/bin/{exe}`.

        Parameters
        ----------
        target : Path
            Target path of a link.

        Returns
        -------
        Optional[str]
            Compiler name, or None if the pattern does not match.
        """
        parts = target.parts
        # Look for .../compiler/version/bin/exe
        if len(parts) >= 4 and parts[-2] == "bin":
            return parts[-3]
        return None

    def _find_linked_executables(
        self, compiler: str, version: str
    ) -> List[str]:
        """
        Discover executable names from existing versioned links.

        Used when the toolchain directory has already been removed
        but links still exist.

        Parameters
        ----------
        compiler : str
            Compiler name.
        version : str
            Version string.

        Returns
        -------
        List[str]
            Executable base names (without @version).
        """
        major = _extract_major_version(version)
        suffix = f"@{major}"

        executables = []
        for entry in self.link_root.iterdir():
            if entry.name.endswith(suffix):
                base_name = entry.name[: -len(suffix)]
                executables.append(base_name)

        return executables

    def _create_short_links(
        self, compiler: str, version: str
    ) -> int:
        """
        Create short (non-versioned) links for a specific version.

        Parameters
        ----------
        compiler : str
            Compiler name.
        version : str
            Version string.

        Returns
        -------
        int
            Number of short links created.
        """
        layout = ToolchainLayout(compiler, version, self.install_root)
        if not layout.exists():
            return 0

        executables = _find_executables(layout.bin_dir)
        count = 0

        for exe_name in executables:
            exe_path = layout.bin_dir / exe_name
            short_name = _short_link_name(exe_name)
            if self._create_single_link(exe_path, short_name, force=True):
                count += 1

        return count

    def _remove_short_links_for_version(
        self, compiler: str, version: str
    ) -> int:
        """
        Remove short links pointing to a specific version.

        Parameters
        ----------
        compiler : str
            Compiler name.
        version : str
            Version string.

        Returns
        -------
        int
            Number of short links removed.
        """
        layout = ToolchainLayout(compiler, version, self.install_root)
        target_bin = str(layout.bin_dir)
        count = 0

        for entry in self.link_root.iterdir():
            if not entry.is_symlink() and not entry.is_file():
                continue
            if "@" in entry.name:
                continue

            try:
                target = self._read_link_target(entry)
                if target and str(target.parent) == target_bin:
                    if self._remove_single_link(entry):
                        count += 1
            except OSError:
                pass

        return count
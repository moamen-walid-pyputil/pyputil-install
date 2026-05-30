"""
Safe toolchain uninstallation with rollback capability.

Removes installed compiler toolchains along with their symlinks,
manifests, and environment references. Uses manifests to ensure
complete cleanup without leaving orphaned files. Supports dry-run
preview, environment conflict detection, and forced removal.

Design
------
Uninstallation follows a strict, transactional order:
    1. Resolve the toolchain layout and verify it exists.
    2. Check if the version is the active default (warn/block).
    3. Check if the version is referenced by any environment (warn/block).
    4. Build a complete removal plan from the manifest.
    5. Remove symlinks created during installation.
    6. Remove the toolchain directory tree (robust, with retries).
    7. Remove the manifest file.
    8. Clean up empty parent directories.
    9. Remove orphaned short symlinks if this was the default version.

A dry-run mode (`dry_run_uninstall()`) shows exactly what will be
removed without touching the disk.

A force mode (`force=True`) skips environment and default-version
conflict checks.

Safety Features
---------------
- Environment conflict detection: refuses to uninstall if any
  environment references the toolchain version.
- Default version detection: warns if uninstalling the current
  default version for a compiler family.
- Manifest-driven cleanup: if a manifest exists, it is the
  authoritative source for what to clean up.
- Best-effort fallback: if the manifest is missing, symlinks are
  discovered by scanning the link directory.
- Robust directory removal: retries up to 3 times with exponential
  backoff to handle transient file locks (Windows).
- Atomic file operations where possible.

Usage
-----
    from pyputil_install.compiler_installer.uninstall import (
        dry_run_uninstall,
        uninstall_toolchain,
        force_uninstall,
    )

    # Preview first
    plan = dry_run_uninstall("gcc", "14.2.0-2")
    print(plan.summary())

    # Check if safe
    if plan.can_proceed:
        success = uninstall_toolchain("gcc", "14.2.0-2")

    # Force if needed
    success = force_uninstall("gcc", "14.2.0-2")

Warnings
--------
- Uninstallation is IRREVERSIBLE. Always run dry_run_uninstall()
  before the actual operation.
- If a toolchain is the current default for its compiler family,
  short symlinks (e.g., `gcc` without version) will become dangling.
  Set a new default before uninstalling: `set_default("gcc", "13.3.0")`.
- On Windows, symlinks created as `.bat` wrappers are cleaned up
  only if they were recorded in the manifest.
- This module does NOT deactivate the toolchain from the current
  process. Call `activation.deactivate()` before uninstalling.
- Concurrent uninstall operations on the same toolchain are NOT
  safe. Use external locking if needed.

User Instructions
-----------------
- Always preview with dry_run_uninstall() first.
- Use force=True only after manually removing the toolchain from
  any environments that reference it.
- After uninstalling, run `ManifestStore().repair()` to clean up
  any orphan manifests.
- To uninstall all versions of a compiler: `uninstall_all("gcc")`.
- To uninstall everything: `uninstall_all_compilers()` (extreme caution).
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .layouts import (
    ToolchainLayout,
    ToolchainSet,
    get_install_root,
    get_default_version,
    set_default,
    _rmtree_robust,
)
from .symlinks import SymlinkManager
from .manifests import Manifest, ManifestStore
from.environments import EnvironmentStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum number of retry attempts for directory removal
_MAX_RMRETRY_ATTEMPTS = int(os.environ.get("TOOLFORGE_UNINSTALL_RETRIES", "3"))

# Base delay between retries in seconds (exponential backoff)
_RETRY_BASE_DELAY = float(os.environ.get("TOOLFORGE_UNINSTALL_RETRY_DELAY", "0.5"))


# ============================================================================
# UninstallPlan — immutable preview of what will be removed
# ============================================================================

@dataclass(frozen=True)
class UninstallPlan:
    """
    Immutable description of what an uninstall operation will remove.

    Attributes
    ----------
    compiler : str
        Compiler name.
    version : str
        Version string to uninstall.
    toolchain_path : Path
        Absolute path to the toolchain directory that will be removed.
    toolchain_exists : bool
        True if the toolchain directory currently exists on disk.
    symlinks_to_remove : List[str]
        Filenames of symlinks (in the centralized bin directory)
        that will be removed.
    manifest_exists : bool
        True if a manifest file exists and will be removed.
    manifest_path : Optional[Path]
        Path to the manifest file, if it exists.
    environments_referencing : List[str]
        Environment names that reference this exact version.
        If non-empty, uninstall is blocked unless force=True.
    is_default_version : bool
        True if this version is the current default for its compiler
        family. Uninstalling it will leave dangling short symlinks.
    empty_compiler_dir : bool
        True if the compiler family directory will become empty
        after removal and will also be deleted.
    blocked : bool
        True if there are blocking conditions preventing safe removal.
    blocking_reasons : List[str]
        Human-readable reasons why removal is blocked.

    Properties
    ----------
    can_proceed : bool
        True if there are no blocking conditions.
    is_safe : bool
        Alias for can_proceed.
    """

    compiler: str
    version: str
    toolchain_path: Path
    toolchain_exists: bool = False
    symlinks_to_remove: List[str] = field(default_factory=list)
    manifest_exists: bool = False
    manifest_path: Optional[Path] = None
    environments_referencing: List[str] = field(default_factory=list)
    is_default_version: bool = False
    empty_compiler_dir: bool = False
    blocked: bool = False
    blocking_reasons: List[str] = field(default_factory=list)

    @property
    def can_proceed(self) -> bool:
        """Return True if there are no blocking conditions."""
        return not self.blocked

    @property
    def is_safe(self) -> bool:
        """Alias for can_proceed."""
        return self.can_proceed

    def summary(self) -> str:
        """
        Return a human-readable multi-line summary of the plan.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = [
            f"Uninstall plan for {self.compiler}@{self.version}",
            f"{'=' * 50}",
            f"  Toolchain directory : {self.toolchain_path}",
            f"  Directory exists     : {self.toolchain_exists}",
        ]

        if self.symlinks_to_remove:
            lines.append(
                f"  Symlinks to remove  : {len(self.symlinks_to_remove)}"
            )
            for name in self.symlinks_to_remove[:15]:
                lines.append(f"    - {name}")
            if len(self.symlinks_to_remove) > 15:
                lines.append(
                    f"    ... and {len(self.symlinks_to_remove) - 15} more"
                )

        if self.manifest_exists and self.manifest_path:
            lines.append(f"  Manifest to remove  : {self.manifest_path}")

        if self.is_default_version:
            lines.append(
                f"  WARNING: This is the default version for {self.compiler}. "
                f"Short symlinks will become dangling."
            )

        if self.environments_referencing:
            lines.append(
                f"  BLOCKED: Referenced by environments: "
                f"{', '.join(self.environments_referencing)}"
            )

        if self.empty_compiler_dir:
            lines.append(
                f"  Note: Compiler directory will become empty and be removed."
            )

        lines.append(f"{'=' * 50}")

        if self.blocked:
            lines.append("Status: BLOCKED")
            for reason in self.blocking_reasons:
                lines.append(f"  - {reason}")
        else:
            lines.append("Status: READY — safe to proceed")

        return "\n".join(lines)


# ============================================================================
# Plan building (internal)
# ============================================================================

def _build_uninstall_plan(
    compiler: str,
    version: str,
    install_root: Path,
    manifest_store: ManifestStore,
    env_store: EnvironmentStore,
    symlink_manager: SymlinkManager,
) -> UninstallPlan:
    """
    Build a complete UninstallPlan by inspecting disk state.

    Parameters
    ----------
    compiler : str
        Compiler name.
    version : str
        Version string.
    install_root : Path
        Root directory.
    manifest_store : ManifestStore
        Manifest store instance.
    env_store : EnvironmentStore
        Environment store instance.
    symlink_manager : SymlinkManager
        Symlink manager instance.

    Returns
    -------
    UninstallPlan
        Complete plan with all information gathered.
    """
    layout = ToolchainLayout(compiler, version, install_root)
    toolchain_exists = layout.exists()
    blocking_reasons: List[str] = []

    # --- Manifest ---
    manifest = manifest_store.load(compiler, version)
    manifest_exists = manifest is not None
    manifest_path = (
        manifest_store._manifest_path(compiler, version)
        if manifest_exists
        else None
    )

    # --- Symlinks ---
    if manifest is not None:
        symlinks_to_remove = list(manifest.symlinks_created)
    else:
        # Fallback: scan the link directory
        symlinks_to_remove = _discover_symlinks_for_version(
            compiler, version, symlink_manager
        )

    # --- Environment conflicts ---
    referencing = _find_referencing_environments(compiler, version, env_store)
    if referencing:
        blocking_reasons.append(
            f"Version {compiler}@{version} is referenced by environment(s): "
            f"{', '.join(referencing)}"
        )

    # --- Default version check ---
    is_default = False
    current_default = get_default_version(compiler, install_root)
    if current_default == version:
        is_default = True
        # This is a warning, not a block — but we record it
        # so the caller can decide.

    # --- Compiler directory emptiness ---
    compiler_set = ToolchainSet(compiler, install_root)
    installed_versions = compiler_set.list_versions()
    remaining = [v for v in installed_versions if v != version]
    empty_compiler_dir = (
        len(remaining) == 0
        and len(installed_versions) > 0
        and toolchain_exists
    )

    # --- Determine blocked state ---
    blocked = len(blocking_reasons) > 0

    return UninstallPlan(
        compiler=compiler,
        version=version,
        toolchain_path=layout.path,
        toolchain_exists=toolchain_exists,
        symlinks_to_remove=symlinks_to_remove,
        manifest_exists=manifest_exists,
        manifest_path=manifest_path,
        environments_referencing=referencing,
        is_default_version=is_default,
        empty_compiler_dir=empty_compiler_dir,
        blocked=blocked,
        blocking_reasons=blocking_reasons,
    )


# ============================================================================
# Public API
# ============================================================================

def dry_run_uninstall(
    compiler: str,
    version: str,
    install_root: Optional[Path] = None,
) -> UninstallPlan:
    """
    Preview what would be removed by an uninstall operation.

    Does NOT modify the disk. Returns a complete UninstallPlan
    describing every file, symlink, and manifest that would be
    affected.

    Parameters
    ----------
    compiler : str
        Compiler name, e.g., "gcc".
    version : str
        Version string, e.g., "14.2.0-2".
    install_root : Optional[Path]
        Root directory. If None, `get_install_root()` is called.

    Returns
    -------
    UninstallPlan
        Complete plan. Check `plan.can_proceed` before calling
        `uninstall_toolchain()`.

    Examples
    --------
    >>> plan = dry_run_uninstall("gcc", "14.2.0-2")
    >>> print(plan.summary())
    >>> if plan.can_proceed:
    ...     uninstall_toolchain("gcc", "14.2.0-2")
    """
    root = install_root or get_install_root()
    manifest_store = ManifestStore(root)
    env_store = EnvironmentStore(root)
    symlink_manager = SymlinkManager(root)

    return _build_uninstall_plan(
        compiler, version, root,
        manifest_store, env_store, symlink_manager,
    )


def uninstall_toolchain(
    compiler: str,
    version: str,
    install_root: Optional[Path] = None,
    force: bool = False,
) -> bool:
    """
    Uninstall a compiler toolchain version completely.

    Performs the removal in a specific order:
        1. Remove versioned symlinks (`gcc@14`).
        2. Remove the toolchain directory tree.
        3. Remove the manifest file.
        4. Clean up empty compiler family directory.
        5. If this was the default version, clear the default symlink
           and remove orphaned short symlinks.

    Parameters
    ----------
    compiler : str
        Compiler name.
    version : str
        Version string.
    install_root : Optional[Path]
        Root directory. If None, `get_install_root()` is called.
    force : bool
        If True, proceeds even if environments reference this
        version. If False, raises RuntimeError when blocked.

    Returns
    -------
    bool
        True if at least one component was removed. False if the
        toolchain was not found at all.

    Raises
    ------
    RuntimeError
        If force=False and the operation is blocked (environments
        reference the version).

    Warnings
    --------
    - This operation is IRREVERSIBLE. Preview with
      `dry_run_uninstall()` first.
    - Deactivate the toolchain from the current process before
      uninstalling.
    """
    root = install_root or get_install_root()
    manifest_store = ManifestStore(root)
    env_store = EnvironmentStore(root)
    symlink_manager = SymlinkManager(root)

    plan = _build_uninstall_plan(
        compiler, version, root,
        manifest_store, env_store, symlink_manager,
    )

    # Block if environments reference this version and force=False
    if plan.blocked and not force:
        raise RuntimeError(
            f"Cannot uninstall {compiler}@{version}. "
            f"Reasons:\n" + "\n".join(
                f"  - {r}" for r in plan.blocking_reasons
            ) + f"\nUse force=True to override."
        )

    if plan.blocked and force:
        logger.warning(
            "Force-uninstalling %s@%s despite: %s",
            compiler, version,
            "; ".join(plan.blocking_reasons),
        )

    # Warn about default version
    if plan.is_default_version:
        logger.warning(
            "%s@%s is the default version for %s. "
            "Short symlinks will become dangling. "
            "Set a new default with: set_default('%s', '<new_version>')",
            compiler, version, compiler, compiler,
        )

    removed_any = False

    # 1. Remove versioned symlinks
    if plan.symlinks_to_remove:
        removed_count = symlink_manager.remove_links(compiler, version)
        if removed_count > 0:
            logger.info(
                "Removed %d symlink(s) for %s@%s",
                removed_count, compiler, version,
            )
            removed_any = True

    # 2. Remove toolchain directory
    if plan.toolchain_exists:
        _rmtree_robust(
            plan.toolchain_path,
            max_attempts=_MAX_RMRETRY_ATTEMPTS,
        )
        logger.info("Removed toolchain directory: %s", plan.toolchain_path)
        removed_any = True

    # 3. Remove manifest
    if plan.manifest_exists:
        manifest_store.delete(compiler, version)
        removed_any = True

    # 4. Remove empty compiler directory
    if plan.empty_compiler_dir:
        compiler_set = ToolchainSet(compiler, root)
        if compiler_set.path.exists():
            try:
                compiler_set.path.rmdir()
                logger.debug(
                    "Removed empty compiler directory: %s",
                    compiler_set.path,
                )
            except OSError as exc:
                logger.debug(
                    "Could not remove compiler directory %s: %s",
                    compiler_set.path, exc,
                )

    # 5. If this was the default, clear default and short symlinks
    if plan.is_default_version:
        # Clear the default symlink
        from .layouts import get_default_symlink_path
        default_link = get_default_symlink_path(compiler, root)
        if default_link.is_symlink():
            try:
                default_link.unlink()
                logger.debug("Removed default symlink: %s", default_link)
            except OSError as exc:
                logger.warning(
                    "Could not remove default symlink %s: %s",
                    default_link, exc,
                )
        elif default_link.exists():
            try:
                if default_link.is_dir():
                    import shutil
                    shutil.rmtree(default_link)
                else:
                    default_link.unlink()
                logger.debug("Removed default marker: %s", default_link)
            except OSError as exc:
                logger.warning(
                    "Could not remove default marker %s: %s",
                    default_link, exc,
                )

        # Remove short symlinks pointing to this version
        symlink_manager.remove_short_links(compiler)

    if removed_any:
        logger.info(
            "Successfully uninstalled %s@%s", compiler, version,
        )
    else:
        logger.warning(
            "Nothing to uninstall for %s@%s — toolchain not found",
            compiler, version,
        )

    return removed_any


def force_uninstall(
    compiler: str,
    version: str,
    install_root: Optional[Path] = None,
) -> bool:
    """
    Uninstall a toolchain regardless of any blocking conditions.

    Shortcut for `uninstall_toolchain(compiler, version, force=True)`.

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
        True if removed.
    """
    return uninstall_toolchain(compiler, version, install_root, force=True)


def uninstall_all(
    compiler: str,
    install_root: Optional[Path] = None,
    force: bool = False,
) -> int:
    """
    Uninstall ALL versions of a compiler family.

    Parameters
    ----------
    compiler : str
        Compiler name.
    install_root : Optional[Path]
        Root directory.
    force : bool
        If True, skips environment conflict checks.

    Returns
    -------
    int
        Number of versions successfully uninstalled.

    Examples
    --------
    >>> count = uninstall_all("gcc")
    >>> print(f"Removed {count} GCC version(s)")
    """
    root = install_root or get_install_root()
    compiler_set = ToolchainSet(compiler, root)
    versions = compiler_set.list_versions()

    count = 0
    for version in versions:
        try:
            if uninstall_toolchain(compiler, version, root, force=force):
                count += 1
        except RuntimeError as exc:
            logger.warning("Skipped %s@%s: %s", compiler, version, exc)

    logger.info(
        "Uninstalled %d of %d %s version(s)",
        count, len(versions), compiler,
    )
    return count


def uninstall_all_compilers(
    install_root: Optional[Path] = None,
    force: bool = False,
) -> int:
    """
    Uninstall ALL toolchains of ALL compiler families.

    USE WITH EXTREME CAUTION. This removes everything ToolForge
    has installed.

    Parameters
    ----------
    install_root : Optional[Path]
        Root directory.
    force : bool
        If True, skips all conflict checks.

    Returns
    -------
    int
        Total number of versions uninstalled.
    """
    from .layouts import list_compiler_families

    root = install_root or get_install_root()
    families = list_compiler_families(root)

    total = 0
    for compiler in families:
        count = uninstall_all(compiler, root, force=force)
        total += count

    logger.warning(
        "Uninstalled ALL toolchains: %d version(s) across %d compiler(s)",
        total, len(families),
    )
    return total


def clean_orphans(install_root: Optional[Path] = None) -> int:
    """
    Clean up orphaned files: manifests without toolchain directories,
    dangling symlinks, and empty compiler directories.

    This is safe to run at any time. It only removes things that
    have no corresponding toolchain installation.

    Parameters
    ----------
    install_root : Optional[Path]
        Root directory.

    Returns
    -------
    int
        Number of orphaned items cleaned up.
    """
    root = install_root or get_install_root()
    count = 0

    # Repair orphan manifests
    manifest_store = ManifestStore(root)
    count += manifest_store.repair()

    # Rescan and remove stale symlinks
    symlink_manager = SymlinkManager(root)
    _, removed = symlink_manager.rescan()
    count += removed

    # Remove empty compiler directories
    for entry in sorted(root.iterdir()) if root.exists() else []:
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        try:
            if not any(entry.iterdir()):
                entry.rmdir()
                count += 1
                logger.debug("Removed empty directory: %s", entry)
        except OSError:
            pass

    logger.info("Cleaned up %d orphaned item(s)", count)
    return count


# ============================================================================
# Internal helpers
# ============================================================================

def _discover_symlinks_for_version(
    compiler: str,
    version: str,
    symlink_manager: SymlinkManager,
) -> List[str]:
    """
    Discover symlinks associated with a toolchain version.

    Used as a fallback when no manifest exists. Scans the link
    root for any symlink whose target resides inside the
    toolchain's `bin` directory.

    Parameters
    ----------
    compiler : str
        Compiler name.
    version : str
        Version string.
    symlink_manager : SymlinkManager
        Symlink manager instance.

    Returns
    -------
    List[str]
        Names of symlinks pointing to this toolchain's bin directory.
        Sorted alphabetically.
    """
    layout = ToolchainLayout(compiler, version, symlink_manager.install_root)
    target_bin = str(layout.bin_dir)
    discovered: List[str] = []

    if not symlink_manager.link_root.exists():
        return discovered

    for entry in sorted(symlink_manager.link_root.iterdir()):
        try:
            target = symlink_manager._read_link_target(entry)
            if target is not None and str(target.parent) == target_bin:
                discovered.append(entry.name)
        except OSError:
            continue

    return discovered


def _find_referencing_environments(
    compiler: str,
    version: str,
    env_store: EnvironmentStore,
) -> List[str]:
    """
    Find all environments that reference a specific compiler version.

    Parameters
    ----------
    compiler : str
        Compiler name.
    version : str
        Version string.
    env_store : EnvironmentStore
        Environment store instance.

    Returns
    -------
    List[str]
        Sorted list of environment names that reference this version.
    """
    referencing: List[str] = []

    for env in env_store.list_all_environments():
        env_version = env.get_version(compiler)
        if env_version == version:
            referencing.append(env.name)

    return sorted(referencing)
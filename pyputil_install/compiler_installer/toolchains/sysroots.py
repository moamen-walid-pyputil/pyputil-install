"""
Sysroot detection and management for cross-compilation toolchains.

A sysroot is a directory that mimics the root filesystem of a target
platform. It contains headers, libraries, and support files that a
cross-compiler needs to build software for a different architecture
or operating system than the host.

This module detects sysroots that are:
    - Bundled with a toolchain (e.g., Android NDK's sysroot).
    - Installed separately (e.g., in /sysroot or $SYSROOT).
    - Managed by ToolForge (under the installation root).
    - System sysroots for native compilation.

Design
------
Sysroot detection is independent of any specific toolchain.
Functions accept a Toolchain object or a direct Path, examine
the toolchain's configuration, and return SysrootInfo objects
describing what was found.

Usage
-----
    from pathlib import Path
    from pyputil_install.compiler_installer.toolchains.sysroots import detect_sysroot, SysrootInfo
    from pyputil_install.compiler_installer.toolchains.gcc import GCCToolchain

    gcc = GCCToolchain(Path("/opt/arm-gnu-toolchain"))
    sysroot = detect_sysroot(gcc)
    if sysroot:
        print(sysroot.path)        # /opt/arm-gnu-toolchain/arm-none-eabi/sysroot
        print(sysroot.include_dir) # /opt/.../sysroot/usr/include
        print(sysroot.lib_dir)     # /opt/.../sysroot/usr/lib

Warnings
--------
- Not all toolchains have a sysroot. Native compilers typically
  use the host system root (/) as their sysroot.
- The detected sysroot path may not be valid if the toolchain
  was moved after installation.
- Some toolchains have multiple sysroots (e.g., one per API level
  in Android NDK). This module returns the default one.

User Instructions
-----------------
- Use detect_sysroot() to find the sysroot for a given toolchain.
- Use list_available_sysroots() for toolchains with multiple sysroots
  (like Android NDK with its per-API-level platform directories).
"""

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .base import Toolchain

logger = logging.getLogger(__name__)

# ============================================================================
# SysrootInfo
# ============================================================================


@dataclass(frozen=True)
class SysrootInfo:
    """
    Information about a detected sysroot.

    Attributes
    ----------
    path : Path
        Absolute path to the sysroot root directory.
    include_dir : Optional[Path]
        Path to the usr/include directory within the sysroot.
        This is where standard headers (stdio.h, stdlib.h) are found.
    lib_dir : Optional[Path]
        Path to the usr/lib directory within the sysroot.
        This is where runtime libraries (libc.so, crt0.o) are found.
    is_relative : bool
        True if the sysroot path is relative to the toolchain
        installation directory. Such sysroots should not be moved
        independently of the toolchain.
    source : str
        How the sysroot was found:
        - "toolchain" : reported by the compiler itself.
        - "environment" : from SYSROOT or {target}_SYSROOT env var.
        - "bundled" : found in a well-known subdirectory of the toolchain.
        - "managed" : found in the ToolForge managed sysroots directory.
        - "system" : the host root (/) for native compilation.
        - "manual" : explicitly provided by the user.
    """

    path: Path
    include_dir: Optional[Path] = None
    lib_dir: Optional[Path] = None
    is_relative: bool = False
    source: str = "toolchain"


# ============================================================================
# Detection
# ============================================================================


def detect_sysroot(
    toolchain,
    target: Optional[str] = None,
) -> Optional[SysrootInfo]:
    """
    Detect the sysroot for a given toolchain.

    Uses multiple strategies in order:
    1. Ask the compiler directly via -print-sysroot.
    2. Check environment variables (SYSROOT, {target}_SYSROOT).
    3. Look for a bundled sysroot in the toolchain directory.
    4. Look in the ToolForge managed sysroots directory.
    5. Fall back to the host root (/) for native compilers.

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain to query, or a direct path to a compiler.
    target : Optional[str]
        Target triplet to help locate the correct sysroot when
        the toolchain has multiple. Example: "aarch64-linux-android".

    Returns
    -------
    Optional[SysrootInfo]
        Detected sysroot, or None if no sysroot was found and
        the compiler is not native.
    """
    # Strategy 1: Ask the compiler
    sysroot = _detect_from_compiler(toolchain)
    if sysroot and sysroot.path != Path("/"):
        return sysroot

    # Strategy 2: Environment variables
    sysroot = _detect_from_environment(target)
    if sysroot:
        return sysroot

    # Strategy 3: Bundled with the toolchain
    sysroot = _detect_bundled(toolchain, target)
    if sysroot:
        return sysroot

    # Strategy 4: ToolForge managed
    sysroot = _detect_managed(toolchain, target)
    if sysroot:
        return sysroot

    # Strategy 5: Native fallback
    if sysroot and sysroot.path == Path("/"):
        return SysrootInfo(
            path=Path("/"),
            include_dir=Path("/usr/include"),
            lib_dir=Path("/usr/lib"),
            source="system",
        )

    return None


def _detect_from_compiler(toolchain) -> Optional[SysrootInfo]:
    """
    Ask the compiler for its sysroot via -print-sysroot.

    Works with GCC and Clang. The compiler must be accessible
    and respond within 10 seconds.

    Parameters
    ----------
    toolchain : Toolchain or Path
        Toolchain object or compiler path.

    Returns
    -------
    Optional[SysrootInfo]
        Sysroot reported by the compiler, or None.
    """
    compiler = _get_compiler_path(toolchain)
    if compiler is None:
        return None

    try:
        result = subprocess.run(
            [str(compiler), "-print-sysroot"],
            capture_output=True,
            text=True,
            timeout=10,
            shell=False,
        )
        if result.returncode != 0:
            return None

        sysroot_path = Path(result.stdout.strip().splitlines()[0].strip())
        if not sysroot_path.exists():
            return None

        include_dir = sysroot_path / "usr" / "include"
        lib_dir = sysroot_path / "usr" / "lib"
        if not include_dir.is_dir():
            include_dir = None
        if not lib_dir.is_dir():
            lib_dir = None

        return SysrootInfo(
            path=sysroot_path,
            include_dir=include_dir,
            lib_dir=lib_dir,
            is_relative=not sysroot_path.is_absolute(),
            source="toolchain",
        )
    except Exception as exc:
        logger.debug("Sysroot detection via compiler failed: %s", exc)

    return None


def _detect_from_environment(target: Optional[str] = None) -> Optional[SysrootInfo]:
    """
    Check environment variables for a sysroot path.

    Checks SYSROOT first, then {target}_SYSROOT if a target
    triplet is provided (e.g., AARCH64_LINUX_ANDROID_SYSROOT).

    Parameters
    ----------
    target : Optional[str]
        Target triplet for target-specific env var lookup.

    Returns
    -------
    Optional[SysrootInfo]
        Sysroot from environment, or None.
    """
    # Generic SYSROOT
    sysroot_env = os.environ.get("SYSROOT")
    if sysroot_env:
        path = Path(sysroot_env)
        if path.is_dir():
            return SysrootInfo(
                path=path,
                include_dir=path / "usr" / "include" if (path / "usr" / "include").is_dir() else None,
                lib_dir=path / "usr" / "lib" if (path / "usr" / "lib").is_dir() else None,
                source="environment",
            )

    # Target-specific
    if target:
        target_var = target.upper().replace("-", "_") + "_SYSROOT"
        target_env = os.environ.get(target_var)
        if target_env:
            path = Path(target_env)
            if path.is_dir():
                return SysrootInfo(
                    path=path,
                    include_dir=path / "usr" / "include" if (path / "usr" / "include").is_dir() else None,
                    lib_dir=path / "usr" / "lib" if (path / "usr" / "lib").is_dir() else None,
                    source="environment",
                )

    return None


def _detect_bundled(toolchain, target: Optional[str] = None) -> Optional[SysrootInfo]:
    """
    Look for a sysroot bundled inside the toolchain directory.

    Common locations:
        {prefix}/{target}/sysroot    (GCC cross-compiler)
        {prefix}/sysroot             (Android NDK symlink)
        {prefix}/../sysroot          (sibling directory)

    Parameters
    ----------
    toolchain : Toolchain or Path
        Toolchain object or path.
    target : Optional[str]
        Target triplet to locate target-specific sysroots.

    Returns
    -------
    Optional[SysrootInfo]
        Bundled sysroot, or None.
    """
    prefix = _get_prefix_path(toolchain)
    if prefix is None:
        return None

    candidates = []

    # If we have a target triplet, look for {prefix}/{target}/sysroot
    if target:
        candidates.append(prefix / target / "sysroot")
        candidates.append(prefix / target)

    # Standard bundled paths
    candidates.append(prefix / "sysroot")
    candidates.append(prefix.parent / "sysroot")

    # Android NDK style
    toolchains_dir = prefix / "toolchains" / "llvm" / "prebuilt"
    if toolchains_dir.is_dir():
        for host_dir in toolchains_dir.iterdir() if toolchains_dir.is_dir() else []:
            candidates.append(host_dir / "sysroot")

    for candidate in candidates:
        if candidate.is_dir() and _looks_like_sysroot(candidate):
            include_dir = candidate / "usr" / "include"
            lib_dir = candidate / "usr" / "lib"
            return SysrootInfo(
                path=candidate,
                include_dir=include_dir if include_dir.is_dir() else None,
                lib_dir=lib_dir if lib_dir.is_dir() else None,
                is_relative=not candidate.is_absolute(),
                source="bundled",
            )

    return None


def _detect_managed(toolchain, target: Optional[str] = None) -> Optional[SysrootInfo]:
    """
    Look for a sysroot in the ToolForge managed sysroots directory.

    Structure:
        {toolforge_home}/sysroots/{target}/

    Parameters
    ----------
    toolchain : Toolchain or Path
        Toolchain object or path.
    target : Optional[str]
        Target triplet.

    Returns
    -------
    Optional[SysrootInfo]
        Managed sysroot, or None.
    """
    if target is None:
        return None

    toolforge_home = os.environ.get("TOOLFORGE_HOME")
    if toolforge_home is None:
        xdg = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
        toolforge_home = str(Path(xdg) / "toolforge")

    managed = Path(toolforge_home) / "sysroots" / target
    if managed.is_dir() and _looks_like_sysroot(managed):
        return SysrootInfo(
            path=managed,
            include_dir=managed / "usr" / "include" if (managed / "usr" / "include").is_dir() else None,
            lib_dir=managed / "usr" / "lib" if (managed / "usr" / "lib").is_dir() else None,
            source="managed",
        )

    return None


# ============================================================================
# Multiple sysroots (e.g., Android NDK)
# ============================================================================


def list_available_sysroots(toolchain) -> List[SysrootInfo]:
    """
    List all sysroots available for a toolchain.

    Useful for toolchains like Android NDK that provide one
    sysroot per API level or architecture.

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain to examine.

    Returns
    -------
    List[SysrootInfo]
        All sysroots found. May be empty.
    """
    sysroots = []

    # Start with the default
    default = detect_sysroot(toolchain)
    if default:
        sysroots.append(default)

    # Look for platform-specific sysroots (Android NDK)
    prefix = _get_prefix_path(toolchain)
    if prefix:
        platforms_dir = prefix / "platforms"
        if platforms_dir.is_dir():
            try:
                for api_dir in sorted(platforms_dir.iterdir()):
                    if not api_dir.is_dir():
                        continue
                    for arch_dir in api_dir.iterdir():
                        if arch_dir.is_dir() and _looks_like_sysroot(arch_dir):
                            sysroots.append(SysrootInfo(
                                path=arch_dir,
                                include_dir=arch_dir / "usr" / "include" if (arch_dir / "usr" / "include").is_dir() else None,
                                lib_dir=arch_dir / "usr" / "lib" if (arch_dir / "usr" / "lib").is_dir() else None,
                                source="bundled",
                            ))
            except PermissionError:
                pass

    return sysroots


# ============================================================================
# Helpers
# ============================================================================


def _looks_like_sysroot(path: Path) -> bool:
    """
    Check if a directory has the structure of a sysroot.

    A valid sysroot must have at least one of:
        - usr/include directory
        - usr/lib directory
        - lib directory with .so or .a files

    Parameters
    ----------
    path : Path
        Directory to examine.

    Returns
    -------
    bool
        True if the directory looks like a sysroot.
    """
    if not path.is_dir():
        return False

    # Standard Linux sysroot layout
    if (path / "usr" / "include").is_dir():
        return True
    if (path / "usr" / "lib").is_dir():
        return True

    # Minimal sysroot (just lib/)
    lib_dir = path / "lib"
    if lib_dir.is_dir():
        try:
            for entry in lib_dir.iterdir():
                if entry.name.endswith(".so") or entry.name.endswith(".a"):
                    return True
                if entry.name == "crt0.o" or entry.name == "crt1.o":
                    return True
        except PermissionError:
            pass

    # Android sysroot layout
    if (path / "usr" / "include").is_dir():
        return True
    if (path / "lib").is_dir() and (path / "include").is_dir():
        return True

    return False


def _get_compiler_path(toolchain) -> Optional[Path]:
    """
    Extract the C compiler path from a Toolchain or Path.

    Parameters
    ----------
    toolchain : Toolchain or Path
        Toolchain object or direct compiler path.

    Returns
    -------
    Optional[Path]
        Path to the compiler executable.
    """
    if isinstance(toolchain, Path):
        return toolchain
    if isinstance(toolchain, Toolchain):
        return toolchain.c_compiler
    return None


def _get_prefix_path(toolchain) -> Optional[Path]:
    """
    Extract the installation prefix from a Toolchain or Path.

    Parameters
    ----------
    toolchain : Toolchain or Path
        Toolchain object or path.

    Returns
    -------
    Optional[Path]
        The prefix directory.
    """
    if isinstance(toolchain, Path):
        return toolchain
    if isinstance(toolchain, Toolchain):
        return toolchain.path
    return None
"""
Automatic toolchain detection.

Scans the host system for installed compiler toolchains and returns
fully-initialized Toolchain objects. Supports GCC, Clang, MSVC,
Zig, Emscripten, and Android NDK.

Design
------
Detection is a two-phase process:
    1. Path discovery — find directories that look like toolchain
       installations using platform-specific search paths.
    2. Toolchain construction — wrap each discovered path in the
       appropriate Toolchain subclass and validate it.

The module provides both targeted detection (find a specific kind)
and broad detection (find everything).

Search Paths by Platform
------------------------
    Linux:
        - /usr/bin (system compilers)
        - /usr/local/bin (manually installed)
        - /opt/*/bin (vendor toolchains)
        - $HOME/.local/bin (user installs)
        - TOOLFORGE_HOME or ~/.local/share/toolforge/toolchains/

    macOS:
        - /usr/bin (Xcode CLT; gcc is symlink to clang)
        - /Applications/Xcode.app/... (Xcode)
        - /opt/homebrew/bin (Homebrew ARM)
        - /usr/local/bin (Homebrew Intel + manual)
        - TOOLFORGE_HOME or ~/.local/share/toolforge/toolchains/

    Windows:
        - C:/Program Files/Microsoft Visual Studio/ (VS 2017+)
        - C:/Program Files (x86)/Microsoft Visual Studio/ (VS)
        - C:/msys64/mingw64/bin (MSYS2 MinGW)
        - C:/msys64/ucrt64/bin (MSYS2 UCRT)
        - C:/Program Files/LLVM/bin (official LLVM)
        - %LOCALAPPDATA%/toolforge/toolchains/

Environment Variables
---------------------
    CC, CXX : If set, these paths are checked first and given
              priority over auto-detected compilers.
    TOOLFORGE_HOME : Overrides the default toolforge install root.
    ANDROID_NDK_HOME : Path to Android NDK root.

Usage
-----
    from from pyputil_install.compiler_installer.toolchains.detection import (
        detect_all_toolchains,
        detect_gcc,
        detect_clang,
        detect_msvc,
    )

    # Find everything
    all_toolchains = detect_all_toolchains()
    for tc in all_toolchains:
        print(tc.kind.name, tc.version, tc.path)

    # Find only GCC
    gcc_list = detect_gcc()
    for gcc in gcc_list:
        print(gcc.version, gcc.target_triplet)

    # Find and use the best available
    best = detect_best_toolchain()
    if best:
        result = best.compile("source.c", output="prog")

Warnings
--------
- Detection may spawn subprocesses (for version/target queries).
  Each toolchain found triggers at least one subprocess call.
- On macOS, /usr/bin/gcc is typically a symlink to clang.
  GCCToolchain will detect this as GCC but the underlying
  compiler is Clang. Check .target_triplet to distinguish.
- MSVC detection is Windows-only and returns empty list on
  other platforms.
- The search can take several seconds on systems with many
  directories in PATH or large /opt trees.

User Instructions
-----------------
- Use detect_all_toolchains() for a complete system scan.
- Use detect_best_toolchain() when you just need one working compiler.
- Results are NOT cached between calls. Each call re-scans.
- Set CC/CXX to force specific compilers ahead of auto-detection.
"""

import logging
import os
import platform as _platform
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

from .base import Toolchain, ToolchainKind
from .gcc import GCCToolchain
from .clang import ClangToolchain
from .msvc import MSVCToolchain
from .zig import ZigToolchain
from .emscripten import EmscriptenToolchain
from .android import AndroidNDKToolchain

logger = logging.getLogger(__name__)

# ============================================================================
# Mapping from ToolchainKind to implementation class
# ============================================================================

_KIND_TO_CLASS: Dict[ToolchainKind, Type[Toolchain]] = {
    ToolchainKind.GCC: GCCToolchain,
    ToolchainKind.CLANG: ClangToolchain,
    ToolchainKind.MSVC: MSVCToolchain,
    ToolchainKind.ZIG: ZigToolchain,
    ToolchainKind.EMSCRIPTEN: EmscriptenToolchain,
    ToolchainKind.ANDROID_NDK: AndroidNDKToolchain,
}


# ============================================================================
# Internal: path discovery helpers
# ============================================================================

def _get_install_root() -> Path:
    """
    Return the ToolForge toolchain installation root.

    Uses TOOLFORGE_HOME env var if set, otherwise the platform
    default (~/.local/share/toolforge/toolchains on Unix).

    Returns
    -------
    Path
        Absolute path to the toolchains directory.
    """
    env_home = os.environ.get("TOOLFORGE_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()

    if os.name == "nt":
        local_app = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(local_app) / "toolforge" / "toolchains"

    xdg = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    return Path(xdg) / "toolforge" / "toolchains"


def _executable_exists(path: Path, name: str) -> Optional[Path]:
    """
    Check if a named executable exists in a directory.

    Parameters
    ----------
    path : Path
        Directory to check.
    name : str
        Executable name without extension.

    Returns
    -------
    Optional[Path]
        Full path if found and executable, None otherwise.
    """
    exe = path / name
    if exe.is_file() and os.access(str(exe), os.X_OK):
        return exe
    if os.name == "nt":
        exe = path / f"{name}.exe"
        if exe.is_file() and os.access(str(exe), os.X_OK):
            return exe
    return None


def _find_in_path(name: str) -> Optional[Path]:
    """
    Find an executable in the system PATH.

    Parameters
    ----------
    name : str
        Executable name.

    Returns
    -------
    Optional[Path]
        Full path if found, None otherwise.
    """
    result = shutil.which(name)
    return Path(result) if result else None


def _parent_of(path: Path, levels: int = 1) -> Path:
    """
    Return the parent directory N levels up.

    Parameters
    ----------
    path : Path
        Starting path.
    levels : int
        Number of levels to go up.

    Returns
    -------
    Path
        Parent directory.
    """
    result = path
    for _ in range(levels):
        result = result.parent
    return result


# ============================================================================
# Detection: GCC
# ============================================================================

def detect_gcc() -> List[GCCToolchain]:
    """
    Find all installed GCC toolchains on the system.

    Search order:
        1. CC environment variable (if ends with gcc or cc).
        2. `gcc` in PATH.
        3. Common installation directories:
           - /usr, /usr/local, /opt/* (Linux)
           - /opt/homebrew/opt/gcc (macOS Homebrew)
           - /usr/local/opt/gcc (macOS Intel Homebrew)
           - C:/msys64/mingw64, C:/msys64/ucrt64 (Windows MSYS2)
           - C:/mingw-w64 (Windows standalone)
        4. ToolForge managed installations.

    Returns
    -------
    List[GCCToolchain]
        Valid GCC toolchains found. May be empty.
        Each toolchain has been validated (version and target detected).
    """
    found: Dict[str, GCCToolchain] = {}

    # 1. CC environment variable
    cc_env = os.environ.get("CC", "")
    if cc_env and ("gcc" in cc_env or cc_env.endswith("cc")):
        gcc_path = Path(cc_env)
        if gcc_path.is_file():
            prefix = _parent_of(gcc_path.parent, 1)
            tc = GCCToolchain(prefix)
            if tc.is_valid() and str(tc.path) not in found:
                found[str(tc.path)] = tc

    # 2. gcc in PATH
    gcc_in_path = _find_in_path("gcc")
    if gcc_in_path:
        prefix = _parent_of(gcc_in_path.parent, 1)
        tc = GCCToolchain(prefix)
        if tc.is_valid() and str(tc.path) not in found:
            found[str(tc.path)] = tc

    # 3. Common installation directories
    search_roots = _get_gcc_search_roots()
    for root in search_roots:
        if not root.is_dir():
            continue
        # root/bin/gcc ?
        if _executable_exists(root / "bin", "gcc"):
            tc = GCCToolchain(root)
            if tc.is_valid() and str(tc.path) not in found:
                found[str(tc.path)] = tc
        # root itself is a prefix with bin/gcc?
        if _executable_exists(root, "gcc") and root.name == "bin":
            prefix = _parent_of(root, 1)
            tc = GCCToolchain(prefix)
            if tc.is_valid() and str(tc.path) not in found:
                found[str(tc.path)] = tc

    # 4. ToolForge managed installations
    install_root = _get_install_root()
    gcc_dir = install_root / "gcc"
    if gcc_dir.is_dir():
        for version_dir in gcc_dir.iterdir():
            if version_dir.is_dir():
                tc = GCCToolchain(version_dir)
                if tc.is_valid() and str(tc.path) not in found:
                    found[str(tc.path)] = tc

    return list(found.values())


def _get_gcc_search_roots() -> List[Path]:
    """
    Return platform-specific directories to search for GCC.

    Returns
    -------
    List[Path]
        Directory paths that may contain GCC installations.
    """
    system = _platform.system()
    roots = []

    if system == "Linux":
        roots.extend([
            Path("/usr"),
            Path("/usr/local"),
        ])
        # /opt subdirectories
        opt = Path("/opt")
        if opt.is_dir():
            try:
                for entry in opt.iterdir():
                    if entry.is_dir():
                        roots.append(entry)
            except PermissionError:
                pass

    elif system == "Darwin":
        roots.extend([
            Path("/usr"),
            Path("/usr/local"),
            Path("/opt/homebrew/opt/gcc"),
            Path("/usr/local/opt/gcc"),
            Path("/opt/homebrew"),
            Path("/usr/local"),
        ])

    elif system == "Windows":
        roots.extend([
            Path("C:/msys64/mingw64"),
            Path("C:/msys64/ucrt64"),
            Path("C:/msys64/clang64"),
            Path("C:/mingw-w64"),
        ])

    return roots


# ============================================================================
# Detection: Clang
# ============================================================================

def detect_clang() -> List[ClangToolchain]:
    """
    Find all installed Clang/LLVM toolchains on the system.

    Search order:
        1. CC/CXX environment variables (if clang).
        2. `clang` in PATH.
        3. Common installation directories:
           - /usr (system Clang on Linux)
           - /opt/homebrew/opt/llvm (macOS Homebrew)
           - /usr/local/opt/llvm (macOS Intel Homebrew)
           - Xcode toolchain path (macOS)
           - C:/Program Files/LLVM (Windows official)
        4. ToolForge managed installations.

    Returns
    -------
    List[ClangToolchain]
        Valid Clang toolchains found. May be empty.
    """
    found: Dict[str, ClangToolchain] = {}

    # 1. Environment variables
    for env_var in ("CC", "CXX"):
        env_val = os.environ.get(env_var, "")
        if "clang" in env_val.lower():
            clang_path = Path(env_val)
            if clang_path.is_file():
                prefix = _parent_of(clang_path.parent, 1)
                tc = ClangToolchain(prefix)
                if tc.is_valid() and str(tc.path) not in found:
                    found[str(tc.path)] = tc

    # 2. clang in PATH
    clang_in_path = _find_in_path("clang")
    if clang_in_path:
        prefix = _parent_of(clang_in_path.parent, 1)
        tc = ClangToolchain(prefix)
        if tc.is_valid() and str(tc.path) not in found:
            found[str(tc.path)] = tc

    # 3. Common directories
    search_roots = _get_clang_search_roots()
    for root in search_roots:
        if not root.is_dir():
            continue
        if _executable_exists(root / "bin", "clang"):
            tc = ClangToolchain(root)
            if tc.is_valid() and str(tc.path) not in found:
                found[str(tc.path)] = tc

    # 4. ToolForge managed
    install_root = _get_install_root()
    clang_dir = install_root / "clang"
    if clang_dir.is_dir():
        for version_dir in clang_dir.iterdir():
            if version_dir.is_dir():
                tc = ClangToolchain(version_dir)
                if tc.is_valid() and str(tc.path) not in found:
                    found[str(tc.path)] = tc

    return list(found.values())


def _get_clang_search_roots() -> List[Path]:
    """Return platform-specific directories to search for Clang."""
    system = _platform.system()
    roots = []

    if system == "Linux":
        roots.append(Path("/usr"))
        roots.append(Path("/usr/local"))

    elif system == "Darwin":
        roots.extend([
            Path("/usr"),
            Path("/usr/local"),
            Path("/opt/homebrew/opt/llvm"),
            Path("/usr/local/opt/llvm"),
            Path("/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr"),
        ])

    elif system == "Windows":
        roots.append(Path("C:/Program Files/LLVM"))

    return roots


# ============================================================================
# Detection: MSVC (Windows only)
# ============================================================================

def detect_msvc() -> List[MSVCToolchain]:
    """
    Find installed Microsoft Visual C++ toolchains.

    Windows only. Returns an empty list on other platforms.
    Searches Visual Studio 2022, 2019, and 2017 installation
    directories for BuildTools, Community, Professional, and
    Enterprise editions.

    Returns
    -------
    List[MSVCToolchain]
        Valid MSVC toolchains found. May be empty.
    """
    if _platform.system() != "Windows":
        return []

    found: Dict[str, MSVCToolchain] = {}

    vs_roots = _get_msvc_search_roots()
    for vs_base in vs_roots:
        if not vs_base.is_dir():
            continue

        # Look for VC/Tools/MSVC/* directories
        vc_tools = vs_base / "VC" / "Tools" / "MSVC"
        if not vc_tools.is_dir():
            continue

        try:
            for version_dir in vc_tools.iterdir():
                if not version_dir.is_dir():
                    continue
                tc = MSVCToolchain(version_dir)
                if tc.is_valid() and str(tc.path) not in found:
                    found[str(tc.path)] = tc
        except PermissionError:
            continue

    return list(found.values())


def _get_msvc_search_roots() -> List[Path]:
    """Return Visual Studio base directories to search for MSVC."""
    roots = []
    program_files = os.environ.get("ProgramFiles", "C:/Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")

    for base in (Path(program_files), Path(program_files_x86)):
        for vs_year in ("2022", "2019", "2017"):
            vs_dir = base / "Microsoft Visual Studio" / vs_year
            if vs_dir.is_dir():
                for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
                    edition_dir = vs_dir / edition
                    if edition_dir.is_dir():
                        roots.append(edition_dir)

    return roots


# ============================================================================
# Detection: Zig
# ============================================================================

def detect_zig() -> List[ZigToolchain]:
    """
    Find installed Zig toolchains.

    Search order:
        1. `zig` in PATH.
        2. Common directories: /usr/local/zig, ~/zig, C:/zig.
        3. ToolForge managed installations.

    Returns
    -------
    List[ZigToolchain]
        Valid Zig toolchains found.
    """
    found: Dict[str, ZigToolchain] = {}

    # 1. PATH
    zig_path = _find_in_path("zig")
    if zig_path:
        tc = ZigToolchain(zig_path.parent)
        if tc.is_valid() and str(tc.path) not in found:
            found[str(tc.path)] = tc

    # 2. Common directories
    search_roots = [
        Path("/usr/local/zig"),
        Path.home() / "zig",
        Path("C:/zig"),
    ]
    for root in search_roots:
        if root.is_dir():
            tc = ZigToolchain(root)
            if tc.is_valid() and str(tc.path) not in found:
                found[str(tc.path)] = tc

    # 3. ToolForge managed
    install_root = _get_install_root()
    zig_dir = install_root / "zig"
    if zig_dir.is_dir():
        for version_dir in zig_dir.iterdir():
            if version_dir.is_dir():
                tc = ZigToolchain(version_dir)
                if tc.is_valid() and str(tc.path) not in found:
                    found[str(tc.path)] = tc

    return list(found.values())


# ============================================================================
# Detection: Emscripten
# ============================================================================

def detect_emscripten() -> List[EmscriptenToolchain]:
    """
    Find installed Emscripten SDKs.

    Search order:
        1. `emcc` in PATH.
        2. Common directories: ~/emsdk, /usr/local/emsdk.
        3. EMSDK environment variable.
        4. ToolForge managed installations.

    Returns
    -------
    List[EmscriptenToolchain]
        Valid Emscripten toolchains found.
    """
    found: Dict[str, EmscriptenToolchain] = {}

    # 1. PATH
    emcc_path = _find_in_path("emcc")
    if emcc_path:
        # emcc is typically in emsdk/upstream/emscripten/
        root = _parent_of(emcc_path.parent, 2)  # upstream/emscripten -> emsdk
        tc = EmscriptenToolchain(root)
        if tc.is_valid() and str(tc.path) not in found:
            found[str(tc.path)] = tc

    # 2. Common directories
    search_roots = [
        Path.home() / "emsdk",
        Path("/usr/local/emsdk"),
    ]
    for root in search_roots:
        if root.is_dir():
            tc = EmscriptenToolchain(root)
            if tc.is_valid() and str(tc.path) not in found:
                found[str(tc.path)] = tc

    # 3. EMSDK env var
    emsdk_env = os.environ.get("EMSDK")
    if emsdk_env:
        emsdk_path = Path(emsdk_env)
        if emsdk_path.is_dir():
            tc = EmscriptenToolchain(emsdk_path)
            if tc.is_valid() and str(tc.path) not in found:
                found[str(tc.path)] = tc

    # 4. ToolForge managed
    install_root = _get_install_root()
    emsdk_dir = install_root / "emscripten"
    if emsdk_dir.is_dir():
        for version_dir in emsdk_dir.iterdir():
            if version_dir.is_dir():
                tc = EmscriptenToolchain(version_dir)
                if tc.is_valid() and str(tc.path) not in found:
                    found[str(tc.path)] = tc

    return list(found.values())


# ============================================================================
# Detection: Android NDK
# ============================================================================

def detect_android_ndk() -> List[AndroidNDKToolchain]:
    """
    Find installed Android NDK installations.

    Search order:
        1. ANDROID_NDK_HOME environment variable.
        2. ANDROID_HOME/ndk/* (Android SDK NDK directory).
        3. Common directories: ~/Android/Sdk/ndk/*.
        4. /usr/local/android-ndk.

    Returns
    -------
    List[AndroidNDKToolchain]
        Valid Android NDK toolchains found.
    """
    found: Dict[str, AndroidNDKToolchain] = {}

    ndk_roots = _get_ndk_search_roots()
    for ndk_root in ndk_roots:
        if not ndk_root.is_dir():
            continue
        tc = AndroidNDKToolchain(ndk_root)
        if tc.is_valid() and str(tc.path) not in found:
            found[str(tc.path)] = tc

    return list(found.values())


def _get_ndk_search_roots() -> List[Path]:
    """Return directories to search for Android NDK."""
    roots = []

    # ANDROID_NDK_HOME
    ndk_home = os.environ.get("ANDROID_NDK_HOME")
    if ndk_home:
        roots.append(Path(ndk_home))

    # ANDROID_HOME/ndk/*
    android_home = os.environ.get("ANDROID_HOME")
    if android_home:
        ndk_dir = Path(android_home) / "ndk"
        if ndk_dir.is_dir():
            try:
                for entry in sorted(ndk_dir.iterdir(), reverse=True):
                    if entry.is_dir():
                        roots.append(entry)
            except PermissionError:
                pass

    # Common paths
    if _platform.system() == "Windows":
        roots.append(Path.home() / "AppData" / "Local" / "Android" / "Sdk" / "ndk")
    else:
        roots.append(Path.home() / "Android" / "Sdk" / "ndk")
        roots.append(Path("/usr/local/android-ndk"))

    # Expand any ndk/* directories
    expanded = []
    for root in roots:
        if root.is_dir() and any(
            f.is_dir() and (f / "toolchains" / "llvm").is_dir()
            for f in root.iterdir() if f.is_dir()
        ):
            try:
                for entry in root.iterdir():
                    if entry.is_dir():
                        expanded.append(entry)
            except PermissionError:
                pass
        else:
            expanded.append(root)

    return expanded


# ============================================================================
# Detection: all
# ============================================================================

def detect_all_toolchains() -> List[Toolchain]:
    """
    Find all installed compiler toolchains on the system.

    Runs all individual detection functions and returns a combined
    list. Detection order: GCC, Clang, MSVC, Zig, Emscripten, NDK.

    Returns
    -------
    List[Toolchain]
        All valid toolchains found. May be empty if no compilers
        are installed.

    Example
    -------
    >>> all_tcs = detect_all_toolchains()
    >>> for tc in all_tcs:
    ...     print(f"{tc.kind.name:15} {tc.version:10} {tc.path}")
    """
    detectors = [
        detect_gcc,
        detect_clang,
        detect_msvc,
        detect_zig,
        detect_emscripten,
        detect_android_ndk,
    ]

    all_found: List[Toolchain] = []
    for detector in detectors:
        try:
            found = detector()
            all_found.extend(found)
            logger.debug("%s found %d toolchain(s)", detector.__name__, len(found))
        except Exception as exc:
            logger.warning("%s failed: %s", detector.__name__, exc)

    return all_found


def detect_best_toolchain(
    prefer_native: bool = True,
) -> Optional[Toolchain]:
    """
    Find the best available toolchain for general use.

    Selects the highest-priority toolchain from all detected.
    Priority order: GCC > Clang > MSVC > Zig > Emscripten.
    Within each kind, prefers the highest version number.

    Parameters
    ----------
    prefer_native : bool
        If True (default), prefers native compilers over cross-compilers.
        If False, cross-compilers are treated equally.

    Returns
    -------
    Optional[Toolchain]
        The best toolchain, or None if no compilers are installed.

    Example
    -------
    >>> best = detect_best_toolchain()
    >>> if best:
    ...     result = best.compile("source.c", output="prog")
    """
    all_tcs = detect_all_toolchains()
    if not all_tcs:
        return None

    # Priority ordering
    kind_priority = {
        ToolchainKind.GCC: 0,
        ToolchainKind.CLANG: 1,
        ToolchainKind.MSVC: 2,
        ToolchainKind.ZIG: 3,
        ToolchainKind.EMSCRIPTEN: 4,
        ToolchainKind.ANDROID_NDK: 5,
        ToolchainKind.UNKNOWN: 99,
    }

    def sort_key(tc: Toolchain) -> tuple:
        """Sort by: native preferred, kind priority, version descending."""
        is_cross = 1
        if hasattr(tc, 'is_cross_compiler'):
            is_cross = 1 if tc.is_cross_compiler else 0
        elif hasattr(tc, 'is_native'):
            is_cross = 0 if tc.is_native else 1

        kind_prio = kind_priority.get(tc.kind, 99)

        # Parse version for numeric comparison
        version_tuple = (0, 0, 0)
        try:
            parts = tc.version.split(".")
            version_tuple = tuple(int(p) for p in parts[:3])
        except (ValueError, AttributeError):
            pass

        return (
            is_cross if prefer_native else 0,
            kind_prio,
            -version_tuple[0],
            -version_tuple[1] if len(version_tuple) > 1 else 0,
            -version_tuple[2] if len(version_tuple) > 2 else 0,
        )

    all_tcs.sort(key=sort_key)
    return all_tcs[0]
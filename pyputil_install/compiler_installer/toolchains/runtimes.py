"""
Runtime library detection for compiler toolchains.

Detects which runtime libraries are available for a given toolchain,
including C standard library, C++ standard library, and compiler
support libraries (libgcc, libclang_rt, compiler-rt).

Design
------
Runtime detection queries the compiler for its library search paths
(-print-search-dirs), then scans those directories for known
runtime library files. Results are returned as a RuntimeInfo object
listing what was found.

Each toolchain family uses different runtime libraries:
    - GCC: libgcc, libstdc++, libgcc_s, libgomp
    - Clang: libclang_rt (compiler-rt), libc++, libc++abi
    - MSVC: msvcrt, libcmt, msvcp, vcruntime
    - Zig: bundles libc, libc++, libunwind, compiler-rt
    - Emscripten: libc, libc++, libc++abi (compiled to Wasm)
    - Android NDK: libc (bionic), libc++, libstdc++

Usage
-----
    from pathlib import Path
    from pyputil_install.compiler_installer.toolchains.runtimes import detect_runtimes, RuntimeInfo
    from pyputil_install.compiler_installer.toolchains.gcc import GCCToolchain

    gcc = GCCToolchain(Path("/usr"))
    runtimes = detect_runtimes(gcc)
    print(runtimes.c_library)          # "glibc" or "unknown"
    print(runtimes.cxx_library)        # "libstdc++" or "libc++"
    print(runtimes.compiler_rt)        # Path to libgcc.a or libclang_rt.a
    print(runtimes.search_paths)       # Library search paths from compiler

Warnings
--------
- Detection runs the compiler with -print-search-dirs, which
  may be slow on network-mounted filesystems.
- Runtime files found on disk may not be the ones actually
  used at link time. The linker may select different files
  based on -m32/-m64, -static, or other flags.
- Some toolchains (Zig, Emscripten) bundle their runtimes
  internally and do not expose them via search paths.

User Instructions
-----------------
- Use detect_runtimes() to get a complete picture of available
  runtime libraries.
- Check RuntimeInfo.c_library to determine which C library is
  being targeted (glibc, musl, bionic, etc.).
- The search_paths list can be passed to -L flags if needed.
"""

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from .base import Toolchain

logger = logging.getLogger(__name__)

# ============================================================================
# RuntimeInfo
# ============================================================================


@dataclass
class RuntimeInfo:
    """
    Information about runtime libraries available for a toolchain.

    Attributes
    ----------
    c_library : str
        Detected C standard library. Values:
        "glibc", "musl", "bionic" (Android), "msvcrt" (Windows),
        "newlib", "picolibc", "wasm-libc" (Emscripten),
        "unknown" if detection failed.
    cxx_library : str
        Detected C++ standard library. Values:
        "libstdc++", "libc++", "msvcp" (MSVC), "libc++abi",
        "none" if no C++ library found.
    compiler_rt : Optional[Path]
        Path to the compiler runtime library (libgcc.a, libclang_rt.a,
        or equivalent). None if not found.
    search_paths : List[Path]
        Library search paths reported by the compiler. These are
        directories the linker will search for libraries.
    available_libs : Set[str]
        Set of library base names found in the search paths.
        Example: {"c", "m", "pthread", "stdc++", "gcc_s"}.
    is_static_only : bool
        True if only static runtime libraries were found (.a but
        no .so). Common for embedded and cross-compilation targets.
    """

    c_library: str = "unknown"
    cxx_library: str = "none"
    compiler_rt: Optional[Path] = None
    search_paths: List[Path] = field(default_factory=list)
    available_libs: Set[str] = field(default_factory=set)
    is_static_only: bool = False


# ============================================================================
# Detection
# ============================================================================


def detect_runtimes(
    toolchain,
    extra_flags: Optional[List[str]] = None,
) -> RuntimeInfo:
    """
    Detect runtime libraries available for a toolchain.

    Runs the compiler with -print-search-dirs to find library
    search paths, then scans those paths for known runtime files.

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain to query, or a direct compiler path.
    extra_flags : Optional[List[str]]
        Additional flags to pass to the compiler (e.g., -m32, -target).

    Returns
    -------
    RuntimeInfo
        Detected runtime information.

    Example
    -------
    >>> runtimes = detect_runtimes(my_gcc)
    >>> print(runtimes.c_library)
    'glibc'
    >>> print(runtimes.cxx_library)
    'libstdc++'
    >>> for lib in sorted(runtimes.available_libs):
    ...     print(lib)
    c
    gcc_s
    m
    pthread
    stdc++
    """
    info = RuntimeInfo()
    compiler = _get_compiler_path(toolchain)
    if compiler is None:
        return info

    # Get library search paths from the compiler
    info.search_paths = _get_search_paths(compiler, extra_flags)
    if not info.search_paths:
        logger.debug("No library search paths found for %s", compiler)
        return info

    # Scan for libraries
    info.available_libs = _scan_libraries(info.search_paths)

    # Detect C library
    info.c_library = _detect_c_library(info.available_libs, info.search_paths)

    # Detect C++ library
    info.cxx_library = _detect_cxx_library(info.available_libs)

    # Find compiler runtime
    info.compiler_rt = _find_compiler_rt(info.search_paths, info.available_libs)

    # Check if static only
    info.is_static_only = _check_static_only(info.search_paths, info.available_libs)

    return info


def _get_search_paths(compiler: Path, extra_flags: Optional[List[str]] = None) -> List[Path]:
    """
    Get library search paths from the compiler.

    Runs: compiler -print-search-dirs
    Parses the "libraries:" line for colon-separated paths.

    Parameters
    ----------
    compiler : Path
        Path to the compiler.
    extra_flags : Optional[List[str]]
        Extra flags to pass.

    Returns
    -------
    List[Path]
        Library search paths that exist on disk.
    """
    cmd = [str(compiler), "-print-search-dirs"]
    if extra_flags:
        cmd.extend(extra_flags)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            shell=False,
        )
        output = result.stdout or result.stderr
    except Exception as exc:
        logger.debug("Failed to get search dirs: %s", exc)
        return []

    paths = []
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("libraries:"):
            libs_part = line.split(":", 1)[1].strip()
            # Remove leading '=' if present
            if libs_part.startswith("="):
                libs_part = libs_part[1:]
            for raw_path in libs_part.split(":"):
                raw_path = raw_path.strip()
                if raw_path:
                    p = Path(raw_path)
                    if p.is_dir() and p not in paths:
                        paths.append(p)
            break

    return paths


def _scan_libraries(search_paths: List[Path]) -> Set[str]:
    """
    Scan search paths for library files.

    Looks for .a, .so, .dylib, .dll, and .lib files and extracts
    the library base name (e.g., "libc.so" -> "c").

    Parameters
    ----------
    search_paths : List[Path]
        Directories to scan.

    Returns
    -------
    Set[str]
        Set of library base names found.
    """
    libs = set()
    extensions = (".a", ".so", ".dylib", ".dll", ".lib")

    for directory in search_paths:
        try:
            for entry in directory.iterdir():
                if not entry.is_file():
                    continue
                name = entry.name
                if not any(name.endswith(ext) for ext in extensions):
                    continue

                # Extract library name: lib{name}.{ext}
                base = _library_base_name(name)
                if base:
                    libs.add(base)
        except PermissionError:
            continue

    return libs


def _library_base_name(filename: str) -> Optional[str]:
    """
    Extract the library base name from a filename.

    Examples:
        "libc.so.6" -> "c"
        "libstdc++.a" -> "stdc++"
        "libgcc_s.so" -> "gcc_s"
        "crt0.o" -> None (not a library)
        "vcruntime.lib" -> "vcruntime" (Windows .lib without lib prefix)

    Parameters
    ----------
    filename : str
        Library filename.

    Returns
    -------
    Optional[str]
        Base name, or None if not a recognized library.
    """
    # Strip version suffix: libc.so.6 -> libc.so
    name = filename
    while True:
        stem, dot, rest = name.partition(".")
        if rest and rest[0].isdigit():
            name = stem + dot + rest.split(".", 1)[-1] if "." in rest else stem
            continue
        break

    # Strip known extensions
    for ext in (".a", ".so", ".dylib", ".dll", ".lib"):
        if name.endswith(ext):
            name = name[:-len(ext)]
            break

    # Strip "lib" prefix
    if name.startswith("lib"):
        name = name[3:]

    # Skip non-library files
    if name in ("", "crt0", "crt1", "crti", "crtn", "crtbegin", "crtend"):
        return None
    if name.startswith("crt"):
        return None

    return name


def _detect_c_library(available_libs: Set[str], search_paths: List[Path]) -> str:
    """
    Detect which C standard library is available.

    Checks for known marker files:
        glibc: libc.so with versioned symbols (glibc 2.x)
        musl: libc.so that reports musl
        bionic: libc.so in Android paths
        msvcrt: msvcrt.lib or libcmt.lib
        newlib: libc.a alongside newlib headers
        wasm-libc: emscripten paths

    Parameters
    ----------
    available_libs : Set[str]
        Library base names found.
    search_paths : List[Path]
        Library search paths.

    Returns
    -------
    str
        C library identifier.
    """
    # MSVC C runtimes
    if "msvcrt" in available_libs or "libcmt" in available_libs or "vcruntime" in available_libs:
        return "msvcrt"

    # Check paths for known markers
    for directory in search_paths:
        dir_str = str(directory).lower()

        if "android" in dir_str or "bionic" in dir_str:
            if "c" in available_libs:
                return "bionic"

        if "emscripten" in dir_str or "emsdk" in dir_str:
            return "wasm-libc"

    # Check for libc
    if "c" not in available_libs:
        return "unknown"

    # Look for glibc or musl marker
    for directory in search_paths:
        # glibc typically has libc.so.6
        libc_so = directory / "libc.so.6"
        if libc_so.is_file():
            return "glibc"

        # musl has libc.so pointing to ld-musl
        libc_so = directory / "libc.so"
        if libc_so.is_file():
            try:
                target = os.readlink(str(libc_so))
                if "musl" in target:
                    return "musl"
            except OSError:
                pass

    return "glibc"  # Most common on Linux


def _detect_cxx_library(available_libs: Set[str]) -> str:
    """
    Detect which C++ standard library is available.

    Checks for known library names in the scanned set.

    Parameters
    ----------
    available_libs : Set[str]
        Library base names found.

    Returns
    -------
    str
        C++ library identifier.
    """
    if "stdc++" in available_libs:
        return "libstdc++"
    if "c++" in available_libs:
        return "libc++"
    if "c++abi" in available_libs:
        return "libc++abi"
    if "msvcp" in available_libs or "msvcprt" in available_libs:
        return "msvcp"
    return "none"


def _find_compiler_rt( search_paths: List[Path], available_libs: Set[str]) -> Optional[Path]:
    """
    Find the compiler runtime library.

    Looks for:
        libgcc.a (GCC)
        libclang_rt.builtins.a (Clang/LLVM)
        libgcc_s.so (GCC shared)
        compiler-rt (various)

    Parameters
    ----------
    search_paths : List[Path]
        Directories to search.
    available_libs : Set[str]
        Library base names found.

    Returns
    -------
    Optional[Path]
        Path to the runtime library, or None.
    """
    candidates = [
        "libclang_rt.builtins.a",
        "libclang_rt.builtins.so",
        "libgcc.a",
        "libgcc_s.so",
    ]

    for directory in search_paths:
        for candidate in candidates:
            full = directory / candidate
            if full.is_file():
                return full

    return None


def _check_static_only(search_paths: List[Path], available_libs: Set[str]) -> bool:
    """
    Check if only static libraries are available.

    Scans search paths for any .so or .dylib file. If none
    are found, the toolchain is static-only.

    Parameters
    ----------
    search_paths : List[Path]
        Library search paths.
    available_libs : Set[str]
        Library names found.

    Returns
    -------
    bool
        True if no shared libraries were found.
    """
    for directory in search_paths:
        try:
            for entry in directory.iterdir():
                if entry.is_file() and (entry.name.endswith(".so") or entry.name.endswith(".dylib")):
                    return False
        except PermissionError:
            continue
    return True


# ============================================================================
# Helpers
# ============================================================================


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
        Path to the compiler.
    """
    if isinstance(toolchain, Path):
        return toolchain
    if isinstance(toolchain, Toolchain):
        return toolchain.c_compiler
    return None
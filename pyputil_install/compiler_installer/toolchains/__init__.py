"""
Toolchain abstraction layer — runtime intelligence for installed compilers.

Provides classes and utilities for working with installed compiler
toolchains as objects with known properties, capabilities, and
behaviors. This layer sits above raw compiler discovery and below
build system integration.

Modules
-------
base.py         : Toolchain abstract base class, ToolRole, CompileResult
gcc.py          : GCC implementation
clang.py        : Clang/LLVM implementation
msvc.py         : MSVC implementation (Windows only)
zig.py          : Zig implementation
emscripten.py   : Emscripten SDK implementation
android.py      : Android NDK implementation
capabilities.py : Feature detection (C++20, OpenMP, sanitizers, etc.)
detection.py    : Automatic toolchain discovery on the host system
sysroots.py     : Sysroot detection for cross-compilation
runtimes.py     : Runtime library detection (libc, libstdc++, etc.)
abi.py          : ABI detection (calling convention, type sizes, etc.)
environments.py : Named sets of toolchains for reproducible builds

Usage
-----
    from pyputil_install.compiler_installer.toolchains import (
        detect_all_toolchains,
        detect_best_toolchain,
        detect_gcc,
        detect_clang,
        Toolchain,
        ToolchainKind,
        ToolRole,
    )

    # Find everything installed
    all_tcs = detect_all_toolchains()
    for tc in all_tcs:
        print(tc.kind.name, tc.version)

    # Use the best available
    best = detect_best_toolchain()
    if best:
        result = best.compile("source.c", output="program")

    # Check capabilities
    from .capabilities import detect_all
    caps = detect_all(best)
    print("C++20:", caps["c++20"])
    print("OpenMP:", caps["openmp"])

Warnings
--------
- Detection spawns subprocesses. First call may take seconds.
- MSVC detection is Windows-only.
- Toolchain objects are not thread-safe for activation/deactivation.
- ABI and capability checks compile test programs. Ensure a working
  toolchain is selected before running comprehensive detection.

User Instructions
-----------------
- Start with detect_all_toolchains() or detect_best_toolchain().
- Use Toolchain subclasses directly when you know the exact compiler.
- Capability and ABI detection are separate modules — import them
  explicitly when needed.
- Environments provide reproducible builds across machines.
"""

# ============================================================================
# Base classes and enums
# ============================================================================

from .base import (
    Toolchain,
    ToolchainKind,
    ToolRole,
    CompileResult,
)

# ============================================================================
# Toolchain implementations
# ============================================================================

from .gcc import GCCToolchain
from .clang import ClangToolchain
from .msvc import MSVCToolchain
from .zig import ZigToolchain
from .emscripten import EmscriptenToolchain
from .android import AndroidNDKToolchain

# ============================================================================
# Detection
# ============================================================================

from .detection import (
    detect_gcc,
    detect_clang,
    detect_msvc,
    detect_zig,
    detect_emscripten,
    detect_android_ndk,
    detect_all_toolchains,
    detect_best_toolchain,
)

# ============================================================================
# Capabilities
# ============================================================================

from .capabilities import (
    supports_cpp20,
    supports_cpp17,
    supports_cpp14,
    supports_cpp11,
    supports_c11,
    supports_openmp,
    supports_lto,
    supports_asan,
    supports_ubsan,
    supports_thread_sanitizer,
    supports_stack_protector,
    supports_pic,
    supports_rtti,
    supports_exceptions,
    detect_all,
    clear_detection_cache,
)

# ============================================================================
# Sysroots
# ============================================================================

from .sysroots import (
    SysrootInfo,
    detect_sysroot,
    list_available_sysroots,
)

# ============================================================================
# Runtimes
# ============================================================================

from .runtimes import (
    RuntimeInfo,
    detect_runtimes,
)

# ============================================================================
# ABI
# ============================================================================

from .abi import (
    ABIInfo,
    detect_abi,
    get_abi_compatibility,
)

# ============================================================================
# Environments
# ============================================================================

from .environments import (
    Environment,
    EnvironmentStore,
)

# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Base
    "Toolchain",
    "ToolchainKind",
    "ToolRole",
    "CompileResult",
    # Implementations
    "GCCToolchain",
    "ClangToolchain",
    "MSVCToolchain",
    "ZigToolchain",
    "EmscriptenToolchain",
    "AndroidNDKToolchain",
    # Detection
    "detect_gcc",
    "detect_clang",
    "detect_msvc",
    "detect_zig",
    "detect_emscripten",
    "detect_android_ndk",
    "detect_all_toolchains",
    "detect_best_toolchain",
    # Capabilities
    "supports_cpp20",
    "supports_cpp17",
    "supports_cpp14",
    "supports_cpp11",
    "supports_c11",
    "supports_openmp",
    "supports_lto",
    "supports_asan",
    "supports_ubsan",
    "supports_thread_sanitizer",
    "supports_stack_protector",
    "supports_pic",
    "supports_rtti",
    "supports_exceptions",
    "detect_all",
    "clear_detection_cache",
    # Sysroots
    "SysrootInfo",
    "detect_sysroot",
    "list_available_sysroots",
    # Runtimes
    "RuntimeInfo",
    "detect_runtimes",
    # ABI
    "ABIInfo",
    "detect_abi",
    "get_abi_compatibility",
    # Environments
    "Environment",
    "EnvironmentStore",
]
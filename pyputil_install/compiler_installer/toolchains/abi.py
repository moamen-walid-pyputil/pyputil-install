"""
ABI detection for compiler toolchains.

Detects the Application Binary Interface (ABI) that a toolchain
targets. The ABI defines how functions are called, how arguments
are passed, how data is laid out in memory, and how exceptions
are handled — all of which must be consistent between compilation
units that are linked together.

This module determines ABI characteristics by:
    1. Parsing the target triplet for architecture and OS hints.
    2. Running the preprocessor to check built-in defines.
    3. Querying the compiler with -dumpmachine and other flags.
    4. Examining the compiler's default flags.

Design
------
ABI detection is a standalone module. It takes a Toolchain (or
compiler path) and returns an ABIInfo dataclass. Each field
represents one axis of ABI variation.

Usage
-----
    from pathlib import Path
    from pyputil_install.compiler_installer.toolchains.abi import detect_abi, ABIInfo
    from pyputil_install.compiler_installer.toolchains.gcc import GCCToolchain

    gcc = GCCToolchain(Path("/usr"))
    abi = detect_abi(gcc)
    print(abi.pointer_size)       # 8 (x86_64) or 4 (x86)
    print(abi.endianness)         # "little" or "big"
    print(abi.calling_convention) # "sysv" or "msvc" or "aapcs"
    print(abi.exception_model)    # "dwarf" or "sjlj" or "seh"

Warnings
--------
- ABI detection runs the compiler and preprocessor. Each call
  spawns subprocesses with a timeout.
- Results reflect the DEFAULT target of the compiler. Use
  compiler flags (-m32, -target) to query alternative ABIs.
- Some fields (exception_model, name_mangling) are inferred
  from the target OS and may not reflect custom toolchain
  configurations.

User Instructions
-----------------
- Use detect_abi() for a complete ABI profile.
- Use get_abi_compatibility() to check if two toolchains can
  link their output together.
- The ABIInfo dataclass is hashable and can be used as a
  dictionary key for caching compiled objects.
"""

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .base import Toolchain

logger = logging.getLogger(__name__)

# ============================================================================
# ABIInfo
# ============================================================================


@dataclass(frozen=True)
class ABIInfo:
    """
    Complete ABI description for a compiler target.

    Attributes
    ----------
    target_triplet : str
        Full target triplet, e.g., "x86_64-linux-gnu".
    architecture : str
        CPU architecture: "x86_64", "aarch64", "arm", "x86", "riscv64".
    pointer_size : int
        Size of void* in bits: 64 or 32.
    endianness : str
        "little" or "big".
    sizeof_int : int
        sizeof(int) in bytes. Typically 4.
    sizeof_long : int
        sizeof(long) in bytes. 8 on 64-bit Unix, 4 on Windows and 32-bit.
    sizeof_wchar_t : int
        sizeof(wchar_t) in bytes. 4 on Linux, 2 on Windows.
    sizeof_long_double : int
        sizeof(long double) in bytes. 16 on x86_64, 8 on ARM64.
    char_is_signed : bool
        True if `char` is signed by default.
    calling_convention : str
        Default calling convention:
        "sysv" (x86_64 Unix), "msvc" (x86_64 Windows),
        "aapcs" (ARM), "aapcs64" (AArch64), "lp64" (RISC-V).
    exception_model : str
        Exception handling model:
        "dwarf" (Linux/macOS GCC), "sjlj" (MinGW, embedded),
        "seh" (Windows x86_64), "table" (ARM EH ABI).
    name_mangling : str
        C++ name mangling scheme: "itanium" (GCC/Clang Unix),
        "msvc" (MSVC, MinGW with -fms-runtime).
    float_abi : str
        Floating-point ABI: "hard", "soft", "softfp".
    stack_alignment : int
        Default stack alignment in bytes. 16 on most modern ABIs.
    is_ilp32 : bool
        True if int, long, and pointers are all 32-bit (e.g., x32 ABI).
    is_lp64 : bool
        True if long and pointers are 64-bit, int is 32-bit.
    is_llp64 : bool
        True if long long and pointers are 64-bit, long is 32-bit (Windows).
    """

    target_triplet: str = ""
    architecture: str = ""
    pointer_size: int = 0
    endianness: str = ""
    sizeof_int: int = 0
    sizeof_long: int = 0
    sizeof_wchar_t: int = 0
    sizeof_long_double: int = 0
    char_is_signed: bool = True
    calling_convention: str = ""
    exception_model: str = ""
    name_mangling: str = ""
    float_abi: str = ""
    stack_alignment: int = 16
    is_ilp32: bool = False
    is_lp64: bool = False
    is_llp64: bool = False


# ============================================================================
# Detection
# ============================================================================


def detect_abi(
    toolchain,
    extra_flags: Optional[List[str]] = None,
) -> ABIInfo:
    """
    Detect the complete ABI for a toolchain's default target.

    Runs the preprocessor to query built-in defines for type
    sizes, endianness, and other ABI-relevant macros.

    Parameters
    ----------
    toolchain : Toolchain or Path
        Toolchain object or direct compiler path.
    extra_flags : Optional[List[str]]
        Additional compiler flags to query a different ABI
        (e.g., ["-m32"] for 32-bit mode on x86_64).

    Returns
    -------
    ABIInfo
        Complete ABI description.

    Example
    -------
    >>> abi = detect_abi(my_gcc)
    >>> print(abi.pointer_size)
    64
    >>> print(abi.endianness)
    'little'
    >>> print(abi.calling_convention)
    'sysv'
    """
    compiler = _get_compiler_path(toolchain)
    if compiler is None:
        return ABIInfo()

    # Build query flags
    flags = list(extra_flags) if extra_flags else []

    # Get target triplet
    triplet = _get_target_triplet(compiler, flags)

    # Get preprocessor defines
    defines = _get_preprocessor_defines(compiler, flags)

    # Parse each ABI axis
    arch = _detect_architecture(triplet, defines)
    pointer_size = _detect_pointer_size(defines, arch)
    endianness = _detect_endianness(defines)
    sizeof_int = _detect_type_size(defines, "int")
    sizeof_long = _detect_type_size(defines, "long")
    sizeof_wchar_t = _detect_type_size(defines, "wchar_t")
    sizeof_long_double = _detect_type_size(defines, "long double")
    char_is_signed = _detect_char_signed(defines)
    calling_convention = _detect_calling_convention(arch, triplet)
    exception_model = _detect_exception_model(arch, triplet)
    name_mangling = _detect_name_mangling(triplet, defines)
    float_abi = _detect_float_abi(defines, arch)
    stack_alignment = _detect_stack_alignment(arch, defines)

    # Determine LP model
    is_ilp32 = pointer_size == 32 and sizeof_long == 4 and sizeof_int == 4
    is_lp64 = pointer_size == 64 and sizeof_long == 8 and sizeof_int == 4
    is_llp64 = pointer_size == 64 and sizeof_long == 4 and sizeof_int == 4

    return ABIInfo(
        target_triplet=triplet,
        architecture=arch,
        pointer_size=pointer_size,
        endianness=endianness,
        sizeof_int=sizeof_int,
        sizeof_long=sizeof_long,
        sizeof_wchar_t=sizeof_wchar_t,
        sizeof_long_double=sizeof_long_double,
        char_is_signed=char_is_signed,
        calling_convention=calling_convention,
        exception_model=exception_model,
        name_mangling=name_mangling,
        float_abi=float_abi,
        stack_alignment=stack_alignment,
        is_ilp32=is_ilp32,
        is_lp64=is_lp64,
        is_llp64=is_llp64,
    )


# ============================================================================
# Compatibility
# ============================================================================


def get_abi_compatibility(abi1: ABIInfo, abi2: ABIInfo) -> bool:
    """
    Check if two ABIs are compatible for linking.

    Two ABIs are compatible if they share the same calling
    convention, exception model, name mangling scheme,
    pointer size, and endianness.

    Parameters
    ----------
    abi1 : ABIInfo
        First ABI.
    abi2 : ABIInfo
        Second ABI.

    Returns
    -------
    bool
        True if objects compiled for these ABIs can be linked
        together.

    Example
    -------
    >>> gcc_abi = detect_abi(gcc)
    >>> clang_abi = detect_abi(clang)
    >>> if get_abi_compatibility(gcc_abi, clang_abi):
    ...     print("Can link GCC and Clang objects")
    """
    if abi1.pointer_size != abi2.pointer_size:
        return False
    if abi1.endianness != abi2.endianness:
        return False
    if abi1.calling_convention != abi2.calling_convention:
        return False
    if abi1.exception_model != abi2.exception_model:
        return False
    if abi1.name_mangling != abi2.name_mangling:
        return False
    return True


# ============================================================================
# Internal: compiler queries
# ============================================================================


def _get_target_triplet(compiler: Path, flags: List[str]) -> str:
    """Get the target triplet from -dumpmachine."""
    try:
        result = subprocess.run(
            [str(compiler)] + flags + ["-dumpmachine"],
            capture_output=True, text=True, timeout=10, shell=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return ""


def _get_preprocessor_defines(compiler: Path, flags: List[str]) -> Dict[str, str]:
    """
    Get preprocessor defines by running the compiler with -dM -E.

    Returns a dict of MACRO_NAME -> value (or "1" if defined without value).
    """
    cmd = [str(compiler)] + flags + ["-dM", "-E", "-"]
    try:
        result = subprocess.run(
            cmd,
            input="",
            capture_output=True, text=True, timeout=15, shell=False,
        )
        if result.returncode != 0:
            return {}

        defines = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("#define "):
                parts = line[8:].split(None, 2)
                if len(parts) == 1:
                    defines[parts[0]] = "1"
                elif len(parts) >= 2:
                    defines[parts[0]] = parts[1]
        return defines
    except Exception:
        return {}


# ============================================================================
# Internal: individual ABI axis detection
# ============================================================================


def _detect_architecture(triplet: str, defines: Dict[str, str]) -> str:
    """Detect architecture from triplet and defines."""
    arch = triplet.split("-")[0] if triplet else ""

    arch_map = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
        "arm": "arm",
        "armv7": "arm",
        "armv7l": "arm",
        "armv6": "arm",
        "i686": "x86",
        "i386": "x86",
        "i586": "x86",
        "riscv64": "riscv64",
        "riscv32": "riscv32",
        "wasm32": "wasm32",
        "wasm64": "wasm64",
    }
    return arch_map.get(arch, arch)


def _detect_pointer_size(defines: Dict[str, str], arch: str) -> int:
    """Detect pointer size from __SIZEOF_POINTER__ or architecture."""
    if "__SIZEOF_POINTER__" in defines:
        return int(defines["__SIZEOF_POINTER__"]) * 8
    if arch in ("x86_64", "aarch64", "riscv64"):
        return 64
    if arch in ("x86", "arm", "riscv32", "wasm32"):
        return 32
    return 64


def _detect_endianness(defines: Dict[str, str]) -> str:
    """Detect endianness from __BYTE_ORDER__."""
    if "__BYTE_ORDER__" in defines:
        order = defines["__BYTE_ORDER__"]
        if "LITTLE" in order or "1234" in order:
            return "little"
        if "BIG" in order or "4321" in order:
            return "big"
    if "__LITTLE_ENDIAN__" in defines:
        return "little"
    if "__BIG_ENDIAN__" in defines:
        return "big"
    return "little"  # Default assumption


def _detect_type_size(defines: Dict[str, str], type_name: str) -> int:
    """Detect sizeof for a type from preprocessor defines."""
    macro_map = {
        "int": "__SIZEOF_INT__",
        "long": "__SIZEOF_LONG__",
        "wchar_t": "__SIZEOF_WCHAR_T__",
        "long double": "__SIZEOF_LONG_DOUBLE__",
    }
    macro = macro_map.get(type_name)
    if macro and macro in defines:
        return int(defines[macro])

    # Fallback for wchar_t
    if type_name == "wchar_t":
        if "__WCHAR_TYPE__" in defines:
            wchar_type = defines["__WCHAR_TYPE__"]
            if "int" in wchar_type:
                return _detect_type_size(defines, "int")
            if "short" in wchar_type:
                return 2
    return 0


def _detect_char_signed(defines: Dict[str, str]) -> bool:
    """Detect if char is signed."""
    if "__CHAR_UNSIGNED__" in defines and defines["__CHAR_UNSIGNED__"] == "1":
        return False
    return True


def _detect_calling_convention(arch: str, triplet: str) -> str:
    """Detect the default calling convention."""
    if "mingw" in triplet.lower() or "windows" in triplet.lower() or "msvc" in triplet.lower():
        return "msvc" if arch == "x86_64" else "cdecl"
    if arch in ("aarch64",):
        return "aapcs64"
    if arch == "arm":
        return "aapcs"
    if arch in ("riscv64", "riscv32"):
        return "lp64" if "64" in arch else "ilp32"
    if arch == "x86":
        return "cdecl"
    if arch == "x86_64":
        return "sysv"
    return "unknown"


def _detect_exception_model(arch: str, triplet: str) -> str:
    """Detect the exception handling model."""
    lower = triplet.lower()
    if "mingw" in lower and arch == "x86":
        return "sjlj"
    if "mingw" in lower and arch == "x86_64":
        return "seh"
    if "windows" in lower or "msvc" in lower:
        return "seh" if arch == "x86_64" else "sjlj"
    if arch == "arm":
        return "table"
    return "dwarf"


def _detect_name_mangling(triplet: str, defines: Dict[str, str]) -> str:
    """Detect the C++ name mangling scheme."""
    lower = triplet.lower()
    if "mingw" in lower or "windows" in lower or "msvc" in lower:
        if "__MSVC_RUNTIME_CHECKS" in defines or "_MSC_VER" in defines:
            return "msvc"
        # MinGW GCC uses itanium by default, but can use msvc with flags
        return "itanium"
    return "itanium"


def _detect_float_abi(defines: Dict[str, str], arch: str) -> str:
    """Detect the floating-point ABI."""
    if arch == "arm":
        if "__ARM_PCS_VFP" in defines:
            return "hard"
        if "__SOFTFP__" in defines:
            return "softfp"
        return "soft"
    return "hard"


def _detect_stack_alignment(arch: str, defines: Dict[str, str]) -> int:
    """Detect the default stack alignment."""
    if arch in ("x86_64", "aarch64", "riscv64"):
        return 16
    if arch == "x86":
        return 16  # Modern GCC on x86 defaults to 16
    if arch == "arm":
        return 8
    return 16


# ============================================================================
# Helpers
# ============================================================================


def _get_compiler_path(toolchain) -> Optional[Path]:
    """Extract the C compiler path from a Toolchain or Path."""
    if isinstance(toolchain, Path):
        return toolchain
    if isinstance(toolchain, Toolchain):
        return toolchain.c_compiler
    return None
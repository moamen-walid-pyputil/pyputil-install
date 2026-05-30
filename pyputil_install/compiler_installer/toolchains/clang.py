"""
Clang toolchain implementation.

Represents an installed LLVM Clang compiler suite. Detects
executables, version, vendor, target triplet, sysroot, and
runtime configuration from a standard Clang/LLVM installation.

Layout
------
A standard LLVM/Clang installation follows GCC conventions:
    {prefix}/
        bin/
            clang               # C compiler
            clang++             # C++ compiler (symlink to clang)
            clang-cl            # MSVC-compatible driver (Windows)
            clang-cpp           # C preprocessor
            lld                 # LLVM linker (ELF)
            ld.lld              # Alternative linker name
            ld64.lld            # macOS linker
            lld-link            # Windows linker
            wasm-ld             # WebAssembly linker
            llvm-ar             # LLVM archiver
            llvm-ranlib         # LLVM ranlib
            llvm-nm             # LLVM nm
            llvm-objcopy        # LLVM objcopy
            llvm-objdump        # LLVM objdump
            llvm-readelf        # LLVM readelf
            llvm-size           # LLVM size
            llvm-strings        # LLVM strings
            llvm-strip          # LLVM strip
            llvm-as             # LLVM assembler
            llvm-link           # LLVM bitcode linker
            llvm-dis            # LLVM bitcode disassembler
            llvm-config         # Build configuration utility
            scan-build          # Static analyzer driver
        lib/
            clang/{version}/
                include/         # Built-in headers (stddef.h, stdarg.h, etc.)
                lib/
                    linux/       # Runtime libraries (libclang_rt.*.a)
        include/
            c++/v1/             # libc++ headers (if included)
        share/
            clang/              # Clang data files (sanitizer blacklists, etc.)

Apple Clang (Xcode) layout:
    /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/
        bin/
            clang               # Apple Clang
            clang++             # Apple Clang++
            ar, ranlib, nm, ... # Apple's cctools (not LLVM)
        lib/
            clang/{apple_version}/

Homebrew LLVM layout:
    /opt/homebrew/opt/llvm/
        bin/
            clang               # Upstream LLVM
            clang++             # Upstream LLVM
            lld, llvm-ar, ...   # Full LLVM toolchain

Usage
-----
    from pathlib import Path
    from pyputil_install.compiler_installer.toolchains.clang import ClangToolchain

    # System Clang (Linux)
    clang = ClangToolchain(Path("/usr"))
    if clang.is_valid():
        print(clang.version)          # "18.1.8"
        print(clang.vendor)           # "LLVM"
        print(clang.target_triplet)   # "x86_64-linux-gnu"
        print(clang.c_compiler)       # Path("/usr/bin/clang")
        print(clang.is_apple_clang)   # False

    # Apple Clang (macOS)
    xcode = ClangToolchain(
        Path("/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr")
    )
    print(xcode.vendor)              # "Apple"
    print(xcode.is_apple_clang)      # True

    # Homebrew LLVM (macOS)
    brew = ClangToolchain(Path("/opt/homebrew/opt/llvm"))
    print(brew.llvm_config_path)     # Path to llvm-config

Warnings
--------
- Clang is often installed alongside GCC/binutils. On Linux,
  /usr/bin/clang may use the system's ld, ar, etc. This class
  detects LLVM-specific tools first, then falls back to GNU
  binutils names for tools not provided by LLVM.
- Apple Clang uses a DIFFERENT version numbering scheme. The
  version from `clang --version` on macOS (e.g., "16.0.0")
  does NOT correspond to upstream LLVM versions. Check the
  `vendor` property before comparing versions.
- On macOS, /usr/bin/clang is Apple Clang. Homebrew installs
  upstream LLVM at /opt/homebrew/opt/llvm/bin/clang.
- The clang binary dispatches by argv[0]. clang++ and clang-cl
  are typically symlinks to the same clang binary.
- Some LLVM tools (llvm-config, scan-build) are not part of
  the core compilation pipeline and may be absent in minimal
  installations.

User Instructions
-----------------
- Pass the installation PREFIX, not the bin directory:
    Correct: ClangToolchain(Path("/usr"))
    Wrong:   ClangToolchain(Path("/usr/bin"))
- For system Clang: ClangToolchain(Path("/usr"))
- For Xcode Clang: Use the full path to the toolchain:
    Path("/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr")
- For Homebrew LLVM: ClangToolchain(Path("/opt/homebrew/opt/llvm"))
- For official LLVM binaries: ClangToolchain(Path("/usr/local/llvm"))
"""

import logging
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .base import (
    Toolchain,
    ToolchainKind,
    ToolRole,
    CompileResult,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Executable name maps for Clang/LLVM
# ============================================================================

# Primary map: LLVM-specific tool names. These are checked first.
_LLVM_EXECUTABLE_MAP: Dict[str, ToolRole] = {
    "clang": ToolRole.C_COMPILER,
    "clang++": ToolRole.CXX_COMPILER,
    "clang-cl": ToolRole.C_COMPILER,       # MSVC-compatible driver (Windows)
    "clang-cpp": ToolRole.C_COMPILER,       # C preprocessor mode
    "lld": ToolRole.LINKER,
    "ld.lld": ToolRole.LINKER,
    "ld64.lld": ToolRole.LINKER,            # macOS Mach-O linker
    "lld-link": ToolRole.LINKER,            # Windows COFF linker
    "wasm-ld": ToolRole.LINKER,             # WebAssembly linker
    "llvm-ar": ToolRole.ARCHIVER,
    "llvm-ranlib": ToolRole.RANLIB,
    "llvm-nm": ToolRole.NM,
    "llvm-objcopy": ToolRole.OBJCOPY,
    "llvm-objdump": ToolRole.OBJDUMP,
    "llvm-readelf": ToolRole.READELF,
    "llvm-size": ToolRole.SIZE,
    "llvm-strings": ToolRole.STRINGS,
    "llvm-strip": ToolRole.STRIP,
    "llvm-as": ToolRole.ASSEMBLER,
}

# Fallback map: GNU binutils names used when LLVM tools are absent.
# On Linux, Clang often relies on the system's binutils.
_GNU_FALLBACK_MAP: Dict[str, ToolRole] = {
    "ar": ToolRole.ARCHIVER,
    "ranlib": ToolRole.RANLIB,
    "nm": ToolRole.NM,
    "strip": ToolRole.STRIP,
    "objcopy": ToolRole.OBJCOPY,
    "objdump": ToolRole.OBJDUMP,
    "readelf": ToolRole.READELF,
    "size": ToolRole.SIZE,
    "strings": ToolRole.STRINGS,
    "as": ToolRole.ASSEMBLER,
    "ld": ToolRole.LINKER,
}

# Supplementary tools: useful but not required for compilation
_SUPPLEMENTARY_NAMES: Dict[str, str] = {
    "llvm-config": "llvm_config",
    "llvm-link": "llvm_link",
    "llvm-dis": "llvm_dis",
    "scan-build": "scan_build",
}


class ClangToolchain(Toolchain):
    """
    Represents an installed LLVM Clang compiler suite.

    Detects native and Apple Clang installations by scanning the
    bin/ directory for LLVM-specific and fallback GNU tool names.
    Handles Apple Clang (Xcode), Homebrew LLVM, and official LLVM
    binary releases.

    Parameters
    ----------
    path : Path
        Root directory of the Clang/LLVM installation. This is
        the prefix directory (e.g., /usr, /opt/homebrew/opt/llvm).
    validate : bool
        If True (default), scans for executables, detects version,
        vendor, target triplet, sysroot, and resource directory.
        Set to False for offline or testing scenarios.

    Attributes (from Toolchain)
    ---------------------------
    path : Path
        Resolved installation root directory.
    bin_dir : Path
        The bin/ subdirectory (path / "bin").
    kind : ToolchainKind
        Always ToolchainKind.CLANG.
    version : str
        Detected Clang version string.
    target_triplet : str
        Detected default target triplet.
    executables : Dict[ToolRole, Path]
        Mapping of ToolRole to discovered executable paths.

    Properties
    ----------
    is_apple_clang : bool
        True if this is Apple's Clang (Xcode), not upstream LLVM.
    vendor : str
        "Apple", "LLVM", or "unknown".
    is_clang_cl_available : bool
        True if clang-cl (MSVC-compatible driver) is available.
    resource_dir : Optional[Path]
        Path to Clang's resource directory (built-in headers).
    sysroot : Optional[Path]
        Path to the default sysroot.
    llvm_config_path : Optional[Path]
        Path to llvm-config, if found.
    scan_build_path : Optional[Path]
        Path to scan-build, if found.
    """

    def __init__(self, path: Path, validate: bool = True) -> None:
        self.kind = ToolchainKind.CLANG
        self._vendor: str = "unknown"
        self._is_apple_clang: bool = False
        self._resource_dir: Optional[Path] = None
        self._sysroot: Optional[Path] = None

        # Supplementary tool paths
        self._supplementary: Dict[str, Optional[Path]] = {
            name: None for name in _SUPPLEMENTARY_NAMES.values()
        }

        super().__init__(path, validate)

    # ==================================================================
    # Executable discovery
    # ==================================================================

    def _find_executables(self) -> None:
        """
        Scan for Clang/LLVM executables in the bin directory.

        Tries multiple bin directory locations:
        1. {path}/bin (standard layout)
        2. {path} itself (if user passed the bin directory directly)

        Checks LLVM-specific tool names first, then falls back to
        GNU binutils names for missing roles.
        """
        bin_candidates = [self.path / "bin"]

        # If the path itself looks like a bin directory
        if self.path.name == "bin" and self.path.is_dir():
            bin_candidates.insert(0, self.path)

        actual_bin_dir: Optional[Path] = None
        for candidate in bin_candidates:
            if candidate.is_dir():
                actual_bin_dir = candidate
                break

        if actual_bin_dir is None:
            logger.warning("No bin directory found for Clang at %s", self.path)
            return

        self.bin_dir = actual_bin_dir
        logger.debug("Using Clang bin directory: %s", self.bin_dir)

        # Phase 1: LLVM-specific tools
        for name, role in _LLVM_EXECUTABLE_MAP.items():
            if role in self.executables:
                continue
            exe_path = self._find_file_in_dir(self.bin_dir, name)
            if exe_path:
                self.executables[role] = exe_path

        # Phase 2: GNU binutils fallback for missing tools
        for name, role in _GNU_FALLBACK_MAP.items():
            if role in self.executables:
                continue
            exe_path = self._find_file_in_dir(self.bin_dir, name)
            if exe_path:
                self.executables[role] = exe_path

        # Phase 3: Supplementary tools
        for exe_name, attr_name in _SUPPLEMENTARY_NAMES.items():
            exe_path = self._find_file_in_dir(self.bin_dir, exe_name)
            if exe_path:
                self._supplementary[attr_name] = exe_path

        if ToolRole.C_COMPILER in self.executables:
            logger.debug(
                "Found Clang C compiler: %s",
                self.executables[ToolRole.C_COMPILER],
            )
        else:
            logger.warning("No Clang C compiler found in %s", self.bin_dir)

    # ==================================================================
    # Version and vendor detection
    # ==================================================================

    def _get_version(self) -> str:
        """
        Detect Clang version and vendor from `clang --version`.

        Clang output formats:
            Upstream LLVM:
                clang version 18.1.8 (https://github.com/llvm/llvm-project.git ...)
            Apple Clang:
                Apple clang version 16.0.0 (clang-1600.0.26.4)
                Target: x86_64-apple-darwin23.0.0
            Ubuntu Clang:
                Ubuntu clang version 18.1.3 (1ubuntu1)

        The vendor is detected from keywords in the first line.
        Apple Clang version numbers do NOT correspond to upstream
        LLVM versions.

        Returns
        -------
        str
            Detected version string, e.g., "18.1.8" or "16.0.0".
            Returns "0.0.0" if detection fails.
        """
        compiler = self.executables.get(ToolRole.C_COMPILER)
        if compiler is None:
            return "0.0.0"

        try:
            result = subprocess.run(
                [str(compiler), "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
            if result.returncode != 0:
                logger.debug("clang --version returned non-zero: %d", result.returncode)
                return "0.0.0"

            output = result.stdout or result.stderr
            if not output.strip():
                return "0.0.0"

            lines = output.splitlines()
            first_line = lines[0] if lines else ""

            # ---- Detect vendor from the first line ----
            lower_line = first_line.lower()
            if "apple" in lower_line:
                self._vendor = "Apple"
                self._is_apple_clang = True
            elif "clang" in lower_line:
                self._vendor = "LLVM"
            else:
                self._vendor = "unknown"

            # ---- Detect target triplet from output ----
            for line in lines:
                if line.strip().startswith("Target:"):
                    detected_target = line.split(":", 1)[1].strip()
                    if detected_target and not self.target_triplet:
                        self.target_triplet = detected_target
                    break

            # ---- Extract version ----
            # Pattern: "version 18.1.8"
            match = re.search(r"version\s+(\d+\.\d+\.\d+)", first_line)
            if match:
                return match.group(1)

            # Fallback: any three-part number
            match = re.search(r"(\d+\.\d+\.\d+)", first_line)
            if match:
                return match.group(1)

            # Fallback: two-part number
            match = re.search(r"(\d+\.\d+)", first_line)
            if match:
                return match.group(1)

        except subprocess.TimeoutExpired:
            logger.warning("clang --version timed out after 10s")
        except Exception as exc:
            logger.debug("Clang version detection failed: %s", exc)

        return "0.0.0"

    # ==================================================================
    # Resource directory and sysroot
    # ==================================================================

    @property
    def resource_dir(self) -> Optional[Path]:
        """
        Return Clang's resource directory containing built-in headers.

        Detected by running: clang -print-resource-dir
        This directory contains stddef.h, stdarg.h, limits.h, etc.

        Returns
        -------
        Optional[Path]
            Path to the resource directory, or None if detection fails.
        """
        if self._resource_dir is not None:
            return self._resource_dir

        compiler = self.executables.get(ToolRole.C_COMPILER)
        if compiler is None:
            return None

        try:
            result = subprocess.run(
                [str(compiler), "-print-resource-dir"],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                resource_str = result.stdout.strip().splitlines()[0].strip()
                resource_path = Path(resource_str)
                if resource_path.exists():
                    self._resource_dir = resource_path
                    return self._resource_dir
        except Exception as exc:
            logger.debug("Failed to detect resource dir: %s", exc)

        return None

    @property
    def sysroot(self) -> Optional[Path]:
        """
        Return the default sysroot for this Clang installation.

        Detected by running: clang -print-sysroot
        On macOS, this typically returns the Xcode SDK path.
        On Linux, this returns an empty string or "/".

        Returns
        -------
        Optional[Path]
            Path to the sysroot, or None if not configured.
        """
        if self._sysroot is not None:
            return self._sysroot

        compiler = self.executables.get(ToolRole.C_COMPILER)
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
            if result.returncode == 0 and result.stdout.strip():
                sysroot_str = result.stdout.strip().splitlines()[0].strip()
                sysroot_path = Path(sysroot_str)
                if sysroot_path.exists() and sysroot_str != "/":
                    self._sysroot = sysroot_path
                    return self._sysroot
        except Exception as exc:
            logger.debug("Failed to detect sysroot: %s", exc)

        return None

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def is_apple_clang(self) -> bool:
        """
        Check if this is Apple's Clang (Xcode), not upstream LLVM.

        Apple Clang uses a different version numbering scheme and
        may be missing some LLVM tools that upstream provides.

        Returns
        -------
        bool
            True if the vendor is "Apple".
        """
        return self._is_apple_clang or self._vendor == "Apple"

    @property
    def vendor(self) -> str:
        """
        Return the vendor string for this Clang installation.

        Returns
        -------
        str
            "Apple", "LLVM", or "unknown" if detection failed.
        """
        return self._vendor

    @property
    def is_clang_cl_available(self) -> bool:
        """
        Check if clang-cl (MSVC-compatible driver) is available.

        clang-cl accepts MSVC-style flags and is useful on Windows
        for compatibility with Visual Studio build systems.

        Returns
        -------
        bool
            True if a clang-cl executable was found in bin/.
        """
        return self._find_file_in_dir(self.bin_dir, "clang-cl") is not None

    @property
    def llvm_config_path(self) -> Optional[Path]:
        """
        Return the path to llvm-config, if found.

        llvm-config provides build flags (--cflags, --ldflags, --libs)
        useful for integrating LLVM into custom build systems.

        Returns
        -------
        Optional[Path]
            Path to llvm-config, or None.
        """
        return self._supplementary.get("llvm_config")

    @property
    def llvm_link_path(self) -> Optional[Path]:
        """
        Return the path to llvm-link (bitcode linker), if found.

        Returns
        -------
        Optional[Path]
            Path to llvm-link, or None.
        """
        return self._supplementary.get("llvm_link")

    @property
    def llvm_dis_path(self) -> Optional[Path]:
        """
        Return the path to llvm-dis (bitcode disassembler), if found.

        Returns
        -------
        Optional[Path]
            Path to llvm-dis, or None.
        """
        return self._supplementary.get("llvm_dis")

    @property
    def scan_build_path(self) -> Optional[Path]:
        """
        Return the path to scan-build (static analyzer driver), if found.

        Returns
        -------
        Optional[Path]
            Path to scan-build, or None.
        """
        return self._supplementary.get("scan_build")

    # ==================================================================
    # Internal helpers
    # ==================================================================

    @staticmethod
    def _find_file_in_dir(directory: Path, name: str) -> Optional[Path]:
        """
        Find an executable file in a directory.

        Checks for the bare name and, on Windows, the name with
        .exe extension. Verifies the file is executable.

        Parameters
        ----------
        directory : Path
            Directory to search. Must exist.
        name : str
            Base filename without extension.

        Returns
        -------
        Optional[Path]
            Absolute path to the executable if found and executable,
            None otherwise.
        """
        if not directory.is_dir():
            return None

        candidate = directory / name
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return candidate

        if os.name == "nt":
            candidate = directory / f"{name}.exe"
            if candidate.is_file() and os.access(str(candidate), os.X_OK):
                return candidate

        return None
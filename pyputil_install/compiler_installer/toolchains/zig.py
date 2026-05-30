"""
Zig toolchain implementation.

Represents an installed Zig compiler toolchain. Zig bundles a C/C++
compiler (clang), linker (lld), and its own build system, making it
a self-contained cross-compilation toolchain.

Unlike GCC or Clang, Zig is a single executable that provides all
functionality through subcommands:
    zig cc       # C compiler (clang-compatible)
    zig c++      # C++ compiler (clang-compatible)
    zig build    # Build system
    zig build-exe # Direct executable compilation
    zig build-lib # Direct library compilation
    zig build-obj # Direct object compilation

Layout
------
A Zig installation is minimal:
    {prefix}/
        zig                # Main executable (Unix)
        zig.exe            # Main executable (Windows)
        lib/
            std/            # Zig standard library
            libc/           # Bundled libc headers
            libcxx/         # Bundled libc++ headers
            libunwind/      # Bundled libunwind
        doc/                # Documentation

There is no separate bin/ directory. All tools are accessed via
`zig {subcommand}`.

Usage
-----
    from pathlib import Path
    from pyputil_install.compiler_installer.toolchains.zig import ZigToolchain

    zig = ZigToolchain(Path("/usr/local/zig"))
    if zig.is_valid():
        print(zig.version)              # "0.14.0"
        print(zig.c_compiler)           # Path to zig
        print(zig.target_triplet)       # Native target

        # Compile via zig cc
        result = zig.compile("source.c", output="program")
        # Equivalent to: zig cc -o program source.c

Warnings
--------
- Zig uses its own versioning (e.g., 0.14.0), not the bundled
  LLVM/Clang version.
- zig cc and zig c++ are clang-compatible but not identical.
  Some clang flags may not be supported.
- Cross-compilation targets are passed via -target flag, not
  a separate cross-compiler binary.
- Zig does not provide separate ar, ranlib, or nm executables.
  These are accessed via `zig ar`, `zig ranlib`, etc.
"""

import logging
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .base import (
    Toolchain,
    ToolchainKind,
    ToolRole,
    CompileResult,
)

logger = logging.getLogger(__name__)


class ZigToolchain(Toolchain):
    """
    Represents an installed Zig compiler toolchain.

    Parameters
    ----------
    path : Path
        Path to the Zig installation directory (containing `zig`
        executable) or directly to the zig executable itself.
    validate : bool
        If True, detects version and target. Default True.

    Attributes
    ----------
    path : Path
        Resolved installation root or zig executable path.
    bin_dir : Path
        Same as path (Zig has no separate bin directory).
    kind : ToolchainKind
        Always ToolchainKind.ZIG.
    version : str
        Detected Zig version.
    target_triplet : str
        Default target triplet.
    executables : Dict[ToolRole, Path]
        All tools point to the same zig executable.

    Properties
    ----------
    zig_path : Path
        Path to the zig executable.
    lib_dir : Optional[Path]
        Path to the lib/ directory containing std library.
    """

    def __init__(self, path: Path, validate: bool = True) -> None:
        self.kind = ToolchainKind.ZIG
        self._zig_path: Optional[Path] = None
        self._lib_dir: Optional[Path] = None
        super().__init__(path, validate)

    # ==================================================================
    # Executable discovery
    # ==================================================================

    def _find_executables(self) -> None:
        """
        Locate the zig executable.

        Checks:
        1. {path}/zig (if path is a directory)
        2. {path} (if path is a file named zig or zig.exe)
        3. {path}/bin/zig (standard layout)
        """
        candidates = [
            self.path / "zig",
            self.path / "zig.exe",
            self.path,
            self.path / "bin" / "zig",
            self.path / "bin" / "zig.exe",
        ]

        for candidate in candidates:
            if candidate.is_file() and os.access(str(candidate), os.X_OK):
                self._zig_path = candidate
                self.bin_dir = candidate.parent
                break

        if self._zig_path is None:
            logger.warning("Zig executable not found at %s", self.path)
            return

        # All roles point to the same zig executable with subcommands
        zig_path = self._zig_path
        self.executables[ToolRole.C_COMPILER] = zig_path
        self.executables[ToolRole.CXX_COMPILER] = zig_path
        self.executables[ToolRole.LINKER] = zig_path
        self.executables[ToolRole.ARCHIVER] = zig_path
        self.executables[ToolRole.RANLIB] = zig_path
        self.executables[ToolRole.NM] = zig_path
        self.executables[ToolRole.STRIP] = zig_path
        self.executables[ToolRole.OBJCOPY] = zig_path
        self.executables[ToolRole.OBJDUMP] = zig_path
        self.executables[ToolRole.READELF] = zig_path
        self.executables[ToolRole.SIZE] = zig_path
        self.executables[ToolRole.STRINGS] = zig_path

        logger.debug("Found Zig at %s", self._zig_path)

        # Detect lib directory
        lib_dir = self.bin_dir.parent / "lib"
        if lib_dir.is_dir():
            self._lib_dir = lib_dir

    # ==================================================================
    # Version and target
    # ==================================================================

    def _get_version(self) -> str:
        """Detect Zig version from `zig version`."""
        if self._zig_path is None:
            return "0.0.0"

        try:
            result = subprocess.run(
                [str(self._zig_path), "version"],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                match = re.search(r"(\d+\.\d+\.\d+)", version)
                if match:
                    return match.group(1)
                return version
        except Exception:
            pass
        return "0.0.0"

    def _get_target_triplet(self) -> str:
        """Detect default target from `zig targets`."""
        if self._zig_path is None:
            return ""

        try:
            result = subprocess.run(
                [str(self._zig_path), "target"],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
            if result.returncode == 0:
                return result.stdout.strip().splitlines()[0].strip()
        except Exception:
            pass
        return ""

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def zig_path(self) -> Optional[Path]:
        """Return the path to the zig executable."""
        return self._zig_path

    @property
    def lib_dir(self) -> Optional[Path]:
        """Return the lib/ directory, if found."""
        return self._lib_dir

    # ==================================================================
    # Compile override
    # ==================================================================

    def compile(
        self,
        source: str,
        output: Optional[str] = None,
        flags: Optional[List[str]] = None,
        language: Optional[str] = None,
        timeout: int = 120,
    ) -> CompileResult:
        """
        Compile using `zig cc` for C or `zig c++` for C++.

        Parameters
        ----------
        source : str
            Source file path.
        output : Optional[str]
            Output file path.
        flags : Optional[List[str]]
            Additional flags.
        language : Optional[str]
            "c" or "c++". Auto-detected from extension if None.
        timeout : int
            Seconds before timeout.
        """
        if not self.is_valid():
            raise RuntimeError(f"Zig toolchain at {self.path} is not valid")

        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Source not found: {source}")

        compiler = self._zig_path
        cmd = [str(compiler)]

        # Select subcommand
        if language == "c++" or source_path.suffix in (".cpp", ".cxx", ".cc", ".C"):
            cmd.append("c++")
        else:
            cmd.append("cc")

        if flags:
            cmd.extend(flags)

        cmd.append(str(source_path))

        if output:
            cmd.extend(["-o", output])
            output_file = Path(output)
        else:
            output_file = source_path.with_suffix("")
            if os.name == "nt":
                output_file = output_file.with_suffix(".exe")
            cmd.extend(["-o", str(output_file)])

        import time
        start = time.monotonic()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=False)
        except subprocess.TimeoutExpired:
            return CompileResult(returncode=-1, stderr=f"Timeout after {timeout}s", command=cmd)

        elapsed = (time.monotonic() - start) * 1000
        return CompileResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            command=cmd,
            output_file=output_file if proc.returncode == 0 and output_file.exists() else None,
            elapsed_ms=elapsed,
        )
"""
Android NDK toolchain implementation.

Represents an installed Android NDK (Native Development Kit).
The Android NDK provides cross-compilation toolchains targeting
Android devices (ARM, ARM64, x86, x86_64) using GCC (older NDK)
or LLVM/Clang (NDK r19+).

Layout
------
Android NDK r25+ (current):
    {ndk_root}/
        toolchains/
            llvm/
                prebuilt/
                    {host_platform}/
                        bin/
                            clang                  # C/C++ compiler
                            clang++                # C++ compiler
                            ld.lld                 # Linker
                            llvm-ar                # Archiver
                            llvm-strip             # Strip
                            llvm-nm                # NM
                            llvm-objcopy           # Objcopy
                            llvm-objdump           # Objdump
                            llvm-readelf           # Readelf
                            aapt2                  # Android Asset Packaging Tool
                            ndk-stack              # Stack trace tool
                            ndk-which              # Locate NDK tools
                        sysroot/
                            usr/
                                include/            # Unified headers
                                lib/
                                    aarch64-linux-android/   # ARM64
                                    arm-linux-androideabi/    # ARM32
                                    x86_64-linux-android/     # x86_64
                                    i686-linux-android/       # x86
        platforms/
            android-{api}/
                arch-{arch}/
                    usr/
                        lib/                      # Platform libraries (libc, libm, libdl)
        sysroot/ -> toolchains/llvm/prebuilt/{host}/sysroot

The NDK uses a single Clang binary that cross-compiles to all
Android architectures via the -target flag:
    clang -target aarch64-linux-android21
    clang -target armv7a-linux-androideabi21
    clang -target x86_64-linux-android21
    clang -target i686-linux-android21

Usage
-----
    from pathlib import Path
    from pyputil_install.compiler_installer.toolchains.android import AndroidNDKToolchain

    ndk = AndroidNDKToolchain(Path("~/Android/Sdk/ndk/25.2.9519653"))
    if ndk.is_valid():
        print(ndk.version)                 # "25.2.9519653"
        print(ndk.host_platform)           # "linux-x86_64"
        print(ndk.available_targets)       # ["aarch64-linux-android", ...]
        print(ndk.sysroot)                 # Path to unified sysroot

        # Compile for ARM64 Android
        result = ndk.compile(
            "source.c",
            output="program",
            target="aarch64-linux-android21",
            flags=["-O2"],
        )

Warnings
--------
- Android NDK targets are specified by API level (e.g., android21).
  The minimum API level determines which Android versions the
  compiled binary can run on.
- NDK r19+ uses Clang. GCC was removed in NDK r19. This class
  only supports the Clang-based toolchain.
- The host platform directory under prebuilt/ varies:
  linux-x86_64, darwin-x86_64, darwin-arm64, windows-x86_64.
- ndk-build and CMake toolchain files are available in the NDK
  but not managed by this class.
- The unified sysroot is a symlink to the LLVM toolchain sysroot
  in NDK r25+.

User Instructions
-----------------
- Pass the NDK root directory:
    Correct: Path("~/Android/Sdk/ndk/25.2.9519653")
- Set ANDROID_NDK_HOME environment variable for auto-detection.
- Use the `target` parameter to specify the Android target triplet
  with API level (e.g., "aarch64-linux-android21").
- Default API level is 21 (Android 5.0+) if not specified.
"""

import logging
import os
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

# Known Android target architectures
_ANDROID_TARGETS: Dict[str, str] = {
    "aarch64-linux-android": "arm64-v8a",
    "arm-linux-androideabi": "armeabi-v7a",
    "x86_64-linux-android": "x86_64",
    "i686-linux-android": "x86",
}

# Default API level if not specified in the target triplet
_DEFAULT_API_LEVEL = 21


class AndroidNDKToolchain(Toolchain):
    """
    Represents an installed Android NDK compiler toolchain.

    Parameters
    ----------
    path : Path
        Root directory of the Android NDK installation.
    validate : bool
        If True, scans for executables and detects version.
        Default True.

    Attributes
    ----------
    path : Path
        NDK root directory.
    bin_dir : Path
        Path to the LLVM bin/ directory.
    kind : ToolchainKind
        Always ToolchainKind.ANDROID_NDK.
    version : str
        NDK version string.
    target_triplet : str
        Default target triplet (empty, user must specify).
    executables : Dict[ToolRole, Path]
        LLVM tool paths.

    Properties
    ----------
    host_platform : Optional[str]
        Detected host platform directory name.
    sysroot : Optional[Path]
        Path to the unified sysroot.
    available_targets : List[str]
        List of available Android target triplets (without API level).
    clang_path : Optional[Path]
        Path to clang.
    """

    def __init__(self, path: Path, validate: bool = True) -> None:
        self.kind = ToolchainKind.ANDROID_NDK
        self._host_platform: Optional[str] = None
        self._sysroot: Optional[Path] = None
        self._clang_path: Optional[Path] = None
        self._available_targets: List[str] = []
        super().__init__(path, validate)

    # ==================================================================
    # Executable discovery
    # ==================================================================

    def _find_executables(self) -> None:
        """
        Locate the NDK LLVM toolchain directory.

        The NDK layout is:
            {ndk}/toolchains/llvm/prebuilt/{host_platform}/bin/
        """
        llvm_base = self.path / "toolchains" / "llvm" / "prebuilt"
        if not llvm_base.is_dir():
            logger.warning("LLVM toolchain not found at %s", llvm_base)
            return

        # Find the host platform directory
        try:
            for entry in llvm_base.iterdir():
                if entry.is_dir():
                    bin_dir = entry / "bin"
                    if bin_dir.is_dir() and (bin_dir / "clang").is_file():
                        self._host_platform = entry.name
                        self.bin_dir = bin_dir
                        break
        except PermissionError:
            pass

        if self._host_platform is None:
            logger.warning("No host platform found in %s", llvm_base)
            return

        logger.debug("NDK host platform: %s", self._host_platform)

        # Map LLVM tools
        tool_map = {
            "clang": ToolRole.C_COMPILER,
            "clang++": ToolRole.CXX_COMPILER,
            "ld.lld": ToolRole.LINKER,
            "llvm-ar": ToolRole.ARCHIVER,
            "llvm-strip": ToolRole.STRIP,
            "llvm-nm": ToolRole.NM,
            "llvm-objcopy": ToolRole.OBJCOPY,
            "llvm-objdump": ToolRole.OBJDUMP,
            "llvm-readelf": ToolRole.READELF,
        }

        for name, role in tool_map.items():
            exe = self._find_file(self.bin_dir, name)
            if exe:
                self.executables[role] = exe

        if ToolRole.C_COMPILER in self.executables:
            self._clang_path = self.executables[ToolRole.C_COMPILER]
            logger.debug("Found NDK Clang at %s", self._clang_path)

        # Detect sysroot
        sysroot = self.bin_dir.parent / "sysroot"
        if sysroot.is_dir():
            self._sysroot = sysroot
        else:
            alt_sysroot = self.path / "sysroot"
            if alt_sysroot.is_dir():
                self._sysroot = alt_sysroot

        # Detect available target architectures from sysroot lib/
        if self._sysroot:
            usr_lib = self._sysroot / "usr" / "lib"
            if usr_lib.is_dir():
                try:
                    for entry in usr_lib.iterdir():
                        if entry.is_dir() and entry.name in _ANDROID_TARGETS:
                            self._available_targets.append(entry.name)
                except PermissionError:
                    pass

        if not self._available_targets:
            self._available_targets = list(_ANDROID_TARGETS.keys())

    # ==================================================================
    # Version
    # ==================================================================

    def _get_version(self) -> str:
        """
        Detect NDK version from the source.properties file.

        Format:
            Pkg.Desc = Android NDK
            Pkg.Revision = 25.2.9519653

        Returns
        -------
        str
            NDK version string.
        """
        props_file = self.path / "source.properties"
        if props_file.is_file():
            try:
                content = props_file.read_text()
                for line in content.splitlines():
                    if "Pkg.Revision" in line:
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            return parts[1].strip()
            except Exception:
                pass

        # Fallback: try directory name
        dir_match = re.search(r"(\d+\.\d+\.\d+)", self.path.name)
        if dir_match:
            return dir_match.group(1)

        # Fallback: clang --version
        return super()._get_version()

    def _get_target_triplet(self) -> str:
        """NDK does not have a single default target."""
        return ""

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def host_platform(self) -> Optional[str]:
        """Return the NDK host platform directory name."""
        return self._host_platform

    @property
    def sysroot(self) -> Optional[Path]:
        """Return the unified sysroot path."""
        return self._sysroot

    @property
    def clang_path(self) -> Optional[Path]:
        """Return the path to clang."""
        return self._clang_path

    @property
    def available_targets(self) -> List[str]:
        """
        Return available Android target triplets.

        Returns
        -------
        List[str]
            e.g., ["aarch64-linux-android", "arm-linux-androideabi", ...]
        """
        return list(self._available_targets)

    # ==================================================================
    # Compile
    # ==================================================================

    def compile(
        self,
        source: str,
        output: Optional[str] = None,
        flags: Optional[List[str]] = None,
        language: Optional[str] = None,
        target: str = "",
        timeout: int = 120,
    ) -> CompileResult:
        """
        Compile for Android using the NDK Clang.

        Parameters
        ----------
        source : str
            Source file path.
        output : Optional[str]
            Output file path.
        flags : Optional[List[str]]
            Additional compiler flags.
        language : Optional[str]
            "c" or "c++".
        target : str
            Android target triplet with API level.
            Examples: "aarch64-linux-android21", "armv7a-linux-androideabi21".
            If empty, uses the first available target with default API.
        timeout : int
            Seconds before timeout.
        """
        if not self.is_valid():
            raise RuntimeError(f"NDK toolchain at {self.path} is not valid")

        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Source not found: {source}")

        # Determine target
        if not target:
            base = self._available_targets[0] if self._available_targets else "aarch64-linux-android"
            target = f"{base}{_DEFAULT_API_LEVEL}"

        compiler = self._clang_path
        cmd = [str(compiler), "-target", target]

        # Add sysroot
        if self._sysroot:
            cmd.extend(["--sysroot", str(self._sysroot)])

        if flags:
            cmd.extend(flags)
        else:
            cmd.append("-O2")

        cmd.append(str(source_path))

        if output:
            cmd.extend(["-o", output])
            output_file = Path(output)
        else:
            output_file = source_path.with_suffix("")
            cmd.extend(["-o", str(output_file)])

        import time
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, shell=False
            )
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

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _find_file(directory: Path, name: str) -> Optional[Path]:
        """Find an executable in a directory."""
        candidate = directory / name
        if candidate.is_file():
            return candidate
        if os.name == "nt":
            candidate = directory / f"{name}.exe"
            if candidate.is_file():
                return candidate
        return None
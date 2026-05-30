"""
GCC toolchain implementation.

Represents an installed GNU Compiler Collection installation.
Detects executables, version, target triplet, sysroot, and
multiarch configuration from a standard GCC layout.

Layout
------
A native GCC installation:
    {prefix}/
        bin/
            gcc                 # C compiler
            g++                 # C++ compiler
            cpp                 # C preprocessor
            gcc-ar              # GCC-specific archiver wrapper
            gcc-nm              # GCC-specific nm wrapper
            gcc-ranlib          # GCC-specific ranlib wrapper
            ar, as, ld, nm      # binutils (may be in separate dir)
            strip, objcopy, objdump, readelf, size, strings
        lib/
            gcc/{target}/{version}/
                include/         # GCC internal headers
                libgcc.a         # GCC support library
                crtbegin.o       # Start files
                crtend.o         # End files
                specs            # GCC specs file
        libexec/
            gcc/{target}/{version}/
                cc1              # C compiler proper
                cc1plus          # C++ compiler proper
                collect2         # Linker wrapper
                lto-wrapper      # LTO wrapper
                lto1             # LTO compiler
        include/
            c++/{version}/       # C++ headers
        share/
            man/

A cross-compiler GCC installation:
    {prefix}/
        bin/
            {target}-gcc
            {target}-g++
            {target}-ar
            ...
        {target}/
            sysroot/             # Optional sysroot

Usage
-----
    from pathlib import Path
    from pyputil_install.compiler_installer.toolchains.gcc import GCCToolchain

    # System GCC
    gcc = GCCToolchain(Path("/usr"))
    if gcc.is_valid():
        print(gcc.version)          # "14.2.0"
        print(gcc.target_triplet)   # "x86_64-linux-gnu"
        print(gcc.multiarch)        # "x86_64-linux-gnu"
        print(gcc.sysroot)          # Path("/") or None
        print(gcc.c_compiler)       # Path("/usr/bin/gcc")

    # Cross-compiler
    arm = GCCToolchain(Path("/opt/arm-gnu-toolchain"))
    print(arm.target_triplet)       # "arm-none-eabi"
    print(arm.is_cross_compiler)    # True
    print(arm.target_prefix)        # "arm-none-eabi"

    # Versioned GCC on Ubuntu
    gcc14 = GCCToolchain(Path("/usr"))
    # If gcc is not found directly, scans for gcc-14, gcc-13, etc.

Warnings
--------
- GCC installations using ccache or distcc wrappers may confuse
  detection. The _find_executables method looks for real executables
  directly in bin/, not through PATH resolution.
- On macOS, /usr/bin/gcc is often a symlink to clang. GCCToolchain
  will follow the symlink and detect Clang. Check the version output
  or target_triplet to distinguish: GCC uses "x86_64-linux-gnu",
  Apple Clang uses "x86_64-apple-darwin".
- MinGW-w64 GCC installations on Windows follow cross-compiler naming
  with prefix "x86_64-w64-mingw32-". These are detected correctly
  by _scan_cross_executables().
- The sysroot is detected by running `gcc -print-sysroot`. This
  subprocess call has a 10-second timeout.
- Version detection parses the first line of `gcc --version`.
  Custom or vendor-patched GCC builds may use different formatting.

User Instructions
-----------------
- Pass the installation PREFIX, not the bin directory:
    Correct: GCCToolchain(Path("/usr"))
    Wrong:   GCCToolchain(Path("/usr/bin"))
- For system GCC: GCCToolchain(Path("/usr"))
- For Homebrew GCC on macOS: GCCToolchain(Path("/opt/homebrew/opt/gcc"))
- For xPack GCC: GCCToolchain(Path("~/.local/share/toolforge/toolchains/gcc/14.2.0-2"))
- To use a specific version when multiple are installed, set
  TOOLFORGE_GCC_VERSION=14 to prefer gcc-14 over gcc.
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
# Executable name maps
# ============================================================================

# Mapping of bare executable names to ToolRole for native GCC
_NATIVE_EXECUTABLE_MAP: Dict[str, ToolRole] = {
    "gcc": ToolRole.C_COMPILER,
    "g++": ToolRole.CXX_COMPILER,
    "cpp": ToolRole.C_COMPILER,  # Preprocessor — treated as C compiler
    "cc": ToolRole.C_COMPILER,   # Common alias for C compiler
    "c++": ToolRole.CXX_COMPILER, # Common alias for C++ compiler
    "ar": ToolRole.ARCHIVER,
    "gcc-ar": ToolRole.ARCHIVER,
    "as": ToolRole.ASSEMBLER,
    "ld": ToolRole.LINKER,
    "ld.bfd": ToolRole.LINKER,
    "ld.gold": ToolRole.LINKER,
    "ranlib": ToolRole.RANLIB,
    "gcc-ranlib": ToolRole.RANLIB,
    "nm": ToolRole.NM,
    "gcc-nm": ToolRole.NM,
    "strip": ToolRole.STRIP,
    "objcopy": ToolRole.OBJCOPY,
    "objdump": ToolRole.OBJDUMP,
    "readelf": ToolRole.READELF,
    "size": ToolRole.SIZE,
    "strings": ToolRole.STRINGS,
    "dlltool": ToolRole.DLLTOOL,
    "windres": ToolRole.WINDRES,
}

# Mapping of cross-compiler suffixes to ToolRole
_CROSS_SUFFIX_MAP: Dict[str, ToolRole] = {
    "gcc": ToolRole.C_COMPILER,
    "g++": ToolRole.CXX_COMPILER,
    "cpp": ToolRole.C_COMPILER,
    "ar": ToolRole.ARCHIVER,
    "gcc-ar": ToolRole.ARCHIVER,
    "as": ToolRole.ASSEMBLER,
    "ld": ToolRole.LINKER,
    "ld.bfd": ToolRole.LINKER,
    "ranlib": ToolRole.RANLIB,
    "gcc-ranlib": ToolRole.RANLIB,
    "nm": ToolRole.NM,
    "gcc-nm": ToolRole.NM,
    "strip": ToolRole.STRIP,
    "objcopy": ToolRole.OBJCOPY,
    "objdump": ToolRole.OBJDUMP,
    "readelf": ToolRole.READELF,
    "size": ToolRole.SIZE,
    "strings": ToolRole.STRINGS,
    "dlltool": ToolRole.DLLTOOL,
    "windres": ToolRole.WINDRES,
}


class GCCToolchain(Toolchain):
    """
    Represents an installed GCC compiler suite.

    Detects native and cross-compiler GCC installations by scanning
    the bin/ directory for standard GCC executable names. Supports
    unprefixed (native), target-prefixed (cross), and version-suffixed
    (gcc-14) naming conventions.

    Parameters
    ----------
    path : Path
        Root directory of the GCC installation (the --prefix used
        during configuration). For a system install this is /usr.
        For xPack this is the toolchain version directory.
    validate : bool
        If True (default), scans for executables, detects version,
        target triplet, sysroot, and multiarch directory.
        Set to False for offline or testing scenarios.

    Attributes (from Toolchain)
    ---------------------------
    path : Path
        Resolved installation root directory.
    bin_dir : Path
        The bin/ subdirectory (path / "bin").
    kind : ToolchainKind
        Always ToolchainKind.GCC.
    version : str
        Detected GCC version string, e.g., "14.2.0".
    target_triplet : str
        Detected target triplet, e.g., "x86_64-linux-gnu".
    executables : Dict[ToolRole, Path]
        Mapping of ToolRole to discovered executable absolute paths.

    Properties
    ----------
    is_cross_compiler : bool
        True if the target triplet differs from the host machine.
    target_prefix : Optional[str]
        The cross-compiler prefix (e.g., "arm-none-eabi"), or None.
    sysroot : Optional[Path]
        The sysroot directory reported by `gcc -print-sysroot`.
    multiarch : Optional[str]
        The multiarch directory name, e.g., "x86_64-linux-gnu".
    gcc_path : Optional[Path]
        Alias for c_compiler.
    gpp_path : Optional[Path]
        Alias for cxx_compiler.
    gcc_lib_dir : Optional[Path]
        Path to the GCC internal library directory (lib/gcc/{target}/{version}).
    is_native : bool
        Inverse of is_cross_compiler.
    """

    def __init__(self, path: Path, validate: bool = True) -> None:
        self.kind = ToolchainKind.GCC
        self._target_prefix: Optional[str] = None
        self._sysroot: Optional[Path] = None
        self._multiarch: Optional[str] = None
        self._gcc_lib_dir: Optional[Path] = None
        super().__init__(path, validate)

    # ==================================================================
    # Executable discovery
    # ==================================================================

    def _find_executables(self) -> None:
        """
        Scan the bin/ directory for GCC executables.

        Attempts three strategies in order:
        1. Native unprefixed names (gcc, g++, ar, ...)
        2. Cross-compiler prefixed names ({target}-gcc, ...)
        3. Version-suffixed names (gcc-14, g++-14, ...)

        Sets self.executables to a mapping of ToolRole -> Path.
        Each role is assigned the first matching executable found.
        """
        if not self.bin_dir.exists() or not self.bin_dir.is_dir():
            logger.warning("GCC bin directory not found: %s", self.bin_dir)
            return

        # Strategy 1: native names
        self._scan_native_executables()

        # Strategy 2: cross-compiler prefixed names
        if ToolRole.C_COMPILER not in self.executables:
            self._scan_cross_executables()

        # Strategy 3: version-suffixed names (gcc-14)
        if ToolRole.C_COMPILER not in self.executables:
            self._scan_versioned_executables()

        # If still nothing, try the parent directory in case
        # the user passed bin/ instead of the prefix
        if ToolRole.C_COMPILER not in self.executables and self.bin_dir.parent != self.path:
            # Check if self.path itself is a bin directory
            pass

        if ToolRole.C_COMPILER in self.executables:
            logger.debug(
                "Found GCC C compiler: %s",
                self.executables[ToolRole.C_COMPILER],
            )
        else:
            logger.warning("No GCC C compiler found in %s", self.bin_dir)

    def _scan_native_executables(self) -> None:
        """
        Look for standard unprefixed executable names in bin/.

        Checks each name in _NATIVE_EXECUTABLE_MAP. Only sets
        a role if it has not already been assigned.
        """
        for name, role in _NATIVE_EXECUTABLE_MAP.items():
            if role in self.executables:
                continue
            exe_path = self._find_file_in_dir(self.bin_dir, name)
            if exe_path:
                self.executables[role] = exe_path

    def _scan_cross_executables(self) -> None:
        """
        Look for cross-compiler executables with target- prefix.

        Scans bin/ for files matching {anything}-gcc. Extracts
        the prefix from the first match and uses it to locate
        other tools (e.g., {prefix}-g++, {prefix}-ar).

        Sets self._target_prefix to the detected prefix.
        """
        if not self.bin_dir.is_dir():
            return

        try:
            all_files = list(self.bin_dir.iterdir())
        except PermissionError:
            logger.warning("Permission denied reading %s", self.bin_dir)
            return

        # Find any file ending with "-gcc" or "-gcc.exe"
        gcc_candidates = [
            f for f in all_files
            if f.is_file() and (f.name.endswith("-gcc") or f.name.endswith("-gcc.exe"))
        ]

        if not gcc_candidates:
            return

        # Use the first candidate to determine the target prefix
        gcc_name = gcc_candidates[0].stem  # e.g., "arm-none-eabi-gcc"
        if not gcc_name.endswith("-gcc"):
            return

        prefix = gcc_name[:-4]  # Strip "-gcc" suffix
        if not prefix:
            return

        self._target_prefix = prefix
        logger.debug("Detected GCC cross-compiler prefix: %s", prefix)

        # Map all cross-compiler tools
        for suffix, role in _CROSS_SUFFIX_MAP.items():
            if role in self.executables:
                continue
            exe_name = f"{prefix}-{suffix}"
            exe_path = self._find_file_in_dir(self.bin_dir, exe_name)
            if exe_path:
                self.executables[role] = exe_path

    def _scan_versioned_executables(self) -> None:
        """
        Look for version-suffixed executables like gcc-14, g++-14.

        Some distributions (Ubuntu, Debian) install GCC with version
        numbers appended to executable names. This method finds the
        highest available version.

        Respects TOOLFORGE_GCC_VERSION environment variable to prefer
        a specific version.
        """
        if not self.bin_dir.is_dir():
            return

        try:
            all_files = list(self.bin_dir.iterdir())
        except PermissionError:
            return

        preferred_version = os.environ.get("TOOLFORGE_GCC_VERSION", "")

        # ---- C compiler: gcc-{version} ----
        gcc_versioned = self._collect_versioned(all_files, "gcc")
        if gcc_versioned:
            best = self._pick_best_version(gcc_versioned, preferred_version)
            self.executables[ToolRole.C_COMPILER] = best

        # ---- C++ compiler: g++-{version} ----
        gpp_versioned = self._collect_versioned(all_files, "g++")
        if gpp_versioned:
            best = self._pick_best_version(gpp_versioned, preferred_version)
            self.executables[ToolRole.CXX_COMPILER] = best

        # ---- Other tools: ar-{version}, ranlib-{version}, nm-{version}, strip-{version} ----
        versioned_roles: Dict[str, ToolRole] = {
            "ar": ToolRole.ARCHIVER,
            "ranlib": ToolRole.RANLIB,
            "nm": ToolRole.NM,
            "strip": ToolRole.STRIP,
        }

        for suffix, role in versioned_roles.items():
            if role in self.executables:
                continue
            candidates = self._collect_versioned(all_files, suffix)
            if candidates:
                self.executables[role] = self._pick_best_version(candidates, preferred_version)

    # ==================================================================
    # Sysroot and multiarch detection
    # ==================================================================

    @property
    def sysroot(self) -> Optional[Path]:
        """
        Return the sysroot directory for this GCC installation.

        Detected by running: gcc -print-sysroot
        Returns None if detection fails or the path does not exist.
        A return value of Path("/") means no sysroot is configured.

        Returns
        -------
        Optional[Path]
            The sysroot directory path.
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
                if sysroot_path.exists():
                    self._sysroot = sysroot_path
                    return self._sysroot
        except Exception as exc:
            logger.debug("Failed to detect sysroot: %s", exc)

        return None

    @property
    def multiarch(self) -> Optional[str]:
        """
        Return the multiarch directory name.

        Detected by running: gcc -print-multiarch
        Returns None if detection fails.

        On Debian/Ubuntu systems, this is typically the same as
        the target triplet, e.g., "x86_64-linux-gnu".

        Returns
        -------
        Optional[str]
            The multiarch directory name.
        """
        if self._multiarch is not None:
            return self._multiarch

        compiler = self.executables.get(ToolRole.C_COMPILER)
        if compiler is None:
            return None

        try:
            result = subprocess.run(
                [str(compiler), "-print-multiarch"],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                self._multiarch = result.stdout.strip().splitlines()[0].strip()
                return self._multiarch
        except Exception as exc:
            logger.debug("Failed to detect multiarch: %s", exc)

        return None

    @property
    def gcc_lib_dir(self) -> Optional[Path]:
        """
        Return the GCC internal library directory.

        Path: {prefix}/lib/gcc/{target}/{version}/

        This directory contains libgcc.a, crtbegin.o, crtend.o,
        and the specs file. It is where GCC looks for its own
        support libraries and start files.

        Returns
        -------
        Optional[Path]
            Path to the GCC library directory, or None if the
            target triplet or version is unknown.
        """
        if self._gcc_lib_dir is not None:
            return self._gcc_lib_dir

        if not self.target_triplet or not self.version:
            return None

        candidate = self.path / "lib" / "gcc" / self.target_triplet / self.version
        if candidate.exists() and candidate.is_dir():
            self._gcc_lib_dir = candidate
            return self._gcc_lib_dir

        return None

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def is_cross_compiler(self) -> bool:
        """
        Check if this GCC installation is a cross-compiler.

        Compares the target triplet against the host machine type.
        Uses a normalization map to handle common variations
        (e.g., "x86_64" vs "amd64", "aarch64" vs "arm64").

        Returns
        -------
        bool
            True if the target triplet machine differs from the
            host machine. False if target_triplet is empty or
            matches the host.
        """
        if not self.target_triplet:
            return False

        host_machine = platform.machine().lower()

        # Normalize common variations to canonical forms
        host_normalize: Dict[str, str] = {
            "x86_64": "x86_64",
            "amd64": "x86_64",
            "i686": "x86",
            "i386": "x86",
            "aarch64": "aarch64",
            "arm64": "aarch64",
            "armv7l": "arm",
            "armv6l": "arm",
        }
        host = host_normalize.get(host_machine, host_machine)

        target_lower = self.target_triplet.lower()
        return host not in target_lower

    @property
    def is_native(self) -> bool:
        """Inverse of is_cross_compiler."""
        return not self.is_cross_compiler

    @property
    def target_prefix(self) -> Optional[str]:
        """
        Return the cross-compiler target prefix.

        Returns
        -------
        Optional[str]
            The prefix (e.g., "arm-none-eabi", "x86_64-w64-mingw32"),
            or None if this is a native compiler.
        """
        return self._target_prefix

    @property
    def gcc_path(self) -> Optional[Path]:
        """Alias for c_compiler. Returns the path to gcc."""
        return self.c_compiler

    @property
    def gpp_path(self) -> Optional[Path]:
        """Alias for cxx_compiler. Returns the path to g++."""
        return self.cxx_compiler

    # ==================================================================
    # Version and target overrides
    # ==================================================================

    def _get_version(self) -> str:
        """
        Detect GCC version from `gcc --version`.

        Parses the first line for a version pattern like "14.2.0".
        GCC output format:
            gcc (GCC) 14.2.0
            gcc (Ubuntu 14.2.0-4ubuntu2) 14.2.0

        Returns "0.0.0" if detection fails.

        Returns
        -------
        str
            Detected version string.
        """
        version = super()._get_version()
        if version != "0.0.0":
            return version

        # Additional fallback: try -dumpversion
        compiler = self.executables.get(ToolRole.C_COMPILER)
        if compiler is None:
            return "0.0.0"

        try:
            result = subprocess.run(
                [str(compiler), "-dumpversion"],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().splitlines()[0].strip()
        except Exception:
            pass

        return "0.0.0"

    # ==================================================================
    # Internal helpers
    # ==================================================================

    @staticmethod
    def _find_file_in_dir(directory: Path, name: str) -> Optional[Path]:
        """
        Find a file in a directory, with optional .exe suffix on Windows.

        Parameters
        ----------
        directory : Path
            Directory to search.
        name : str
            Base filename without extension.

        Returns
        -------
        Optional[Path]
            Absolute path to the file if found and executable,
            None otherwise.
        """
        exe_path = directory / name
        if exe_path.is_file() and os.access(str(exe_path), os.X_OK):
            return exe_path

        if os.name == "nt":
            exe_path = directory / f"{name}.exe"
            if exe_path.is_file() and os.access(str(exe_path), os.X_OK):
                return exe_path

        return None

    @staticmethod
    def _collect_versioned(all_files: List[Path], base_name: str) -> Dict[int, Path]:
        """
        Collect versioned executables from a directory listing.

        Parameters
        ----------
        all_files : List[Path]
            All entries in the bin directory.
        base_name : str
            Base executable name (e.g., "gcc", "g++", "ar").

        Returns
        -------
        Dict[int, Path]
            Mapping of major version number to executable path.
            Only includes files that are executable.
        """
        pattern = re.compile(rf"^{re.escape(base_name)}-(\d+)(\.exe)?$")
        result: Dict[int, Path] = {}

        for f in all_files:
            if not f.is_file():
                continue
            match = pattern.match(f.name)
            if not match:
                continue
            if not os.access(str(f), os.X_OK):
                continue
            version_num = int(match.group(1))
            result[version_num] = f

        return result

    @staticmethod
    def _pick_best_version(candidates: Dict[int, Path], preferred: str) -> Path:
        """
        Pick the best versioned executable from candidates.

        If `preferred` is set and matches a candidate, returns that one.
        Otherwise returns the highest version number.

        Parameters
        ----------
        candidates : Dict[int, Path]
            Mapping of version number to path.
        preferred : str
            Preferred version string (e.g., "14").

        Returns
        -------
        Path
            The selected executable path.
        """
        if preferred and preferred.isdigit():
            preferred_int = int(preferred)
            if preferred_int in candidates:
                return candidates[preferred_int]

        # Return highest version
        return candidates[max(candidates.keys())]
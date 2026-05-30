"""
MSVC toolchain implementation.

Represents an installed Microsoft Visual C++ compiler suite.
Detects executables, version, target architecture, and Windows SDK
integration from Visual Studio or Build Tools installations.

Layout
------
Visual Studio 2022 (typical):
    C:/Program Files/Microsoft Visual Studio/2022/{edition}/
        VC/
            Tools/
                MSVC/
                    {version}/
                        bin/
                            Hostx64/x64/cl.exe       # x64 compiler
                            Hostx64/x86/cl.exe       # x86 cross-compiler
                            Hostx64/arm64/cl.exe     # ARM64 cross-compiler
                            Hostx86/x64/cl.exe       # x86->x64 cross-compiler
                            Hostx86/x86/cl.exe       # x86 native compiler
                        include/                      # C/C++ headers
                        lib/
                            x64/                      # x64 libraries
                            x86/                      # x86 libraries
                            arm64/                    # ARM64 libraries

Build Tools (standalone, no IDE):
    C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/
        VC/Tools/MSVC/{version}/
            ...

Older Visual Studio versions:
    VS 2019: C:/Program Files (x86)/Microsoft Visual Studio/2019/{edition}/
    VS 2017: C:/Program Files (x86)/Microsoft Visual Studio/2017/{edition}/

Host and Target Architecture
----------------------------
MSVC uses a host/target directory convention:
    bin/Host{host_arch}/{target_arch}/cl.exe

Common combinations:
    Hostx64/x64   — native 64-bit compiler
    Hostx64/x86   — 64-bit host, 32-bit target
    Hostx64/arm64 — 64-bit host, ARM64 target
    Hostx86/x86   — native 32-bit compiler

This class detects ALL available host/target combinations and
allows selecting which one to use via the `host_arch` and
`target_arch` parameters.

Environment
-----------
MSVC requires specific environment variables to function:
    INCLUDE, LIB, LIBPATH, PATH
These are typically set by vcvarsall.bat or VsDevCmd.bat.

This class provides `generate_env()` to produce the necessary
environment overrides without modifying the current process.
Use `apply_env()` to apply them to os.environ directly.

Usage
-----
    from pathlib import Path
    from pyputil_install.compiler_installer.toolchains.msvc import MSVCToolchain

    # Auto-detect from common installation paths
    msvc = MSVCToolchain(Path(
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.38.33130"
    ))
    if msvc.is_valid():
        print(msvc.version)              # "14.38.33130"
        print(msvc.c_compiler)           # Path to cl.exe
        print(msvc.available_archs)      # ["x64", "x86", "arm64"]
        print(msvc.windows_sdk_version)  # "10.0.22621.0"

        # Generate environment for x64 native compilation
        env = msvc.generate_env("x64", "x64")
        print(env["INCLUDE"])
        print(env["LIB"])

Warnings
--------
- MSVC is Windows-only. This class raises RuntimeError if
  instantiated on a non-Windows platform.
- The compiler (cl.exe) cannot be run directly. It requires
  the environment variables set by vcvarsall.bat or the
  `generate_env()` method.
- Visual Studio and Build Tools must be installed separately.
  This class detects existing installations but does NOT install
  or download anything.
- Microsoft changes installation paths between Visual Studio
  versions. The auto-detection in this class covers VS 2017,
  2019, and 2022.
- Windows SDK is a separate install and may be located at
  C:/Program Files (x86)/Windows Kits/10/.

User Instructions
-----------------
- Pass the MSVC tools directory (containing bin/, include/, lib/):
    Correct: Path(".../VC/Tools/MSVC/14.38.33130")
- To compile, first generate the environment:
    env = msvc.generate_env("x64", "x64")
    os.environ.update(env)
- Then call msvc.compile() normally.
- For cross-compilation, specify the target_arch parameter.
"""

import logging
import os
import platform as _platform
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


class MSVCToolchain(Toolchain):
    """
    Represents an installed Microsoft Visual C++ compiler suite.

    Detects cl.exe, link.exe, lib.exe, and other MSVC tools from
    a Visual Studio or Build Tools installation. Provides host/target
    architecture selection and environment variable generation.

    Parameters
    ----------
    path : Path
        Path to the MSVC tools directory. This is the directory
        containing bin/, include/, and lib/.
        Example: .../VC/Tools/MSVC/14.38.33130
    validate : bool
        If True (default), scans for executables and detects
        version, available architectures, and Windows SDK.
    host_arch : Optional[str]
        Default host architecture to use. One of "x64", "x86".
        If None, auto-detected from the host machine.
    target_arch : Optional[str]
        Default target architecture. One of "x64", "x86", "arm64".
        If None, defaults to the same as host_arch.

    Attributes
    ----------
    path : Path
        MSVC tools root directory.
    bin_dir : Path
        Base bin/ directory.
    kind : ToolchainKind
        Always ToolchainKind.MSVC.
    version : str
        MSVC toolchain version, e.g., "14.38.33130".
    target_triplet : str
        Target architecture (e.g., "x64", "x86", "arm64").
        Not a traditional triplet — MSVC uses simple arch names.
    executables : Dict[ToolRole, Path]
        Mapping of roles to discovered executable paths for the
        current host/target combination.

    Properties
    ----------
    available_host_archs : List[str]
        Host architectures with available compilers.
    available_target_archs : List[str]
        Target architectures with available compilers.
    windows_sdk_version : Optional[str]
        Detected Windows SDK version.
    windows_sdk_path : Optional[Path]
        Path to the Windows SDK root.
    """

    def __init__(
        self,
        path: Path,
        validate: bool = True,
        host_arch: Optional[str] = None,
        target_arch: Optional[str] = None,
    ) -> None:
        if _platform.system() != "Windows":
            raise RuntimeError(
                "MSVCToolchain is only available on Windows. "
                f"Current platform: {_platform.system()}"
            )

        self.kind = ToolchainKind.MSVC
        self._host_arch = host_arch or self._detect_native_arch()
        self._target_arch = target_arch or self._host_arch

        # Available host/target combinations found during scanning
        self._available_combinations: Set[Tuple[str, str]] = set()

        # Windows SDK
        self._windows_sdk_version: Optional[str] = None
        self._windows_sdk_path: Optional[Path] = None

        # MSVC uses a different layout: bin/Host{host}/{target}/
        self._compiler_bin_dir: Optional[Path] = None

        super().__init__(path, validate)

    # ==================================================================
    # Executable discovery
    # ==================================================================

    def _find_executables(self) -> None:
        """
        Scan for MSVC executables.

        MSVC executables are organized by host/target architecture
        in bin/Host{host}/{target}/ directories. This method scans
        all available combinations and sets self.executables to
        the one matching the selected host_arch and target_arch.
        """
        if not self.path.exists():
            logger.warning("MSVC path does not exist: %s", self.path)
            return

        # Verify this looks like an MSVC tools directory
        if not (self.path / "bin").is_dir():
            logger.warning("No bin/ directory in MSVC path: %s", self.path)
            return
        if not (self.path / "include").is_dir():
            logger.warning("No include/ directory in MSVC path: %s", self.path)
            return
        if not (self.path / "lib").is_dir():
            logger.warning("No lib/ directory in MSVC path: %s", self.path)
            return

        self.bin_dir = self.path / "bin"

        # Scan all Host*/ directories
        self._scan_host_target_dirs()

        # Select the best matching compiler for the requested archs
        self._select_compiler()

        # Detect version from the path or from cl.exe
        self.version = self._get_version()

        # Detect Windows SDK
        self._detect_windows_sdk()

    def _scan_host_target_dirs(self) -> None:
        """
        Scan bin/ for all Host{arch}/{arch}/ directories.

        Populates self._available_combinations with tuples of
        (host_arch, target_arch) for every directory containing
        cl.exe.
        """
        if not self.bin_dir.is_dir():
            return

        try:
            for host_entry in self.bin_dir.iterdir():
                if not host_entry.is_dir():
                    continue
                if not host_entry.name.startswith("Host"):
                    continue

                host_arch = self._parse_arch_from_dir(host_entry.name)
                if host_arch is None:
                    continue

                for target_entry in host_entry.iterdir():
                    if not target_entry.is_dir():
                        continue

                    target_arch = self._parse_arch_from_dir(target_entry.name)
                    if target_arch is None:
                        continue

                    # Check for cl.exe
                    cl_path = target_entry / "cl.exe"
                    if cl_path.is_file():
                        self._available_combinations.add((host_arch, target_arch))
                        logger.debug(
                            "Found MSVC compiler: Host%s -> %s",
                            host_arch, target_arch,
                        )
        except PermissionError:
            logger.warning("Permission denied scanning %s", self.bin_dir)

    def _select_compiler(self) -> None:
        """
        Select the best compiler for the requested host/target archs.

        If the exact combination is not available, tries to find
        the closest match. Sets self.executables and self._compiler_bin_dir.
        """
        # Exact match
        if (self._host_arch, self._target_arch) in self._available_combinations:
            self._compiler_bin_dir = (
                self.bin_dir / f"Host{self._host_arch}" / self._target_arch
            )
        # Same target, different host
        else:
            for host, target in self._available_combinations:
                if target == self._target_arch:
                    logger.info(
                        "Host%s not available, using Host%s for target %s",
                        self._host_arch, host, target,
                    )
                    self._host_arch = host
                    self._compiler_bin_dir = (
                        self.bin_dir / f"Host{host}" / target
                    )
                    break

        if self._compiler_bin_dir is None and self._available_combinations:
            # Pick first available
            host, target = next(iter(self._available_combinations))
            logger.warning(
                "Requested Host%s/%s not available. Falling back to Host%s/%s",
                self._host_arch, self._target_arch, host, target,
            )
            self._host_arch = host
            self._target_arch = target
            self._compiler_bin_dir = self.bin_dir / f"Host{host}" / target

        if self._compiler_bin_dir is None:
            logger.warning("No MSVC compiler found in %s", self.bin_dir)
            return

        # Map executables
        msvc_tools = {
            "cl.exe": ToolRole.C_COMPILER,
            "link.exe": ToolRole.LINKER,
            "lib.exe": ToolRole.ARCHIVER,
            "ml64.exe": ToolRole.ASSEMBLER,
            "dumpbin.exe": ToolRole.OBJDUMP,
            "editbin.exe": ToolRole.OBJCOPY,
        }

        for exe_name, role in msvc_tools.items():
            exe_path = self._compiler_bin_dir / exe_name
            if exe_path.is_file():
                self.executables[role] = exe_path

        # cl.exe serves as both C and C++ compiler
        if ToolRole.C_COMPILER in self.executables:
            self.executables[ToolRole.CXX_COMPILER] = self.executables[ToolRole.C_COMPILER]

        # Set target_triplet to the target arch (MSVC doesn't use triplets)
        self.target_triplet = self._target_arch

    # ==================================================================
    # Version and SDK detection
    # ==================================================================

    def _get_version(self) -> str:
        """
        Detect MSVC version.

        First tries to extract the version from the directory path
        (e.g., .../MSVC/14.38.33130). Falls back to running cl.exe
        to get the version from its output.

        Returns
        -------
        str
            Version string like "14.38.33130". Returns "0.0.0" on failure.
        """
        # Extract from path (most reliable)
        version_match = re.search(r"(\d+\.\d+\.\d+)", str(self.path))
        if version_match:
            return version_match.group(1)

        # Fall back to cl.exe
        cl_path = self.executables.get(ToolRole.C_COMPILER)
        if cl_path is None:
            return "0.0.0"

        try:
            result = subprocess.run(
                [str(cl_path)],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
            output = result.stdout or result.stderr
            if output:
                match = re.search(r"Version\s+(\d+\.\d+\.\d+)", output)
                if match:
                    return match.group(1)
                match = re.search(r"(\d+\.\d+\.\d+)", output)
                if match:
                    return match.group(1)
        except Exception as exc:
            logger.debug("MSVC version detection failed: %s", exc)

        return "0.0.0"

    def _detect_windows_sdk(self) -> None:
        """
        Detect the Windows SDK version and path.

        Checks common installation locations for the Windows SDK.
        """
        sdk_base = Path("C:/Program Files (x86)/Windows Kits/10")
        if not sdk_base.exists():
            return

        # Find the include directory with the highest version
        include_dir = sdk_base / "Include"
        if include_dir.is_dir():
            versions = []
            try:
                for entry in include_dir.iterdir():
                    if entry.is_dir() and re.match(r"^\d+\.\d+\.\d+\.\d+$", entry.name):
                        versions.append(entry.name)
            except PermissionError:
                pass

            if versions:
                # Sort by version number (newest first)
                versions.sort(
                    key=lambda v: tuple(int(x) for x in v.split(".")),
                    reverse=True,
                )
                self._windows_sdk_version = versions[0]

        self._windows_sdk_path = sdk_base

    # ==================================================================
    # Environment generation
    # ==================================================================

    def generate_env(
        self,
        host_arch: Optional[str] = None,
        target_arch: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate the environment variables needed to use this MSVC toolchain.

        MSVC requires INCLUDE, LIB, and LIBPATH to find headers and
        libraries. This method produces a dictionary of environment
        overrides that can be applied via os.environ.update().

        Parameters
        ----------
        host_arch : Optional[str]
            Host architecture ("x64" or "x86"). Defaults to the
            toolchain's configured host_arch.
        target_arch : Optional[str]
            Target architecture ("x64", "x86", "arm64"). Defaults
            to the toolchain's configured target_arch.

        Returns
        -------
        Dict[str, str]
            Dictionary of environment variables:
            - PATH: includes the MSVC bin directory and Windows SDK
            - INCLUDE: MSVC headers and Windows SDK headers
            - LIB: MSVC libraries and Windows SDK libraries
            - LIBPATH: MSVC library path
        """
        host = host_arch or self._host_arch
        target = target_arch or self._target_arch

        env: Dict[str, str] = {}

        # PATH additions
        paths_to_add = []

        # MSVC bin directory
        compiler_bin = self.bin_dir / f"Host{host}" / target
        if compiler_bin.is_dir():
            paths_to_add.append(str(compiler_bin))

        # Windows SDK bin directory
        if self._windows_sdk_path and self._windows_sdk_version:
            sdk_bin = self._windows_sdk_path / "bin" / self._windows_sdk_version / target
            if sdk_bin.is_dir():
                paths_to_add.append(str(sdk_bin))

        if paths_to_add:
            existing_path = os.environ.get("PATH", "")
            env["PATH"] = os.pathsep.join(paths_to_add + [existing_path])

        # INCLUDE
        includes = []

        # MSVC includes
        msvc_include = self.path / "include"
        if msvc_include.is_dir():
            includes.append(str(msvc_include))

        # Windows SDK includes
        if self._windows_sdk_path and self._windows_sdk_version:
            sdk_include = self._windows_sdk_path / "Include" / self._windows_sdk_version
            if sdk_include.is_dir():
                for subdir in ("ucrt", "shared", "um", "winrt", "cppwinrt"):
                    sub_path = sdk_include / subdir
                    if sub_path.is_dir():
                        includes.append(str(sub_path))

        if includes:
            existing_include = os.environ.get("INCLUDE", "")
            env["INCLUDE"] = os.pathsep.join(includes + [existing_include])

        # LIB
        libs = []

        # MSVC libraries
        msvc_lib = self.path / "lib" / target
        if msvc_lib.is_dir():
            libs.append(str(msvc_lib))

        # Windows SDK libraries
        if self._windows_sdk_path and self._windows_sdk_version:
            sdk_lib = self._windows_sdk_path / "Lib" / self._windows_sdk_version
            if sdk_lib.is_dir():
                for subdir in ("ucrt", "um"):
                    sub_path = sdk_lib / subdir / target
                    if sub_path.is_dir():
                        libs.append(str(sub_path))

        if libs:
            existing_lib = os.environ.get("LIB", "")
            env["LIB"] = os.pathsep.join(libs + [existing_lib])

        # LIBPATH
        libpaths = []
        if msvc_lib.is_dir():
            libpaths.append(str(msvc_lib))
        if libpaths:
            existing_libpath = os.environ.get("LIBPATH", "")
            env["LIBPATH"] = os.pathsep.join(libpaths + [existing_libpath])

        return env

    def apply_env(
        self,
        host_arch: Optional[str] = None,
        target_arch: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Apply the MSVC environment to the current process.

        Calls generate_env() and updates os.environ with the result.

        Parameters
        ----------
        host_arch : Optional[str]
            Host architecture.
        target_arch : Optional[str]
            Target architecture.

        Returns
        -------
        Dict[str, str]
            The environment overrides that were applied.
        """
        env = self.generate_env(host_arch, target_arch)
        os.environ.update(env)
        return env

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def available_host_archs(self) -> List[str]:
        """
        Return all host architectures with available compilers.

        Returns
        -------
        List[str]
            Sorted list, e.g., ["x64", "x86"].
        """
        return sorted({h for h, _ in self._available_combinations})

    @property
    def available_target_archs(self) -> List[str]:
        """
        Return all target architectures with available compilers.

        Returns
        -------
        List[str]
            Sorted list, e.g., ["arm64", "x64", "x86"].
        """
        return sorted({t for _, t in self._available_combinations})

    @property
    def windows_sdk_version(self) -> Optional[str]:
        """
        Return the detected Windows SDK version.

        Returns
        -------
        Optional[str]
            Version string like "10.0.22621.0", or None.
        """
        return self._windows_sdk_version

    @property
    def windows_sdk_path(self) -> Optional[Path]:
        """
        Return the Windows SDK root path.

        Returns
        -------
        Optional[Path]
            Path to C:/Program Files (x86)/Windows Kits/10.
        """
        return self._windows_sdk_path

    # ==================================================================
    # Internal helpers
    # ==================================================================

    @staticmethod
    def _detect_native_arch() -> str:
        """
        Detect the native host architecture.

        Returns
        -------
        str
            "x64" or "x86".
        """
        machine = _platform.machine().lower()
        if machine in ("x86_64", "amd64", "arm64"):
            return "x64"
        return "x86"

    @staticmethod
    def _parse_arch_from_dir(dirname: str) -> Optional[str]:
        """
        Extract architecture from a directory name like "Hostx64" or "x64".

        Parameters
        ----------
        dirname : str
            Directory name.

        Returns
        -------
        Optional[str]
            "x64", "x86", "arm64", or None if not recognized.
        """
        # Strip "Host" prefix if present
        name = dirname.replace("Host", "", 1).lower()
        if name in ("x64", "x86", "arm64", "arm", "arm64ec"):
            return name
        return None
"""
Emscripten toolchain implementation.

Represents an installed Emscripten SDK (Emscripten Compiler Frontend).
Emscripten compiles C/C++ code to WebAssembly (Wasm) or asm.js
using the `emcc` and `em++` compiler drivers, which wrap Clang
and the Emscripten runtime libraries.

Layout
------
A standard Emscripten SDK installation:
    {emsdk}/
        upstream/
            emscripten/
                emcc                    # C compiler driver (Python/Node)
                em++                    # C++ compiler driver
                emcc.py                 # Python entry point
                emscripten.py           # Core Emscripten logic
                src/                    # Compiler source files
                system/
                    include/             # libc, libc++, SDL, GL headers
                    lib/                 # Precompiled runtime libraries (.a)
            bin/
                clang                   # Bundled Clang
                clang++                 # Bundled Clang++
                llvm-ar                 # Bundled LLVM archiver
                llvm-nm                 # Bundled LLVM nm
                lld                     # Bundled LLVM linker (wasm-ld)
                node                    # Node.js (required by emcc)
            lib/
                clang/{version}/         # Clang resource directory
        node/{version}/                 # Node.js runtime
        python/                         # Bundled Python (optional)

Emscripten can also be installed system-wide via package managers,
where emcc is in PATH and the system directory is elsewhere.

Usage
-----
    from pathlib import Path
    from pyputil_install.compiler_installer.toolchains.emscripten import EmscriptenToolchain

    emsdk = EmscriptenToolchain(Path("/path/to/emsdk"))
    if emsdk.is_valid():
        print(emsdk.version)              # "4.0.8"
        print(emsdk.emcc_path)            # Path to emcc
        print(emsdk.node_path)            # Path to Node.js

        # Compile to WebAssembly
        result = emsdk.compile(
            "source.c",
            output="program.html",
            flags=["-O2", "-s", "EXPORTED_FUNCTIONS=['_main']"],
        )

Warnings
--------
- Emscripten requires Node.js to run. It must be either bundled
  with the SDK or available on PATH.
- The emcc/em++ drivers are Python scripts that invoke Clang.
  They require Python 3.6+ to be installed.
- Compilation always targets WebAssembly (wasm32) by default.
  The target triplet reported is "wasm32-unknown-emscripten".
- Emscripten runtime libraries (.a files) are precompiled for
  wasm32. Rebuilding them requires the full SDK and Python.
- The version reported is the Emscripten version, not the
  bundled Clang/LLVM version.

User Instructions
-----------------
- Pass the emsdk root directory:
    Correct: EmscriptenToolchain(Path("~/emsdk"))
- For system-installed emscripten, use:
    EmscriptenToolchain(Path("/usr"))
- Ensure Node.js is installed and on PATH if not bundled.
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


class EmscriptenToolchain(Toolchain):
    """
    Represents an installed Emscripten SDK compiler toolchain.

    Parameters
    ----------
    path : Path
        Root directory of the Emscripten SDK installation.
    validate : bool
        If True, scans for executables and detects version.
        Default True.

    Attributes
    ----------
    path : Path
        Resolved installation root.
    bin_dir : Path
        The upstream/emscripten/ or bin/ directory.
    kind : ToolchainKind
        Always ToolchainKind.EMSCRIPTEN.
    version : str
        Emscripten version, e.g., "4.0.8".
    target_triplet : str
        Always "wasm32-unknown-emscripten".
    executables : Dict[ToolRole, Path]
        Mapping of roles to discovered executables.

    Properties
    ----------
    emcc_path : Optional[Path]
        Path to emcc.
    empp_path : Optional[Path]
        Path to em++.
    node_path : Optional[Path]
        Path to Node.js.
    clang_path : Optional[Path]
        Path to the bundled Clang.
    system_include_dir : Optional[Path]
        Path to Emscripten system headers.
    system_lib_dir : Optional[Path]
        Path to Emscripten precompiled libraries.
    cache_dir : Optional[Path]
        Path to the Emscripten cache directory.
    """

    def __init__(self, path: Path, validate: bool = True) -> None:
        self.kind = ToolchainKind.EMSCRIPTEN
        self._emcc_path: Optional[Path] = None
        self._empp_path: Optional[Path] = None
        self._node_path: Optional[Path] = None
        self._clang_path: Optional[Path] = None
        self._system_include: Optional[Path] = None
        self._system_lib: Optional[Path] = None
        self._cache_dir: Optional[Path] = None
        super().__init__(path, validate)

    # ==================================================================
    # Executable discovery
    # ==================================================================

    def _find_executables(self) -> None:
        """
        Scan for emcc, em++, Node.js, and bundled Clang.

        Checks the emsdk upstream/emscripten/ directory first,
        then falls back to searching PATH.
        """
        emscripten_candidates = [
            self.path / "upstream" / "emscripten",
            self.path,
            self.path / "share" / "emscripten",
        ]

        emscripten_dir = None
        for candidate in emscripten_candidates:
            if (candidate / "emcc").is_file() or (candidate / "emcc.py").is_file():
                emscripten_dir = candidate
                break

        if emscripten_dir is None:
            # Check if emcc is in PATH
            emcc_in_path = self._which("emcc")
            if emcc_in_path:
                emscripten_dir = emcc_in_path.parent
            else:
                logger.warning("emcc not found in %s", self.path)
                return

        self.bin_dir = emscripten_dir
        self._emcc_path = self._find_file(emscripten_dir, "emcc") or self._find_file(emscripten_dir, "emcc.py")
        self._empp_path = self._find_file(emscripten_dir, "em++") or self._find_file(emscripten_dir, "emcc")

        # Map core roles
        if self._emcc_path:
            self.executables[ToolRole.C_COMPILER] = self._emcc_path
        if self._empp_path:
            self.executables[ToolRole.CXX_COMPILER] = self._empp_path

        # Find Node.js
        node_candidates = [
            self.path / "node" / "current" / "bin" / "node",
            self.path / "node" / "current" / "bin" / "node.exe",
            self.path / "upstream" / "bin" / "node",
            self.path / "upstream" / "bin" / "node.exe",
        ]
        for node_candidate in node_candidates:
            if node_candidate.is_file():
                self._node_path = node_candidate
                break
        if self._node_path is None:
            # Check PATH
            node_in_path = self._which("node")
            if node_in_path:
                self._node_path = node_in_path

        # Find bundled Clang
        upstream_bin = self.path / "upstream" / "bin"
        if upstream_bin.is_dir():
            clang = self._find_file(upstream_bin, "clang")
            if clang:
                self._clang_path = clang

        # System directories
        system_dir = emscripten_dir / "system"
        if system_dir.is_dir():
            if (system_dir / "include").is_dir():
                self._system_include = system_dir / "include"
            if (system_dir / "lib").is_dir():
                self._system_lib = system_dir / "lib"

        # Cache directory
        cache_dir = self.path / ".emscripten_cache"
        if cache_dir.is_dir():
            self._cache_dir = cache_dir
        else:
            home_cache = Path.home() / ".emscripten_cache"
            if home_cache.is_dir():
                self._cache_dir = home_cache

        logger.debug("Found Emscripten at %s", emscripten_dir)

    # ==================================================================
    # Version and target
    # ==================================================================

    def _get_version(self) -> str:
        """
        Detect Emscripten version from `emcc --version`.

        Output format:
            emcc (Emscripten gcc/clang-like replacement + linker ...) 4.0.8
            ...
            clang version 20.0.0

        Returns
        -------
        str
            Emscripten version string.
        """
        emcc = self._emcc_path
        if emcc is None:
            return "0.0.0"

        try:
            result = subprocess.run(
                [str(emcc), "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
            if result.returncode != 0:
                return "0.0.0"

            first_line = result.stdout.splitlines()[0] if result.stdout else ""
            match = re.search(r"\)\s+(\d+\.\d+\.\d+)", first_line)
            if match:
                return match.group(1)
            match = re.search(r"(\d+\.\d+\.\d+)", first_line)
            if match:
                return match.group(1)
        except Exception as exc:
            logger.debug("Emscripten version detection failed: %s", exc)

        return "0.0.0"

    def _get_target_triplet(self) -> str:
        """Return the fixed target triplet for Emscripten."""
        return "wasm32-unknown-emscripten"

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def emcc_path(self) -> Optional[Path]:
        """Return path to emcc."""
        return self._emcc_path

    @property
    def empp_path(self) -> Optional[Path]:
        """Return path to em++."""
        return self._empp_path

    @property
    def node_path(self) -> Optional[Path]:
        """Return path to Node.js."""
        return self._node_path

    @property
    def clang_path(self) -> Optional[Path]:
        """Return path to the bundled Clang."""
        return self._clang_path

    @property
    def system_include_dir(self) -> Optional[Path]:
        """Return path to Emscripten system headers."""
        return self._system_include

    @property
    def system_lib_dir(self) -> Optional[Path]:
        """Return path to Emscripten runtime libraries."""
        return self._system_lib

    @property
    def cache_dir(self) -> Optional[Path]:
        """Return path to the Emscripten cache directory."""
        return self._cache_dir

    # ==================================================================
    # Compile override
    # ==================================================================

    def compile(
        self,
        source: str,
        output: Optional[str] = None,
        flags: Optional[List[str]] = None,
        language: Optional[str] = None,
        timeout: int = 300,
    ) -> CompileResult:
        """
        Compile using emcc or em++.

        Parameters
        ----------
        source : str
            Source file path.
        output : Optional[str]
            Output file path (usually .html, .js, or .wasm).
        flags : Optional[List[str]]
            Additional Emscripten flags.
        language : Optional[str]
            "c" or "c++". Auto-detected from extension if None.
        timeout : int
            Seconds before timeout (default 300 for Emscripten).
        """
        if not self.is_valid():
            raise RuntimeError(f"Emscripten toolchain at {self.path} is not valid")

        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Source not found: {source}")

        # Use em++ for C++, emcc for C
        if language == "c++" or source_path.suffix in (".cpp", ".cxx", ".cc", ".C"):
            compiler = self._empp_path or self._emcc_path
        else:
            compiler = self._emcc_path

        if compiler is None:
            raise RuntimeError("No Emscripten compiler found")

        cmd = [str(compiler)]
        if flags:
            cmd.extend(flags)
        else:
            cmd.append("-O2")

        cmd.append(str(source_path))

        if output:
            cmd.extend(["-o", output])
            output_file = Path(output)
        else:
            output_file = source_path.with_suffix(".html")
            cmd.extend(["-o", str(output_file)])

        import time
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, shell=False
            )
        except subprocess.TimeoutExpired:
            return CompileResult(
                returncode=-1,
                stderr=f"Timeout after {timeout}s",
                command=cmd,
            )

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
        """Find a file in a directory, with optional .exe on Windows."""
        candidate = directory / name
        if candidate.is_file():
            return candidate
        if os.name == "nt":
            candidate = directory / f"{name}.exe"
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _which(name: str) -> Optional[Path]:
        """Find an executable on PATH."""
        import shutil
        result = shutil.which(name)
        return Path(result) if result else None
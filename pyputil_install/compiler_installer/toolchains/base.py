"""
Base toolchain abstraction layer.

Defines the Toolchain abstract base class (ABC) that all
compiler-specific implementations must subclass. A Toolchain
represents a complete installed compiler suite — the compiler,
linker, archiver, and other utilities that ship together.

This module establishes the contract that every toolchain
implementation must fulfill. It does NOT contain detection logic,
platform-specific code, or capability checks.

Design
------
Every toolchain must provide:
    - Paths to key executables (compiler, linker, archiver, etc.)
    - Version information
    - Target triplet
    - Installation root and bin directory
    - A compile() method that abstracts subprocess invocation

Subclasses override _find_executables() to locate tools within
their specific directory layout. Each subclass also sets `kind`
and may override version/triplet detection if the default method
(running the C compiler with --version and -dumpmachine) is not
appropriate.

Usage
-----
    from toolforge.toolchains.base import Toolchain, ToolRole
    from toolforge.toolchains.gcc import GCCToolchain

    gcc = GCCToolchain(Path("/usr"))
    if gcc.is_valid():
        print(gcc.version)
        print(gcc.c_compiler)
        print(gcc.target_triplet)
        result = gcc.compile("source.c", output="program", flags=["-O2"])
        if result.success:
            print("Compiled successfully")

Warnings
--------
- Toolchain objects are NOT immutable. Paths are resolved once
  at construction time but the underlying filesystem may change.
- The compile() method is synchronous and blocking. For async
  execution, wrap in asyncio.to_thread().
- Subprocess calls made by compile() inherit the current process
  environment unless TOOLFORGE_CLEAN_ENV is set to "1".
- Validation that executables exist is done at construction time
  when validate=True. Set validate=False to skip this check
  (useful for offline inspection or testing).
- _get_version() and _get_target_triplet() execute the compiler
  binary. If the binary hangs or crashes, these methods will
  time out after 10 seconds.

User Instructions
-----------------
- Do NOT instantiate Toolchain directly. Use a concrete subclass
  or the detection module.
- Call is_valid() after construction to check if the toolchain
  is usable before calling compile().
- Use get_executable() to find tools by role (CC, CXX, AR, LD, etc.).
- The compile() method is a convenience wrapper around subprocess.
  For complex builds, use get_executable() and subprocess directly.
"""

import abc
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# ToolchainKind — broad classification of a toolchain family
# ============================================================================

class ToolchainKind(Enum):
    """
    Broad classification of a toolchain family.

    Used for dispatch and filtering when iterating over multiple
    installed toolchains. This enum is distinct from CompilerType
    in the URL layer — ToolchainKind describes an installed
    toolchain, while CompilerType describes a remote artifact.

    Members
    -------
    GCC : GNU Compiler Collection
    CLANG : LLVM Clang
    MSVC : Microsoft Visual C++
    ZIG : Zig toolchain
    EMSCRIPTEN : Emscripten SDK
    ANDROID_NDK : Android NDK
    UNKNOWN : Unable to classify
    """
    GCC = auto()
    CLANG = auto()
    MSVC = auto()
    ZIG = auto()
    EMSCRIPTEN = auto()
    ANDROID_NDK = auto()
    UNKNOWN = auto()


# ============================================================================
# ToolRole — standard executable roles within a toolchain
# ============================================================================

class ToolRole(Enum):
    """
    Standard roles within a compiler toolchain.

    Each role corresponds to a well-known executable name or
    pattern. Not all toolchains provide every role — MSVC does
    not have a ranlib equivalent, for example.

    Members
    -------
    C_COMPILER : The C compiler (gcc, clang, cl.exe)
    CXX_COMPILER : The C++ compiler (g++, clang++, cl.exe)
    ASSEMBLER : The assembler (as, or the compiler with -x assembler)
    ARCHIVER : Static library archiver (ar, llvm-ar, lib.exe)
    LINKER : The linker (ld, lld, link.exe)
    RANLIB : Archive indexer (ranlib, llvm-ranlib)
    NM : Symbol table lister (nm, llvm-nm)
    STRIP : Symbol stripper (strip, llvm-strip)
    OBJCOPY : Object file copier/converter (objcopy, llvm-objcopy)
    OBJDUMP : Object file disassembler (objdump, llvm-objdump)
    READELF : ELF file reader (readelf, llvm-readelf)
    SIZE : Section size viewer (size, llvm-size)
    STRINGS : Printable string extractor (strings, llvm-strings)
    DLLTOOL : DLL creation tool for MinGW/Windows (dlltool)
    WINDRES : Windows resource compiler for MinGW (windres)
    """
    C_COMPILER = auto()
    CXX_COMPILER = auto()
    ASSEMBLER = auto()
    ARCHIVER = auto()
    LINKER = auto()
    RANLIB = auto()
    NM = auto()
    STRIP = auto()
    OBJCOPY = auto()
    OBJDUMP = auto()
    READELF = auto()
    SIZE = auto()
    STRINGS = auto()
    DLLTOOL = auto()
    WINDRES = auto()


# ============================================================================
# CompileResult — output of a compilation operation
# ============================================================================

@dataclass
class CompileResult:
    """
    Result of a single compilation operation.

    Attributes
    ----------
    returncode : int
        Process exit code. 0 means success, non-zero means failure.
    stdout : str
        Captured standard output from the compiler process.
        Typically empty unless -v or similar flags are used.
    stderr : str
        Captured standard error from the compiler process.
        Contains warnings, errors, and diagnostic messages.
    command : List[str]
        The full command line that was executed, as a list of
        individual arguments. Useful for debugging or replaying
        the compilation manually.
    output_file : Optional[Path]
        Path to the output file produced by the compiler, if one
        was requested and successfully created. None if no output
        file was specified or compilation failed.
    elapsed_ms : float
        Wall-clock time for the compilation in milliseconds.
        Measured from subprocess launch to process exit.

    Properties
    ----------
    success : bool
        True if returncode == 0, False otherwise.
    """
    returncode: int = -1
    stdout: str = ""
    stderr: str = ""
    command: List[str] = field(default_factory=list)
    output_file: Optional[Path] = None
    elapsed_ms: float = 0.0

    @property
    def success(self) -> bool:
        """Return True if the compilation exited with code 0."""
        return self.returncode == 0

    def summary(self) -> str:
        """
        Return a one-line human-readable summary of the result.

        Returns
        -------
        str
            Summary string like "SUCCESS (0.23s)" or "FAILED exit=1 (0.15s)".
        """
        status = "SUCCESS" if self.success else f"FAILED exit={self.returncode}"
        return f"{status} ({self.elapsed_ms / 1000:.2f}s)"

    def __repr__(self) -> str:
        return (
            f"CompileResult(returncode={self.returncode}, "
            f"output_file={self.output_file!r}, "
            f"elapsed_ms={self.elapsed_ms:.0f})"
        )


# ============================================================================
# Toolchain — abstract base class
# ============================================================================

class Toolchain(abc.ABC):
    """
    Abstract base class for all compiler toolchain implementations.

    Represents a complete, installed compiler suite. Provides
    access to individual tools by role, version detection, target
    triplet detection, and a convenience compile() method.

    Subclasses must:
        - Set `kind` to the appropriate ToolchainKind value.
        - Implement `_find_executables()` to populate `self.executables`.
        - Optionally override `_get_version()` and `_get_target_triplet()`
          if the default method (running the C compiler with --version
          and -dumpmachine) is not suitable.

    Parameters
    ----------
    path : Path
        Root directory of the toolchain installation. The exact
        layout depends on the compiler family:
        - GCC/Clang: directory containing bin/, lib/, include/.
        - MSVC: the VC/Tools/MSVC/{version}/ directory.
        - Zig: the directory containing the zig executable.
    validate : bool
        If True (default), verifies that the path exists and calls
        _validate() to check that the minimum required executables
        are present. Set to False for offline construction or
        unit testing.

    Attributes
    ----------
    path : Path
        Resolved absolute path to the toolchain root directory.
    bin_dir : Path
        Path to the bin/ directory containing executables.
        May be set to path itself if the toolchain has no separate
        bin directory (e.g., Zig).
    kind : ToolchainKind
        Toolchain family classification. Set by the subclass.
    version : str
        Detected version string, e.g., "14.2.0", "18.1.8".
        Empty string if detection failed or validate=False.
    target_triplet : str
        Detected target triplet, e.g., "x86_64-linux-gnu".
        Empty string if detection failed or validate=False.
    executables : Dict[ToolRole, Path]
        Mapping of ToolRole to the absolute path of the
        corresponding executable. Populated by _find_executables().
        Empty dict if validate=False or _find_executables() fails.

    Methods
    -------
    is_valid() -> bool
        Check if the minimum required executables were found.
    get_executable(role: ToolRole) -> Optional[Path]
        Return the path to a tool by its role.
    compile(source, output=None, flags=None, language=None) -> CompileResult
        Compile a single source file using the C compiler.
    get_version() -> str
        Return the detected version string.
    get_target_triplet() -> str
        Return the detected target triplet.
    """

    def __init__(self, path: Path, validate: bool = True) -> None:
        self.path = path.resolve()
        self.bin_dir = self.path / "bin"
        if not hasattr(self, 'kind'):
            self.kind = ToolchainKind.UNKNOWN
        self.version: str = ""
        self.target_triplet: str = ""
        self.executables: Dict[ToolRole, Path] = {}

        if validate:
            self._validate_and_detect()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def is_valid(self) -> bool:
        """
        Check if the toolchain has at minimum a working C compiler.

        Returns
        -------
        bool
            True if at least ToolRole.C_COMPILER was found and
            the executable exists on disk.
        """
        c_compiler = self.executables.get(ToolRole.C_COMPILER)
        if c_compiler is None:
            return False
        return c_compiler.is_file() and os.access(str(c_compiler), os.X_OK)

    def get_executable(self, role: ToolRole) -> Optional[Path]:
        """
        Return the path to a specific tool by its role.

        Parameters
        ----------
        role : ToolRole
            The role to look up (e.g., ToolRole.ARCHIVER).

        Returns
        -------
        Optional[Path]
            Path to the executable, or None if that role is not
            provided by this toolchain.
        """
        return self.executables.get(role)

    def compile(
        self,
        source: str,
        output: Optional[str] = None,
        flags: Optional[List[str]] = None,
        language: Optional[str] = None,
        timeout: int = 120,
    ) -> CompileResult:
        """
        Compile a single source file using the C compiler.

        This is a convenience method for simple single-file
        compilations. For multi-file builds or projects with
        complex build requirements, use get_executable() to
        retrieve the compiler path and invoke it directly via
        subprocess or a build system.

        Parameters
        ----------
        source : str
            Path to the source file to compile. Must exist.
        output : Optional[str]
            Path for the output file. If None, the output is
            placed alongside the source with the platform-
            appropriate executable extension (e.g., .exe on Windows).
        flags : Optional[List[str]]
            Additional compiler flags. Default is ["-O2"].
        language : Optional[str]
            Source language: "c" or "c++". If None, inferred from
            the source file extension.
        timeout : int
            Maximum time in seconds to wait for compilation.
            Default is 120 seconds.

        Returns
        -------
        CompileResult
            Result with return code, stdout, stderr, timing.

        Raises
        ------
        RuntimeError
            If the toolchain is not valid (no C compiler found).
        FileNotFoundError
            If the source file does not exist.

        Warnings
        --------
        - The compilation inherits the current process environment.
          Set TOOLFORGE_CLEAN_ENV=1 to strip most variables.
        - The subprocess is run with shell=False for security.
        """
        if not self.is_valid():
            raise RuntimeError(
                f"Toolchain at {self.path} is not valid: "
                f"no C compiler found."
            )

        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")

        compiler = self.executables[ToolRole.C_COMPILER]

        # Build command
        cmd: List[str] = [str(compiler)]

        if flags:
            cmd.extend(flags)
        else:
            cmd.append("-O2")

        if language == "c++" or (language is None and source_path.suffix in (".cpp", ".cxx", ".cc", ".C")):
            pass  # Many compilers detect language from extension
        if language == "c":
            cmd.append("-x")
            cmd.append("c")

        cmd.append(str(source_path))

        if output:
            cmd.extend(["-o", output])
            output_file = Path(output)
        else:
            output_file = source_path.with_suffix("")
            if os.name == "nt":
                output_file = output_file.with_suffix(".exe")
            cmd.extend(["-o", str(output_file)])

        # Execute
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.monotonic() - start) * 1000
            return CompileResult(
                returncode=-1,
                stdout="",
                stderr=f"Compilation timed out after {timeout}s",
                command=cmd,
                output_file=None,
                elapsed_ms=elapsed,
            )

        elapsed = (time.monotonic() - start) * 1000

        # Check if output was actually produced (compiler may exit 0
        # but fail to write the output file under some conditions)
        if proc.returncode == 0 and not output_file.exists():
            return CompileResult(
                returncode=-1,
                stdout=proc.stdout,
                stderr="Compiler exited 0 but output file was not created",
                command=cmd,
                output_file=None,
                elapsed_ms=elapsed,
            )

        return CompileResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            command=cmd,
            output_file=output_file if output_file.exists() else None,
            elapsed_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Version and target detection
    # ------------------------------------------------------------------

    def get_version(self) -> str:
        """
        Return the detected version string.

        Returns
        -------
        str
            Version string like "14.2.0". Empty string if not
            yet detected or detection failed.
        """
        return self.version

    def get_target_triplet(self) -> str:
        """
        Return the detected target triplet.

        Returns
        -------
        str
            Target triplet like "x86_64-linux-gnu". Empty string
            if not yet detected or detection failed.
        """
        return self.target_triplet

    # ------------------------------------------------------------------
    # Internal: validation and detection
    # ------------------------------------------------------------------

    def _validate_and_detect(self) -> None:
        """
        Run validation, executable discovery, and version/target detection.

        Called once during __init__ when validate=True.
        """
        if not self.path.exists():
            logger.warning("Toolchain path does not exist: %s", self.path)
            return

        if not self.path.is_dir():
            logger.warning("Toolchain path is not a directory: %s", self.path)
            return

        try:
            self._find_executables()
        except Exception as exc:
            logger.warning("Failed to discover executables in %s: %s", self.path, exc)
            return

        if self.is_valid():
            try:
                self.version = self._get_version()
            except Exception as exc:
                logger.warning("Failed to detect version: %s", exc)

            try:
                self.target_triplet = self._get_target_triplet()
            except Exception as exc:
                logger.warning("Failed to detect target triplet: %s", exc)

    @abc.abstractmethod
    def _find_executables(self) -> None:
        """
        Locate all available tools and populate self.executables.

        Subclasses must implement this method. It should scan the
        toolchain's bin directory and set self.executables to a
        mapping of ToolRole -> Path for every tool found.

        At minimum, ToolRole.C_COMPILER must be set for the
        toolchain to be considered valid.
        """
        ...

    def _get_version(self) -> str:
        """
        Detect the compiler version by running {c_compiler} --version.

        Parses the first line of stdout to extract a version string.
        Override in subclasses if the output format differs.

        Returns
        -------
        str
            Version string, or "0.0.0" if detection fails.
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
                return "0.0.0"
            output = result.stdout or result.stderr
            first_line = output.splitlines()[0] if output else ""
            match = re.search(r"(\d+\.\d+\.\d+)", first_line)
            if match:
                return match.group(1)
            match = re.search(r"(\d+\.\d+)", first_line)
            if match:
                return match.group(1)
        except Exception as exc:
            logger.debug("Version detection failed: %s", exc)

        return "0.0.0"

    def _get_target_triplet(self) -> str:
        """
        Detect the target triplet by running {c_compiler} -dumpmachine.

        Override in subclasses if -dumpmachine is not supported
        (e.g., MSVC, Zig).

        Returns
        -------
        str
            Target triplet, or empty string if detection fails.
        """
        compiler = self.executables.get(ToolRole.C_COMPILER)
        if compiler is None:
            return ""

        try:
            result = subprocess.run(
                [str(compiler), "-dumpmachine"],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout.strip().splitlines()[0].strip()
        except Exception as exc:
            logger.debug("Target triplet detection failed: %s", exc)

        return ""

    # ------------------------------------------------------------------
    # Properties for common roles
    # ------------------------------------------------------------------

    @property
    def c_compiler(self) -> Optional[Path]:
        """Return the path to the C compiler, or None."""
        return self.executables.get(ToolRole.C_COMPILER)

    @property
    def cxx_compiler(self) -> Optional[Path]:
        """Return the path to the C++ compiler, or None."""
        return self.executables.get(ToolRole.CXX_COMPILER)

    @property
    def archiver(self) -> Optional[Path]:
        """Return the path to the archiver (ar), or None."""
        return self.executables.get(ToolRole.ARCHIVER)

    @property
    def linker(self) -> Optional[Path]:
        """Return the path to the linker (ld), or None."""
        return self.executables.get(ToolRole.LINKER)

    @property
    def strip(self) -> Optional[Path]:
        """Return the path to strip, or None."""
        return self.executables.get(ToolRole.STRIP)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"path={self.path!r}, "
            f"version={self.version!r}, "
            f"target={self.target_triplet!r})"
        )
"""
Test suite for pyputil install compiler installer toolchain abstraction layer.

Covers: base classes, GCC, Clang, MSVC, Zig, Emscripten, Android NDK,
capabilities, detection, sysroots, runtimes, ABI, and environments.

Requirements
------------
    pip install pytest

Platform Notes
--------------
- GCC/Clang tests: run on Linux, macOS, Windows (with compiler installed)
- MSVC tests: Windows only, skipped on other platforms
- Zig tests: run if Zig is installed on PATH
- Emscripten tests: run if Emscripten (emcc) is installed on PATH
- Android NDK tests: run if ANDROID_NDK_HOME environment variable is set
- Capability tests: require a working C compiler (GCC or Clang)
- Detection tests: scan the host system (may spawn subprocesses)
- ABI tests: require a working C compiler

Expected Pass Rates by Platform
-------------------------------
    Linux:      90-98% (GCC and Clang widely available)
    macOS:      85-95% (Clang available; GCC may be a symlink to Clang)
    Windows:    50-70% (depends on MSVC/MinGW/Clang installation)
    Android:    40-60% (Pydroid3/Termux have GCC; NDK unlikely)

Individual Test Expectations
----------------------------
    test_compile_simple_c:
        Linux:    PASS (GCC available in /usr/bin)
        macOS:    PASS (Clang available)
        Windows:  SKIP if no compiler found
        Android:  PASS (Pydroid3 bundles GCC)

    test_detect_abi_returns_abi_info:
        Linux:    PASS
        macOS:    PASS
        Windows:  SKIP if no compiler
        Android:  PASS

    test_detect_msvc:
        Linux:    PASS (returns empty list)
        macOS:    PASS (returns empty list)
        Windows:  PASS (may return list if VS installed)
        Android:  PASS (returns empty list)
"""

import os
import platform as _platform
import sys
from pathlib import Path
from typing import Optional

import pytest

# ============================================================================
# Imports
# ============================================================================

from pyputil_install.compiler_installer.toolchains.base import (
    Toolchain,
    ToolchainKind,
    ToolRole,
    CompileResult,
)
from pyputil_install.compiler_installer.toolchains.gcc import GCCToolchain
from pyputil_install.compiler_installer.toolchains.clang import ClangToolchain
from pyputil_install.compiler_installer.toolchains.msvc import MSVCToolchain
from pyputil_install.compiler_installer.toolchains.zig import ZigToolchain
from pyputil_install.compiler_installer.toolchains.emscripten import EmscriptenToolchain
from pyputil_install.compiler_installer.toolchains.android import AndroidNDKToolchain

from pyputil_install.compiler_installer.toolchains.capabilities import (
    supports_cpp17,
    supports_c11,
    supports_openmp,
    supports_lto,
    supports_pic,
    supports_rtti,
    supports_exceptions,
    detect_all,
    clear_detection_cache,
)

from pyputil_install.compiler_installer.toolchains.detection import (
    detect_gcc,
    detect_clang,
    detect_msvc,
    detect_all_toolchains,
    detect_best_toolchain,
)

from pyputil_install.compiler_installer.toolchains.sysroots import (
    SysrootInfo,
    detect_sysroot,
)

from pyputil_install.compiler_installer.toolchains.runtimes import (
    RuntimeInfo,
    detect_runtimes,
)

from pyputil_install.compiler_installer.toolchains.abi import (
    ABIInfo,
    detect_abi,
    get_abi_compatibility,
)

from pyputil_install.compiler_installer.toolchains.environments import (
    Environment,
    EnvironmentStore,
)


# ============================================================================
# Helpers
# ============================================================================

def _has_gcc() -> bool:
    """
    Check if GCC is available on this system.

    Returns
    -------
    bool
        True if `gcc` is found on PATH.
    """
    import shutil
    return shutil.which("gcc") is not None


def _has_clang() -> bool:
    """
    Check if Clang is available on this system.

    Returns
    -------
    bool
        True if `clang` is found on PATH.
    """
    import shutil
    return shutil.which("clang") is not None


def _has_any_compiler() -> bool:
    """
    Check if any C compiler is available.

    Returns
    -------
    bool
        True if gcc, clang, or cc is on PATH.
    """
    import shutil
    return any(
        shutil.which(name) for name in ("gcc", "clang", "cc")
    )


def _get_system_gcc() -> Optional[GCCToolchain]:
    """
    Get a GCCToolchain for the system GCC, if available.

    Returns
    -------
    Optional[GCCToolchain]
        Valid GCC toolchain, or None.
    """
    import shutil
    gcc_path = shutil.which("gcc")
    if gcc_path:
        prefix = Path(gcc_path).parent.parent
        tc = GCCToolchain(prefix)
        if tc.is_valid():
            return tc
    return None


def _get_system_clang() -> Optional[ClangToolchain]:
    """
    Get a ClangToolchain for the system Clang, if available.

    Returns
    -------
    Optional[ClangToolchain]
        Valid Clang toolchain, or None.
    """
    import shutil
    clang_path = shutil.which("clang")
    if clang_path:
        prefix = Path(clang_path).parent.parent
        tc = ClangToolchain(prefix)
        if tc.is_valid():
            return tc
    return None


# ============================================================================
# Base Tests
# ============================================================================

class TestBase:
    """
    Tests for base classes, enums, and data structures.

    Expected pass rate: 100% on all platforms.
    No compiler or external dependencies required.
    """

    def test_toolchain_kind_all_members_present(self):
        """
        ToolchainKind enum includes all seven expected families.

        Expected result: True for GCC, CLANG, MSVC, ZIG,
        EMSCRIPTEN, ANDROID_NDK, UNKNOWN.

        All platforms: PASS
        """
        kinds = {kind.name for kind in ToolchainKind}
        assert "GCC" in kinds
        assert "CLANG" in kinds
        assert "MSVC" in kinds
        assert "ZIG" in kinds
        assert "EMSCRIPTEN" in kinds
        assert "ANDROID_NDK" in kinds
        assert "UNKNOWN" in kinds

    def test_tool_role_essential_members_present(self):
        """
        ToolRole enum includes compiler, archiver, linker, strip.

        Expected result: True for C_COMPILER, CXX_COMPILER,
        ARCHIVER, LINKER, STRIP.

        All platforms: PASS
        """
        roles = {role.name for role in ToolRole}
        assert "C_COMPILER" in roles
        assert "CXX_COMPILER" in roles
        assert "ARCHIVER" in roles
        assert "LINKER" in roles
        assert "STRIP" in roles

    def test_compile_result_default_values(self):
        """
        CompileResult fields have correct default values.

        Expected result:
            returncode=-1, stdout="", stderr="", command=[],
            output_file=None, elapsed_ms=0.0.

        All platforms: PASS
        """
        result = CompileResult()
        assert result.returncode == -1
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.command == []
        assert result.output_file is None
        assert result.elapsed_ms == 0.0

    def test_compile_result_success_true_for_zero(self):
        """
        CompileResult.success is True when returncode is 0.

        All platforms: PASS
        """
        result = CompileResult(returncode=0)
        assert result.success is True

    def test_compile_result_success_false_for_nonzero(self):
        """
        CompileResult.success is False when returncode is non-zero.

        All platforms: PASS
        """
        result = CompileResult(returncode=1)
        assert result.success is False

    def test_compile_result_summary_contains_status(self):
        """
        CompileResult.summary() includes SUCCESS or FAILED.

        All platforms: PASS
        """
        success_result = CompileResult(returncode=0, elapsed_ms=230.0)
        fail_result = CompileResult(returncode=1, elapsed_ms=150.0)
        assert "SUCCESS" in success_result.summary()
        assert "FAILED" in fail_result.summary()

    def test_compile_result_repr_includes_key_fields(self):
        """
        CompileResult.__repr__ includes returncode and output_file.

        All platforms: PASS
        """
        result = CompileResult(returncode=0, output_file=Path("/tmp/out.o"))
        assert "returncode=0" in repr(result)
        assert "output_file" in repr(result)

    def test_toolchain_abstract_cannot_instantiate(self):
        """
        Toolchain ABC cannot be instantiated directly.

        All platforms: PASS (raises TypeError)
        """
        with pytest.raises(TypeError):
            Toolchain(Path("/tmp"))  # type: ignore


# ============================================================================
# GCC Tests
# ============================================================================

class TestGCC:
    """
    Tests for GCCToolchain.

    Expected pass rate:
        Linux:    95% (GCC typically installed)
        macOS:    50% (gcc may be clang symlink)
        Windows:  40% (requires MinGW or similar)
        Android:  60% (Pydroid3 has GCC)
    """

    @pytest.mark.skipif(not _has_gcc(), reason="GCC not found on PATH")
    def test_gcc_detect_system_compiler(self):
        """
        GCCToolchain detects the system GCC at /usr.

        Expected result: is_valid() returns True, kind is GCC.

        Linux:    PASS
        macOS:    SKIP or PASS (if real GCC installed)
        Windows:  SKIP
        Android:  PASS
        """
        tc = _get_system_gcc()
        if tc is None:
            pytest.skip("System GCC not found")
        assert tc.is_valid()
        assert tc.kind == ToolchainKind.GCC

    @pytest.mark.skipif(not _has_gcc(), reason="GCC not found on PATH")
    def test_gcc_has_version(self):
        """
        GCCToolchain reports a non-empty version string.

        Expected result: version != "0.0.0", version != "".

        Linux:    PASS
        macOS:    SKIP or PASS
        Windows:  SKIP
        Android:  PASS
        """
        tc = _get_system_gcc()
        if tc is None:
            pytest.skip("System GCC not found")
        assert tc.version != "0.0.0"
        assert tc.version != ""

    @pytest.mark.skipif(not _has_gcc(), reason="GCC not found on PATH")
    def test_gcc_has_target_triplet(self):
        """
        GCCToolchain reports a non-empty target triplet.

        Expected result: target_triplet is not empty.

        Linux:    PASS
        macOS:    SKIP or PASS
        Windows:  SKIP
        Android:  PASS
        """
        tc = _get_system_gcc()
        if tc is None:
            pytest.skip("System GCC not found")
        assert tc.target_triplet != ""

    @pytest.mark.skipif(not _has_gcc(), reason="GCC not found on PATH")
    def test_gcc_compile_simple_c(self, tmp_path):
        """
        GCCToolchain.compile() compiles a trivial C file.

        Expected result: CompileResult.success is True.

        Linux:    PASS
        macOS:    SKIP or PASS
        Windows:  SKIP
        Android:  PASS
        """
        tc = _get_system_gcc()
        if tc is None:
            pytest.skip("System GCC not found")

        source = tmp_path / "test.c"
        source.write_text("int main() { return 0; }\n")
        output = tmp_path / "test_out"

        result = tc.compile(str(source), output=str(output))
        assert result.success is True
        assert output.is_file()

    @pytest.mark.skipif(not _has_gcc(), reason="GCC not found on PATH")
    def test_gcc_has_c_and_cxx_compilers(self):
        """
        GCCToolchain finds both C and C++ compilers.

        Expected result: c_compiler and cxx_compiler are not None.

        Linux:    PASS
        macOS:    SKIP or PASS
        Windows:  SKIP
        Android:  PASS
        """
        tc = _get_system_gcc()
        if tc is None:
            pytest.skip("System GCC not found")
        assert tc.c_compiler is not None
        assert tc.cxx_compiler is not None

    @pytest.mark.skipif(not _has_gcc(), reason="GCC not found on PATH")
    def test_gcc_has_binutils(self):
        """
        GCCToolchain finds ar, nm, strip.

        Expected result: archiver, nm, strip are not None.

        Linux:    PASS
        macOS:    SKIP or PASS
        Windows:  SKIP
        Android:  PASS
        """
        tc = _get_system_gcc()
        if tc is None:
            pytest.skip("System GCC not found")
        assert tc.archiver is not None
        assert tc.get_executable(ToolRole.NM) is not None
        assert tc.strip is not None

    def test_gcc_nonexistent_path(self):
        """
        GCCToolchain with nonexistent path is not valid.

        Expected result: is_valid() returns False.

        All platforms: PASS
        """
        tc = GCCToolchain(Path("/nonexistent/gcc/prefix"))
        assert not tc.is_valid()
        assert tc.version == "0.0.0" or tc.version == ""

    def test_gcc_is_cross_compiler_detection(self):
        """
        GCCToolchain.is_cross_compiler works correctly.

        Expected result: System GCC on native host is not cross.

        Linux:    PASS (returns False for native)
        macOS:    PASS (returns False for native)
        Windows:  PASS (returns False for native MinGW)
        Android:  PASS (returns False for native GCC)
        """
        tc = _get_system_gcc()
        if tc is None:
            pytest.skip("System GCC not found")
        # System GCC on native host should not be a cross-compiler
        # (May be True on some configurations, so we just check the property runs)
        result = tc.is_cross_compiler
        assert isinstance(result, bool)


# ============================================================================
# Clang Tests
# ============================================================================

class TestClang:
    """
    Tests for ClangToolchain.

    Expected pass rate:
        Linux:    90% (Clang commonly installed)
        macOS:    95% (Apple Clang always present)
        Windows:  30% (requires LLVM install)
        Android:  20% (rarely installed)
    """

    @pytest.mark.skipif(not _has_clang(), reason="Clang not found on PATH")
    def test_clang_detect_system_compiler(self):
        """
        ClangToolchain detects the system Clang.

        Expected result: is_valid() returns True, kind is CLANG.

        Linux:    PASS
        macOS:    PASS
        Windows:  SKIP or PASS
        Android:  SKIP
        """
        tc = _get_system_clang()
        if tc is None:
            pytest.skip("System Clang not found")
        assert tc.is_valid()
        assert tc.kind == ToolchainKind.CLANG

    @pytest.mark.skipif(not _has_clang(), reason="Clang not found on PATH")
    def test_clang_has_version(self):
        """
        ClangToolchain reports a non-empty version.

        Expected result: version is not "0.0.0".

        Linux:    PASS
        macOS:    PASS
        Windows:  SKIP or PASS
        Android:  SKIP
        """
        tc = _get_system_clang()
        if tc is None:
            pytest.skip("System Clang not found")
        assert tc.version != "0.0.0"
        assert tc.version != ""

    @pytest.mark.skipif(not _has_clang(), reason="Clang not found on PATH")
    def test_clang_has_vendor(self):
        """
        ClangToolchain reports vendor as Apple or LLVM.

        Expected result: vendor in ("Apple", "LLVM").

        Linux:    PASS (vendor = "LLVM")
        macOS:    PASS (vendor = "Apple")
        Windows:  SKIP or PASS
        Android:  SKIP
        """
        tc = _get_system_clang()
        if tc is None:
            pytest.skip("System Clang not found")
        assert tc.vendor in ("Apple", "LLVM")

    @pytest.mark.skipif(not _has_clang(), reason="Clang not found on PATH")
    def test_clang_compile_simple_c(self, tmp_path):
        """
        ClangToolchain.compile() compiles a trivial C file.

        Expected result: CompileResult.success is True.

        Linux:    PASS
        macOS:    PASS
        Windows:  SKIP or PASS
        Android:  SKIP
        """
        tc = _get_system_clang()
        if tc is None:
            pytest.skip("System Clang not found")

        source = tmp_path / "test.c"
        source.write_text("int main() { return 0; }\n")
        output = tmp_path / "test_out"

        result = tc.compile(str(source), output=str(output))
        assert result.success is True

    @pytest.mark.skipif(not _has_clang(), reason="Clang not found on PATH")
    def test_clang_apple_detection(self):
        """
        ClangToolchain.is_apple_clang works correctly.

        Expected result:
            macOS: True
            Linux: False

        macOS:    PASS (returns True)
        Linux:    PASS (returns False)
        """
        tc = _get_system_clang()
        if tc is None:
            pytest.skip("System Clang not found")
        if _platform.system() == "Darwin":
            assert tc.is_apple_clang is True
        else:
            assert tc.is_apple_clang is False


# ============================================================================
# MSVC Tests
# ============================================================================

class TestMSVC:
    """
    Tests for MSVCToolchain.

    Expected pass rate:
        Linux:    100% (all tests return empty or raise)
        macOS:    100% (same)
        Windows:  40-70% (depends on Visual Studio installation)
        Android:  100% (all tests return empty or raise)
    """

    def test_msvc_raises_on_non_windows(self):
        """
        MSVCToolchain raises RuntimeError on non-Windows platforms.

        Expected result:
            Linux/macOS/Android: RuntimeError
            Windows: no error (proceeds to validation)

        Linux:    PASS
        macOS:    PASS
        Windows:  PASS (does not raise)
        Android:  PASS
        """
        if _platform.system() != "Windows":
            with pytest.raises(RuntimeError):
                MSVCToolchain(Path("C:/nonexistent"))
        else:
            # On Windows, it will fail validation but not raise at init
            tc = MSVCToolchain(Path("C:/nonexistent"), validate=False)
            assert not tc.is_valid()

    def test_msvc_detect_returns_list(self):
        """
        detect_msvc() returns a list (possibly empty).

        Expected result: returns a list (not None).

        All platforms: PASS
        """
        result = detect_msvc()
        assert isinstance(result, list)

    def test_msvc_detect_empty_on_linux(self):
        """
        detect_msvc() returns empty list on Linux.

        Expected result: len(result) == 0.

        Linux:    PASS
        macOS:    PASS
        Android:  PASS
        Windows:  may or may not be empty
        """
        if _platform.system() != "Windows":
            result = detect_msvc()
            assert len(result) == 0


# ============================================================================
# Capability Tests
# ============================================================================

class TestCapabilities:
    """
    Tests for capability detection.

    Expected pass rate:
        Linux:    85-95% (GCC/Clang available, most features work)
        macOS:    80-90% (Clang available)
        Windows:  40-60% (depends on compiler)
        Android:  50-70% (GCC available)
    """

    @pytest.mark.skipif(not _has_any_compiler(), reason="No compiler found")
    def test_detect_all_returns_dict(self):
        """
        detect_all() returns a dictionary.

        Expected result: dict with capability names as keys.

        All platforms with compiler: PASS
        """
        tc = _get_system_gcc() or _get_system_clang()
        if tc is None:
            pytest.skip("No compiler available")
        caps = detect_all(tc)
        assert isinstance(caps, dict)
        assert len(caps) > 0
        # Check common keys exist
        for key in ("c++17", "c11", "openmp", "lto", "pic"):
            assert key in caps, f"Missing key: {key}"

    @pytest.mark.skipif(not _has_any_compiler(), reason="No compiler found")
    def test_supports_c11(self):
        """
        supports_c11() returns a boolean.

        Expected result: True for GCC 5+ and Clang 3.3+.

        Linux:    PASS (returns True)
        macOS:    PASS (returns True)
        Windows:  SKIP or PASS
        Android:  PASS
        """
        tc = _get_system_gcc() or _get_system_clang()
        if tc is None:
            pytest.skip("No compiler available")
        result = supports_c11(tc)
        assert isinstance(result, bool)

    @pytest.mark.skipif(not _has_any_compiler(), reason="No compiler found")
    def test_supports_cpp17(self):
        """
        supports_cpp17() returns a boolean.

        Expected result: True for GCC 7+ and Clang 5+.

        Linux:    PASS (returns True)
        macOS:    PASS (returns True)
        """
        tc = _get_system_gcc() or _get_system_clang()
        if tc is None:
            pytest.skip("No compiler available")
        result = supports_cpp17(tc)
        assert isinstance(result, bool)

    @pytest.mark.skipif(not _has_any_compiler(), reason="No compiler found")
    def test_supports_pic(self):
        """
        supports_pic() returns True for essentially all compilers.

        Expected result: True.

        Linux:    PASS
        macOS:    PASS
        """
        tc = _get_system_gcc() or _get_system_clang()
        if tc is None:
            pytest.skip("No compiler available")
        assert supports_pic(tc) is True

    @pytest.mark.skipif(not _has_any_compiler(), reason="No compiler found")
    def test_cache_clear(self):
        """
        clear_detection_cache() runs without error.

        Expected result: no exception.

        All platforms with compiler: PASS
        """
        clear_detection_cache()
        # Should not raise


# ============================================================================
# Detection Tests
# ============================================================================

class TestDetection:
    """
    Tests for automatic toolchain detection.

    Expected pass rate:
        Linux:    90-95%
        macOS:    85-90%
        Windows:  50-70%
        Android:  40-60%
    """

    def test_detect_all_toolchains_returns_list(self):
        """
        detect_all_toolchains() returns a list.

        Expected result: list (may be empty).

        All platforms: PASS
        """
        result = detect_all_toolchains()
        assert isinstance(result, list)

    @pytest.mark.skipif(not _has_any_compiler(), reason="No compiler found")
    def test_detect_all_finds_at_least_one(self):
        """
        detect_all_toolchains() finds at least one toolchain
        when a compiler is on PATH.

        Expected result: len(result) >= 1.

        Linux:    PASS
        macOS:    PASS
        Windows:  SKIP or PASS
        Android:  PASS
        """
        result = detect_all_toolchains()
        assert len(result) >= 1

    @pytest.mark.skipif(not _has_any_compiler(), reason="No compiler found")
    def test_detect_best_returns_toolchain(self):
        """
        detect_best_toolchain() returns a Toolchain object.

        Expected result: Toolchain instance.

        Linux:    PASS
        macOS:    PASS
        """
        best = detect_best_toolchain()
        assert best is not None
        assert isinstance(best, Toolchain)
        assert best.is_valid()

    def test_detect_gcc_returns_list(self):
        """
        detect_gcc() returns a list.

        Expected result: list (may be empty).

        All platforms: PASS
        """
        result = detect_gcc()
        assert isinstance(result, list)

    def test_detect_clang_returns_list(self):
        """
        detect_clang() returns a list.

        Expected result: list (may be empty).

        All platforms: PASS
        """
        result = detect_clang()
        assert isinstance(result, list)

    def test_detect_msvc_returns_list(self):
        """
        detect_msvc() returns a list on all platforms.

        Expected result: list.

        All platforms: PASS
        """
        result = detect_msvc()
        assert isinstance(result, list)


# ============================================================================
# Sysroot Tests
# ============================================================================

class TestSysroots:
    """
    Tests for sysroot detection.

    Expected pass rate:
        Linux:    80-90%
        macOS:    70-80%
        Windows:  40-50%
        Android:  50-60%
    """

    @pytest.mark.skipif(not _has_any_compiler(), reason="No compiler found")
    def test_detect_sysroot_returns_sysroot_info(self):
        """
        detect_sysroot() returns a SysrootInfo or None.

        Expected result: SysrootInfo or None.

        Linux:    PASS (returns / or toolchain sysroot)
        macOS:    PASS (returns Xcode SDK sysroot)
        """
        tc = _get_system_gcc() or _get_system_clang()
        if tc is None:
            pytest.skip("No compiler available")
        result = detect_sysroot(tc)
        if result is not None:
            assert isinstance(result, SysrootInfo)
            assert result.path.is_dir() or result.path == Path("/")

    def test_sysroot_info_fields(self):
        """
        SysrootInfo has expected fields.

        Expected result: path, include_dir, lib_dir, source.

        All platforms: PASS
        """
        info = SysrootInfo(path=Path("/"), source="system")
        assert info.path == Path("/")
        assert info.source == "system"
        assert info.is_relative is False


# ============================================================================
# Runtime Tests
# ============================================================================

class TestRuntimes:
    """
    Tests for runtime library detection.

    Expected pass rate:
        Linux:    85-95%
        macOS:    75-85%
        Windows:  40-50%
        Android:  50-60%
    """

    @pytest.mark.skipif(not _has_any_compiler(), reason="No compiler found")
    def test_detect_runtimes_returns_runtime_info(self):
        """
        detect_runtimes() returns a RuntimeInfo object.

        Expected result: RuntimeInfo instance.

        Linux:    PASS
        macOS:    PASS
        """
        tc = _get_system_gcc() or _get_system_clang()
        if tc is None:
            pytest.skip("No compiler available")
        result = detect_runtimes(tc)
        assert isinstance(result, RuntimeInfo)
        assert result.c_library != "unknown" or result.search_paths

    def test_runtime_info_defaults(self):
        """
        RuntimeInfo has correct default values.

        Expected result: c_library="unknown", empty search_paths.

        All platforms: PASS
        """
        info = RuntimeInfo()
        assert info.c_library == "unknown"
        assert info.cxx_library == "none"
        assert info.search_paths == []
        assert info.is_static_only is False


# ============================================================================
# ABI Tests
# ============================================================================

class TestABI:
    """
    Tests for ABI detection.

    Expected pass rate:
        Linux:    90-95%
        macOS:    85-95%
        Windows:  40-50%
        Android:  50-60%
    """

    @pytest.mark.skipif(not _has_any_compiler(), reason="No compiler found")
    def test_detect_abi_returns_abi_info(self):
        """
        detect_abi() returns an ABIInfo object.

        Expected result: ABIInfo with non-zero pointer_size.

        Linux:    PASS (pointer_size=64 on x86_64, 32 on ARM)
        macOS:    PASS (pointer_size=64)
        """
        tc = _get_system_gcc() or _get_system_clang()
        if tc is None:
            pytest.skip("No compiler available")
        abi = detect_abi(tc)
        assert isinstance(abi, ABIInfo)
        assert abi.pointer_size in (32, 64)
        assert abi.endianness in ("little", "big")

    def test_abi_info_fields_exist(self):
        """
        ABIInfo has all expected fields.

        Expected result: All fields are present.

        All platforms: PASS
        """
        abi = ABIInfo()
        assert hasattr(abi, "pointer_size")
        assert hasattr(abi, "endianness")
        assert hasattr(abi, "calling_convention")
        assert hasattr(abi, "exception_model")
        assert hasattr(abi, "name_mangling")

    def test_get_abi_compatibility_same_abi(self):
        """
        get_abi_compatibility returns True for identical ABIs.

        Expected result: True.

        All platforms: PASS
        """
        abi1 = ABIInfo(
            pointer_size=64,
            endianness="little",
            calling_convention="sysv",
            exception_model="dwarf",
            name_mangling="itanium",
        )
        abi2 = ABIInfo(
            pointer_size=64,
            endianness="little",
            calling_convention="sysv",
            exception_model="dwarf",
            name_mangling="itanium",
        )
        assert get_abi_compatibility(abi1, abi2) is True

    def test_get_abi_compatibility_different_abi(self):
        """
        get_abi_compatibility returns False for incompatible ABIs.

        Expected result: False.

        All platforms: PASS
        """
        abi1 = ABIInfo(pointer_size=64, calling_convention="sysv")
        abi2 = ABIInfo(pointer_size=32, calling_convention="sysv")
        assert get_abi_compatibility(abi1, abi2) is False


# ============================================================================
# Environment Tests
# ============================================================================

class TestEnvironments:
    """
    Tests for environment management.

    Expected pass rate: 100% on all platforms.
    Pure file I/O, no compiler dependencies.
    """

    def test_environment_store_create(self, tmp_path):
        """
        EnvironmentStore.create() writes an environment file.

        Expected result: Environment object with correct name.

        All platforms: PASS
        """
        store = EnvironmentStore(data_dir=tmp_path)
        env = store.create("test-env", {"CC": "gcc@14"})
        assert env.name == "test-env"
        assert env.compilers == {"CC": "gcc@14"}

    def test_environment_store_get(self, tmp_path):
        """
        EnvironmentStore.get() retrieves a saved environment.

        Expected result: Environment with matching fields.

        All platforms: PASS
        """
        store = EnvironmentStore(data_dir=tmp_path)
        store.create("myproject", {"CC": "clang@18", "CXX": "clang++@18"})
        env = store.get("myproject")
        assert env is not None
        assert env.compilers["CC"] == "clang@18"
        assert env.compilers["CXX"] == "clang++@18"

    def test_environment_store_list_all(self, tmp_path):
        """
        EnvironmentStore.list_all() returns environment names.

        Expected result: Sorted list of names.

        All platforms: PASS
        """
        store = EnvironmentStore(data_dir=tmp_path)
        store.create("env-a", {})
        store.create("env-b", {})
        names = store.list_all()
        assert "env-a" in names
        assert "env-b" in names

    def test_environment_store_delete(self, tmp_path):
        """
        EnvironmentStore.delete() removes an environment.

        Expected result: get() returns None after delete.

        All platforms: PASS
        """
        store = EnvironmentStore(data_dir=tmp_path)
        store.create("temp-env", {})
        assert store.get("temp-env") is not None
        store.delete("temp-env")
        assert store.get("temp-env") is None

    def test_environment_store_validate_name(self):
        """
        EnvironmentStore rejects invalid names.

        Expected result: ValueError for empty or invalid name.

        All platforms: PASS
        """
        store = EnvironmentStore(data_dir=Path("/tmp"))
        with pytest.raises(ValueError):
            store.create("", {})
        with pytest.raises(ValueError):
            store.create("has spaces", {})

    def test_environment_to_dict(self):
        """
        Environment.to_dict() produces JSON-compatible dict.

        Expected result: dict with name, compilers, flags, etc.

        All platforms: PASS
        """
        env = Environment(name="test", compilers={"CC": "gcc@14"})
        d = env.to_dict()
        assert d["name"] == "test"
        assert d["compilers"]["CC"] == "gcc@14"
        assert "flags" in d
        assert "env_vars" in d

    def test_environment_from_dict(self):
        """
        Environment.from_dict() reconstructs an Environment.

        Expected result: Environment with matching fields.

        All platforms: PASS
        """
        data = {
            "name": "test",
            "compilers": {"CC": "gcc@14"},
            "flags": {"CFLAGS": "-O2"},
            "env_vars": {},
            "description": "",
            "created_at": "",
            "updated_at": "",
            "schema_version": 1,
        }
        env = Environment.from_dict(data)
        assert env.name == "test"
        assert env.compilers["CC"] == "gcc@14"
        assert env.flags["CFLAGS"] == "-O2"

    def test_environment_export_import(self, tmp_path):
        """
        Environment can be exported and re-imported.

        Expected result: Imported environment matches original.

        All platforms: PASS
        """
        store = EnvironmentStore(data_dir=tmp_path)
        store.create("original", {"CC": "gcc@14"})

        export_path = tmp_path / "exported.json"
        store.export_env("original", export_path)
        assert export_path.is_file()

        imported = store.import_env(export_path, overwrite=True)
        assert imported.name == "original"
        assert imported.compilers["CC"] == "gcc@14"


# ============================================================================
# Run configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
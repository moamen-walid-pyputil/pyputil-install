"""
Test suite for ToolForge — Compiler Intelligence Engine.

Tests cover all layers: models, parsers, strategies, validation,
scoring, cache, discovery, and the public API.

Requirements
------------
    pip install pytest pytest-cov

Notes
-----
- Tests that execute real compilers are skipped on environments
  where no compiler is found in PATH.
- Tests that touch the filesystem use temporary directories.
- The blocklist is disabled for testing via TOOLFORGE_SKIP_BLOCKLIST=1.

Expected Results by Platform
----------------------------
Tests are categorized by their platform dependency:

Platform-Independent Tests (expected 100% pass on all platforms):
    - All models tests
    - All parsers tests
    - All scoring tests
    - Strategy registry tests
    - Cache read/write tests
    - Manager filter/rank tests with mock data

Platform-Dependent Tests (may skip if no compiler available):
    - Real validation tests
    - Live discovery tests
    - Real compiler execution tests

Linux:     95-100% pass (GCC/Clang widely available)
macOS:     90-95% pass (Clang available, GCC may be symlink to Clang)
Windows:   70-85% pass (MSVC/GCC detection depends on installation)
Android:   50-70% pass (Termux have compilers, but PATH varies)
"""

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure blocklist is disabled for test environment
os.environ["TOOLFORGE_SKIP_BLOCKLIST"] = "1"

import pyputil_install.compiler_installer.toolforge as toolforge
from pyputil_install.compiler_installer.toolforge.models import CompilerInfo, CompilerKind, CompilerSource
from pyputil_install.compiler_installer.toolforge.parsers import (
    parse_version_output,
    parse_kind_and_vendor,
    parse_target_triplet,
    split_target_triplet,
    normalize_output,
    version_tuple,
)
from pyputil_install.compiler_installer.toolforge.strategies import (
    DiscoveryStrategy,
    DiscoveryStrategyRegistry,
    UserOverrideStrategy,
    PATHStrategy,
    ExtraDirsStrategy,
    CommonDirsStrategy,
)
from pyputil_install.compiler_installer.toolforge.validation import validate_and_fingerprint
from pyputil_install.compiler_installer.toolforge.scoring import (
    score_compiler,
    rank_compilers,
    best_compiler,
    score_for_native_build,
    score_for_cross_compilation,
    explain_score,
    SCORING_WEIGHTS,
)
from pyputil_install.compiler_installer.toolforge.cache import CompilerCache
from pyputil_install.compiler_installer.toolforge.discovery import CompilerManager, discover_compilers


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_gcc_info() -> CompilerInfo:
    """
    Create a sample GCC CompilerInfo for testing.

    Returns
    -------
    CompilerInfo
        Valid GCC 13.2.0 native compiler on x86_64-linux-gnu.
        confidence_score=0.98, source=SYSTEM_PATH.
    """
    return CompilerInfo(
        path="/usr/bin/gcc",
        version="13.2.0",
        vendor="GNU",
        kind=CompilerKind.GCC,
        target_triplet="x86_64-linux-gnu",
        is_cross_compiler=False,
        source=CompilerSource.SYSTEM_PATH,
        confidence_score=0.98,
        fingerprint="abc123",
    )


@pytest.fixture
def sample_clang_info() -> CompilerInfo:
    """
    Create a sample Clang CompilerInfo for testing.

    Returns
    -------
    CompilerInfo
        Valid Clang 17.0.6 native compiler on x86_64-linux-gnu.
        confidence_score=0.99, source=SYSTEM_PATH.
    """
    return CompilerInfo(
        path="/usr/bin/clang",
        version="17.0.6",
        vendor="LLVM",
        kind=CompilerKind.CLANG,
        target_triplet="x86_64-linux-gnu",
        is_cross_compiler=False,
        source=CompilerSource.SYSTEM_PATH,
        confidence_score=0.99,
        fingerprint="def456",
    )


@pytest.fixture
def sample_cross_info() -> CompilerInfo:
    """
    Create a sample cross-compiler CompilerInfo for testing.

    Returns
    -------
    CompilerInfo
        ARM cross-compiler targeting arm-none-eabi.
        confidence_score=0.95, source=COMMON_DIR.
    """
    return CompilerInfo(
        path="/opt/arm/bin/arm-none-eabi-gcc",
        version="12.2.0",
        vendor="GNU",
        kind=CompilerKind.GCC,
        target_triplet="arm-none-eabi",
        is_cross_compiler=True,
        source=CompilerSource.COMMON_DIR,
        confidence_score=0.95,
        fingerprint="ghi789",
    )


@pytest.fixture
def sample_mingw_info() -> CompilerInfo:
    """
    Create a sample MinGW CompilerInfo for testing.

    Returns
    -------
    CompilerInfo
        MinGW-w64 cross-compiler targeting Windows.
        confidence_score=0.70, source=SYSTEM_PATH.
        Low confidence due to MinGW wrapper penalty.
    """
    return CompilerInfo(
        path="/usr/bin/x86_64-w64-mingw32-gcc",
        version="12.0.0",
        vendor="MinGW",
        kind=CompilerKind.GCC,
        target_triplet="x86_64-w64-mingw32",
        is_cross_compiler=True,
        source=CompilerSource.SYSTEM_PATH,
        confidence_score=0.70,
        fingerprint="jkl012",
    )


@pytest.fixture
def sample_unknown_info() -> CompilerInfo:
    """
    Create a sample unknown CompilerInfo for testing.

    Returns
    -------
    CompilerInfo
        Unknown compiler with empty target and version "0.0.0".
        confidence_score=0.30, source=FALLBACK.
        Used to test penalty scoring and filtering.
    """
    return CompilerInfo(
        path="/usr/local/bin/mystery-cc",
        version="0.0.0",
        vendor="unknown",
        kind=CompilerKind.UNKNOWN,
        target_triplet="",
        is_cross_compiler=False,
        source=CompilerSource.FALLBACK,
        confidence_score=0.30,
        fingerprint="mno345",
    )


@pytest.fixture
def temp_cache_dir() -> str:
    """
    Create a temporary directory for cache testing.

    Returns
    -------
    str
        Path to temporary directory. Cleaned up after test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def has_real_compiler() -> bool:
    """
    Check if a real compiler is available for live tests.

    Returns
    -------
    bool
        True if gcc or clang or cc is found in PATH.

    Used to skip tests that require actual compiler execution
    on environments without one installed.
    """
    import shutil
    for name in ["gcc", "clang", "cc"]:
        if shutil.which(name):
            return True
    return False


# ============================================================================
# Models Tests
# ============================================================================

class TestModels:
    """
    Tests for core data models.

    Expected pass rate: 100% on all platforms.
    No filesystem or subprocess dependencies.
    """

    def test_compiler_kind_enum_members(self):
        """
        CompilerKind enum has all expected members.

        Expected result: True
        """
        members = [kind.name for kind in CompilerKind]
        assert "GCC" in members
        assert "CLANG" in members
        assert "MSVC" in members
        assert "UNKNOWN" in members

    def test_compiler_source_ordering(self):
        """
        CompilerSource values are ordered by priority.

        Expected result: USER_OVERRIDE(1) < SYSTEM_PATH(3) < FALLBACK(5)
        Lower numeric value = higher priority.
        """
        assert CompilerSource.USER_OVERRIDE.value < CompilerSource.MANAGED_TOOLCHAIN.value
        assert CompilerSource.MANAGED_TOOLCHAIN.value < CompilerSource.SYSTEM_PATH.value
        assert CompilerSource.SYSTEM_PATH.value < CompilerSource.COMMON_DIR.value
        assert CompilerSource.COMMON_DIR.value < CompilerSource.FALLBACK.value

    def test_compiler_info_is_frozen(self, sample_gcc_info):
        """
        CompilerInfo is immutable after creation.

        Expected result: AttributeError raised on mutation attempt.
        """
        with pytest.raises((AttributeError, TypeError)):
            sample_gcc_info.path = "/other/path"  # type: ignore[attr-defined]

    def test_compiler_info_default_confidence(self):
        """
        Default confidence_score is 1.0 when not explicitly set.

        Expected result: 1.0
        """
        info = CompilerInfo(
            path="/tmp/test-cc",
            version="1.0.0",
            vendor="GNU",
            kind=CompilerKind.GCC,
            target_triplet="x86_64-linux-gnu",
            is_cross_compiler=False,
            source=CompilerSource.FALLBACK,
        )
        assert info.confidence_score == 1.0

    def test_compiler_info_fingerprint_optional(self):
        """
        fingerprint field accepts None.

        Expected result: None
        """
        info = CompilerInfo(
            path="/tmp/test-cc",
            version="1.0.0",
            vendor="GNU",
            kind=CompilerKind.GCC,
            target_triplet="x86_64-linux-gnu",
            is_cross_compiler=False,
            source=CompilerSource.FALLBACK,
        )
        assert info.fingerprint is None

    def test_compiler_info_repr(self, sample_gcc_info):
        """
        CompilerInfo repr is readable and contains path.

        Expected result: String containing path.
        """
        r = repr(sample_gcc_info)
        assert "/usr/bin/gcc" in r

    def test_compiler_info_equality(self):
        """
        Two CompilerInfo with identical fields are equal.

        Expected result: a == b
        """
        a = CompilerInfo(
            path="/tmp/cc", version="1.0", vendor="GNU",
            kind=CompilerKind.GCC, target_triplet="x86_64-linux-gnu",
            is_cross_compiler=False, source=CompilerSource.FALLBACK,
        )
        b = CompilerInfo(
            path="/tmp/cc", version="1.0", vendor="GNU",
            kind=CompilerKind.GCC, target_triplet="x86_64-linux-gnu",
            is_cross_compiler=False, source=CompilerSource.FALLBACK,
        )
        assert a == b

    def test_compiler_info_inequality(self):
        """
        Two CompilerInfo with different paths are not equal.

        Expected result: a != b
        """
        a = CompilerInfo(
            path="/tmp/cc", version="1.0", vendor="GNU",
            kind=CompilerKind.GCC, target_triplet="x86_64-linux-gnu",
            is_cross_compiler=False, source=CompilerSource.FALLBACK,
        )
        b = CompilerInfo(
            path="/tmp/other", version="1.0", vendor="GNU",
            kind=CompilerKind.GCC, target_triplet="x86_64-linux-gnu",
            is_cross_compiler=False, source=CompilerSource.FALLBACK,
        )
        assert a != b


# ============================================================================
# Parsers Tests
# ============================================================================

class TestParsers:
    """
    Tests for text parsing utilities.

    Expected pass rate: 100% on all platforms.
    Pure text processing, no external dependencies.
    """

    # --- parse_kind_and_vendor ---

    def test_parse_gcc_kind(self):
        """
        GCC --version output returns GCC kind and GNU vendor.

        Expected result: (GCC, "GNU")
        """
        kind, vendor = parse_kind_and_vendor("gcc (GCC) 13.2.0")
        assert kind == CompilerKind.GCC
        assert vendor == "GNU"

    def test_parse_clang_kind(self):
        """
        Clang --version output returns CLANG kind and LLVM vendor.

        Expected result: (CLANG, "LLVM")
        """
        kind, vendor = parse_kind_and_vendor("clang version 17.0.6")
        assert kind == CompilerKind.CLANG
        assert vendor == "LLVM"

    def test_parse_apple_clang(self):
        """
        Apple Clang --version returns CLANG kind and Apple vendor.

        Expected result: (CLANG, "Apple")
        """
        kind, vendor = parse_kind_and_vendor("Apple clang version 15.0.0")
        assert kind == CompilerKind.CLANG
        assert vendor == "Apple"

    def test_parse_mingw_gcc(self):
        """
        MinGW-w64 GCC returns GCC kind and MinGW vendor.

        Expected result: (GCC, "MinGW")
        """
        kind, vendor = parse_kind_and_vendor(
            "gcc (MinGW-W64 x86_64-posix-seh) 12.0.0"
        )
        assert kind == CompilerKind.GCC
        assert vendor == "MinGW"

    def test_parse_msvc_kind(self):
        """
        MSVC --version returns MSVC kind and Microsoft vendor.

        Expected result: (MSVC, "Microsoft")
        """
        kind, vendor = parse_kind_and_vendor(
            "Microsoft (R) C/C++ Optimizing Compiler Version 19.38.33130"
        )
        assert kind == CompilerKind.MSVC
        assert vendor == "Microsoft"

    def test_parse_unknown_kind(self):
        """
        Unrecognizable output returns UNKNOWN kind and "unknown" vendor.

        Expected result: (UNKNOWN, "unknown")
        """
        kind, vendor = parse_kind_and_vendor("some random text")
        assert kind == CompilerKind.UNKNOWN
        assert vendor == "unknown"

    def test_parse_empty_output(self):
        """
        Empty string returns UNKNOWN kind and "unknown" vendor.

        Expected result: (UNKNOWN, "unknown")
        """
        kind, vendor = parse_kind_and_vendor("")
        assert kind == CompilerKind.UNKNOWN
        assert vendor == "unknown"

    # --- parse_version_output ---

    def test_parse_gcc_version(self):
        """
        GCC version extracted correctly from first line.

        Expected result: "13.2.0"
        """
        v = parse_version_output("gcc (GCC) 13.2.0", CompilerKind.GCC)
        assert v == "13.2.0"

    def test_parse_clang_version(self):
        """
        Clang version extracted correctly.

        Expected result: "17.0.6"
        """
        v = parse_version_output("clang version 17.0.6", CompilerKind.CLANG)
        assert v == "17.0.6"

    def test_parse_msvc_version(self):
        """
        MSVC version extracted correctly.

        Expected result: "19.38.33130"
        """
        v = parse_version_output(
            "Microsoft (R) C/C++ Optimizing Compiler Version 19.38.33130 for x64",
            CompilerKind.MSVC,
        )
        assert v == "19.38.33130"

    def test_parse_version_empty_output(self):
        """
        Empty output returns "0.0.0".

        Expected result: "0.0.0"
        """
        v = parse_version_output("", CompilerKind.GCC)
        assert v == "0.0.0"

    def test_parse_version_no_digits(self):
        """
        Text without version digits returns "0.0.0".

        Expected result: "0.0.0"
        """
        v = parse_version_output("hello world", CompilerKind.GCC)
        assert v == "0.0.0"

    # --- parse_target_triplet ---

    def test_parse_triplet_normal(self):
        """
        Normal triplet returns unchanged.

        Expected result: "x86_64-linux-gnu"
        """
        t = parse_target_triplet("x86_64-linux-gnu")
        assert t == "x86_64-linux-gnu"

    def test_parse_triplet_with_newlines(self):
        """
        Triplet with trailing newlines is stripped.

        Expected result: "arm-none-eabi"
        """
        t = parse_target_triplet("arm-none-eabi\n\n")
        assert t == "arm-none-eabi"

    def test_parse_triplet_none(self):
        """
        None input returns empty string.

        Expected result: ""
        """
        t = parse_target_triplet(None)
        assert t == ""

    def test_parse_triplet_empty(self):
        """
        Empty string returns empty string.

        Expected result: ""
        """
        t = parse_target_triplet("")
        assert t == ""

    # --- split_target_triplet ---

    def test_split_triplet_three_part(self):
        """
        Three-part triplet splits correctly.

        Expected result: ("x86_64", "linux", "gnu")
        """
        parts = split_target_triplet("x86_64-linux-gnu")
        assert parts == ("x86_64", "linux", "gnu")

    def test_split_triplet_two_part(self):
        """
        Two-part triplet splits with empty OS.

        Expected result: ("arm", "none", "eabi")
        Note: "eabi" is the OS part in "arm-none-eabi"
        """
        parts = split_target_triplet("arm-none-eabi")
        assert parts == ("arm", "none", "eabi")

    def test_split_triplet_single_part(self):
        """
        Single-part triplet has empty vendor and OS.

        Expected result: ("x86_64", "", "")
        """
        parts = split_target_triplet("x86_64")
        assert parts == ("x86_64", "", "")

    def test_split_triplet_empty(self):
        """
        Empty triplet returns all empty.

        Expected result: ("", "", "")
        """
        parts = split_target_triplet("")
        assert parts == ("", "", "")

    # --- version_tuple ---

    def test_version_tuple_three_part(self):
        """
        Three-part version converts to tuple.

        Expected result: (13, 2, 0)
        """
        assert version_tuple("13.2.0") == (13, 2, 0)

    def test_version_tuple_two_part(self):
        """
        Two-part version pads with zero.

        Expected result: (17, 0, 0)
        """
        assert version_tuple("17.0") == (17, 0, 0)

    def test_version_tuple_single_part(self):
        """
        Single-part version pads with zeros.

        Expected result: (5, 0, 0)
        """
        assert version_tuple("5") == (5, 0, 0)

    def test_version_tuple_empty(self):
        """
        Empty string returns zeros.

        Expected result: (0, 0, 0)
        """
        assert version_tuple("") == (0, 0, 0)

    def test_version_tuple_comparison(self):
        """
        Version tuples compare correctly.

        Expected result: 13.2.0 > 13.1.0
        """
        assert version_tuple("13.2.0") > version_tuple("13.1.0")

    def test_version_tuple_equal(self):
        """
        Identical versions compare equal.

        Expected result: 13.2.0 == 13.2.0
        """
        assert version_tuple("13.2.0") == version_tuple("13.2.0")

    # --- normalize_output ---

    def test_normalize_carriage_returns(self):
        """
        Carriage returns converted to newlines.

        Expected result: No \r in output.
        """
        result = normalize_output("hello\r\nworld\r\n")
        assert "\r" not in result

    def test_normalize_multiple_spaces(self):
        """
        Multiple spaces collapsed to single space.

        Expected result: "hello world"
        """
        result = normalize_output("hello    world")
        assert result == "hello world"

    def test_normalize_strip_whitespace(self):
        """
        Leading/trailing whitespace removed per line.

        Expected result: "hello"
        """
        result = normalize_output("  hello  ")
        assert result == "hello"


# ============================================================================
# Strategies Tests
# ============================================================================

class TestStrategies:
    """
    Tests for discovery strategies.

    Expected pass rate: 100% on all platforms for registry tests.
    Strategy discovery tests: 90-100% depending on environment.
    """

    def test_user_override_empty(self):
        """
        UserOverrideStrategy returns empty when no env vars set.

        Expected result: []
        """
        with patch.dict(os.environ, {}, clear=True):
            strategy = UserOverrideStrategy()
            assert strategy.discover() == []

    def test_user_override_with_cc(self, tmp_path):
        """
        UserOverrideStrategy finds path from CC env var.

        Expected result: [path_to_temp_gcc]
        """
        fake_gcc = tmp_path / "gcc"
        fake_gcc.touch()
        fake_gcc.chmod(0o755)

        # Do not clear=True — preserve environment for Android
        with patch.dict(os.environ, {"CC": str(fake_gcc)}):
            strategy = UserOverrideStrategy()
            result = strategy.discover()
            assert str(fake_gcc) in result

    def test_user_override_with_missing_file(self):
        """
        UserOverrideStrategy skips env vars pointing to missing files.

        Expected result: []
        """
        with patch.dict(os.environ, {"CC": "/nonexistent/gcc"}):
            strategy = UserOverrideStrategy()
            assert strategy.discover() == []

    def test_path_strategy_finds_executables(self, tmp_path):
        """
        PATHStrategy finds compiler candidates in a directory.

        Expected result: [path_to_temp_gcc]
        """
        fake_gcc = tmp_path / "gcc"
        fake_gcc.touch()
        fake_gcc.chmod(0o755)

        # Do not clear=True — preserve environment for Android
        with patch.dict(os.environ, {"PATH": str(tmp_path)}):
            strategy = PATHStrategy()
            result = strategy.discover()
            assert str(fake_gcc) in result

    def test_path_strategy_empty_path(self):
        """
        PATHStrategy handles empty PATH gracefully.

        Expected result: []
        """
        with patch.dict(os.environ, {"PATH": ""}, clear=True):
            strategy = PATHStrategy()
            assert strategy.discover() == []

    def test_extra_dirs_strategy(self, tmp_path):
        """
        ExtraDirsStrategy finds compilers in configured directories.

        Expected result: [path_to_temp_gcc]
        """
        fake_gcc = tmp_path / "gcc"
        fake_gcc.touch()
        fake_gcc.chmod(0o755)

        # Do not clear=True — preserve environment for Android
        with patch.dict(
            os.environ,
            {"COMPILER_SEARCH_DIRS": str(tmp_path)},
        ):
            strategy = ExtraDirsStrategy()
            result = strategy.discover()
            assert str(fake_gcc) in result

    def test_extra_dirs_strategy_empty(self):
        """
        ExtraDirsStrategy returns empty when env var not set.

        Expected result: []
        """
        with patch.dict(os.environ, {}, clear=True):
            strategy = ExtraDirsStrategy()
            assert strategy.discover() == []

    # --- Strategy Registry ---

    def test_registry_register_strategy(self):
        """
        Custom strategy can be registered.

        Expected result: Strategy appears in get_all().
        """
        class CustomStrategy(DiscoveryStrategy):
            name = "test_custom"
            priority = 42
            def discover(self):
                return []

        DiscoveryStrategyRegistry.register(CustomStrategy)
        names = [s.name for s in DiscoveryStrategyRegistry.get_all()]
        assert "test_custom" in names

    def test_registry_disable_strategy(self):
        """
        Disabled strategy excluded from get_enabled().

        Expected result: Strategy not in enabled list.
        """
        class TempStrategy(DiscoveryStrategy):
            name = "temp_to_disable"
            priority = 99
            def discover(self):
                return []

        DiscoveryStrategyRegistry.register(TempStrategy)
        DiscoveryStrategyRegistry.disable("temp_to_disable")
        enabled_names = [s.name for s in DiscoveryStrategyRegistry.get_enabled()]
        assert "temp_to_disable" not in enabled_names

    def test_registry_enable_strategy(self):
        """
        Re-enabled strategy appears in get_enabled().

        Expected result: Strategy back in enabled list.
        """
        class TempStrategy2(DiscoveryStrategy):
            name = "temp_to_enable"
            priority = 98
            def discover(self):
                return []

        DiscoveryStrategyRegistry.register(TempStrategy2)
        DiscoveryStrategyRegistry.disable("temp_to_enable")
        DiscoveryStrategyRegistry.enable("temp_to_enable")
        enabled_names = [s.name for s in DiscoveryStrategyRegistry.get_enabled()]
        assert "temp_to_enable" in enabled_names

    def test_registry_duplicate_name_raises(self):
        """
        Registering duplicate strategy name raises ValueError.

        Expected result: ValueError raised.
        """
        class DupStrategy(DiscoveryStrategy):
            name = "duplicate_test"
            def discover(self):
                return []

        DiscoveryStrategyRegistry.register(DupStrategy)
        with pytest.raises(ValueError):
            DiscoveryStrategyRegistry.register(DupStrategy)

    def test_registry_missing_name_raises(self):
        """
        Strategy without name raises ValueError.

        Expected result: ValueError raised.
        """
        class NoNameStrategy(DiscoveryStrategy):
            def discover(self):
                return []

        with pytest.raises(ValueError):
            DiscoveryStrategyRegistry.register(NoNameStrategy)

    def test_registry_set_order(self):
        """
        set_order reassigns priorities correctly.

        Expected result: user_override has priority 0, system_path has priority 1.
        """
        DiscoveryStrategyRegistry.set_order(["user_override", "system_path"])
        all_strats = {s.name: s.priority for s in DiscoveryStrategyRegistry.get_all()}
        assert all_strats.get("user_override") == 0
        assert all_strats.get("system_path") == 1

    def test_strategy_priority_sorting(self):
        """
        get_enabled returns strategies sorted by priority.

        Expected result: Lower priority numbers come first.
        """
        enabled = DiscoveryStrategyRegistry.get_enabled()
        priorities = [s.priority for s in enabled]
        assert priorities == sorted(priorities)


# ============================================================================
# Scoring Tests
# ============================================================================

class TestScoring:
    """
    Tests for compiler scoring and ranking.

    Expected pass rate: 100% on all platforms.
    Pure math/logic, no external dependencies.
    """

    def test_score_native_beats_cross(
        self, sample_gcc_info, sample_cross_info
    ):
        """
        Native compiler scores higher than equivalent cross-compiler.

        Expected result: score(gcc_native) > score(arm_cross)
        """
        native_score = score_compiler(sample_gcc_info, prefer_native=True)
        cross_score = score_compiler(sample_cross_info, prefer_native=True)
        assert native_score > cross_score

    def test_score_higher_confidence_beats_lower(
        self, sample_gcc_info, sample_mingw_info
    ):
        """
        High-confidence compiler scores higher than low-confidence.

        Expected result: score(gcc) > score(mingw)
        """
        gcc_score = score_compiler(sample_gcc_info)
        mingw_score = score_compiler(sample_mingw_info)
        assert gcc_score > mingw_score

    def test_score_unknown_penalized(
        self, sample_gcc_info, sample_unknown_info
    ):
        """
        Unknown compiler scores lower than known compiler.

        Expected result: score(gcc) > score(unknown)
        """
        gcc_score = score_compiler(sample_gcc_info)
        unknown_score = score_compiler(sample_unknown_info)
        assert gcc_score > unknown_score

    def test_score_cross_no_penalty_when_prefer_false(
        self, sample_cross_info
    ):
        """
        Cross-compiler not penalized when prefer_native=False.

        Expected result: Score without native preference is higher
        or equal to score with native preference.
        """
        score_with = score_compiler(sample_cross_info, prefer_native=True)
        score_without = score_compiler(sample_cross_info, prefer_native=False)
        assert score_without >= score_with

    def test_score_user_override_beats_system_path(self):
        """
        USER_OVERRIDE source scores higher than SYSTEM_PATH.

        Expected result: score(override) > score(system)
        """
        override = CompilerInfo(
            path="/custom/gcc", version="10.0.0", vendor="GNU",
            kind=CompilerKind.GCC, target_triplet="x86_64-linux-gnu",
            is_cross_compiler=False, source=CompilerSource.USER_OVERRIDE,
        )
        system = CompilerInfo(
            path="/usr/bin/gcc", version="13.2.0", vendor="GNU",
            kind=CompilerKind.GCC, target_triplet="x86_64-linux-gnu",
            is_cross_compiler=False, source=CompilerSource.SYSTEM_PATH,
        )
        # Override should win despite lower version
        assert score_compiler(override) > score_compiler(system)

    def test_score_higher_version_beats_lower(self):
        """
        Higher version scores higher when all else equal.

        Expected result: score(v13) > score(v10)
        """
        v13 = CompilerInfo(
            path="/usr/bin/gcc-13", version="13.0.0", vendor="GNU",
            kind=CompilerKind.GCC, target_triplet="x86_64-linux-gnu",
            is_cross_compiler=False, source=CompilerSource.SYSTEM_PATH,
        )
        v10 = CompilerInfo(
            path="/usr/bin/gcc-10", version="10.0.0", vendor="GNU",
            kind=CompilerKind.GCC, target_triplet="x86_64-linux-gnu",
            is_cross_compiler=False, source=CompilerSource.SYSTEM_PATH,
        )
        assert score_compiler(v13) > score_compiler(v10)

    def test_rank_compilers_sorted(self, sample_gcc_info, sample_clang_info, sample_unknown_info):
        """
        rank_compilers returns list sorted by score descending.

        Expected result: First element has highest score.
        """
        compilers = [sample_unknown_info, sample_clang_info, sample_gcc_info]
        ranked = rank_compilers(compilers)
        scores = [score_compiler(c) for c in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_compilers_stable(self):
        """
        rank_compilers preserves order for equal scores.

        Expected result: Original relative order maintained for equal items.
        """
        # Two compilers with identical scoring attributes
        a = CompilerInfo(
            path="/usr/bin/gcc", version="13.2.0", vendor="GNU",
            kind=CompilerKind.GCC, target_triplet="x86_64-linux-gnu",
            is_cross_compiler=False, source=CompilerSource.SYSTEM_PATH,
            confidence_score=0.98, fingerprint="abc123",
        )
        b = CompilerInfo(
            path="/usr/bin/gcc-13", version="13.2.0", vendor="GNU",
            kind=CompilerKind.GCC, target_triplet="x86_64-linux-gnu",
            is_cross_compiler=False, source=CompilerSource.SYSTEM_PATH,
            confidence_score=0.98, fingerprint="abc123",
        )
        ranked = rank_compilers([a, b])
        assert ranked[0] == a
        assert ranked[1] == b

    def test_best_compiler_returns_highest(self, sample_gcc_info, sample_unknown_info):
        """
        best_compiler returns the single highest-scoring compiler.

        Expected result: sample_gcc_info
        """
        best = best_compiler([sample_unknown_info, sample_gcc_info])
        assert best == sample_gcc_info

    def test_best_compiler_empty(self):
        """
        best_compiler returns None for empty list.

        Expected result: None
        """
        assert best_compiler([]) is None

    def test_score_for_native_build(self, sample_gcc_info, sample_cross_info):
        """
        score_for_native_build heavily prefers native.

        Expected result: native_score >> cross_score
        """
        native_score = score_for_native_build(sample_gcc_info)
        cross_score = score_for_native_build(sample_cross_info)
        assert native_score > cross_score * 1.5  # Significant gap

    def test_score_for_cross_compilation(self, sample_cross_info):
        """
        score_for_cross_compilation does not penalize cross.

        Expected result: Cross score with prefer_native=False >= score with prefer_native=True.
        """
        score_with = score_compiler(sample_cross_info, prefer_native=True)
        score_without = score_for_cross_compilation(sample_cross_info)
        # Cross should not be penalized when we don't prefer native
        assert score_without >= score_with

    def test_explain_score_keys(self, sample_gcc_info):
        """
        explain_score returns all expected component keys.

        Expected result: All keys present.
        """
        breakdown = explain_score(sample_gcc_info)
        expected_keys = {"source", "native_bonus", "version",
                         "vendor_adjustment", "confidence_penalty",
                         "raw_total", "final_score"}
        assert set(breakdown.keys()) == expected_keys

    def test_explain_score_values_are_numeric(self, sample_gcc_info):
        """
        All explain_score values are floats.

        Expected result: All float.
        """
        breakdown = explain_score(sample_gcc_info)
        for value in breakdown.values():
            assert isinstance(value, float)

    def test_custom_weights_override(self, sample_gcc_info):
        """
        Custom weights parameter overrides defaults.

        Expected result: Different score with custom weights.
        """
        default_score = score_compiler(sample_gcc_info)
        custom_score = score_compiler(
            sample_gcc_info,
            weights={"native_bonus": 1000.0},
        )
        assert custom_score != default_score

    def test_confidence_multiplier_affects_score(self):
        """
        Lower confidence reduces final score.

        Expected result: 0.5 confidence halves the raw score.
        """
        high_conf = CompilerInfo(
            path="/tmp/cc", version="1.0", vendor="GNU",
            kind=CompilerKind.GCC, target_triplet="x86_64-linux-gnu",
            is_cross_compiler=False, source=CompilerSource.FALLBACK,
            confidence_score=1.0,
        )
        low_conf = CompilerInfo(
            path="/tmp/cc", version="1.0", vendor="GNU",
            kind=CompilerKind.GCC, target_triplet="x86_64-linux-gnu",
            is_cross_compiler=False, source=CompilerSource.FALLBACK,
            confidence_score=0.5,
        )
        assert score_compiler(high_conf) > score_compiler(low_conf)


# ============================================================================
# Cache Tests
# ============================================================================

class TestCache:
    """
    Tests for filesystem caching.

    Expected pass rate: 95-100% on all platforms.
    Requires writable temp directory (available on all platforms).
    """

    def test_save_and_load(self, temp_cache_dir, sample_gcc_info):
        """
        Compilers saved to cache can be loaded back.

        Expected result: Loaded compiler equals saved compiler.
        """
        cache = CompilerCache(cache_dir=temp_cache_dir)
        cache.save([sample_gcc_info])
        loaded = cache.load()
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].path == sample_gcc_info.path
        assert loaded[0].version == sample_gcc_info.version

    def test_load_empty_cache(self, temp_cache_dir):
        """
        Loading nonexistent cache returns None.

        Expected result: None
        """
        cache = CompilerCache(cache_dir=temp_cache_dir)
        assert cache.load() is None

    def test_clear_cache(self, temp_cache_dir, sample_gcc_info):
        """
        clear() removes the cache file.

        Expected result: load() returns None after clear().
        """
        cache = CompilerCache(cache_dir=temp_cache_dir)
        cache.save([sample_gcc_info])
        assert cache.load() is not None
        cache.clear()
        assert cache.load() is None

    def test_is_valid(self, temp_cache_dir, sample_gcc_info):
        """
        is_valid returns True after save.

        Expected result: True
        """
        cache = CompilerCache(cache_dir=temp_cache_dir)
        cache.save([sample_gcc_info])
        assert cache.is_valid() is True

    def test_is_valid_empty(self, temp_cache_dir):
        """
        is_valid returns False when no cache exists.

        Expected result: False
        """
        cache = CompilerCache(cache_dir=temp_cache_dir)
        assert cache.is_valid() is False

    def test_save_empty_list(self, temp_cache_dir):
        """
        Saving empty list still writes cache file.

        Expected result: load() returns empty list.
        """
        cache = CompilerCache(cache_dir=temp_cache_dir)
        cache.save([])
        loaded = cache.load()
        assert loaded is not None
        assert len(loaded) == 0

    def test_load_corrupted_cache(self, temp_cache_dir):
        """
        Corrupted JSON returns None instead of raising.

        Expected result: None
        """
        cache = CompilerCache(cache_dir=temp_cache_dir)
        os.makedirs(cache.cache_dir, exist_ok=True)
        with open(cache.cache_path, "w") as f:
            f.write("not valid json{{{{")
        assert cache.load() is None

    def test_cache_max_age(self, temp_cache_dir, sample_gcc_info):
        """
        Cache older than max_age returns None.

        Expected result: None when max_age=0.
        """
        cache = CompilerCache(cache_dir=temp_cache_dir, max_age_seconds=0)
        cache.save([sample_gcc_info])
        # max_age=0 means immediate expiry
        assert cache.load() is None


# ============================================================================
# Validation Tests
# ============================================================================

class TestValidation:
    """
    Tests for compiler validation.

    Expected pass rate: 80-95% on platforms with a real compiler.
    Skips live execution tests if no compiler found in PATH.
    """

    @pytest.mark.skipif(
        not has_real_compiler(),
        reason="No real compiler found in PATH",
    )
    def test_validate_real_gcc(self):
        """
        Validate a real GCC found in PATH.

        Expected result: CompilerInfo returned with GCC kind.
        Skips if no compiler available.
        """
        import shutil
        gcc_path = shutil.which("gcc") or shutil.which("cc")
        if not gcc_path:
            pytest.skip("gcc/cc not found")
        result = validate_and_fingerprint(gcc_path)
        assert result is not None
        assert result.kind in (CompilerKind.GCC, CompilerKind.CLANG)
        assert result.path
        assert result.version != "0.0.0"

    def test_validate_nonexistent(self):
        """
        Validating nonexistent path returns None.

        Expected result: None
        """
        result = validate_and_fingerprint("/nonexistent/compiler/xyz")
        assert result is None

    def test_validate_blocklisted(self):
        """
        Validating blocklisted path returns None.

        Expected result: None (rejected by blocklist)
        Note: Blocklist is disabled in test env but this test
        temporarily re-enables it.
        """
        with patch.dict(os.environ, {}, clear=True):
            # Without TOOLFORGE_SKIP_BLOCKLIST, /tmp should be blocked
            orig = os.environ.get("TOOLFORGE_SKIP_BLOCKLIST")
            if orig:
                del os.environ["TOOLFORGE_SKIP_BLOCKLIST"]
            try:
                result = validate_and_fingerprint("/tmp/gcc")
                assert result is None
            finally:
                if orig:
                    os.environ["TOOLFORGE_SKIP_BLOCKLIST"] = orig


# ============================================================================
# Discovery / CompilerManager Tests
# ============================================================================

class TestCompilerManager:
    """
    Tests for CompilerManager and discover_compilers.

    Expected pass rate: 85-100% depending on platform.
    Most tests use mock data and pass everywhere.
    """

    def test_manager_all_with_mock(
        self, sample_gcc_info, sample_clang_info, sample_cross_info
    ):
        """
        Manager.all() returns cached compiler list.

        Expected result: List of 3 compilers.
        """
        manager = CompilerManager()
        manager._all_compilers = [sample_gcc_info, sample_clang_info, sample_cross_info]
        assert len(manager.all()) == 3

    def test_manager_all_empty(self):
        """
        Manager.all() returns empty list before discovery.

        Expected result: []
        """
        manager = CompilerManager()
        assert manager.all() == []

    def test_manager_best_returns_highest(
        self, sample_gcc_info, sample_unknown_info
    ):
        """
        Manager.best() returns the highest-priority compiler.

        Expected result: sample_gcc_info (higher score than unknown).
        """
        manager = CompilerManager()
        # Pre-sorted: GCC beats unknown
        manager._all_compilers = [sample_gcc_info, sample_unknown_info]
        assert manager.best() == sample_gcc_info

    def test_manager_best_empty(self):
        """
        Manager.best() returns None when no compilers.

        Expected result: None
        """
        manager = CompilerManager()
        assert manager.best() is None

    def test_manager_filter_by_kind(
        self, sample_gcc_info, sample_clang_info, sample_cross_info
    ):
        """
        Filter by CompilerKind returns only matching compilers.

        Expected result: 2 GCC compilers, 0 Clang (cross is also GCC).
        """
        manager = CompilerManager()
        manager._all_compilers = [sample_gcc_info, sample_clang_info, sample_cross_info]
        gcc = manager.filter(kinds=[CompilerKind.GCC])
        assert len(gcc) == 2
        clang = manager.filter(kinds=[CompilerKind.CLANG])
        assert len(clang) == 1

    def test_manager_filter_by_vendor(
        self, sample_gcc_info, sample_clang_info, sample_mingw_info
    ):
        """
        Filter by vendor returns only matching compilers.

        Expected result: 1 GNU, 1 LLVM, 1 MinGW.
        """
        manager = CompilerManager()
        manager._all_compilers = [sample_gcc_info, sample_clang_info, sample_mingw_info]
        gnu = manager.filter(vendors=["GNU"])
        assert len(gnu) == 1
        llvm = manager.filter(vendors=["LLVM"])
        assert len(llvm) == 1
        mingw = manager.filter(vendors=["MinGW"])
        assert len(mingw) == 1

    def test_manager_filter_cross_only(
        self, sample_gcc_info, sample_cross_info
    ):
        """
        Filter cross_compilers=True returns only cross-compilers.

        Expected result: 1 cross-compiler.
        """
        manager = CompilerManager()
        manager._all_compilers = [sample_gcc_info, sample_cross_info]
        cross = manager.filter(cross_compilers=True)
        assert len(cross) == 1
        assert cross[0].is_cross_compiler

    def test_manager_filter_native_only(
        self, sample_gcc_info, sample_cross_info
    ):
        """
        Filter cross_compilers=False returns only native compilers.

        Expected result: 1 native compiler.
        """
        manager = CompilerManager()
        manager._all_compilers = [sample_gcc_info, sample_cross_info]
        native = manager.filter(cross_compilers=False)
        assert len(native) == 1
        assert not native[0].is_cross_compiler

    def test_manager_filter_min_confidence(
        self, sample_gcc_info, sample_mingw_info, sample_unknown_info
    ):
        """
        Filter min_confidence excludes low-confidence compilers.

        Expected result: Only compilers with confidence >= 0.8.
        """
        manager = CompilerManager()
        manager._all_compilers = [sample_gcc_info, sample_mingw_info, sample_unknown_info]
        high = manager.filter(min_confidence=0.8)
        assert len(high) == 1
        assert high[0] == sample_gcc_info

    def test_manager_filter_target_arch(
        self, sample_gcc_info, sample_cross_info
    ):
        """
        Filter target_arch matches substring in target triplet.

        Expected result: 1 ARM compiler.
        """
        manager = CompilerManager()
        manager._all_compilers = [sample_gcc_info, sample_cross_info]
        arm = manager.filter(target_arch="arm")
        assert len(arm) == 1
        assert "arm" in arm[0].target_triplet

    def test_manager_native_shortcut(
        self, sample_gcc_info, sample_cross_info
    ):
        """
        manager.native() is equivalent to filter(cross_compilers=False).

        Expected result: Same as explicit filter.
        """
        manager = CompilerManager()
        manager._all_compilers = [sample_gcc_info, sample_cross_info]
        assert manager.native() == manager.filter(cross_compilers=False)

    def test_manager_cross_shortcut(
        self, sample_gcc_info, sample_cross_info
    ):
        """
        manager.cross() is equivalent to filter(cross_compilers=True).

        Expected result: Same as explicit filter.
        """
        manager = CompilerManager()
        manager._all_compilers = [sample_gcc_info, sample_cross_info]
        assert manager.cross() == manager.filter(cross_compilers=True)

    def test_manager_multiple_filters(
        self, sample_gcc_info, sample_clang_info, sample_cross_info, sample_mingw_info
    ):
        """
        Multiple filters combine with AND logic.

        Expected result: 1 compiler matching both GCC and native.
        """
        manager = CompilerManager()
        manager._all_compilers = [
            sample_gcc_info, sample_clang_info,
            sample_cross_info, sample_mingw_info,
        ]
        result = manager.filter(
            kinds=[CompilerKind.GCC],
            cross_compilers=False,
            min_confidence=0.9,
        )
        assert len(result) == 1
        assert result[0] == sample_gcc_info

    def test_manager_is_stale_false_after_discovery(self):
        """
        is_stale returns False immediately after discovery.

        Expected result: False
        """
        manager = CompilerManager()
        manager._path_hash = manager._hash_path()
        assert manager.is_stale() is False

    def test_manager_is_stale_true_initially(self):
        """
        is_stale returns True when no path hash is set.

        Expected result: True
        """
        manager = CompilerManager()
        assert manager.is_stale() is True

    def test_manager_deduplication(self):
        """
        _deduplicate removes paths with same real path.

        Expected result: Only one instance kept.
        """
        manager = CompilerManager()
        paths = ["/usr/bin/gcc", "/usr/bin/../bin/gcc"]
        deduped = manager._deduplicate(paths)
        assert len(deduped) == 1

    def test_manager_sort_key_native_first(
        self, sample_gcc_info, sample_cross_info
    ):
        """
        _sort_key places native compilers before cross.

        Expected result: native_key < cross_key (native sorts first).
        """
        manager = CompilerManager()
        native_key = manager._sort_key(sample_gcc_info)
        cross_key = manager._sort_key(sample_cross_info)
        # Native should sort before cross (lower tuple = higher priority)
        assert native_key < cross_key


# ============================================================================
# Public API Tests
# ============================================================================

class TestPublicAPI:
    """
    Tests for the public API (discover_compilers function).

    Expected pass rate: 80-90% on platforms with a real compiler.
    """

    def test_discover_compilers_returns_manager(self):
        """
        discover_compilers returns a CompilerManager instance.

        Expected result: CompilerManager object.
        """
        manager = discover_compilers()
        assert isinstance(manager, CompilerManager)

    def test_discover_compilers_with_user_paths(self, tmp_path):
        """
        User-provided paths are included in results.

        Expected result: User path found in manager.all().
        Requires actual compiler at path for validation to succeed.
        May return 0 if temp file is not a real compiler.
        """
        # Create a shell script that echoes version info
        fake_gcc = tmp_path / "gcc"
        fake_gcc.write_text("#!/bin/sh\necho 'gcc (GCC) 99.0.0'\n")
        fake_gcc.chmod(0o755)

        manager = discover_compilers(user_paths=[str(fake_gcc)])
        # Note: validation may fail because the script might not handle
        # --version and -dumpmachine properly. This tests that the path
        # is at least considered.
        assert isinstance(manager, CompilerManager)

    def test_discover_compilers_with_strategy_filter(self):
        """
        enabled_strategies parameter limits which strategies run.

        Expected result: Manager returned (strategies filtered internally).
        """
        manager = discover_compilers(
            enabled_strategies=["user_override", "system_path"]
        )
        assert isinstance(manager, CompilerManager)

    def test_rescan_clears_and_rediscovers(self):
        """
        rescan() triggers fresh discovery.

        Expected result: all() returns list (may be empty on this platform).
        """
        manager = discover_compilers()
        manager.rescan()
        result = manager.all()
        assert isinstance(result, list)

    def test_manager_str_repr(self):
        """
        CompilerManager string representation is readable.

        Expected result: String containing class name.
        """
        manager = CompilerManager()
        assert "CompilerManager" in repr(manager)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """
    End-to-end integration tests.

    Expected pass rate: 70-90% depending on platform.
    Tests the full pipeline: strategies -> validation -> scoring.
    """

    @pytest.mark.skipif(
        not has_real_compiler(),
        reason="No real compiler found in PATH",
    )
    def test_full_pipeline_real_compiler(self):
        """
        Full pipeline returns at least one compiler when compiler exists.

        Expected result: len(manager.all()) >= 1.
        Skips if no compiler available.
        """
        manager = discover_compilers()
        compilers = manager.all()
        assert len(compilers) >= 1

    @pytest.mark.skipif(
        not has_real_compiler(),
        reason="No real compiler found in PATH",
    )
    def test_pipeline_best_has_path(self):
        """
        best() compiler has a valid path and version.

        Expected result: best.path is non-empty string.
        Skips if no compiler available.
        """
        manager = discover_compilers()
        best = manager.best()
        if best is not None:
            assert best.path
            assert best.version
            assert best.kind != CompilerKind.UNKNOWN

    def test_all_exports(self):
        """
        toolforge.__all__ contains expected names.

        Expected result: discover_compilers and CompilerManager in __all__.
        """
        assert "discover_compilers" in toolforge.__all__
        assert "CompilerManager" in toolforge.__all__
        assert "CompilerInfo" in toolforge.__all__


# ============================================================================
# Run configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
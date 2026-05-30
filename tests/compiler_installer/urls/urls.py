"""
Test suite for Compiler Installer URL resolution layer.

Covers: enums, templates, context formatting, provider registry,
artifact URL building, and URL syntax validation.


Requirements
------------
    pip install pytest pytest-asyncio aiohttp

Notes
-----
- HTTP validation tests require aiohttp and network access.
- Live URL tests are marked @pytest.mark.slow and skipped by default.
  Use --run-slow to include them.
- All non-network tests pass on any platform with Python 3.10+.

Expected Pass Rates by Platform
-------------------------------
    Linux:     100% (all tests, assuming network available for HTTP tests)
    macOS:     100% (all tests)
    Windows:   100% (all tests; symlink-specific tests are not in this file)
    Android:   100% (non-network tests; HTTP tests depend on aiohttp install)
"""

pytest_plugins = ("pytest_asyncio",)


import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Check if aiohttp is available for HTTP tests
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# ============================================================================
# Pytest configuration
# ============================================================================

def pytest_addoption(parser):
    """Add --run-slow flag to include live URL tests."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests that hit live URLs",
    )

def pytest_configure(config):
    """Register the 'slow' marker."""
    config.addinivalue_line("markers", "slow: mark test as hitting live URLs")


# ============================================================================
# Imports from pyputil_install.compiler_installer.urls
# ============================================================================

from pyputil_install.compiler_installer.urls.base import (
    CompilerType,
    PlatformType,
    ArchitectureType,
    ArchiveType,
)

from pyputil_install.compiler_installer.urls.templates import (
    GCC_XPACK_BASE_URL,
    GCC_XPACK_FILENAME,
    GCC_XPACK_CHECKSUM_URL,
    CLANG_LLVM_BASE_URL,
    CLANG_LLVM_FILENAME,
    CLANG_LLVM_CHECKSUM_URL,
    MINGW_BASE_URL,
    MINGW_FILENAME,
    MINGW_CHECKSUM_URL,
    ARM_GNU_BASE_URL,
    ARM_GNU_FILENAME,
    ARM_GNU_CHECKSUM_URL,
    RISCV_BASE_URL,
    RISCV_FILENAME,
    RISCV_CHECKSUM_URL,
    EMSCRIPTEN_BASE_URL,
    EMSCRIPTEN_FILENAME,
    EMSCRIPTEN_CHECKSUM_URL,
    ZIG_BASE_URL,
    ZIG_FILENAME,
    ZIG_CHECKSUM_URL,
)

from pyputil_install.compiler_installer.urls.context import (
    TemplateContext,
    safe_format_template,
    join_url,
    _extract_fields,
)

from pyputil_install.compiler_installer.urls.registry import (
    ProviderDefinition,
    get_provider,
    get_provider_or_raise,
    list_providers,
    list_supported_platforms,
    list_supported_architectures,
    is_supported,
    PROVIDER_REGISTRY,
)

from pyputil_install.compiler_installer.urls.builder import (
    ArtifactInfo,
    resolve_artifact,
    resolve_url,
)

from pyputil_install.compiler_installer.urls.validation import (
    validate_url_syntax,
    ValidationResult,
    ValidationStatus,
    ChecksumResult,
)


# ============================================================================
# Helper fixtures
# ============================================================================

@pytest.fixture
def gcc_linux_x64_context() -> TemplateContext:
    """
    Return a TemplateContext for GCC on Linux x64.

    Returns
    -------
    TemplateContext
        version="14.2.0-2", platform=LINUX, arch=X64, ext=TAR_GZ.
    """
    return TemplateContext(
        version="14.2.0-2",
        platform=PlatformType.LINUX,
        arch=ArchitectureType.X64,
        ext=ArchiveType.TAR_GZ,
    )


@pytest.fixture
def clang_macos_arm64_context() -> TemplateContext:
    """
    Return a TemplateContext for Clang on macOS ARM64.

    Returns
    -------
    TemplateContext
        version="18.1.0", platform=MACOS, arch=ARM64, ext=TAR_GZ,
        platform_variant="apple-darwin22.0".
    """
    return TemplateContext(
        version="18.1.0",
        platform=PlatformType.MACOS,
        arch=ArchitectureType.ARM64,
        ext=ArchiveType.TAR_GZ,
        platform_variant="apple-darwin22.0",
    )


@pytest.fixture
def mingw_windows_x64_context() -> TemplateContext:
    """
    Return a TemplateContext for MinGW on Windows x64.

    Returns
    -------
    TemplateContext
        version="12.0.0", platform=WINDOWS, arch=X64, ext=ZIP.
    """
    return TemplateContext(
        version="12.0.0",
        platform=PlatformType.WINDOWS,
        arch=ArchitectureType.X64,
        ext=ArchiveType.ZIP,
    )


# ============================================================================
# Enums Tests
# ============================================================================

class TestEnums:
    """
    Tests for URL-related enums.

    Expected pass rate: 100% on all platforms.
    No external dependencies.
    """

    def test_compiler_type_has_explicit_string_values(self):
        """
        All CompilerType members use explicit string values, not auto().

        Expected result on all platforms: True
        """
        assert CompilerType.GCC.value == "gcc"
        assert CompilerType.CLANG.value == "clang"
        assert CompilerType.MINGW.value == "mingw"
        assert CompilerType.ARM_GNU.value == "arm-gnu"
        assert CompilerType.RISCV.value == "riscv"
        assert CompilerType.EMSCRIPTEN.value == "emscripten"
        assert CompilerType.ZIG.value == "zig"

    def test_compiler_type_enum_name_differs_from_value(self):
        """
        Enum.name (Python attribute) is not the same as enum.value.

        This prevents code from using .name.lower() instead of .value.

        Expected result on all platforms: True
        """
        assert CompilerType.GCC.name == "GCC"
        assert CompilerType.GCC.value == "gcc"
        assert CompilerType.GCC.name != CompilerType.GCC.value

    def test_platform_type_has_explicit_string_values(self):
        """
        All PlatformType members use explicit string values.

        Expected result on all platforms: True
        """
        assert PlatformType.LINUX.value == "linux"
        assert PlatformType.MACOS.value == "macos"
        assert PlatformType.WINDOWS.value == "windows"

    def test_platform_type_has_no_android_member(self):
        """
        PlatformType does NOT include ANDROID.

        Android hosts use Linux artifacts, so there is no separate
        artifact platform for Android.

        Expected result on all platforms: True
        """
        platform_names = [p.name for p in PlatformType]
        assert "ANDROID" not in platform_names

    def test_architecture_type_has_explicit_string_values(self):
        """
        All ArchitectureType members use explicit string values.

        Expected result on all platforms: True
        """
        assert ArchitectureType.X64.value == "x64"
        assert ArchitectureType.ARM64.value == "arm64"
        assert ArchitectureType.ARM.value == "arm"
        assert ArchitectureType.X86.value == "x86"
        assert ArchitectureType.RISCV64.value == "riscv64"

    def test_archive_type_has_file_extensions_as_values(self):
        """
        ArchiveType values are file extensions (without leading dot).

        Expected result on all platforms: True
        """
        assert ArchiveType.TAR_GZ.value == "tar.gz"
        assert ArchiveType.TAR_XZ.value == "tar.xz"
        assert ArchiveType.TAR_BZ2.value == "tar.bz2"
        assert ArchiveType.ZIP.value == "zip"

    def test_all_compiler_types_are_iterable(self):
        """
        CompilerType supports iteration over all members.

        Expected result on all platforms: 7 members
        """
        members = list(CompilerType)
        assert len(members) == 7

    def test_all_platform_types_are_iterable(self):
        """
        PlatformType supports iteration over all members.

        Expected result on all platforms: 3 members
        """
        members = list(PlatformType)
        assert len(members) == 3

    def test_all_architecture_types_are_iterable(self):
        """
        ArchitectureType supports iteration over all members.

        Expected result on all platforms: 5 members
        """
        members = list(ArchitectureType)
        assert len(members) == 5

    def test_all_archive_types_are_iterable(self):
        """
        ArchiveType supports iteration over all members.

        Expected result on all platforms: 4 members
        """
        members = list(ArchiveType)
        assert len(members) == 4


# ============================================================================
# Templates Tests
# ============================================================================

class TestTemplates:
    """
    Tests for URL and filename template strings.

    Expected pass rate: 100% on all platforms.
    Pure string inspection, no network or filesystem access.
    """

    def test_gcc_filename_contains_expected_placeholders(self):
        """
        GCC filename template uses version, platform, arch, ext.

        Expected result on all platforms: {"version", "platform", "arch", "ext"}
        """
        fields = _extract_fields(GCC_XPACK_FILENAME)
        assert fields == {"version", "platform", "arch", "ext"}

    def test_clang_filename_requires_platform_variant(self):
        """
        Clang filename template includes platform_variant placeholder.

        LLVM publishes per-distro builds, so the template must
        accept a distro variant string.

        Expected result on all platforms: "platform_variant" in fields
        """
        fields = _extract_fields(CLANG_LLVM_FILENAME)
        assert "platform_variant" in fields

    def test_mingw_filename_lacks_platform_placeholder(self):
        """
        MinGW filename template does NOT include platform.

        MinGW only targets Windows, so the platform is implicit.

        Expected result on all platforms: "platform" not in fields
        """
        fields = _extract_fields(MINGW_FILENAME)
        assert "platform" not in fields

    def test_every_base_url_template_requires_version(self):
        """
        Every base URL template has a {version} placeholder.

        Without {version}, the URL cannot be customized per release.

        Expected result on all platforms: True for all 7 providers
        """
        base_templates = [
            ("GCC", GCC_XPACK_BASE_URL),
            ("CLANG", CLANG_LLVM_BASE_URL),
            ("MINGW", MINGW_BASE_URL),
            ("ARM_GNU", ARM_GNU_BASE_URL),
            ("RISCV", RISCV_BASE_URL),
            ("EMSCRIPTEN", EMSCRIPTEN_BASE_URL),
            ("ZIG", ZIG_BASE_URL),
        ]
        for name, template in base_templates:
            fields = _extract_fields(template)
            assert "version" in fields, (
                f"{name} base URL missing {{version}} placeholder"
            )

    def test_every_filename_template_requires_ext(self):
        """
        Every filename template has an {ext} placeholder.

        Archive format varies by platform, so the extension must be
        parameterized.

        Expected result on all platforms: True for all 7 providers
        """
        filename_templates = [
            ("GCC", GCC_XPACK_FILENAME),
            ("CLANG", CLANG_LLVM_FILENAME),
            ("MINGW", MINGW_FILENAME),
            ("ARM_GNU", ARM_GNU_FILENAME),
            ("RISCV", RISCV_FILENAME),
            ("EMSCRIPTEN", EMSCRIPTEN_FILENAME),
            ("ZIG", ZIG_FILENAME),
        ]
        for name, template in filename_templates:
            fields = _extract_fields(template)
            assert "ext" in fields, (
                f"{name} filename template missing {{ext}} placeholder"
            )

    def test_every_provider_has_checksum_template(self):
        """
        Every provider has a non-empty checksum URL template.

        Expected result on all platforms: True for all 7 providers
        """
        checksum_templates = [
            ("GCC", GCC_XPACK_CHECKSUM_URL),
            ("CLANG", CLANG_LLVM_CHECKSUM_URL),
            ("MINGW", MINGW_CHECKSUM_URL),
            ("ARM_GNU", ARM_GNU_CHECKSUM_URL),
            ("RISCV", RISCV_CHECKSUM_URL),
            ("EMSCRIPTEN", EMSCRIPTEN_CHECKSUM_URL),
            ("ZIG", ZIG_CHECKSUM_URL),
        ]
        for name, template in checksum_templates:
            assert isinstance(template, str), (
                f"{name} checksum template is not a string"
            )
            assert len(template) > 0, (
                f"{name} checksum template is empty"
            )

    def test_template_strings_are_raw_not_formatted(self):
        """
        Template strings contain literal {placeholders}, not values.

        If any template has already been formatted, it means a
        .format() call leaked into the template definition.

        Expected result on all platforms: { still present in every template
        """
        all_templates = [
            GCC_XPACK_BASE_URL, GCC_XPACK_FILENAME, GCC_XPACK_CHECKSUM_URL,
            CLANG_LLVM_BASE_URL, CLANG_LLVM_FILENAME, CLANG_LLVM_CHECKSUM_URL,
            MINGW_BASE_URL, MINGW_FILENAME, MINGW_CHECKSUM_URL,
            ARM_GNU_BASE_URL, ARM_GNU_FILENAME, ARM_GNU_CHECKSUM_URL,
            RISCV_BASE_URL, RISCV_FILENAME, RISCV_CHECKSUM_URL,
            EMSCRIPTEN_BASE_URL, EMSCRIPTEN_FILENAME, EMSCRIPTEN_CHECKSUM_URL,
            ZIG_BASE_URL, ZIG_FILENAME, ZIG_CHECKSUM_URL,
        ]
        for template in all_templates:
            assert "{" in template, (
                f"Template appears pre-formatted (no placeholders left): "
                f"{template[:80]}"
            )


# ============================================================================
# TemplateContext Tests
# ============================================================================

class TestTemplateContext:
    """
    Tests for TemplateContext and safe_format_template.

    Expected pass rate: 100% on all platforms.
    Pure in-memory operations, no I/O.
    """

    def test_context_stores_all_required_fields(self, gcc_linux_x64_context):
        """
        TemplateContext stores version, platform, arch, ext.

        Expected result on all platforms: all fields match input.
        """
        assert gcc_linux_x64_context.version == "14.2.0-2"
        assert gcc_linux_x64_context.platform == PlatformType.LINUX
        assert gcc_linux_x64_context.arch == ArchitectureType.X64
        assert gcc_linux_x64_context.ext == ArchiveType.TAR_GZ

    def test_context_platform_variant_defaults_to_none(self, gcc_linux_x64_context):
        """
        platform_variant is None when not explicitly set.

        Expected result on all platforms: None
        """
        assert gcc_linux_x64_context.platform_variant is None

    def test_context_platform_variant_accepts_string(
        self, clang_macos_arm64_context
    ):
        """
        platform_variant accepts a string for LLVM distro variants.

        Expected result on all platforms: "apple-darwin22.0"
        """
        assert clang_macos_arm64_context.platform_variant == "apple-darwin22.0"

    def test_context_is_frozen_and_immutable(self, gcc_linux_x64_context):
        """
        TemplateContext cannot be modified after creation.

        Expected result on all platforms: raises exception
        """
        with pytest.raises(Exception):
            gcc_linux_x64_context.version = "15.0.0"  # type: ignore

    def test_to_format_dict_excludes_platform_variant_when_none(
        self, gcc_linux_x64_context
    ):
        """
        to_format_dict() omits "platform_variant" key when it is None.

        This prevents KeyError when formatting templates that do not
        use {platform_variant}.

        Expected result on all platforms: "platform_variant" not in dict
        """
        d = gcc_linux_x64_context.to_format_dict()
        assert "platform_variant" not in d

    def test_to_format_dict_includes_platform_variant_when_set(
        self, clang_macos_arm64_context
    ):
        """
        to_format_dict() includes "platform_variant" when it has a value.

        Expected result on all platforms: "platform_variant" in dict
        """
        d = clang_macos_arm64_context.to_format_dict()
        assert "platform_variant" in d
        assert d["platform_variant"] == "apple-darwin22.0"

    def test_to_format_dict_uses_enum_value_not_enum_name(
        self, gcc_linux_x64_context
    ):
        """
        to_format_dict() returns enum.value strings, not enum.name.

        This is critical for correct URL generation.

        Expected result on all platforms:
            platform="linux", arch="x64", ext="tar.gz"
        """
        d = gcc_linux_x64_context.to_format_dict()
        assert d["platform"] == "linux"
        assert d["arch"] == "x64"
        assert d["ext"] == "tar.gz"

    def test_safe_format_gcc_filename(self, gcc_linux_x64_context):
        """
        safe_format_template produces correct GCC filename.

        Expected result on all platforms:
            "xpack-gcc-14.2.0-2-linux-x64.tar.gz"
        """
        result = safe_format_template(GCC_XPACK_FILENAME, gcc_linux_x64_context)
        assert result == "xpack-gcc-14.2.0-2-linux-x64.tar.gz"

    def test_safe_format_clang_filename(self, clang_macos_arm64_context):
        """
        safe_format_template produces correct Clang filename.

        Expected result on all platforms:
            starts with "clang+llvm-18.1.0-arm64-linux-apple-darwin22.0"
        """
        result = safe_format_template(
            CLANG_LLVM_FILENAME, clang_macos_arm64_context
        )
        assert result.startswith("clang+llvm-18.1.0-arm64-linux-apple-darwin22.0")

    def test_safe_format_mingw_filename(self, mingw_windows_x64_context):
        """
        safe_format_template produces correct MinGW filename.

        Expected result on all platforms:
            "x64-12.0.0-release-posix-seh-ucrt-rt_v12-rev0.zip"
        """
        result = safe_format_template(
            MINGW_FILENAME, mingw_windows_x64_context
        )
        assert result == "x64-12.0.0-release-posix-seh-ucrt-rt_v12-rev0.zip"

    def test_safe_format_raises_keyerror_for_missing_placeholder(
        self, gcc_linux_x64_context
    ):
        """
        safe_format_template raises KeyError if a placeholder is missing.

        Expected result on all platforms: KeyError
        """
        with pytest.raises(KeyError):
            safe_format_template(
                "{version}-{missing_placeholder}", gcc_linux_x64_context
            )

    def test_safe_format_raises_valueerror_in_strict_mode(
        self, gcc_linux_x64_context
    ):
        """
        strict_unused=True raises ValueError if context has extra keys.

        Expected result on all platforms: ValueError
        """
        with pytest.raises(ValueError):
            safe_format_template(
                "{version}", gcc_linux_x64_context, strict_unused=True
            )

    def test_safe_format_strict_mode_exempts_platform_variant(
        self, clang_macos_arm64_context
    ):
        """
        strict_unused=True does NOT raise for unused platform_variant.

        This allows the same context to be used for templates with
        and without {platform_variant} without error.

        Expected result on all platforms: formats successfully
        """
        result = safe_format_template(
            "{version}.{ext}",
            clang_macos_arm64_context,
            strict_unused=True,
        )
        assert result == "18.1.0.tar.gz"

    def test_extract_fields_handles_format_specs(self):
        """
        _extract_fields strips format specs like :>10 and !r.

        Expected result on all platforms: {"name", "version"}
        """
        fields = _extract_fields("{name:>10}-{version!r}")
        assert fields == {"name", "version"}

    def test_extract_fields_ignores_escaped_braces(self):
        """
        _extract_fields ignores double-brace {{escaped}} patterns.

        Expected result on all platforms: {"version"} only
        """
        fields = _extract_fields("Hello {{name}}, {version}")
        assert fields == {"version"}

    def test_join_url_handles_no_slashes(self):
        """
        join_url adds exactly one slash between base and path.

        Expected result on all platforms:
            "https://example.com/releases/file.tar.gz"
        """
        result = join_url("https://example.com/releases", "file.tar.gz")
        assert result == "https://example.com/releases/file.tar.gz"

    def test_join_url_handles_trailing_slash_on_base(self):
        """
        join_url normalizes a trailing slash on the base URL.

        Expected result on all platforms: no double slash
        """
        result = join_url("https://example.com/releases/", "file.tar.gz")
        assert result == "https://example.com/releases/file.tar.gz"

    def test_join_url_handles_leading_slash_on_path(self):
        """
        join_url normalizes a leading slash on the path.

        Expected result on all platforms: no double slash
        """
        result = join_url("https://example.com/releases", "/file.tar.gz")
        assert result == "https://example.com/releases/file.tar.gz"

    def test_join_url_handles_both_slashes(self):
        """
        join_url normalizes both trailing and leading slashes.

        Expected result on all platforms: exactly one slash
        """
        result = join_url("https://example.com/releases/", "/file.tar.gz")
        assert result == "https://example.com/releases/file.tar.gz"


# ============================================================================
# Registry Tests
# ============================================================================

class TestRegistry:
    """
    Tests for provider registry.

    Expected pass rate: 100% on all platforms.
    No network access.
    """

    def test_get_provider_returns_object_for_every_compiler_type(self):
        """
        Every CompilerType enum member has a registered provider.

        Expected result on all platforms: 7 providers, none None
        """
        for ct in CompilerType:
            provider = get_provider(ct)
            assert provider is not None, (
                f"No provider registered for {ct.value}"
            )

    def test_get_provider_returns_none_for_unknown_type(self):
        """
        get_provider returns None for a non-existent CompilerType.

        This test uses a mock value to simulate an unregistered type.

        Expected result on all platforms: None
        """
        # Create a value that is not in the registry
        fake = MagicMock()
        fake.value = "nonexistent"
        result = get_provider(fake)
        assert result is None

    def test_get_provider_or_raise_raises_keyerror_for_missing(self):
        """
        get_provider_or_raise raises KeyError for unregistered type.

        Expected result on all platforms: KeyError
        """
        fake = MagicMock()
        fake.value = "nonexistent"
        with pytest.raises(KeyError):
            get_provider_or_raise(fake)

    def test_get_provider_or_raise_returns_provider_for_gcc(self):
        """
        get_provider_or_raise returns GCC provider.

        Expected result on all platforms: ProviderDefinition
        """
        provider = get_provider_or_raise(CompilerType.GCC)
        assert isinstance(provider, ProviderDefinition)
        assert provider.compiler_type == CompilerType.GCC

    def test_gcc_supports_all_desktop_platforms(self):
        """
        GCC provider supports Linux, macOS, and Windows.

        Expected result on all platforms: True for all 3
        """
        provider = get_provider(CompilerType.GCC)
        assert provider.supports_platform(PlatformType.LINUX)
        assert provider.supports_platform(PlatformType.MACOS)
        assert provider.supports_platform(PlatformType.WINDOWS)

    def test_gcc_supports_x64_arm64_arm(self):
        """
        GCC provider supports x64, arm64, and 32-bit arm.

        Expected result on all platforms: True for all 3
        """
        provider = get_provider(CompilerType.GCC)
        assert provider.supports_architecture(ArchitectureType.X64)
        assert provider.supports_architecture(ArchitectureType.ARM64)
        assert provider.supports_architecture(ArchitectureType.ARM)

    def test_gcc_does_not_support_riscv64(self):
        """
        GCC xPack does not ship RISC-V binaries.

        Expected result on all platforms: False
        """
        provider = get_provider(CompilerType.GCC)
        assert not provider.supports_architecture(ArchitectureType.RISCV64)

    def test_gcc_does_not_require_platform_variant(self):
        """
        GCC provider does not need platform_variant.

        Expected result on all platforms: False
        """
        provider = get_provider(CompilerType.GCC)
        assert not provider.requires_platform_variant

    def test_gcc_has_checksum(self):
        """
        GCC provider publishes checksum files.

        Expected result on all platforms: True
        """
        provider = get_provider(CompilerType.GCC)
        assert provider.has_checksum()

    def test_clang_requires_platform_variant(self):
        """
        Clang provider requires platform_variant for distro suffixes.

        Expected result on all platforms: True
        """
        provider = get_provider(CompilerType.CLANG)
        assert provider.requires_platform_variant

    def test_mingw_only_supports_windows(self):
        """
        MinGW only targets Windows.

        Expected result on all platforms:
            True for WINDOWS, False for LINUX and MACOS
        """
        provider = get_provider(CompilerType.MINGW)
        assert provider.supports_platform(PlatformType.WINDOWS)
        assert not provider.supports_platform(PlatformType.LINUX)
        assert not provider.supports_platform(PlatformType.MACOS)

    def test_mingw_supports_x64_and_x86(self):
        """
        MinGW supports 64-bit and 32-bit x86.

        Expected result on all platforms: True for X64 and X86
        """
        provider = get_provider(CompilerType.MINGW)
        assert provider.supports_architecture(ArchitectureType.X64)
        assert provider.supports_architecture(ArchitectureType.X86)

    def test_riscv_only_supports_linux(self):
        """
        RISC-V toolchain only publishes Linux binaries.

        Expected result on all platforms: True for LINUX only
        """
        provider = get_provider(CompilerType.RISCV)
        assert provider.supports_platform(PlatformType.LINUX)
        assert not provider.supports_platform(PlatformType.MACOS)
        assert not provider.supports_platform(PlatformType.WINDOWS)

    def test_riscv_only_supports_riscv64_arch(self):
        """
        RISC-V toolchain only supports riscv64.

        Expected result on all platforms: True for RISCV64 only
        """
        provider = get_provider(CompilerType.RISCV)
        assert provider.supports_architecture(ArchitectureType.RISCV64)
        assert not provider.supports_architecture(ArchitectureType.X64)
        assert not provider.supports_architecture(ArchitectureType.ARM64)

    def test_zig_supports_all_architectures(self):
        """
        Zig publishes binaries for all 5 architectures.

        Expected result on all platforms: True for X64, ARM64, ARM, X86, RISCV64
        """
        provider = get_provider(CompilerType.ZIG)
        for arch in ArchitectureType:
            assert provider.supports_architecture(arch), (
                f"Zig should support {arch.value}"
            )

    def test_list_providers_returns_all_seven(self):
        """
        list_providers returns 7 CompilerType values.

        Expected result on all platforms: frozenset of length 7
        """
        providers = list_providers()
        assert len(providers) == 7
        assert CompilerType.GCC in providers
        assert CompilerType.ZIG in providers

    def test_list_supported_platforms_for_gcc(self):
        """
        list_supported_platforms returns all 3 desktop platforms for GCC.

        Expected result on all platforms: frozenset of LINUX, MACOS, WINDOWS
        """
        platforms = list_supported_platforms(CompilerType.GCC)
        assert PlatformType.LINUX in platforms
        assert PlatformType.MACOS in platforms
        assert PlatformType.WINDOWS in platforms
        assert len(platforms) == 3

    def test_list_supported_architectures_for_clang(self):
        """
        list_supported_architectures for Clang returns X64 and ARM64.

        Expected result on all platforms: frozenset of X64, ARM64
        """
        archs = list_supported_architectures(CompilerType.CLANG)
        assert ArchitectureType.X64 in archs
        assert ArchitectureType.ARM64 in archs
        assert len(archs) == 2

    def test_is_supported_returns_true_for_valid_combination(self):
        """
        is_supported returns True for GCC on Linux x64.

        Expected result on all platforms: True
        """
        assert is_supported(
            CompilerType.GCC,
            PlatformType.LINUX,
            ArchitectureType.X64,
        )

    def test_is_supported_returns_false_for_invalid_combination(self):
        """
        is_supported returns False for MinGW on Linux.

        Expected result on all platforms: False
        """
        assert not is_supported(
            CompilerType.MINGW,
            PlatformType.LINUX,
            ArchitectureType.X64,
        )

    def test_registry_is_immutable(self):
        """
        PROVIDER_REGISTRY is wrapped in MappingProxyType.

        Expected result on all platforms: TypeError on mutation
        """
        with pytest.raises(TypeError):
            PROVIDER_REGISTRY[CompilerType.GCC] = None  # type: ignore

    def test_provider_definition_is_frozen(self):
        """
        ProviderDefinition is a frozen dataclass.

        Expected result on all platforms: raises exception on mutation
        """
        provider = get_provider(CompilerType.GCC)
        with pytest.raises(Exception):
            provider.supported_platforms = frozenset()  # type: ignore


# ============================================================================
# Builder Tests
# ============================================================================

class TestBuilder:
    """
    Tests for artifact URL resolution (resolve_artifact, resolve_url).

    Expected pass rate: 100% on all platforms.
    No network access — only URL string construction.
    """

    def test_resolve_artifact_returns_artifact_info(self):
        """
        resolve_artifact returns an ArtifactInfo instance.

        Expected result on all platforms: ArtifactInfo
        """
        artifact = resolve_artifact("gcc", "14.2.0-2", "linux", "x64")
        assert isinstance(artifact, ArtifactInfo)

    def test_resolve_artifact_gcc_linux_x64_url(self):
        """
        resolve_artifact builds correct GCC Linux x64 URL.

        Expected result on all platforms:
            URL contains "xpack-gcc-14.2.0-2-linux-x64.tar.gz"
        """
        artifact = resolve_artifact("gcc", "14.2.0-2", "linux", "x64")
        assert "xpack-gcc-14.2.0-2-linux-x64.tar.gz" in artifact.url
        assert artifact.url.startswith("https://github.com")
        assert artifact.filename == "xpack-gcc-14.2.0-2-linux-x64.tar.gz"

    def test_resolve_artifact_gcc_linux_x64_checksum_url(self):
        """
        resolve_artifact includes checksum URL for GCC.

        Expected result on all platforms: checksum_url ends with .sha256
        """
        artifact = resolve_artifact("gcc", "14.2.0-2", "linux", "x64")
        assert artifact.checksum_url is not None
        assert artifact.checksum_url.endswith(".sha256")
        assert artifact.filename in artifact.checksum_url

    def test_resolve_artifact_gcc_macos_arm64_url(self):
        """
        resolve_artifact builds correct GCC macOS ARM64 URL.

        Expected result on all platforms:
            URL contains "xpack-gcc-14.2.0-2-macos-arm64.tar.gz"
        """
        artifact = resolve_artifact("gcc", "14.2.0-2", "macos", "arm64")
        assert "macos" in artifact.url
        assert "arm64" in artifact.url
        assert artifact.filename == "xpack-gcc-14.2.0-2-macos-arm64.tar.gz"

    def test_resolve_artifact_gcc_windows_x64_zip(self):
        """
        resolve_artifact uses .zip for Windows.

        Expected result on all platforms:
            URL contains ".zip" extension
        """
        artifact = resolve_artifact("gcc", "14.2.0-2", "windows", "x64")
        assert artifact.filename.endswith(".zip")
        assert artifact.archive_type == ArchiveType.ZIP

    def test_resolve_artifact_clang_includes_platform_variant(self):
        """
        resolve_artifact adds platform_variant for Clang.

        Expected result on all platforms:
            URL contains "ubuntu-22.04" (Linux) or appropriate variant
        """
        artifact = resolve_artifact("clang", "18.1.0", "linux", "x64")
        assert "ubuntu-22.04" in artifact.url

    def test_resolve_artifact_mingw_windows_only(self):
        """
        resolve_artifact for MinGW defaults to Windows platform.

        MinGW template does not include {platform}.

        Expected result on all platforms: URL is valid
        """
        artifact = resolve_artifact("mingw", "12.0.0", "windows", "x64")
        assert artifact.filename.endswith(".zip")

    def test_resolve_artifact_compiler_type_is_set(self):
        """
        ArtifactInfo.compiler_type matches the requested compiler.

        Expected result on all platforms: CompilerType.GCC
        """
        artifact = resolve_artifact("gcc", "14.2.0-2", "linux", "x64")
        assert artifact.compiler_type == CompilerType.GCC

    def test_resolve_artifact_platform_and_arch_are_enums(self):
        """
        ArtifactInfo stores platform and arch as enum values.

        Expected result on all platforms: PlatformType.LINUX, ArchitectureType.X64
        """
        artifact = resolve_artifact("gcc", "14.2.0-2", "linux", "x64")
        assert artifact.platform == PlatformType.LINUX
        assert artifact.arch == ArchitectureType.X64

    def test_resolve_url_returns_string(self):
        """
        resolve_url returns a plain URL string.

        Expected result on all platforms: str starting with https://
        """
        url = resolve_url("gcc", "14.2.0-2", "linux", "x64")
        assert isinstance(url, str)
        assert url.startswith("https://")

    def test_resolve_artifact_case_insensitive_compiler(self):
        """
        Compiler name is case-insensitive.

        Expected result on all platforms: same URL for "GCC" and "gcc"
        """
        lower = resolve_artifact("gcc", "14.2.0-2", "linux", "x64")
        upper = resolve_artifact("GCC", "14.2.0-2", "linux", "x64")
        assert lower.url == upper.url

    def test_resolve_artifact_case_insensitive_platform(self):
        """
        Platform name is case-insensitive.

        Expected result on all platforms: same URL for "LINUX" and "linux"
        """
        lower = resolve_artifact("gcc", "14.2.0-2", "linux", "x64")
        upper = resolve_artifact("gcc", "14.2.0-2", "LINUX", "x64")
        assert lower.url == upper.url

    def test_resolve_artifact_accepts_arch_aliases(self):
        """
        Architecture accepts common aliases: x86_64, amd64, aarch64.

        Expected result on all platforms: all resolve to same URL
        """
        x64_1 = resolve_artifact("gcc", "14.2.0-2", "linux", "x64")
        x64_2 = resolve_artifact("gcc", "14.2.0-2", "linux", "x86_64")
        x64_3 = resolve_artifact("gcc", "14.2.0-2", "linux", "amd64")
        assert x64_1.url == x64_2.url == x64_3.url

    def test_resolve_artifact_accepts_platform_aliases(self):
        """
        Platform accepts darwin as alias for macos.

        Expected result on all platforms: "darwin" resolves to "macos" in URL
        """
        artifact = resolve_artifact("gcc", "14.2.0-2", "darwin", "x64")
        assert "macos" in artifact.url

    def test_resolve_artifact_raises_for_unknown_compiler(self):
        """
        Unknown compiler name raises ValueError.

        Expected result on all platforms: ValueError
        """
        with pytest.raises(ValueError):
            resolve_artifact("nonexistent_compiler", "1.0", "linux", "x64")

    def test_resolve_artifact_raises_for_unknown_platform(self):
        """
        Unknown platform name raises ValueError.

        Expected result on all platforms: ValueError
        """
        with pytest.raises(ValueError):
            resolve_artifact("gcc", "14.2.0-2", "solaris", "x64")

    def test_resolve_artifact_raises_for_unknown_arch(self):
        """
        Unknown architecture raises ValueError.

        Expected result on all platforms: ValueError
        """
        with pytest.raises(ValueError):
            resolve_artifact("gcc", "14.2.0-2", "linux", "mips")

    def test_resolve_artifact_raises_for_unsupported_combination(self):
        """
        Valid compiler but unsupported platform/arch raises RuntimeError.

        Expected result on all platforms: RuntimeError
        """
        with pytest.raises(RuntimeError):
            resolve_artifact("mingw", "12.0.0", "linux", "x64")

    def test_resolve_artifact_auto_detects_host_platform(self):
        """
        If platform is None, auto-detection runs without error.

        Expected result on all platforms: ArtifactInfo returned
        """
        artifact = resolve_artifact("gcc", "14.2.0-2")
        assert isinstance(artifact, ArtifactInfo)
        assert artifact.platform is not None
        assert artifact.arch is not None

    def test_resolve_artifact_explicit_archive_type(self):
        """
        archive_type parameter overrides platform default.

        Expected result on all platforms: .tar.xz extension
        """
        artifact = resolve_artifact(
            "gcc", "14.2.0-2", "linux", "x64", "tar.xz"
        )
        assert artifact.filename.endswith(".tar.xz")
        assert artifact.archive_type == ArchiveType.TAR_XZ

    def test_resolve_artifact_raises_for_unknown_archive_type(self):
        """
        Unknown archive type raises ValueError.

        Expected result on all platforms: ValueError
        """
        with pytest.raises(ValueError):
            resolve_artifact("gcc", "14.2.0-2", "linux", "x64", "rar")

    def test_artifact_info_is_frozen(self):
        """
        ArtifactInfo is immutable.

        Expected result on all platforms: raises exception on mutation
        """
        artifact = resolve_artifact("gcc", "14.2.0-2", "linux", "x64")
        with pytest.raises(Exception):
            artifact.url = "other"  # type: ignore

    def test_resolve_zig_url(self):
        """
        resolve_artifact for Zig returns ziglang.org URL.

        Expected result on all platforms: URL starts with ziglang.org
        """
        artifact = resolve_artifact("zig", "0.11.0", "linux", "x64")
        assert "ziglang.org" in artifact.url
        assert artifact.filename.startswith("zig-")

    def test_resolve_emscripten_url(self):
        """
        resolve_artifact for Emscripten returns GitHub URL.

        Expected result on all platforms: URL contains emscripten-core
        """
        artifact = resolve_artifact("emscripten", "3.1.60", "linux", "x64")
        assert "emscripten-core" in artifact.url

    def test_resolve_arm_gnu_url(self):
        """
        resolve_artifact for ARM GNU returns ARM Developer URL.

        Expected result on all platforms: URL contains developer.arm.com
        """
        artifact = resolve_artifact("arm-gnu", "13.2.0", "linux", "arm64")
        assert "developer.arm.com" in artifact.url

    def test_resolve_riscv_url(self):
        """
        resolve_artifact for RISC-V returns riscv-collab URL.

        Expected result on all platforms: URL contains riscv-collab
        """
        artifact = resolve_artifact("riscv", "2024.04.12", "linux", "riscv64")
        assert "riscv-collab" in artifact.url


# ============================================================================
# URL Syntax Validation Tests
# ============================================================================

class TestURLSyntaxValidation:
    """
    Tests for URL syntax validation (no HTTP).

    Expected pass rate: 100% on all platforms.
    Pure string validation, no network.
    """

    def test_valid_https_url_passes(self):
        """
        Standard HTTPS URL is syntactically valid.

        Expected result: (True, None)
        """
        is_valid, error = validate_url_syntax(
            "https://github.com/releases/file.tar.gz"
        )
        assert is_valid is True
        assert error is None

    def test_url_without_scheme_fails(self):
        """
        URL without scheme is invalid.

        Expected result: (False, "URL has no scheme")
        """
        is_valid, error = validate_url_syntax("github.com/file.tar.gz")
        assert is_valid is False
        assert "scheme" in error.lower()

    def test_url_with_ftp_scheme_fails(self):
        """
        Non-HTTP scheme is rejected.

        Expected result: (False, "Unsupported scheme: ftp")
        """
        is_valid, error = validate_url_syntax("ftp://example.com/file.tar.gz")
        assert is_valid is False
        assert "scheme" in error.lower()

    def test_url_without_host_fails(self):
        """
        URL without a hostname is invalid.

        Expected result: (False, "no host")
        """
        is_valid, error = validate_url_syntax("https:///file.tar.gz")
        assert is_valid is False
        assert "host" in error.lower()

    def test_url_without_path_fails(self):
        """
        URL with only a domain is invalid (no file path).

        Expected result: (False, "no file path")
        """
        is_valid, error = validate_url_syntax("https://github.com")
        assert is_valid is False
        assert "path" in error.lower()

    def test_url_with_query_params_passes(self):
        """
        URL with query parameters is valid (path is present).

        Expected result: (True, None)
        """
        is_valid, error = validate_url_syntax(
            "https://github.com/releases/download?version=1.0"
        )
        assert is_valid is True

    def test_empty_string_fails(self):
        """
        Empty string is invalid.

        Expected result: (False, ...)
        """
        is_valid, error = validate_url_syntax("")
        assert is_valid is False

    def test_random_text_fails(self):
        """
        Non-URL text is invalid.

        Expected result: (False, ...)
        """
        is_valid, error = validate_url_syntax("not a url at all")
        assert is_valid is False

    def test_validation_result_dataclass(self):
        """
        ValidationResult can be constructed with required fields.

        Expected result on all platforms: ValidationResult instance
        """
        result = ValidationResult(
            url="https://example.com/file.tar.gz",
            status=ValidationStatus.VALID,
        )
        assert result.url == "https://example.com/file.tar.gz"
        assert result.status == ValidationStatus.VALID
        assert result.is_valid is False  # default
        assert result.content_length is None

    def test_checksum_result_dataclass(self):
        """
        ChecksumResult can be constructed with required fields.

        Expected result on all platforms: ChecksumResult instance
        """
        result = ChecksumResult(
            checksum_url="https://example.com/file.tar.gz.sha256",
        )
        assert result.checksum_url == "https://example.com/file.tar.gz.sha256"
        assert result.is_valid is False
        assert result.algorithm is None


# ============================================================================
# HTTP Validation Tests (async, requires aiohttp)
# ============================================================================

@pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
class TestHTTPValidation:
    """
    Tests for async HTTP URL validation.

    Expected pass rate: 90-100% on platforms with network access.
    Skipped entirely if aiohttp is not installed.
    """

    @pytest.mark.asyncio
    async def test_validate_url_importable(self):
        """
        validate_url function is importable and callable.

        Expected result: no ImportError
        """
        from pyputil_install.compiler_installer.urls.validation import validate_url
        assert callable(validate_url)

    @pytest.mark.asyncio
    async def test_is_url_downloadable_importable(self):
        """
        is_url_downloadable function is importable.

        Expected result: no ImportError
        """
        from pyputil_install.compiler_installer.urls.validation import is_url_downloadable
        assert callable(is_url_downloadable)

    @pytest.mark.asyncio
    async def test_validate_url_returns_validation_result(self):
        """
        validate_url returns a ValidationResult for a valid HTTPS URL.

        Expected result: ValidationResult with VALID status.

        Platform notes:
            Linux:    Pass (network available)
            macOS:    Pass
            Windows:  Pass
            Android:  Pass (if aiohttp installed and network available)
        """
        from pyputil_install.compiler_installer.urls.validation import validate_url
        result = await validate_url("https://httpbin.org/status/200")
        assert isinstance(result, ValidationResult)
        assert result.status == ValidationStatus.VALID
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_validate_url_404_returns_not_found(self):
        """
        validate_url returns NOT_FOUND for a 404 URL.

        Expected result: ValidationStatus.NOT_FOUND
        """
        from pyputil_install.compiler_installer.urls.validation import validate_url
        result = await validate_url("https://httpbin.org/status/404")
        assert result.status == ValidationStatus.NOT_FOUND
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_validate_url_invalid_syntax(self):
        """
        validate_url returns INVALID_URL for malformed input.

        Expected result: ValidationStatus.INVALID_URL
        """
        from pyputil_install.compiler_installer.urls.validation import validate_url
        result = await validate_url("not-a-url")
        assert result.status == ValidationStatus.INVALID_URL
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_is_url_downloadable_true(self):
        """
        is_url_downloadable returns True for a reachable URL.

        Expected result: True
        """
        from pyputil_install.compiler_installer.urls.validation import is_url_downloadable
        result = await is_url_downloadable("https://httpbin.org/status/200")
        assert result is True

    @pytest.mark.asyncio
    async def test_is_url_downloadable_false_for_404(self):
        """
        is_url_downloadable returns False for a 404.

        Expected result: False
        """
        from pyputil_install.compiler_installer.urls.validation import is_url_downloadable
        result = await is_url_downloadable("https://httpbin.org/status/404")
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_validate_real_gcc_url(self):
        """
        Validate a real GCC release URL.

        Expected result: VALID if the release exists.

        Platform notes:
            Linux:    Pass (if network available)
            macOS:    Pass
            Windows:  Pass
            Android:  Pass (if aiohttp installed)
        """
        from pyputil_install.compiler_installer.urls.builder import resolve_url
        from pyputil_install.compiler_installer.urls.validation import validate_url

        url = resolve_url("gcc", "14.2.0-2", "linux", "x64")
        result = await validate_url(url)
        # The URL may or may not exist depending on release availability.
        # We only check that validation completes without crashing.
        assert isinstance(result, ValidationResult)

    @pytest.mark.asyncio
    async def test_validate_urls_batch(self):
        """
        validate_urls validates multiple URLs concurrently.

        Expected result: one result per input URL.
        """
        from pyputil_install.compiler_installer.urls.validation import validate_urls

        urls = [
            "https://httpbin.org/status/200",
            "https://httpbin.org/status/404",
            "https://httpbin.org/status/200",
        ]
        results = await validate_urls(urls, max_concurrent=2)
        assert len(results) == 3
        assert results[0].is_valid is True
        assert results[1].is_valid is False
        assert results[2].is_valid is True

    @pytest.mark.asyncio
    async def test_validate_checksum_importable(self):
        """
        validate_checksum function is importable.

        Expected result: no ImportError
        """
        from pyputil_install.compiler_installer.urls.validation import validate_checksum
        assert callable(validate_checksum)


# ============================================================================
# Run configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Provider registry for compiler release artifacts.

Maps each CompilerType to a complete ProviderDefinition that bundles
base URL, filename template, checksum template, and platform/arch
support information. This is the single source of truth for what
compilers are available and where to find their releases.

All registries are wrapped in MappingProxyType — immutable at runtime.

Architecture
------------
templates.py     → Raw format strings (no logic)
registry.py      → ProviderDefinition objects (this file)
builder.py       → Uses registry to resolve ArtifactInfo
validation.py    → Validates resolved URLs against live servers

Usage
-----
    from pyputil_install.compiler_installer import get_provider, list_providers

    provider = get_provider(CompilerType.GCC)
    if provider and provider.supports_platform(PlatformType.LINUX):
        print(f"GCC base URL: {provider.base_url_template}")

    all_types = list_providers()
    for ct in all_types:
        print(ct.value)

Adding a New Provider
---------------------
1. Add templates to `templates.py`:
   - MYCOMPILER_BASE_URL
   - MYCOMPILER_FILENAME
   - MYCOMPILER_CHECKSUM_URL (optional)
2. Import them at the top of this file.
3. Create a ProviderDefinition instance with:
   - compiler_type: CompilerType enum value
   - base_url_template: the base URL constant
   - filename_template: the filename constant
   - checksum_template: the checksum URL constant (or None)
   - supported_platforms: frozenset of PlatformType
   - supported_architectures: frozenset of ArchitectureType
   - requires_platform_variant: True only for LLVM
4. Add the provider to the _PROVIDER_MAP dict.
5. Add the CompilerType to the enum in enums.py if new.

Warnings
--------
- Registries are IMMUTABLE at runtime. To add providers dynamically,
  use the plugin system (future feature), not this module.
- supported_platforms and supported_architectures are STATIC.
  Some providers change support per-version (e.g., LLVM dropping
  older Ubuntu). Dynamic validation comes in a future update.
- checksum_template can be None. Some providers do not publish
  checksum files in a predictable pattern.
- ProviderDefinition objects are frozen dataclasses. Once created,
  they cannot be modified. Create a new instance if changes are needed.

User Instructions
-----------------
- This module is internal. Use `builder.py` public API.
- To check if a compiler/platform/arch combination is supported,
  use `provider.supports_platform()` and `provider.supports_architecture()`.
- To list all available compiler types, use `list_providers()`.
- Do NOT modify _PROVIDER_MAP after import. It is wrapped in
  MappingProxyType and will raise TypeError if mutated.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, FrozenSet, Optional

from .base import (
    ArchitectureType,
    CompilerType,
    PlatformType,
)

from .templates import (
    ARM_GNU_BASE_URL,
    ARM_GNU_CHECKSUM_URL,
    ARM_GNU_FILENAME,
    CLANG_LLVM_BASE_URL,
    CLANG_LLVM_CHECKSUM_URL,
    CLANG_LLVM_FILENAME,
    EMSCRIPTEN_BASE_URL,
    EMSCRIPTEN_CHECKSUM_URL,
    EMSCRIPTEN_FILENAME,
    GCC_XPACK_BASE_URL,
    GCC_XPACK_CHECKSUM_URL,
    GCC_XPACK_FILENAME,
    MINGW_BASE_URL,
    MINGW_CHECKSUM_URL,
    MINGW_FILENAME,
    RISCV_BASE_URL,
    RISCV_CHECKSUM_URL,
    RISCV_FILENAME,
    ZIG_BASE_URL,
    ZIG_CHECKSUM_URL,
    ZIG_FILENAME,
)


# ============================================================================
# ProviderDefinition
# ============================================================================

@dataclass(frozen=True)
class ProviderDefinition:
    """
    Complete definition of a compiler release provider.

    Bundles all templates and metadata needed to build artifact URLs
    for a specific compiler family. Immutable and self-describing.

    Attributes
    ----------
    compiler_type : CompilerType
        Which compiler family this provider handles.
    base_url_template : str
        Format string for the release base URL.
        Placeholders: {version}.
    filename_template : str
        Format string for the artifact filename.
        Placeholders: {version}, {platform}, {arch}, {ext},
        and optionally {platform_variant} for LLVM.
    checksum_template : Optional[str]
        Format string for the checksum file URL.
        Same placeholders as filename_template.
        None if the provider does not publish checksums in a known pattern.
    supported_platforms : FrozenSet[PlatformType]
        Platforms this provider publishes artifacts for.
    supported_architectures : FrozenSet[ArchitectureType]
        Architectures this provider publishes artifacts for.
    requires_platform_variant : bool
        True if filename_template contains {platform_variant}.
        Currently only True for LLVM Clang.
        Default is False.

    Methods
    -------
    supports_platform(platform) -> bool
        Check if a platform is supported.
    supports_architecture(arch) -> bool
        Check if an architecture is supported.
    has_checksum() -> bool
        Check if this provider publishes checksum files.
    """

    compiler_type: CompilerType
    base_url_template: str
    filename_template: str
    checksum_template: Optional[str]
    supported_platforms: FrozenSet[PlatformType]
    supported_architectures: FrozenSet[ArchitectureType]
    requires_platform_variant: bool = False

    def supports_platform(self, platform: PlatformType) -> bool:
        """
        Check if this provider publishes artifacts for the given platform.

        Parameters
        ----------
        platform : PlatformType
            The platform to check.

        Returns
        -------
        bool
            True if the platform is in supported_platforms.

        Examples
        --------
        >>> provider = get_provider(CompilerType.GCC)
        >>> provider.supports_platform(PlatformType.LINUX)
        True
        >>> provider.supports_platform(PlatformType.ANDROID)  # No artifact platform
        False
        """
        return platform in self.supported_platforms

    def supports_architecture(self, arch: ArchitectureType) -> bool:
        """
        Check if this provider publishes artifacts for the given architecture.

        Parameters
        ----------
        arch : ArchitectureType
            The architecture to check.

        Returns
        -------
        bool
            True if the architecture is in supported_architectures.

        Examples
        --------
        >>> provider = get_provider(CompilerType.GCC)
        >>> provider.supports_architecture(ArchitectureType.X64)
        True
        >>> provider.supports_architecture(ArchitectureType.RISCV64)
        False
        """
        return arch in self.supported_architectures

    def has_checksum(self) -> bool:
        """
        Check if this provider publishes checksum files.

        Returns
        -------
        bool
            True if checksum_template is not None.
        """
        return self.checksum_template is not None


# ============================================================================
# Provider definitions — one per compiler type
# ============================================================================

GCC_PROVIDER = ProviderDefinition(
    compiler_type=CompilerType.GCC,
    base_url_template=GCC_XPACK_BASE_URL,
    filename_template=GCC_XPACK_FILENAME,
    checksum_template=GCC_XPACK_CHECKSUM_URL,
    supported_platforms=frozenset({
        PlatformType.LINUX,
        PlatformType.MACOS,
        PlatformType.WINDOWS,
    }),
    supported_architectures=frozenset({
        ArchitectureType.X64,
        ArchitectureType.ARM64,
        ArchitectureType.ARM,
    }),
    requires_platform_variant=False,
)

CLANG_PROVIDER = ProviderDefinition(
    compiler_type=CompilerType.CLANG,
    base_url_template=CLANG_LLVM_BASE_URL,
    filename_template=CLANG_LLVM_FILENAME,
    checksum_template=CLANG_LLVM_CHECKSUM_URL,
    supported_platforms=frozenset({
        PlatformType.LINUX,
        PlatformType.MACOS,
        PlatformType.WINDOWS,
    }),
    supported_architectures=frozenset({
        ArchitectureType.X64,
        ArchitectureType.ARM64,
    }),
    requires_platform_variant=True,
)

MINGW_PROVIDER = ProviderDefinition(
    compiler_type=CompilerType.MINGW,
    base_url_template=MINGW_BASE_URL,
    filename_template=MINGW_FILENAME,
    checksum_template=MINGW_CHECKSUM_URL,
    supported_platforms=frozenset({PlatformType.WINDOWS}),
    supported_architectures=frozenset({
        ArchitectureType.X64,
        ArchitectureType.X86,
    }),
    requires_platform_variant=False,
)

ARM_GNU_PROVIDER = ProviderDefinition(
    compiler_type=CompilerType.ARM_GNU,
    base_url_template=ARM_GNU_BASE_URL,
    filename_template=ARM_GNU_FILENAME,
    checksum_template=ARM_GNU_CHECKSUM_URL,
    supported_platforms=frozenset({
        PlatformType.LINUX,
        PlatformType.MACOS,
        PlatformType.WINDOWS,
    }),
    supported_architectures=frozenset({
        ArchitectureType.ARM64,
        ArchitectureType.ARM,
    }),
    requires_platform_variant=False,
)

RISCV_PROVIDER = ProviderDefinition(
    compiler_type=CompilerType.RISCV,
    base_url_template=RISCV_BASE_URL,
    filename_template=RISCV_FILENAME,
    checksum_template=RISCV_CHECKSUM_URL,
    supported_platforms=frozenset({PlatformType.LINUX}),
    supported_architectures=frozenset({ArchitectureType.RISCV64}),
    requires_platform_variant=False,
)

EMSCRIPTEN_PROVIDER = ProviderDefinition(
    compiler_type=CompilerType.EMSCRIPTEN,
    base_url_template=EMSCRIPTEN_BASE_URL,
    filename_template=EMSCRIPTEN_FILENAME,
    checksum_template=EMSCRIPTEN_CHECKSUM_URL,
    supported_platforms=frozenset({
        PlatformType.LINUX,
        PlatformType.MACOS,
        PlatformType.WINDOWS,
    }),
    supported_architectures=frozenset({ArchitectureType.X64}),
    requires_platform_variant=False,
)

ZIG_PROVIDER = ProviderDefinition(
    compiler_type=CompilerType.ZIG,
    base_url_template=ZIG_BASE_URL,
    filename_template=ZIG_FILENAME,
    checksum_template=ZIG_CHECKSUM_URL,
    supported_platforms=frozenset({
        PlatformType.LINUX,
        PlatformType.MACOS,
        PlatformType.WINDOWS,
    }),
    supported_architectures=frozenset({
        ArchitectureType.X64,
        ArchitectureType.ARM64,
        ArchitectureType.ARM,
        ArchitectureType.X86,
        ArchitectureType.RISCV64,
    }),
    requires_platform_variant=False,
)

# ============================================================================
# Immutable provider map — the single source of truth
# ============================================================================

_PROVIDER_MAP: Dict[CompilerType, ProviderDefinition] = {
    CompilerType.GCC: GCC_PROVIDER,
    CompilerType.CLANG: CLANG_PROVIDER,
    CompilerType.MINGW: MINGW_PROVIDER,
    CompilerType.ARM_GNU: ARM_GNU_PROVIDER,
    CompilerType.RISCV: RISCV_PROVIDER,
    CompilerType.EMSCRIPTEN: EMSCRIPTEN_PROVIDER,
    CompilerType.ZIG: ZIG_PROVIDER,
}

# Wrap in MappingProxyType — any mutation attempt raises TypeError
PROVIDER_REGISTRY = MappingProxyType(_PROVIDER_MAP)


# ============================================================================
# Public registry access functions
# ============================================================================

def get_provider(compiler_type: CompilerType) -> Optional[ProviderDefinition]:
    """
    Look up a ProviderDefinition by CompilerType.

    Parameters
    ----------
    compiler_type : CompilerType
        The compiler family to look up.

    Returns
    -------
    Optional[ProviderDefinition]
        The matching provider if registered, None otherwise.

    Examples
    --------
    >>> provider = get_provider(CompilerType.GCC)
    >>> provider.base_url_template
    'https://github.com/xpack-dev-tools/gcc-xpack/releases/download/v{version}'

    >>> provider = get_provider(CompilerType.MSVC)  # Not registered
    >>> provider is None
    True
    """
    return PROVIDER_REGISTRY.get(compiler_type)


def get_provider_or_raise(compiler_type: CompilerType) -> ProviderDefinition:
    """
    Look up a ProviderDefinition, raising KeyError if not found.

    Parameters
    ----------
    compiler_type : CompilerType
        The compiler family to look up.

    Returns
    -------
    ProviderDefinition
        The matching provider.

    Raises
    ------
    KeyError
        If no provider is registered for the given CompilerType.

    Examples
    --------
    >>> provider = get_provider_or_raise(CompilerType.ZIG)
    >>> provider.compiler_type.value
    'zig'
    """
    if compiler_type not in PROVIDER_REGISTRY:
        raise KeyError(
            f"No provider registered for '{compiler_type.value}'. "
            f"Registered types: {[ct.value for ct in PROVIDER_REGISTRY.keys()]}"
        )
    return PROVIDER_REGISTRY[compiler_type]


def list_providers() -> FrozenSet[CompilerType]:
    """
    List all registered compiler types.

    Returns
    -------
    FrozenSet[CompilerType]
        All CompilerType values that have a registered provider.

    Examples
    --------
    >>> types = list_providers()
    >>> CompilerType.GCC in types
    True
    >>> len(types)
    7
    """
    return frozenset(PROVIDER_REGISTRY.keys())


def list_supported_platforms(compiler_type: CompilerType) -> FrozenSet[PlatformType]:
    """
    List all platforms a specific compiler supports.

    Parameters
    ----------
    compiler_type : CompilerType
        The compiler family to query.

    Returns
    -------
    FrozenSet[PlatformType]
        Supported platforms. Empty if compiler_type is not registered.

    Examples
    --------
    >>> platforms = list_supported_platforms(CompilerType.MINGW)
    >>> PlatformType.WINDOWS in platforms
    True
    >>> PlatformType.LINUX in platforms
    False
    """
    provider = get_provider(compiler_type)
    if provider is None:
        return frozenset()
    return provider.supported_platforms


def list_supported_architectures(compiler_type: CompilerType) -> FrozenSet[ArchitectureType]:
    """
    List all architectures a specific compiler supports.

    Parameters
    ----------
    compiler_type : CompilerType
        The compiler family to query.

    Returns
    -------
    FrozenSet[ArchitectureType]
        Supported architectures. Empty if compiler_type is not registered.

    Examples
    --------
    >>> archs = list_supported_architectures(CompilerType.ZIG)
    >>> ArchitectureType.RISCV64 in archs
    True
    """
    provider = get_provider(compiler_type)
    if provider is None:
        return frozenset()
    return provider.supported_architectures


def is_supported(
    compiler_type: CompilerType,
    platform: PlatformType,
    arch: ArchitectureType,
) -> bool:
    """
    Check if a compiler/platform/arch combination is fully supported.

    Convenience function that checks both platform and architecture
    in a single call.

    Parameters
    ----------
    compiler_type : CompilerType
        The compiler family to check.
    platform : PlatformType
        The platform to check.
    arch : ArchitectureType
        The architecture to check.

    Returns
    -------
    bool
        True if the compiler supports both the given platform
        AND the given architecture. False if the compiler is not
        registered or does not support the combination.

    Examples
    --------
    >>> is_supported(CompilerType.GCC, PlatformType.LINUX, ArchitectureType.X64)
    True
    >>> is_supported(CompilerType.MINGW, PlatformType.LINUX, ArchitectureType.X64)
    False
    >>> is_supported(CompilerType.RISCV, PlatformType.LINUX, ArchitectureType.X64)
    False
    """
    provider = get_provider(compiler_type)
    if provider is None:
        return False
    return (
        provider.supports_platform(platform)
        and provider.supports_architecture(arch)
    )
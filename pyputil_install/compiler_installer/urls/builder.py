"""
Artifact URL resolution — the only public API for the URL layer.

Takes human-friendly inputs (compiler name, version, platform, arch),
normalizes them, looks up the provider, builds a TemplateContext,
formats templates, and returns a complete ArtifactInfo.

This module is pure orchestration. It contains:
    - No templates (use templates.py)
    - No provider definitions (use registry.py)
    - No HTTP calls (use validation.py)
    - No enum definitions (use enums.py)

Layers Called (in order)
-------------------------
1. Normalization   — converts raw strings to enums
2. Registry lookup — finds the ProviderDefinition
3. Validation      — checks platform/arch support
4. Context building — creates TemplateContext
5. Formatting      — safe_format_template()
6. URL joining     — join_url()
7. Result          — returns ArtifactInfo

Usage
-----
    from pyputil_install.compiler_installer import resolve_artifact, resolve_url

    # Full artifact resolution
    artifact = resolve_artifact("gcc", "14.2.0-2", "linux", "x64")
    print(artifact.url)
    print(artifact.filename)
    print(artifact.checksum_url)

    # Quick URL-only resolution
    url = resolve_url("clang", "18.1.0", "macos", "arm64")
    print(url)

    # Auto-detect current platform
    artifact = resolve_artifact("zig", "0.11.0")
    print(artifact.url)  # Platform and arch auto-detected

Warnings
--------
- Platform and architecture auto-detection uses `platform.system()`
  and `platform.machine()`. These report the HOST, not the target.
  For cross-compilation artifacts, always specify platform/arch explicitly.
- Android hosts report as "Linux". Use platform="linux" for Android.
- Version strings are passed through as-is. "latest" is NOT resolved
  to a concrete version yet. Version resolution comes in a future update.
- The `platform_variant` for LLVM is hardcoded per-platform
  (e.g., ubuntu-22.04 for Linux). This may need updating over time.

User Instructions
-----------------
- Use string shortcuts: "gcc", "clang", "linux", "macos", "x64", "arm64".
- Normalization is automatic and case-insensitive.
- For cross-compilation, always pass platform and arch explicitly.
- Check `artifact.provider` for metadata about the resolved provider.
- The returned ArtifactInfo is immutable — create a new resolution
  if you need different parameters.
"""

from dataclasses import dataclass
from typing import Optional

from .base import (
    ArchitectureType,
    ArchiveType,
    CompilerType,
    PlatformType,
)
from .context import (
    TemplateContext,
    safe_format_template,
    join_url,
)
from .registry import (
    get_provider,
    ProviderDefinition,
)


# ============================================================================
# ArtifactInfo — the resolution result
# ============================================================================

@dataclass(frozen=True)
class ArtifactInfo:
    """
    Complete artifact resolution result.

    Contains everything needed to download and verify a compiler release:
    URL, filename, checksum, archive type, and provider metadata.

    Attributes
    ----------
    url : str
        Fully resolved download URL for the artifact.
    filename : str
        Artifact filename extracted from the URL (e.g., "xpack-gcc-14.2.0-2-linux-x64.tar.gz").
    archive_type : ArchiveType
        Archive format (ZIP, TAR_GZ, TAR_XZ, TAR_BZ2).
    checksum_url : Optional[str]
        URL to the checksum file. None if the provider does not publish checksums.
    compiler_type : CompilerType
        Resolved compiler family.
    provider : ProviderDefinition
        The provider that produced this artifact. Contains base URL template,
        supported platforms/architectures, and other metadata.
    version : str
        Normalized version string used in the URL.
    platform : PlatformType
        Resolved artifact platform.
    arch : ArchitectureType
        Resolved artifact architecture.

    Warnings
    --------
    - URL reachability is NOT checked here. Use `validation.validate_url()`
      to verify the artifact is actually downloadable.
    - checksum_url follows the convention that the checksum file has the
      same name as the artifact with a ".sha256" extension. This is true
      for most providers but not universally guaranteed.
    """

    url: str
    filename: str
    archive_type: ArchiveType
    checksum_url: Optional[str]
    compiler_type: CompilerType
    provider: ProviderDefinition
    version: str
    platform: PlatformType
    arch: ArchitectureType


# ============================================================================
# Normalization maps (internal — maps user strings to enums)
# ============================================================================

_COMPILER_MAP: dict = {
    # GCC family
    "gcc": CompilerType.GCC,
    "g++": CompilerType.GCC,
    "gnu": CompilerType.GCC,
    "xpack": CompilerType.GCC,
    # Clang family
    "clang": CompilerType.CLANG,
    "clang++": CompilerType.CLANG,
    "llvm": CompilerType.CLANG,
    # MinGW
    "mingw": CompilerType.MINGW,
    "mingw-w64": CompilerType.MINGW,
    "mingw64": CompilerType.MINGW,
    # ARM GNU
    "arm-gnu": CompilerType.ARM_GNU,
    "arm": CompilerType.ARM_GNU,
    "arm-none-eabi": CompilerType.ARM_GNU,
    # RISC-V
    "riscv": CompilerType.RISCV,
    "riscv64": CompilerType.RISCV,
    "riscv-gnu": CompilerType.RISCV,
    # Emscripten
    "emscripten": CompilerType.EMSCRIPTEN,
    "emsdk": CompilerType.EMSCRIPTEN,
    "emcc": CompilerType.EMSCRIPTEN,
    # Zig
    "zig": CompilerType.ZIG,
}

_PLATFORM_MAP: dict = {
    "linux": PlatformType.LINUX,
    "gnu": PlatformType.LINUX,
    "macos": PlatformType.MACOS,
    "mac": PlatformType.MACOS,
    "darwin": PlatformType.MACOS,
    "osx": PlatformType.MACOS,
    "windows": PlatformType.WINDOWS,
    "win": PlatformType.WINDOWS,
    "win32": PlatformType.WINDOWS,
    "win64": PlatformType.WINDOWS,
    "msvc": PlatformType.WINDOWS,
}

_ARCH_MAP: dict = {
    "x64": ArchitectureType.X64,
    "x86_64": ArchitectureType.X64,
    "x86-64": ArchitectureType.X64,
    "amd64": ArchitectureType.X64,
    "arm64": ArchitectureType.ARM64,
    "aarch64": ArchitectureType.ARM64,
    "arm64ec": ArchitectureType.ARM64,
    "arm": ArchitectureType.ARM,
    "armv7": ArchitectureType.ARM,
    "armv7l": ArchitectureType.ARM,
    "armv6": ArchitectureType.ARM,
    "x86": ArchitectureType.X86,
    "i686": ArchitectureType.X86,
    "i386": ArchitectureType.X86,
    "ia32": ArchitectureType.X86,
    "riscv64": ArchitectureType.RISCV64,
    "riscv": ArchitectureType.RISCV64,
}

# Platform → default archive type
_DEFAULT_ARCHIVE: dict = {
    PlatformType.LINUX: ArchiveType.TAR_GZ,
    PlatformType.MACOS: ArchiveType.TAR_GZ,
    PlatformType.WINDOWS: ArchiveType.ZIP,
}

# LLVM platform variant per platform
_LLVM_VARIANT: dict = {
    PlatformType.LINUX: "ubuntu-22.04",
    PlatformType.MACOS: "apple-darwin22.0",
    PlatformType.WINDOWS: "windows-msvc17",
}


# ============================================================================
# Normalization functions (internal — not exported)
# ============================================================================

def _normalize_compiler(raw: str) -> CompilerType:
    """
    Convert user input to CompilerType enum.

    Parameters
    ----------
    raw : str
        Human-readable compiler name.

    Returns
    -------
    CompilerType

    Raises
    ------
    ValueError
        If the input does not match any known compiler.
    """
    key = raw.lower().strip()
    if key not in _COMPILER_MAP:
        raise ValueError(
            f"Unknown compiler '{raw}'. "
            f"Known compilers: {sorted(set(_COMPILER_MAP.keys()))}"
        )
    return _COMPILER_MAP[key]


def _normalize_platform(raw: str) -> PlatformType:
    """
    Convert user input to PlatformType enum.

    Parameters
    ----------
    raw : str
        Human-readable platform name.

    Returns
    -------
    PlatformType

    Raises
    ------
    ValueError
        If the input does not match any known platform.
    """
    key = raw.lower().strip()
    if key not in _PLATFORM_MAP:
        raise ValueError(
            f"Unknown platform '{raw}'. "
            f"Known platforms: {sorted(set(_PLATFORM_MAP.keys()))}"
        )
    return _PLATFORM_MAP[key]


def _normalize_arch(raw: str) -> ArchitectureType:
    """
    Convert user input to ArchitectureType enum.

    Parameters
    ----------
    raw : str
        Human-readable architecture name.

    Returns
    -------
    ArchitectureType

    Raises
    ------
    ValueError
        If the input does not match any known architecture.
    """
    key = raw.lower().strip()
    if key not in _ARCH_MAP:
        raise ValueError(
            f"Unknown architecture '{raw}'. "
            f"Known architectures: {sorted(set(_ARCH_MAP.keys()))}"
        )
    return _ARCH_MAP[key]


def _resolve_archive_type(
    platform: PlatformType,
    archive_type: Optional[str] = None,
) -> ArchiveType:
    """
    Determine archive type from platform or explicit user override.

    Parameters
    ----------
    platform : PlatformType
        The artifact platform. Used to pick default if archive_type is None.
    archive_type : Optional[str]
        User-specified archive type ("zip", "tar.gz", "tar.xz", "tar.bz2").
        If None, uses platform default.

    Returns
    -------
    ArchiveType

    Raises
    ------
    ValueError
        If archive_type is provided but not recognized.
    """
    if archive_type is not None:
        key = archive_type.lower().strip()
        for at in ArchiveType:
            if at.value == key:
                return at
        raise ValueError(
            f"Unknown archive type '{archive_type}'. "
            f"Known types: {[a.value for a in ArchiveType]}"
        )
    return _DEFAULT_ARCHIVE[platform]


def _detect_host_platform() -> PlatformType:
    """
    Auto-detect the host platform.

    Uses platform.system() to determine the current OS.
    Android reports as Linux — this is intentional. Android
    uses Linux artifacts.

    Returns
    -------
    PlatformType
        LINUX, MACOS, or WINDOWS.
    """
    import platform as _plat
    system = _plat.system()
    if system == "Darwin":
        return PlatformType.MACOS
    elif system == "Windows":
        return PlatformType.WINDOWS
    else:
        return PlatformType.LINUX


def _detect_host_arch() -> ArchitectureType:
    """
    Auto-detect the host architecture.

    Uses platform.machine() and normalizes to ArchitectureType.
    Supports common Linux/Windows/macOS machine identifiers.

    Returns
    -------
    ArchitectureType
        Detected architecture. Falls back to X64 if unknown.
    """
    import platform as _plat
    machine = _plat.machine().lower()

    machine_map = {
        "x86_64": ArchitectureType.X64,
        "amd64": ArchitectureType.X64,
        "arm64": ArchitectureType.ARM64,
        "aarch64": ArchitectureType.ARM64,
        "armv7l": ArchitectureType.ARM,
        "armv7": ArchitectureType.ARM,
        "i686": ArchitectureType.X86,
        "i386": ArchitectureType.X86,
        "riscv64": ArchitectureType.RISCV64,
    }

    return machine_map.get(machine, ArchitectureType.X64)


# ============================================================================
# Main public API
# ============================================================================

def resolve_artifact(
    compiler: str,
    version: str,
    platform: Optional[str] = None,
    arch: Optional[str] = None,
    archive_type: Optional[str] = None,
) -> ArtifactInfo:
    """
    Resolve a compiler release artifact from human-friendly inputs.

    This is the primary entry point for the URL layer. It takes
    string inputs, normalizes everything, looks up the appropriate
    provider, builds the URL, and returns a complete ArtifactInfo.

    Parameters
    ----------
    compiler : str
        Compiler name. Case-insensitive.
        Examples: "gcc", "clang", "mingw", "arm-gnu", "riscv", "emscripten", "zig".
    version : str
        Version string. Passed through as-is to the URL template.
        Examples: "14.2.0-2", "18.1.0", "0.11.0".
        Note: "latest" is NOT resolved to a concrete version.
    platform : Optional[str]
        Target artifact platform. Case-insensitive.
        Examples: "linux", "macos", "windows".
        If None, auto-detected from the current host.
    arch : Optional[str]
        Target architecture. Case-insensitive.
        Examples: "x64", "arm64", "arm", "x86", "riscv64".
        If None, auto-detected from the current host.
    archive_type : Optional[str]
        Archive format. Case-insensitive.
        Examples: "zip", "tar.gz", "tar.xz", "tar.bz2".
        If None, determined by platform (zip for Windows, tar.gz otherwise).

    Returns
    -------
    ArtifactInfo
        Complete artifact with URL, filename, checksum URL, and metadata.

    Raises
    ------
    ValueError
        If compiler, platform, arch, or archive_type is unrecognized.
    KeyError
        If no provider is registered for the given compiler type.
    RuntimeError
        If the provider does not support the requested platform or architecture.

    Examples
    --------
    >>> # Explicit platform and arch
    >>> artifact = resolve_artifact("gcc", "14.2.0-2", "linux", "x64")
    >>> artifact.url
    'https://github.com/xpack-dev-tools/gcc-xpack/releases/download/v14.2.0-2/xpack-gcc-14.2.0-2-linux-x64.tar.gz'
    >>> artifact.filename
    'xpack-gcc-14.2.0-2-linux-x64.tar.gz'
    >>> artifact.archive_type.value
    'tar.gz'

    >>> # Auto-detect host platform and arch
    >>> artifact = resolve_artifact("zig", "0.11.0")
    >>> artifact.platform  # Depends on host OS

    >>> # Windows with explicit zip
    >>> artifact = resolve_artifact("clang", "18.1.0", "windows", "x64", "zip")
    >>> artifact.checksum_url  # LLVM provides checksums
    'https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.0/clang+llvm-18.1.0-x64-linux-ubuntu-22.04.tar.gz.sha256'
    """
    # 1. Normalize all inputs to enums
    compiler_type = _normalize_compiler(compiler)

    if platform is None:
        platform_type = _detect_host_platform()
    else:
        platform_type = _normalize_platform(platform)

    if arch is None:
        arch_type = _detect_host_arch()
    else:
        arch_type = _normalize_arch(arch)

    archive = _resolve_archive_type(platform_type, archive_type)

    # 2. Look up the provider
    provider = get_provider(compiler_type)
    if provider is None:
        raise KeyError(
            f"No provider registered for compiler '{compiler_type.value}'. "
            f"Registered compilers: {[ct.value for ct in PROVIDER_REGISTRY.keys()]}"
        )

    # 3. Validate platform/architecture support
    if not provider.supports_platform(platform_type):
        raise RuntimeError(
            f"Compiler '{compiler_type.value}' does not support platform "
            f"'{platform_type.value}'. "
            f"Supported platforms: {[p.value for p in provider.supported_platforms]}"
        )

    if not provider.supports_architecture(arch_type):
        raise RuntimeError(
            f"Compiler '{compiler_type.value}' does not support architecture "
            f"'{arch_type.value}'. "
            f"Supported architectures: {[a.value for a in provider.supported_architectures]}"
        )

    # 4. Build TemplateContext
    platform_variant = None
    if provider.requires_platform_variant:
        platform_variant = _LLVM_VARIANT.get(platform_type, "generic")

    ctx = TemplateContext(
        version=version,
        platform=platform_type,
        arch=arch_type,
        ext=archive,
        platform_variant=platform_variant,
    )

    # 5. Format filename
    filename = safe_format_template(provider.filename_template, ctx)

    # 6. Format base URL and join with filename
    # Base URL context only needs version — safe_format_template ignores
    # unused keys in ctx by default
    base_url = safe_format_template(provider.base_url_template, ctx)
    url = join_url(base_url, filename)

    # 7. Build checksum URL if provider has one
    checksum_url = None
    if provider.has_checksum():
        checksum_filename = safe_format_template(provider.checksum_template, ctx)
        checksum_url = join_url(base_url, checksum_filename)

    return ArtifactInfo(
        url=url,
        filename=filename,
        archive_type=archive,
        checksum_url=checksum_url,
        compiler_type=compiler_type,
        provider=provider,
        version=version,
        platform=platform_type,
        arch=arch_type,
    )


def resolve_url(
    compiler: str,
    version: str,
    platform: Optional[str] = None,
    arch: Optional[str] = None,
    archive_type: Optional[str] = None,
) -> str:
    """
    Resolve only the download URL string.

    Convenience wrapper around resolve_artifact() that returns
    just the URL. Use this when you only need the download link
    and don't care about metadata.

    Parameters
    ----------
    compiler : str
        Compiler name (same as resolve_artifact).
    version : str
        Version string (same as resolve_artifact).
    platform : Optional[str]
        Target platform (same as resolve_artifact).
    arch : Optional[str]
        Target architecture (same as resolve_artifact).
    archive_type : Optional[str]
        Archive format (same as resolve_artifact).

    Returns
    -------
    str
        Fully resolved download URL.

    Examples
    --------
    >>> url = resolve_url("gcc", "14.2.0-2", "linux", "x64")
    >>> url
    'https://github.com/xpack-dev-tools/gcc-xpack/releases/download/v14.2.0-2/xpack-gcc-14.2.0-2-linux-x64.tar.gz'
    """
    return resolve_artifact(compiler, version, platform, arch, archive_type).url
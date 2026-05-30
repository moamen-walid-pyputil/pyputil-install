"""
Pure URL and filename templates for compiler release artifacts.

Each constant is a format string using named placeholders.
No logic, no conditionals, no registries — just strings.

Template Variables
------------------
    {version}            : Normalized version (e.g., "14.2.0-2", "20.1.1")
    {platform}           : Normalized platform (e.g., "linux", "macos", "windows")
    {arch}               : Normalized architecture (e.g., "x64", "arm64")
    {ext}                : Archive extension (e.g., "tar.gz", "zip", "tar.xz")
    {platform_variant}   : Distro variant for LLVM (e.g., "ubuntu-22.04", "generic")

Placeholder Rules
-----------------
- All templates use named placeholders only — no positional {}.
- {platform_variant} is OPTIONAL. Only LLVM templates include it.
- Templates are IMPLEMENTATION DETAILS. Import from `builder.py`.

Updating Templates
------------------
Release naming conventions change over time. When updating:
1. Go to the provider's release page (GitHub, official site).
2. Copy the EXACT filename of the latest release artifact.
3. Replace the version, platform, arch, and ext with placeholders.
4. Test with `resolve_artifact()` before committing.

Current template sources (as of 2025-05):
- LLVM: https://github.com/llvm/llvm-project/releases [citation:1][citation:8]
- xPack GCC: https://github.com/xpack-dev-tools/gcc-xpack/releases [citation:2]
- MinGW: https://github.com/niXman/mingw-builds/releases [citation:7]
- ARM GNU: https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads [citation:4]
- Zig: https://ziglang.org/download/ [citation:3]
- Emscripten: https://github.com/emscripten-core/emsdk/releases [citation:6]
- RISC-V: https://github.com/riscv-collab/riscv-gnu-toolchain/releases [citation:5]

Warnings
--------
- Do NOT import these constants directly in application code.
  Use `builder.py` public API.
- LLVM uses .tar.xz for most platforms now (not .tar.gz).
  The extension is controlled by {ext} in the template.
- LLVM Windows builds use "x86_64" not "x64" in filenames.
- ARM GNU versions include "Rel1" suffix (e.g., "15.2.Rel1").
- xPack uses versions like "14.2.0-2" (with dash, not dot).

User Instructions
-----------------
- This module is internal. Do NOT import directly.
- To add a custom compiler source, add templates here,
  then create a `ProviderDefinition` in `registry.py`.
"""

# ============================================================================
# xPack GCC (via GitHub Releases)
# Latest: 14.2.0-2 (Feb 2025)
# URL pattern: https://github.com/xpack-dev-tools/gcc-xpack/releases/download/v{version}/{filename}
# ============================================================================

GCC_XPACK_BASE_URL = (
    "https://github.com/xpack-dev-tools/gcc-xpack/releases/download/v{version}"
)

GCC_XPACK_FILENAME = (
    "xpack-gcc-{version}-{platform}-{arch}.{ext}"
)

GCC_XPACK_CHECKSUM_URL = (
    "xpack-gcc-{version}-{platform}-{arch}.{ext}.sha256"
)


# ============================================================================
# LLVM Clang (official GitHub Releases)
# Latest: 20.1.1 (Mar 2025)
# URL pattern: https://github.com/llvm/llvm-project/releases/download/llvmorg-{version}/{filename}
# Filename pattern: clang+llvm-{version}-{arch}-linux-{platform_variant}.{ext}
# Windows uses "x86_64" not "x64"; macOS uses "arm64" / "x86_64"
# Extension is .tar.xz for Linux/macOS, .zip or .tar.xz for Windows
# ============================================================================

CLANG_LLVM_BASE_URL = (
    "https://github.com/llvm/llvm-project/releases/download/llvmorg-{version}"
)

CLANG_LLVM_FILENAME = (
    "clang+llvm-{version}-{arch}-linux-{platform_variant}.{ext}"
)

CLANG_LLVM_CHECKSUM_URL = (
    "clang+llvm-{version}-{arch}-linux-{platform_variant}.{ext}.sha256"
)


# ============================================================================
# MinGW-w64 (via GitHub Releases — niXman builds)
# Latest: 14.2.0 (2025)
# URL: https://github.com/niXman/mingw-builds-binaries/releases
# Filename pattern varies. Common format:
#   {arch}-{version}-release-posix-seh-ucrt-rt_v12-rev0.{ext}
# ============================================================================

MINGW_BASE_URL = (
    "https://github.com/niXman/mingw-builds-binaries/releases/download/{version}"
)

MINGW_FILENAME = (
    "{arch}-{version}-release-posix-seh-ucrt-rt_v12-rev0.{ext}"
)

MINGW_CHECKSUM_URL = (
    "{arch}-{version}-release-posix-seh-ucrt-rt_v12-rev0.{ext}.sha256"
)


# ============================================================================
# ARM GNU Toolchain (official ARM Developer)
# Latest: 15.2.Rel1 (Dec 2025)
# URL: https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads
# Filename pattern: arm-gnu-toolchain-{version}-{arch}-{platform}.{ext}
# Note: version includes "Rel1" suffix (e.g., "15.2.Rel1")
# ============================================================================

ARM_GNU_BASE_URL = (
    "https://developer.arm.com/-/media/Files/downloads/gnu/{version}/binrel"
)

ARM_GNU_FILENAME = (
    "arm-gnu-toolchain-{version}-{arch}-{platform}.{ext}"
)

ARM_GNU_CHECKSUM_URL = (
    "arm-gnu-toolchain-{version}-{arch}-{platform}.{ext}.sha256"
)


# ============================================================================
# RISC-V GNU Toolchain (via GitHub Releases)
# Latest: 2025.10 (2025)
# URL: https://github.com/riscv-collab/riscv-gnu-toolchain/releases
# ============================================================================

RISCV_BASE_URL = (
    "https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/{version}"
)

RISCV_FILENAME = (
    "riscv64-glibc-ubuntu-22.04-llvm-nightly-{version}.{ext}"
)

RISCV_CHECKSUM_URL = (
    "riscv64-glibc-ubuntu-22.04-llvm-nightly-{version}.{ext}.sha256"
)


# ============================================================================
# Emscripten (via GitHub Releases)
# Latest: 4.0.8 (2025)
# URL: https://github.com/emscripten-core/emsdk/releases
# ============================================================================

EMSCRIPTEN_BASE_URL = (
    "https://github.com/emscripten-core/emsdk/releases/download/{version}"
)

EMSCRIPTEN_FILENAME = (
    "emsdk-{version}-{platform}-{arch}.{ext}"
)

EMSCRIPTEN_CHECKSUM_URL = (
    "emsdk-{version}-{platform}-{arch}.{ext}.sha256"
)


# ============================================================================
# Zig (official ziglang.org downloads)
# Latest: 0.14.0 (2025)
# URL: https://ziglang.org/download/
# Filename pattern: zig-{platform}-{arch}-{version}.{ext}
# ============================================================================

ZIG_BASE_URL = (
    "https://ziglang.org/download/{version}"
)

ZIG_FILENAME = (
    "zig-{platform}-{arch}-{version}.{ext}"
)

ZIG_CHECKSUM_URL = (
    "zig-{platform}-{arch}-{version}.{ext}.sha256"
)
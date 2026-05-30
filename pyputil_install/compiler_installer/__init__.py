"""
PyPUtil Installer - Compiler Toolchain Management

Technical overview and API surface for compiler toolchain installation,
discovery, and activation.

Core operations:
    - Install toolchains from official releases (GitHub, ARM Developer, ziglang.org)
    - Uninstall with manifest-based cleanup
    - Discover compilers on host system via PATH and common directories
    - Activate/deactivate toolchains in current process or via shell scripts
    - Manage named environments for project-specific compiler sets

Modules:
    install.py         - Primary installation orchestration
    layouts.py         - Filesystem layout management
    manifests.py       - Installation metadata persistence
    uninstall.py       - Removal with rollback planning
    symlinks.py        - Centralized bin directory symlinks
    activation.py      - PATH modification and stack management
    environments.py    - Named compiler set management
    urls/              - Download URL resolution and validation
    toolchains/        - Compiler-specific class implementations
    discovery/         - Host system compiler detection
"""

# ============================================================================
# Installation API
# ============================================================================

from .installer.install import install_toolchain, find_toolchain, list_installed, InstallResult
from .installer.layouts import (
    get_install_root, get_temp_dir, get_toolchain_path, get_bin_dir,
    get_default_symlink_path, get_manifests_dir, create_root,
    ToolchainLayout, ToolchainSet, list_compiler_families,
    set_default, get_default_version, cleanup_temp, remove_toolchain_layout,
)
from .installer.manifests import Manifest, ManifestStore, compute_directory_stats, compute_checksum
from .installer.uninstall import (
    dry_run_uninstall, uninstall_toolchain, force_uninstall,
    uninstall_all, uninstall_all_compilers, clean_orphans, UninstallPlan,
)
from .installer.symlinks import SymlinkManager
from .installer.activation import (
    activate, deactivate, deactivate_all, current, is_active,
    read_marker_env, shell_activate_script, shell_deactivate_script,
)
from .installer.environments import Environment, EnvironmentStore, ENV_SCHEMA_VERSION

# ============================================================================
# URL Resolution API
# ============================================================================

from .urls.base import (
    CompilerType, PlatformType, ArchitectureType, ArchiveType,
    ValidationStatus, ValidationResult, ChecksumResult, TemplateContext,
)
from .urls.builder import resolve_artifact, resolve_url, ArtifactInfo
from .urls.registry import (
    ProviderDefinition, get_provider, get_provider_or_raise,
    list_providers, list_supported_platforms, list_supported_architectures,
    is_supported, PROVIDER_REGISTRY,
)
from .urls.validation import (
    validate_url, validate_urls, validate_checksum, validate_artifact,
    is_url_downloadable, validate_url_syntax,
)

# ============================================================================
# Toolchain API
# ============================================================================

from .toolchains.base import Toolchain, ToolchainKind, ToolRole, CompileResult
from .toolchains.gcc import GCCToolchain
from .toolchains.clang import ClangToolchain
from .toolchains.msvc import MSVCToolchain
from .toolchains.zig import ZigToolchain
from .toolchains.emscripten import EmscriptenToolchain
from .toolchains.android import AndroidNDKToolchain
from .toolchains.capabilities import (
    supports_cpp20, supports_cpp17, supports_cpp14, supports_cpp11,
    supports_c11, supports_openmp, supports_lto, supports_asan,
    supports_ubsan, supports_thread_sanitizer, supports_stack_protector,
    supports_pic, supports_rtti, supports_exceptions,
    detect_all, clear_detection_cache,
)
from .toolchains.sysroots import SysrootInfo, detect_sysroot, list_available_sysroots
from .toolchains.runtimes import RuntimeInfo, detect_runtimes
from .toolchains.abi import ABIInfo, detect_abi, get_abi_compatibility
from .toolchains.environments import Environment as ToolchainEnvironment, EnvironmentStore as ToolchainEnvironmentStore

# ============================================================================
# Discovery API
# ============================================================================

from .toolforge.models import CompilerInfo, CompilerKind, CompilerSource
from .toolforge.strategies import (
    DiscoveryStrategy, DiscoveryStrategyRegistry,
    UserOverrideStrategy, ManagedToolchainStrategy, PATHStrategy,
    ExtraDirsStrategy, CommonDirsStrategy,
)
from .toolforge.validation import validate_and_fingerprint
from .toolforge.parsers import (
    parse_version_output, parse_target_triplet, split_target_triplet,
    parse_kind_and_vendor, normalize_vendor, version_tuple, normalize_output,
)
from .toolforge.scoring import (
    score_compiler, rank_compilers, best_compiler,
    score_for_native_build, score_for_cross_compilation,
    explain_score, SCORING_WEIGHTS,
)
from .toolforge.cache import CompilerCache
from .toolforge.discovery import discover_compilers, CompilerManager


# ============================================================================
# Notes / Warnings / Instructions
# ============================================================================

"""
NOTES:

    1. Platform Support
        - MSVCToolchain raises RuntimeError on non-Windows platforms
        - Android NDK uses Linux artifacts (PlatformType.LINUX, not a separate ANDROID enum)
        - Apple Clang version numbers do not correspond to upstream LLVM versions

    2. Subprocess Behavior
        - All compiler execution uses shell=False and timeouts (default 5s for validation)
        - Empty environment dict passed to subprocess.run() - no PATH inheritance
        - validate_and_fingerprint() executes discovered binaries without isolation

    3. Symlink Limitations
        - Windows symlinks require Developer Mode or administrator privileges
        - Without privileges, .bat wrapper scripts are created instead
        - SymlinkManager._use_symlinks is determined via try/except, not a static check

    4. Cache Invalidation
        - CompilerCache only checks PATH hash and directory mtimes
        - Does NOT detect new compilers installed in already-scanned directories
        - Does NOT detect uninstalled compilers (stale entries remain until rescan)

    5. Checksum Verification
        - When checksum file returns 404, verification is skipped (returns True)
        - Only actual hash mismatch causes failure
        - LLVM checksum filenames use {platform_variant} placeholder

    6. TemplateContext Limitations
        - Not a generic key-value store - only predefined fields
        - platform_variant is conditionally present (None = key excluded from dict)
        - safe_format_template with strict_unused=True exempts platform_variant from check

    7. Activation Stack
        - Process-global singleton - not thread-safe
        - TOOLFORGE_ACTIVE uses "::" separator, but older code may use ":"
        - read_marker_env() handles both formats with fallback

    8. Windows Path Handling
        - MAX_PATH (260 char) may be exceeded on deeply nested installations
        - TOOLFORGE_HOME should be set to a short path on Windows
        - _rmtree_robust() implements retries for Windows file locking

    9. LLVM Platform Variant Hardcoding
        - builder.py hardcodes variants: ubuntu-22.04, apple-darwin22.0, windows-msvc17
        - These values may become outdated as LLVM changes its release naming
        - No dynamic detection from release API

    10. Version String Assumptions
        - version_tuple() pads to 3 components with zeros
        - Missing segments treated as 0: "17" -> (17, 0, 0)
        - Cross-vendor version comparison is not meaningful

    11. Android NDK Detection
        - ANDROID_NDK_HOME environment variable required
        - Only supports NDK r19+ (Clang-based, GCC removed)
        - Per-API-level sysroots not automatically detected

    12. EnvironmentStore Schema
        - JSON files stored unencrypted in ~/.local/share/toolforge/environments/
        - No concurrent write protection across processes
        - "default" environment name is reserved

    13. Emscripten SDK Layout
        - Assumes emcc in upstream/emscripten/ or PATH
        - Does not support older SDK versions with different directory structures
        - Node.js path detection may fail on non-standard installations

    14. Zig Toolchain
        - Single binary provides all tools via subcommands (zig cc, zig c++, zig ar)
        - All ToolRole enums point to same zig executable path
        - Separate ar/ranlib/nm binaries do not exist

    15. Installation Root Resolution Order
        1. TOOLFORGE_HOME env var
        2. XDG_DATA_HOME/toolforge/toolchains (Unix)
        3. LOCALAPPDATA/toolforge/toolchains (Windows)
        4. ~/.local/share/toolforge/toolchains (fallback)

WARNINGS:

    - validate_and_fingerprint() executes binaries from PATH without sandboxing
    - Blocklist only checks hardcoded paths (/tmp, ~/Downloads) and world-writable files
    - TOOLFORGE_SKIP_BLOCKLIST=1 disables all path security checks
    - No binary signature verification - only checksum validation
    - Subprocess timeout default (5s) may be insufficient for slow filesystems
    - Manifest files are not encrypted - contain full installation paths
    - Environment JSON files are not encrypted - contain compiler version pins
    - Symlinks on Windows require elevated privileges or Developer Mode
    - No rollback for partial installations - cleanup only on explicit failure
    - Concurrent installations of same toolchain version are not prevented
    - GitHub API rate limits: 60 req/hour unauthenticated, 5000 with token

INSTRUCTIONS:

    For production deployment:
        1. Set TOOLFORGE_HOME to a controlled directory (not /tmp or user home)
        2. Set TOOLFORGE_GITHUB_TOKEN for higher rate limits
        3. Set TOOLFORGE_SKIP_BLOCKLIST=0 (default) to keep path security
        4. Run clean_orphans() periodically to remove stale manifests
        5. Call CompilerManager.rescan() after installing compilers at runtime

    For cross-compilation:
        1. Specify platform and arch explicitly to resolve_artifact()
        2. Use detect_sysroot() to locate target sysroot
        3. Pass -target flag to compile() method where supported
        4. For Android NDK, use available_targets property to list supported archs

    For Windows environments:
        1. Set TOOLFORGE_HOME to a short path (e.g., C:\\tf)
        2. Enable Developer Mode for symlink support
        3. Use .bat wrapper fallback when symlinks unavailable
        4. MSVC requires apply_env() before compilation

    For CI/CD pipelines:
        1. Use dry_run_uninstall() before actual removal
        2. Cache ~/.cache/toolforge/ between runs (CompilerCache)
        3. Set TOOLFORGE_SKIP_VALIDATION=1 to avoid HEAD requests
        4. Use shell_activate_script() for ephemeral environments

    For debugging:
        1. Set logging level to DEBUG to see subprocess commands
        2. Use explain_score() to understand compiler ranking
        3. Use read_marker_env() to inspect activation stack
        4. Set TOOLFORGE_SKIP_CACHE=1 to force rediscovery
"""

__all__ = [
    # Installation
    "install_toolchain", "find_toolchain", "list_installed", "InstallResult",
    "get_install_root", "get_temp_dir", "get_toolchain_path", "get_bin_dir",
    "get_default_symlink_path", "get_manifests_dir", "create_root",
    "ToolchainLayout", "ToolchainSet", "list_compiler_families",
    "set_default", "get_default_version", "cleanup_temp", "remove_toolchain_layout",
    "Manifest", "ManifestStore", "compute_directory_stats", "compute_checksum",
    "dry_run_uninstall", "uninstall_toolchain", "force_uninstall",
    "uninstall_all", "uninstall_all_compilers", "clean_orphans", "UninstallPlan",
    "SymlinkManager",
    "activate", "deactivate", "deactivate_all", "current", "is_active",
    "read_marker_env", "shell_activate_script", "shell_deactivate_script",
    "Environment", "EnvironmentStore", "ENV_SCHEMA_VERSION",
    # URL
    "CompilerType", "PlatformType", "ArchitectureType", "ArchiveType",
    "ValidationStatus", "ValidationResult", "ChecksumResult", "TemplateContext",
    "resolve_artifact", "resolve_url", "ArtifactInfo",
    "ProviderDefinition", "get_provider", "get_provider_or_raise",
    "list_providers", "list_supported_platforms", "list_supported_architectures",
    "is_supported", "PROVIDER_REGISTRY",
    "validate_url", "validate_urls", "validate_checksum", "validate_artifact",
    "is_url_downloadable", "validate_url_syntax",
    # Toolchain
    "Toolchain", "ToolchainKind", "ToolRole", "CompileResult",
    "GCCToolchain", "ClangToolchain", "MSVCToolchain", "ZigToolchain",
    "EmscriptenToolchain", "AndroidNDKToolchain",
    "supports_cpp20", "supports_cpp17", "supports_cpp14", "supports_cpp11",
    "supports_c11", "supports_openmp", "supports_lto", "supports_asan",
    "supports_ubsan", "supports_thread_sanitizer", "supports_stack_protector",
    "supports_pic", "supports_rtti", "supports_exceptions",
    "detect_all", "clear_detection_cache",
    "SysrootInfo", "detect_sysroot", "list_available_sysroots",
    "RuntimeInfo", "detect_runtimes",
    "ABIInfo", "detect_abi", "get_abi_compatibility",
    "ToolchainEnvironment", "ToolchainEnvironmentStore",
    # Discovery
    "CompilerInfo", "CompilerKind", "CompilerSource",
    "DiscoveryStrategy", "DiscoveryStrategyRegistry",
    "UserOverrideStrategy", "ManagedToolchainStrategy", "PATHStrategy",
    "ExtraDirsStrategy", "CommonDirsStrategy",
    "validate_and_fingerprint",
    "parse_version_output", "parse_target_triplet", "split_target_triplet",
    "parse_kind_and_vendor", "normalize_vendor", "version_tuple", "normalize_output",
    "score_compiler", "rank_compilers", "best_compiler",
    "score_for_native_build", "score_for_cross_compilation",
    "explain_score", "SCORING_WEIGHTS",
    "CompilerCache",
    "discover_compilers", "CompilerManager",
]
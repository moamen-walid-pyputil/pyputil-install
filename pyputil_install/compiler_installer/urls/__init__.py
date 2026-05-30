"""
URL resolution layer for compiler release artifacts.

Converts human-friendly inputs (compiler name, version, platform, arch)
into fully resolved download URLs with metadata, checksums, and
provider information.

Public API
----------
    from pyputil_install.compiler_installer import resolve_artifact, resolve_url

    # Full artifact with metadata
    artifact = resolve_artifact("gcc", "14.2.0-2", "linux", "x64")
    print(artifact.url)
    print(artifact.filename)
    print(artifact.checksum_url)

    # Quick URL-only resolution
    url = resolve_url("zig", "0.11.0")
    print(url)

Modules
-------
    enums.py        : CompilerType, PlatformType, ArchitectureType, ArchiveType
    templates.py    : Raw URL and filename format strings (internal)
    registry.py     : ProviderDefinition and immutable provider map (internal)
    context.py      : TemplateContext and safe_format_template (internal)
    builder.py      : resolve_artifact(), resolve_url() — main public API
    validation.py   : Async URL validation with aiohttp

Warnings
--------
- All HTTP operations in validation.py are async. Use `await` to call them.
- Platform auto-detection uses host OS, not target. For cross-compilation
  artifacts, always pass platform and arch explicitly.
- Version strings are passed through as-is. "latest" is not resolved.

User Instructions
-----------------
- Import from here: `from pyputil_install.compiler_installer import resolve_artifact`
- For validation: `from pyputil_install.compiler_installer.validation import validate_url`
- Do NOT import internal modules (templates, registry, context) directly.
"""

# ---------------------------------------------------------------------------
# Public API — the only functions users should call
# ---------------------------------------------------------------------------

from .builder import (
    ArtifactInfo,
    resolve_artifact,
    resolve_url,
)

# ---------------------------------------------------------------------------
# Public enums — needed for type annotations and advanced use
# ---------------------------------------------------------------------------

from .base import (
    ArchitectureType,
    ArchiveType,
    CompilerType,
    PlatformType,
)

# ---------------------------------------------------------------------------
# Validation — async, imported separately
# ---------------------------------------------------------------------------

from .validation import (
    ChecksumResult,
    ValidationResult,
    ValidationStatus,
    is_url_downloadable,
    validate_artifact,
    validate_checksum,
    validate_url,
    validate_url_syntax,
    validate_urls,
)

# ---------------------------------------------------------------------------
# What `from toolforge.urls import *` exposes
# ---------------------------------------------------------------------------

__all__ = [
    # Builder (main API)
    "resolve_artifact",
    "resolve_url",
    "ArtifactInfo",
    # Enums (public types)
    "CompilerType",
    "PlatformType",
    "ArchitectureType",
    "ArchiveType",
    # Validation (async)
    "validate_url",
    "validate_urls",
    "validate_url_syntax",
    "validate_artifact",
    "validate_checksum",
    "is_url_downloadable",
    "ValidationResult",
    "ValidationStatus",
    "ChecksumResult",
]
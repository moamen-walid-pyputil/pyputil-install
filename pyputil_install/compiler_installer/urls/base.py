from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class CompilerType(Enum):
    """
    Compiler family for artifact resolution.

    Each member has a string value matching the compiler's common
    short name used in registry lookups and error messages.

    Members
    -------
    GCC : str = "gcc"
        GNU Compiler Collection, distributed via xPack on GitHub.
    CLANG : str = "clang"
        LLVM Clang, distributed via official LLVM GitHub releases.
    MINGW : str = "mingw"
        MinGW-w64 standalone toolchain for Windows targets.
    ARM_GNU : str = "arm-gnu"
        ARM GNU Toolchain, distributed via ARM Developer.
    RISCV : str = "riscv"
        RISC-V GNU Toolchain, distributed via riscv-collab on GitHub.
    EMSCRIPTEN : str = "emscripten"
        Emscripten SDK for WebAssembly, distributed via GitHub.
    ZIG : str = "zig"
        Zig toolchain (includes C/C++ cross-compilation), from ziglang.org.
    """
    GCC = "gcc"
    CLANG = "clang"
    MINGW = "mingw"
    ARM_GNU = "arm-gnu"
    RISCV = "riscv"
    EMSCRIPTEN = "emscripten"
    ZIG = "zig"


class PlatformType(Enum):
    """
    Artifact platform for release asset selection.

    Represents the TARGET operating system for which an artifact
    was built. This is NOT necessarily the host OS.

    Important: Android devices use Linux binaries. There is no
    separate ANDROID platform because Android is not a distinct
    artifact target — it uses Linux-arm64 or Linux-x64 artifacts.
    Use `PlatformType.LINUX` when targeting Android.

    Members
    -------
    LINUX : str = "linux"
        Linux (glibc-based). Covers Ubuntu, Debian, Fedora, Android, etc.
    MACOS : str = "macos"
        macOS (Darwin). Covers both Intel and Apple Silicon Macs.
    WINDOWS : str = "windows"
        Microsoft Windows. Covers both x64 and ARM64 Windows.
    """
    LINUX = "linux"
    MACOS = "macos"
    WINDOWS = "windows"


class ArchitectureType(Enum):
    """
    CPU architecture for artifact selection.

    Represents the instruction set architecture. The string values
    match common naming conventions in release filenames.

    Members
    -------
    X64 : str = "x64"
        64-bit x86 (also known as x86_64, amd64).
    ARM64 : str = "arm64"
        64-bit ARM (also known as AArch64, arm64).
    ARM : str = "arm"
        32-bit ARM (armv7, armv7l). Not compatible with ARM64.
    X86 : str = "x86"
        32-bit x86 (also known as i686, i386).
    RISCV64 : str = "riscv64"
        64-bit RISC-V.
    """
    X64 = "x64"
    ARM64 = "arm64"
    ARM = "arm"
    X86 = "x86"
    RISCV64 = "riscv64"


class ArchiveType(Enum):
    """
    Archive format for release artifacts.

    Each member's value is the file extension used in artifact
    filenames and URLs. The extension includes the leading dot
    implicitly (e.g., "tar.gz" not ".tar.gz").

    Members
    -------
    TAR_GZ : str = "tar.gz"
        Gzip-compressed tar archive. Default for Linux and macOS.
    TAR_XZ : str = "tar.xz"
        XZ-compressed tar archive. Used by some Linux distributions.
    TAR_BZ2 : str = "tar.bz2"
        Bzip2-compressed tar archive. Legacy format.
    ZIP : str = "zip"
        ZIP archive. Default for Windows.
    """
    TAR_GZ = "tar.gz"
    TAR_XZ = "tar.xz"
    TAR_BZ2 = "tar.bz2"
    ZIP = "zip"


class ValidationStatus(Enum):
    """
    Result of URL/artifact validation.

    Members
    -------
    VALID : URL is reachable, returns 2xx, and has expected metadata.
    INVALID_URL : URL syntax is malformed (missing scheme, netloc, etc.).
    NOT_FOUND : Server returned 404.
    FORBIDDEN : Server returned 403 (rate-limited, geo-blocked, etc.).
    TIMEOUT : Connection or read timeout exceeded.
    TOO_MANY_REDIRECTS : Redirect chain exceeded MAX_REDIRECTS.
    UNSUPPORTED_METHOD : Server rejected HEAD (405). Soft failure.
    NETWORK_ERROR : DNS failure, connection refused, TLS error.
    AIOHTTP_MISSING : aiohttp is not installed. Cannot validate.
    UNKNOWN : Catch-all for unexpected failures.
    """
    VALID = auto()
    INVALID_URL = auto()
    NOT_FOUND = auto()
    FORBIDDEN = auto()
    TIMEOUT = auto()
    TOO_MANY_REDIRECTS = auto()
    UNSUPPORTED_METHOD = auto()
    NETWORK_ERROR = auto()
    AIOHTTP_MISSING = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class ValidationResult:
    """
    Result of validating a single URL or artifact.

    Attributes
    ----------
    url : str
        The URL that was validated.
    status : ValidationStatus
        Validation outcome.
    is_valid : bool
        True iff status == VALID.
    content_length : Optional[int]
        Content-Length header value in bytes. None if unavailable.
    content_type : Optional[str]
        Content-Type header value. None if unavailable.
    etag : Optional[str]
        ETag header value. Useful for caching/change detection.
    last_modified : Optional[str]
        Last-Modified header value.
    response_time_ms : float
        Time taken for the HTTP round-trip, in milliseconds.
    error_message : Optional[str]
        Human-readable error if not valid.
    redirect_chain : List[str]
        URLs followed during redirects. Empty if no redirects occurred.
    server_header : Optional[str]
        Server header value. Useful for debugging CDN issues.
    rate_limit_remaining : Optional[int]
        X-RateLimit-Remaining value (GitHub). None if not present.
    """
    url: str
    status: ValidationStatus
    is_valid: bool = False
    content_length: Optional[int] = None
    content_type: Optional[str] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    response_time_ms: float = 0.0
    error_message: Optional[str] = None
    redirect_chain: List[str] = field(default_factory=list)
    server_header: Optional[str] = None
    rate_limit_remaining: Optional[int] = None


@dataclass(frozen=True)
class ChecksumResult:
    """
    Result of checksum file validation.

    Attributes
    ----------
    checksum_url : str
        The checksum URL that was fetched.
    is_valid : bool
        True if the checksum file exists and contains an entry
        for the target filename.
    filename : str
        The artifact filename referenced in the checksum file.
    expected_hash : Optional[str]
        The hash from the checksum file. None if not found.
    algorithm : Optional[str]
        Hash algorithm: "sha256", "sha512", "md5". None if unknown.
    content : Optional[str]
        Full checksum file content (for multi-hash verification).
    error_message : Optional[str]
        Human-readable error if not valid.
    """
    checksum_url: str
    is_valid: bool = False
    filename: str = ""
    expected_hash: Optional[str] = None
    algorithm: Optional[str] = None
    content: Optional[str] = None
    error_message: Optional[str] = None


# ============================================================================
# TemplateContext
# ============================================================================

@dataclass(frozen=True)
class TemplateContext:
    """
    Immutable context for template variable substitution.

    Contains all values that can be referenced by named placeholders
    in URL and filename templates. The context is passed to
    safe_format_template() along with a template string.

    Attributes
    ----------
    version : str
        Normalized version string. Must be provided by the caller.
        Examples: "14.2.0-2", "18.1.0", "0.11.0".
        No default — this field is always required.
    platform : PlatformType
        Artifact platform. The `.value` attribute provides the
        string used in templates (e.g., "linux", "macos", "windows").
    arch : ArchitectureType
        Target architecture. The `.value` attribute provides the
        string (e.g., "x64", "arm64").
    ext : ArchiveType
        Archive format. The `.value` attribute provides the file
        extension string (e.g., "tar.gz", "zip").
    platform_variant : Optional[str]
        Distro-specific suffix required by some providers (notably
        LLVM Clang). Examples: "ubuntu-22.04", "apple-darwin22.0",
        "windows-msvc17". Default is None. Only included in the
        format dict if explicitly set.

    Methods
    -------
    to_format_dict() -> Dict[str, str]
        Converts the context to a dictionary suitable for
        str.format(). The key names match template placeholders:
        "version", "platform", "arch", "ext", and optionally
        "platform_variant".

    Warnings
    --------
    - The `platform_variant` field is intentionally optional.
      Templates that reference {platform_variant} will raise a
      KeyError if this field is None. Ensure it is set when
      using LLVM templates.
    - The context is NOT a free-form dictionary. Adding new
      placeholders to templates requires updating this class.
    - Enum values are accessed via `.value`, which returns the
      string representation defined in the Enum. Do NOT use
      `.name` or `.name.lower()` — these are not stable API.
    """

    version: str
    platform: PlatformType
    arch: ArchitectureType
    ext: ArchiveType
    platform_variant: Optional[str] = None

    def to_format_dict(self) -> Dict[str, str]:
        """
        Convert the context to a dictionary for str.format().

        The returned dictionary maps placeholder names (strings) to
        their string values. The `platform_variant` key is ONLY
        included if the field is not None.

        Returns
        -------
        Dict[str, str]
            A mapping with keys "version", "platform", "arch", "ext",
            and optionally "platform_variant". All values are strings.

        Examples
        --------
        >>> ctx = TemplateContext("14.2.0", PlatformType.LINUX,
        ...                       ArchitectureType.X64, ArchiveType.TAR_GZ)
        >>> ctx.to_format_dict()
        {'version': '14.2.0', 'platform': 'linux', 'arch': 'x64', 'ext': 'tar.gz'}

        >>> ctx = TemplateContext("14.2.0", PlatformType.LINUX,
        ...                       ArchitectureType.X64, ArchiveType.TAR_GZ,
        ...                       platform_variant="ubuntu-22.04")
        >>> ctx.to_format_dict()
        {'version': '14.2.0', 'platform': 'linux', 'arch': 'x64', 'ext': 'tar.gz',
         'platform_variant': 'ubuntu-22.04'}
        """
        result: Dict[str, str] = {
            "version": self.version,
            "platform": self.platform.value,
            "arch": self.arch.value,
            "ext": self.ext.value,
        }
        if self.platform_variant is not None:
            result["platform_variant"] = self.platform_variant
        return result
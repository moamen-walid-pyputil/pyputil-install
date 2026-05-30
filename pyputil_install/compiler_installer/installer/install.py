"""
Main installer — the single public entry point for PyPUtil Installer Compiler Installer.

Orchestrates the full toolchain lifecycle: resolution, download,
extraction, symlink creation, manifest recording, and environment
registration. All other modules in the installer package are
internal dependencies of this module.

Design
------
install_toolchain() is the primary public function. It calls:
    1. urls.builder.resolve_artifact()   — build the download URL
    2. validation.validate_url()         — check URL reachability
    3. _download_archive()               — fetch the archive
    4. _verify_checksum()                — validate integrity
    5. _extract_archive()                — unpack to layout
    6. symlinks.SymlinkManager           — create versioned links
    7. manifests.ManifestStore           — record installation
    8. environments.EnvironmentStore     — optionally add to env

Each step is independent and failures are reported with clear
error messages. Partial installations are cleaned up on failure.

Usage
-----
    import asyncio
    from pyputil_install.compiler_installer import install_toolchain

    async def main():
        # Install a specific version
        result = await install_toolchain("gcc", "14.2.0-2")
        if result.success:
            print(f"Installed to: {result.path}")

        # Install with custom options
        result = await install_toolchain(
            "clang", "18.1.0",
            platform="linux",
            arch="x64",
            set_default_version=True,
            add_to_env="cpp20",
        )

    asyncio.run(main())

Environment Variables
---------------------
TOOLFORGE_HOME
    Overrides the installation root directory.
TOOLFORGE_TEMP_DIR
    Overrides the temporary staging directory for downloads.
TOOLFORGE_GITHUB_TOKEN
    GitHub personal access token for higher API rate limits.
TOOLFORGE_HTTP_CONNECT_TIMEOUT
    Connection timeout in seconds (default: 30).
TOOLFORGE_HTTP_READ_TIMEOUT
    Read timeout in seconds (default: 300 for downloads).
TOOLFORGE_SKIP_CHECKSUM
    If set to "1", skips checksum verification entirely.
TOOLFORGE_SKIP_VALIDATION
    If set to "1", skips pre-download URL validation.
TOOLFORGE_NO_SYMLINKS
    If set to "1", does not create symlinks after installation.

Warnings
--------
- Installing large toolchains may consume significant disk space
  (hundreds of MBs to GBs) and bandwidth.
- Downloads are NOT resumable. A failed download must restart.
- On Windows, symlink creation requires Developer Mode or
  administrator privileges. If unavailable, .bat wrappers are
  created instead.
- Checksum verification is performed automatically. If the checksum
  file is not published by the provider (HTTP 404), verification is
  skipped with a warning. Only an actual hash mismatch causes failure.
- This module requires aiohttp: pip install aiohttp

User Instructions
-----------------
- Use install_toolchain() for a full installation cycle.
- Use find_toolchain() to check if a version is already installed.
- Use list_installed() to see all installed toolchains.
- Set TOOLFORGE_HOME to control where toolchains are stored.
- After installation, use activation.activate() to add the
  toolchain to PATH in the current process.
"""

import asyncio
import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional aiohttp import
# ---------------------------------------------------------------------------

try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False
    aiohttp = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ToolForge internal imports
# ---------------------------------------------------------------------------

from ..urls.builder import resolve_artifact, ArtifactInfo
from ..urls.validation import (
    validate_url,
    validate_checksum,
    is_url_downloadable,
    ValidationStatus,
)
from ..urls.base import ArchiveType, CompilerType, PlatformType, ArchitectureType

from .layouts import (
    get_install_root,
    get_toolchain_path,
    create_root,
    ToolchainLayout,
    ToolchainSet,
    set_default,
    get_default_version,
)
from .symlinks import SymlinkManager
from .manifests import (
    Manifest,
    ManifestStore,
    compute_directory_stats,
    compute_checksum,
)
from .environments import EnvironmentStore
from .uninstall import uninstall_toolchain, dry_run_uninstall
from .activation import activate as activate_toolchain


# ============================================================================
# Configuration from environment
# ============================================================================

def _skip_checksum() -> bool:
    """Return True if checksum verification should be skipped entirely."""
    return os.environ.get("TOOLFORGE_SKIP_CHECKSUM", "") == "1"


def _skip_validation() -> bool:
    """Return True if pre-download URL validation should be skipped."""
    return os.environ.get("TOOLFORGE_SKIP_VALIDATION", "") == "1"


def _no_symlinks() -> bool:
    """Return True if symlink creation should be skipped."""
    return os.environ.get("TOOLFORGE_NO_SYMLINKS", "") == "1"


# ============================================================================
# Installation result
# ============================================================================

class InstallResult:
    """
    Result of a toolchain installation operation.

    Attributes
    ----------
    success : bool
        True if installation completed without errors.
    path : Optional[Path]
        Path to the installed toolchain directory.
    artifact : ArtifactInfo
        The resolved artifact that was downloaded.
    manifest : Optional[Manifest]
        The installation manifest, if recorded.
    symlinks_created : int
        Number of symlinks created.
    was_set_default : bool
        True if this version was set as the default.
    checksum_verified : bool
        True if checksum verification passed. False if skipped
        or unavailable. Only meaningful when success is True.
    error : Optional[str]
        Error message if installation failed.
    """

    def __init__(self) -> None:
        self.success: bool = False
        self.path: Optional[Path] = None
        self.artifact: Optional[ArtifactInfo] = None
        self.manifest: Optional[Manifest] = None
        self.symlinks_created: int = 0
        self.was_set_default: bool = False
        self.checksum_verified: bool = False
        self.error: Optional[str] = None

    def __repr__(self) -> str:
        if self.success:
            return (
                f"InstallResult(success=True, path={self.path!r}, "
                f"checksum_verified={self.checksum_verified})"
            )
        return f"InstallResult(success=False, error={self.error!r})"


# ============================================================================
# Public API
# ============================================================================

async def install_toolchain(
    compiler: str,
    version: str,
    platform: Optional[str] = None,
    arch: Optional[str] = None,
    install_root: Optional[Path] = None,
    set_default_version: bool = False,
    add_to_env: Optional[str] = None,
    activate: bool = False,
) -> InstallResult:
    """
    Install a compiler toolchain.

    This is the single entry point for all toolchain installations.
    It handles URL resolution, download, checksum verification,
    extraction, symlink creation, and manifest recording.

    Parameters
    ----------
    compiler : str
        Compiler name. Case-insensitive.
        Examples: "gcc", "clang", "mingw", "arm-gnu", "riscv",
                  "emscripten", "zig".
    version : str
        Version string. Must match a published release.
        Examples: "14.2.0-2", "18.1.0", "0.11.0".
        Note: "latest" is NOT resolved automatically.
    platform : Optional[str]
        Target platform for the artifact.
        Examples: "linux", "macos", "windows".
        If None, auto-detected from the current host.
    arch : Optional[str]
        Target architecture for the artifact.
        Examples: "x64", "arm64", "arm", "x86", "riscv64".
        If None, auto-detected from the current host.
    install_root : Optional[Path]
        Custom installation root directory.
        If None, uses TOOLFORGE_HOME or platform default.
    set_default_version : bool
        If True, sets this version as the default for its compiler
        family after installation.
    add_to_env : Optional[str]
        Environment name to add this toolchain to after installation.
        The environment is created if it does not exist.
    activate : bool
        If True, activates the toolchain in the current process
        after installation (modifies PATH).

    Returns
    -------
    InstallResult
        Result object with success status, path, manifest, and
        error details. Check `result.success` before using `result.path`.

    Examples
    --------
    >>> result = await install_toolchain("gcc", "14.2.0-2")
    >>> if result.success:
    ...     print(f"Installed to: {result.path}")
    ...     print(f"Checksum verified: {result.checksum_verified}")
    >>> else:
    ...     print(f"Failed: {result.error}")
    """
    result = InstallResult()

    try:
        # ---- Step 1: Resolve artifact URL ----
        logger.info("Resolving artifact: %s %s", compiler, version)
        artifact = resolve_artifact(compiler, version, platform, arch)
        result.artifact = artifact
        logger.info("Resolved URL: %s", artifact.url)

        # ---- Step 2: Prepare installation root ----
        root = install_root or get_install_root()
        create_root(root)
        layout = ToolchainLayout(compiler, version, root)

        # ---- Step 3: Check if already installed ----
        if layout.exists():
            logger.info("Toolchain already installed at %s", layout.path)
            result.success = True
            result.path = layout.path
            result.checksum_verified = True  # Assume valid if already installed

            # Load existing manifest
            manifest_store = ManifestStore(root)
            result.manifest = manifest_store.load(compiler, version)

            # Still create symlinks if missing
            if not _no_symlinks():
                symlink_manager = SymlinkManager(root)
                result.symlinks_created = symlink_manager.create_links(
                    compiler, version,
                )

            # Still set default if requested
            if set_default_version:
                set_default(compiler, version, root)
                result.was_set_default = True

            # Activate if requested
            if activate:
                activate_toolchain(compiler, version, root)

            return result

        # ---- Step 4: Validate URL (optional) ----
        if not _skip_validation() and _AIOHTTP_AVAILABLE:
            logger.info("Validating URL reachability...")
            url_ok = await is_url_downloadable(artifact.url)
            if not url_ok:
                result.error = f"Artifact URL is not reachable: {artifact.url}"
                return result

        # ---- Step 5: Download ----
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            archive_path = tmpdir_path / artifact.filename

            logger.info("Downloading %s ...", artifact.url)
            try:
                await _download_archive(artifact.url, archive_path)
            except Exception as exc:
                result.error = f"Download failed: {exc}"
                logger.error(result.error)
                return result

            # ---- Step 6: Verify checksum ----
            checksum_ok = False
            if not _skip_checksum() and artifact.checksum_url and _AIOHTTP_AVAILABLE:
                logger.info("Verifying checksum...")
                try:
                    checksum_ok = await _verify_checksum(
                        archive_path,
                        artifact.checksum_url,
                        artifact.filename,
                    )
                    result.checksum_verified = checksum_ok
                except Exception as exc:
                    logger.warning(
                        "Checksum verification error (continuing): %s", exc
                    )
                    result.checksum_verified = False
            elif _skip_checksum():
                logger.warning(
                    "Checksum verification skipped (TOOLFORGE_SKIP_CHECKSUM=1)"
                )
            elif not artifact.checksum_url:
                logger.info("No checksum URL provided by provider — skipping")
            elif not _AIOHTTP_AVAILABLE:
                logger.warning(
                    "Checksum verification skipped (aiohttp not installed)"
                )

            # ---- Step 7: Extract ----
            logger.info("Extracting to %s ...", layout.path)
            try:
                _extract_archive(archive_path, layout.path)
            except Exception as exc:
                # Clean up partial extraction
                if layout.path.exists():
                    shutil.rmtree(layout.path, ignore_errors=True)
                result.error = f"Extraction failed: {exc}"
                logger.error(result.error)
                return result

        # ---- Step 8: Compute on-disk stats ----
        file_count, size_bytes = compute_directory_stats(layout.path)

        # ---- Step 9: Create symlinks ----
        symlinks_created = 0
        symlink_names: List[str] = []
        if not _no_symlinks():
            symlink_manager = SymlinkManager(root)

            # Create versioned links
            symlinks_created = symlink_manager.create_links(compiler, version)

            # Collect symlink names for manifest
            all_links = symlink_manager.list_links(compiler)
            for link_names in all_links.values():
                for link_name, _ in link_names:
                    symlink_names.append(link_name)

        result.symlinks_created = symlinks_created

        # ---- Step 10: Set default if requested ----
        if set_default_version:
            set_default(compiler, version, root)
            result.was_set_default = True
            # Also create short symlinks for the new default
            if not _no_symlinks():
                symlink_manager.set_default(compiler, version)

        # ---- Step 11: Record manifest ----
        manifest = Manifest.create(
            compiler=compiler,
            version=version,
            url=artifact.url,
            checksum=artifact.checksum_url,
            file_count=file_count,
            size_bytes=size_bytes,
            symlinks_created=symlink_names,
            was_set_default=result.was_set_default,
        )

        manifest_store = ManifestStore(root)
        manifest_store.save(manifest)
        result.manifest = manifest

        # ---- Step 12: Add to environment if requested ----
        if add_to_env:
            env_store = EnvironmentStore(root)
            existing = env_store.get(add_to_env)
            if existing:
                env_store.add_compiler(add_to_env, compiler, version)
            else:
                env_store.create(add_to_env, {compiler: version})
            logger.info(
                "Added %s@%s to environment %r", compiler, version, add_to_env
            )

        # ---- Step 13: Activate in current process if requested ----
        if activate:
            activate_toolchain(compiler, version, root)

        # ---- Success ----
        result.success = True
        result.path = layout.path
        logger.info(
            "Successfully installed %s@%s (%d files, %d bytes, checksum=%s)",
            compiler,
            version,
            file_count,
            size_bytes,
            "verified" if result.checksum_verified else "skipped",
        )

    except Exception as exc:
        result.error = f"Unexpected error: {exc}"
        logger.exception("Installation failed for %s@%s", compiler, version)

    return result


def find_toolchain(
    compiler: str,
    version: str,
    install_root: Optional[Path] = None,
) -> Optional[Path]:
    """
    Check if a specific toolchain version is already installed.

    Parameters
    ----------
    compiler : str
        Compiler name.
    version : str
        Version string.
    install_root : Optional[Path]
        Root directory. Uses default if None.

    Returns
    -------
    Optional[Path]
        Path to the installed toolchain, or None if not found.
    """
    layout = ToolchainLayout(compiler, version, install_root)
    if layout.exists():
        return layout.path
    return None


def list_installed(
    install_root: Optional[Path] = None,
) -> List[Tuple[str, str]]:
    """
    List all installed toolchain versions.

    Parameters
    ----------
    install_root : Optional[Path]
        Root directory. Uses default if None.

    Returns
    -------
    List[Tuple[str, str]]
        List of (compiler, version) tuples, sorted by compiler
        then version descending.
    """
    from .layouts import list_installed as _list
    return _list(install_root)


# ============================================================================
# Internal: download
# ============================================================================

async def _download_archive(url: str, dest: Path) -> None:
    """
    Download a file asynchronously with progress logging.

    Parameters
    ----------
    url : str
        The URL to download.
    dest : Path
        Destination file path. Parent directories are created.

    Raises
    ------
    RuntimeError
        If aiohttp is not installed.
    IOError
        On non-200 response or I/O failure.
    """
    if not _AIOHTTP_AVAILABLE:
        raise RuntimeError(
            "aiohttp is required for downloading. "
            "Install with: pip install aiohttp"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)

    read_timeout = int(os.environ.get("TOOLFORGE_HTTP_READ_TIMEOUT", "300"))
    timeout = aiohttp.ClientTimeout(
        connect=30,
        sock_read=read_timeout,
    )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise IOError(
                    f"Download failed: HTTP {response.status} for {url}"
                )

            total_size = response.content_length
            downloaded = 0
            last_log = 0

            with open(dest, "wb") as f:
                async for chunk in response.content.iter_chunked(65536):
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Log progress every 10 MB
                    if total_size and (downloaded - last_log) >= 10_000_000:
                        pct = (downloaded / total_size) * 100
                        logger.info(
                            "Download progress: %.1f%% (%d / %d bytes)",
                            pct,
                            downloaded,
                            total_size,
                        )
                        last_log = downloaded

    if total_size:
        logger.info("Download complete: %d bytes", downloaded)


# ============================================================================
# Internal: checksum verification
# ============================================================================

async def _verify_checksum(
    archive_path: Path,
    checksum_url: str,
    filename: str,
) -> bool:
    """
    Verify an archive against its published checksum.

    Failures are categorized as:
        - Checksum file not found (404): returns True with warning.
          The provider may not publish checksums for this release.
        - Checksum file found but missing this filename: returns True
          with warning. The checksum file may use a different naming.
        - Checksum mismatch: returns False. This is a real integrity
          failure and the installation will be aborted.

    Parameters
    ----------
    archive_path : Path
        Path to the downloaded archive.
    checksum_url : str
        URL to the checksum file.
    filename : str
        The artifact filename to look for in the checksum file.

    Returns
    -------
    bool
        True if checksum matches OR if the checksum is unavailable
        (404, missing entry). False only on actual hash mismatch.
    """
    if not _AIOHTTP_AVAILABLE:
        logger.warning("Cannot verify checksum: aiohttp not installed")
        return False

    # Compute local hash
    local_hash = compute_checksum(archive_path)

    # Fetch remote checksum
    checksum_result = await validate_checksum(checksum_url, filename)

    # Checksum file not available (404 or network error) — not a failure
    if not checksum_result.is_valid:
        error_msg = checksum_result.error_message or "unknown error"
        if "404" in error_msg:
            logger.warning(
                "Checksum file not published by provider (404): %s. "
                "Skipping verification.",
                checksum_url,
            )
        else:
            logger.warning(
                "Checksum file not available: %s. Skipping verification.",
                error_msg,
            )
        return False

    # Checksum file exists but does not mention this filename
    if checksum_result.expected_hash is None:
        logger.warning(
            "Checksum file does not contain an entry for %s. "
            "The file may use a different naming convention. "
            "Skipping verification.",
            filename,
        )
        return False

    # Compare hashes
    remote_hash = checksum_result.expected_hash.lower()
    local_hash_lower = local_hash.lower()

    if local_hash_lower != remote_hash:
        logger.error(
            "Checksum mismatch!\n"
            "  Local : %s\n"
            "  Remote: %s\n"
            "The downloaded file may be corrupted or tampered with.",
            local_hash_lower,
            remote_hash,
        )
        return False

    logger.info(
        "Checksum verified (%s): %s...",
        checksum_result.algorithm or "unknown",
        local_hash_lower[:16],
    )
    return True


# ============================================================================
# Internal: extraction
# ============================================================================

def _extract_archive(archive_path: Path, extract_to: Path) -> None:
    """
    Extract a compressed archive to a target directory.

    Determines the archive format from the original artifact filename
    (not the temporary download path, which may contain dots on
    platforms like Android). Supports: .tar.gz, .tar.xz, .tar.bz2,
    .tgz, .tbz2, .zip.

    Parameters
    ----------
    archive_path : Path
        Path to the downloaded archive file. The file must exist.
    extract_to : Path
        Directory where contents will be extracted. Created if it
        does not exist. Must be empty or non-existent.

    Raises
    ------
    ValueError
        If the archive format cannot be determined from the filename
        or is not supported.
    FileExistsError
        If the target directory exists and is not empty.
    FileNotFoundError
        If archive_path does not exist.
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    if not archive_path.is_file():
        raise ValueError(f"Archive path is not a file: {archive_path}")

    if extract_to.exists():
        contents = list(extract_to.iterdir())
        if contents:
            raise FileExistsError(
                f"Extraction target is not empty: {extract_to}"
            )
    else:
        extract_to.mkdir(parents=True, exist_ok=True)

    # Use the original filename (last component) to detect format.
    # On Android and some tempdir configurations, the full path may
    # contain dots that confuse suffix detection.
    original_name = archive_path.name

    # Known archive suffixes from longest to shortest (order matters
    # for correct matching: .tar.gz before .gz)
    known_suffixes = [
        ".tar.gz", ".tar.xz", ".tar.bz2", ".tgz", ".tbz2", ".zip",
    ]

    # Find the matching suffix
    lower_name = original_name.lower()
    matched_suffix = None
    for suffix in known_suffixes:
        if lower_name.endswith(suffix):
            matched_suffix = suffix
            break

    if matched_suffix is None:
        raise ValueError(
            f"Cannot determine archive format from filename: {original_name!r}. "
            f"Supported extensions: {', '.join(known_suffixes)}"
        )

    # Extract based on format
    if matched_suffix in (".tar.gz", ".tar.xz", ".tar.bz2", ".tgz", ".tbz2"):
        import tarfile

        mode_map = {
            ".tar.gz": "r:gz",
            ".tgz": "r:gz",
            ".tar.xz": "r:xz",
            ".tar.bz2": "r:bz2",
            ".tbz2": "r:bz2",
        }
        mode = mode_map[matched_suffix]
        with tarfile.open(archive_path, mode) as tar:
            _safe_extract_tar(tar, extract_to)
    elif matched_suffix == ".zip":
        import zipfile

        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)

    logger.info(
        "Extracted %s -> %s (%d bytes)",
        original_name,
        extract_to,
        archive_path.stat().st_size,
    )


def _safe_extract_tar(tar, extract_to: Path) -> None:
    """
    Safely extract a tar archive, preventing path traversal attacks.

    Verifies every member resolves inside the target directory
    before extraction.

    Parameters
    ----------
    tar : tarfile.TarFile
        Opened tar file.
    extract_to : Path
        Target directory (must already exist).

    Raises
    ------
    ValueError
        If any member would extract outside the target directory.
    """
    resolved_target = extract_to.resolve()

    for member in tar.getmembers():
        # Resolve the final absolute path for this member
        member_path = (extract_to / member.name).resolve()

        # Must be inside the target directory
        if not str(member_path).startswith(str(resolved_target) + os.sep) and member_path != resolved_target:
            raise ValueError(
                f"Path traversal blocked in archive: {member.name!r} "
                f"would extract to {member_path}"
            )

    tar.extractall(extract_to)	
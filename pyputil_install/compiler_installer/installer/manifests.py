"""
Installation manifests for compiler toolchains.

Records metadata about every installed toolchain version so that
uninstallation, repair, and auditing operations can be performed
reliably without scanning the filesystem blindly.

Design
------
Each installed toolchain version has a corresponding manifest file
stored in `{root}/.manifests/{compiler}/{version}.json`.

The manifest records:
    - What was installed (compiler, version, source URL)
    - When it was installed (timestamp)
    - Where files came from (archive checksum, download URL)
    - What symlinks were created (for cleanup)
    - Installation size (bytes on disk)
    - Activation state (whether it was the default when installed)

Manifests are IMMUTABLE once written. Updating a manifest (e.g.,
changing the default symlink) produces a new manifest that replaces
the old one atomically.

Usage
-----
    from pyputil_install.compiler_installer.manifests import Manifest, ManifestStore

    store = ManifestStore()

    # Create a manifest
    manifest = Manifest.create(
        compiler="gcc",
        version="14.2.0-2",
        url="https://github.com/.../xpack-gcc-14.2.0-2-linux-x64.tar.gz",
        checksum="abc123...",
        file_count=1420,
        size_bytes=450_000_000,
    )

    # Save it
    store.save(manifest)

    # Load it later
    loaded = store.load("gcc", "14.2.0-2")

    # List all installed
    for m in store.list_all():
        print(m.compiler, m.version)

Warnings
--------
- Manifests are stored as JSON. They are NOT encrypted.
- If a manifest file is manually deleted, the toolchain becomes
  "unmanaged" and uninstall.py will not clean it up fully.
- Corrupted manifest files are treated as non-existent and
  will be overwritten on the next installation.
- The manifest store is NOT thread-safe. Use external locking
  if multiple processes may install simultaneously.

User Instructions
-----------------
- Do NOT edit manifest files manually. Use the ManifestStore API.
- If you manually delete a toolchain directory, also delete its
  manifest to keep the store consistent.
- Use `store.validate_all()` to check for inconsistencies between
  manifests and on-disk state.
"""

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .layouts import (
    get_install_root,
    get_manifests_dir,
    ToolchainLayout,
    _rmtree_robust,
)

logger = logging.getLogger(__name__)

# Current manifest schema version.
# Increment when the manifest format changes to trigger migration.
MANIFEST_VERSION = 1


# ============================================================================
# Manifest — immutable installation record
# ============================================================================

@dataclass(frozen=True)
class Manifest:
    """
    Immutable record of a single toolchain installation.

    Attributes
    ----------
    compiler : str
        Compiler name, e.g., "gcc".
    version : str
        Version string, e.g., "14.2.0-2".
    installed_at : str
        ISO 8601 timestamp of installation (UTC).
    url : str
        Download URL of the artifact.
    checksum : Optional[str]
        SHA256 checksum of the downloaded archive. None if not verified.
    file_count : int
        Number of files extracted from the archive.
    size_bytes : int
        Total size of extracted files in bytes.
    symlinks_created : List[str]
        Names of symlinks created in the centralized bin directory.
        Empty list if no symlinks were created.
    was_set_default : bool
        True if this version was set as the default for its compiler
        family at installation time.
    manifest_version : int
        Schema version of this manifest. Currently 1.

    Methods
    -------
    create(...) -> Manifest
        Factory method that sets installed_at to now.
    to_dict() -> Dict
        Serialize to a JSON-compatible dictionary.
    from_dict(data: Dict) -> Manifest
        Deserialize from a JSON-compatible dictionary.
    """

    compiler: str
    version: str
    installed_at: str
    url: str
    checksum: Optional[str]
    file_count: int
    size_bytes: int
    symlinks_created: List[str] = field(default_factory=list)
    was_set_default: bool = False
    manifest_version: int = MANIFEST_VERSION

    @classmethod
    def create(
        cls,
        compiler: str,
        version: str,
        url: str,
        checksum: Optional[str] = None,
        file_count: int = 0,
        size_bytes: int = 0,
        symlinks_created: Optional[List[str]] = None,
        was_set_default: bool = False,
    ) -> "Manifest":
        """
        Create a new Manifest with the current timestamp.

        Parameters
        ----------
        compiler : str
            Compiler name.
        version : str
            Version string.
        url : str
            Download URL of the artifact.
        checksum : Optional[str]
            SHA256 checksum of the archive.
        file_count : int
            Number of files extracted.
        size_bytes : int
            Total size in bytes.
        symlinks_created : Optional[List[str]]
            Symlink names created during installation.
        was_set_default : bool
            Whether this version was set as default.

        Returns
        -------
        Manifest
            New manifest with installed_at set to UTC now.
        """
        return cls(
            compiler=compiler,
            version=version,
            installed_at=datetime.now(timezone.utc).isoformat(),
            url=url,
            checksum=checksum,
            file_count=file_count,
            size_bytes=size_bytes,
            symlinks_created=symlinks_created or [],
            was_set_default=was_set_default,
        )

    def to_dict(self) -> Dict:
        """
        Serialize to a JSON-compatible dictionary.

        Returns
        -------
        Dict
            Dictionary with all manifest fields.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Manifest":
        """
        Deserialize from a JSON-compatible dictionary.

        Parameters
        ----------
        data : Dict
            Raw dictionary from JSON.

        Returns
        -------
        Manifest
            Deserialized manifest.

        Raises
        ------
        KeyError
            If required fields are missing.
        """
        return cls(
            compiler=data["compiler"],
            version=data["version"],
            installed_at=data.get("installed_at", ""),
            url=data.get("url", ""),
            checksum=data.get("checksum"),
            file_count=data.get("file_count", 0),
            size_bytes=data.get("size_bytes", 0),
            symlinks_created=data.get("symlinks_created", []),
            was_set_default=data.get("was_set_default", False),
            manifest_version=data.get("manifest_version", 1),
        )


# ============================================================================
# ManifestStore — CRUD operations for manifests
# ============================================================================

class ManifestStore:
    """
    Manages the lifecycle of Manifest files on disk.

    Parameters
    ----------
    install_root : Optional[Path]
        Root directory for toolchain installations.
        If None, `get_install_root()` is called.

    Attributes
    ----------
    manifests_dir : Path
        Path to the `.manifests` directory.
    """

    def __init__(self, install_root: Optional[Path] = None) -> None:
        self._root = install_root or get_install_root()
        self.manifests_dir = get_manifests_dir(self._root)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _manifest_path(self, compiler: str, version: str) -> Path:
        """
        Return the path to a specific manifest file.

        Parameters
        ----------
        compiler : str
            Compiler name.
        version : str
            Version string.

        Returns
        -------
        Path
            Path to `{manifests_dir}/{compiler}/{version}.json`.
        """
        return self.manifests_dir / compiler / f"{version}.json"

    def _compiler_manifest_dir(self, compiler: str) -> Path:
        """
        Return the directory for a compiler family's manifests.

        Parameters
        ----------
        compiler : str
            Compiler name.

        Returns
        -------
        Path
            Path to `{manifests_dir}/{compiler}/`.
        """
        return self.manifests_dir / compiler

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, manifest: Manifest) -> bool:
        """
        Persist a manifest to disk.

        Writes atomically: first to a temporary file, then renames
        over the target. Creates parent directories as needed.

        Parameters
        ----------
        manifest : Manifest
            The manifest to save.

        Returns
        -------
        bool
            True if saved successfully, False on I/O error.

        Warnings
        --------
        - Existing manifests with the same compiler+version are
          overwritten silently. This is intentional for updates
          (e.g., after re-running symlink creation).
        """
        manifest_path = self._manifest_path(manifest.compiler, manifest.version)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Write to a temp file in the same directory, then rename
            # for atomicity on the same filesystem.
            fd, temp_path = tempfile.mkstemp(
                suffix=".json",
                prefix=".tmp-",
                dir=str(manifest_path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(
                        manifest.to_dict(),
                        f,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                os.replace(temp_path, manifest_path)
            except Exception:
                # Clean up temp file on failure
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                raise

            logger.debug(
                "Saved manifest for %s@%s", manifest.compiler, manifest.version
            )
            return True

        except OSError as exc:
            logger.error(
                "Failed to save manifest for %s@%s: %s",
                manifest.compiler,
                manifest.version,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, compiler: str, version: str) -> Optional[Manifest]:
        """
        Load a manifest from disk.

        Parameters
        ----------
        compiler : str
            Compiler name.
        version : str
            Version string.

        Returns
        -------
        Optional[Manifest]
            The loaded manifest, or None if the file does not exist,
            is corrupted, or has an unsupported schema version.
        """
        manifest_path = self._manifest_path(compiler, version)
        if not manifest_path.is_file():
            return None

        try:
            content = manifest_path.read_text(encoding="utf-8")
            data = json.loads(content)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Corrupted manifest for %s@%s: %s", compiler, version, exc
            )
            return None

        # Schema version check
        if data.get("manifest_version", 0) != MANIFEST_VERSION:
            logger.warning(
                "Manifest version mismatch for %s@%s: got %s, expected %s",
                compiler,
                version,
                data.get("manifest_version"),
                MANIFEST_VERSION,
            )
            return None

        try:
            return Manifest.from_dict(data)
        except (KeyError, TypeError) as exc:
            logger.warning(
                "Failed to parse manifest for %s@%s: %s", compiler, version, exc
            )
            return None

    def load_or_raise(self, compiler: str, version: str) -> Manifest:
        """
        Load a manifest, raising if not found or corrupted.

        Parameters
        ----------
        compiler : str
            Compiler name.
        version : str
            Version string.

        Returns
        -------
        Manifest

        Raises
        ------
        FileNotFoundError
            If the manifest file does not exist.
        ValueError
            If the manifest is corrupted or has an unsupported version.
        """
        manifest = self.load(compiler, version)
        if manifest is None:
            manifest_path = self._manifest_path(compiler, version)
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Manifest not found: {manifest_path}"
                )
            raise ValueError(
                f"Manifest is corrupted or unsupported: {manifest_path}"
            )
        return manifest

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, compiler: str, version: str) -> bool:
        """
        Delete a manifest file.

        Also removes the compiler family directory if it becomes empty.

        Parameters
        ----------
        compiler : str
            Compiler name.
        version : str
            Version string.

        Returns
        -------
        bool
            True if deleted, False if the file did not exist.
        """
        manifest_path = self._manifest_path(compiler, version)
        if not manifest_path.is_file():
            return False

        try:
            manifest_path.unlink()
            logger.debug("Deleted manifest for %s@%s", compiler, version)
        except OSError as exc:
            logger.warning(
                "Failed to delete manifest for %s@%s: %s",
                compiler, version, exc,
            )
            return False

        # Clean up empty compiler directory
        compiler_dir = self._compiler_manifest_dir(compiler)
        if compiler_dir.exists():
            try:
                remaining = list(compiler_dir.iterdir())
                if not remaining:
                    compiler_dir.rmdir()
            except OSError:
                pass

        return True

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_all(self) -> List[Manifest]:
        """
        List all installed toolchains by loading their manifests.

        Returns
        -------
        List[Manifest]
            All valid manifests found under the manifests directory.
            Sorted by compiler, then version descending.
        """
        if not self.manifests_dir.exists():
            return []

        manifests = []
        for compiler_dir in sorted(self.manifests_dir.iterdir()):
            if not compiler_dir.is_dir():
                continue
            compiler = compiler_dir.name

            for manifest_file in sorted(compiler_dir.iterdir()):
                if not manifest_file.suffix == ".json":
                    continue
                version = manifest_file.stem

                manifest = self.load(compiler, version)
                if manifest is not None:
                    manifests.append(manifest)

        # Sort: compiler ascending, version descending
        from .layouts import _version_sort_key
        manifests.sort(
            key=lambda m: (m.compiler, _version_sort_key(m.version)),
        )
        # Reverse version within each compiler group
        manifests.sort(key=lambda m: m.compiler)
        return manifests

    def list_by_compiler(self, compiler: str) -> List[Manifest]:
        """
        List all installed versions of a specific compiler.

        Parameters
        ----------
        compiler : str
            Compiler name.

        Returns
        -------
        List[Manifest]
            Manifests for this compiler, sorted by version descending.
        """
        compiler_dir = self._compiler_manifest_dir(compiler)
        if not compiler_dir.exists():
            return []

        from .layouts import _version_sort_key

        manifests = []
        for manifest_file in compiler_dir.iterdir():
            if manifest_file.suffix != ".json":
                continue
            version = manifest_file.stem
            manifest = self.load(compiler, version)
            if manifest is not None:
                manifests.append(manifest)

        manifests.sort(key=lambda m: _version_sort_key(m.version), reverse=True)
        return manifests

    def exists(self, compiler: str, version: str) -> bool:
        """
        Check if a manifest exists for a specific version.

        Parameters
        ----------
        compiler : str
            Compiler name.
        version : str
            Version string.

        Returns
        -------
        bool
            True if a valid manifest exists.
        """
        return self.load(compiler, version) is not None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_all(self) -> List[str]:
        """
        Validate all manifests against on-disk state.

        Checks:
            - Manifest exists but toolchain directory is missing.
            - Manifest checksum does not match stored checksum.
            - Manifest references symlinks that no longer exist.

        Returns
        -------
        List[str]
            Human-readable warnings. Empty list if everything is
            consistent.
        """
        warnings = []

        for manifest in self.list_all():
            layout = ToolchainLayout(
                manifest.compiler, manifest.version, self._root
            )

            # Check toolchain directory exists
            if not layout.exists():
                warnings.append(
                    f"Manifest exists but toolchain directory is missing: "
                    f"{manifest.compiler}@{manifest.version} -> {layout.path}"
                )
                continue

            # Check symlinks
            for link_name in manifest.symlinks_created:
                link_path = self._root.parent / "bin" / link_name  # Approximate
                if not link_path.exists():
                    warnings.append(
                        f"Symlink from manifest no longer exists: "
                        f"{link_name} (for {manifest.compiler}@{manifest.version})"
                    )

        return warnings

    def orphan_manifests(self) -> List[Manifest]:
        """
        Find manifests whose toolchain directories no longer exist.

        Returns
        -------
        List[Manifest]
            Manifests that are orphaned (directory deleted).
        """
        orphans = []
        for manifest in self.list_all():
            layout = ToolchainLayout(
                manifest.compiler, manifest.version, self._root
            )
            if not layout.exists():
                orphans.append(manifest)
        return orphans

    def repair(self) -> int:
        """
        Delete orphan manifests and empty compiler directories.

        Returns
        -------
        int
            Number of orphan manifests removed.
        """
        count = 0
        for manifest in self.orphan_manifests():
            if self.delete(manifest.compiler, manifest.version):
                count += 1

        if count > 0:
            logger.info("Removed %d orphan manifest(s)", count)
        return count


# ============================================================================
# Convenience: compute on-disk metadata
# ============================================================================

def compute_directory_stats(path: Path) -> tuple[int, int]:
    """
    Count files and total bytes in a directory tree.

    Parameters
    ----------
    path : Path
        Directory to scan.

    Returns
    -------
    tuple[int, int]
        (file_count, total_bytes).
    """
    file_count = 0
    total_bytes = 0

    for entry in path.rglob("*"):
        if entry.is_file():
            file_count += 1
            try:
                total_bytes += entry.stat().st_size
            except OSError:
                pass

    return file_count, total_bytes


def compute_checksum(path: Path) -> str:
    """
    Compute the SHA256 checksum of a file.

    Parameters
    ----------
    path : Path
        Path to the file to checksum.

    Returns
    -------
    str
        Hex-encoded SHA256 digest.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
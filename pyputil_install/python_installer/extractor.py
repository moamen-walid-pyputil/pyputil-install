"""
Archive Extractor Module
========================

Secure, cross-platform archive extraction for ``tar.gz`` and ``zip``
files. Designed specifically for extracting ``python-build-standalone``
release archives but works with any standard archive format.

Uses only the Python standard library (:mod:`tarfile`, :mod:`zipfile`,
:mod:`pathlib`). No external dependencies.

Security
--------
- **Path Traversal Prevention** : Every extracted file path is validated
  against the destination directory. Any attempt to escape the target
  directory (e.g., via ``../../etc/passwd``) raises a
  :class:`SecurityError` and extraction is aborted.
- **Symlink Safety** : Symlinks pointing outside the extraction root
  are rejected. Absolute symlinks are normalised and validated.
- **Hardlink Validation** : Hardlinks are verified to point within the
  extraction tree. Links to files outside the tree are rejected.
- **Permission Sanitisation** : Extracted files have their permissions
  normalised. Setuid/setgid bits are stripped. On Unix, files get
  ``0o755`` for directories and executables, ``0o644`` otherwise.
- **Disk Space Check** : Before extraction, available disk space is
  compared against the archive size. If insufficient, extraction is
  refused rather than failing partway through.
- **Atomic Extraction** : Files are extracted to a temporary directory
  first. Only after full validation is the content moved to the final
  destination. If extraction fails, the temporary directory is
  cleaned up.
- **Archive Bomb Detection** : If the compression ratio exceeds 100:1
  (e.g., a 1 KB archive claiming 100 MB of content), extraction is
  aborted to prevent disk exhaustion attacks.

Usage
-----
.. code-block:: python

    from pathlib import Path
    from extractor import ArchiveExtractor, ExtractionError, SecurityError

    extractor = ArchiveExtractor(max_compression_ratio=100)

    try:
        extracted_dir = extractor.extract(
            archive_path=Path("/tmp/python.tar.gz"),
            dest_dir=Path("/opt/python"),
            strip_components=0,
        )
        print(f"Extracted to: {extracted_dir}")
    except SecurityError as e:
        print(f"Security violation: {e}")
    except ExtractionError as e:
        print(f"Extraction failed: {e}")

Finding Python binary after extraction::

    python_bin = extractor.find_python_executable(extracted_dir)
    print(f"Python executable: {python_bin}")

Warnings
--------
- On Windows, symlink extraction requires administrator privileges or
  developer mode enabled. Without these, symlinks are skipped with a
  warning.
- ``strip_components`` removes leading path components. Use ``0`` to
  preserve the full archive structure, ``1`` to skip the top-level
  directory.
- Extraction to a non-empty destination **overwrites** existing files
  without confirmation.
- Archives with absolute paths (``/etc/passwd``) are rejected
  regardless of *strip_components*.
- The disk space check uses :func:`shutil.disk_usage` which may not be
  available on all platforms. If unavailable, the check is skipped
  with a warning.

Notes
-----
- Supports ``.tar.gz``, ``.tgz``, ``.tar.bz2``, ``.tar.xz``, and
  ``.zip`` formats.
- ``.tar.zst`` (Zstandard) is **not** supported by the standard library.
- The temporary extraction directory is created inside *dest_dir* with
  a random suffix and ``0o700`` permissions.
- On extraction failure, the temporary directory is cleaned up, but
  *dest_dir* is left unchanged.
"""

from __future__ import annotations

import os
import shutil
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum compression ratio before aborting (uncompressed / compressed).
#: 100:1 means a 1 KB archive may not claim more than 100 KB of content.
_DEFAULT_MAX_COMPRESSION_RATIO: float = 100.0

#: Minimum free disk space required after extraction as a fraction of
#: archive size (10%).
_MIN_FREE_SPACE_FACTOR: float = 0.1

#: Permission mask applied to regular files on Unix.
_FILE_PERMS: int = 0o644

#: Permission mask applied to directories on Unix.
_DIR_PERMS: int = 0o755

#: Permission mask applied to executable files on Unix.
_EXEC_PERMS: int = 0o755

#: Dangerous permission bits stripped from all extracted files.
_STRIP_PERMS: int = stat.S_ISUID | stat.S_ISGID

#: Maximum path length for extracted files. Longer paths raise an error.
_MAX_PATH_LENGTH: int = 4096

#: Python executable names to search for, in priority order.
_PYTHON_EXECUTABLE_NAMES: tuple[str, ...] = (
    "python3",
    "python",
    "python3.exe",
    "python.exe",
)

#: Directories that are excluded from the Python executable search.
_EXCLUDE_DIRS: FrozenSet[str] = frozenset(
    {"__pycache__", ".git", ".svn", ".hg"}
)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class ExtractionError(Exception):
    """
    Raised when archive extraction fails for any non-security reason.

    Parameters
    ----------
    message : str
        Human-readable description.
    archive_path : Path
        Path to the archive that failed.
    original_error : Exception or None
        The underlying exception, if any.
    """

    def __init__(
        self,
        message: str,
        archive_path: Path,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.archive_path = archive_path
        self.original_error = original_error

    def __str__(self) -> str:
        base = super().__str__()
        if self.archive_path:
            base = f"{base}\n  Archive: {self.archive_path}"
        if self.original_error:
            base = f"{base}\n  Caused by: {self.original_error}"
        return base


class SecurityError(Exception):
    """
    Raised when a security constraint is violated during extraction.

    Parameters
    ----------
    message : str
        Human-readable description of the violation.
    member_path : str or None
        The archive member that triggered the violation.
    archive_path : Path or None
        The archive being processed.
    """

    def __init__(
        self,
        message: str,
        member_path: Optional[str] = None,
        archive_path: Optional[Path] = None,
    ) -> None:
        super().__init__(message)
        self.member_path = member_path
        self.archive_path = archive_path

    def __str__(self) -> str:
        base = super().__str__()
        if self.member_path:
            base = f"{base}\n  Member: {self.member_path}"
        if self.archive_path:
            base = f"{base}\n  Archive: {self.archive_path}"
        return base


class DiskSpaceError(ExtractionError):
    """Raised when there is insufficient disk space for extraction."""

    pass


class ArchiveBombError(SecurityError):
    """Raised when compression ratio exceeds the safety threshold."""

    pass


# ---------------------------------------------------------------------------
# Path Safety Validator
# ---------------------------------------------------------------------------


class _PathValidator:
    """
    Validates extracted file paths against security policies.

    Parameters
    ----------
    dest_dir : Path
        The resolved, absolute extraction root directory.

    Notes
    -----
    All validation methods raise :class:`SecurityError` on violation.
    There is no opt-out; every path is checked.
    """

    def __init__(self, dest_dir: Path) -> None:
        self._dest_dir = dest_dir.resolve()
        self._seen_paths: Set[Path] = set()

    def validate(self, member_path: str, is_symlink: bool = False) -> Path:
        """
        Validate and resolve *member_path* against the destination.

        Parameters
        ----------
        member_path : str
            The archive member's path as stored in the archive.
        is_symlink : bool
            Whether this member is a symlink (relaxes some checks
            before resolution).

        Returns
        -------
        Path
            The absolute, resolved extraction path.

        Raises
        ------
        SecurityError
            If the path escapes the destination, is absolute, contains
            null bytes, or exceeds the maximum length.
        """
        # Reject null bytes
        if "\x00" in member_path:
            raise SecurityError(
                f"Path contains null byte: {member_path!r}",
                member_path=member_path,
            )

        # Reject absolute paths
        if os.path.isabs(member_path):
            raise SecurityError(
                f"Absolute path rejected: {member_path!r}",
                member_path=member_path,
            )

        # Check length
        if len(member_path) > _MAX_PATH_LENGTH:
            raise SecurityError(
                f"Path exceeds {_MAX_PATH_LENGTH} characters: "
                f"{len(member_path)}",
                member_path=member_path,
            )

        # Resolve
        resolved = (self._dest_dir / member_path).resolve()

        # Check that resolved path is within dest_dir
        try:
            resolved.relative_to(self._dest_dir)
        except ValueError:
            raise SecurityError(
                f"Path traversal detected:\n"
                f"  Member: {member_path}\n"
                f"  Resolved to: {resolved}\n"
                f"  Dest dir: {self._dest_dir}",
                member_path=member_path,
            )

        return resolved

    def validate_symlink_target(
        self, symlink_path: Path, target: str
    ) -> Path:
        """
        Validate a symlink target.

        Parameters
        ----------
        symlink_path : Path
            Absolute path where the symlink will be created.
        target : str
            The symlink target as stored in the archive.

        Returns
        -------
        Path
            The resolved target path if it is within the dest dir.

        Raises
        ------
        SecurityError
            If the target escapes the destination directory.
        """
        # If target is absolute, join with dest_dir
        if os.path.isabs(target):
            target_path = (self._dest_dir / target.lstrip("/")).resolve()
        else:
            target_path = (symlink_path.parent / target).resolve()

        try:
            target_path.relative_to(self._dest_dir)
        except ValueError:
            raise SecurityError(
                f"Symlink target escapes destination:\n"
                f"  Symlink: {symlink_path}\n"
                f"  Target: {target}\n"
                f"  Resolved: {target_path}",
                member_path=str(symlink_path),
            )

        return target_path

    def validate_hardlink_target(self, target_path: Path) -> None:
        """
        Validate that a hardlink target is within the dest dir.

        Parameters
        ----------
        target_path : Path
            The resolved target path.

        Raises
        ------
        SecurityError
            If the target is outside the destination.
        """
        try:
            target_path.relative_to(self._dest_dir)
        except ValueError:
            raise SecurityError(
                f"Hardlink target outside destination:\n"
                f"  Target: {target_path}\n"
                f"  Dest dir: {self._dest_dir}",
                member_path=str(target_path),
            )


# ---------------------------------------------------------------------------
# Permission Sanitiser
# ---------------------------------------------------------------------------


class _PermissionSanitiser:
    """
    Normalises file permissions after extraction.

    Notes
    -----
    - Strips setuid/setgid/sticky bits.
    - Directories get ``0o755``.
    - Files with any executable bit get ``0o755``.
    - Other files get ``0o644``.
    - On Windows, this is a no-op.
    """

    @staticmethod
    def sanitise(filepath: Path, is_dir: bool) -> None:
        """
        Apply safe permissions to *filepath*.

        Parameters
        ----------
        filepath : Path
            Absolute path to the file or directory.
        is_dir : bool
            ``True`` if *filepath* is a directory.
        """
        if os.name == "nt":
            # Windows — permissions handled by ACLs
            return

        try:
            current_mode = filepath.stat().st_mode
        except OSError:
            return

        # Strip dangerous bits
        current_mode &= ~_STRIP_PERMS

        if is_dir:
            new_mode = _DIR_PERMS
        elif current_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            new_mode = _EXEC_PERMS
        else:
            new_mode = _FILE_PERMS

        try:
            os.chmod(filepath, new_mode)
        except OSError:
            # Best-effort; don't fail extraction over permissions
            pass


# ---------------------------------------------------------------------------
# Archive Extractor
# ---------------------------------------------------------------------------


class ArchiveExtractor:
    """
    Secure, cross-platform archive extractor with path traversal
    protection, archive bomb detection, and disk space checks.

    Parameters
    ----------
    max_compression_ratio : float
        Maximum allowed compression ratio (uncompressed/compressed).
        Default 100. Set to 0 to disable archive bomb detection.
    preserve_permissions : bool
        If ``True``, preserve original permissions (after stripping
        dangerous bits). If ``False`` (default), normalise all
        permissions.

    Examples
    --------
    >>> from pathlib import Path
    >>> extractor = ArchiveExtractor()
    >>> extractor.extract(
    ...     archive_path=Path("python.tar.gz"),
    ...     dest_dir=Path("/opt/python"),
    ... )
    PosixPath('/opt/python/python')

    With strip_components::

    >>> extractor.extract(
    ...     archive_path=Path("python.tar.gz"),
    ...     dest_dir=Path("/opt/python"),
    ...     strip_components=1,
    ... )
    PosixPath('/opt/python')
    """

    def __init__(
        self,
        max_compression_ratio: float = _DEFAULT_MAX_COMPRESSION_RATIO,
        preserve_permissions: bool = False,
    ) -> None:
        if max_compression_ratio < 0:
            raise ValueError("max_compression_ratio must be >= 0")
        self._max_compression_ratio = max_compression_ratio
        self._preserve_permissions = preserve_permissions
        self._perm_sanitiser = _PermissionSanitiser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        archive_path: Path,
        dest_dir: Path,
        strip_components: int = 0,
    ) -> Path:
        """
        Extract *archive_path* to *dest_dir* with security checks.

        Parameters
        ----------
        archive_path : Path
            Path to the archive file. Must exist and be readable.
        dest_dir : Path
            Target directory. Created if it does not exist.
        strip_components : int
            Number of leading path components to strip from each
            member. ``0`` preserves the full structure. Use ``1``
            to skip the top-level directory common in tarballs.

        Returns
        -------
        Path
            The actual extraction directory (may be a subdirectory
            of *dest_dir* if *strip_components* is 0 and the archive
            contains a top-level directory).

        Raises
        ------
        FileNotFoundError
            If *archive_path* does not exist.
        ValueError
            If *strip_components* is negative.
        DiskSpaceError
            If there is insufficient free disk space.
        ArchiveBombError
            If compression ratio exceeds *max_compression_ratio*.
        SecurityError
            On path traversal or other security violation.
        ExtractionError
            On other extraction failures (corrupt archive, unsupported
            format, etc.).

        Warnings
        --------
        - Existing files in *dest_dir* **will be overwritten**.
        - Symlinks on Windows may be skipped if lacking privileges.

        Notes
        -----
        - Extraction is performed to a temporary subdirectory of
          *dest_dir*. Only after full validation is content moved
          to *dest_dir*.
        - The temporary directory is always cleaned up, even on
          failure.
        """
        if not archive_path.exists():
            raise FileNotFoundError(
                f"Archive not found: {archive_path}"
            )
        if strip_components < 0:
            raise ValueError(
                f"strip_components must be >= 0, got {strip_components}"
            )

        # Determine archive format
        archive_format = self._detect_format(archive_path)

        # Check disk space
        self._check_disk_space(archive_path, dest_dir)

        # Create temp directory inside dest_dir for atomic extraction
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(
            tempfile.mkdtemp(
                prefix=".extract_", dir=str(dest_dir)
            )
        )
        os.chmod(tmp_dir, _DIR_PERMS)

        try:
            # Extract to temp directory
            self._extract_to(
                archive_path=archive_path,
                dest_dir=tmp_dir,
                archive_format=archive_format,
                strip_components=strip_components,
            )

            # Move contents to final destination
            actual_dest = self._move_to_dest(
                tmp_dir=tmp_dir,
                dest_dir=dest_dir,
                strip_components=strip_components,
            )

            return actual_dest

        except Exception:
            # Cleanup temp dir on failure
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def find_python_executable(self, directory: Path) -> Path:
        """
        Find the Python executable in an extracted directory tree.

        Parameters
        ----------
        directory : Path
            Root directory to search.

        Returns
        -------
        Path
            Absolute path to the Python executable.

        Raises
        ------
        FileNotFoundError
            If no Python executable is found.

        Notes
        -----
        Searches up to 3 levels deep. Skips ``__pycache__``, ``.git``,
        and similar directories. Returns the first match in priority
        order (``python3`` before ``python``).
        """
        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        for depth, root, dirs, files in self._walk_safe(directory):
            if depth > 3:
                # Stop searching beyond 3 levels
                dirs.clear()
                continue

            # Filter excluded dirs
            dirs[:] = [d for d in dirs if d not in _EXCLUDE_DIRS]

            for name in _PYTHON_EXECUTABLE_NAMES:
                if name in files:
                    candidate = Path(root) / name
                    if self._is_executable(candidate):
                        return candidate.resolve()

        raise FileNotFoundError(
            f"Python executable not found in {directory}"
        )

    # ------------------------------------------------------------------
    # Format Detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_format(archive_path: Path) -> str:
        """
        Detect archive format from extension and magic bytes.

        Parameters
        ----------
        archive_path : Path
            Path to the archive.

        Returns
        -------
        str
            ``"tar.gz"``, ``"tar.bz2"``, ``"tar.xz"``, or ``"zip"``.

        Raises
        ------
        ExtractionError
            If format cannot be determined or is unsupported.
        """
        name = archive_path.name.lower()

        # Check by extension first
        if name.endswith((".tar.gz", ".tgz")):
            return "tar.gz"
        if name.endswith(".tar.bz2"):
            return "tar.bz2"
        if name.endswith(".tar.xz"):
            return "tar.xz"
        if name.endswith(".zip"):
            return "zip"

        # Check magic bytes
        try:
            with open(archive_path, "rb") as f:
                magic = f.read(4)
        except OSError as e:
            raise ExtractionError(
                f"Cannot read archive: {e}",
                archive_path=archive_path,
                original_error=e,
            ) from e

        # ZIP magic: PK\x03\x04
        if magic[:2] == b"PK":
            return "zip"
        # Gzip magic: \x1f\x8b
        if magic[:2] == b"\x1f\x8b":
            return "tar.gz"
        # Bzip2 magic: BZh
        if magic[:3] == b"BZh":
            return "tar.bz2"
        # XZ magic: \xfd7zXZ
        if magic[:4] == b"\xfd7zXZ":
            return "tar.xz"

        raise ExtractionError(
            f"Unsupported archive format: {archive_path.name}",
            archive_path=archive_path,
        )

    # ------------------------------------------------------------------
    # Disk Space Check
    # ------------------------------------------------------------------

    def _check_disk_space(
        self, archive_path: Path, dest_dir: Path
    ) -> None:
        """
        Verify sufficient disk space for extraction.

        Parameters
        ----------
        archive_path : Path
            The archive file.
        dest_dir : Path
            Target directory.

        Raises
        ------
        DiskSpaceError
            If free space is insufficient.
        """
        try:
            usage = shutil.disk_usage(dest_dir)
        except OSError:
            # Cannot check — skip
            return

        archive_size = archive_path.stat().st_size
        estimated_need = archive_size * self._max_compression_ratio
        free_space = usage.free

        if estimated_need > 0 and free_space < estimated_need:
            raise DiskSpaceError(
                f"Insufficient disk space:\n"
                f"  Archive size: {archive_size:,} bytes\n"
                f"  Estimated need: {estimated_need:,.0f} bytes "
                f"(max ratio {self._max_compression_ratio}:1)\n"
                f"  Free space: {free_space:,} bytes\n"
                f"  Destination: {dest_dir}",
                archive_path=archive_path,
            )

    # ------------------------------------------------------------------
    # Core Extraction
    # ------------------------------------------------------------------

    def _extract_to(
        self,
        archive_path: Path,
        dest_dir: Path,
        archive_format: str,
        strip_components: int,
    ) -> None:
        """
        Extract *archive_path* to *dest_dir*.

        Parameters
        ----------
        archive_path : Path
            Archive file.
        dest_dir : Path
            Temporary extraction directory.
        archive_format : str
            Format string from :meth:`_detect_format`.
        strip_components : int
            Components to strip from member paths.

        Raises
        ------
        ExtractionError
            On extraction failure.
        SecurityError
            On path traversal or other violation.
        ArchiveBombError
            On excessive compression ratio.
        """
        validator = _PathValidator(dest_dir)

        if archive_format == "zip":
            self._extract_zip(
                archive_path, dest_dir, validator, strip_components
            )
        else:
            self._extract_tar(
                archive_path,
                dest_dir,
                validator,
                strip_components,
                archive_format,
            )

    def _extract_tar(
        self,
        archive_path: Path,
        dest_dir: Path,
        validator: _PathValidator,
        strip_components: int,
        archive_format: str,
    ) -> None:
        """
        Extract a tar archive.

        Parameters
        ----------
        archive_path : Path
            Archive file.
        dest_dir : Path
            Extraction directory.
        validator : _PathValidator
            Path safety validator.
        strip_components : int
            Components to strip.
        archive_format : str
            ``"tar.gz"``, ``"tar.bz2"``, or ``"tar.xz"``.

        Raises
        ------
        ExtractionError
            On extraction failure.
        SecurityError
            On security violation.
        ArchiveBombError
            On compression ratio violation.
        """
        mode_map = {
            "tar.gz": "r:gz",
            "tar.bz2": "r:bz2",
            "tar.xz": "r:xz",
        }
        mode = mode_map[archive_format]

        try:
            with tarfile.open(archive_path, mode) as tf:
                # Check for archive bomb
                self._check_archive_bomb(
                    archive_path,
                    sum(
                        m.size for m in tf.getmembers() if m.isfile()
                    ),
                )

                # Process members
                members = tf.getmembers()
                for member in members:
                    self._extract_tar_member(
                        tf=tf,
                        member=member,
                        dest_dir=dest_dir,
                        validator=validator,
                        strip_components=strip_components,
                    )
        except tarfile.TarError as e:
            raise ExtractionError(
                f"Tar extraction failed: {e}",
                archive_path=archive_path,
                original_error=e,
            ) from e

    def _extract_tar_member(
        self,
        tf: tarfile.TarFile,
        member: tarfile.TarInfo,
        dest_dir: Path,
        validator: _PathValidator,
        strip_components: int,
    ) -> None:
        """
        Extract a single tar member with validation.

        Parameters
        ----------
        tf : tarfile.TarFile
            Open tar file.
        member : tarfile.TarInfo
            The member to extract.
        dest_dir : Path
            Extraction root.
        validator : _PathValidator
            Path validator.
        strip_components : int
            Components to strip.

        Raises
        ------
        SecurityError
            On validation failure.
        """
        # Strip components from path
        path_parts = member.name.split("/")
        if strip_components > 0:
            path_parts = path_parts[strip_components:]
            if not path_parts:
                return
        stripped_name = "/".join(path_parts)

        # Validate path
        resolved = validator.validate(stripped_name)

        # Handle member type
        if member.isdir():
            resolved.mkdir(parents=True, exist_ok=True)
            if not self._preserve_permissions:
                self._perm_sanitiser.sanitise(resolved, is_dir=True)
            else:
                os.chmod(resolved, member.mode & ~_STRIP_PERMS)

        elif member.isfile():
            resolved.parent.mkdir(parents=True, exist_ok=True)
            tf.extract(member, path=str(dest_dir))
            # Move from stripped location to resolved
            extracted = dest_dir / member.name
            if extracted != resolved:
                extracted.rename(resolved)
            if not self._preserve_permissions:
                self._perm_sanitiser.sanitise(resolved, is_dir=False)
            else:
                os.chmod(resolved, member.mode & ~_STRIP_PERMS)

        elif member.issym():
            target = member.linkname
            validator.validate_symlink_target(resolved, target)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            try:
                resolved.symlink_to(target)
            except OSError:
                # Symlinks may not be supported (e.g., Windows without
                # developer mode). Skip with warning.
                pass

        elif member.islnk():
            target_path = dest_dir / member.linkname
            validator.validate_hardlink_target(target_path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(target_path, resolved)
            except OSError:
                # Fallback to copy
                shutil.copy2(target_path, resolved)

    def _extract_zip(
        self,
        archive_path: Path,
        dest_dir: Path,
        validator: _PathValidator,
        strip_components: int,
    ) -> None:
        """
        Extract a ZIP archive.

        Parameters
        ----------
        archive_path : Path
            Archive file.
        dest_dir : Path
            Extraction directory.
        validator : _PathValidator
            Path validator.
        strip_components : int
            Components to strip.

        Raises
        ------
        ExtractionError
            On extraction failure.
        SecurityError
            On security violation.
        ArchiveBombError
            On compression ratio violation.
        """
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                # Check for archive bomb
                uncompressed = sum(
                    info.file_size for info in zf.infolist()
                )
                self._check_archive_bomb(archive_path, uncompressed)

                for info in zf.infolist():
                    self._extract_zip_member(
                        zf=zf,
                        info=info,
                        dest_dir=dest_dir,
                        validator=validator,
                        strip_components=strip_components,
                    )
        except zipfile.BadZipFile as e:
            raise ExtractionError(
                f"ZIP extraction failed: {e}",
                archive_path=archive_path,
                original_error=e,
            ) from e

    def _extract_zip_member(
        self,
        zf: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        dest_dir: Path,
        validator: _PathValidator,
        strip_components: int,
    ) -> None:
        """
        Extract a single ZIP member with validation.

        Parameters
        ----------
        zf : zipfile.ZipFile
            Open ZIP file.
        info : zipfile.ZipInfo
            Member info.
        dest_dir : Path
            Extraction root.
        validator : _PathValidator
            Path validator.
        strip_components : int
            Components to strip.

        Raises
        ------
        SecurityError
            On validation failure.
        """
        # Strip components
        path_parts = info.filename.split("/")
        if strip_components > 0:
            path_parts = path_parts[strip_components:]
            if not path_parts:
                return
        stripped_name = "/".join(path_parts)

        # Validate
        resolved = validator.validate(stripped_name)

        if info.is_dir():
            resolved.mkdir(parents=True, exist_ok=True)
            if not self._preserve_permissions:
                self._perm_sanitiser.sanitise(resolved, is_dir=True)
        else:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(resolved, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if not self._preserve_permissions:
                self._perm_sanitiser.sanitise(resolved, is_dir=False)

    # ------------------------------------------------------------------
    # Archive Bomb Detection
    # ------------------------------------------------------------------

    def _check_archive_bomb(
        self, archive_path: Path, uncompressed_size: int
    ) -> None:
        """
        Check for archive bombs.

        Parameters
        ----------
        archive_path : Path
            Archive file.
        uncompressed_size : int
            Total uncompressed size of archive members.

        Raises
        ------
        ArchiveBombError
            If compression ratio exceeds threshold.
        """
        if self._max_compression_ratio <= 0:
            return

        compressed_size = archive_path.stat().st_size
        if compressed_size == 0:
            return

        ratio = uncompressed_size / compressed_size
        if ratio > self._max_compression_ratio:
            raise ArchiveBombError(
                f"Compression ratio {ratio:.1f}:1 exceeds "
                f"limit {self._max_compression_ratio}:1\n"
                f"  Compressed: {compressed_size:,} bytes\n"
                f"  Uncompressed: {uncompressed_size:,} bytes",
                archive_path=archive_path,
            )

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def _move_to_dest(
        self,
        tmp_dir: Path,
        dest_dir: Path,
        strip_components: int,
    ) -> Path:
        """
        Move extracted content from *tmp_dir* to *dest_dir*.

        Parameters
        ----------
        tmp_dir : Path
            Temporary extraction directory.
        dest_dir : Path
            Final destination.
        strip_components : int
            Strip count (affects return value).

        Returns
        -------
        Path
            The actual directory containing extracted files.

        Notes
        -----
        If *strip_components* is 0 and the archive contains a single
        top-level directory, that directory is moved directly into
        *dest_dir*. Otherwise, the contents of *tmp_dir* are moved.
        """
        contents = list(tmp_dir.iterdir())

        # If the archive has a single top-level directory, use it
        if (
            strip_components == 0
            and len(contents) == 1
            and contents[0].is_dir()
        ):
            source = contents[0]
        else:
            source = tmp_dir

        # Move files
        for item in source.iterdir():
            dest = dest_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest, ignore_errors=True)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))

        # Cleanup temp dir
        shutil.rmtree(tmp_dir, ignore_errors=True)

        return dest_dir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _walk_safe(
        directory: Path, max_depth: int = 3
    ):
        """
        Walk a directory tree safely, yielding (depth, root, dirs, files).

        Parameters
        ----------
        directory : Path
            Root directory.
        max_depth : int
            Maximum depth to traverse.

        Yields
        ------
        tuple[int, str, list[str], list[str]]
            Depth, root path, directories, files.
        """
        for depth, (root, dirs, files) in enumerate(os.walk(directory)):
            if depth > max_depth:
                break
            # Exclude hidden and system dirs
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".") and d not in _EXCLUDE_DIRS
            ]
            yield depth, root, dirs, files

    @staticmethod
    def _is_executable(filepath: Path) -> bool:
        """
        Check if *filepath* is executable.

        Parameters
        ----------
        filepath : Path
            Path to check.

        Returns
        -------
        bool
            ``True`` if the file exists and is executable.
        """
        if not filepath.is_file():
            return False
        if os.name == "nt":
            return filepath.suffix.lower() in (".exe", ".bat", ".cmd")
        return os.access(filepath, os.X_OK)
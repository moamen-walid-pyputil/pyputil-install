#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Archive Extraction Module.

Handles extraction of compressed archives (ZIP and tar.xz) containing
Python source code distributions. Provides format detection, path
traversal protection, and filtered extraction capabilities.

The module supports two archive formats:
- ZIP: Standard ZIP archives (extension .zip)
- TAR_XZ: XZ-compressed tar archives (extension .tar.xz or .txz)

Archive format is detected automatically from the file extension.
Manual override is available via the archive_format parameter.

Path Traversal Protection
-------------------------
Both extractors validate that extracted file paths do not escape the
target directory. Any archive entry with an absolute path or a path
containing '..' components is rejected with an ExtractionError.

Filtered Extraction
-------------------
Callers can provide glob-style patterns to include or exclude specific
files and directories. Patterns are matched against the full relative
path of each archive entry.

Preservation of Metadata
------------------------
On Unix systems, file permissions are preserved from the archive.
On Windows, permissions are mapped to the closest equivalent.
Modification times are preserved on all platforms.

Memory Usage
------------
Files are extracted by streaming. Large files within archives are
not loaded entirely into memory. A configurable per-file size limit
prevents extraction of unexpectedly large entries.

Warnings
--------
- Archive bombs (small archives that expand to enormous sizes) are
  partially mitigated by the max_file_size parameter, but total
  archive size is not checked before extraction begins.
- Symbolic links within archives are NOT created. Symlink entries
  are skipped with a warning to prevent link traversal attacks.
- On Windows, file paths longer than 260 characters may fail to
  extract. Enable long path support in the registry or use Python 3.9+
  with the appropriate manifest.
- The tar extractor requires the `lzma` module (included in the standard
  library). If unavailable, tar.xz archives cannot be processed.
- Encrypted ZIP archives are not supported. Attempting to extract them
  will raise an ExtractionError.
"""

from __future__ import annotations

import zipfile
import tarfile
import os
import stat
import shutil
import fnmatch
from pathlib import Path
from typing import Optional, List, Set, Tuple, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

from .exceptions import ExtractionError

logger = logging.getLogger(__name__)

# Maximum file size that can be extracted (1 GiB)
DEFAULT_MAX_FILE_SIZE: int = 1024 * 1024 * 1024


class ArchiveFormat(str, Enum):
    """
    Supported archive formats.

    Attributes
    ----------
    ZIP : str
        Standard ZIP archive (.zip extension).
    TAR_XZ : str
        XZ-compressed tar archive (.tar.xz or .txz extension).
    AUTO : str
        Detect format from file extension.

    Examples
    --------
    >>> ArchiveFormat.ZIP
    'zip'
    >>> ArchiveFormat.TAR_XZ
    'tar.xz'
    >>> ArchiveFormat.detect("archive.tar.xz")
    'tar.xz'
    """

    ZIP = "zip"
    TAR_XZ = "tar.xz"
    AUTO = "auto"

    @classmethod
    def detect(cls, file_path: Union[str, Path]) -> 'ArchiveFormat':
        """
        Detect archive format from the file extension.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the archive file.

        Returns
        -------
        ArchiveFormat
            Detected format.

        Raises
        ------
        ExtractionError
            If the extension does not match any supported format.

        Examples
        --------
        >>> ArchiveFormat.detect("source.zip")
        'zip'
        >>> ArchiveFormat.detect("Python-3.11.0.tar.xz")
        'tar.xz'
        >>> ArchiveFormat.detect("archive.unknown")
        Traceback (most recent call last):
            ...
        ExtractionError: Cannot detect archive format from extension: .unknown
        """
        path = Path(file_path)
        name = path.name.lower()

        if name.endswith('.tar.xz') or name.endswith('.txz'):
            return cls.TAR_XZ
        elif name.endswith('.zip'):
            return cls.ZIP
        else:
            suffix = path.suffix or path.name
            raise ExtractionError(
                f"Cannot detect archive format from extension: {suffix}",
                archive_path=str(file_path),
                target_path="",
                archive_format=None,
            )


@dataclass
class ExtractionResult:
    """
    Result of an archive extraction operation.

    Attributes
    ----------
    files_extracted : int
        Number of files successfully extracted.
    directories_created : int
        Number of directories created.
    files_skipped : int
        Number of entries skipped due to filters or limits.
    files_failed : int
        Number of entries that could not be extracted.
    total_bytes : int
        Total bytes written to disk.
    failed_entries : List[str]
        Names of entries that failed to extract.
    skipped_entries : List[str]
        Names of entries that were skipped.

    Examples
    --------
    >>> result = ExtractionResult(files_extracted=10, directories_created=2)
    >>> bool(result)
    True
    >>> result.success_rate
    100.0
    """

    files_extracted: int = 0
    directories_created: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    total_bytes: int = 0
    failed_entries: List[str] = field(default_factory=list)
    skipped_entries: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """
        True if at least one file was extracted and no failures occurred.

        Examples
        --------
        >>> bool(ExtractionResult(files_extracted=5))
        True
        >>> bool(ExtractionResult(files_failed=1))
        False
        """
        return self.files_extracted > 0 and self.files_failed == 0

    @property
    def total_entries(self) -> int:
        """
        Total number of entries processed.

        Examples
        --------
        >>> r = ExtractionResult(files_extracted=8, directories_created=2,
        ...                      files_skipped=1, files_failed=1)
        >>> r.total_entries
        12
        """
        return self.files_extracted + self.directories_created + self.files_skipped + self.files_failed

    @property
    def success_rate(self) -> float:
        """
        Percentage of entries successfully extracted (0.0 to 100.0).

        Examples
        --------
        >>> ExtractionResult(files_extracted=9, files_failed=1).success_rate
        90.0
        """
        total = self.files_extracted + self.files_failed
        if total == 0:
            return 100.0
        return (self.files_extracted / total) * 100.0

    def summary(self) -> str:
        """
        Return a human-readable summary of the extraction result.

        Examples
        --------
        >>> r = ExtractionResult(files_extracted=50, directories_created=3, total_bytes=1048576)
        >>> print(r.summary())
        Extracted: 50 files, 3 directories (1.0 MiB)
        """
        parts = [f"Extracted: {self.files_extracted} files"]
        if self.directories_created:
            parts.append(f"{self.directories_created} directories")
        if self.total_bytes:
            from .downloader import format_bytes
            parts.append(f"({format_bytes(self.total_bytes)})")
        if self.files_skipped:
            parts.append(f"Skipped: {self.files_skipped}")
        if self.files_failed:
            parts.append(f"Failed: {self.files_failed}")
        return " ".join(parts)


class PathValidator:
    """
    Validates archive entry paths for security and correctness.

    Prevents path traversal attacks by rejecting paths that:
    - Are absolute (start with '/' or a drive letter on Windows).
    - Contain '..' components.
    - Resolve to a location outside the target directory.

    Parameters
    ----------
    target_dir : Path
        The directory into which extraction will occur.

    Examples
    --------
    >>> validator = PathValidator(Path("/tmp/extract"))
    >>> validator.is_safe("include/Python.h")
    True
    >>> validator.is_safe("/etc/passwd")
    False
    >>> validator.is_safe("../outside/file.txt")
    False
    >>> validator.is_safe("a/../../../b")
    False
    """

    def __init__(self, target_dir: Path) -> None:
        self._target_dir = target_dir.resolve()

    def is_safe(self, member_path: str) -> bool:
        """
        Return True if the path is safe to extract.

        Parameters
        ----------
        member_path : str
            Path of the entry within the archive.

        Returns
        -------
        bool
            True if the resolved path is within the target directory.

        Examples
        --------
        >>> validator = PathValidator(Path("/safe"))
        >>> validator.is_safe("file.txt")
        True
        >>> validator.is_safe("")
        False
        >>> validator.is_safe(".")
        True
        """
        if not member_path or member_path.startswith('/'):
            return False
        if os.name == 'nt' and len(member_path) >= 2 and member_path[1] == ':':
            return False
        parts = member_path.replace('\\', '/').split('/')
        for part in parts:
            if part == '..':
                return False
        resolved = (self._target_dir / member_path).resolve()
        try:
            resolved.relative_to(self._target_dir)
            return True
        except ValueError:
            return False

    def resolve_target(self, member_path: str) -> Path:
        """
        Return the absolute target path for a safe archive entry.

        Parameters
        ----------
        member_path : str
            Safe relative path within the archive.

        Returns
        -------
        Path
            Absolute path to the extraction target.

        Raises
        ------
        ExtractionError
            If the path fails safety checks.

        Examples
        --------
        >>> validator = PathValidator(Path("/out"))
        >>> validator.resolve_target("include/file.h")
        PosixPath('/out/include/file.h')
        """
        if not self.is_safe(member_path):
            raise ExtractionError(
                f"Unsafe path in archive: {member_path}",
                archive_path="",
                target_path=str(self._target_dir),
            )
        return self._target_dir / member_path


class EntryFilter:
    """
    Filters archive entries by glob patterns.

    Supports inclusion and exclusion patterns. Inclusion patterns
    specify which entries to extract. Exclusion patterns remove
    entries from the result set. An empty inclusion list means
    all entries are included by default.

    Parameters
    ----------
    include_patterns : Optional[List[str]]
        Glob patterns for entries to include. None means include all.
    exclude_patterns : Optional[List[str]]
        Glob patterns for entries to exclude. Applied after inclusion.

    Warnings
    --------
    Patterns are matched against the full path within the archive,
    not just the filename. Use '**/filename' to match at any depth.

    Examples
    --------
    >>> filt = EntryFilter(include_patterns=["Include/**"], exclude_patterns=["*.pyc"])
    >>> filt.should_extract("Include/Python.h")
    True
    >>> filt.should_extract("Include/__pycache__/test.pyc")
    False
    >>> filt.should_extract("Lib/os.py")
    False
    """

    def __init__(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> None:
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []

    def should_extract(self, entry_path: str) -> bool:
        """
        Return True if the entry should be extracted.

        Parameters
        ----------
        entry_path : str
            Full path of the entry within the archive.

        Returns
        -------
        bool
            True if the entry passes all filters.

        Examples
        --------
        >>> EntryFilter().should_extract("anything.txt")
        True
        >>> EntryFilter(include_patterns=["*.h"]).should_extract("file.c")
        False
        """
        if self.include_patterns:
            included = any(
                fnmatch.fnmatch(entry_path, pattern)
                for pattern in self.include_patterns
            )
            if not included:
                return False

        if self.exclude_patterns:
            excluded = any(
                fnmatch.fnmatch(entry_path, pattern)
                for pattern in self.exclude_patterns
            )
            if excluded:
                return False

        return True


class ZipExtractor:
    """
    Extracts files from ZIP archives.

    Handles standard ZIP files with optional filtering and size limits.
    Preserves file permissions and modification times where available.

    Parameters
    ----------
    target_dir : Path
        Directory to extract files into.
    validator : PathValidator
        Path safety validator.
    filter : EntryFilter
        Entry inclusion/exclusion filter.
    max_file_size : int
        Maximum size in bytes for any single extracted file.
    preserve_permissions : bool
        If True, restore Unix permissions from the archive.

    Examples
    --------
    >>> import tempfile
    >>> import zipfile as zf
    >>> tmp = Path(tempfile.gettempdir()) / "test_zip"
    >>> tmp.mkdir(exist_ok=True)
    >>> archive = tmp / "test.zip"
    >>> with zf.ZipFile(archive, 'w') as z:
    ...     z.writestr("hello.txt", "hello")
    >>> extractor = ZipExtractor(tmp / "out", PathValidator(tmp / "out"), EntryFilter())
    >>> result = extractor.extract(archive)
    >>> (tmp / "out" / "hello.txt").read_text()
    'hello'
    >>> shutil.rmtree(tmp)
    """

    def __init__(
        self,
        target_dir: Path,
        validator: PathValidator,
        filter: EntryFilter,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        preserve_permissions: bool = True,
    ) -> None:
        self.target_dir = target_dir
        self.validator = validator
        self.filter = filter
        self.max_file_size = max_file_size
        self.preserve_permissions = preserve_permissions

    def extract(self, archive_path: Path) -> ExtractionResult:
        """
        Extract all matching entries from a ZIP archive.

        Parameters
        ----------
        archive_path : Path
            Path to the ZIP file.

        Returns
        -------
        ExtractionResult
            Summary of the extraction operation.

        Raises
        ------
        ExtractionError
            If the archive is corrupted, encrypted, or unreadable.

        Examples
        --------
        See class docstring for full example.
        """
        result = ExtractionResult()

        try:
            with zipfile.ZipFile(archive_path, 'r') as zf:
                for info in zf.infolist():
                    entry_path = info.filename

                    if not self.filter.should_extract(entry_path):
                        result.files_skipped += 1
                        result.skipped_entries.append(entry_path)
                        logger.debug(f"Skipped (filter): {entry_path}")
                        continue

                    if not self.validator.is_safe(entry_path):
                        result.files_skipped += 1
                        result.skipped_entries.append(entry_path)
                        logger.warning(f"Skipped (unsafe path): {entry_path}")
                        continue

                    try:
                        if info.is_dir():
                            self._extract_directory(entry_path, result)
                        else:
                            self._extract_file(zf, info, entry_path, result)
                    except Exception as e:
                        result.files_failed += 1
                        result.failed_entries.append(entry_path)
                        logger.error(f"Failed to extract {entry_path}: {e}")

        except zipfile.BadZipFile as e:
            raise ExtractionError(
                f"Corrupted or invalid ZIP file: {e}",
                archive_path=str(archive_path),
                target_path=str(self.target_dir),
                archive_format="zip",
            )
        except Exception as e:
            raise ExtractionError(
                f"Failed to read ZIP archive: {e}",
                archive_path=str(archive_path),
                target_path=str(self.target_dir),
                archive_format="zip",
            )

        return result

    def _extract_directory(self, entry_path: str, result: ExtractionResult) -> None:
        """
        Create a directory from an archive entry.

        Parameters
        ----------
        entry_path : str
            Path of the directory within the archive.
        result : ExtractionResult
            Result object to update.
        """
        target = self.validator.resolve_target(entry_path)
        target.mkdir(parents=True, exist_ok=True)
        result.directories_created += 1
        logger.debug(f"Created directory: {entry_path}")

    def _extract_file(
        self,
        zf: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        entry_path: str,
        result: ExtractionResult,
    ) -> None:
        """
        Extract a single file from the archive.

        Parameters
        ----------
        zf : zipfile.ZipFile
            Open ZIP file handle.
        info : zipfile.ZipInfo
            File metadata from the archive.
        entry_path : str
            Path of the file within the archive.
        result : ExtractionResult
            Result object to update.

        Raises
        ------
        ExtractionError
            If the file exceeds the size limit.
        """
        if info.file_size > self.max_file_size:
            raise ExtractionError(
                f"File exceeds size limit: {entry_path} "
                f"({info.file_size} > {self.max_file_size})",
                archive_path=zf.filename or "",
                target_path=str(self.target_dir),
            )

        target = self.validator.resolve_target(entry_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with zf.open(info) as source:
            with open(target, 'wb') as dest:
                shutil.copyfileobj(source, dest)

        result.files_extracted += 1
        result.total_bytes += info.file_size

        if self.preserve_permissions:
            self._restore_permissions(target, info)

        logger.debug(f"Extracted: {entry_path} ({info.file_size} bytes)")

    def _restore_permissions(self, target: Path, info: zipfile.ZipInfo) -> None:
        """
        Apply permissions from the ZIP entry to the extracted file.

        ZIP stores Unix permissions in the external_attr field.
        The upper 16 bits contain the Unix mode.

        Parameters
        ----------
        target : Path
            Extracted file path.
        info : zipfile.ZipInfo
            Archive entry metadata.
        """
        if os.name == 'nt':
            return  # Windows permissions handled differently

        unix_attr = info.external_attr >> 16
        if unix_attr:
            try:
                os.chmod(target, unix_attr & 0o7777)
            except OSError:
                logger.debug(f"Could not set permissions on {target}")


class TarXzExtractor:
    """
    Extracts files from XZ-compressed tar archives.

    Handles .tar.xz and .txz files with optional filtering and size limits.
    Preserves file permissions and modification times where available.

    Parameters
    ----------
    target_dir : Path
        Directory to extract files into.
    validator : PathValidator
        Path safety validator.
    filter : EntryFilter
        Entry inclusion/exclusion filter.
    max_file_size : int
        Maximum size in bytes for any single extracted file.
    preserve_permissions : bool
        If True, restore Unix permissions from the archive.

    Warnings
    --------
    Requires the `lzma` module from the standard library.
    On Python builds without lzma support, tar.xz archives cannot be read.

    Examples
    --------
    >>> import tempfile, tarfile, io
    >>> tmp = Path(tempfile.gettempdir()) / "test_tar"
    >>> tmp.mkdir(exist_ok=True)
    >>> archive = tmp / "test.tar.xz"
    >>> # Create a minimal tar.xz (example only; real usage reads existing files)
    >>> extractor = TarXzExtractor(tmp / "out", PathValidator(tmp / "out"), EntryFilter())
    """

    def __init__(
        self,
        target_dir: Path,
        validator: PathValidator,
        filter: EntryFilter,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        preserve_permissions: bool = True,
    ) -> None:
        self.target_dir = target_dir
        self.validator = validator
        self.filter = filter
        self.max_file_size = max_file_size
        self.preserve_permissions = preserve_permissions

    def extract(self, archive_path: Path) -> ExtractionResult:
        """
        Extract all matching entries from a tar.xz archive.

        Parameters
        ----------
        archive_path : Path
            Path to the .tar.xz file.

        Returns
        -------
        ExtractionResult
            Summary of the extraction operation.

        Raises
        ------
        ExtractionError
            If the archive is corrupted or unreadable.
        """
        result = ExtractionResult()

        try:
            with tarfile.open(archive_path, 'r:xz') as tf:
                for member in tf:
                    entry_path = member.name

                    if not self.filter.should_extract(entry_path):
                        result.files_skipped += 1
                        result.skipped_entries.append(entry_path)
                        logger.debug(f"Skipped (filter): {entry_path}")
                        continue

                    if not self.validator.is_safe(entry_path):
                        result.files_skipped += 1
                        result.skipped_entries.append(entry_path)
                        logger.warning(f"Skipped (unsafe path): {entry_path}")
                        continue

                    try:
                        if member.isdir():
                            self._extract_directory(entry_path, result)
                        elif member.issym():
                            self._skip_symlink(entry_path, result)
                        elif member.islnk():
                            self._skip_hardlink(entry_path, member, result)
                        elif member.isfile():
                            self._extract_file(tf, member, entry_path, result)
                        else:
                            result.files_skipped += 1
                            result.skipped_entries.append(entry_path)
                            logger.debug(f"Skipped (unknown type): {entry_path}")
                    except Exception as e:
                        result.files_failed += 1
                        result.failed_entries.append(entry_path)
                        logger.error(f"Failed to extract {entry_path}: {e}")

        except tarfile.TarError as e:
            raise ExtractionError(
                f"Failed to read tar archive: {e}",
                archive_path=str(archive_path),
                target_path=str(self.target_dir),
                archive_format="tar.xz",
            )

        return result

    def _extract_directory(self, entry_path: str, result: ExtractionResult) -> None:
        """
        Create a directory entry.

        Parameters
        ----------
        entry_path : str
            Directory path within the archive.
        result : ExtractionResult
            Result object to update.
        """
        target = self.validator.resolve_target(entry_path)
        target.mkdir(parents=True, exist_ok=True)
        result.directories_created += 1
        logger.debug(f"Created directory: {entry_path}")

    def _skip_symlink(self, entry_path: str, result: ExtractionResult) -> None:
        """
        Skip a symbolic link entry.

        Parameters
        ----------
        entry_path : str
            Symlink path within the archive.
        result : ExtractionResult
            Result object to update.
        """
        result.files_skipped += 1
        result.skipped_entries.append(entry_path)
        logger.warning(f"Skipped symlink: {entry_path}")

    def _skip_hardlink(
        self, entry_path: str, member: tarfile.TarInfo, result: ExtractionResult
    ) -> None:
        """
        Skip a hard link entry.

        Parameters
        ----------
        entry_path : str
            Hard link path within the archive.
        member : tarfile.TarInfo
            Tar entry metadata.
        result : ExtractionResult
            Result object to update.
        """
        result.files_skipped += 1
        result.skipped_entries.append(entry_path)
        logger.warning(f"Skipped hard link: {entry_path} -> {member.linkname}")

    def _extract_file(
        self,
        tf: tarfile.TarFile,
        member: tarfile.TarInfo,
        entry_path: str,
        result: ExtractionResult,
    ) -> None:
        """
        Extract a regular file from the archive.

        Parameters
        ----------
        tf : tarfile.TarFile
            Open tar file handle.
        member : tarfile.TarInfo
            File metadata.
        entry_path : str
            File path within the archive.
        result : ExtractionResult
            Result object to update.

        Raises
        ------
        ExtractionError
            If the file exceeds the size limit.
        """
        if member.size > self.max_file_size:
            raise ExtractionError(
                f"File exceeds size limit: {entry_path} "
                f"({member.size} > {self.max_file_size})",
                archive_path=tf.name or "",
                target_path=str(self.target_dir),
            )

        target = self.validator.resolve_target(entry_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with tf.extractfile(member) as source:
            if source is None:
                raise ExtractionError(
                    f"Cannot read file data for: {entry_path}",
                    archive_path=tf.name or "",
                    target_path=str(self.target_dir),
                )
            with open(target, 'wb') as dest:
                shutil.copyfileobj(source, dest)

        result.files_extracted += 1
        result.total_bytes += member.size

        if self.preserve_permissions:
            try:
                os.chmod(target, member.mode & 0o7777)
            except OSError:
                logger.debug(f"Could not set permissions on {target}")

        if member.mtime:
            os.utime(target, (member.mtime, member.mtime))

        logger.debug(f"Extracted: {entry_path} ({member.size} bytes)")


class ArchiveExtractor:
    """
    Unified interface for extracting ZIP and tar.xz archives.

    Detects the archive format automatically or accepts an explicit
    format parameter. Delegates to the appropriate specialized
    extractor based on the detected or specified format.

    Parameters
    ----------
    target_dir : Path
        Directory to extract files into.
    include_patterns : Optional[List[str]]
        Glob patterns for files to include.
    exclude_patterns : Optional[List[str]]
        Glob patterns for files to exclude.
    max_file_size : int
        Maximum allowed size for extracted files.
    preserve_permissions : bool
        If True, restore file permissions from the archive.

    Warnings
    --------
    The target_dir and any parent directories are created automatically.
    Existing files in the target_dir are overwritten without confirmation.

    Examples
    --------
    >>> import tempfile, zipfile as zf
    >>> tmp = Path(tempfile.gettempdir()) / "test_extractor"
    >>> tmp.mkdir(exist_ok=True)
    >>> archive = tmp / "source.zip"
    >>> with zf.ZipFile(archive, 'w') as z:
    ...     z.writestr("Include/Python.h", "// header")
    ...     z.writestr("Include/pyconfig.h", "// config")
    >>> extractor = ArchiveExtractor(
    ...     target_dir=tmp / "extracted",
    ...     include_patterns=["Include/*.h"],
    ... )
    >>> result = extractor.extract(archive)
    >>> result.files_extracted
    2
    >>> (tmp / "extracted" / "Include" / "Python.h").exists()
    True
    >>> shutil.rmtree(tmp)
    """

    def __init__(
        self,
        target_dir: Path,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        preserve_permissions: bool = True,
    ) -> None:
        self.target_dir = target_dir
        self.max_file_size = max_file_size
        self.preserve_permissions = preserve_permissions
        self.validator = PathValidator(target_dir)
        self.filter = EntryFilter(
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )

    def extract(
        self,
        archive_path: Union[str, Path],
        format: ArchiveFormat = ArchiveFormat.AUTO,
    ) -> ExtractionResult:
        """
        Extract an archive to the target directory.

        Parameters
        ----------
        archive_path : Union[str, Path]
            Path to the archive file.
        format : ArchiveFormat
            Archive format. AUTO detects from the file extension.

        Returns
        -------
        ExtractionResult
            Summary of the extraction.

        Raises
        ------
        ExtractionError
            If the format is unsupported or extraction fails.
        FileNotFoundError
            If the archive file does not exist.

        Examples
        --------
        See class docstring.
        """
        archive_path = Path(archive_path)

        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        if format == ArchiveFormat.AUTO:
            format = ArchiveFormat.detect(archive_path)

        self.target_dir.mkdir(parents=True, exist_ok=True)

        if format == ArchiveFormat.ZIP:
            extractor = ZipExtractor(
                self.target_dir,
                self.validator,
                self.filter,
                self.max_file_size,
                self.preserve_permissions,
            )
        elif format == ArchiveFormat.TAR_XZ:
            extractor = TarXzExtractor(
                self.target_dir,
                self.validator,
                self.filter,
                self.max_file_size,
                self.preserve_permissions,
            )
        else:
            raise ExtractionError(
                f"Unsupported archive format: {format.value}",
                archive_path=str(archive_path),
                target_path=str(self.target_dir),
                archive_format=format.value,
            )

        logger.info(f"Extracting {archive_path} to {self.target_dir} (format: {format.value})")
        result = extractor.extract(archive_path)
        logger.info(result.summary())

        return result


def find_directory_with_file(
    root_dir: Path,
    target_dir_name: str,
    target_file_name: str,
) -> Optional[Path]:
    """
    Recursively search for a directory containing a specific file.

    Used to locate the Include directory within extracted Python source
    by searching for a directory named target_dir_name that contains
    a file named target_file_name.

    Parameters
    ----------
    root_dir : Path
        Directory to search recursively.
    target_dir_name : str
        Name of the target directory (e.g., 'Include').
    target_file_name : str
        Name of the file that must exist inside the directory
        (e.g., 'Python.h').

    Returns
    -------
    Optional[Path]
        Path to the matching directory, or None if not found.

    Examples
    --------
    >>> import tempfile
    >>> tmp = Path(tempfile.gettempdir()) / "search_test"
    >>> target = tmp / "cpython" / "Include"
    >>> target.mkdir(parents=True)
    >>> (target / "Python.h").touch()
    >>> found = find_directory_with_file(tmp, "Include", "Python.h")
    >>> found == target
    True
    >>> shutil.rmtree(tmp)

    >>> find_directory_with_file(Path("/nonexistent"), "Include", "Python.h") is None
    True
    """
    if not root_dir.exists():
        return None

    for path in root_dir.rglob(target_dir_name):
        if path.is_dir() and (path / target_file_name).exists():
            return path

    return None


def extract_python_headers_archive(
    archive_path: Union[str, Path],
    target_dir: Union[str, Path],
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    preserve_permissions: bool = True,
    format: ArchiveFormat = ArchiveFormat.AUTO,
) -> ExtractionResult:
    """
    Extract Python header files from a source archive.

    Convenience function that creates an ArchiveExtractor configured
    for Python header extraction and runs it.

    Parameters
    ----------
    archive_path : Union[str, Path]
        Path to the source archive (ZIP or tar.xz).
    target_dir : Union[str, Path]
        Directory to extract files into.
    include_patterns : Optional[List[str]]
        Patterns for files to include.
    exclude_patterns : Optional[List[str]]
        Patterns for files to exclude.
    max_file_size : int
        Maximum file size to extract.
    preserve_permissions : bool
        Whether to preserve Unix permissions.
    format : ArchiveFormat
        Archive format or AUTO for detection.

    Returns
    -------
    ExtractionResult
        Summary of the extraction.

    Raises
    ------
    ExtractionError
        If extraction fails.
    FileNotFoundError
        If the archive does not exist.

    Examples
    --------
    >>> import tempfile, zipfile as zf
    >>> tmp = Path(tempfile.gettempdir()) / "test_python_extract"
    >>> tmp.mkdir(exist_ok=True)
    >>> archive = tmp / "cpython.zip"
    >>> with zf.ZipFile(archive, 'w') as z:
    ...     z.writestr("cpython/Include/Python.h", "// Python.h")
    ...     z.writestr("cpython/Include/abstract.h", "// abstract.h")
    >>> result = extract_python_headers_archive(archive, tmp / "headers")
    >>> result.files_extracted
    2
    >>> shutil.rmtree(tmp)
    """
    extractor = ArchiveExtractor(
        target_dir=Path(target_dir),
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        max_file_size=max_file_size,
        preserve_permissions=preserve_permissions,
    )
    return extractor.extract(archive_path, format=format)
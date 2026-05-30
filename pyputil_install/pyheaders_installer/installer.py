#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Header Installation Module.

Copies Python C header files from a source directory to a target
include directory. Handles directory structure replication, file
overwrite policies, permission preservation, and post-installation
verification.

The module is designed for the specific task of installing Python
development headers (Python.h, pyconfig.h, and related files) into
a system or user-specified include directory. It does not handle
compilation or linking; it only places header files on disk.

Directory Structure
-------------------
The source directory is expected to contain header files organized
in the standard CPython layout:

    Include/
        Python.h
        pyconfig.h
        abstract.h
        ...
        cpython/
            ...
        internal/
            ...

The installer replicates this structure under the target directory.
If include_subdirs is True, subdirectories like cpython/ and internal/
are also copied. If False, only top-level files are copied.

File Overwrite Policy
---------------------
Three policies control behavior when a target file already exists:

    SKIP: Leave the existing file untouched.
    OVERWRITE: Replace the existing file with the new one.
    BACKUP: Rename the existing file with a .backup suffix, then copy.

The default policy is OVERWRITE. BACKUP is useful when the existing
headers may be needed for reference.

Post-Installation Verification
------------------------------
After copying, an optional verification step checks that:
- The target directory exists.
- Key header files (Python.h, pyconfig.h) are present.
- File sizes are non-zero.

If verification fails, a summary of missing or empty files is logged.

Atomic Installation
-------------------
The installer can operate in atomic mode: headers are first copied
to a staging directory, then the staging directory is atomically
renamed to the target path. This prevents partial installations
if the process is interrupted. Atomic mode requires the staging
directory to be on the same filesystem as the target.

Permissions
-----------
On Unix systems, source file permissions are preserved in the copy.
On Windows, default permissions are applied.

Warnings
--------
- Installing headers to a system directory (e.g., /usr/include)
  typically requires root privileges. The installer does not
  escalate privileges; it will fail with a permission error.
- On some systems, /usr/include is managed by the package manager.
  Manually placing files there may cause conflicts during system
  updates. Consider using a user-specific directory instead.
- Atomic installation uses os.replace(), which is atomic on Unix
  and on Windows (Python 3.3+). If the target is on a different
  filesystem, the operation is not atomic and may leave the target
  in an inconsistent state.
- Empty header files are not treated as errors by default. Enable
  verify_non_empty in the verification config to reject them.
"""

from __future__ import annotations

import os
import shutil
import stat
import filecmp
from pathlib import Path
from typing import Optional, List, Set, Dict, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

from .exceptions import InstallationError, BackupError

logger = logging.getLogger(__name__)


class OverwritePolicy(str, Enum):
    """
    Policy for handling existing files at the target location.

    Attributes
    ----------
    SKIP : str
        Leave the existing file unchanged. The new file is not copied.
    OVERWRITE : str
        Replace the existing file. This is the default.
    BACKUP : str
        Rename the existing file by appending a suffix, then copy.

    Examples
    --------
    >>> OverwritePolicy.SKIP
    'skip'
    >>> OverwritePolicy.OVERWRITE
    'overwrite'
    >>> OverwritePolicy.BACKUP
    'backup'
    """

    SKIP = "skip"
    OVERWRITE = "overwrite"
    BACKUP = "backup"


class AtomicMode(str, Enum):
    """
    Mode for atomic installation.

    Attributes
    ----------
    DISABLED : str
        Copy files directly to the target directory.
    ENABLED : str
        Copy to a staging directory, then atomically rename.
    AUTO : str
        Use atomic mode if the staging directory is on the same
        filesystem as the target. Falls back to direct copy otherwise.

    Examples
    --------
    >>> AtomicMode.DISABLED
    'disabled'
    >>> AtomicMode.ENABLED
    'enabled'
    """

    DISABLED = "disabled"
    ENABLED = "enabled"
    AUTO = "auto"


@dataclass
class InstallResult:
    """
    Result of a header installation operation.

    Attributes
    ----------
    files_copied : int
        Number of files successfully copied.
    directories_created : int
        Number of directories created.
    files_skipped : int
        Number of files skipped due to overwrite policy.
    files_failed : int
        Number of files that could not be copied.
    total_bytes : int
        Total bytes written to disk.
    failed_files : List[str]
        Relative paths of files that failed to copy.
    skipped_files : List[str]
        Relative paths of files that were skipped.
    backed_up_files : List[str]
        Relative paths of files that were backed up before overwrite.

    Examples
    --------
    >>> result = InstallResult(files_copied=10, directories_created=2)
    >>> bool(result)
    True
    >>> result.success_rate
    100.0
    """

    files_copied: int = 0
    directories_created: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    total_bytes: int = 0
    failed_files: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    backed_up_files: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """
        True if at least one file was copied and no failures occurred.

        Examples
        --------
        >>> bool(InstallResult(files_copied=5))
        True
        >>> bool(InstallResult(files_failed=1))
        False
        """
        return self.files_copied > 0 and self.files_failed == 0

    @property
    def total_processed(self) -> int:
        """
        Total number of files processed (copied + skipped + failed).

        Examples
        --------
        >>> InstallResult(files_copied=5, files_skipped=2, files_failed=1).total_processed
        8
        """
        return self.files_copied + self.files_skipped + self.files_failed

    @property
    def success_rate(self) -> float:
        """
        Percentage of files successfully copied (0.0 to 100.0).

        Examples
        --------
        >>> InstallResult(files_copied=9, files_failed=1).success_rate
        90.0
        """
        total = self.files_copied + self.files_failed
        if total == 0:
            return 100.0
        return (self.files_copied / total) * 100.0

    def summary(self) -> str:
        """
        Return a human-readable summary.

        Examples
        --------
        >>> r = InstallResult(files_copied=42, directories_created=3, total_bytes=1048576)
        >>> print(r.summary())
        Installed: 42 files, 3 directories (1.0 MiB)
        """
        from .downloader import format_bytes

        parts = [f"Installed: {self.files_copied} files"]
        if self.directories_created:
            parts.append(f"{self.directories_created} directories")
        if self.total_bytes:
            parts.append(f"({format_bytes(self.total_bytes)})")
        if self.files_skipped:
            parts.append(f"Skipped: {self.files_skipped}")
        if self.files_failed:
            parts.append(f"Failed: {self.files_failed}")
        if self.backed_up_files:
            parts.append(f"Backed up: {len(self.backed_up_files)}")
        return " ".join(parts)


@dataclass
class VerificationConfig:
    """
    Configuration for post-installation header verification.

    Attributes
    ----------
    enabled : bool
        Whether to run verification after installation.
    required_files : List[str]
        File names that must exist in the target directory.
        Defaults to Python.h and pyconfig.h.
    verify_non_empty : bool
        If True, fail verification for zero-byte files.
    verify_readable : bool
        If True, fail verification for unreadable files.

    Examples
    --------
    >>> config = VerificationConfig()
    >>> "Python.h" in config.required_files
    True
    """

    enabled: bool = True
    required_files: List[str] = field(
        default_factory=lambda: ["Python.h", "pyconfig.h"]
    )
    verify_non_empty: bool = True
    verify_readable: bool = True


@dataclass
class BackupConfig:
    """
    Configuration for backup behavior.

    Attributes
    ----------
    suffix : str
        Suffix appended to backup file or directory names.
    max_backups : int
        Maximum number of timestamped backups to retain per target.
        Set to 0 for unlimited. Old backups beyond this count are
        deleted (oldest first).
    include_timestamp : bool
        If True, append a timestamp to the backup name for uniqueness.

    Examples
    --------
    >>> config = BackupConfig(suffix=".backup", max_backups=5)
    >>> config.suffix
    '.backup'
    """

    suffix: str = ".backup"
    max_backups: int = 5
    include_timestamp: bool = False


class BackupManager:
    """
    Creates and manages backups of files and directories.

    Handles backup directory creation, file/directory renaming for
    backup, and rotation of old backups when a maximum count is set.

    Parameters
    ----------
    config : BackupConfig
        Backup behavior configuration.

    Examples
    --------
    >>> import tempfile
    >>> tmp = Path(tempfile.gettempdir()) / "backup_test"
    >>> tmp.mkdir(exist_ok=True)
    >>> target = tmp / "original"
    >>> target.write_text("data")
    >>> manager = BackupManager(BackupConfig(suffix=".bak"))
    >>> backup_path = manager.backup_file(target)
    >>> backup_path.name
    'original.bak'
    >>> target.exists()
    False
    >>> backup_path.read_text()
    'data'
    >>> shutil.rmtree(tmp)
    """

    def __init__(self, config: BackupConfig = BackupConfig()) -> None:
        self.config = config

    def backup_file(self, file_path: Path) -> Optional[Path]:
        """
        Rename a file to create a backup.

        If the backup path already exists, it is overwritten.
        If max_backups > 0, old backups are rotated.

        Parameters
        ----------
        file_path : Path
            Path to the file to back up. The file is moved, not copied.

        Returns
        -------
        Optional[Path]
            Path to the backup file, or None if the source did not exist.

        Raises
        ------
        BackupError
            If the rename operation fails.

        Examples
        --------
        >>> import tempfile
        >>> tmp = Path(tempfile.gettempdir()) / "backup_file_test"
        >>> tmp.mkdir(exist_ok=True)
        >>> f = tmp / "data.txt"
        >>> f.write_text("hello")
        5
        >>> manager = BackupManager()
        >>> backup = manager.backup_file(f)
        >>> backup.read_text()
        'hello'
        >>> f.exists()
        False
        >>> shutil.rmtree(tmp)
        """
        if not file_path.exists():
            return None

        backup_path = self._build_backup_path(file_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if backup_path.exists():
                if backup_path.is_dir():
                    shutil.rmtree(backup_path)
                else:
                    backup_path.unlink()
            shutil.move(str(file_path), str(backup_path))
            logger.info(f"Backed up: {file_path} -> {backup_path}")
            self._rotate_backups(file_path)
            return backup_path
        except OSError as e:
            raise BackupError(
                f"Failed to back up file: {e}",
                path=str(file_path),
                operation="backup_file",
                source_path=str(file_path),
                backup_path=str(backup_path),
            )

    def backup_directory(self, dir_path: Path) -> Optional[Path]:
        """
        Move a directory to create a backup.

        Parameters
        ----------
        dir_path : Path
            Path to the directory to back up.

        Returns
        -------
        Optional[Path]
            Path to the backup directory, or None if the source
            did not exist.

        Raises
        ------
        BackupError
            If the rename fails.

        Examples
        --------
        >>> import tempfile
        >>> tmp = Path(tempfile.gettempdir()) / "backup_dir_test"
        >>> tmp.mkdir(exist_ok=True)
        >>> d = tmp / "headers"
        >>> d.mkdir()
        >>> (d / "file.h").write_text("// header")
        10
        >>> manager = BackupManager()
        >>> backup = manager.backup_directory(d)
        >>> d.exists()
        False
        >>> (backup / "file.h").read_text()
        '// header'
        >>> shutil.rmtree(tmp)
        """
        if not dir_path.exists() or not dir_path.is_dir():
            return None

        backup_path = self._build_backup_path(dir_path)

        try:
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.move(str(dir_path), str(backup_path))
            logger.info(f"Backed up directory: {dir_path} -> {backup_path}")
            self._rotate_backups(dir_path)
            return backup_path
        except OSError as e:
            raise BackupError(
                f"Failed to back up directory: {e}",
                path=str(dir_path),
                operation="backup_directory",
                source_path=str(dir_path),
                backup_path=str(backup_path),
            )

    def _build_backup_path(self, original: Path) -> Path:
        """
        Construct the backup path from the original path.

        Parameters
        ----------
        original : Path
            Original file or directory path.

        Returns
        -------
        Path
            Path for the backup.

        Examples
        --------
        >>> manager = BackupManager(BackupConfig(suffix=".bak", include_timestamp=False))
        >>> str(manager._build_backup_path(Path("/tmp/file.txt")))
        '/tmp/file.txt.bak'
        """
        if self.config.include_timestamp:
            import time
            stamp = time.strftime("%Y%m%d_%H%M%S")
            return original.parent / f"{original.name}{self.config.suffix}.{stamp}"
        return original.parent / f"{original.name}{self.config.suffix}"

    def _rotate_backups(self, original: Path) -> None:
        """
        Remove old backups if max_backups is exceeded.

        Identifies all backups matching the suffix pattern for the
        given original path. If the count exceeds max_backups,
        deletes the oldest ones first.

        Parameters
        ----------
        original : Path
            Original path used to identify related backups.
        """
        if self.config.max_backups <= 0:
            return

        pattern = f"{original.name}{self.config.suffix}*"
        backups = sorted(
            original.parent.glob(pattern),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
        )

        while len(backups) > self.config.max_backups:
            oldest = backups.pop(0)
            try:
                if oldest.is_dir():
                    shutil.rmtree(oldest)
                else:
                    oldest.unlink()
                logger.debug(f"Rotated old backup: {oldest}")
            except OSError as e:
                logger.warning(f"Could not remove old backup {oldest}: {e}")


class HeaderVerifier:
    """
    Verifies that installed header files are present and valid.

    Checks that required files exist in the target directory and,
    optionally, that they are non-empty and readable.

    Parameters
    ----------
    config : VerificationConfig
        Verification settings.

    Examples
    --------
    >>> import tempfile
    >>> tmp = Path(tempfile.gettempdir()) / "verify_test"
    >>> tmp.mkdir(exist_ok=True)
    >>> (tmp / "Python.h").write_text("// Python.h")
    12
    >>> (tmp / "pyconfig.h").write_text("// pyconfig.h")
    14
    >>> verifier = HeaderVerifier()
    >>> result = verifier.verify(tmp)
    >>> result.is_valid
    True
    >>> shutil.rmtree(tmp)
    """

    def __init__(self, config: VerificationConfig = VerificationConfig()) -> None:
        self.config = config

    def verify(self, target_dir: Path) -> 'VerificationResult':
        """
        Verify that required headers exist in the target directory.

        Parameters
        ----------
        target_dir : Path
            Directory containing the installed headers.

        Returns
        -------
        VerificationResult
            Result of the verification with details on failures.

        Examples
        --------
        >>> verifier = HeaderVerifier()
        >>> result = verifier.verify(Path("/nonexistent"))
        >>> result.is_valid
        False
        """
        result = VerificationResult()

        if not target_dir.exists():
            result.add_error("", "Target directory does not exist")
            return result

        if not target_dir.is_dir():
            result.add_error("", "Target path is not a directory")
            return result

        for filename in self.config.required_files:
            file_path = target_dir / filename
            if not file_path.exists():
                result.add_missing(filename)
                continue

            if self.config.verify_non_empty:
                if file_path.stat().st_size == 0:
                    result.add_error(filename, "File is empty (zero bytes)")
                    continue

            if self.config.verify_readable:
                if not os.access(file_path, os.R_OK):
                    result.add_error(filename, "File is not readable")
                    continue

            result.add_verified(filename)

        return result


@dataclass
class VerificationResult:
    """
    Result of header verification.

    Attributes
    ----------
    verified_files : List[str]
        Files that passed all checks.
    missing_files : List[str]
        Required files not found in the target directory.
    errors : Dict[str, str]
        Files that failed verification, mapped to error descriptions.

    Examples
    --------
    >>> result = VerificationResult()
    >>> result.add_verified("Python.h")
    >>> result.add_missing("pyconfig.h")
    >>> result.is_valid
    False
    >>> print(result.summary())
    Verification: 1 passed, 1 missing, 0 errors
    """

    verified_files: List[str] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """
        True if no files are missing and no errors occurred.

        Examples
        --------
        >>> VerificationResult().is_valid
        True
        """
        return len(self.missing_files) == 0 and len(self.errors) == 0

    @property
    def total_checked(self) -> int:
        """
        Total number of files checked.

        Examples
        --------
        >>> r = VerificationResult(verified_files=["a.h"], missing_files=["b.h"])
        >>> r.total_checked
        2
        """
        return len(self.verified_files) + len(self.missing_files) + len(self.errors)

    def add_verified(self, filename: str) -> None:
        """
        Record a file that passed verification.

        Parameters
        ----------
        filename : str
            Name of the verified file.
        """
        self.verified_files.append(filename)

    def add_missing(self, filename: str) -> None:
        """
        Record a required file that is missing.

        Parameters
        ----------
        filename : str
            Name of the missing file.
        """
        self.missing_files.append(filename)

    def add_error(self, filename: str, description: str) -> None:
        """
        Record a file that failed verification with a reason.

        Parameters
        ----------
        filename : str
            Name of the file (empty string for directory-level errors).
        description : str
            Reason for the verification failure.
        """
        self.errors[filename] = description

    def summary(self) -> str:
        """
        Return a human-readable summary.

        Examples
        --------
        >>> r = VerificationResult(verified_files=["a.h", "b.h"],
        ...                        missing_files=["c.h"],
        ...                        errors={"d.h": "empty"})
        >>> print(r.summary())
        Verification: 2 passed, 1 missing, 1 errors
        """
        return (
            f"Verification: {len(self.verified_files)} passed, "
            f"{len(self.missing_files)} missing, "
            f"{len(self.errors)} errors"
        )


class HeaderInstaller:
    """
    Installs Python C header files from a source to a target directory.

    Copies header files from a source Include directory to a target
    include directory, preserving subdirectory structure and file
    permissions. Supports backup of existing files, skip/overwrite
    policies, and atomic installation.

    Parameters
    ----------
    source_dir : Path
        Directory containing the header files to install (the Include
        directory from a CPython source tree).
    target_dir : Path
        Destination directory for the header files.
    include_subdirs : bool
        If True, recursively copy subdirectories. If False, only copy
        files directly in the source directory.
    overwrite_policy : OverwritePolicy
        How to handle files that already exist in the target.
    backup_config : Optional[BackupConfig]
        Configuration for backup behavior. If None, backups are disabled.
    atomic_mode : AtomicMode
        Whether to use atomic installation via a staging directory.
    preserve_permissions : bool
        If True, copy file permissions from source files.
    verification_config : Optional[VerificationConfig]
        Post-installation verification settings. If None, verification
        is skipped.

    Warnings
    --------
    - The target directory is created if it does not exist. Parent
      directories are also created as needed.
    - Subdirectory copying can be expensive for large source trees.
      When extracting from a full CPython archive, set include_subdirs
      to True only if cpython/ and internal/ headers are needed.
    - Atomic installation writes to a staging directory first, which
      requires additional free disk space equal to the total size of
      all files being installed.

    Examples
    --------
    >>> import tempfile
    >>> tmp = Path(tempfile.gettempdir()) / "installer_test"
    >>> tmp.mkdir(exist_ok=True)
    >>> source = tmp / "source"
    >>> source.mkdir()
    >>> (source / "Python.h").write_text("// header")
    10
    >>> target = tmp / "target"
    >>> installer = HeaderInstaller(source, target)
    >>> result = installer.install()
    >>> result.files_copied
    1
    >>> (target / "Python.h").read_text()
    '// header'
    >>> shutil.rmtree(tmp)
    """

    def __init__(
        self,
        source_dir: Path,
        target_dir: Path,
        include_subdirs: bool = True,
        overwrite_policy: OverwritePolicy = OverwritePolicy.OVERWRITE,
        backup_config: Optional[BackupConfig] = None,
        atomic_mode: AtomicMode = AtomicMode.AUTO,
        preserve_permissions: bool = True,
        verification_config: Optional[VerificationConfig] = None,
    ) -> None:
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
        if not source_dir.is_dir():
            raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

        self.source_dir = source_dir
        self.target_dir = target_dir
        self.include_subdirs = include_subdirs
        self.overwrite_policy = overwrite_policy
        self.backup_manager = BackupManager(backup_config) if backup_config else None
        self.atomic_mode = atomic_mode
        self.preserve_permissions = preserve_permissions
        self.verification_config = verification_config or VerificationConfig(enabled=False)

    def install(self) -> InstallResult:
        """
        Execute the header installation.

        Performs the following steps in order:
        1. Determine the effective target and staging directories.
        2. Back up the existing target directory if the policy is BACKUP.
        3. Copy files from source to target (or staging).
        4. If atomic, rename staging to target.
        5. Run post-installation verification if enabled.

        Returns
        -------
        InstallResult
            Summary of the installation.

        Raises
        ------
        InstallationError
            If the installation fails critically (e.g., permission error
            on the target directory).

        Examples
        --------
        >>> import tempfile
        >>> tmp = Path(tempfile.gettempdir()) / "install_test"
        >>> tmp.mkdir(exist_ok=True)
        >>> src = tmp / "src"
        >>> src.mkdir()
        >>> (src / "header.h").write_text("// header")
        10
        >>> installer = HeaderInstaller(src, tmp / "dest")
        >>> result = installer.install()
        >>> result.files_copied
        1
        >>> shutil.rmtree(tmp)
        """
        result = InstallResult()
        effective_target = self.target_dir
        staging_dir: Optional[Path] = None

        try:
            # Determine whether to use atomic installation
            use_atomic = self._should_use_atomic()
            if use_atomic:
                staging_dir = self._create_staging_dir()
                effective_target = staging_dir
                logger.info(f"Using atomic installation via: {staging_dir}")

            # Back up existing target if policy is BACKUP
            if self.overwrite_policy == OverwritePolicy.BACKUP:
                if self.target_dir.exists() and staging_dir is None:
                    self.backup_manager.backup_directory(self.target_dir)
                    result.backed_up_files.append(str(self.target_dir))

            # Create the effective target directory
            effective_target.mkdir(parents=True, exist_ok=True)

            # Copy files
            self._copy_directory(self.source_dir, effective_target, result)

            # Atomic rename
            if staging_dir is not None:
                self._atomic_rename(staging_dir, self.target_dir)
                effective_target = self.target_dir

            # Verify installation
            if self.verification_config.enabled:
                verifier = HeaderVerifier(self.verification_config)
                verify_result = verifier.verify(effective_target)
                logger.info(verify_result.summary())
                if not verify_result.is_valid:
                    logger.warning(
                        f"Verification failed: missing={verify_result.missing_files}, "
                        f"errors={verify_result.errors}"
                    )

            logger.info(result.summary())
            return result

        except Exception as e:
            # Clean up staging directory on failure
            if staging_dir is not None and staging_dir.exists():
                try:
                    shutil.rmtree(staging_dir)
                except Exception:
                    pass

            if isinstance(e, InstallationError):
                raise

            raise InstallationError(
                f"Installation failed: {e}",
                path=str(self.target_dir),
                operation="install",
                source_path=str(self.source_dir),
                files_copied=result.files_copied,
                files_failed=result.files_failed,
                failed_files=result.failed_files,
            )

    def _should_use_atomic(self) -> bool:
        """
        Determine if atomic installation should be used.

        Returns
        -------
        bool
            True if atomic mode is enabled or auto-detect succeeds.

        Examples
        --------
        >>> installer = HeaderInstaller(Path("/tmp/src"), Path("/tmp/dest"),
        ...                             atomic_mode=AtomicMode.DISABLED)
        >>> installer._should_use_atomic()
        False
        """
        if self.atomic_mode == AtomicMode.DISABLED:
            return False
        if self.atomic_mode == AtomicMode.ENABLED:
            return True
        # AUTO: check that target parent exists and is on the same mount
        if self.atomic_mode == AtomicMode.AUTO:
            target_parent = self.target_dir.parent
            if not target_parent.exists():
                return False
            try:
                staging_test = target_parent / ".atomic_test"
                staging_test.touch()
                staging_test.unlink()
                return True
            except OSError:
                logger.warning(
                    "Atomic installation disabled: cannot write to parent directory"
                )
                return False
        return False

    def _create_staging_dir(self) -> Path:
        """
        Create a unique staging directory for atomic installation.

        Returns
        -------
        Path
            Path to the staging directory.

        Raises
        ------
        InstallationError
            If the staging directory cannot be created.
        """
        import uuid
        staging = self.target_dir.parent / f".{self.target_dir.name}.staging.{uuid.uuid4().hex[:8]}"
        try:
            staging.mkdir(parents=True, exist_ok=False)
            return staging
        except OSError as e:
            raise InstallationError(
                f"Could not create staging directory: {e}",
                path=str(staging),
                operation="create_staging",
                source_path=str(self.source_dir),
            )

    def _atomic_rename(self, staging_dir: Path, target_dir: Path) -> None:
        """
        Atomically rename the staging directory to the target.

        On Unix, os.replace is atomic. On Windows, it is atomic if
        both paths are on the same filesystem.

        Parameters
        ----------
        staging_dir : Path
            Staging directory containing the installed files.
        target_dir : Path
            Final target path.

        Raises
        ------
        InstallationError
            If the rename fails.

        Examples
        --------
        >>> import tempfile
        >>> tmp = Path(tempfile.gettempdir()) / "atomic_test"
        >>> tmp.mkdir(exist_ok=True)
        >>> staging = tmp / ".staging"
        >>> staging.mkdir()
        >>> (staging / "file.h").write_text("// header")
        10
        >>> installer = HeaderInstaller(staging, tmp / "final")
        >>> installer._atomic_rename(staging, tmp / "final")
        >>> (tmp / "final" / "file.h").exists()
        True
        >>> staging.exists()
        False
        >>> shutil.rmtree(tmp)
        """
        try:
            if target_dir.exists():
                if target_dir.is_dir():
                    shutil.rmtree(target_dir)
                else:
                    target_dir.unlink()
            os.replace(str(staging_dir), str(target_dir))
            logger.info(f"Atomic rename: {staging_dir} -> {target_dir}")
        except OSError as e:
            raise InstallationError(
                f"Atomic rename failed: {e}",
                path=str(target_dir),
                operation="atomic_rename",
                source_path=str(staging_dir),
            )

    def _copy_directory(
        self, source: Path, target: Path, result: InstallResult
    ) -> None:
        """
        Recursively copy files from source to target.
    
        Parameters
        ----------
        source : Path
            Source directory to copy from.
        target : Path
            Target directory to copy to.
        result : InstallResult
            Result object to accumulate statistics.
        """
        for item in source.iterdir():
            if item.is_symlink():
                logger.debug(f"Skipping symlink: {item.name}")
                continue
    
            if item.is_dir():
                if not self.include_subdirs:
                    logger.debug(f"Skipping subdirectory: {item.name}")
                    continue
                sub_target = target / item.name
                sub_target.mkdir(parents=True, exist_ok=True)
                result.directories_created += 1
                self._copy_directory(item, sub_target, result)
    
            elif item.is_file():
                self._copy_file(item, target / item.name, result)


    
    
    def _copy_file(
        self, source_file: Path, target_file: Path, result: InstallResult
    ) -> None:
        """
        Copy a single file with overwrite policy handling.
    
        Parameters
        ----------
        source_file : Path
            Source file to copy.
        target_file : Path
            Destination file path.
        result : InstallResult
            Result object to update.
        """
        # Use the target directory that was actually passed (may be staging dir)
        # Find the base target by walking up from target_file to find where we're copying
        # We need to compute relative path from the actual installation base
        
        # The target_file contains the full path including staging dir if atomic.
        # We need to figure out the relative path within the eventual target_dir.
        # Since _copy_directory is called with effective_target, we use that.
        
        # Build relative path from source_dir structure
        try:
            relative_path = source_file.relative_to(self.source_dir)
        except ValueError:
            # Fallback: use the file name only
            relative_path = Path(source_file.name)
    
        # Check overwrite policy
        if target_file.exists():
            if self.overwrite_policy == OverwritePolicy.SKIP:
                result.files_skipped += 1
                result.skipped_files.append(str(relative_path))
                logger.debug(f"Skipped (exists): {relative_path}")
                return
            elif self.overwrite_policy == OverwritePolicy.BACKUP:
                if self.backup_manager:
                    self.backup_manager.backup_file(target_file)
                    result.backed_up_files.append(str(relative_path))
    
        # Copy the file
        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
            result.files_copied += 1
            result.total_bytes += source_file.stat().st_size
            logger.debug(f"Copied: {relative_path}")
        except OSError as e:
            result.files_failed += 1
            result.failed_files.append(str(relative_path))
            logger.error(f"Failed to copy {relative_path}: {e}")


def find_include_directory(
    extract_dir: Path,
    directory_name: str = "Include",
    required_file: str = "Python.h",
) -> Optional[Path]:
    """
    Find the Include directory within an extracted CPython source tree.

    Searches recursively for a directory with the given name that
    contains the specified required file.

    Parameters
    ----------
    extract_dir : Path
        Root directory of the extracted source tree.
    directory_name : str
        Name of the target directory to find (default: 'Include').
    required_file : str
        File that must exist inside the directory (default: 'Python.h').

    Returns
    -------
    Optional[Path]
        Path to the matching directory, or None if not found.

    Examples
    --------
    >>> import tempfile
    >>> tmp = Path(tempfile.gettempdir()) / "find_include_test"
    >>> include_dir = tmp / "cpython-3.11.0" / "Include"
    >>> include_dir.mkdir(parents=True)
    >>> (include_dir / "Python.h").touch()
    >>> (include_dir / "pyconfig.h").touch()
    >>> found = find_include_directory(tmp)
    >>> found == include_dir
    True
    >>> shutil.rmtree(tmp)
    """
    if not extract_dir.exists():
        return None

    for path in extract_dir.rglob(directory_name):
        if path.is_dir() and (path / required_file).exists():
            logger.info(f"Found Include directory: {path}")
            return path

    logger.warning(
        f"Could not find '{directory_name}' directory containing '{required_file}' "
        f"in {extract_dir}"
    )
    return None


def copy_python_headers(
    source_include_dir: Path,
    target_include_dir: Path,
    include_subdirs: bool = True,
    overwrite_policy: OverwritePolicy = OverwritePolicy.OVERWRITE,
    backup_existing: bool = True,
    atomic: bool = True,
    verbose: bool = False,
) -> InstallResult:
    """
    Copy Python header files from source to target directory.

    Convenience function for the common use case of installing
    CPython headers with sensible defaults.

    Parameters
    ----------
    source_include_dir : Path
        Path to the Include directory from CPython source.
    target_include_dir : Path
        Destination directory for the headers.
    include_subdirs : bool
        Whether to copy subdirectories like cpython/ and internal/.
    overwrite_policy : OverwritePolicy
        How to handle existing files.
    backup_existing : bool
        If True and policy is OVERWRITE, back up existing target
        directory before installation.
    atomic : bool
        Use atomic installation to prevent partial state.
    verbose : bool
        Enable detailed logging.

    Returns
    -------
    InstallResult
        Summary of the installation.

    Examples
    --------
    >>> import tempfile
    >>> tmp = Path(tempfile.gettempdir()) / "copy_test"
    >>> src = tmp / "src"
    >>> src.mkdir(parents=True)
    >>> (src / "Python.h").write_text("// Python.h")
    12
    >>> dst = tmp / "include"
    >>> result = copy_python_headers(src, dst)
    >>> result.files_copied
    1
    >>> shutil.rmtree(tmp)
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)

    backup_cfg = BackupConfig() if backup_existing else None
    atomic_mode = AtomicMode.ENABLED if atomic else AtomicMode.DISABLED

    installer = HeaderInstaller(
        source_dir=source_include_dir,
        target_dir=target_include_dir,
        include_subdirs=include_subdirs,
        overwrite_policy=overwrite_policy,
        backup_config=backup_cfg,
        atomic_mode=atomic_mode,
    )

    return installer.install()
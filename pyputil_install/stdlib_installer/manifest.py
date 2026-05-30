"""
Package manifest and metadata management for stdlib_installer.

Tracks installed packages, their versions, constituent files, checksums,
and installation timestamps. Provides the authoritative record of what
is installed and where, enabling accurate removal, updates, and queries.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timezone

from .exceptions import (
    ModuleNotInstalledError,
    AlreadyInstalledError,
)

logger = logging.getLogger(__name__)


class Manifest:
    """
    Persistent JSON-backed record of installed standard library packages.

    Stores metadata for each installed package including the source
    Python version, list of installed file paths relative to the storage
    directory, per-file SHA-256 checksums, dependency names, and the
    UTC timestamp of installation.

    The manifest file is written atomically using a temporary file and
    rename to prevent corruption on concurrent writes or crashes.

    Parameters
    ----------
    storage_dir : Path
        Root directory where installed packages reside. The manifest
        file is stored as ``stdlib_installer.json`` inside this directory.

    Attributes
    ----------
    storage_dir : Path
        The root directory for installed packages.
    manifest_path : Path
        Full path to the manifest JSON file.
    _packages : dict
        In-memory dictionary of all manifest entries keyed by package name.

    Examples
    --------
    >>> manifest = Manifest(Path("/home/user/.stdlib-packages"))
    >>> manifest.add(
    ...     name="json",
    ...     version="3.11",
    ...     files=["json/__init__.py", "json/encoder.py"],
    ...     checksums={"json/__init__.py": "abc123...", "json/encoder.py": "def456..."},
    ...     dependencies=["os", "sys"]
    ... )
    >>> manifest.has("json")
    True
    >>> entry = manifest.get("json")
    >>> print(entry["version"])
    3.11
    """

    MANIFEST_FILENAME = "stdlib_installer.json"

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.storage_dir / self.MANIFEST_FILENAME
        self._packages: Dict[str, Dict[str, Any]] = {}

        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """
        Load the manifest from disk.

        If the manifest file does not exist, starts with an empty
        registry. If the file contains invalid JSON, logs a warning,
        backs up the corrupted file, and starts fresh.

        Returns
        -------
        None
            Populates ``self._packages`` in place.
        """
        if not self.manifest_path.exists():
            logger.debug(
                "No existing manifest at %s", self.manifest_path
            )
            return

        try:
            raw = self.manifest_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning(
                "Cannot read manifest at %s: %s", self.manifest_path, e
            )
            return

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(
                "Manifest JSON is corrupted: %s. Backing up and starting fresh.", e
            )
            self._backup_corrupted(raw)
            return

        if not isinstance(data, dict):
            logger.warning(
                "Manifest root is not a dict, backing up and starting fresh."
            )
            self._backup_corrupted(raw)
            return

        # Validate structure of each entry
        for name, entry in data.items():
            if isinstance(entry, dict):
                self._packages[name] = entry
            else:
                logger.warning(
                    "Skipping invalid entry for '%s': not a dict", name
                )

        logger.debug(
            "Loaded %d packages from manifest", len(self._packages)
        )

    def save(self) -> None:
        """
        Persist the manifest to disk atomically.

        Writes to a temporary file first, then renames it over the
        actual manifest path. This prevents partial writes from
        corrupting the manifest on crashes or power loss.

        Raises
        ------
        OSError
            If the write or rename fails.
        """
        temp_path = self.manifest_path.with_suffix(".tmp")

        try:
            temp_path.write_text(
                json.dumps(
                    self._packages, indent=2, ensure_ascii=False
                ),
                encoding="utf-8",
            )
            temp_path.replace(self.manifest_path)
        except OSError:
            # Clean up temp file if rename fails
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise

        logger.debug(
            "Saved manifest with %d packages", len(self._packages)
        )

    def _backup_corrupted(self, raw_content: str) -> None:
        """
        Rename a corrupted manifest file for later inspection.

        The corrupted file is moved to a name with a ``.corrupted``
        suffix appended. If the backup target already exists, a numeric
        suffix is added (e.g., ``.corrupted.1``).

        Parameters
        ----------
        raw_content : str
            The raw text content of the corrupted manifest, preserved
            in the backup file.
        """
        backup_path = self.manifest_path.with_suffix(
            self.manifest_path.suffix + ".corrupted"
        )

        # Find a unique backup name if one already exists
        counter = 1
        while backup_path.exists():
            backup_path = self.manifest_path.with_suffix(
                f"{self.manifest_path.suffix}.corrupted.{counter}"
            )
            counter += 1

        try:
            backup_path.write_text(raw_content, encoding="utf-8")
            logger.info(
                "Corrupted manifest backed up to %s", backup_path
            )
        except OSError as e:
            logger.error(
                "Failed to backup corrupted manifest: %s", e
            )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def has(self, name: str) -> bool:
        """
        Check if a package is recorded in the manifest.

        Parameters
        ----------
        name : str
            Package name to check.

        Returns
        -------
        bool
            ``True`` if the package exists in the manifest.

        Examples
        --------
        >>> manifest.has("json")
        True
        >>> manifest.has("nonexistent")
        False
        """
        return name in self._packages

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the full manifest entry for a package.

        Parameters
        ----------
        name : str
            Package name.

        Returns
        -------
        dict or None
            The manifest entry dict if found, ``None`` otherwise. The
            dict contains keys: ``version`` (str), ``installed_at``
            (str ISO timestamp), ``files`` (list of str relative paths),
            ``checksums`` (dict of str path to str hash),
            ``dependencies`` (list of str names).

        Examples
        --------
        >>> entry = manifest.get("json")
        >>> entry["version"]
        '3.11'
        >>> entry["files"]
        ['json/__init__.py', 'json/encoder.py', 'json/decoder.py']
        """
        return self._packages.get(name)

    def list_names(self) -> List[str]:
        """
        Return all installed package names in alphabetical order.

        Returns
        -------
        list of str
            Sorted list of package names.

        Examples
        --------
        >>> manifest.list_names()
        ['csv', 'json', 'xml']
        """
        return sorted(self._packages.keys())

    def list_all(self) -> List[Dict[str, Any]]:
        """
        Return full entries for all installed packages.

        Returns
        -------
        list of dict
            All manifest entries, sorted alphabetically by name.

        Examples
        --------
        >>> for entry in manifest.list_all():
        ...     print(entry["name"], entry["version"])
        csv 3.13
        json 3.13
        """
        return [
            self._packages[name] for name in self.list_names()
        ]

    def count(self) -> int:
        """
        Return the number of registered packages.

        Returns
        -------
        int
            Total package count.

        Examples
        --------
        >>> manifest.count()
        3
        """
        return len(self._packages)

    # ------------------------------------------------------------------
    # Modification methods
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        version: str,
        files: List[str],
        checksums: Dict[str, str],
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """
        Add or update a package entry in the manifest.

        Parameters
        ----------
        name : str
            Package name.
        version : str
            CPython version the package was installed from
            (e.g., ``"3.11"``).
        files : list of str
            Relative file paths from the storage directory for all
            installed files belonging to this package.
        checksums : dict
            Mapping of relative file path to SHA-256 hex digest.
        dependencies : list of str or None, optional
            Names of packages this package depends on. Defaults to
            empty list.

        Raises
        ------
        AlreadyInstalledError
            If a package with this name already exists in the manifest.

        Examples
        --------
        >>> manifest.add(
        ...     name="csv",
        ...     version="3.11",
        ...     files=["csv.py"],
        ...     checksums={"csv.py": "e3b0c44298fc1c149afbf4c8996fb924..."},
        ...     dependencies=["os", "io"]
        ... )
        """
        if name in self._packages:
            existing_path = str(self.storage_dir / name)
            raise AlreadyInstalledError(
                package_name=name,
                installed_path=existing_path,
            )

        now = datetime.now(timezone.utc).isoformat()

        self._packages[name] = {
            "name": name,
            "version": version,
            "installed_at": now,
            "files": sorted(files),
            "checksums": checksums,
            "dependencies": sorted(dependencies or []),
        }

        self.save()
        logger.info(
            "Added '%s' to manifest (version=%s, %d files)",
            name,
            version,
            len(files),
        )

    def remove(self, name: str) -> Dict[str, Any]:
        """
        Remove a package from the manifest.

        Does **not** delete files from disk. The caller is responsible
        for cleaning up the actual files before or after calling this
        method.

        Parameters
        ----------
        name : str
            Package name to remove.

        Returns
        -------
        dict
            The removed entry for reference (caller may use the
            ``"files"`` list to clean up disk).

        Raises
        ------
        ModuleNotInstalledError
            If the package is not found in the manifest.

        Examples
        --------
        >>> entry = manifest.remove("json")
        >>> entry["files"]
        ['json/__init__.py', 'json/encoder.py']
        >>> # Now delete the actual files:
        >>> for f in entry["files"]:
        ...     (manifest.storage_dir / f).unlink(missing_ok=True)
        """
        if name not in self._packages:
            raise ModuleNotInstalledError(module_name=name)

        entry = self._packages.pop(name)
        self.save()
        logger.info("Removed '%s' from manifest", name)
        return entry

    def update_dependencies(
        self, name: str, dependencies: List[str]
    ) -> None:
        """
        Update the dependency list for an installed package.

        Parameters
        ----------
        name : str
            Package name.
        dependencies : list of str
            New list of dependency names.

        Raises
        ------
        ModuleNotInstalledError
            If the package is not in the manifest.

        Examples
        --------
        >>> manifest.update_dependencies("json", ["os", "sys", "codecs"])
        """
        if name not in self._packages:
            raise ModuleNotInstalledError(module_name=name)

        self._packages[name]["dependencies"] = sorted(dependencies)
        self.save()
        logger.debug("Updated dependencies for '%s'", name)

    # ------------------------------------------------------------------
    # File-related queries
    # ------------------------------------------------------------------

    def get_files(self, name: str) -> List[str]:
        """
        Return the list of tracked file paths for a package.

        Parameters
        ----------
        name : str
            Package name.

        Returns
        -------
        list of str
            Relative file paths. Empty list if package not found.

        Examples
        --------
        >>> manifest.get_files("json")
        ['json/__init__.py', 'json/encoder.py', 'json/decoder.py']
        """
        entry = self._packages.get(name)
        if entry is None:
            return []
        return entry.get("files", [])

    def get_dependencies(self, name: str) -> List[str]:
        """
        Return the dependency list for a package.

        Parameters
        ----------
        name : str
            Package name.

        Returns
        -------
        list of str
            Dependency package names. Empty list if package not found
            or has no dependencies.

        Examples
        --------
        >>> manifest.get_dependencies("email")
        ['base64', 'quopri', 'uu']
        """
        entry = self._packages.get(name)
        if entry is None:
            return []
        return entry.get("dependencies", [])

    def get_file_checksum(
        self, name: str, file_path: str
    ) -> Optional[str]:
        """
        Retrieve the stored checksum for a specific file of a package.

        Parameters
        ----------
        name : str
            Package name.
        file_path : str
            Relative file path as stored in the manifest.

        Returns
        -------
        str or None
            SHA-256 hex digest if found, ``None`` if the package or
            file is not tracked.

        Examples
        --------
        >>> manifest.get_file_checksum("json", "json/__init__.py")
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
        """
        entry = self._packages.get(name)
        if entry is None:
            return None
        return entry.get("checksums", {}).get(file_path)

    def find_package_by_file(self, file_path: Path) -> Optional[str]:
        """
        Find which package owns a given file.

        Parameters
        ----------
        file_path : Path
            Absolute or relative path to a file.

        Returns
        -------
        str or None
            Package name if the file belongs to a registered package,
            ``None`` otherwise.

        Examples
        --------
        >>> name = manifest.find_package_by_file(
        ...     Path("/home/user/.stdlib-packages/json/encoder.py")
        ... )
        >>> name
        'json'
        """
        try:
            relative = str(file_path.relative_to(self.storage_dir))
        except ValueError:
            return None

        for name, entry in self._packages.items():
            if relative in entry.get("files", []):
                return name
        return None

    # ------------------------------------------------------------------
    # Integrity verification
    # ------------------------------------------------------------------

    def verify_integrity(self, name: str) -> Dict[str, bool]:
        """
        Verify checksums of all tracked files for a package.

        Compares stored checksums against actual file contents on disk.
        Files that are missing or have mismatched checksums are reported.

        Parameters
        ----------
        name : str
            Package name to verify.

        Returns
        -------
        dict
            Mapping of file path to boolean indicating whether the
            checksum matched. Missing files are reported as ``False``.

        Raises
        ------
        ModuleNotInstalledError
            If the package is not in the manifest.

        Examples
        --------
        >>> results = manifest.verify_integrity("json")
        >>> for path, ok in results.items():
        ...     if not ok:
        ...         print(f"Corrupted: {path}")
        """
        if name not in self._packages:
            raise ModuleNotInstalledError(module_name=name)

        entry = self._packages[name]
        stored_checksums = entry.get("checksums", {})
        results: Dict[str, bool] = {}

        for rel_path in entry.get("files", []):
            full_path = self.storage_dir / rel_path
            expected = stored_checksums.get(rel_path)

            if expected is None:
                results[rel_path] = False
                continue

            if not full_path.exists():
                results[rel_path] = False
                continue

            try:
                hasher = hashlib.sha256()
                with open(full_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
                actual = hasher.hexdigest()
                results[rel_path] = actual == expected
            except OSError:
                results[rel_path] = False

        return results

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def clear(self) -> int:
        """
        Remove all entries from the manifest.

        Returns
        -------
        int
            Number of entries removed.

        Examples
        --------
        >>> manifest.clear()
        5
        """
        count = len(self._packages)
        self._packages.clear()
        self.save()
        logger.info(
            "Cleared manifest (%d entries removed)", count
        )
        return count

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Return the full manifest data as a dictionary.

        Returns
        -------
        dict
            Full manifest contents keyed by package name.

        Examples
        --------
        >>> data = manifest.to_dict()
        >>> len(data)
        3
        """
        return dict(self._packages)

    # ------------------------------------------------------------------
    # Magic methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """
        Return the number of packages in the manifest.

        Returns
        -------
        int
            Package count.

        Examples
        --------
        >>> len(manifest)
        3
        """
        return len(self._packages)

    def __contains__(self, name: str) -> bool:
        """
        Support ``in`` operator for package name membership check.

        Parameters
        ----------
        name : str
            Package name.

        Returns
        -------
        bool
            ``True`` if the package is in the manifest.

        Examples
        --------
        >>> "json" in manifest
        True
        """
        return self.has(name)

    def __iter__(self):
        """
        Iterate over package names in the manifest.

        Yields
        ------
        str
            Package names in alphabetical order.

        Examples
        --------
        >>> for name in manifest:
        ...     print(name)
        csv
        json
        xml
        """
        return iter(self.list_names())

    def __repr__(self) -> str:
        """
        Return a concise string representation.

        Returns
        -------
        str
        """
        return (
            f"Manifest({len(self._packages)} packages at "
            f"{self.storage_dir})"
        )
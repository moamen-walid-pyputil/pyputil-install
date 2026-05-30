"""
Compiler discovery result caching.

Provides filesystem-based caching for compiler discovery results
to avoid expensive system scans on every application startup.

Scope
-----
- Save validated compiler lists to disk
- Load cached results with staleness checks
- Invalidate cache when system state changes

This module does NOT:
    - Discover compilers (use strategies.py / discovery.py)
    - Validate compilers (use validation.py)
    - Execute any binaries

Cache Location
--------------
Default: ~/.cache/toolforge/compilers.json (Linux/macOS)
         %LOCALAPPDATA%/toolforge/compilers.json (Windows)

Override with TOOLFORGE_CACHE_DIR environment variable.

Cache Structure
---------------
    {
        "version": 1,
        "created": "2026-05-23T10:30:00",
        "path_hash": "abc123...",
        "compilers": [
            {
                "path": "/usr/bin/gcc",
                "version": "13.2.0",
                "vendor": "GNU",
                "kind": "GCC",
                "target_triplet": "x86_64-linux-gnu",
                "is_cross_compiler": false,
                "source": "SYSTEM_PATH",
                "confidence_score": 0.98,
                "fingerprint": "def456..."
            }
        ]
    }

Staleness Detection
-------------------
Cache is considered stale when:
    1. PATH environment variable hash differs from cached value
    2. Cache file age exceeds max_age_seconds (default: 1 hour)
    3. A monitored directory's modification time changed
    4. Cache version number differs (schema migration)

Usage
-----
    from .cache import CompilerCache

    cache = CompilerCache()
    compilers = cache.load()  # Returns None if stale or missing
    cache.save(compilers)     # Persists results to disk

Warnings
--------
- Cache is NOT encrypted. Paths and versions are stored in plain JSON.
- Cache file permissions are set to 0o600 (owner read/write only).
- Stale cache detection is heuristic, not exhaustive. New compilers
  installed in previously-empty directories already in PATH will NOT
  be detected by staleness checks. Use rescan() to force refresh.

User Instructions
-----------------
- Set TOOLFORGE_CACHE_DIR to change cache location.
- Set TOOLFORGE_CACHE_MAX_AGE to change staleness threshold (seconds).
- Set TOOLFORGE_CACHE_MONITOR_DIRS to add directories to watch for changes
  (colon-separated, like PATH).
- Delete the cache file manually to force fresh discovery.
"""

import hashlib
import json
import logging
import os
import platform
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .models import CompilerInfo, CompilerKind, CompilerSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

CACHE_VERSION = 1

DEFAULT_CACHE_DIR = os.environ.get(
    "TOOLFORGE_CACHE_DIR",
    os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "toolforge",
    )
    if platform.system() != "Windows"
    else os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~/AppData/Local")),
        "toolforge",
    ),
)

DEFAULT_MAX_AGE_SECONDS = int(
    os.environ.get("TOOLFORGE_CACHE_MAX_AGE", "3600")  # 1 hour
)

CACHE_FILENAME = "compilers.json"


# ---------------------------------------------------------------------------
# CompilerCache
# ---------------------------------------------------------------------------

class CompilerCache:
    """
    Filesystem cache for compiler discovery results.

    Handles serialization, deserialization, and staleness checks.
    All I/O errors are caught and logged — cache failures are never
    fatal to the calling code.

    Parameters
    ----------
    cache_dir : Optional[str]
        Directory for cache files. Defaults to DEFAULT_CACHE_DIR.
    max_age_seconds : Optional[int]
        Maximum cache age before considered stale.
        Defaults to DEFAULT_MAX_AGE_SECONDS.

    Attributes
    ----------
    cache_dir : str
        Resolved cache directory path.
    cache_path : str
        Full path to the cache JSON file.
    max_age_seconds : int
        Maximum age in seconds.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_age_seconds: Optional[int] = None,
    ) -> None:
        self.cache_dir = os.path.abspath(
            cache_dir or DEFAULT_CACHE_DIR
        )
        self.cache_path = os.path.join(self.cache_dir, CACHE_FILENAME)
        self.max_age_seconds = (
            max_age_seconds
            if max_age_seconds is not None
            else DEFAULT_MAX_AGE_SECONDS
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def load(self) -> Optional[List[CompilerInfo]]:
        """
        Load compilers from cache if valid and not stale.

        Returns None if:
            - Cache file does not exist
            - Cache file is unreadable
            - Cache version mismatch
            - Cache is older than max_age_seconds
            - PATH hash has changed since cache was written
            - Monitored directories have changed
            - JSON parsing fails
            - Any CompilerInfo field fails to deserialize

        Returns
        -------
        Optional[List[CompilerInfo]]
            Cached compiler list, or None if cache is invalid/stale.
        """
        if not os.path.isfile(self.cache_path):
            logger.debug("Cache file not found: %s", self.cache_path)
            return None

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, PermissionError, OSError) as exc:
            logger.warning("Failed to read cache: %s", exc)
            return None

        # Version check
        if data.get("version") != CACHE_VERSION:
            logger.debug("Cache version mismatch, invalidating")
            return None

        # Age check
        created_str = data.get("created", "")
        if self._is_expired(created_str):
            logger.debug("Cache expired (age > %s seconds)", self.max_age_seconds)
            return None

        # PATH hash check
        cached_path_hash = data.get("path_hash", "")
        current_path_hash = self._hash_current_path()
        if cached_path_hash != current_path_hash:
            logger.debug("Cache stale: PATH hash changed")
            return None

        # Monitored directories check
        cached_dir_hashes = data.get("dir_hashes", {})
        if self._dirs_changed(cached_dir_hashes):
            logger.debug("Cache stale: monitored directory changed")
            return None

        # Deserialize compilers
        try:
            compilers = self._deserialize_compilers(data.get("compilers", []))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to deserialize cache: %s", exc)
            return None

        logger.info(
            "Cache loaded: %d compilers from %s",
            len(compilers),
            created_str,
        )
        return compilers

    def save(self, compilers: List[CompilerInfo]) -> bool:
        """
        Save compiler list to cache.

        Creates the cache directory if it does not exist.
        Sets file permissions to 0o600 (owner read/write only).

        Parameters
        ----------
        compilers : List[CompilerInfo]
            Validated compiler list to cache.

        Returns
        -------
        bool
            True if saved successfully, False on any I/O error.

        Warnings
        --------
        - Existing cache is overwritten atomically (write to temp, then rename).
        - If the list is empty, the cache file is still written.
          This is intentional — it prevents repeated scans when
          no compilers exist.
        """
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError as exc:
            logger.warning("Failed to create cache directory: %s", exc)
            return False

        data = {
            "version": CACHE_VERSION,
            "created": datetime.now().isoformat(),
            "path_hash": self._hash_current_path(),
            "dir_hashes": self._hash_monitored_dirs(),
            "compilers": self._serialize_compilers(compilers),
        }

        # Write to temp file first, then rename (atomic on same filesystem)
        temp_path = self.cache_path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)

            # Set restrictive permissions
            os.chmod(temp_path, 0o600)

            # Atomic rename
            os.replace(temp_path, self.cache_path)
        except OSError as exc:
            logger.warning("Failed to write cache: %s", exc)
            # Clean up temp file
            try:
                os.remove(temp_path)
            except OSError:
                pass
            return False

        logger.info(
            "Cache saved: %d compilers to %s",
            len(compilers),
            self.cache_path,
        )
        return True

    def clear(self) -> bool:
        """
        Delete the cache file.

        Returns
        -------
        bool
            True if cache was deleted or did not exist.
            False if deletion failed.
        """
        try:
            if os.path.isfile(self.cache_path):
                os.remove(self.cache_path)
                logger.debug("Cache cleared: %s", self.cache_path)
        except OSError as exc:
            logger.warning("Failed to clear cache: %s", exc)
            return False
        return True

    def is_valid(self) -> bool:
        """
        Check if cache exists and is not stale.

        Returns
        -------
        bool
            True if cache can be loaded successfully.
        """
        return self.load() is not None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_compilers(compilers: List[CompilerInfo]) -> List[Dict]:
        """
        Convert CompilerInfo list to JSON-serializable dicts.

        Parameters
        ----------
        compilers : List[CompilerInfo]
            Compilers to serialize.

        Returns
        -------
        List[Dict]
            List of dicts with all CompilerInfo fields.
        """
        return [
            {
                "path": c.path,
                "version": c.version,
                "vendor": c.vendor,
                "kind": c.kind.name,
                "target_triplet": c.target_triplet,
                "is_cross_compiler": c.is_cross_compiler,
                "source": c.source.name,
                "confidence_score": c.confidence_score,
                "fingerprint": c.fingerprint,
            }
            for c in compilers
        ]

    @staticmethod
    def _deserialize_compilers(raw: List[Dict]) -> List[CompilerInfo]:
        """
        Convert JSON dicts back to CompilerInfo objects.

        Parameters
        ----------
        raw : List[Dict]
            Raw dicts from JSON.

        Returns
        -------
        List[CompilerInfo]
            Deserialized compiler objects.

        Raises
        ------
        KeyError
            If a required field is missing.
        ValueError
            If an enum field has an invalid value.
        """
        compilers: List[CompilerInfo] = []
        for item in raw:
            compilers.append(CompilerInfo(
                path=item["path"],
                version=item["version"],
                vendor=item["vendor"],
                kind=CompilerKind[item["kind"]],
                target_triplet=item["target_triplet"],
                is_cross_compiler=item["is_cross_compiler"],
                source=CompilerSource[item["source"]],
                confidence_score=item["confidence_score"],
                fingerprint=item.get("fingerprint"),
            ))
        return compilers

    # ------------------------------------------------------------------
    # Staleness checks
    # ------------------------------------------------------------------

    def _is_expired(self, created_str: str) -> bool:
        """
        Check if cache is older than max_age_seconds.

        Parameters
        ----------
        created_str : str
            ISO-format datetime string from cache.

        Returns
        -------
        bool
            True if cache age exceeds max_age_seconds.
        """
        if not created_str:
            return True

        try:
            created = datetime.fromisoformat(created_str)
            age = datetime.now() - created
            return age > timedelta(seconds=self.max_age_seconds)
        except (ValueError, TypeError):
            return True

    @staticmethod
    def _hash_current_path() -> str:
        """
        Hash the current PATH environment variable.

        Returns
        -------
        str
            SHA256 hex digest of PATH contents.
        """
        path_val = os.environ.get("PATH", "")
        return hashlib.sha256(path_val.encode()).hexdigest()

    def _dirs_changed(self, cached_hashes: Dict[str, str]) -> bool:
        """
        Check if any monitored directory has changed.

        Compares modification times of directories in PATH and
        TOOLFORGE_CACHE_MONITOR_DIRS against cached hashes.

        Parameters
        ----------
        cached_hashes : Dict[str, str]
            Dir path → hash from cache.

        Returns
        -------
        bool
            True if any monitored directory changed.
        """
        current_hashes = self._hash_monitored_dirs()

        # Check for new directories not in cache
        for path in current_hashes:
            if path not in cached_hashes:
                return True

        # Check for removed directories
        for path in cached_hashes:
            if path not in current_hashes:
                return True

        # Check for changed hashes
        for path, hash_val in cached_hashes.items():
            if current_hashes.get(path) != hash_val:
                return True

        return False

    @staticmethod
    def _hash_monitored_dirs() -> Dict[str, str]:
        """
        Build hash map of monitored directories.

        Monitors:
            - All directories in PATH
            - Directories from _CACHE_MONITOR_DIRS env var

        Hash is based on directory modification time and entry count.
        This is NOT cryptographically secure — it only detects changes.

        Returns
        -------
        Dict[str, str]
            Directory path → hash string.
        """
        dirs_to_monitor: List[str] = []

        # PATH directories
        path_env = os.environ.get("PATH", "")
        for d in path_env.split(os.pathsep):
            d = d.strip()
            if d and os.path.isdir(d):
                dirs_to_monitor.append(d)

        # Extra directories from env var
        extra = os.environ.get("TOOLFORGE_CACHE_MONITOR_DIRS", "")
        separator = ";" if platform.system() == "Windows" else ":"
        for d in extra.split(separator):
            d = d.strip()
            if d and os.path.isdir(d):
                dirs_to_monitor.append(d)

        hashes: Dict[str, str] = {}
        for d in dirs_to_monitor:
            try:
                stat = os.stat(d)
                entry_count = len(os.listdir(d))
                raw = f"{stat.st_mtime}:{entry_count}:{stat.st_ino if hasattr(stat, 'st_ino') else 0}"
                hashes[d] = hashlib.sha256(raw.encode()).hexdigest()
            except (PermissionError, OSError):
                continue

        return hashes
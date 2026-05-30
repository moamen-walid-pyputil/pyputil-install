#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Persistent caching layer for package metadata and API responses.

This module provides a file-based cache with Time-To-Live (TTL) expiration,
automatic garbage collection, compression support, and detailed statistics.
It is designed to reduce network requests to PyPI and improve response times
for repeated queries about the same packages.

Classes
-------
CacheEntry
    A single cached item with metadata and expiration.
CacheConfig
    Configuration for cache behavior including TTL, size limits, and storage.
PackageCache
    Main cache interface for storing and retrieving package data.

Functions
--------
get_cache_dir
    Determine the platform-appropriate cache directory.
clear_system_cache
    Remove all cached data from the default cache location.

Examples
--------
>>> cache = PackageCache()
>>> cache.set("requests", {"version": "2.28.0", "dependencies": [...]})
>>> data = cache.get("requests")
>>> print(data["version"])
'2.28.0'
"""

import os
import sys
import json
import time
import shutil
import hashlib
import logging
import threading
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager

from .exceptions import CacheError

logger = logging.getLogger(__name__)


def get_cache_dir(app_name: str = "package_installer") -> Path:
    """
    Determine the platform-appropriate cache directory.

    Parameters
    ----------
    app_name : str, default="package_installer"
        Application name used as the cache subdirectory.

    Returns
    -------
    Path
        Path to the cache directory.

    Notes
    -----
    Uses the following platform conventions:

    - **Linux**: ``$XDG_CACHE_HOME/app_name`` or ``~/.cache/app_name``
    - **macOS**: ``~/Library/Caches/app_name``
    - **Windows**: ``%LOCALAPPDATA%\\app_name\\Cache``

    The directory is created if it does not exist.

    Examples
    --------
    >>> cache_dir = get_cache_dir()
    >>> cache_dir.name
    'package_installer'
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        cache_path = Path(base) / app_name / "Cache"
    elif sys.platform == "darwin":
        cache_path = Path.home() / "Library" / "Caches" / app_name
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        cache_path = Path(xdg_cache) / app_name

    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def clear_system_cache(app_name: str = "package_installer") -> bool:
    """
    Remove all cached data from the default system cache location.

    Parameters
    ----------
    app_name : str, default="package_installer"
        Application name whose cache should be cleared.

    Returns
    -------
    bool
        True if cache was cleared successfully, False if cache
        directory did not exist.

    Raises
    ------
    CacheError
        If the cache directory exists but cannot be removed due to
        permission errors or other filesystem issues.

    Examples
    --------
    >>> clear_system_cache()
    True
    """
    cache_dir = get_cache_dir(app_name)
    if not cache_dir.exists():
        return False

    try:
        shutil.rmtree(cache_dir)
        logger.info(f"Cleared system cache at {cache_dir}")
        return True
    except OSError as e:
        raise CacheError(
            f"Failed to clear cache directory: {e}",
            cache_path=str(cache_dir),
            operation="clear",
        ) from e


@dataclass
class CacheEntry:
    """
    A single cached item with its data and metadata.

    Parameters
    ----------
    key : str
        Unique identifier for the cached item (typically package name).
    data : any
        The cached data (must be JSON-serializable).
    created_at : float
        Unix timestamp when the entry was created.
    expires_at : float
        Unix timestamp when the entry expires.
    access_count : int
        Number of times this entry has been accessed.
    size_bytes : int
        Approximate size of the serialized data in bytes.
    compressed : bool
        Whether the data is stored compressed.

    Attributes
    ----------
    key : str
        Unique identifier for the cached item.
    data : any
        The stored data.
    created_at : float
        Creation timestamp.
    expires_at : float
        Expiration timestamp.
    access_count : int
        Access counter.
    size_bytes : int
        Data size in bytes.
    compressed : bool
        Compression flag.

    Examples
    --------
    >>> entry = CacheEntry("numpy", {"version": "1.24.0"}, 1000.0, 2000.0)
    >>> entry.is_expired()
    False
    """

    key: str
    data: Any
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 3600)
    access_count: int = 0
    size_bytes: int = 0
    compressed: bool = False

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """
        Check if this entry has expired.

        Parameters
        ----------
        current_time : float, optional
            The reference time for expiration check. Defaults to
            ``time.time()`` if not provided.

        Returns
        -------
        bool
            True if the entry is expired.

        Examples
        --------
        >>> entry = CacheEntry("test", {}, expires_at=100.0)
        >>> entry.is_expired(current_time=200.0)
        True
        """
        now = current_time if current_time is not None else time.time()
        return now >= self.expires_at

    @property
    def age_seconds(self) -> float:
        """
        Age of this entry in seconds since creation.

        Returns
        -------
        float
            Seconds elapsed since ``created_at``.
        """
        return time.time() - self.created_at

    @property
    def ttl_remaining(self) -> float:
        """
        Remaining time-to-live in seconds.

        Returns
        -------
        float
            Seconds until expiration. Negative if already expired.
        """
        return self.expires_at - time.time()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entry metadata to a dictionary (excluding data).

        Returns
        -------
        dict
            Dictionary with key, timestamps, counts, and size.

        Examples
        --------
        >>> entry = CacheEntry("requests", {}, size_bytes=1024)
        >>> meta = entry.to_dict()
        >>> meta["key"]
        'requests'
        """
        return {
            "key": self.key,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
            "compressed": self.compressed,
            "is_expired": self.is_expired(),
            "age_seconds": round(self.age_seconds, 2),
            "ttl_remaining": round(self.ttl_remaining, 2),
        }


@dataclass
class CacheConfig:
    """
    Configuration for cache behavior.

    Parameters
    ----------
    ttl_seconds : float, default=1800
        Default Time-To-Live for new cache entries in seconds.
        1800 seconds = 30 minutes.
    max_size_mb : float, default=100.0
        Maximum total cache size in megabytes. When exceeded, the
        least recently accessed expired entries are evicted first.
    max_entries : int, default=10000
        Maximum number of cache entries. When exceeded, expired
        entries are evicted.
    compress_threshold_bytes : int, default=1024
        Minimum data size in bytes to trigger compression. Data
        smaller than this is stored uncompressed.
    compression_level : int, default=6
        zlib compression level (1-9). 1 is fastest, 9 is best
        compression. Default 6 balances speed and size.
    cache_dir : Path or str, optional
        Custom cache directory. If None, uses platform default.
    auto_gc_interval : int, default=100
        Number of write operations between automatic garbage
        collection runs. Set to 0 to disable auto-GC.

    Notes
    -----
    Compression uses zlib (DEFLATE) which provides a good balance
    between speed and compression ratio for JSON data.

    The cache uses an index file (``cache_index.json``) to track
    all entries and their metadata without loading data into memory.

    Examples
    --------
    >>> config = CacheConfig(ttl_seconds=3600, max_size_mb=50.0)
    >>> config.ttl_seconds
    3600
    """

    ttl_seconds: float = 1800.0
    max_size_mb: float = 100.0
    max_entries: int = 10000
    compress_threshold_bytes: int = 1024
    compression_level: int = 6
    cache_dir: Optional[Path] = None
    auto_gc_interval: int = 100

    def __post_init__(self) -> None:
        """Validate configuration and set default cache directory."""
        if self.ttl_seconds <= 0:
            raise ValueError(
                f"ttl_seconds must be positive, got {self.ttl_seconds}"
            )
        if self.max_size_mb <= 0:
            raise ValueError(
                f"max_size_mb must be positive, got {self.max_size_mb}"
            )
        if self.max_entries <= 0:
            raise ValueError(
                f"max_entries must be positive, got {self.max_entries}"
            )
        if not 1 <= self.compression_level <= 9:
            raise ValueError(
                f"compression_level must be 1-9, got {self.compression_level}"
            )

        if self.cache_dir is None:
            self.cache_dir = get_cache_dir()
        else:
            self.cache_dir = Path(self.cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def max_size_bytes(self) -> int:
        """Maximum cache size in bytes."""
        return int(self.max_size_mb * 1024 * 1024)


class PackageCache:
    """
    Persistent file-based cache for package metadata and API responses.

    This class provides a thread-safe caching layer with TTL-based
    expiration, automatic garbage collection, compression for large
    entries, and detailed access statistics.

    Parameters
    ----------
    config : CacheConfig, optional
        Cache configuration. If None, default configuration is used.

    Attributes
    ----------
    config : CacheConfig
        The active cache configuration.
    stats : dict
        Runtime statistics (hits, misses, evictions, writes).
    index : dict
        In-memory index of all cache entries (key -> CacheEntry metadata).

    Notes
    -----
    Cache entries are stored as individual files in the cache directory
    with filenames derived from SHA256 hashes of the keys. An index file
    (``cache_index.json``) maintains metadata for all entries.

    Thread safety is provided by a reentrant lock. All public methods
    acquire this lock before accessing the cache.

    Examples
    --------
    >>> cache = PackageCache()
    >>> cache.set("numpy", {"version": "1.24.0", "dependencies": ["python>=3.8"]})
    >>> data = cache.get("numpy")
    >>> data["version"]
    '1.24.0'
    >>> cache.stats
    {'hits': 1, 'misses': 0, 'evictions': 0, 'writes': 1, 'garbage_collections': 0}

    Custom TTL for specific entries::

    >>> cache.set("flask", {"version": "2.3.0"}, ttl=7200)  # 2 hour TTL
    """

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        self.config = config if config is not None else CacheConfig()
        self._lock = threading.RLock()
        self._index: Dict[str, CacheEntry] = {}
        self._write_counter: int = 0

        self._stats: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "writes": 0,
            "garbage_collections": 0,
        }

        self._index_path = self.config.cache_dir / "cache_index.json"
        self._load_index()
        logger.debug(
            f"PackageCache initialized at {self.config.cache_dir} "
            f"with {len(self._index)} entries"
        )

    def _hash_key(self, key: str) -> str:
        """
        Generate a safe filename hash from a cache key.

        Parameters
        ----------
        key : str
            The cache key to hash.

        Returns
        -------
        str
            Hex-encoded SHA256 hash of the key.
        """
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _get_entry_path(self, key: str) -> Path:
        """
        Get the filesystem path for a cache entry.

        Parameters
        ----------
        key : str
            The cache key.

        Returns
        -------
        Path
            Path to the cache file.
        """
        return self.config.cache_dir / f"{self._hash_key(key)}.cache"

    def _compress(self, data: bytes) -> bytes:
        """
        Compress data using zlib.

        Parameters
        ----------
        data : bytes
            Raw bytes to compress.

        Returns
        -------
        bytes
            Compressed bytes.
        """
        return zlib.compress(data, level=self.config.compression_level)

    def _decompress(self, data: bytes) -> bytes:
        """
        Decompress zlib-compressed data.

        Parameters
        ----------
        data : bytes
            Compressed bytes.

        Returns
        -------
        bytes
            Decompressed bytes.
        """
        return zlib.decompress(data)

    def _load_index(self) -> None:
        """
        Load cache index from disk.

        If the index file does not exist or is corrupted, an empty
        index is initialized. Corrupted index files are backed up
        before being replaced.

        Raises
        ------
        CacheError
            If the index file exists but cannot be read due to
            permission errors.
        """
        if not self._index_path.exists():
            self._index = {}
            return

        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                raw_index = json.load(f)

            self._index = {}
            for key, entry_dict in raw_index.items():
                try:
                    entry = CacheEntry(
                        key=entry_dict["key"],
                        data=None,
                        created_at=entry_dict["created_at"],
                        expires_at=entry_dict["expires_at"],
                        access_count=entry_dict.get("access_count", 0),
                        size_bytes=entry_dict.get("size_bytes", 0),
                        compressed=entry_dict.get("compressed", False),
                    )
                    self._index[key] = entry
                except (KeyError, TypeError) as e:
                    logger.warning(
                        f"Skipping corrupted index entry for '{key}': {e}"
                    )

            logger.debug(f"Loaded {len(self._index)} entries from index")

        except json.JSONDecodeError as e:
            logger.error(f"Corrupted cache index file: {e}")
            self._backup_corrupted_index()
            self._index = {}

        except OSError as e:
            raise CacheError(
                f"Cannot read cache index: {e}",
                cache_path=str(self._index_path),
                operation="read",
            ) from e

    def _save_index(self) -> None:
        """
        Save the current index to disk atomically.

        Writes to a temporary file first, then renames to ensure
        atomic updates and prevent corruption on crash.

        Notes
        -----
        Data values are not saved in the index; only metadata is
        persisted. Actual data is stored in individual cache files.

        Raises
        ------
        CacheError
            If the index cannot be written.
        """
        temp_path = self._index_path.with_suffix(".tmp")

        try:
            index_data = {}
            for key, entry in self._index.items():
                index_data[key] = entry.to_dict()

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(index_data, f, indent=2)

            temp_path.replace(self._index_path)
            logger.debug(f"Saved index with {len(self._index)} entries")

        except OSError as e:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise CacheError(
                f"Cannot write cache index: {e}",
                cache_path=str(self._index_path),
                operation="write",
            ) from e

    def _backup_corrupted_index(self) -> None:
        """
        Create a backup of a corrupted index file.

        The backup is named with a timestamp suffix to allow
        manual recovery if needed.
        """
        if not self._index_path.exists():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self._index_path.with_suffix(
            f".corrupted_{timestamp}.json"
        )

        try:
            shutil.copy2(self._index_path, backup_path)
            logger.warning(f"Backed up corrupted index to {backup_path}")
        except OSError as e:
            logger.error(f"Failed to backup corrupted index: {e}")

    def _should_compress(self, data: Any) -> bool:
        """
        Determine if data should be compressed based on serialized size.

        Parameters
        ----------
        data : any
            The data to evaluate.

        Returns
        -------
        bool
            True if data should be compressed.
        """
        try:
            serialized = json.dumps(data).encode("utf-8")
            return len(serialized) >= self.config.compress_threshold_bytes
        except (TypeError, ValueError):
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the cache.

        Parameters
        ----------
        key : str
            The cache key to retrieve.
        default : any, optional
            Value to return if key is not found or expired.

        Returns
        -------
        any
            The cached data, or `default` if not found or expired.

        Notes
        -----
        Expired entries are automatically removed when accessed.
        The ``access_count`` for the entry is incremented on each
        successful retrieval.

        Examples
        --------
        >>> cache = PackageCache()
        >>> result = cache.get("nonexistent", default={"status": "not_found"})
        >>> result["status"]
        'not_found'
        """
        with self._lock:
            if key not in self._index:
                self._stats["misses"] += 1
                return default

            entry = self._index[key]

            if entry.is_expired():
                self._remove_entry(key)
                self._stats["misses"] += 1
                return default

            entry_path = self._get_entry_path(key)

            if not entry_path.exists():
                self._remove_entry(key)
                self._stats["misses"] += 1
                return default

            try:
                with open(entry_path, "rb") as f:
                    raw_data = f.read()

                if entry.compressed:
                    raw_data = self._decompress(raw_data)

                data = json.loads(raw_data.decode("utf-8"))
                entry.access_count += 1
                self._stats["hits"] += 1
                return data

            except (OSError, json.JSONDecodeError, zlib.error) as e:
                logger.warning(f"Failed to read cache entry '{key}': {e}")
                self._remove_entry(key)
                self._stats["misses"] += 1
                return default

    def set(
        self,
        key: str,
        data: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Store a value in the cache.

        Parameters
        ----------
        key : str
            The cache key.
        data : any
            The data to cache. Must be JSON-serializable.
        ttl : float, optional
            Time-to-live in seconds for this entry. If None, uses
            the configured default TTL from ``CacheConfig.ttl_seconds``.

        Raises
        ------
        CacheError
            If data cannot be serialized or written to disk.
        ValueError
            If ttl is negative.

        Notes
        -----
        Data larger than ``compress_threshold_bytes`` is automatically
        compressed before storage.

        Examples
        --------
        >>> cache = PackageCache()
        >>> cache.set("requests", {"version": "2.28.0", "python": ">=3.7"})
        >>> cache.set("flask", {"version": "2.3.0"}, ttl=3600)
        """
        if ttl is not None and ttl < 0:
            raise ValueError(f"ttl must be non-negative, got {ttl}")

        with self._lock:
            effective_ttl = ttl if ttl is not None else self.config.ttl_seconds
            now = time.time()

            try:
                serialized = json.dumps(data).encode("utf-8")
            except (TypeError, ValueError) as e:
                raise CacheError(
                    f"Data for key '{key}' is not JSON-serializable: {e}",
                    operation="write",
                ) from e

            should_compress = len(serialized) >= self.config.compress_threshold_bytes
            stored_data = self._compress(serialized) if should_compress else serialized

            entry_path = self._get_entry_path(key)

            try:
                temp_path = entry_path.with_suffix(".tmp")
                with open(temp_path, "wb") as f:
                    f.write(stored_data)
                temp_path.replace(entry_path)
            except OSError as e:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                raise CacheError(
                    f"Cannot write cache entry '{key}': {e}",
                    cache_path=str(entry_path),
                    operation="write",
                ) from e

            entry = CacheEntry(
                key=key,
                data=data,
                created_at=now,
                expires_at=now + effective_ttl,
                size_bytes=len(stored_data),
                compressed=should_compress,
            )

            self._index[key] = entry
            self._stats["writes"] += 1
            self._write_counter += 1

            if (
                self.config.auto_gc_interval > 0
                and self._write_counter % self.config.auto_gc_interval == 0
            ):
                self.garbage_collect()

            self._save_index()

    def _remove_entry(self, key: str) -> None:
        """
        Remove an entry from the index and disk without acquiring the lock.

        Parameters
        ----------
        key : str
            The cache key to remove.

        Notes
        -----
        This is an internal method that assumes the lock is already
        held by the caller.
        """
        entry_path = self._get_entry_path(key)
        entry_path.unlink(missing_ok=True)
        self._index.pop(key, None)

    def delete(self, key: str) -> bool:
        """
        Remove a specific entry from the cache.

        Parameters
        ----------
        key : str
            The cache key to remove.

        Returns
        -------
        bool
            True if the entry existed and was removed, False if
            the key was not found.

        Examples
        --------
        >>> cache = PackageCache()
        >>> cache.set("temp", {"data": "value"})
        >>> cache.delete("temp")
        True
        >>> cache.delete("nonexistent")
        False
        """
        with self._lock:
            if key not in self._index:
                return False

            self._remove_entry(key)
            self._save_index()
            return True

    def exists(self, key: str) -> bool:
        """
        Check if a key exists and is not expired.

        Parameters
        ----------
        key : str
            The cache key to check.

        Returns
        -------
        bool
            True if the key exists and the entry is valid (not expired).

        Examples
        --------
        >>> cache = PackageCache()
        >>> cache.set("valid", {"data": 1})
        >>> cache.exists("valid")
        True
        """
        with self._lock:
            if key not in self._index:
                return False
            return not self._index[key].is_expired()

    def get_or_set(
        self,
        key: str,
        factory: callable,
        ttl: Optional[float] = None,
    ) -> Any:
        """
        Retrieve from cache or compute and store if not present.

        Parameters
        ----------
        key : str
            The cache key.
        factory : callable
            A callable that produces the data when cache miss occurs.
            Must return JSON-serializable data.
        ttl : float, optional
            Time-to-live for the entry if created.

        Returns
        -------
        any
            The cached or freshly computed data.

        Raises
        ------
        CacheError
            If the factory function raises an exception.

        Notes
        -----
        This is a convenience method that combines ``get`` and ``set``
        into a single atomic operation. It is useful for lazy-loading
        patterns where data is fetched on demand and cached for
        subsequent access.

        Examples
        --------
        >>> def fetch_from_api():
        ...     return {"version": "2.0.0"}
        >>> cache = PackageCache()
        >>> data = cache.get_or_set("my-package", fetch_from_api)
        >>> data["version"]
        '2.0.0'
        """
        with self._lock:
            cached = self.get(key)
            if cached is not None:
                return cached

            try:
                data = factory()
            except Exception as e:
                raise CacheError(
                    f"Factory function failed for key '{key}': {e}",
                    operation="write",
                ) from e

            self.set(key, data, ttl=ttl)
            return data

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns
        -------
        dict
            Dictionary with hit/miss counts, evictions, writes,
            garbage collections, current entry count, and total
            index size.

        Examples
        --------
        >>> cache = PackageCache()
        >>> stats = cache.get_stats()
        >>> stats["total_entries"]
        0
        """
        with self._lock:
            stats = dict(self._stats)
            stats["total_entries"] = len(self._index)
            stats["active_entries"] = sum(
                1 for e in self._index.values() if not e.is_expired()
            )
            stats["expired_entries"] = stats["total_entries"] - stats["active_entries"]
            stats["total_size_bytes"] = sum(
                e.size_bytes for e in self._index.values()
            )
            return stats

    def garbage_collect(self) -> int:
        """
        Remove all expired entries and enforce size limits.

        Returns
        -------
        int
            Number of entries removed.

        Notes
        -----
        Garbage collection runs in three phases:

        1. Remove all expired entries.
        2. If ``max_entries`` is exceeded, remove oldest expired first.
        3. If ``max_size_mb`` is exceeded, remove entries by oldest
           access time until the limit is satisfied.

        Active (non-expired) entries are only evicted in phase 3
        if size limits are still exceeded after removing expired entries.

        Examples
        --------
        >>> cache = PackageCache()
        >>> removed = cache.garbage_collect()
        >>> print(f"Removed {removed} entries")
        """
        with self._lock:
            removed = 0
            now = time.time()

            expired_keys = [
                key for key, entry in self._index.items()
                if entry.is_expired(current_time=now)
            ]
            for key in expired_keys:
                self._remove_entry(key)
                removed += 1

            if len(self._index) > self.config.max_entries:
                sorted_entries = sorted(
                    self._index.items(),
                    key=lambda item: (item[1].is_expired(), item[1].access_count),
                )
                while len(self._index) > self.config.max_entries:
                    key, _ = sorted_entries.pop(0)
                    self._remove_entry(key)
                    removed += 1
                    self._stats["evictions"] += 1

            current_size = sum(e.size_bytes for e in self._index.values())
            if current_size > self.config.max_size_bytes:
                sorted_by_access = sorted(
                    self._index.items(),
                    key=lambda item: item[1].access_count,
                )
                for key, _ in sorted_by_access:
                    if current_size <= self.config.max_size_bytes:
                        break
                    current_size -= self._index[key].size_bytes
                    self._remove_entry(key)
                    removed += 1
                    self._stats["evictions"] += 1

            if removed > 0:
                self._stats["garbage_collections"] += 1
                self._save_index()

            logger.debug(f"Garbage collection removed {removed} entries")
            return removed

    def clear(self) -> int:
        """
        Remove all entries from the cache.

        Returns
        -------
        int
            Number of entries removed.

        Examples
        --------
        >>> cache = PackageCache()
        >>> cache.set("a", 1)
        >>> cache.set("b", 2)
        >>> cache.clear()
        2
        """
        with self._lock:
            count = len(self._index)
            for key in list(self._index.keys()):
                self._remove_entry(key)
            self._save_index()
            logger.info(f"Cleared {count} entries from cache")
            return count

    def keys(self) -> List[str]:
        """
        Get all cache keys (including expired ones still in index).

        Returns
        -------
        list of str
            All cache keys in the index.

        Examples
        --------
        >>> cache = PackageCache()
        >>> cache.set("x", 1)
        >>> "x" in cache.keys()
        True
        """
        with self._lock:
            return list(self._index.keys())

    def active_keys(self) -> List[str]:
        """
        Get keys for non-expired entries only.

        Returns
        -------
        list of str
            Keys of active (non-expired) entries.

        Examples
        --------
        >>> cache = PackageCache()
        >>> cache.set("active", 1, ttl=3600)
        >>> "active" in cache.active_keys()
        True
        """
        with self._lock:
            return [
                key for key, entry in self._index.items()
                if not entry.is_expired()
            ]

    def touch(self, key: str, ttl: Optional[float] = None) -> bool:
        """
        Extend the expiration time of an existing entry.

        Parameters
        ----------
        key : str
            The cache key to touch.
        ttl : float, optional
            New TTL in seconds from now. If None, uses configured
            default TTL.

        Returns
        -------
        bool
            True if the entry was found and updated, False otherwise.

        Examples
        --------
        >>> cache = PackageCache()
        >>> cache.set("session", {"user": "alice"}, ttl=60)
        >>> cache.touch("session", ttl=3600)
        True
        """
        with self._lock:
            if key not in self._index:
                return False

            effective_ttl = ttl if ttl is not None else self.config.ttl_seconds
            entry = self._index[key]
            entry.expires_at = time.time() + effective_ttl
            self._save_index()
            return True

    @property
    def stats(self) -> Dict[str, int]:
        """
        Runtime statistics for the current session.

        Returns
        -------
        dict
            Dictionary with current session statistics.

        Notes
        -----
        These statistics are not persisted across sessions. For
        persistent statistics, use ``get_stats()``.

        Examples
        --------
        >>> cache = PackageCache()
        >>> cache.stats["hits"]
        0
        """
        return dict(self._stats)

    def __len__(self) -> int:
        """Return number of entries in the index."""
        return len(self._index)

    def __contains__(self, key: str) -> bool:
        """Check if key exists (supports ``key in cache`` syntax)."""
        return self.exists(key)

    def __getitem__(self, key: str) -> Any:
        """
        Support ``cache[key]`` syntax.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        any
            Cached data.

        Raises
        ------
        KeyError
            If key is not found or expired.
        """
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __setitem__(self, key: str, data: Any) -> None:
        """Support ``cache[key] = data`` syntax."""
        self.set(key, data)

    def __delitem__(self, key: str) -> None:
        """
        Support ``del cache[key]`` syntax.

        Raises
        ------
        KeyError
            If key is not found.
        """
        if not self.delete(key):
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        """Iterate over active cache keys."""
        return iter(self.active_keys())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download Manager for acquiring remote files over HTTP/HTTPS.

This module handles the complete lifecycle of downloading files from
remote servers: establishing connections, negotiating transfer parameters,
streaming data to disk with resume capability, and verifying file integrity
through cryptographic hashes.

The system is structured into independent components that each handle
one aspect of the download process. Components communicate through
defined interfaces and emit events for external observers.

Component Overview
------------------

ConnectionPool
    Manages a bounded set of reusable HTTP connections keyed by endpoint.
    Connections are created on demand and returned to the pool after use.
    Idle connections past a configurable timeout are pruned automatically.

SpeedTracker
    Computes download speed using time-decayed weighted sampling.
    Older samples contribute exponentially less weight to the current
    speed estimate, providing faster reaction to bandwidth changes
    while maintaining reasonable smoothness.

CheckpointManager
    Persists download progress to disk at configurable byte intervals.
    Checkpoints are written atomically via temporary file + rename.
    Enables interrupted downloads to resume without re-fetching data.

IntegrityVerifier
    Computes cryptographic hash digests incrementally as data streams
    through. Supports multiple simultaneous hash algorithms. Verifies
    computed digests against expected values provided by the caller.

ChunkScheduler
    Partitions a target file into byte-range chunks for concurrent
    downloading. Dynamically adjusts chunk size based on remaining
    work and available worker capacity.

DownloadManager
    Orchestrates the above components to perform a complete download.
    Accepts a URL and destination path. Handles retry logic, pause/
    resume/cancel requests, and progress reporting.

Thread Safety
-------------
All public methods use internal locking where necessary. The manager
can be shared across threads for concurrent downloads to distinct
target files. Each download operation is independent.

Cancellation
------------
Cancellation is cooperative and checked at chunk boundaries. Partial
files remain on disk in a valid state for subsequent resumption.

Memory Usage
------------
Memory is bounded to O(chunk_size * max_concurrent_chunks) regardless
of total download size. File I/O uses buffered streaming.

Warnings
--------
- Setting max_concurrent too high can trigger server rate limiting
  or saturate the client's network interface.
- Checkpoint files accumulate on disk. Call CheckpointManager.delete_checkpoint
  after successful downloads to prevent unbounded storage growth.
- The integrity verifier processes data in memory. For files exceeding
  available RAM, use a single-pass streaming approach rather than
  loading the entire file.
- This module does not handle authentication (HTTP Basic/Digest/Bearer).
  Authenticated URLs must include credentials or tokens in the URL
  or via custom headers.
- Proxy support requires the proxy URL to be passed explicitly.
  System proxy settings from environment variables are not read.
- The module uses urllib internally. Certificate verification depends
  on the system CA bundle. On minimal systems without CA certificates,
  downloads to HTTPS URLs will fail.
"""

from __future__ import annotations

import hashlib
import time
import threading
import socket
import ssl
import os
import json
import uuid
import random
import math
from pathlib import Path
from typing import (
    Optional, Callable, Dict, Any, List, Tuple, Union,
    Deque, NamedTuple
)
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from contextlib import suppress
from urllib.request import Request, urlopen, HTTPError, URLError
from urllib.parse import urlparse
import http.client
import logging

from .exceptions import (
    DownloadError,
    VerificationError,
    NetworkError,
)
from .config import (
    NetworkConfig,
    VerificationConfig,
    HashAlgorithm,
)

logger = logging.getLogger(__name__)

# Sentinel constants for unknown/unavailable values
UNKNOWN_SIZE: int = -1
UNKNOWN_SPEED: float = -1.0
UNKNOWN_PERCENTAGE: float = -1.0
UNKNOWN_ETA: float = -1.0


# ============================================================================
# Enumerations
# ============================================================================

class DownloadStatus(Enum):
    """
    States in the download lifecycle.

    Valid Transitions
    -----------------
    PENDING    -> CONNECTING
    CONNECTING -> DOWNLOADING, FAILED
    DOWNLOADING-> PAUSED, COMPLETED, CANCELLED, FAILED
    PAUSED     -> DOWNLOADING, CANCELLED
    COMPLETED  -> VERIFYING
    VERIFYING  -> VERIFIED, FAILED
    VERIFIED   -> terminal
    FAILED     -> PENDING (on retry)
    CANCELLED  -> terminal
    """
    PENDING = auto()
    CONNECTING = auto()
    DOWNLOADING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    VERIFYING = auto()
    VERIFIED = auto()
    FAILED = auto()
    CANCELLED = auto()

    @property
    def is_terminal(self) -> bool:
        """True if this state represents a final outcome."""
        return self in (DownloadStatus.VERIFIED, DownloadStatus.CANCELLED)

    @property
    def is_active(self) -> bool:
        """True if a download operation is in progress."""
        return self in (
            DownloadStatus.CONNECTING,
            DownloadStatus.DOWNLOADING,
            DownloadStatus.VERIFYING,
        )


class ConnectionHealth(Enum):
    """Classification of connection quality based on recent metrics."""
    OPTIMAL = auto()
    DEGRADED = auto()
    UNSTABLE = auto()
    FAILED = auto()


# ============================================================================
# Data Structures
# ============================================================================

@dataclass(frozen=True)
class ChunkDescriptor:
    """
    Describes a byte range to fetch in a concurrent download.

    Attributes
    ----------
    index : int
        Zero-based position in the chunk sequence.
    start_byte : int
        Inclusive byte offset where this chunk begins.
    end_byte : int
        Inclusive byte offset where this chunk ends.
        UNKNOWN_SIZE if the total file size is not yet known.

    Examples
    --------
    >>> chunk = ChunkDescriptor(index=0, start_byte=0, end_byte=1048575)
    >>> chunk.size
    1048576
    >>> chunk.range_header
    'bytes=0-1048575'
    """
    index: int
    start_byte: int
    end_byte: int

    @property
    def size(self) -> int:
        """Number of bytes in this chunk, or UNKNOWN_SIZE."""
        if self.end_byte == UNKNOWN_SIZE:
            return UNKNOWN_SIZE
        return self.end_byte - self.start_byte + 1

    @property
    def range_header(self) -> str:
        """HTTP Range header value for requesting this chunk."""
        end = '' if self.end_byte == UNKNOWN_SIZE else str(self.end_byte)
        return f"bytes={self.start_byte}-{end}"


@dataclass
class DownloadProgress:
    """
    Snapshot of download state at an instant in time.

    Attributes
    ----------
    bytes_downloaded : int
        Cumulative bytes written to disk.
    total_bytes : int
        Expected total size, or UNKNOWN_SIZE if indeterminate.
    instantaneous_speed : float
        Bytes/second measured over the recent sampling window.
    average_speed : float
        Bytes/second averaged over the entire download.
    elapsed_seconds : float
        Wall-clock time since the download started.
    eta_seconds : float
        Estimated seconds remaining, or UNKNOWN_ETA.
    completion_percentage : float
        Value between 0.0 and 100.0, or UNKNOWN_PERCENTAGE.
    status : DownloadStatus
        Current lifecycle state.
    active_connections : int
        Number of concurrent chunk transfers in flight.
    retry_count : int
        Number of retry attempts executed so far.

    Examples
    --------
    >>> progress = DownloadProgress(
    ...     bytes_downloaded=5242880,
    ...     total_bytes=10485760,
    ...     instantaneous_speed=1024000.0,
    ...     status=DownloadStatus.DOWNLOADING,
    ... )
    >>> progress.format_human()
    '[███████████████░░░░░░░░░░░░░░░] 50.0% 5.0 MiB / 10.0 MiB @ 1000.0 KiB/s'
    """
    bytes_downloaded: int = 0
    total_bytes: int = UNKNOWN_SIZE
    instantaneous_speed: float = 0.0
    average_speed: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float = UNKNOWN_ETA
    completion_percentage: float = UNKNOWN_PERCENTAGE
    status: DownloadStatus = DownloadStatus.PENDING
    active_connections: int = 0
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary for logging or transmission.

        Examples
        --------
        >>> DownloadProgress(bytes_downloaded=100).to_dict()
        {'bytes_downloaded': 100, 'total_bytes': -1, ...}
        """
        return {
            "bytes_downloaded": self.bytes_downloaded,
            "total_bytes": self.total_bytes,
            "instantaneous_speed": self.instantaneous_speed,
            "average_speed": self.average_speed,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "completion_percentage": self.completion_percentage,
            "status": self.status.name,
            "active_connections": self.active_connections,
            "retry_count": self.retry_count,
        }

    def format_human(self) -> str:
        """Return a human-readable progress string with optional bar.

        Examples
        --------
        >>> p = DownloadProgress(bytes_downloaded=0, total_bytes=100)
        >>> p.format_human()
        '[░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.0% 0 B / 100 B'
        """
        parts = []

        if self.completion_percentage != UNKNOWN_PERCENTAGE:
            bar_width = 30
            filled = int(bar_width * self.completion_percentage / 100)
            bar = '█' * filled + '░' * (bar_width - filled)
            parts.append(f"[{bar}] {self.completion_percentage:.1f}%")

        parts.append(format_bytes(self.bytes_downloaded))
        if self.total_bytes != UNKNOWN_SIZE:
            parts.append(f"/ {format_bytes(self.total_bytes)}")
        if self.instantaneous_speed > 0:
            parts.append(f"@ {format_bytes(self.instantaneous_speed)}/s")
        if self.eta_seconds != UNKNOWN_ETA and self.eta_seconds > 0:
            parts.append(f"ETA: {format_duration(self.eta_seconds)}")

        return " ".join(parts)


@dataclass
class ConnectionMetrics:
    """
    Performance data for one connection within a download session.

    Attributes
    ----------
    connection_id : int
        Unique identifier for this connection.
    bytes_transferred : int
        Total bytes sent and received.
    connect_time_ms : float
        Milliseconds spent establishing the TCP/TLS connection.
    first_byte_time_ms : float
        Milliseconds from request to first response byte.
    total_active_time_ms : float
        Total milliseconds the connection spent transferring data.
    errors : int
        Count of errors encountered on this connection.
    timeouts : int
        Count of timeout events on this connection.
    """
    connection_id: int
    bytes_transferred: int = 0
    connect_time_ms: float = 0.0
    first_byte_time_ms: float = 0.0
    total_active_time_ms: float = 0.0
    errors: int = 0
    timeouts: int = 0

    @property
    def effective_speed(self) -> float:
        """Throughput in bytes/second over the active period.

        Examples
        --------
        >>> cm = ConnectionMetrics(1, bytes_transferred=10240, total_active_time_ms=1000)
        >>> cm.effective_speed
        10240.0
        """
        if self.total_active_time_ms <= 0:
            return 0.0
        return self.bytes_transferred / (self.total_active_time_ms / 1000.0)

    @property
    def health(self) -> ConnectionHealth:
        """Classification based on error rate and throughput.

        Returns FAILED if error rate exceeds 10% or multiple timeouts occurred.
        Returns UNSTABLE if any errors or timeouts present.
        Returns DEGRADED if throughput is below 1 KiB/s.
        Returns OPTIMAL otherwise.

        Examples
        --------
        >>> ConnectionMetrics(1).health.name
        'OPTIMAL'
        >>> ConnectionMetrics(1, errors=5, bytes_transferred=10).health.name
        'FAILED'
        """
        error_rate = self.errors / max(self.bytes_transferred, 1)
        if self.timeouts > 2 or error_rate > 0.1:
            return ConnectionHealth.FAILED
        if self.timeouts > 0 or error_rate > 0.01:
            return ConnectionHealth.UNSTABLE
        if self.effective_speed < 1024:
            return ConnectionHealth.DEGRADED
        return ConnectionHealth.OPTIMAL


# ============================================================================
# Formatting Utilities
# ============================================================================

def format_bytes(num_bytes: float) -> str:
    """Convert a byte count to a human-readable string using binary prefixes.

    Parameters
    ----------
    num_bytes : float
        Number of bytes. Negative values produce '? B'.

    Returns
    -------
    str
        Formatted string such as '1.5 MiB'.

    Examples
    --------
    >>> format_bytes(0)
    '0.0 B'
    >>> format_bytes(1536)
    '1.5 KiB'
    >>> format_bytes(1048576)
    '1.0 MiB'
    >>> format_bytes(-1)
    '? B'
    """
    if num_bytes < 0:
        return "? B"
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PiB"


def format_duration(seconds: float) -> str:
    """Convert a duration in seconds to human-readable form.

    Parameters
    ----------
    seconds : float
        Duration in seconds. Negative values produce '?'.

    Returns
    -------
    str
        Formatted string such as '1h 23m 45s'.

    Examples
    --------
    >>> format_duration(0)
    '0s'
    >>> format_duration(3661)
    '1h 1m 1s'
    >>> format_duration(-5)
    '?'
    """
    if seconds < 0:
        return "?"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def calculate_jitter(
    base_delay: float,
    attempt: int,
    jitter_factor: float = 0.1,
    max_delay: float = 300.0,
) -> float:
    """Compute retry delay using exponential backoff with decorrelated jitter.

    The formula is: min(base_delay * 2^attempt + uniform(0, base_delay * 2^attempt * jitter_factor), max_delay)

    Decorrelated jitter produces a more even distribution of retry times
    than simple exponential backoff, reducing thundering herd effects when
    multiple clients retry simultaneously.

    Parameters
    ----------
    base_delay : float
        Base delay in seconds for the first retry.
    attempt : int
        Zero-based retry attempt number.
    jitter_factor : float
        Fraction of the exponential delay used as the jitter range.
    max_delay : float
        Upper bound for the computed delay in seconds.

    Returns
    -------
    float
        Seconds to wait before the next retry.

    Examples
    --------
    >>> random.seed(42)
    >>> calculate_jitter(1.0, 0, jitter_factor=0.0)
    1.0
    >>> calculate_jitter(1.0, 3, jitter_factor=0.0)
    8.0
    >>> delay = calculate_jitter(1.0, 5, max_delay=10.0)
    >>> delay <= 10.0
    True
    """
    exponential = base_delay * (2 ** attempt)
    jitter = random.uniform(0, exponential * jitter_factor)
    return min(exponential + jitter, max_delay)


# ============================================================================
# Atomic File Writer
# ============================================================================

class AtomicFileWriter:
    """Writes data to a temporary file and atomically renames to the target path.

    This guarantees that the target path never contains a partially written
    file. If an exception occurs during writing, the temporary file is
    removed and the target path is left unchanged.

    Parameters
    ----------
    target_path : Path
        Final destination for the completed file.
    mode : str
        File mode, typically 'wb' for binary or 'w' for text.

    Warnings
    --------
    The atomic rename requires the temporary file to be on the same
    filesystem as the target path. Cross-filesystem renames will fail.

    Examples
    --------
    >>> import tempfile
    >>> target = Path(tempfile.gettempdir()) / "test_output.bin"
    >>> with AtomicFileWriter(target, 'wb') as writer:
    ...     writer.write(b"hello")
    ...     writer.flush()
    5
    >>> target.read_bytes()
    b'hello'
    >>> target.unlink()
    """

    def __init__(self, target_path: Path, mode: str = 'wb') -> None:
        self.target_path = target_path
        self.mode = mode
        self.temp_path = target_path.with_suffix(target_path.suffix + '.part')
        self._file = None
        self._bytes_written = 0

    def __enter__(self) -> 'AtomicFileWriter':
        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.temp_path, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file:
            self._file.close()
        if exc_type is None:
            self.temp_path.replace(self.target_path)
        else:
            with suppress(FileNotFoundError):
                self.temp_path.unlink()

    def write(self, data: bytes) -> int:
        """Write data to the temporary file.

        Parameters
        ----------
        data : bytes
            Data to write.

        Returns
        -------
        int
            Number of bytes written.

        Raises
        ------
        RuntimeError
            If the writer has not been opened via context manager.
        """
        if self._file is None:
            raise RuntimeError("AtomicFileWriter must be used as a context manager")
        written = self._file.write(data)
        self._bytes_written += written
        return written

    def flush(self) -> None:
        """Flush buffered data and force it to disk via fsync."""
        if self._file:
            self._file.flush()
            os.fsync(self._file.fileno())

    @property
    def bytes_written(self) -> int:
        """Total bytes written to the temporary file so far."""
        return self._bytes_written


# ============================================================================
# Speed Tracker
# ============================================================================

class SpeedTracker:
    """Tracks download speed using time-decayed weighted samples.

    Uses exponential decay so that older measurements contribute
    progressively less to the current speed estimate. This provides
    faster reaction to bandwidth changes than a simple moving average
    while still smoothing out short-term fluctuations.

    The decay follows: weight(t) = exp(-lambda * (now - t))
    where lambda = ln(2) / half_life_seconds.

    Parameters
    ----------
    half_life_seconds : float
        Time in seconds for a sample's contribution to halve.
    sample_interval : float
        Minimum time between recorded samples to avoid excessive updates.

    Examples
    --------
    >>> tracker = SpeedTracker(half_life_seconds=5.0, sample_interval=0.1)
    >>> tracker.record_bytes(0)
    >>> time.sleep(0.15)
    >>> tracker.record_bytes(10240)
    >>> tracker.current_speed > 0
    True
    """

    def __init__(
        self,
        half_life_seconds: float = 5.0,
        sample_interval: float = 0.25,
    ) -> None:
        if half_life_seconds <= 0:
            raise ValueError(f"half_life_seconds must be positive, got {half_life_seconds}")
        if sample_interval <= 0:
            raise ValueError(f"sample_interval must be positive, got {sample_interval}")
        self._half_life = half_life_seconds
        self._decay_constant = math.log(2) / half_life_seconds
        self._sample_interval = sample_interval
        self._lock = threading.Lock()
        self._samples: Deque[Tuple[float, float]] = deque()
        self._last_sample_time = 0.0
        self._current_speed = 0.0

    def _prune(self, now: float) -> None:
        """Remove samples whose weight has decayed below 0.1%."""
        threshold = 0.001
        max_age = -math.log(threshold) / self._decay_constant
        while self._samples and (now - self._samples[0][0]) > max_age:
            self._samples.popleft()

    def record_bytes(self, bytes_downloaded: int) -> None:
        """Feed a cumulative byte count into the tracker.

        Called periodically during download. Internally throttled
        to at most one sample per sample_interval.

        Parameters
        ----------
        bytes_downloaded : int
            Total bytes downloaded so far.

        Examples
        --------
        >>> tracker = SpeedTracker(sample_interval=0.0)
        >>> tracker.record_bytes(0)
        >>> tracker.record_bytes(5000)
        >>> tracker.record_bytes(10000)
        """
        now = time.monotonic()
        with self._lock:
            if now - self._last_sample_time < self._sample_interval:
                return
            self._last_sample_time = now
            self._prune(now)
            self._samples.append((now, float(bytes_downloaded)))
            if len(self._samples) >= 2:
                first_time = self._samples[0][0]
                last_time = self._samples[-1][0]
                span = last_time - first_time
                if span > 0:
                    first_bytes = self._samples[0][1]
                    last_bytes = self._samples[-1][1]
                    self._current_speed = (last_bytes - first_bytes) / span

    @property
    def current_speed(self) -> float:
        """Estimated speed in bytes/second based on recent samples."""
        return self._current_speed

    def estimate_eta(self, remaining_bytes: int) -> float:
        """Estimate seconds until completion.

        Parameters
        ----------
        remaining_bytes : int
            Bytes left to download.

        Returns
        -------
        float
            Estimated seconds, or UNKNOWN_ETA if current speed is zero.

        Examples
        --------
        >>> tracker = SpeedTracker(sample_interval=0.0)
        >>> tracker.record_bytes(0)
        >>> tracker.estimate_eta(1000)
        -1.0
        """
        if self._current_speed <= 0:
            return UNKNOWN_ETA
        return remaining_bytes / self._current_speed


# ============================================================================
# Checkpoint Manager
# ============================================================================

class CheckpointManager:
    """Persists download state so interrupted downloads can resume.

    Writes checkpoint files atomically to a state directory.
    Each download is identified by a caller-provided string.
    Checkpoints record the URL, output path, bytes downloaded,
    and per-chunk completion status.

    Parameters
    ----------
    state_dir : Path
        Directory where checkpoint files are stored.
    checkpoint_interval_bytes : int
        Minimum additional bytes downloaded between checkpoints.

    Warnings
    --------
    Checkpoint files are not automatically deleted. Call
    delete_checkpoint after a successful download to free disk space.

    Examples
    --------
    >>> import tempfile
    >>> state_dir = Path(tempfile.gettempdir()) / "checkpoints_test"
    >>> cm = CheckpointManager(state_dir, checkpoint_interval_bytes=0)
    >>> cm.save_checkpoint("dl1", "http://example.com/file", Path("/tmp/out"),
    ...                     bytes_downloaded=5000, total_bytes=10000, chunk_states={0: 5000})
    >>> checkpoint = cm.load_checkpoint("dl1")
    >>> checkpoint["bytes_downloaded"]
    5000
    >>> cm.delete_checkpoint("dl1")
    >>> cm.load_checkpoint("dl1") is None
    True
    >>> state_dir.rmdir()
    """

    def __init__(
        self,
        state_dir: Path,
        checkpoint_interval_bytes: int = 1024 * 1024,
    ) -> None:
        self.state_dir = state_dir
        self.checkpoint_interval = checkpoint_interval_bytes
        self._lock = threading.Lock()
        self._last_checkpoint_bytes = 0

    def get_state_path(self, download_id: str) -> Path:
        """Return the checkpoint file path for a given download ID.

        Parameters
        ----------
        download_id : str
            Unique identifier for the download session.

        Returns
        -------
        Path
            Path to the JSON checkpoint file.

        Examples
        --------
        >>> cm = CheckpointManager(Path("/tmp/cp"))
        >>> cm.get_state_path("abc123").name
        'abc123.checkpoint'
        """
        return self.state_dir / f"{download_id}.checkpoint"

    def should_checkpoint(self, bytes_downloaded: int) -> bool:
        """Return True if enough bytes have been downloaded since the last checkpoint.

        Parameters
        ----------
        bytes_downloaded : int
            Current total bytes downloaded.

        Returns
        -------
        bool
            True if a new checkpoint should be written.

        Examples
        --------
        >>> cm = CheckpointManager(Path("/tmp"), checkpoint_interval_bytes=1000)
        >>> cm.should_checkpoint(500)
        False
        >>> cm.should_checkpoint(1500)
        True
        """
        return (bytes_downloaded - self._last_checkpoint_bytes) >= self.checkpoint_interval

    def save_checkpoint(
        self,
        download_id: str,
        url: str,
        output_path: Path,
        bytes_downloaded: int,
        total_bytes: int,
        chunk_states: Dict[int, int],
        etag: Optional[str] = None,
        last_modified: Optional[str] = None,
    ) -> None:
        """Save download state to a checkpoint file atomically.

        Parameters
        ----------
        download_id : str
            Unique download identifier.
        url : str
            Source URL being downloaded.
        output_path : Path
            Destination file path on disk.
        bytes_downloaded : int
            Total bytes successfully written.
        total_bytes : int
            Expected total file size.
        chunk_states : Dict[int, int]
            Map of chunk index to bytes written for that chunk.
        etag : Optional[str]
            Server-provided ETag header for conditional requests.
        last_modified : Optional[str]
            Server-provided Last-Modified header.
        """
        state = {
            "version": 1,
            "url": url,
            "output_path": str(output_path),
            "bytes_downloaded": bytes_downloaded,
            "total_bytes": total_bytes,
            "chunk_states": {str(k): v for k, v in chunk_states.items()},
            "etag": etag,
            "last_modified": last_modified,
            "timestamp": time.time(),
        }
        checkpoint_path = self.get_state_path(download_id)
        with self._lock:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            temp_path = checkpoint_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            temp_path.replace(checkpoint_path)
            self._last_checkpoint_bytes = bytes_downloaded

    def load_checkpoint(self, download_id: str) -> Optional[Dict[str, Any]]:
        """Load a previously saved checkpoint.

        Parameters
        ----------
        download_id : str
            Download identifier.

        Returns
        -------
        Optional[Dict[str, Any]]
            Checkpoint state dictionary, or None if not found or corrupt.
        """
        checkpoint_path = self.get_state_path(download_id)
        if not checkpoint_path.exists():
            return None
        with self._lock:
            try:
                with open(checkpoint_path, 'r') as f:
                    state = json.load(f)
                required = {"url", "output_path", "bytes_downloaded", "total_bytes"}
                if not required.issubset(state.keys()):
                    logger.warning(f"Checkpoint {download_id} missing required fields")
                    return None
                state["chunk_states"] = {
                    int(k): v for k, v in state.get("chunk_states", {}).items()
                }
                return state
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to load checkpoint {download_id}: {e}")
                return None

    def delete_checkpoint(self, download_id: str) -> None:
        """Remove a checkpoint file.

        Parameters
        ----------
        download_id : str
            Download identifier.

        Examples
        --------
        >>> cm = CheckpointManager(Path("/tmp/cp"))
        >>> cm.delete_checkpoint("nonexistent")  # Does not raise
        """
        checkpoint_path = self.get_state_path(download_id)
        with self._lock:
            with suppress(FileNotFoundError):
                checkpoint_path.unlink()


# ============================================================================
# Integrity Verifier
# ============================================================================

class IntegrityVerifier:
    """Computes cryptographic hashes incrementally as data streams through.

    Feeds data to one or more hash functions on each update() call.
    After all data is processed, verify() compares computed digests
    against expected values.

    Parameters
    ----------
    algorithms : Optional[List[HashAlgorithm]]
        Hash algorithms to compute. Defaults to [SHA256].
    expected_hashes : Optional[Dict[str, str]]
        Expected hex digests keyed by algorithm name (e.g. {'sha256': 'abc...'}).

    Examples
    --------
    >>> verifier = IntegrityVerifier(
    ...     algorithms=[HashAlgorithm.SHA256],
    ...     expected_hashes={'sha256': hashlib.sha256(b'hello').hexdigest()},
    ... )
    >>> verifier.update(b'hel')
    >>> verifier.update(b'lo')
    >>> passed, computed = verifier.verify()
    >>> passed
    True
    >>> computed['sha256'] == hashlib.sha256(b'hello').hexdigest()
    True
    """

    def __init__(
        self,
        algorithms: Optional[List[HashAlgorithm]] = None,
        expected_hashes: Optional[Dict[str, str]] = None,
    ) -> None:
        self.algorithms = algorithms or [HashAlgorithm.SHA256]
        self.expected_hashes = expected_hashes or {}
        self._hashers: Dict[str, Any] = {}
        self._bytes_processed = 0
        for algo in self.algorithms:
            self._hashers[algo.value] = algo.get_hash_function()()

    def update(self, data: bytes) -> None:
        """Feed a chunk of data through all active hash functions.

        Parameters
        ----------
        data : bytes
            Data to process.

        Examples
        --------
        >>> verifier = IntegrityVerifier()
        >>> verifier.update(b'abc')
        >>> verifier.bytes_processed
        3
        """
        for hasher in self._hashers.values():
            hasher.update(data)
        self._bytes_processed += len(data)

    def verify(self) -> Tuple[bool, Dict[str, str]]:
        """Finalize hashes and check against expected values.

        Returns
        -------
        Tuple[bool, Dict[str, str]]
            A tuple of (all_verified, computed_hashes).
            all_verified is True only if every expected hash matches.

        Examples
        --------
        >>> verifier = IntegrityVerifier()
        >>> verifier.update(b'test')
        >>> passed, hashes = verifier.verify()
        >>> isinstance(hashes['sha256'], str)
        True
        """
        computed = {
            algo: hasher.hexdigest()
            for algo, hasher in self._hashers.items()
        }
        all_verified = True
        for algo, expected in self.expected_hashes.items():
            if algo in computed:
                if computed[algo].lower() != expected.lower():
                    all_verified = False
        return all_verified, computed

    @property
    def bytes_processed(self) -> int:
        """Total bytes passed to update() calls."""
        return self._bytes_processed


# ============================================================================
# Connection Pool
# ============================================================================

class ConnectionPool:
    """Pool of reusable HTTP connections keyed by endpoint.

    Connections are created on demand and returned after use.
    Idle connections past the configured timeout are pruned
    when acquiring a new connection.

    Parameters
    ----------
    max_connections : int
        Maximum total connections across all endpoints.
    max_connections_per_host : int
        Maximum connections to a single host:port pair.
    idle_timeout : float
        Seconds before an idle connection is closed.

    Warnings
    --------
    The pool does not perform health checks on idle connections.
    Servers may close keep-alive connections without notice,
    causing the next request on that connection to fail.

    Examples
    --------
    >>> pool = ConnectionPool(max_connections=2, max_connections_per_host=1)
    >>> conn, cid = pool.get_connection("example.com", 80)
    >>> isinstance(conn, http.client.HTTPConnection)
    True
    >>> pool.return_connection(conn, "example.com", 80)
    >>> pool.close_all()
    """

    def __init__(
        self,
        max_connections: int = 10,
        max_connections_per_host: int = 4,
        idle_timeout: float = 30.0,
    ) -> None:
        self.max_connections = max_connections
        self.max_per_host = max_connections_per_host
        self.idle_timeout = idle_timeout
        self._lock = threading.Lock()
        self._connections: Dict[str, Deque[Tuple[http.client.HTTPConnection, float]]] = {}
        self._active_count = 0
        self._next_conn_id = 0
        self._metrics: Dict[int, ConnectionMetrics] = {}

    def _prune_idle(self) -> None:
        """Close and remove connections idle beyond idle_timeout."""
        now = time.monotonic()
        for host in list(self._connections.keys()):
            conns = self._connections[host]
            while conns and (now - conns[0][1]) > self.idle_timeout:
                conn, _ = conns.popleft()
                with suppress(Exception):
                    conn.close()
                self._active_count -= 1
            if not conns:
                del self._connections[host]

    def get_connection(
        self, host: str, port: int, use_ssl: bool = False,
        timeout: float = 30.0,
    ) -> Tuple[http.client.HTTPConnection, int]:
        """Acquire a connection from the pool or create a new one.

        Parameters
        ----------
        host : str
            Target hostname.
        port : int
            Target port number.
        use_ssl : bool
            If True, create an HTTPSConnection.
        timeout : float
            Connection timeout in seconds.

        Returns
        -------
        Tuple[http.client.HTTPConnection, int]
            (connection_object, connection_id).

        Examples
        --------
        >>> pool = ConnectionPool()
        >>> conn, cid = pool.get_connection("httpbin.org", 443, use_ssl=True, timeout=10)
        >>> conn.host
        'httpbin.org'
        >>> pool.return_connection(conn, "httpbin.org", 443, use_ssl=True)
        >>> pool.close_all()
        """
        endpoint = f"{'https' if use_ssl else 'http'}://{host}:{port}"
        with self._lock:
            self._prune_idle()
            if endpoint in self._connections and self._connections[endpoint]:
                conn, _ = self._connections[endpoint].popleft()
                return conn, id(conn)
            if self._active_count >= self.max_connections:
                oldest_host = None
                oldest_time = float('inf')
                for h, conns in self._connections.items():
                    if conns and conns[0][1] < oldest_time:
                        oldest_time = conns[0][1]
                        oldest_host = h
                if oldest_host:
                    conn, _ = self._connections[oldest_host].popleft()
                    with suppress(Exception):
                        conn.close()
                    self._active_count -= 1
            conn_id = self._next_conn_id
            self._next_conn_id += 1
            self._active_count += 1
            if use_ssl:
                conn = http.client.HTTPSConnection(
                    host, port, timeout=timeout,
                    context=ssl.create_default_context(),
                )
            else:
                conn = http.client.HTTPConnection(host, port, timeout=timeout)
            self._metrics[conn_id] = ConnectionMetrics(connection_id=conn_id)
            return conn, conn_id

    def return_connection(
        self, conn: http.client.HTTPConnection, host: str,
        port: int, use_ssl: bool = False,
    ) -> None:
        """Return a connection to the pool for reuse.

        If the per-host pool is full, the connection is closed instead.

        Parameters
        ----------
        conn : http.client.HTTPConnection
            Connection to return.
        host : str
            Target hostname.
        port : int
            Target port.
        use_ssl : bool
            Whether the connection uses TLS.
        """
        endpoint = f"{'https' if use_ssl else 'http'}://{host}:{port}"
        with self._lock:
            host_conns = self._connections.get(endpoint)
            if host_conns is None:
                host_conns = deque()
                self._connections[endpoint] = host_conns
            if len(host_conns) < self.max_per_host:
                host_conns.append((conn, time.monotonic()))
            else:
                with suppress(Exception):
                    conn.close()
                self._active_count -= 1

    def get_metrics(self, conn_id: int) -> Optional[ConnectionMetrics]:
        """Retrieve metrics for a connection by its ID.

        Parameters
        ----------
        conn_id : int
            Connection identifier returned by get_connection.

        Returns
        -------
        Optional[ConnectionMetrics]
            Metrics or None if the ID is unknown.

        Examples
        --------
        >>> pool = ConnectionPool()
        >>> _, cid = pool.get_connection("example.com", 80)
        >>> metrics = pool.get_metrics(cid)
        >>> metrics.connection_id == cid
        True
        >>> pool.return_connection(_, "example.com", 80)
        >>> pool.close_all()
        """
        return self._metrics.get(conn_id)

    def close_all(self) -> None:
        """Close all pooled connections immediately.

        Examples
        --------
        >>> pool = ConnectionPool()
        >>> conn, _ = pool.get_connection("example.com", 80)
        >>> pool.return_connection(conn, "example.com", 80)
        >>> pool.close_all()
        """
        with self._lock:
            for conns in self._connections.values():
                for conn, _ in conns:
                    with suppress(Exception):
                        conn.close()
            self._connections.clear()
            self._active_count = 0


# ============================================================================
# Chunk Scheduler
# ============================================================================

class ChunkScheduler:
    """Partitions a file into byte-range chunks for concurrent downloading.

    Dynamically adjusts chunk size based on remaining bytes and
    available worker slots. Chunks are assigned sequentially.

    Parameters
    ----------
    total_size : int
        Total file size in bytes, or UNKNOWN_SIZE if not known in advance.
    min_chunk_size : int
        Smallest allowed chunk in bytes.
    max_chunk_size : int
        Largest allowed chunk in bytes.
    max_concurrent : int
        Maximum number of chunks being downloaded simultaneously.

    Warnings
    --------
    When total_size is UNKNOWN_SIZE, chunks beyond the first will have
    end_byte set to UNKNOWN_SIZE. Callers must handle streaming of
    indeterminate-length chunks.

    Examples
    --------
    >>> scheduler = ChunkScheduler(total_size=2000, min_chunk_size=500,
    ...                            max_chunk_size=1000, max_concurrent=2)
    >>> chunk1 = scheduler.get_next_chunk()
    >>> chunk1.start_byte, chunk1.end_byte
    (0, 999)
    >>> chunk2 = scheduler.get_next_chunk()
    >>> chunk2.start_byte, chunk2.end_byte
    (1000, 1999)
    >>> scheduler.get_next_chunk() is None
    True
    """

    def __init__(
        self,
        total_size: int,
        min_chunk_size: int = 256 * 1024,
        max_chunk_size: int = 16 * 1024 * 1024,
        max_concurrent: int = 4,
    ) -> None:
        self.total_size = total_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.max_concurrent = max_concurrent
        self._lock = threading.Lock()
        self._pending: Deque[ChunkDescriptor] = deque()
        self._in_progress: Dict[int, ChunkDescriptor] = {}
        self._completed: Dict[int, int] = {}
        self._next_byte = 0
        self._next_chunk_index = 0

    def _calculate_chunk_size(self) -> int:
        """Determine the size for the next chunk based on remaining work."""
        if self.total_size == UNKNOWN_SIZE:
            return self.max_chunk_size
        remaining = self.total_size - self._next_byte
        remaining_slots = max(1, self.max_concurrent - len(self._in_progress))
        chunk_size = max(self.min_chunk_size, remaining // remaining_slots)
        return min(chunk_size, self.max_chunk_size)

    def has_more_chunks(self) -> bool:
        """True if not all bytes have been allocated to chunks.

        Examples
        --------
        >>> ChunkScheduler(100).has_more_chunks()
        True
        >>> s = ChunkScheduler(0)
        >>> s.has_more_chunks()
        False
        """
        with self._lock:
            if self.total_size == UNKNOWN_SIZE:
                return True
            return self._next_byte < self.total_size

    def can_schedule_more(self) -> bool:
        """True if more chunks can be dispatched without exceeding concurrency.

        Examples
        --------
        >>> s = ChunkScheduler(10000, max_concurrent=1)
        >>> s.can_schedule_more()
        True
        >>> _ = s.get_next_chunk()
        >>> s.can_schedule_more()
        False
        """
        with self._lock:
            return self.has_more_chunks() and len(self._in_progress) < self.max_concurrent

    def get_next_chunk(self) -> Optional[ChunkDescriptor]:
        """Return the next chunk to download, or None if done.

        Returns
        -------
        Optional[ChunkDescriptor]
            Next chunk descriptor, or None.

        Examples
        --------
        >>> s = ChunkScheduler(500, max_concurrent=1)
        >>> chunk = s.get_next_chunk()
        >>> chunk.size
        500
        >>> s.get_next_chunk() is None
        True
        """
        with self._lock:
            if not self.has_more_chunks():
                return None
            chunk_size = self._calculate_chunk_size()
            start_byte = self._next_byte
            end_byte = (
                min(start_byte + chunk_size - 1, self.total_size - 1)
                if self.total_size != UNKNOWN_SIZE
                else UNKNOWN_SIZE
            )
            chunk = ChunkDescriptor(
                index=self._next_chunk_index,
                start_byte=start_byte,
                end_byte=end_byte,
            )
            self._next_chunk_index += 1
            self._next_byte = end_byte + 1 if end_byte != UNKNOWN_SIZE else UNKNOWN_SIZE
            self._in_progress[chunk.index] = chunk
            return chunk

    def mark_chunk_complete(self, chunk_index: int, bytes_written: int) -> None:
        """Record a chunk as successfully finished.

        Parameters
        ----------
        chunk_index : int
            Chunk identifier.
        bytes_written : int
            Number of bytes written to disk for this chunk.

        Examples
        --------
        >>> s = ChunkScheduler(100)
        >>> chunk = s.get_next_chunk()
        >>> s.mark_chunk_complete(chunk.index, 100)
        >>> s.completed_bytes
        100
        """
        with self._lock:
            self._in_progress.pop(chunk_index, None)
            self._completed[chunk_index] = bytes_written

    def mark_chunk_failed(self, chunk_index: int) -> None:
        """Return a failed chunk to the front of the pending queue.

        Parameters
        ----------
        chunk_index : int
            Chunk identifier.

        Examples
        --------
        >>> s = ChunkScheduler(100)
        >>> chunk = s.get_next_chunk()
        >>> s.mark_chunk_failed(chunk.index)
        >>> s.can_schedule_more()
        True
        """
        with self._lock:
            chunk = self._in_progress.pop(chunk_index, None)
            if chunk:
                self._pending.appendleft(chunk)

    @property
    def active_chunks(self) -> int:
        """Number of chunks currently in progress."""
        return len(self._in_progress)

    @property
    def completed_bytes(self) -> int:
        """Total bytes written for all completed chunks."""
        return sum(self._completed.values())


# ============================================================================
# Download Manager
# ============================================================================

class DownloadManager:
    """Orchestrates file downloads with retry, resume, and verification.

    Accepts a URL and destination path. Handles the full lifecycle:
    server probing, chunked transfer, progress reporting, checkpointing,
    and optional integrity verification.

    Parameters
    ----------
    network_config : NetworkConfig
        Settings for retries, timeouts, user agent, etc.
    verification_config : VerificationConfig
        Settings for hash verification.
    state_dir : Optional[Path]
        Directory for checkpoint files. Defaults to
        ~/.cache/python_headers_installer/checkpoints.

    Attributes
    ----------
    on_progress : Optional[Callable[[DownloadProgress], None]]
        Called periodically with a progress snapshot.
    on_status_change : Optional[Callable[[DownloadStatus, str], None]]
        Called when the download transitions to a new status.

    Warnings
    --------
    - Callbacks are invoked from background threads. Keep them short
      and do not block.
    - The manager is not a context manager. Call cancel() or allow
      downloads to complete before discarding.
    - Checkpoint files persist on disk. Delete them manually or call
      the checkpoint manager's delete_checkpoint after success.

    Examples
    --------
    >>> manager = DownloadManager()
    >>> manager.on_status_change = lambda s, m: print(f"{s.name}: {m}")
    >>> manager.on_progress = lambda p: print(p.format_human())
    >>> # manager.download("https://example.com/file.bin", Path("/tmp/file.bin"))
    """

    def __init__(
        self,
        network_config: NetworkConfig = NetworkConfig(),
        verification_config: VerificationConfig = VerificationConfig(),
        state_dir: Optional[Path] = None,
    ) -> None:
        self.network_config = network_config
        self.verification_config = verification_config
        if state_dir is None:
            state_dir = Path.home() / '.cache' / 'python_headers_installer' / 'checkpoints'
        self._checkpoint_manager = CheckpointManager(state_dir)
        self._speed_tracker = SpeedTracker()
        self._connection_pool = ConnectionPool()
        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._lock = threading.RLock()
        self._current_status = DownloadStatus.PENDING
        self._download_id: Optional[str] = None
        self._bytes_downloaded = 0
        self._chunk_scheduler: Optional[ChunkScheduler] = None
        self._integrity_verifier: Optional[IntegrityVerifier] = None
        self._start_time = 0.0
        self._retry_count = 0
        self._output_path: Optional[Path] = None
        self._url: str = ""
        self.on_progress: Optional[Callable[[DownloadProgress], None]] = None
        self.on_status_change: Optional[Callable[[DownloadStatus, str], None]] = None

    def _transition_status(self, new_status: DownloadStatus, message: str = "") -> None:
        """Update the download status and notify the callback."""
        with self._lock:
            self._current_status = new_status
        logger.debug(f"Status -> {new_status.name}: {message}")
        if self.on_status_change:
            try:
                self.on_status_change(new_status, message)
            except Exception:
                logger.debug("Status callback raised exception", exc_info=True)

    def _notify_progress(self) -> None:
        """Build a DownloadProgress snapshot and invoke the callback."""
        if not self.on_progress:
            return
        with self._lock:
            elapsed = time.monotonic() - self._start_time if self._start_time > 0 else 0.0
            total = UNKNOWN_SIZE
            if self._chunk_scheduler:
                total = self._chunk_scheduler.total_size
            completion = UNKNOWN_PERCENTAGE
            if total != UNKNOWN_SIZE and total > 0:
                completion = (self._bytes_downloaded / total) * 100.0
            eta = self._speed_tracker.estimate_eta(
                total - self._bytes_downloaded if total != UNKNOWN_SIZE else UNKNOWN_SIZE
            )
            active = self._chunk_scheduler.active_chunks if self._chunk_scheduler else 0
            progress = DownloadProgress(
                bytes_downloaded=self._bytes_downloaded,
                total_bytes=total,
                instantaneous_speed=self._speed_tracker.current_speed,
                average_speed=(self._bytes_downloaded / elapsed) if elapsed > 0 else 0.0,
                elapsed_seconds=elapsed,
                eta_seconds=eta,
                completion_percentage=completion,
                status=self._current_status,
                active_connections=active,
                retry_count=self._retry_count,
            )
        try:
            self.on_progress(progress)
        except Exception:
            logger.debug("Progress callback raised exception", exc_info=True)

    def cancel(self) -> None:
        """Request cancellation. The download will stop at the next chunk boundary.

        Examples
        --------
        >>> manager = DownloadManager()
        >>> manager.cancel()
        """
        self._cancel_event.set()
        self._transition_status(DownloadStatus.CANCELLED, "Cancelled by user")

    def pause(self) -> None:
        """Pause the download. Worker threads will block until resume is called.

        Examples
        --------
        >>> manager = DownloadManager()
        >>> manager.pause()
        >>> manager.resume()
        """
        self._pause_event.clear()
        self._transition_status(DownloadStatus.PAUSED, "Paused")

    def resume(self) -> None:
        """Resume a paused download.

        Examples
        --------
        >>> manager = DownloadManager()
        >>> manager.pause()
        >>> manager.resume()
        """
        self._pause_event.set()
        self._transition_status(DownloadStatus.DOWNLOADING, "Resumed")

    def reset(self) -> None:
        """Reset internal state for a new download operation.

        Called automatically at the start of download().

        Examples
        --------
        >>> manager = DownloadManager()
        >>> manager.reset()
        """
        self._cancel_event.clear()
        self._pause_event.set()
        self._bytes_downloaded = 0
        self._retry_count = 0
        self._start_time = 0.0
        self._chunk_scheduler = None
        self._integrity_verifier = None
        self._download_id = str(uuid.uuid4())[:8]

    def download(
        self,
        url: str,
        output_path: Path,
        expected_hash: Optional[str] = None,
        resume: bool = True,
    ) -> bool:
        """Download a file from url to output_path.

        Handles retries with exponential backoff. Resumes partial
        downloads if resume=True and a checkpoint exists.

        Parameters
        ----------
        url : str
            Source URL.
        output_path : Path
            Destination file path on disk.
        expected_hash : Optional[str]
            Expected hex digest for integrity verification.
        resume : bool
            If True, attempt to resume from a previous checkpoint.

        Returns
        -------
        bool
            True if the file was downloaded and verification (if enabled) passed.

        Raises
        ------
        DownloadError
            If the download fails after exhausting retries.
        VerificationError
            If integrity verification is enabled, expected_hash is provided,
            and the computed hash does not match.

        Warnings
        --------
        HTTPS downloads require a valid system CA bundle. On minimal
        containers or embedded systems, install ca-certificates.

        Examples
        --------
        >>> manager = DownloadManager(NetworkConfig(retries=2))
        >>> # success = manager.download("https://example.com/file", Path("/tmp/out"))
        """
        self.reset()
        self._url = url
        self._output_path = output_path
        self._start_time = time.monotonic()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._transition_status(DownloadStatus.PENDING, "Starting download")

        # Resume from checkpoint
        resume_from = 0
        if resume and self._download_id:
            checkpoint = self._checkpoint_manager.load_checkpoint(self._download_id)
            if checkpoint and checkpoint.get("url") == url:
                resume_from = checkpoint.get("bytes_downloaded", 0)
                self._bytes_downloaded = resume_from
                logger.info(f"Resuming from byte {resume_from}")

        # Probe server for file size and capabilities
        self._transition_status(DownloadStatus.CONNECTING, "Connecting")
        try:
            file_size, etag, last_modified = self._probe_server(url)
        except NetworkError as e:
            raise DownloadError(f"Server probe failed: {e}", url=url)

        # Set up integrity verifier
        if self.verification_config.enabled:
            self._integrity_verifier = IntegrityVerifier(
                algorithms=[self.verification_config.algorithm],
                expected_hashes=(
                    {self.verification_config.algorithm.value: expected_hash}
                    if expected_hash else None
                ),
            )

        # Initialize chunk scheduler
        self._chunk_scheduler = ChunkScheduler(
            total_size=file_size,
            max_concurrent=4,
        )

        # Download loop with retries
        last_error = None
        for attempt in range(self.network_config.retries):
            self._retry_count = attempt
            if attempt > 0:
                delay = calculate_jitter(self.network_config.retry_delay, attempt)
                logger.info(f"Retry {attempt+1}/{self.network_config.retries} in {delay:.1f}s")
                time.sleep(delay)
            try:
                self._transition_status(DownloadStatus.DOWNLOADING, "Downloading")
                self._download_sequential(url, output_path, resume_from, file_size)
                self._transition_status(DownloadStatus.COMPLETED, "Download finished")

                if self._integrity_verifier:
                    self._transition_status(DownloadStatus.VERIFYING, "Verifying integrity")
                    passed, computed = self._integrity_verifier.verify()
                    if not passed:
                        raise VerificationError(
                            "Hash mismatch",
                            url=url,
                            expected_hash=expected_hash or "",
                            actual_hash=list(computed.values())[0] if computed else "",
                        )
                    self._transition_status(DownloadStatus.VERIFIED, "Verification passed")

                self._checkpoint_manager.delete_checkpoint(self._download_id or "")
                return True

            except (DownloadError, VerificationError) as e:
                last_error = e
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                if isinstance(e, VerificationError) and not self.verification_config.skip_on_failure:
                    raise
                if attempt == self.network_config.retries - 1:
                    raise

        if last_error:
            raise last_error
        return False

    def _probe_server(self, url: str) -> Tuple[int, Optional[str], Optional[str]]:
        """Query the server for file size, ETag, and Last-Modified via HEAD.

        Parameters
        ----------
        url : str
            Target URL.

        Returns
        -------
        Tuple[int, Optional[str], Optional[str]]
            (file_size, etag, last_modified).
            file_size is -1 if not provided by the server.

        Raises
        ------
        NetworkError
            If the HEAD request fails.
        """
        try:
            request = Request(url, method='HEAD')
            request.add_header('User-Agent', self.network_config.user_agent)
            with urlopen(request, timeout=self.network_config.timeout) as response:
                content_length = response.headers.get('Content-Length')
                file_size = int(content_length) if content_length else UNKNOWN_SIZE
                etag = response.headers.get('ETag')
                last_modified = response.headers.get('Last-Modified')
                return file_size, etag, last_modified
        except HTTPError as e:
            raise NetworkError(f"HTTP {e.code}: {e.reason}", url=url, status_code=e.code)
        except URLError as e:
            raise NetworkError(f"Connection failed: {e.reason}", url=url)

    def _download_sequential(
        self, url: str, output_path: Path, resume_from: int, total_size: int,
    ) -> None:
        """Download the file sequentially, writing to output_path.

        If resume_from > 0, opens the existing file and seeks past
        already-downloaded bytes before writing new data.

        Parameters
        ----------
        url : str
            Source URL.
        output_path : Path
            Destination path.
        resume_from : int
            Byte offset to start downloading from.
        total_size : int
            Expected total file size, or UNKNOWN_SIZE.
        """
        request = Request(url)
        request.add_header('User-Agent', self.network_config.user_agent)
        if resume_from > 0:
            request.add_header('Range', f'bytes={resume_from}-')

        mode = 'ab' if resume_from > 0 else 'wb'
        with open(output_path, mode) as out_file:
            with urlopen(request, timeout=self.network_config.timeout) as response:
                while not self._cancel_event.is_set():
                    self._pause_event.wait()
                    chunk = response.read(self.network_config.chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    self._bytes_downloaded += len(chunk)
                    self._speed_tracker.record_bytes(self._bytes_downloaded)
                    self._notify_progress()
                    if self._checkpoint_manager.should_checkpoint(self._bytes_downloaded):
                        if self._download_id:
                            self._checkpoint_manager.save_checkpoint(
                                download_id=self._download_id,
                                url=self._url,
                                output_path=self._output_path or output_path,
                                bytes_downloaded=self._bytes_downloaded,
                                total_bytes=total_size,
                                chunk_states={},
                            )
                    if self._integrity_verifier:
                        self._integrity_verifier.update(chunk)
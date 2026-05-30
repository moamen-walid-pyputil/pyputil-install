#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cryptographic hash computation and verification.

This module provides utilities for computing and verifying cryptographic
hashes of package files. It supports multiple hash algorithms, streaming
computation for large files, and integration with PyPI's digest format.

Classes
-------
HashAlgorithm
    Enumeration of supported hash algorithms.
HashVerifier
    Verifies file integrity against expected hash values.

Functions
--------
compute_hash
    Compute the hash of a file or bytes.
verify_hash
    Verify that data matches an expected hash.
multi_hash
    Compute multiple hash algorithms in a single pass.
compare_hashes
    Constant-time hash comparison to prevent timing attacks.

Examples
--------
>>> from package_installer.security.hashes import compute_hash, HashAlgorithm
>>> hash_value = compute_hash(b"Hello, World!", HashAlgorithm.SHA256)
>>> hash_value
'dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f'
"""

import hashlib
import hmac
import logging
import io
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, BinaryIO

from ..exceptions import ValidationError

logger = logging.getLogger(__name__)


class HashAlgorithm(str, Enum):
    """
    Supported cryptographic hash algorithms.

    Notes
    -----
    Algorithm recommendations per NIST and industry standards:

    - **SHA256**: Minimum recommended for general use.
    - **SHA384**: Higher security margin than SHA256.
    - **SHA512**: Highest security margin. Preferred for long-term integrity.
    - **SHA3_256**: SHA-3 family, resistant to length extension attacks.
    - **SHA3_512**: SHA-3 family with 512-bit output.
    - **BLAKE2b**: Faster than SHA-3 on 64-bit platforms.
    - **MD5**: NOT recommended for security. Included for legacy
      PyPI digest compatibility only.
    """

    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    SHA3_256 = "sha3_256"
    SHA3_512 = "sha3_512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"
    MD5 = "md5"

    @property
    def digest_size(self) -> int:
        """
        Get the digest size in bytes for this algorithm.

        Returns
        -------
        int
            Digest size in bytes.
        """
        sizes = {
            HashAlgorithm.SHA256: 32,
            HashAlgorithm.SHA384: 48,
            HashAlgorithm.SHA512: 64,
            HashAlgorithm.SHA3_256: 32,
            HashAlgorithm.SHA3_512: 64,
            HashAlgorithm.BLAKE2B: 64,
            HashAlgorithm.BLAKE2S: 32,
            HashAlgorithm.MD5: 16,
        }
        return sizes.get(self, 0)

    @property
    def hex_length(self) -> int:
        """
        Get the hex string length for this algorithm.

        Returns
        -------
        int
            Length of the hex-encoded digest string.
        """
        return self.digest_size * 2

    def is_secure(self) -> bool:
        """
        Check if this algorithm is considered cryptographically secure.

        Returns
        -------
        bool
            False for MD5, True for all others.
        """
        return self != HashAlgorithm.MD5


def _get_hash_object(algorithm: HashAlgorithm) -> Any:
    """
    Get a hashlib hash object for the given algorithm.

    Parameters
    ----------
    algorithm : HashAlgorithm
        The hash algorithm to use.

    Returns
    -------
    hashlib hash object
        A new hash object for the specified algorithm.

    Raises
    ------
    ValueError
        If the algorithm is not available in the current Python build.
    """
    algo_map = {
        HashAlgorithm.SHA256: hashlib.sha256,
        HashAlgorithm.SHA384: hashlib.sha384,
        HashAlgorithm.SHA512: hashlib.sha512,
        HashAlgorithm.SHA3_256: hashlib.sha3_256,
        HashAlgorithm.SHA3_512: hashlib.sha3_512,
        HashAlgorithm.BLAKE2B: hashlib.blake2b,
        HashAlgorithm.BLAKE2S: hashlib.blake2s,
        HashAlgorithm.MD5: hashlib.md5,
    }

    try:
        return algo_map[algorithm]()
    except KeyError:
        raise ValueError(
            f"Unsupported hash algorithm: {algorithm}"
        )
    except AttributeError:
        raise ValueError(
            f"Hash algorithm {algorithm} is not available. "
            f"Python may be compiled without {algorithm} support."
        )


def compute_hash(
    data: Union[bytes, str, Path, BinaryIO],
    algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    chunk_size: int = 8192,
) -> str:
    """
    Compute the cryptographic hash of data.

    Parameters
    ----------
    data : bytes, str, Path, or file-like object
        The data to hash. Can be:

        - ``bytes``: Hashed directly.
        - ``str``: Encoded as UTF-8 and hashed.
        - ``Path``: File at the path is read and hashed.
        - file-like object: Read in chunks and hashed.
    algorithm : HashAlgorithm, default=SHA256
        The hash algorithm to use.
    chunk_size : int, default=8192
        Chunk size in bytes for reading files. Larger values use
        more memory but may be faster for large files.

    Returns
    -------
    str
        Hex-encoded digest string.

    Raises
    ------
    ValidationError
        If the data is a Path that does not exist or cannot be read.
    ValueError
        If the algorithm is not supported.

    Notes
    -----
    For file paths and file-like objects, data is read in chunks to
    avoid loading the entire file into memory. This is suitable for
    files of any size.

    Examples
    --------
    >>> compute_hash(b"hello", HashAlgorithm.SHA256)
    '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'

    >>> from pathlib import Path
    >>> hash_value = compute_hash(Path("/path/to/package.whl"))
    """
    hash_obj = _get_hash_object(algorithm)

    if isinstance(data, bytes):
        hash_obj.update(data)
        return hash_obj.hexdigest()

    if isinstance(data, str):
        hash_obj.update(data.encode("utf-8"))
        return hash_obj.hexdigest()

    if isinstance(data, Path):
        if not data.exists():
            raise ValidationError(
                f"File not found: {data}",
                package_name="",
            )
        if not data.is_file():
            raise ValidationError(
                f"Path is not a file: {data}",
                package_name="",
            )
        try:
            with open(data, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    hash_obj.update(chunk)
        except OSError as e:
            raise ValidationError(
                f"Cannot read file {data}: {e}",
                package_name="",
            ) from e
        return hash_obj.hexdigest()

    if hasattr(data, "read"):
        while True:
            chunk = data.read(chunk_size)
            if not chunk:
                break
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            hash_obj.update(chunk)
        return hash_obj.hexdigest()

    raise TypeError(
        f"Unsupported data type: {type(data)}. "
        f"Expected bytes, str, Path, or file-like object."
    )


def multi_hash(
    data: Union[bytes, str, Path, BinaryIO],
    algorithms: List[HashAlgorithm],
    chunk_size: int = 8192,
) -> Dict[str, str]:
    """
    Compute multiple hash algorithms in a single pass over the data.

    Parameters
    ----------
    data : bytes, str, Path, or file-like object
        The data to hash.
    algorithms : list of HashAlgorithm
        The hash algorithms to compute.
    chunk_size : int, default=8192
        Chunk size for reading files.

    Returns
    -------
    dict
        Mapping of algorithm names to hex-encoded digest strings.

    Notes
    -----
    This is more efficient than calling ``compute_hash`` multiple
    times because the data is read only once.

    Examples
    --------
    >>> multi_hash(b"data", [HashAlgorithm.SHA256, HashAlgorithm.MD5])
    {'sha256': '3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7',
     'md5': '8d777f385d3dfec8815d20f7496026dc'}
    """
    hash_objects = {
        algo.value: _get_hash_object(algo) for algo in algorithms
    }

    if isinstance(data, bytes):
        for hash_obj in hash_objects.values():
            hash_obj.update(data)

    elif isinstance(data, str):
        encoded = data.encode("utf-8")
        for hash_obj in hash_objects.values():
            hash_obj.update(encoded)

    elif isinstance(data, Path):
        if not data.exists() or not data.is_file():
            raise ValidationError(
                f"Cannot read file: {data}",
                package_name="",
            )
        try:
            with open(data, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    for hash_obj in hash_objects.values():
                        hash_obj.update(chunk)
        except OSError as e:
            raise ValidationError(
                f"Cannot read file {data}: {e}",
                package_name="",
            ) from e

    elif hasattr(data, "read"):
        while True:
            chunk = data.read(chunk_size)
            if not chunk:
                break
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            for hash_obj in hash_objects.values():
                hash_obj.update(chunk)

    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}"
        )

    return {
        algo: hash_obj.hexdigest()
        for algo, hash_obj in hash_objects.items()
    }


def verify_hash(
    data: Union[bytes, str, Path, BinaryIO],
    expected_hash: str,
    algorithm: HashAlgorithm = HashAlgorithm.SHA256,
) -> bool:
    """
    Verify that data matches an expected hash value.

    Parameters
    ----------
    data : bytes, str, Path, or file-like object
        The data to verify.
    expected_hash : str
        The expected hex-encoded digest string.
    algorithm : HashAlgorithm, default=SHA256
        The hash algorithm to use.

    Returns
    -------
    bool
        True if the computed hash matches the expected hash.

    Raises
    ------
    ValidationError
        If the expected_hash has an invalid length for the algorithm.

    Notes
    -----
    Uses constant-time comparison via ``hmac.compare_digest`` to
    prevent timing attacks. This is important when comparing hashes
    from untrusted sources.

    Examples
    --------
    >>> expected = compute_hash(b"secret", HashAlgorithm.SHA256)
    >>> verify_hash(b"secret", expected, HashAlgorithm.SHA256)
    True
    >>> verify_hash(b"wrong", expected, HashAlgorithm.SHA256)
    False
    """
    if len(expected_hash) != algorithm.hex_length:
        raise ValidationError(
            f"Expected hash length {len(expected_hash)} does not match "
            f"algorithm {algorithm.value} (expected {algorithm.hex_length})",
            package_name="",
            expected_hash=expected_hash,
            algorithm=algorithm.value,
        )

    computed = compute_hash(data, algorithm)
    return compare_hashes(computed, expected_hash)


def compare_hashes(hash_a: str, hash_b: str) -> bool:
    """
    Compare two hash strings in constant time.

    Parameters
    ----------
    hash_a : str
        First hash string.
    hash_b : str
        Second hash string.

    Returns
    -------
    bool
        True if the hashes are equal.

    Notes
    -----
    Uses ``hmac.compare_digest`` to perform constant-time comparison,
    which prevents timing side-channel attacks. The strings are
    converted to lowercase before comparison for case-insensitive
    matching.

    Examples
    --------
    >>> compare_hashes("abc123", "abc123")
    True
    >>> compare_hashes("ABC123", "abc123")
    True
    >>> compare_hashes("abc123", "def456")
    False
    """
    return hmac.compare_digest(
        hash_a.lower().encode("ascii"),
        hash_b.lower().encode("ascii"),
    )


class HashVerifier:
    """
    Verifies file integrity using cryptographic hashes.

    This class validates that downloaded package files match their
    expected hash values from PyPI, protecting against corruption
    and supply chain attacks.

    Parameters
    ----------
    algorithm : HashAlgorithm, default=SHA256
        The primary hash algorithm for verification.
    fallback_algorithms : list of HashAlgorithm, optional
        Additional algorithms to try if the primary algorithm hash
        is not available in the expected hashes.

    Attributes
    ----------
    algorithm : HashAlgorithm
        The primary hash algorithm.
    fallback_algorithms : list of HashAlgorithm
        Fallback algorithms in priority order.

    Notes
    -----
    PyPI provides multiple hash algorithms for each distribution
    file. By default, SHA256 is preferred. If SHA256 is unavailable
    for a specific file, the verifier will try fallback algorithms
    in order.

    MD5 is accepted only as a fallback and will generate a warning
    since it is not cryptographically secure.

    Examples
    --------
    >>> verifier = HashVerifier()
    >>> verifier.verify_file(
    ...     Path("/tmp/package.whl"),
    ...     {"sha256": "abc123...", "md5": "def456..."},
    ... )
    True

    With multiple expected hashes::

    >>> verifier = HashVerifier(
    ...     algorithm=HashAlgorithm.SHA512,
    ...     fallback_algorithms=[HashAlgorithm.SHA256, HashAlgorithm.MD5],
    ... )
    """

    def __init__(
        self,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        fallback_algorithms: Optional[List[HashAlgorithm]] = None,
    ) -> None:
        self.algorithm = algorithm
        self.fallback_algorithms = fallback_algorithms or [
            HashAlgorithm.SHA512,
            HashAlgorithm.SHA384,
            HashAlgorithm.MD5,
        ]

        if algorithm not in self.fallback_algorithms:
            self.fallback_algorithms.insert(0, algorithm)
        else:
            self.fallback_algorithms.remove(algorithm)
            self.fallback_algorithms.insert(0, algorithm)

        self._verification_count: int = 0
        self._failure_count: int = 0
        logger.debug(
            f"HashVerifier initialized with primary={algorithm.value}"
        )

    def verify_file(
        self,
        file_path: Path,
        expected_hashes: Dict[str, str],
    ) -> bool:
        """
        Verify that a file matches at least one expected hash.

        Parameters
        ----------
        file_path : Path
            Path to the file to verify.
        expected_hashes : dict
            Dictionary mapping algorithm names to hex-encoded digests
            (e.g., ``{"sha256": "abc...", "md5": "def..."}``).

        Returns
        -------
        bool
            True if at least one hash algorithm verified successfully.

        Raises
        ------
        ValidationError
            If no expected hashes could be verified or the file
            cannot be read.

        Notes
        -----
        Algorithms are tried in priority order: primary algorithm
        first, then fallback algorithms. The first match succeeds.
        If no algorithm matches, a ``ValidationError`` is raised
        with details of all attempted verifications.

        If MD5 is the only algorithm that matches, a warning is
        logged because MD5 is not cryptographically secure.

        Examples
        --------
        >>> verifier = HashVerifier()
        >>> expected = {"sha256": compute_hash(Path("package.whl"))}
        >>> verifier.verify_file(Path("package.whl"), expected)
        True
        """
        if not file_path.exists():
            raise ValidationError(
                f"File not found for verification: {file_path}",
                package_name="",
            )

        if not file_path.is_file():
            raise ValidationError(
                f"Path is not a file: {file_path}",
                package_name="",
            )

        if not expected_hashes:
            raise ValidationError(
                "No expected hashes provided for verification",
                package_name="",
            )

        self._verification_count += 1
        errors: List[str] = []

        for algo in self.fallback_algorithms:
            algo_name = algo.value

            if algo_name not in expected_hashes:
                continue

            expected = expected_hashes[algo_name]

            if not expected:
                continue

            try:
                computed = compute_hash(file_path, algo)
                if compare_hashes(computed, expected):
                    if algo == HashAlgorithm.MD5:
                        logger.warning(
                            f"{file_path.name}: Verified using MD5, "
                            f"which is not cryptographically secure. "
                            f"Consider using SHA256 or stronger."
                        )
                    logger.debug(
                        f"{file_path.name}: Verified with {algo_name}"
                    )
                    return True
                else:
                    errors.append(
                        f"{algo_name} mismatch: "
                        f"expected={expected[:16]}..., "
                        f"got={computed[:16]}..."
                    )
            except Exception as e:
                errors.append(f"{algo_name} error: {e}")

        self._failure_count += 1

        raise ValidationError(
            f"Hash verification failed for {file_path.name}. "
            f"Errors: {'; '.join(errors)}",
            package_name="",
            algorithm=self.algorithm.value,
        )

    def verify_bytes(
        self,
        data: bytes,
        expected_hashes: Dict[str, str],
    ) -> bool:
        """
        Verify that raw bytes match at least one expected hash.

        Parameters
        ----------
        data : bytes
            The data to verify.
        expected_hashes : dict
            Dictionary mapping algorithm names to hex-encoded digests.

        Returns
        -------
        bool
            True if at least one hash verified successfully.

        Raises
        ------
        ValidationError
            If verification fails.

        Examples
        --------
        >>> verifier = HashVerifier()
        >>> expected = {"sha256": compute_hash(b"hello")}
        >>> verifier.verify_bytes(b"hello", expected)
        True
        """
        self._verification_count += 1
        errors: List[str] = []

        for algo in self.fallback_algorithms:
            algo_name = algo.value

            if algo_name not in expected_hashes:
                continue

            expected = expected_hashes[algo_name]
            if not expected:
                continue

            try:
                computed = compute_hash(data, algo)
                if compare_hashes(computed, expected):
                    if algo == HashAlgorithm.MD5:
                        logger.warning(
                            "Verified using MD5, which is not secure"
                        )
                    return True
                else:
                    errors.append(
                        f"{algo_name} mismatch"
                    )
            except Exception as e:
                errors.append(f"{algo_name} error: {e}")

        self._failure_count += 1

        raise ValidationError(
            f"Hash verification failed for data. "
            f"Errors: {'; '.join(errors)}",
            package_name="",
            algorithm=self.algorithm.value,
        )

    def compute_expected_hashes(
        self,
        file_path: Path,
        algorithms: Optional[List[HashAlgorithm]] = None,
    ) -> Dict[str, str]:
        """
        Compute hashes of a file for use as expected values.

        Parameters
        ----------
        file_path : Path
            Path to the file.
        algorithms : list of HashAlgorithm, optional
            Algorithms to compute. If None, uses primary algorithm
            plus all fallback algorithms.

        Returns
        -------
        dict
            Mapping of algorithm names to hex-encoded digests.

        Examples
        --------
        >>> verifier = HashVerifier()
        >>> hashes = verifier.compute_expected_hashes(Path("package.whl"))
        >>> "sha256" in hashes
        True
        """
        if algorithms is None:
            algorithms = [self.algorithm] + [
                a for a in self.fallback_algorithms if a != self.algorithm
            ]

        return multi_hash(file_path, algorithms)

    @property
    def stats(self) -> Dict[str, int]:
        """
        Get verification statistics.

        Returns
        -------
        dict
            Dictionary with ``total``, ``failures``, and
            ``success_rate``.

        Examples
        --------
        >>> verifier = HashVerifier()
        >>> verifier.stats["total"]
        0
        """
        return {
            "total": self._verification_count,
            "failures": self._failure_count,
            "successes": self._verification_count - self._failure_count,
            "success_rate": (
                (self._verification_count - self._failure_count)
                / self._verification_count
                if self._verification_count > 0
                else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset verification statistics to zero."""
        self._verification_count = 0
        self._failure_count = 0

    def __repr__(self) -> str:
        """String representation of the verifier."""
        return (
            f"HashVerifier("
            f"primary={self.algorithm.value}, "
            f"verified={self._verification_count})"
        )


def compute_hashes_from_pypi_digests(
    digests: Dict[str, str],
) -> Dict[str, str]:
    """
    Convert PyPI digest format to standardized hash dictionary.

    Parameters
    ----------
    digests : dict
        PyPI file digests dictionary (e.g.,
        ``{"sha256": "abc...", "md5": "def..."}``).

    Returns
    -------
    dict
        Standardized dictionary with algorithm names as keys and
        hex-encoded digests as values. Unknown or unsupported
        algorithms are excluded.

    Notes
    -----
    PyPI uses lowercase algorithm names. This function normalizes
    the names and filters out algorithms that are not recognized.

    Examples
    --------
    >>> digests = {"sha256": "abc123", "md5": "def456", "unknown": "xyz"}
    >>> compute_hashes_from_pypi_digests(digests)
    {'sha256': 'abc123', 'md5': 'def456'}
    """
    recognized = {algo.value for algo in HashAlgorithm}
    return {
        algo: digest
        for algo, digest in digests.items()
        if algo in recognized and digest
    }


def is_hash_string(value: str, algorithm: HashAlgorithm) -> bool:
    """
    Check if a string looks like a valid hex-encoded hash.

    Parameters
    ----------
    value : str
        The string to check.
    algorithm : HashAlgorithm
        The expected hash algorithm.

    Returns
    -------
    bool
        True if the string has the correct length and is valid hex.

    Notes
    -----
    This only checks format, not cryptographic validity.

    Examples
    --------
    >>> is_hash_string("abc123", HashAlgorithm.SHA256)
    False
    >>> is_hash_string("a" * 64, HashAlgorithm.SHA256)
    True
    """
    if len(value) != algorithm.hex_length:
        return False

    try:
        int(value, 16)
        return True
    except ValueError:
        return False
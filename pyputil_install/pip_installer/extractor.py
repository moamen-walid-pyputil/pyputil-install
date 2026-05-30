"""
Archive Extraction Module.

Detects and decompresses archives in any common format without requiring
the caller to know the compression type in advance. Handles ZIP, gzip,
bzip2, xz, lzma, tar (uncompressed and compressed), and raw deflate.

Why this module exists:
    - Network downloads may arrive compressed (gzip, deflate) even when
      the URL suggests an uncompressed resource. Servers and CDNs can
      transparently compress responses based on Accept-Encoding headers.
    - Wheel files are ZIP archives but may be served with Content-Encoding:
      gzip, producing a gzip-compressed ZIP that must be decompressed
      before use.
    - get-pip.py fetched from some mirrors returns gzip-compressed data
      that appears as a binary blob starting with ``\\x1f\\x8b`` instead
      of valid Python source code.
    - Python's standard library has all the tools (``zipfile``, ``gzip``,
      ``bz2``, ``lzma``, ``tarfile``) but using them requires knowing
      the format in advance. This module detects the format from magic
      bytes and applies the correct decompressor automatically.
    - Nested archives (e.g., ``.tar.gz`` inside a ``.zip``) are common
      in Python packaging. This module handles one level of nesting
      to extract the final payload.

Detection is based on magic bytes (file signatures) at the start of
the data stream. This is more reliable than file extensions because:
    - Files may have no extension (temporary files, pipes).
    - Files may have misleading extensions (``.whl`` is a ZIP but
      doesn't end with ``.zip``).
    - Network streams have no filenames at all.

Warnings
--------
- This module reads the entire archive into memory for detection and
  decompression. Archives larger than available RAM will cause
  MemoryError. For large archives, use streaming extraction via the
  ``extract_to_dir`` function which writes to disk incrementally.
- Nested archive detection is limited to one level. A ``.tar.gz``
  inside a ``.zip`` will be extracted to a ``.tar`` file, not fully
  unpacked. Call ``extract_archive`` twice for full extraction.
- The module does not handle encrypted or password-protected archives.
  Attempting to extract them will raise ``ValueError``.
- Some formats (``.xz``, ``.lzma``) require the ``lzma`` module which
  is available in Python 3.3+ but may be missing in minimal builds.

Examples
--------
Extract any archive to a directory:

    >>> from extractor import extract_to_dir
    >>> files = extract_to_dir(Path("downloads/pip-24.0.tar.gz"), Path("output"))
    >>> print(files)
    ['output/pip-24.0/setup.py', 'output/pip-24.0/README.md', ...]

Decompress data from network response:

    >>> from extractor import decompress_bytes
    >>> compressed = b'\\x1f\\x8b\\x08\\x00...'  # gzip data
    >>> decompressed = decompress_bytes(compressed)
    >>> decompressed[:50]
    b'#!/usr/bin/env python\\n...'

Detect archive type from bytes:

    >>> from extractor import detect_archive_type
    >>> detect_archive_type(b'PK\\x03\\x04...')
    <ArchiveType.ZIP: 2>

Extract wheel contents directly:

    >>> from extractor import extract_wheel
    >>> extract_wheel(Path("pip-24.0-py3-none-any.whl"), Path("site-packages"))

Handle nested compression (gzip-compressed tar):

    >>> from extractor import decompress_bytes
    >>> with open("archive.tar.gz", "rb") as f:
    ...     data = f.read()
    >>> tar_bytes = decompress_bytes(data)  # First layer: gzip
    >>> files = extract_tar_bytes(tar_bytes, Path("output"))  # Second: tar
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Public Enumerations
# ---------------------------------------------------------------------------


class ArchiveType(Enum):
    """
    Identifies the compression or archive format detected from magic bytes.

    Why an enum instead of strings:
        - Magic byte matching returns one of these variants; strings would
          risk typos in format names.
        - Enables exhaustive handling: ``match archive_type: case ArchiveType.ZIP: ...``
        - Each variant carries semantic meaning beyond the format name,
          such as whether it's a container (contains files) or just a
          compression layer (wraps a single stream).

    Attributes
    ----------
    ZIP : ArchiveType
        ZIP archive (``PK`` magic). Container format.
    GZIP : ArchiveType
        gzip compression (``\\x1f\\x8b`` magic). Single-stream compression.
    BZIP2 : ArchiveType
        bzip2 compression (``BZh`` magic). Single-stream compression.
    XZ : ArchiveType
        xz/lzma compression (``\\xfd7zXZ`` magic). Single-stream compression.
    LZMA : ArchiveType
        Legacy lzma compression (``\\x5d\\x00`` magic). Single-stream compression.
    TAR : ArchiveType
        Uncompressed tar archive (``ustar`` magic at offset 257). Container format.
    DEFLATE : ArchiveType
        Raw deflate/zlib compression. Detected when no other magic matches
        but data appears compressed.
    UNKNOWN : ArchiveType
        Format could not be identified. Data may be uncompressed or in
        an unsupported format.

    Examples
    --------
    >>> atype = ArchiveType.GZIP
    >>> atype.name
    'GZIP'
    >>> atype == ArchiveType.ZIP
    False
    """

    ZIP = auto()
    """ZIP archive (container format, multiple files)."""

    GZIP = auto()
    """gzip compression (single stream)."""

    BZIP2 = auto()
    """bzip2 compression (single stream)."""

    XZ = auto()
    """xz/lzma compression (single stream)."""

    LZMA = auto()
    """Legacy lzma compression (single stream)."""

    TAR = auto()
    """Uncompressed tar archive (container format, multiple files)."""

    DEFLATE = auto()
    """Raw deflate/zlib compression (single stream)."""

    UNKNOWN = auto()
    """Format could not be identified."""


class CompressionType(Enum):
    """
    Identifies compression-only formats (not container formats).

    Why separate from ArchiveType:
        - Compression formats (gzip, bzip2, xz) wrap a single byte stream.
          Container formats (ZIP, tar) hold multiple files.
        - The extraction logic differs: compression needs decompression
          only; containers need file-by-file extraction.
        - Some formats are both (tar.gz is gzip wrapping tar). We
          distinguish the outer compression from the inner container.

    Attributes
    ----------
    GZIP : CompressionType
        gzip compression.
    BZIP2 : CompressionType
        bzip2 compression.
    XZ : CompressionType
        xz compression.
    LZMA : CompressionType
        Legacy lzma compression.
    DEFLATE : CompressionType
        Raw deflate/zlib compression.
    NONE : CompressionType
        No compression detected.

    Examples
    --------
    >>> ctype = CompressionType.GZIP
    >>> ctype.name
    'GZIP'
    """

    GZIP = auto()
    BZIP2 = auto()
    XZ = auto()
    LZMA = auto()
    DEFLATE = auto()
    NONE = auto()


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------


@dataclass
class ArchiveInfo:
    """
    Information about a detected archive format.

    Why a dataclass instead of a tuple:
        - Named fields are self-documenting: ``info.is_container`` is
          clearer than ``info[2]``.
        - Enables adding fields (like ``compression_ratio``) in the
          future without breaking callers.

    Attributes
    ----------
    archive_type : ArchiveType
        The detected archive or compression format.
    compression_type : CompressionType
        The compression layer, if any. ``CompressionType.NONE`` for
        uncompressed containers.
    is_container : bool
        ``True`` if the format contains multiple files (ZIP, tar).
    is_compressed : bool
        ``True`` if the data is compressed (gzip, bzip2, xz, deflate).
    extension : str
        Standard file extension for this format (e.g., ``'.zip'``,
        ``'.tar.gz'``).
    mime_type : str
        Standard MIME type for this format.

    Examples
    --------
    >>> info = ArchiveInfo(
    ...     archive_type=ArchiveType.GZIP,
    ...     compression_type=CompressionType.GZIP,
    ...     is_container=False,
    ...     is_compressed=True,
    ...     extension='.gz',
    ...     mime_type='application/gzip',
    ... )
    >>> info.is_compressed
    True
    """

    archive_type: ArchiveType
    compression_type: CompressionType
    is_container: bool
    is_compressed: bool
    extension: str
    mime_type: str


@dataclass
class ExtractionResult:
    """
    Result of an extraction operation.

    Why a dedicated result type:
        - Reports both success/failure and the list of extracted files.
        - Includes metadata about the detected format and any warnings.
        - Enables callers to verify what was extracted without scanning
          the output directory.

    Attributes
    ----------
    success : bool
        ``True`` if extraction completed without errors.
    archive_type : ArchiveType
        The detected format that was extracted.
    files_extracted : List[Path]
        Absolute paths of all extracted files.
    bytes_decompressed : int
        Number of bytes after decompression (0 for container formats).
    warnings : List[str]
        Non-fatal issues encountered during extraction.
    errors : List[str]
        Fatal errors if extraction failed.

    Examples
    --------
    >>> result = ExtractionResult(
    ...     success=True,
    ...     archive_type=ArchiveType.ZIP,
    ...     files_extracted=[Path("/output/pip/__init__.py")],
    ...     bytes_decompressed=0,
    ...     warnings=[],
    ...     errors=[],
    ... )
    >>> result.success
    True
    """

    success: bool
    archive_type: ArchiveType
    files_extracted: List[Path] = field(default_factory=list)
    bytes_decompressed: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Magic Bytes Database
# ---------------------------------------------------------------------------

# Why a dict mapping magic bytes to ArchiveType:
#   - Detection is O(n) where n is the number of formats (small constant).
#   - Magic bytes are checked in order; longer prefixes are checked first
#     to avoid false matches (e.g., ``\\x1f\\x8b`` for gzip before
#     ``\\x1f`` alone for something else).
#   - Adding a new format requires only adding an entry to this dict.

_MAGIC_BYTES: Dict[bytes, ArchiveType] = {
    b"PK\x03\x04": ArchiveType.ZIP,          # ZIP archive (also .whl, .jar)
    b"PK\x05\x06": ArchiveType.ZIP,          # Empty ZIP archive
    b"PK\x07\x08": ArchiveType.ZIP,          # Spanned ZIP archive
    b"\x1f\x8b\x08": ArchiveType.GZIP,       # gzip compression
    b"BZh": ArchiveType.BZIP2,               # bzip2 compression
    b"\xfd7zXZ": ArchiveType.XZ,            # xz compression
    b"\x5d\x00\x00": ArchiveType.LZMA,       # Legacy lzma compression
}

# Tar magic is at offset 257, not the start of the file.
# We check for it separately in the detection function.
_TAR_MAGIC_OFFSET: int = 257
_TAR_MAGIC_BYTES: bytes = b"ustar\x00"
_TAR_MAGIC_BYTES_ALT: bytes = b"ustar  \x00"  # GNU tar variant

# Archive format metadata
_ARCHIVE_INFO_MAP: Dict[ArchiveType, ArchiveInfo] = {
    ArchiveType.ZIP: ArchiveInfo(
        archive_type=ArchiveType.ZIP,
        compression_type=CompressionType.NONE,
        is_container=True,
        is_compressed=False,
        extension=".zip",
        mime_type="application/zip",
    ),
    ArchiveType.GZIP: ArchiveInfo(
        archive_type=ArchiveType.GZIP,
        compression_type=CompressionType.GZIP,
        is_container=False,
        is_compressed=True,
        extension=".gz",
        mime_type="application/gzip",
    ),
    ArchiveType.BZIP2: ArchiveInfo(
        archive_type=ArchiveType.BZIP2,
        compression_type=CompressionType.BZIP2,
        is_container=False,
        is_compressed=True,
        extension=".bz2",
        mime_type="application/x-bzip2",
    ),
    ArchiveType.XZ: ArchiveInfo(
        archive_type=ArchiveType.XZ,
        compression_type=CompressionType.XZ,
        is_container=False,
        is_compressed=True,
        extension=".xz",
        mime_type="application/x-xz",
    ),
    ArchiveType.LZMA: ArchiveInfo(
        archive_type=ArchiveType.LZMA,
        compression_type=CompressionType.LZMA,
        is_container=False,
        is_compressed=True,
        extension=".lzma",
        mime_type="application/x-lzma",
    ),
    ArchiveType.TAR: ArchiveInfo(
        archive_type=ArchiveType.TAR,
        compression_type=CompressionType.NONE,
        is_container=True,
        is_compressed=False,
        extension=".tar",
        mime_type="application/x-tar",
    ),
    ArchiveType.DEFLATE: ArchiveInfo(
        archive_type=ArchiveType.DEFLATE,
        compression_type=CompressionType.DEFLATE,
        is_container=False,
        is_compressed=True,
        extension=".deflate",
        mime_type="application/zlib",
    ),
    ArchiveType.UNKNOWN: ArchiveInfo(
        archive_type=ArchiveType.UNKNOWN,
        compression_type=CompressionType.NONE,
        is_container=False,
        is_compressed=False,
        extension="",
        mime_type="application/octet-stream",
    ),
}


# ---------------------------------------------------------------------------
# Public API: Detection
# ---------------------------------------------------------------------------


def detect_archive_type(data: bytes) -> ArchiveType:
    """
    Identify the archive or compression format from raw bytes.

    Why detect from bytes instead of filename:
        - Files may have no extension (temporary files, download streams).
        - Extensions can be misleading (``.whl`` is a ZIP archive).
        - Network responses have Content-Type headers but they can be
          wrong or missing.
        - Magic bytes are definitive for well-formed archives.

    Detection order matters:
        1. Check magic bytes at offset 0 for all compression and container
           formats in the ``_MAGIC_BYTES`` dict.
        2. Check for tar magic at offset 257 (tar archives store their
           signature deep in the header).
        3. Check for raw deflate/zlib (``\\x78`` prefix).
        4. Return ``UNKNOWN`` if no signature matches.

    Parameters
    ----------
    data : bytes
        Raw bytes from the beginning of the archive. At least 512 bytes
        are needed for reliable detection (tar header is at offset 257-262).
        Fewer bytes may produce ``UNKNOWN`` for tar archives.

    Returns
    -------
    ArchiveType
        The detected format.

    Examples
    --------
    >>> detect_archive_type(b"PK\\x03\\x04\\x00\\x00...")
    <ArchiveType.ZIP: 1>

    >>> detect_archive_type(b"\\x1f\\x8b\\x08\\x00...")
    <ArchiveType.GZIP: 2>

    >>> detect_archive_type(b"#!/usr/bin/env python\\n...")
    <ArchiveType.UNKNOWN: 8>

    >>> # tar archive detection (magic at offset 257)
    >>> tar_data = b"\\x00" * 257 + b"ustar\\x00" + b"..."
    >>> detect_archive_type(tar_data)
    <ArchiveType.TAR: 6>
    """
    if len(data) < 4:
        return ArchiveType.UNKNOWN

    # Check magic bytes at offset 0
    # Sort by magic length descending to match longest prefix first
    sorted_magics = sorted(_MAGIC_BYTES.items(), key=lambda x: len(x[0]), reverse=True)
    for magic, archive_type in sorted_magics:
        if data[: len(magic)] == magic:
            return archive_type

    # Check for tar magic at offset 257
    if len(data) >= _TAR_MAGIC_OFFSET + len(_TAR_MAGIC_BYTES):
        tar_sig = data[
            _TAR_MAGIC_OFFSET : _TAR_MAGIC_OFFSET + len(_TAR_MAGIC_BYTES)
        ]
        if tar_sig in (_TAR_MAGIC_BYTES, _TAR_MAGIC_BYTES_ALT):
            return ArchiveType.TAR

    # Check for raw deflate/zlib (0x78 prefix)
    if data[0] == 0x78:
        return ArchiveType.DEFLATE

    return ArchiveType.UNKNOWN


def detect_archive_type_from_file(file_path: Path) -> ArchiveType:
    """
    Identify the archive format by reading magic bytes from a file.

    Why a separate function for files:
        - Avoids loading the entire file into memory for detection.
          Only the first 512 bytes are read.
        - Handles file-not-found and permission errors gracefully.
        - Returns ``UNKNOWN`` for empty files rather than raising.

    Parameters
    ----------
    file_path : Path
        Path to the file to inspect.

    Returns
    -------
    ArchiveType
        The detected format, or ``UNKNOWN`` if the file cannot be read
        or the format is not recognized.

    Examples
    --------
    >>> detect_archive_type_from_file(Path("package.zip"))
    <ArchiveType.ZIP: 1>

    >>> detect_archive_type_from_file(Path("nonexistent.file"))
    <ArchiveType.UNKNOWN: 8>
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(512)
        return detect_archive_type(header)
    except (OSError, FileNotFoundError):
        return ArchiveType.UNKNOWN


def get_archive_info(archive_type: ArchiveType) -> ArchiveInfo:
    """
    Return metadata about a detected archive type.

    Why a lookup function instead of attributes on ArchiveType:
        - ArchiveType is an enum; mixing format metadata with enumeration
          values violates separation of concerns.
        - Metadata can be updated without changing the enum definition.
        - Enables callers to get all information about a format in one
          structured object.

    Parameters
    ----------
    archive_type : ArchiveType
        The format to get information about.

    Returns
    -------
    ArchiveInfo
        Structured metadata for the format.

    Examples
    --------
    >>> info = get_archive_info(ArchiveType.GZIP)
    >>> info.is_compressed
    True
    >>> info.extension
    '.gz'
    """
    return _ARCHIVE_INFO_MAP.get(archive_type, _ARCHIVE_INFO_MAP[ArchiveType.UNKNOWN])


def get_compression_type(data: bytes) -> CompressionType:
    """
    Detect only the compression layer (not container) from raw bytes.

    Why separate from archive detection:
        - A file may be a compressed container (e.g., ``.tar.gz``).
          ``detect_archive_type`` returns ``GZIP`` for the outer layer;
          callers need to decompress, then detect again for the inner
          tar container.
        - This function identifies the compression algorithm to use for
          the first decompression step.

    Parameters
    ----------
    data : bytes
        Raw bytes from the beginning of the data.

    Returns
    -------
    CompressionType
        The detected compression format, or ``NONE`` if data appears
        uncompressed.

    Examples
    --------
    >>> get_compression_type(b"\\x1f\\x8b\\x08...")
    <CompressionType.GZIP: 1>

    >>> get_compression_type(b"#!/usr/bin/env python...")
    <CompressionType.NONE: 6>
    """
    archive_type = detect_archive_type(data)
    info = get_archive_info(archive_type)
    return info.compression_type


# ---------------------------------------------------------------------------
# Public API: Decompression
# ---------------------------------------------------------------------------


def decompress_bytes(
    data: bytes,
    compression_type: Optional[CompressionType] = None,
) -> bytes:
    """
    Decompress a byte stream using the appropriate algorithm.

    Why accept optional compression_type:
        - If known in advance (from Content-Encoding header, file extension),
          skip detection and go straight to decompression.
        - If ``None``, auto-detect from magic bytes. This is the common
          case for network responses where Content-Encoding may be missing
          or wrong.

    Parameters
    ----------
    data : bytes
        Compressed data to decompress.
    compression_type : Optional[CompressionType]
        The compression algorithm to use. If ``None``, auto-detect from
        the data's magic bytes.

    Returns
    -------
    bytes
        Decompressed data. If no compression is detected and no type is
        specified, the original data is returned unchanged.

    Raises
    ------
    ValueError
        If the specified compression_type does not match the data
        (e.g., data is not valid gzip but gzip was requested).
    OSError
        If decompression fails due to corrupted data.

    Examples
    --------
    >>> import gzip as gzip_module
    >>> original = b"Hello, World!"
    >>> compressed = gzip_module.compress(original)
    >>> decompress_bytes(compressed)
    b'Hello, World!'

    >>> decompress_bytes(compressed, CompressionType.GZIP)
    b'Hello, World!'

    >>> decompress_bytes(b"uncompressed data")
    b'uncompressed data'
    """
    if compression_type is None:
        compression_type = get_compression_type(data)

    if compression_type == CompressionType.GZIP:
        try:
            return gzip.decompress(data)
        except gzip.BadGzipFile as e:
            raise ValueError(f"Invalid gzip data: {e}") from e

    elif compression_type == CompressionType.BZIP2:
        try:
            return bz2.decompress(data)
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid bzip2 data: {e}") from e

    elif compression_type == CompressionType.XZ:
        try:
            return lzma.decompress(data)
        except lzma.LZMAError as e:
            raise ValueError(f"Invalid xz data: {e}") from e

    elif compression_type == CompressionType.LZMA:
        try:
            # Legacy lzma format may need different parameters
            return lzma.decompress(data, format=lzma.FORMAT_ALONE)
        except lzma.LZMAError:
            # Try without format specification
            try:
                return lzma.decompress(data)
            except lzma.LZMAError as e:
                raise ValueError(f"Invalid lzma data: {e}") from e

    elif compression_type == CompressionType.DEFLATE:
        import zlib

        try:
            return zlib.decompress(data)
        except zlib.error as e:
            # Try with -15 window bits (raw deflate without header)
            try:
                return zlib.decompress(data, -15)
            except zlib.error as e2:
                raise ValueError(f"Invalid deflate data: {e2}") from e2

    else:
        # No compression detected or NONE type
        return data


def decompress_file(
    file_path: Path,
    output_path: Optional[Path] = None,
) -> bytes:
    """
    Read and decompress a compressed file from disk.

    Why a file-specific function:
        - Avoids loading the entire compressed file into memory at once
          when the caller only needs the decompressed bytes.
        - Handles file-not-found and permission errors with clear messages.
        - Optionally writes decompressed data to a new file.

    Parameters
    ----------
    file_path : Path
        Path to the compressed file.
    output_path : Optional[Path]
        If provided, decompressed data is written to this path.
        If ``None``, data is returned as bytes only.

    Returns
    -------
    bytes
        Decompressed data.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.
    PermissionError
        If ``file_path`` cannot be read or ``output_path`` cannot be written.

    Examples
    --------
    >>> decompress_file(Path("data.gz"), Path("data.txt"))
    b'decompressed content...'
    """
    with open(file_path, "rb") as f:
        compressed = f.read()

    decompressed = decompress_bytes(compressed)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(decompressed)

    return decompressed


# ---------------------------------------------------------------------------
# Public API: Archive Extraction
# ---------------------------------------------------------------------------


def extract_to_dir(
    archive_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = True,
    preserve_permissions: bool = False,
) -> List[Path]:
    """
    Extract any supported archive to a directory.

    Why a single function for all formats:
        - Callers don't need to know the format in advance.
        - Handles nested compression transparently (e.g., ``.tar.gz``
          is decompressed as gzip, then extracted as tar).
        - Returns the list of extracted file paths for verification.

    Parameters
    ----------
    archive_path : Path
        Path to the archive file to extract.
    output_dir : Path
        Directory to extract files into. Created if it does not exist.
    overwrite : bool
        If ``True``, overwrite existing files. If ``False``, skip files
        that already exist. Default ``True``.
    preserve_permissions : bool
        If ``True``, restore file permissions from the archive.
        Requires appropriate OS support. Default ``False``.

    Returns
    -------
    List[Path]
        Absolute paths of all extracted files, sorted alphabetically.

    Raises
    ------
    FileNotFoundError
        If ``archive_path`` does not exist.
    ValueError
        If the archive format is not supported or the data is corrupted.
    OSError
        If extraction fails due to filesystem errors.

    Examples
    --------
    >>> files = extract_to_dir(
    ...     Path("downloads/pip-24.0.tar.gz"),
    ...     Path("output/pip-24.0"),
    ... )
    >>> len(files) > 0
    True
    >>> files[0].name
    'PKG-INFO'

    >>> # Extract wheel (ZIP format)
    >>> files = extract_to_dir(
    ...     Path("pip-24.0-py3-none-any.whl"),
    ...     Path("site-packages"),
    ... )
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    # Read the file
    with open(archive_path, "rb") as f:
        data = f.read()

    if len(data) == 0:
        raise ValueError(f"Archive is empty: {archive_path}")

    # Detect format
    archive_type = detect_archive_type(data)

    if archive_type == ArchiveType.UNKNOWN:
        raise ValueError(
            f"Cannot determine archive format for: {archive_path}. "
            f"First bytes: {data[:16].hex()}"
        )

    # Handle compressed formats: decompress first
    info = get_archive_info(archive_type)
    if info.is_compressed and not info.is_container:
        decompressed = decompress_bytes(data, info.compression_type)
        # Detect the inner format (e.g., tar inside gzip)
        inner_type = detect_archive_type(decompressed)
        if inner_type != ArchiveType.UNKNOWN:
            archive_type = inner_type
            data = decompressed

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract based on container type
    extracted_files: List[Path] = []

    if archive_type == ArchiveType.ZIP:
        extracted_files = _extract_zip(data, output_dir, overwrite)

    elif archive_type == ArchiveType.TAR:
        extracted_files = _extract_tar(
            data, output_dir, overwrite, preserve_permissions
        )

    elif archive_type == ArchiveType.GZIP:
        # Standalone gzip (not .tar.gz) — single file
        decompressed = decompress_bytes(data, CompressionType.GZIP)
        # Use the archive name minus .gz as the output filename
        output_name = archive_path.stem  # removes .gz
        if output_name.endswith(".tar"):
            # It was .tar.gz after all; extract tar
            extracted_files = _extract_tar(
                decompressed, output_dir, overwrite, preserve_permissions
            )
        else:
            output_file = output_dir / output_name
            if not overwrite and output_file.exists():
                pass
            else:
                output_file.write_bytes(decompressed)
            extracted_files = [output_file]

    elif archive_type == ArchiveType.BZIP2:
        decompressed = decompress_bytes(data, CompressionType.BZIP2)
        output_name = archive_path.stem
        if output_name.endswith(".tar"):
            extracted_files = _extract_tar(
                decompressed, output_dir, overwrite, preserve_permissions
            )
        else:
            output_file = output_dir / output_name
            if overwrite or not output_file.exists():
                output_file.write_bytes(decompressed)
            extracted_files = [output_file]

    elif archive_type == ArchiveType.XZ:
        decompressed = decompress_bytes(data, CompressionType.XZ)
        output_name = archive_path.stem
        if output_name.endswith(".tar"):
            extracted_files = _extract_tar(
                decompressed, output_dir, overwrite, preserve_permissions
            )
        else:
            output_file = output_dir / output_name
            if overwrite or not output_file.exists():
                output_file.write_bytes(decompressed)
            extracted_files = [output_file]

    else:
        raise ValueError(f"Unsupported archive type: {archive_type.name}")

    # Sort for consistent output
    extracted_files.sort()
    return extracted_files


def extract_wheel(
    wheel_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = True,
) -> List[Path]:
    """
    Extract a Python wheel file (.whl) to a directory.

    Why a dedicated wheel extraction function:
        - Wheels are ZIP archives but have a specific internal structure
          (``.dist-info`` directory, ``.data`` directory).
        - The function name signals intent clearly: "extract this wheel"
          is more readable than "extract this zip to site-packages".
        - Enables wheel-specific behavior in the future (e.g., running
          ``.data/scripts`` post-install hooks).

        Parameters
    ----------
    wheel_path : Path
        Path to the ``.whl`` file.
    output_dir : Path
        Directory to extract wheel contents into (typically site-packages).
    overwrite : bool
        If ``True``, overwrite existing files. Default ``True``.

    Returns
    -------
    List[Path]
        Absolute paths of all extracted files.

    Examples
    --------
    >>> files = extract_wheel(
    ...     Path("pip-24.0-py3-none-any.whl"),
    ...     Path("/usr/lib/python3.11/site-packages"),
    ... )
    >>> any("pip/__init__.py" in str(f) for f in files)
    True
    """
    return extract_to_dir(wheel_path, output_dir, overwrite=overwrite)


def extract_archive(
    data: bytes,
    output_dir: Path,
    *,
    archive_type: Optional[ArchiveType] = None,
    overwrite: bool = True,
) -> ExtractionResult:
    """
    Extract an archive from bytes (in-memory) to a directory.

    Why accept bytes instead of a file path:
        - Network downloads produce bytes directly; saving to a temporary
          file just to extract it is wasteful.
        - Enables extraction of archives embedded in other formats
          (e.g., a ZIP inside another ZIP).
        - Testing with synthetic archive data doesn't require filesystem
          setup.

    Parameters
    ----------
    data : bytes
        The archive contents as bytes.
    output_dir : Path
        Directory to extract files into.
    archive_type : Optional[ArchiveType]
        The format of the archive. If ``None``, auto-detect from magic bytes.
    overwrite : bool
        If ``True``, overwrite existing files. Default ``True``.

    Returns
    -------
    ExtractionResult
        Structured result with success status, list of extracted files,
        and any warnings or errors.

    Examples
    --------
    >>> import zipfile as zf
    >>> import io
    >>> buf = io.BytesIO()
    >>> with zf.ZipFile(buf, 'w') as z:
    ...     z.writestr("test.txt", "content")
    >>> result = extract_archive(buf.getvalue(), Path("/tmp/extract_test"))
    >>> result.success
    True
    >>> len(result.files_extracted)
    1
    """
    warnings: List[str] = []
    errors: List[str] = []

    if archive_type is None:
        archive_type = detect_archive_type(data)

    if archive_type == ArchiveType.UNKNOWN:
        return ExtractionResult(
            success=False,
            archive_type=archive_type,
            errors=["Cannot determine archive format"],
        )

    try:
        # Handle compression layer first
        info = get_archive_info(archive_type)
        bytes_decompressed = 0

        if info.is_compressed and not info.is_container:
            original_size = len(data)
            data = decompress_bytes(data, info.compression_type)
            bytes_decompressed = len(data)
            # Detect inner format
            inner_type = detect_archive_type(data)
            if inner_type != ArchiveType.UNKNOWN:
                archive_type = inner_type

        # Extract
        files: List[Path] = []
        if archive_type == ArchiveType.ZIP:
            files = _extract_zip(data, output_dir, overwrite)
        elif archive_type == ArchiveType.TAR:
            files = _extract_tar(data, output_dir, overwrite, False)
        else:
            # Single compressed file with no container
            output_file = output_dir / "decompressed_output"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not output_file.exists():
                output_file.write_bytes(data)
            files = [output_file]

        return ExtractionResult(
            success=True,
            archive_type=archive_type,
            files_extracted=files,
            bytes_decompressed=bytes_decompressed,
            warnings=warnings,
        )

    except Exception as e:
        errors.append(str(e))
        return ExtractionResult(
            success=False,
            archive_type=archive_type,
            errors=errors,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Private: Format-Specific Extraction
# ---------------------------------------------------------------------------


def _extract_zip(
    data: bytes,
    output_dir: Path,
    overwrite: bool,
) -> List[Path]:
    """
    Extract a ZIP archive from bytes to a directory.

    Why private:
        - Called only by ``extract_to_dir`` and ``extract_archive``.
        - Callers should use the public API, not format-specific functions.
    """
    extracted: List[Path] = []

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            # Skip directories (they're created automatically)
            if member.endswith("/") or member.endswith("\\"):
                continue

            dest_path = output_dir / member

            # Skip if exists and overwrite is False
            if not overwrite and dest_path.exists():
                extracted.append(dest_path)
                continue

            # Create parent directories
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract file
            with zf.open(member) as source:
                dest_path.write_bytes(source.read())

            extracted.append(dest_path)

    return extracted


def _extract_tar(
    data: bytes,
    output_dir: Path,
    overwrite: bool,
    preserve_permissions: bool,
) -> List[Path]:
    """
    Extract a tar archive from bytes to a directory.

    Why private:
        - Called only by ``extract_to_dir`` and ``extract_archive``.
        - Callers should use the public API, not format-specific functions.
    """
    extracted: List[Path] = []

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
        for member in tf.getmembers():
            # Skip directories
            if member.isdir():
                continue

            dest_path = output_dir / member.name

            # Security check: prevent path traversal
            # Why: malicious archives may contain paths like ../../etc/passwd
            resolved = dest_path.resolve()
            if not str(resolved).startswith(str(output_dir.resolve())):
                continue

            # Skip if exists and overwrite is False
            if not overwrite and dest_path.exists():
                extracted.append(dest_path)
                continue

            # Create parent directories
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract file
            with tf.extractfile(member) as source:
                if source is not None:
                    dest_path.write_bytes(source.read())

            # Restore permissions if requested
            if preserve_permissions and hasattr(member, "mode"):
                dest_path.chmod(member.mode)

            extracted.append(dest_path)

    return extracted


# ---------------------------------------------------------------------------
# Public API: Convenience Functions
# ---------------------------------------------------------------------------


def is_compressed(data: bytes) -> bool:
    """
    Check if data appears to be compressed.

    Why this function:
        - Quick check before attempting decompression.
        - Avoids the overhead of detecting the exact format when a
          simple yes/no answer is sufficient.

    Parameters
    ----------
    data : bytes
        Raw bytes to check.

    Returns
    -------
    bool
        ``True`` if the data's magic bytes indicate compression.

    Examples
    --------
    >>> is_compressed(b"\\x1f\\x8b\\x08...")  # gzip
    True
    >>> is_compressed(b"#!/usr/bin/env python...")
    False
    """
    ctype = get_compression_type(data)
    return ctype != CompressionType.NONE


def is_archive(data: bytes) -> bool:
    """
    Check if data appears to be an archive (container or compressed).

    Why separate from is_compressed:
        - ``is_compressed`` returns ``False`` for ZIP files (they're
          containers, not compression streams).
        - ``is_archive`` returns ``True`` for both containers and
          compression streams.
        - Callers wanting "can I extract files from this?" use
          ``is_archive``. Callers wanting "should I decompress this?"
          use ``is_compressed``.

    Parameters
    ----------
    data : bytes
        Raw bytes to check.

    Returns
    -------
    bool
        ``True`` if the data is any recognized archive or compression format.

    Examples
    --------
    >>> is_archive(b"PK\\x03\\x04...")  # ZIP
    True
    >>> is_archive(b"\\x1f\\x8b\\x08...")  # gzip
    True
    >>> is_archive(b"plain text")
    False
    """
    return detect_archive_type(data) != ArchiveType.UNKNOWN


def get_format_description(data: bytes) -> str:
    """
    Return a human-readable description of the detected format.

    Why this function:
        - Error messages and logs benefit from user-friendly format names.
        - "gzip compressed data" is clearer than ``<ArchiveType.GZIP: 2>``.
        - Helps users understand what went wrong when a format is
          unsupported.

    Parameters
    ----------
    data : bytes
        Raw bytes to identify.

    Returns
    -------
    str
        Human-readable format description.

    Examples
    --------
    >>> get_format_description(b"PK\\x03\\x04...")
    'ZIP archive'
    >>> get_format_description(b"\\x1f\\x8b\\x08...")
    'gzip compressed data'
    >>> get_format_description(b"unknown")
    'unknown format'
    """
    archive_type = detect_archive_type(data)

    descriptions = {
        ArchiveType.ZIP: "ZIP archive",
        ArchiveType.GZIP: "gzip compressed data",
        ArchiveType.BZIP2: "bzip2 compressed data",
        ArchiveType.XZ: "xz compressed data",
        ArchiveType.LZMA: "lzma compressed data",
        ArchiveType.TAR: "tar archive",
        ArchiveType.DEFLATE: "deflate/zlib compressed data",
        ArchiveType.UNKNOWN: "unknown format",
    }

    return descriptions.get(archive_type, "unrecognized format")
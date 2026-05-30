"""
Compiler output parsers and text extraction utilities.

This module handles ALL text parsing of compiler version outputs.
It receives raw strings from the validation layer and returns
structured data. It NEVER executes subprocesses or touches the
filesystem — that responsibility belongs exclusively to
`validation.py`.

Scope
-----
- Extract version numbers from --version output
- Identify vendor and compiler family from text patterns
- Normalize target triplets from -dumpmachine output
- Provide comparison utilities for version strings

This module does NOT:
    - Execute any binaries (use validation.py)
    - Discover compiler paths (use strategies.py)
    - Make priority decisions (use scoring.py)

Usage
-----
    from .parsers import parse_version_output, parse_target_triplet

    version_str = parse_version_output(raw_text, CompilerKind.GCC)
    target = parse_target_triplet("-dumpmachine output here")

Warnings
--------
- Version strings are NOT guaranteed to be valid SemVer.
  GCC uses "major.minor.patch", Clang uses "major.minor",
  MSVC uses "major.minor.build".
- Apple Clang versions are deliberately misleading:
  Apple Clang 15.0.0 corresponds to LLVM 16.0.0 upstream.
  Compare by feature inference, not raw version numbers.
- MinGW-w64 GCC wraps standard GCC; the version output
  may contain both "gcc" and "mingw" markers.

User Instructions
-----------------
- Use `version_tuple()` for numeric comparisons, never string comparison.
- Use `normalize_vendor()` to standardize vendor strings across sources.
- For Apple Clang → upstream mapping, see Apple's documentation.
"""

import re
from typing import Optional, Tuple, List

from .models import CompilerKind


# ---------------------------------------------------------------------------
# Version parsing
# ---------------------------------------------------------------------------

def parse_version_output(raw_output: str, kind: CompilerKind) -> str:
    """
    Extract version string from a compiler's --version output.

    Uses the first line only. Scans tokens for the first one that
    starts with a digit or contains a digit-dot pattern.

    Parameters
    ----------
    raw_output : str
        Raw stdout from `compiler --version`.
    kind : CompilerKind
        Compiler family hint. Used only for fallback parsing strategies.
        GCC output: "gcc (GCC) 13.2.0"
        Clang output: "clang version 17.0.6"
        MSVC output: "Microsoft (R) C/C++ Optimizing Compiler Version 19.38.33130"

    Returns
    -------
    str
        Version string. Returns "0.0.0" if no version could be extracted.

    Notes
    -----
    This is intentionally simple. Complex version extraction belongs
    in vendor-specific functions, not here.
    """
    if not raw_output or not raw_output.strip():
        return "0.0.0"

    lines = raw_output.splitlines()
    first_line = lines[0].strip()

    # Try vendor-specific extraction first
    if kind == CompilerKind.CLANG:
        version = _extract_clang_version(first_line)
        if version != "0.0.0":
            return version

    if kind == CompilerKind.MSVC:
        version = _extract_msvc_version(first_line)
        if version != "0.0.0":
            return version

    if kind == CompilerKind.GCC:
        version = _extract_gcc_version(first_line)
        if version != "0.0.0":
            return version

    # Generic fallback: scan tokens for version-like pattern
    return _extract_version_generic(first_line)


def _extract_gcc_version(line: str) -> str:
    """
    Extract version from GCC-style output.

    Patterns handled:
        "gcc (GCC) 13.2.0"
        "gcc (MinGW-W64 x86_64-posix-seh) 13.2.0"
        "gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"

    Returns
    -------
    str
        Version or "0.0.0".
    """
    match = re.search(r"(\d+\.\d+\.\d+)", line)
    if match:
        return match.group(1)
    match = re.search(r"(\d+\.\d+)", line)
    if match:
        return match.group(1)
    return "0.0.0"


def _extract_clang_version(line: str) -> str:
    """
    Extract version from Clang-style output.

    Patterns handled:
        "clang version 17.0.6"
        "Apple clang version 15.0.0 (clang-1500.3.9.4)"
        "Ubuntu clang version 14.0.0-1ubuntu1.1"

    Returns
    -------
    str
        Version or "0.0.0". For Apple Clang, returns the Apple version
        (e.g., "15.0.0"), NOT the upstream LLVM version.
    """
    match = re.search(r"version\s+(\d+\.\d+\.\d+)", line)
    if match:
        return match.group(1)
    match = re.search(r"version\s+(\d+\.\d+)", line)
    if match:
        return match.group(1)
    return "0.0.0"


def _extract_msvc_version(line: str) -> str:
    """
    Extract version from MSVC-style output.

    Pattern handled:
        "Microsoft (R) C/C++ Optimizing Compiler Version 19.38.33130 for x64"

    Returns
    -------
    str
        Version string like "19.38.33130" or "0.0.0".
    """
    match = re.search(r"Version\s+(\d+\.\d+\.\d+)", line, re.IGNORECASE)
    if match:
        return match.group(1)
    return "0.0.0"


def _extract_version_generic(line: str) -> str:
    """
    Fallback version extraction when compiler kind is unknown.

    Scans all tokens for the first one matching a version-like pattern.

    Parameters
    ----------
    line : str
        First line of --version output.

    Returns
    -------
    str
        Version or "0.0.0".
    """
    tokens = line.split()
    for token in tokens:
        cleaned = token.strip("(),;:")
        if re.match(r"^\d+\.\d+", cleaned):
            return cleaned
    return "0.0.0"


# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------

def version_tuple(version_str: str) -> Tuple[int, ...]:
    """
    Convert a version string to a tuple for numeric comparison.

    Parameters
    ----------
    version_str : str
        Version string like "13.2.0" or "17.0".

    Returns
    -------
    Tuple[int, ...]
        Numeric tuple, e.g., (13, 2, 0).

    Warnings
    --------
    - Missing segments are treated as 0: "13" → (13, 0, 0)
    - Non-numeric segments are silently dropped.
    - Do NOT compare tuples from different vendors (GCC vs Clang).
      Version numbers are not comparable across compiler families.

    Examples
    --------
    >>> version_tuple("13.2.0")
    (13, 2, 0)
    >>> version_tuple("17")
    (17, 0, 0)
    >>> version_tuple("13.2.0") > version_tuple("13.1.0")
    True
    """
    parts = version_str.split(".")
    result: List[int] = []
    for part in parts:
        digits = "".join(c for c in part if c.isdigit())
        if digits:
            result.append(int(digits))
    while len(result) < 3:
        result.append(0)
    return tuple(result)


# ---------------------------------------------------------------------------
# Vendor and kind parsing
# ---------------------------------------------------------------------------

def parse_kind_and_vendor(raw_output: str) -> Tuple[CompilerKind, str]:
    """
    Determine compiler family and vendor from --version output.

    Uses case-insensitive substring matching. Order is critical:
    Clang is checked before GCC because some Clang builds include
    "gcc" in their compatibility output.

    Parameters
    ----------
    raw_output : str
        Full --version stdout.

    Returns
    -------
    Tuple[CompilerKind, str]
        (kind, vendor).
        kind: CompilerKind enum value.
        vendor: "GNU", "Apple", "LLVM", "Microsoft", "MinGW", or "unknown".
    """
    lower = raw_output.lower()

    if "clang" in lower:
        if "apple" in lower:
            return CompilerKind.CLANG, "Apple"
        return CompilerKind.CLANG, "LLVM"

    if "gcc" in lower or "gnu" in lower:
        if "mingw" in lower or "mingw64" in lower:
            return CompilerKind.GCC, "MinGW"
        return CompilerKind.GCC, "GNU"

    if any(keyword in lower for keyword in
           ("microsoft", "visual c++", "optimizing compiler")):
        return CompilerKind.MSVC, "Microsoft"

    return CompilerKind.UNKNOWN, "unknown"


def normalize_vendor(vendor: str) -> str:
    """
    Normalize vendor string to a standard form.

    Handles common variations seen in the wild.

    Parameters
    ----------
    vendor : str
        Raw vendor string.

    Returns
    -------
    str
        One of: "GNU", "Apple", "LLVM", "Microsoft", "MinGW", "unknown".
    """
    mapping = {
        "gnu": "GNU",
        "gcc": "GNU",
        "apple": "Apple",
        "llvm": "LLVM",
        "clang": "LLVM",
        "microsoft": "Microsoft",
        "msvc": "Microsoft",
        "mingw": "MinGW",
        "mingw64": "MinGW",
        "mingw-w64": "MinGW",
    }
    return mapping.get(vendor.lower(), "unknown")


# ---------------------------------------------------------------------------
# Target triplet parsing
# ---------------------------------------------------------------------------

def parse_target_triplet(raw_output: Optional[str]) -> str:
    """
    Parse and normalize a target triplet from -dumpmachine output.

    Parameters
    ----------
    raw_output : Optional[str]
        Raw stdout from `compiler -dumpmachine`, or None.

    Returns
    -------
    str
        Normalized target triplet like "x86_64-linux-gnu".
        Returns empty string if input is None or empty.

    Notes
    -----
    Takes only the first line of output.
    Strips trailing newlines, carriage returns, and whitespace.
    Does NOT validate the triplet structure (arch-vendor-os).
    """
    if not raw_output:
        return ""

    first_line = raw_output.strip().splitlines()[0].strip()
    return first_line


def split_target_triplet(triplet: str) -> Tuple[str, str, str]:
    """
    Split a target triplet into architecture, vendor, and OS components.

    Parameters
    ----------
    triplet : str
        Target triplet like "x86_64-linux-gnu" or "aarch64-linux-android".

    Returns
    -------
    Tuple[str, str, str]
        (arch, vendor, os). Missing components are empty strings.

    Examples
    --------
    >>> split_target_triplet("x86_64-linux-gnu")
    ("x86_64", "linux", "gnu")
    >>> split_target_triplet("arm-none-eabi")
    ("arm", "none", "eabi")
    >>> split_target_triplet("x86_64")
    ("x86_64", "", "")
    """
    if not triplet:
        return ("", "", "")

    parts = triplet.split("-")
    if len(parts) == 1:
        return (parts[0], "", "")
    if len(parts) == 2:
        return (parts[0], parts[1], "")
    return (parts[0], parts[1], "-".join(parts[2:]))


# ---------------------------------------------------------------------------
# Output normalization
# ---------------------------------------------------------------------------

def normalize_output(raw_output: str) -> str:
    """
    Normalize raw compiler output for consistent parsing.

    Handles common encoding and formatting issues:
        - Non-ASCII characters replaced
        - Carriage returns stripped
        - Leading/trailing whitespace collapsed
        - Multiple spaces collapsed to single space

    Parameters
    ----------
    raw_output : str
        Raw bytes decoded as UTF-8.

    Returns
    -------
    str
        Cleaned and normalized text.
    """
    cleaned = raw_output.encode("ascii", errors="replace").decode("ascii")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in cleaned.splitlines()]
    return "\n".join(lines)
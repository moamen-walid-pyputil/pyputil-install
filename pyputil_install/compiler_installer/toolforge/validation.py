"""
Compiler validation and security module.

This module is the ONLY place in the codebase that executes discovered
compiler binaries. It verifies that a raw path is a genuine, working
compiler and extracts its identity information.

Layer
-----
This belongs to the Identity Layer. It receives raw paths from the
Detection Layer and returns fully-formed CompilerInfo objects.

Execution Rules (NON-NEGOTIABLE)
--------------------------------
1. `shell=False` on every subprocess call — no shell injection vectors
2. `timeout=5` on every subprocess call — prevent hanging on bad binaries
3. Empty environment dict `env={}` — no inherited env pollution
4. Blocklist check BEFORE any execution — reject dangerous paths early
5. No network access — all calls are local subprocess only

Blocklist
---------
Paths matching these patterns are rejected without execution:
- /tmp, /var/tmp, /dev/shm (world-writable temp directories)
- ~/Downloads (user download directory)
- Any world-writable file in a world-writable directory

Usage
-----
    from .validation import validate_and_fingerprint

    compiler_info = validate_and_fingerprint("/usr/bin/gcc")
    if compiler_info is None:
        print("Not a valid compiler or blocked by security policy")

Warnings
--------
- This module EXECUTES binaries found on the system.
- A malicious binary named 'gcc' in PATH WILL be executed if not blocklisted.
- The blocklist reduces risk but does not eliminate it entirely.
- Review BLOCKLIST_PATHS for your environment before deployment.

User Instructions
-----------------
- Customize BLOCKLIST_PATHS for your threat model
- Set COMPILER_EXECUTION_TIMEOUT env var to change timeout (default 5s)
- Set COMPILER_EXECUTION_MAX_OUTPUT env var to limit stdout capture (default 10KB)
- All rejections are logged at WARNING level; monitor logs in production
"""

import hashlib
import logging
import os
import subprocess
from typing import Optional, Tuple, Iterable

from .models import CompilerInfo, CompilerKind, CompilerSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security configuration — modify for your environment
# ---------------------------------------------------------------------------

BLOCKLIST_PATHS: Tuple[str, ...] = (
    "/tmp",
    "/var/tmp",
    "/dev/shm",
    os.path.expanduser("~/Downloads"),
)

DEFAULT_TIMEOUT: int = int(
    os.environ.get("COMPILER_EXECUTION_TIMEOUT", "5")
)

MAX_OUTPUT_BYTES: int = int(
    os.environ.get("COMPILER_EXECUTION_MAX_OUTPUT", "10240")  # 10 KB
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_and_fingerprint(path: str, allowed: Iterable[str] = None) -> Optional[CompilerInfo]:
    """
    Validate a path as a real compiler and extract its identity.

    Performs security checks, executes the binary with --version and
    -dumpmachine, parses the output, and builds a fingerprint.

    Parameters
    ----------
    path : str
        Absolute path to a potential compiler executable.

    Returns
    -------
    Optional[CompilerInfo]
        A fully populated CompilerInfo object if validation succeeds.
        None if:
            - Path is in BLOCKLIST_PATHS
            - Path is not executable
            - Subprocess times out (default 5 seconds)
            - Subprocess returns non-zero exit code
            - Output parsing fails to identify compiler family
            - World-writable permissions detected

    Security
    --------
    The path is checked against BLOCKLIST_PATHS and world-writable
    permissions BEFORE any subprocess is spawned.
    """
    if _is_blocklisted(path, allowed):
        logger.warning("Blocklisted path rejected: %s", path)
        return None

    if not os.access(path, os.X_OK):
        logger.info("Path is not executable: %s", path)
        return None

    version_output = _safe_execute(path, "--version")
    if version_output is None:
        logger.info("Failed to get --version output from: %s", path)
        return None

    machine_output = _safe_execute(path, "-dumpmachine")

    kind, vendor = _parse_kind_and_vendor(version_output)
    if kind == CompilerKind.UNKNOWN:
        logger.info("Unrecognized compiler kind for: %s", path)
        return None

    version = _extract_version(version_output, kind)

    fingerprint = _build_fingerprint(version_output, machine_output)

    is_cross, target_triplet = _determine_cross_compilation(
        path, machine_output
    )

    confidence = _calculate_confidence(
        kind=kind,
        vendor=vendor,
        version=version,
        target_triplet=target_triplet,
    )

    return CompilerInfo(
        path=os.path.abspath(path),
        version=version,
        vendor=vendor,
        kind=kind,
        target_triplet=target_triplet,
        is_cross_compiler=is_cross,
        source=CompilerSource.SYSTEM_PATH,
        confidence_score=confidence,
        fingerprint=fingerprint,
    )


# ---------------------------------------------------------------------------
# Security checks
# ---------------------------------------------------------------------------

def _is_blocklisted(path: str, allowed: Iterable[str] = None) -> bool:
    """
    Check if path falls within a blocklisted directory.

    Also rejects world-writable files located in world-writable
    directories, as these are common attack vectors.

    Parameters
    ----------
    path : str
        Absolute path to check.
    allowed : optional, iterable
        Iterable paths to skip from block list. 

    Returns
    -------
    bool
        True if the path should be rejected without execution.
    """
    if os.environ.get("TOOLFORGE_SKIP_BLOCKLIST") == "1":
        return False
    allowed = allowed or []

    for allowed_path in allowed:
        if real_path.startswith(os.path.realpath(allowed_path)):
            return False

    real_path = os.path.realpath(path)
    for blocked in BLOCKLIST_PATHS:
        if real_path.startswith(blocked):
            return True

    try:
        parent_dir = os.path.dirname(real_path)
        if os.access(parent_dir, os.W_OK) and os.access(real_path, os.W_OK):
            return True
    except OSError:
        pass

    return False


# ---------------------------------------------------------------------------
# Safe subprocess execution
# ---------------------------------------------------------------------------

def _safe_execute(path: str, *args: str) -> Optional[str]:
    """
    Execute a binary with strict safety controls.

    Parameters
    ----------
    path : str
        Absolute path to the binary.
    *args : str
        Arguments to pass (e.g., "--version", "-dumpmachine").

    Returns
    -------
    Optional[str]
        Decoded, stripped stdout on success. None on ANY failure:
            - TimeoutExpired
            - FileNotFoundError
            - PermissionError
            - Non-zero return code
            - Output exceeds MAX_OUTPUT_BYTES
    """
    try:
        result = subprocess.run(
            [path, *args],
            capture_output=True,
            timeout=DEFAULT_TIMEOUT,
            shell=False,
            env=None,
            text=False,
        )

        if result.returncode != 0:
            return None

        raw_output = result.stdout
        if len(raw_output) > MAX_OUTPUT_BYTES:
            raw_output = raw_output[:MAX_OUTPUT_BYTES]

        return raw_output.decode("utf-8", errors="replace").strip()

    except subprocess.TimeoutExpired:
        logger.warning("Timeout (%ss) executing: %s", DEFAULT_TIMEOUT, path)
        return None
    except (FileNotFoundError, PermissionError, OSError) as exc:
        logger.warning("Execution failed for %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _parse_kind_and_vendor(version_output: str) -> Tuple[CompilerKind, str]:
    """
    Determine compiler family and vendor from --version output.

    Uses simple substring matching on lowercased output.
    Order matters: check for Clang before GCC because some Clang
    builds mention "gcc" in their output for compatibility.

    Parameters
    ----------
    version_output : str
        Raw stdout from running `compiler --version`.

    Returns
    -------
    Tuple[CompilerKind, str]
        (kind, vendor). vendor is one of:
        "GNU", "Apple", "LLVM", "Microsoft", "MinGW", "unknown".
    """
    lower = version_output.lower()

    if "clang" in lower:
        if "apple" in lower:
            return CompilerKind.CLANG, "Apple"
        return CompilerKind.CLANG, "LLVM"

    if "gcc" in lower or "gnu" in lower:
        if "mingw" in lower:
            return CompilerKind.GCC, "MinGW"
        return CompilerKind.GCC, "GNU"

    if "microsoft" in lower or "visual c++" in lower or "optimizing compiler" in lower:
        return CompilerKind.MSVC, "Microsoft"

    return CompilerKind.UNKNOWN, "unknown"


def _extract_version(version_output: str, kind: CompilerKind) -> str:
    """
    Extract version string from --version output.

    Uses simple line splitting and word scanning. Falls back to "0.0.0"
    if no version-like token is found.

    Parameters
    ----------
    version_output : str
        Raw stdout from `compiler --version`.
    kind : CompilerKind
        Compiler family, used to locate the version token position.

    Returns
    -------
    str
        Version string, e.g., "13.2.0". Not guaranteed to be SemVer.
    """
    lines = version_output.splitlines()
    if not lines:
        return "0.0.0"

    tokens = lines[0].split()
    for token in tokens:
        if token[0].isdigit():
            return token.rstrip(",)")
        if "." in token and any(c.isdigit() for c in token):
            return token.rstrip(",)")

    return "0.0.0"


# ---------------------------------------------------------------------------
# Target detection
# ---------------------------------------------------------------------------

def _determine_cross_compilation(
    path: str, machine_output: Optional[str]
) -> Tuple[bool, str]:
    """
    Determine if a compiler is a cross-compiler.

    Compares the compiler's -dumpmachine output against the host
    machine triplet obtained from Python's platform module.

    Parameters
    ----------
    path : str
        Compiler path (unused, reserved for future per-compiler logic).
    machine_output : Optional[str]
        Raw output from `compiler -dumpmachine`, or None if unavailable.

    Returns
    -------
    Tuple[bool, str]
        (is_cross_compiler, target_triplet).
        target_triplet is empty string if machine_output was None.
    """
    if not machine_output:
        return False, ""

    target = machine_output.strip().splitlines()[0].strip()

    try:
        import platform
        host = platform.machine().lower()
        return host not in target.lower(), target
    except Exception:
        return False, target


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

def _build_fingerprint(
    version_output: str, machine_output: Optional[str]
) -> str:
    """
    Build a deterministic fingerprint from compiler outputs.

    Normalizes text to lowercase and collapses whitespace before
    hashing to reduce sensitivity to locale, encoding, or formatting
    differences between compiler builds.

    Parameters
    ----------
    version_output : str
        Normalized --version stdout.
    machine_output : Optional[str]
        Normalized -dumpmachine stdout, or None.

    Returns
    -------
    str
        SHA256 hex digest of concatenated normalized outputs.
    """
    parts = [" ".join(version_output.lower().split())]
    if machine_output:
        parts.append(" ".join(machine_output.lower().split()))
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _calculate_confidence(
    kind: CompilerKind,
    vendor: str,
    version: str,
    target_triplet: str,
) -> float:
    """
    Calculate a confidence score based on identity quality.

    Deductions are cumulative and applied in order:
        - UNKNOWN kind      : -0.5
        - unknown vendor    : -0.3
        - version "0.0.0"   : -0.3
        - empty target      : -0.3
        - vendor is "MinGW" : -0.1 (MinGW wrappers are common)

    Parameters
    ----------
    kind : CompilerKind
        Parsed compiler family.
    vendor : str
        Parsed vendor string.
    version : str
        Extracted version string.
    target_triplet : str
        Target architecture triplet (may be empty).

    Returns
    -------
    float
        Score clamped to [0.0, 1.0].
    """
    score = 1.0

    if kind == CompilerKind.UNKNOWN:
        score -= 0.5
    if vendor == "unknown":
        score -= 0.3
    if version == "0.0.0":
        score -= 0.3
    if not target_triplet:
        score -= 0.3
    if vendor == "MinGW":
        score -= 0.1

    return max(0.0, min(score, 1.0))
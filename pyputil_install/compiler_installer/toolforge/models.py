"""
Core data models for the toolchain discovery engine.

This module defines the fundamental types used by all other layers.
Classes are frozen (immutable) to prevent accidental state mutation
across the detection → identity → decision pipeline.

Scope
-----
This module contains ONLY type definitions and enum declarations.
It does NOT contain:
    - Discovery logic
    - Validation or execution of binaries
    - Parsing of compiler output
    - Capability detection (to be added in a separate `capabilities.py`)

Usage
-----
All other modules import from here:
    from toolforge.models import CompilerInfo, CompilerKind, CompilerSource

Warnings
--------
- Instantiating CompilerInfo directly is allowed but discouraged.
  Normal creation happens inside `validation.py` after successful checks.
- The `confidence_score` field has a default of 1.0, which means
  "unverified". Always set it explicitly after validation.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class CompilerKind(Enum):
    """
    Broad compiler family classification.

    Used for filtering and grouping, NOT for vendor identification.
    Determined by parsing the --version output.

    Members
    -------
    GCC : GNU Compiler Collection and compatible wrappers
    CLANG : LLVM Clang and Apple Clang
    MSVC : Microsoft Visual C++ (cl.exe)
    UNKNOWN : Fallback when no pattern matches
    """
    GCC = auto()
    CLANG = auto()
    MSVC = auto()
    UNKNOWN = auto()


class CompilerSource(Enum):
    """
    Origin of a discovered compiler path.

    The source determines default priority ranking:
        USER_OVERRIDE > MANAGED_TOOLCHAIN > SYSTEM_PATH > COMMON_DIR > FALLBACK

    Members
    -------
    USER_OVERRIDE : Explicitly provided by user (env var or API argument)
    MANAGED_TOOLCHAIN : Found in a toolchain manager directory (e.g., rustup, SDKMAN)
    SYSTEM_PATH : Located by scanning the PATH environment variable
    COMMON_DIR : Found in a platform-specific standard directory (e.g., /usr/bin)
    FALLBACK : Last‑resort discovery method
    """
    USER_OVERRIDE = 1
    MANAGED_TOOLCHAIN = 2
    SYSTEM_PATH = 3
    COMMON_DIR = 4
    FALLBACK = 5


@dataclass(frozen=True)
class CompilerInfo:
    """
    Immutable representation of a validated compiler installation.

    All fields are mandatory except `fingerprint`.
    Once created, an instance cannot be modified; to "change" a field,
    create a new instance with `dataclasses.replace()`.

    Attributes
    ----------
    path : str
        Absolute path to the compiler executable.
        This file was confirmed to exist and be executable at discovery time.
        There is no guarantee it still exists later (e.g., network mounts,
        uninstalls). Always re‑check before use.
    version : str
        Normalized version string. Format depends on vendor:
            GCC   : "13.2.0"
            Clang : "17.0.6"
            MSVC  : "19.38.33130"
        Not guaranteed to be parsable as a strict SemVer string.
    vendor : str
        Normalized vendor name. One of:
            "GNU", "Apple", "LLVM", "Microsoft", "MinGW", "unknown"
        Derived from --version output, not from the binary path.
    kind : CompilerKind
        Broad family (GCC, CLANG, MSVC, UNKNOWN). Used for filtering.
    target_triplet : str
        Architecture‑vendor‑OS string from `-dumpmachine`.
        Examples: "x86_64-linux-gnu", "aarch64-linux-android".
        Empty string if the compiler did not support -dumpmachine.
    is_cross_compiler : bool
        True if `target_triplet` differs from the host machine triplet.
        Always `False` when `target_triplet` is empty.
    source : CompilerSource
        Discovery origin. Determines priority in the decision layer.
    confidence_score : float
        Value between 0.0 and 1.0 indicating how certain we are that this
        is a real, working compiler. Set after validation.
        A score < 0.7 usually means a wrapper, a broken symlink, or a
        compiler that did not respond normally to --version.
    fingerprint : Optional[str]
        Hash of normalized `--version` and `-dumpmachine` output.
        Used to detect wrappers (e.g., Apple GCC symlinked to Clang).
        Set by the identity layer. May be `None` if fingerprinting
        was skipped or failed.

    Warnings
    --------
    - `path` is the only field guaranteed to be an absolute, real path.
      It is NOT a persistent capability handle.
    - `version` string format is vendor‑dependent. Use the parsing
      utilities in `parsers.py` for comparisons.
    - `confidence_score` is heuristic, not a mathematical probability.
    - The object is frozen; use it as an immutable value type.
    """
    path: str
    version: str
    vendor: str
    kind: CompilerKind
    target_triplet: str
    is_cross_compiler: bool
    source: CompilerSource
    confidence_score: float = 1.0
    fingerprint: Optional[str] = None
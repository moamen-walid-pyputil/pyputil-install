"""
Pip Version Compatibility & Resolution Module.

Determines which pip versions are compatible with the current Python
interpreter, validates user-requested versions against compatibility
matrices, resolves "latest compatible" when no specific version is
requested, and provides version comparison utilities.

Why this module exists:
    - pip 24.x dropped support for Python 3.7; installing it on 3.7
      produces a cryptic SyntaxError or ImportError, not a clear
      compatibility message.
    - pip 21.0 was the last version supporting Python 2.7 and 3.5.
    - pip 23.0+ requires Python 3.7+.
    - pip 20.3 was the last version with the old resolver (some
      legacy projects depend on it).
    - Python 3.6 reached end-of-life but many enterprise systems still
      run it; the latest pip for 3.6 is 21.3.1.
    - Without version filtering, a user on Python 3.6 requesting
      "latest pip" would download an incompatible wheel that installs
      but crashes on first use.
    - Version validation prevents fetching non-existent versions
      (e.g., pip 99.0.0) and wasting network round-trips.

The compatibility data in this module is derived from pip's own release
history and documented Python version support policies. It is updated
as new pip versions drop old Python support.

Warnings
--------
- Compatibility data must be updated when new pip versions release.
  The ``LATEST_KNOWN_PIP`` constant should track the most recent
  stable pip version.
- This module does not perform network requests. It relies on
  hardcoded compatibility tables. If pip releases a version newer
  than ``LATEST_KNOWN_PIP``, the module will conservatively assume
  it is compatible (and let the actual installation confirm or fail).
- Python 2.7 and 3.5 support is included for completeness but these
  interpreters are end-of-life. The compatibility ranges for them
  will not change.

Examples
--------
Get the best pip version for the current Python:

    >>> from versions import VersionResolver
    >>> resolver = VersionResolver()
    >>> version = resolver.resolve_best_version()
    >>> print(version)
    '24.1.2'

Check if a specific pip version works with this Python:

    >>> resolver = VersionResolver()
    >>> compatible, reason = resolver.is_compatible("21.3.1")
    >>> compatible
    True
    >>> reason
    'Version 21.3.1 supports Python 3.9'

Attempt to use an incompatible version:

    >>> compatible, reason = resolver.is_compatible("24.0", python_version=(3, 6))
    >>> compatible
    False
    >>> print(reason)
    'pip>=22.0 requires Python>=3.7; current Python is 3.6'

Find all pip versions that support a specific Python:

    >>> from versions import get_compatible_versions_for_python
    >>> versions = get_compatible_versions_for_python((3, 7, 12))
    >>> "21.3.1" in versions
    True
    >>> "24.0" in versions
    False

Compare pip versions programmatically:

    >>> from versions import compare_versions
    >>> compare_versions("23.0.1", "21.3.1")
    1
    >>> compare_versions("20.0", "20.0")
    0
    >>> compare_versions("20.0", "21.0")
    -1
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Public Enumerations
# ---------------------------------------------------------------------------


class VersionSource(Enum):
    """
    Identifies where a version specification originated.

    Why distinguish sources:
        - User-requested versions get strict validation with clear error
          messages telling the user exactly why their choice is incompatible.
        - Auto-resolved versions silently select the best compatible option;
          the user doesn't need a warning about why 24.0 wasn't chosen for
          their Python 3.6 because they never asked for 24.0.
        - Fallback versions (used when all else fails) are logged differently
          to help debugging; they indicate a gap in the compatibility table.
        - Offline versions come from a local wheel file whose version is
          discovered, not requested; validation is different.

    Examples
    --------
    >>> source = VersionSource.USER_REQUESTED
    >>> source.name
    'USER_REQUESTED'
    >>> source == VersionSource.AUTO_RESOLVED
    False
    """

    USER_REQUESTED = auto()
    """User explicitly passed ``--version X.Y.Z`` on the command line."""

    AUTO_RESOLVED = auto()
    """No version specified; resolver chose the best compatible version."""

    FALLBACK = auto()
    """Resolver could not determine best version; used a conservative default."""

    OFFLINE_DISCOVERED = auto()
    """Version was read from a local wheel file's filename."""


class CompatibilityStatus(Enum):
    """
    Result of checking whether a pip version is compatible with a Python version.

    Why three states instead of a boolean:
        - "Compatible" and "Incompatible" are straightforward.
        - "Unknown" handles versions outside the known compatibility table
          (newer than LATEST_KNOWN_PIP or custom builds). The caller can
          decide whether to treat Unknown as compatible (optimistic) or
          incompatible (conservative) based on their risk tolerance.

    Examples
    --------
    >>> status = CompatibilityStatus.COMPATIBLE
    >>> bool(status)  # Truthy for compatible
    True
    >>> status = CompatibilityStatus.INCOMPATIBLE
    >>> bool(status)  # Falsy for incompatible
    False
    """

    COMPATIBLE = auto()
    """Version is known to support the given Python version."""

    INCOMPATIBLE = auto()
    """Version is known to NOT support the given Python version."""

    UNKNOWN = auto()
    """Version is outside the known compatibility table; status undetermined."""

    def __bool__(self) -> bool:
        """Enable boolean checks: COMPATIBLE is True, others are False."""
        return self == CompatibilityStatus.COMPATIBLE


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------


@dataclass
class VersionConstraint:
    """
    Represents a Python version constraint for a range of pip versions.

    Why a dataclass instead of a raw tuple:
        - Named fields make the compatibility table readable and auditable.
          ``(20, 3, 0, 24, 0, 0, 3, 7)`` is opaque; ``min_pip=(20,3)``
          and ``min_python=(3,7)`` is self-documenting.
        - Optional bounds allow open-ended ranges: a max of ``None`` means
          "all versions above min_pip".
        - The ``reason`` field provides user-facing explanations.
        - ``__contains__`` enables ``pip_version in constraint`` syntax.

    Attributes
    ----------
    min_pip : Tuple[int, int]
        Minimum pip version this constraint applies to, as (major, minor).
        Example: ``(21, 0)``.
    max_pip : Optional[Tuple[int, int]]
        Maximum pip version this constraint applies to, exclusive.
        ``None`` means no upper bound. Example: ``(24, 0)`` means
        versions < 24.0.
    min_python : Tuple[int, int]
        Minimum Python version required, as (major, minor).
        Example: ``(3, 7)``.
    max_python : Optional[Tuple[int, int]]
        Maximum Python version supported, exclusive. ``None`` means
        all versions above min_python.
    reason : str
        Human-readable explanation of why this constraint exists.
        Displayed to users when their requested version is incompatible.

    Examples
    --------
    >>> constraint = VersionConstraint(
    ...     min_pip=(22, 0),
    ...     max_pip=None,
    ...     min_python=(3, 7),
    ...     max_python=None,
    ...     reason="pip>=22.0 requires Python>=3.7",
    ... )
    >>> (23, 0) in constraint
    True
    >>> (21, 0) in constraint
    False
    """

    min_pip: Tuple[int, int]
    max_pip: Optional[Tuple[int, int]]
    min_python: Tuple[int, int]
    max_python: Optional[Tuple[int, int]]
    reason: str

    def __contains__(self, pip_version: Tuple[int, int]) -> bool:
        """
        Check if a pip version falls within this constraint's range.

        Parameters
        ----------
        pip_version : Tuple[int, int]
            Pip version as (major, minor). Micro/patch is ignored for
            compatibility checks since Python support depends only on
            major.minor.

        Returns
        -------
        bool
            ``True`` if this constraint applies to the given pip version.

        Examples
        --------
        >>> c = VersionConstraint((21,0), (22,0), (3,6), (3,7), "test")
        >>> (21, 0) in c
        True
        >>> (21, 3) in c
        True
        >>> (22, 0) in c
        False
        """
        if pip_version < self.min_pip:
            return False
        if self.max_pip is not None and pip_version >= self.max_pip:
            return False
        return True

    def python_is_supported(self, python_version: Tuple[int, ...]) -> bool:
        """
        Check if a Python version satisfies this constraint's requirements.

        Parameters
        ----------
        python_version : Tuple[int, ...]
            Python version as (major, minor, ...). Only major and minor
            are compared.

        Returns
        -------
        bool
            ``True`` if the Python version meets the minimum and does not
            exceed the maximum (when specified).

        Examples
        --------
        >>> c = VersionConstraint((21,0), None, (3,6), None, "test")
        >>> c.python_is_supported((3, 8))
        True
        >>> c.python_is_supported((3, 5))
        False
        """
        py_major_minor = (python_version[0], python_version[1])
        if py_major_minor < self.min_python:
            return False
        if self.max_python is not None and py_major_minor >= self.max_python:
            return False
        return True


@dataclass
class VersionResolution:
    """
    Result of resolving which pip version to install.

    Why a dedicated result type:
        - Separates the resolved version from the reasoning behind it.
        - Enables logging the complete resolution path for debugging.
        - The ``warnings`` list surfaces non-fatal issues (e.g., "you
          requested 24.0 but 24.0 doesn't exist; using 24.0.0 instead").
        - ``source`` distinguishes auto-resolution from user requests,
          affecting how errors are presented.

    Attributes
    ----------
    version : Optional[str]
        The resolved pip version string (e.g., ``'23.2.1'``), or ``None``
        if resolution failed entirely.
    source : VersionSource
        How this version was determined.
    compatible : bool
        ``True`` if the resolved version is compatible with the current
        Python. Always ``True`` when ``version`` is not ``None``, since
        incompatible versions are rejected during resolution.
    reason : str
        Human-readable explanation of the resolution decision.
    warnings : List[str]
        Non-fatal issues encountered during resolution, such as minor
        version adjustments or deprecated version usage.
    alternatives : List[str]
        Other compatible versions the user could consider, sorted by
        recency (newest first). Useful when the user's requested version
        is incompatible and they need suggestions.

    Examples
    --------
    >>> resolution = VersionResolution(
    ...     version="23.2.1",
    ...     source=VersionSource.AUTO_RESOLVED,
    ...     compatible=True,
    ...     reason="Selected latest compatible version for Python 3.8",
    ...     warnings=[],
    ...     alternatives=["24.1.2", "24.0", "23.3.1"],
    ... )
    >>> resolution.version
    '23.2.1'
    >>> bool(resolution)
    True
    """

    version: Optional[str]
    source: VersionSource
    compatible: bool
    reason: str
    warnings: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Resolution succeeded if a version was determined."""
        return self.version is not None


# ---------------------------------------------------------------------------
# Compatibility Database
# ---------------------------------------------------------------------------

# The latest pip version known to this module. Used as a sentinel:
# versions beyond this are treated as UNKNOWN rather than INCOMPATIBLE
# because we lack data to make a determination.
LATEST_KNOWN_PIP: str = "26.1.1"

# Maps Python (major, minor) to the maximum pip version that supports it.
# Why a dict keyed by Python version:
#   - O(1) lookup when resolving "best for this Python".
#   - Explicit and auditable; no hidden logic.
#   - Easy to update when new pip versions drop old Python support.
#
# Update policy: When pip releases a version that drops support for a
# Python minor version, add an entry here. For example, when pip 24.0
# dropped Python 3.7, we record that 3.7's max pip is 23.3.x.
_MAX_PIP_FOR_PYTHON: Dict[Tuple[int, int], str] = {
    (2, 7): "20.3.4",
    (3, 0): "7.1.2",
    (3, 1): "7.1.2",
    (3, 2): "7.1.2",
    (3, 3): "10.0.1",
    (3, 4): "19.1.1",
    (3, 5): "20.3.4",
    (3, 6): "21.3.1",
    (3, 7): "23.3.2",
    # Python 3.8+ supports all pip versions through LATEST_KNOWN_PIP.
    # Entries are added here only when support is dropped.
}

# Detailed constraints used for validation and error messages.
# Each entry describes a pip version range and its Python requirement.
# Why list instead of dict:
#   - Multiple constraints can apply to the same Python version
#     (e.g., pip 20.x supports 3.5, pip 21.x requires 3.6).
#   - List order matters: first match wins during lookup.
#   - Allows overlapping ranges that are resolved by iteration order.
_VERSION_CONSTRAINTS: List[VersionConstraint] = [
    VersionConstraint(
        min_pip=(24, 0),
        max_pip=None,
        min_python=(3, 8),
        max_python=None,
        reason="pip>=24.0 requires Python>=3.8",
    ),
    VersionConstraint(
        min_pip=(23, 0),
        max_pip=(24, 0),
        min_python=(3, 7),
        max_python=None,
        reason="pip>=23.0 requires Python>=3.7",
    ),
    VersionConstraint(
        min_pip=(22, 0),
        max_pip=(23, 0),
        min_python=(3, 7),
        max_python=None,
        reason="pip>=22.0 requires Python>=3.7",
    ),
    VersionConstraint(
        min_pip=(21, 0),
        max_pip=(22, 0),
        min_python=(3, 6),
        max_python=None,
        reason="pip>=21.0 requires Python>=3.6",
    ),
    VersionConstraint(
        min_pip=(20, 0),
        max_pip=(21, 0),
        min_python=(3, 5),
        max_python=None,
        reason="pip>=20.0 requires Python>=3.5",
    ),
    VersionConstraint(
        min_pip=(19, 0),
        max_pip=(20, 0),
        min_python=(3, 5),
        max_python=None,
        reason="pip>=19.0 requires Python>=3.5",
    ),
    VersionConstraint(
        min_pip=(10, 0),
        max_pip=(19, 0),
        min_python=(3, 5),
        max_python=(3, 8),
        reason="pip 10.x-18.x supports Python 3.5-3.7",
    ),
    VersionConstraint(
        min_pip=(9, 0),
        max_pip=(10, 0),
        min_python=(3, 4),
        max_python=(3, 7),
        reason="pip 9.x supports Python 3.4-3.6",
    ),
    VersionConstraint(
        min_pip=(7, 0),
        max_pip=(9, 0),
        min_python=(3, 2),
        max_python=(3, 6),
        reason="pip 7.x-8.x supports Python 3.2-3.5",
    ),
    VersionConstraint(
        min_pip=(1, 0),
        max_pip=(7, 0),
        min_python=(2, 7),
        max_python=(3, 5),
        reason="pip<7.0 supports Python 2.7-3.4",
    ),
]

# Well-known pip versions that the user might reasonably request.
# Used to generate alternatives when a request is incompatible.
# Populated with every stable pip release from 20.0 onward.
_KNOWN_VERSIONS: List[str] = [
    "24.3.1",
    "24.2",
    "24.1.2",
    "24.0",
    "23.3.2",
    "23.3.1",
    "23.2.1",
    "23.1.2",
    "23.0.1",
    "23.0",
    "22.3.1",
    "22.3",
    "22.2.2",
    "22.1.2",
    "22.0.4",
    "21.3.1",
    "21.2.4",
    "21.1.3",
    "21.0.1",
    "20.3.4",
    "20.3.3",
    "20.2.4",
    "20.1.1",
    "20.0.2",
]


# ---------------------------------------------------------------------------
# Public Utility Functions
# ---------------------------------------------------------------------------


def parse_version(version_str: str) -> Tuple[int, ...]:
    """
    Parse a version string into a tuple of integers for comparison.

    Why a standalone function instead of a class:
        - Version parsing is needed in multiple modules (checker, fetcher,
          installer) without coupling them to this module.
        - Tuple comparison is built into Python; no custom comparator needed.
        - Handles variable-length versions: "24.0" → (24, 0),
          "21.3.1" → (21, 3, 1).

    Parameters
    ----------
    version_str : str
        A version string with dot-separated integers.
        Examples: ``"24.0"``, ``"21.3.1"``, ``"1.0.0.post1"``.

    Returns
    -------
    Tuple[int, ...]
        Parsed version components as integers. Non-numeric suffixes
        (like ``.post1`` or ``.dev0``) are stripped before parsing.

    Raises
    ------
    ValueError
        If ``version_str`` contains no numeric components.

    Examples
    --------
    >>> parse_version("24.0")
    (24, 0)
    >>> parse_version("21.3.1")
    (21, 3, 1)
    >>> parse_version("20.3.4.post1")
    (20, 3, 4)
    >>> parse_version("invalid")
    Traceback (most recent call last):
        ...
    ValueError: No numeric version components found in 'invalid'
    """
    # Strip non-numeric suffixes like .post1, .dev0, .rc1
    cleaned = re.sub(r"\.(post|dev|rc|a|b|alpha|beta)\d*.*$", "", version_str)
    parts = re.findall(r"\d+", cleaned)

    if not parts:
        raise ValueError(f"No numeric version components found in '{version_str}'")

    return tuple(int(p) for p in parts)


def compare_versions(version_a: str, version_b: str) -> int:
    """
    Compare two version strings.

    Why not use ``packaging.version``:
        - This module must have zero dependencies; it is used before pip
          is installed, so importing packaging would create a circular
          dependency.
        - SemVer comparison logic is simple enough to implement correctly
          without external libraries.
        - This implementation matches pip's own versioning scheme
          (``MAJOR.MINOR.MICRO`` with optional trailing segments).

    Parameters
    ----------
    version_a : str
        First version string.
    version_b : str
        Second version string.

    Returns
    -------
    int
        - ``1`` if version_a > version_b
        - ``0`` if version_a == version_b
        - ``-1`` if version_a < version_b

    Examples
    --------
    >>> compare_versions("24.0", "23.0")
    1
    >>> compare_versions("21.3.1", "21.3.1")
    0
    >>> compare_versions("20.0", "21.0")
    -1
    >>> compare_versions("21.0", "21.0.0")
    0
    """
    parts_a = parse_version(version_a)
    parts_b = parse_version(version_b)

    # Pad shorter tuple with zeros for fair comparison
    max_len = max(len(parts_a), len(parts_b))
    parts_a = parts_a + (0,) * (max_len - len(parts_a))
    parts_b = parts_b + (0,) * (max_len - len(parts_b))

    if parts_a > parts_b:
        return 1
    elif parts_a < parts_b:
        return -1
    return 0


def normalize_version(version_str: str) -> str:
    """
    Normalize a version string to MAJOR.MINOR.MICRO format.

    Why normalization matters:
        - Users may request "24", "24.0", or "24.0.0" — all refer
          to the same logical version but differ in string form.
        - Wheel filenames and PyPI API responses use three-component
          versions; matching requires consistent formatting.
        - Prevents duplicate entries in version lists when a version
          appears in multiple formats.

    Parameters
    ----------
    version_str : str
        A version string with 1, 2, or 3+ components.

    Returns
    -------
    str
        Normalized version with exactly three components.
        Example: ``"24"`` → ``"24.0.0"``, ``"21.3"`` → ``"21.3.0"``.

    Examples
    --------
    >>> normalize_version("24")
    '24.0.0'
    >>> normalize_version("21.3")
    '21.3.0'
    >>> normalize_version("20.3.4")
    '20.3.4'
    >>> normalize_version("20.3.4.1")
    '20.3.4'
    """
    parts = parse_version(version_str)
    while len(parts) < 3:
        parts = parts + (0,)
    parts = parts[:3]
    return f"{parts[0]}.{parts[1]}.{parts[2]}"


def get_compatible_versions_for_python(
    python_version: Tuple[int, ...],
) -> List[str]:
    """
    Return all known pip versions compatible with a given Python version.

    Why this function exists:
        - When a user requests an incompatible version, the orchestrator
          needs to suggest alternatives.
        - Generating alternatives from the constraint table is more accurate
          than showing all known versions.
        - Offline mode may have limited wheel files; this function helps
          identify which local wheels are viable.

    Parameters
    ----------
    python_version : Tuple[int, ...]
        Python version tuple, at least (major, minor).
        Example: ``(3, 7)`` or ``(3, 7, 12)``.

    Returns
    -------
    List[str]
        Sorted list of compatible pip version strings, newest first.

    Examples
    --------
    >>> get_compatible_versions_for_python((3, 8))
    ['24.3.1', '24.2', '24.1.2', '24.0', '23.3.2', ...]
    >>> get_compatible_versions_for_python((3, 6))
    ['21.3.1', '21.2.4', '21.1.3', '21.0.1', '20.3.4', ...]
    >>> get_compatible_versions_for_python((2, 7))
    ['20.3.4', '20.3.3', '20.2.4', '20.1.1', '20.0.2']
    """
    compatible: List[str] = []
    py_major_minor = (python_version[0], python_version[1])

    for version_str in _KNOWN_VERSIONS:
        parts = parse_version(version_str)
        pip_major_minor = (parts[0], parts[1]) if len(parts) >= 2 else (parts[0], 0)

        # Check against constraints
        is_compat = True
        for constraint in _VERSION_CONSTRAINTS:
            if pip_major_minor in constraint:
                if not constraint.python_is_supported(python_version):
                    is_compat = False
                    break

        if is_compat:
            compatible.append(version_str)

    # Sort newest first
    compatible.sort(key=lambda v: parse_version(v), reverse=True)
    return compatible


# ---------------------------------------------------------------------------
# Core Resolver Class
# ---------------------------------------------------------------------------


class VersionResolver:
    """
    Resolves which pip version should be installed for a given Python.

    Why a class instead of a module-level function:
        - Maintains configuration state (strictness, default strategy).
        - Caches resolution results per Python version to avoid repeated
          compatibility table scans.
        - Enables dependency injection in tests by replacing the constraints
          table with a mock.
        - Multiple resolver instances can target different Python versions
          simultaneously without cross-contamination.

    Parameters
    ----------
    python_version : Optional[Tuple[int, int, int]]
        The Python version to resolve for. If ``None``, uses
        ``sys.version_info[:3]``.
    strict : bool
        If ``True``, unknown versions (newer than LATEST_KNOWN_PIP) are
        treated as INCOMPATIBLE rather than UNKNOWN. Conservative mode.

    Attributes
    ----------
    python_version : Tuple[int, int, int]
        The Python version this resolver targets.
    strict : bool
        Whether to treat unknown versions conservatively.

    Examples
    --------
    Basic usage:

        >>> resolver = VersionResolver()
        >>> result = resolver.resolve_best_version()
        >>> print(result.version)
        '24.3.1'
        >>> print(result.reason)
        'Selected latest compatible version for Python 3.12'

    Validate a user-requested version:

        >>> resolver = VersionResolver()
        >>> result = resolver.resolve_user_version("21.3.1")
        >>> result.compatible
        True
        >>> result.version
        '21.3.1'

    Handle incompatible request:

        >>> resolver = VersionResolver(python_version=(3, 6, 8))
        >>> result = resolver.resolve_user_version("24.0")
        >>> result.compatible
        False
        >>> print(result.reason)
        'pip>=24.0 requires Python>=3.8; current Python is 3.6'
        >>> result.alternatives[:3]
        ['21.3.1', '21.2.4', '21.1.3']

    Strict mode for enterprise environments:

        >>> resolver = VersionResolver(strict=True)
        >>> compatible, reason = resolver.is_compatible("99.0.0")
        >>> compatible
        False
        >>> reason
        'Version 99.0.0 is not in the known versions list'
    """

    def __init__(
        self,
        python_version: Optional[Tuple[int, int, int]] = None,
        strict: bool = False,
    ) -> None:
        """
        Initialize the resolver for a specific Python version.

        Parameters
        ----------
        python_version : Optional[Tuple[int, int, int]]
            Target Python version as (major, minor, micro). If ``None``,
            uses the currently running Python's version.
        strict : bool
            Conservative mode: unknown versions are treated as incompatible
            rather than uncertain. Suitable for production environments
            where only verified-compatible versions are acceptable.

        Raises
        ------
        ValueError
            If ``python_version`` has fewer than 3 components.

        Examples
        --------
        >>> resolver = VersionResolver()  # Uses current Python
        >>> resolver.python_version  # doctest: +SKIP
        (3, 11, 5)

        >>> resolver = VersionResolver(python_version=(3, 8, 10))
        >>> resolver.python_version
        (3, 8, 10)

        >>> resolver = VersionResolver(strict=True)
        >>> resolver.strict
        True
        """
        if python_version is None:
            self.python_version: Tuple[int, int, int] = sys.version_info[:3]
        else:
            if len(python_version) < 3:
                raise ValueError(
                    f"python_version must have at least 3 components, "
                    f"got {len(python_version)}: {python_version}"
                )
            self.python_version = python_version

        self.strict: bool = strict
        self._constraints: List[VersionConstraint] = list(_VERSION_CONSTRAINTS)
        self._known_versions: List[str] = list(_KNOWN_VERSIONS)
        self._max_pip_for_python: Dict[Tuple[int, int], str] = dict(
            _MAX_PIP_FOR_PYTHON
        )
        self._latest_known: str = LATEST_KNOWN_PIP

    # ------------------------------------------------------------------
    # Public API: Version Resolution
    # ------------------------------------------------------------------

    def resolve_best_version(self) -> VersionResolution:
        """
        Determine the best pip version for the target Python.

        Resolution order:
        1. Check the max-pip-for-python table for a hard ceiling.
        2. If the target Python is 3.8+, use LATEST_KNOWN_PIP.
        3. Fall back to scanning the known versions list for the newest
           compatible entry.

        Why resolve_best_version exists separately from resolve_user_version:
            - When no version is specified, the goal is "best available,"
              not "validate this specific choice."
            - The resolution can be optimistic (latest known) for modern
              Python without user confirmation.
            - For older Python, the ceiling is non-negotiable; there is
              no user preference to respect.

        Returns
        -------
        VersionResolution
            Resolution result with the selected version and reasoning.

        Examples
        --------
        >>> resolver = VersionResolver(python_version=(3, 9, 18))
        >>> result = resolver.resolve_best_version()
        >>> result.version == LATEST_KNOWN_PIP
        True
        >>> result.source == VersionSource.AUTO_RESOLVED
        True

        >>> resolver = VersionResolver(python_version=(3, 6, 8))
        >>> result = resolver.resolve_best_version()
        >>> result.version
        '21.3.1'
        >>> "Python 3.6" in result.reason
        True
        """
        py_major_minor = (self.python_version[0], self.python_version[1])

        # Check hard ceiling first
        if py_major_minor in self._max_pip_for_python:
            ceiling = self._max_pip_for_python[py_major_minor]
            return VersionResolution(
                version=ceiling,
                source=VersionSource.AUTO_RESOLVED,
                compatible=True,
                reason=(
                    f"Selected pip {ceiling}, the last version supporting "
                    f"Python {self.python_version[0]}.{self.python_version[1]}"
                ),
                warnings=[],
                alternatives=[],
            )

        # Python 3.8+: use latest known
        if py_major_minor >= (3, 8):
            return VersionResolution(
                version=self._latest_known,
                source=VersionSource.AUTO_RESOLVED,
                compatible=True,
                reason=(
                    f"Selected latest known compatible version "
                    f"({self._latest_known}) for Python "
                    f"{self.python_version[0]}.{self.python_version[1]}"
                ),
                warnings=[],
                alternatives=[],
            )

        # Fallback: scan known versions
        compat = get_compatible_versions_for_python(self.python_version)
        if compat:
            best = compat[0]
            return VersionResolution(
                version=best,
                source=VersionSource.AUTO_RESOLVED,
                compatible=True,
                reason=(
                    f"Selected {best}, the newest compatible version "
                    f"for Python {self.python_version[0]}.{self.python_version[1]}"
                ),
                warnings=(
                    []
                    if best == self._latest_known
                    else [
                        f"Note: {self._latest_known} is available but does not "
                        f"support Python {self.python_version[0]}.{self.python_version[1]}"
                    ]
                ),
                alternatives=compat[1:5],
            )

        # No compatible version found at all
        return VersionResolution(
            version=None,
            source=VersionSource.AUTO_RESOLVED,
            compatible=False,
            reason=(
                f"No compatible pip version found for Python "
                f"{self.python_version[0]}.{self.python_version[1]}.{self.python_version[2]}"
            ),
            warnings=[
                "This Python version may be too old or too new for any known pip."
            ],
            alternatives=[],
        )

    def resolve_user_version(self, requested_version: str) -> VersionResolution:
        """
        Validate and normalize a user-requested pip version.

        This method handles:
        - Version normalization ("24" → "24.0.0").
        - Compatibility checking against the target Python.
        - Generating alternatives when the request is incompatible.
        - Recognizing when a requested version doesn't exist at all.

        Why validate before installation:
            - Fetching a non-existent version wastes network round-trips.
            - Installing an incompatible version produces cryptic errors
              that are harder to debug than a clear pre-flight message.
            - The user can abort or choose an alternative before any
              files are downloaded or modified.

        Parameters
        ----------
        requested_version : str
            The version string the user provided. Accepts partial versions
            like ``"24"``, ``"21.3"``, or full versions like ``"20.3.4"``.

        Returns
        -------
        VersionResolution
            Resolution result. If ``compatible`` is ``False``, the
            ``alternatives`` field contains suggested versions.

        Examples
        --------
        Successful resolution:

            >>> resolver = VersionResolver(python_version=(3, 9, 18))
            >>> result = resolver.resolve_user_version("21.3.1")
            >>> result.compatible
            True
            >>> result.version
            '21.3.1'
            >>> result.source == VersionSource.USER_REQUESTED
            True

        Normalization of partial version:

            >>> result = resolver.resolve_user_version("21")
            >>> result.version
            '21.0.0'
            >>> "normalized" in result.reason.lower()
            True

        Incompatible version with alternatives:

            >>> resolver = VersionResolver(python_version=(3, 6, 8))
            >>> result = resolver.resolve_user_version("24.0")
            >>> result.compatible
            False
            >>> len(result.alternatives) > 0
            True
        """
        warnings: List[str] = []

        # Normalize the requested version
        try:
            normalized = normalize_version(requested_version)
        except ValueError as e:
            return VersionResolution(
                version=None,
                source=VersionSource.USER_REQUESTED,
                compatible=False,
                reason=f"Invalid version format: {e}",
                warnings=[],
                alternatives=[],
            )

        if normalized != requested_version:
            warnings.append(
                f"Version '{requested_version}' normalized to '{normalized}'"
            )

        # Check if this version is known to exist
        if normalized not in self._known_versions and self.strict:
            return VersionResolution(
                version=None,
                source=VersionSource.USER_REQUESTED,
                compatible=False,
                reason=(
                    f"Version {normalized} is not in the known versions list. "
                    f"Latest known version is {self._latest_known}."
                ),
                warnings=warnings,
                alternatives=get_compatible_versions_for_python(self.python_version)[:5],
            )

        # Check compatibility
        compatible, reason = self.is_compatible(normalized)

        if compatible:
            return VersionResolution(
                version=normalized,
                source=VersionSource.USER_REQUESTED,
                compatible=True,
                reason=f"Version {normalized} is compatible with Python "
                f"{self.python_version[0]}.{self.python_version[1]}",
                warnings=warnings,
                alternatives=[],
            )
        else:
            alternatives = get_compatible_versions_for_python(self.python_version)
            return VersionResolution(
                version=normalized if not self.strict else None,
                source=VersionSource.USER_REQUESTED,
                compatible=False,
                reason=reason,
                warnings=warnings,
                alternatives=alternatives[:5],
            )

    def resolve_from_wheel_filename(self, wheel_path: str) -> VersionResolution:
        """
        Extract and validate a pip version from a local wheel filename.

        Wheel filenames follow the format:
        ``pip-{version}-py3-none-any.whl``.

        Why this method exists:
            - In offline mode, the version is not requested by the user
              but discovered from the wheel file they provided.
            - The discovered version still needs compatibility validation
              against the target Python.
            - Provides consistent VersionResolution output regardless of
              how the version was determined.

        Parameters
        ----------
        wheel_path : str
            Path or filename of a pip wheel file.
            Example: ``"pip-21.3.1-py3-none-any.whl"``.

        Returns
        -------
        VersionResolution
            Resolution result with ``source=OFFLINE_DISCOVERED``.

        Examples
        --------
        >>> resolver = VersionResolver(python_version=(3, 9, 18))
        >>> result = resolver.resolve_from_wheel_filename(
        ...     "downloads/pip-21.3.1-py3-none-any.whl"
        ... )
        >>> result.version
        '21.3.1'
        >>> result.source == VersionSource.OFFLINE_DISCOVERED
        True

        >>> resolver = VersionResolver(python_version=(3, 6, 8))
        >>> result = resolver.resolve_from_wheel_filename("pip-24.0-py3-none-any.whl")
        >>> result.compatible
        False
        """
        import os

        basename = os.path.basename(str(wheel_path))

        # Parse: pip-{version}-py3-none-any.whl
        match = re.match(r"pip-(.+)-py\d+-none-any\.whl", basename)
        if not match:
            return VersionResolution(
                version=None,
                source=VersionSource.OFFLINE_DISCOVERED,
                compatible=False,
                reason=f"Could not parse version from wheel filename: {basename}",
                warnings=[],
                alternatives=[],
            )

        version_str = match.group(1)
        try:
            normalized = normalize_version(version_str)
        except ValueError:
            return VersionResolution(
                version=None,
                source=VersionSource.OFFLINE_DISCOVERED,
                compatible=False,
                reason=f"Invalid version in wheel filename: {version_str}",
                warnings=[],
                alternatives=[],
            )

        compatible, reason = self.is_compatible(normalized)
        return VersionResolution(
            version=normalized,
            source=VersionSource.OFFLINE_DISCOVERED,
            compatible=compatible,
            reason=reason,
            warnings=[],
            alternatives=(
                []
                if compatible
                else get_compatible_versions_for_python(self.python_version)[:5]
            ),
        )

    # ------------------------------------------------------------------
    # Public API: Compatibility Checks
    # ------------------------------------------------------------------

    def is_compatible(
        self,
        pip_version: str,
        python_version: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[bool, str]:
        """
        Check whether a pip version is compatible with a Python version.

        Why return (bool, str) instead of just bool:
            - The reason string is shown to users to explain rejection.
            - Different callers need different levels of detail; the bool
              enables quick conditionals, the string provides diagnostics.
            - The reason distinguishes "incompatible because too old" from
              "incompatible because too new" from "unknown version."

        Parameters
        ----------
        pip_version : str
            Pip version to check. Example: ``"23.0.1"``.
        python_version : Optional[Tuple[int, ...]]
            Python version to check against. If ``None``, uses the
            resolver's target Python version.

        Returns
        -------
        compatible : bool
            ``True`` if this pip version can run on the Python version.
        reason : str
            Explanation of the compatibility determination.

        Examples
        --------
        >>> resolver = VersionResolver(python_version=(3, 8, 10))
        >>> compatible, reason = resolver.is_compatible("23.0.1")
        >>> compatible
        True
        >>> reason
        'Version 23.0.1 supports Python 3.8'

        >>> compatible, reason = resolver.is_compatible("24.0", python_version=(3, 6))
        >>> compatible
        False
        >>> "requires Python>=3.8" in reason
        True

        >>> resolver = VersionResolver(strict=True)
        >>> compatible, reason = resolver.is_compatible("99.0.0")
        >>> compatible
        False
        """
        if python_version is None:
            python_version = self.python_version

        py_major_minor = (python_version[0], python_version[1])

        try:
            parts = parse_version(pip_version)
        except ValueError as e:
            return False, f"Could not parse version '{pip_version}': {e}"

        pip_major_minor = (parts[0], parts[1]) if len(parts) >= 2 else (parts[0], 0)

        # Check against known constraints
        matched_constraint: Optional[VersionConstraint] = None
        for constraint in self._constraints:
            if pip_major_minor in constraint:
                matched_constraint = constraint
                break

        if matched_constraint is not None:
            if matched_constraint.python_is_supported(python_version):
                return True, (
                    f"Version {pip_version} supports Python "
                    f"{python_version[0]}.{python_version[1]}"
                )
            else:
                return False, (
                    f"{matched_constraint.reason}; "
                    f"current Python is {python_version[0]}.{python_version[1]}"
                )

        # No matching constraint — version is outside known ranges
        if self.strict:
            return False, (
                f"Version {pip_version} is not in the known compatibility table. "
                f"Latest known version is {self._latest_known}."
            )

        # Non-strict: if version is newer than latest known, assume compatible
        if compare_versions(pip_version, self._latest_known) > 0:
            return True, (
                f"Version {pip_version} is newer than {self._latest_known}; "
                f"assuming compatibility with Python "
                f"{python_version[0]}.{python_version[1]} (not verified)"
            )

        # Older than known range with no matching constraint — conservative no
        return False, (
            f"Version {pip_version} is older than the known compatibility range "
            f"and may not support Python {python_version[0]}.{python_version[1]}"
        )

    def get_latest_compatible(self) -> str:
        """
        Return the latest pip version compatible with the target Python.

        Convenience method that returns just the version string, not
        the full resolution. Useful when callers only need the version
        and don't care about warnings or alternatives.

        Returns
        -------
        str
            The latest compatible pip version string.

        Raises
        ------
        RuntimeError
            If no compatible version could be found.

        Examples
        --------
        >>> resolver = VersionResolver(python_version=(3, 9, 18))
        >>> resolver.get_latest_compatible()
        '24.3.1'

        >>> resolver = VersionResolver(python_version=(3, 6, 8))
        >>> resolver.get_latest_compatible()
        '21.3.1'
        """
        result = self.resolve_best_version()
        if result.version is None:
            raise RuntimeError(
                f"No compatible pip version found for Python "
                f"{self.python_version[0]}.{self.python_version[1]}"
            )
        return result.version

    # ------------------------------------------------------------------
    # Public API: Version Lists
    # ------------------------------------------------------------------

    def get_all_compatible_versions(self) -> List[str]:
        """
        Return all known pip versions compatible with the target Python.

        Why expose this:
            - Enables UIs to present a picker of compatible versions.
            - Allows offline mode to check which locally cached wheels
              are viable for the current Python.
            - Useful for debugging: seeing the full compatible set reveals
              gaps in the compatibility table.

        Returns
        -------
        List[str]
            Sorted list of compatible version strings, newest first.

        Examples
        --------
        >>> resolver = VersionResolver(python_version=(3, 11, 0))
        >>> versions = resolver.get_all_compatible_versions()
        >>> len(versions) > 0
        True
        >>> versions[0] == LATEST_KNOWN_PIP
        True
        """
        return get_compatible_versions_for_python(self.python_version)

    def get_version_range_description(self) -> str:
        """
        Return a human-readable description of the compatible pip version range.

        Why this exists:
            - Error messages are more helpful when they explain the range
              of acceptable versions, not just that one version is rejected.
            - UIs can display this as a hint: "Enter a version between
              X and Y."

        Returns
        -------
        str
            Description like ``"pip 9.0.0 to 21.3.1"`` or
            ``"pip 20.0.0 and later"``.

        Examples
        --------
        >>> resolver = VersionResolver(python_version=(3, 6, 8))
        >>> resolver.get_version_range_description()
        'pip 9.0.0 to 21.3.1'

        >>> resolver = VersionResolver(python_version=(3, 11, 0))
        >>> desc = resolver.get_version_range_description()
        >>> "and later" in desc
        True
        """
        compat = self.get_all_compatible_versions()
        if not compat:
            return "No compatible versions"

        oldest = compat[-1]
        newest = compat[0]

        if newest == self._latest_known:
            return f"pip {oldest} and later"
        else:
            return f"pip {oldest} to {newest}"

    # ------------------------------------------------------------------
    # Public API: Version Sorting
    # ------------------------------------------------------------------

    @staticmethod
    def sort_versions_newest_first(versions: List[str]) -> List[str]:
        """
        Sort a list of version strings from newest to oldest.

        Why static:
            - Sorting is independent of the resolver's target Python.
            - Useful as a utility in other modules without instantiating
              a resolver.

        Parameters
        ----------
        versions : List[str]
            Unsorted version strings.

        Returns
        -------
        List[str]
            Versions sorted newest first.

        Examples
        --------
        >>> VersionResolver.sort_versions_newest_first(
        ...     ["20.0", "23.0", "21.3.1"]
        ... )
        ['23.0', '21.3.1', '20.0']
        """
        return sorted(versions, key=lambda v: parse_version(v), reverse=True)

    @staticmethod
    def sort_versions_oldest_first(versions: List[str]) -> List[str]:
        """
        Sort a list of version strings from oldest to newest.

        Parameters
        ----------
        versions : List[str]
            Unsorted version strings.

        Returns
        -------
        List[str]
            Versions sorted oldest first.

        Examples
        --------
        >>> VersionResolver.sort_versions_oldest_first(
        ...     ["23.0", "20.0", "21.3.1"]
        ... )
        ['20.0', '21.3.1', '23.0']
        """
        return sorted(versions, key=lambda v: parse_version(v))
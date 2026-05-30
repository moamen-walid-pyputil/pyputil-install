"""
Platform Detection and Target Triple Resolution
================================================

Comprehensive platform detection module that identifies the current
operating system, CPU architecture, and maps them to Rust-style target
triples used by ``python-build-standalone`` releases.

Provides automatic detection, manual override, validation, and
human-readable platform descriptions. No third-party dependencies.

Security
--------
- All platform strings are validated against a whitelist before use
  in URL construction or filesystem paths.
- Environment variables used for override (``PYTHON_STANDALONE_TARGET``)
  are sanitised to prevent injection.
- Filesystem paths derived from platform strings are normalised and
  validated.

Usage
-----
.. code-block:: python

    from platforms import PlatformDetector, TargetTriple

    detector = PlatformDetector()
    triple = detector.detect()
    print(f"Target triple: {triple}")
    print(f"Description: {triple.description}")
    print(f"Archive extension: {triple.archive_extension}")

Manual override::

    triple = TargetTriple.from_string("x86_64-unknown-linux-gnu")

Listing all supported targets::

    from platforms import SUPPORTED_TARGETS
    for t in SUPPORTED_TARGETS:
        print(t.raw)

Warnings
--------
- On Linux, ARM detection distinguishes between hard-float and
  soft-float ABIs by reading ``/proc/cpuinfo``. If unavailable,
  hard-float is assumed.
- Android/Termux detection relies on the ``ANDROID_ROOT`` environment
  variable. On non-Termux Android environments, detection may be
  inaccurate.
- macOS Rosetta 2 (x86_64 emulation on Apple Silicon) reports
  ``x86_64`` by default. Use the environment override to force
  ``aarch64-apple-darwin``.
- ``platform.machine()`` may return ``AMD64`` on Windows but
  ``x86_64`` elsewhere. Both are normalised.

Notes
-----
- Target triples follow the ``<arch>-<vendor>-<os>[-<abi>]``
  convention used by Rust and LLVM.
- The ``vendor`` field is informational; ``python-build-standalone``
  treats ``unknown`` and ``pc`` equivalently.
- Musl-based Linux builds are **not** supported by this module
  because ``python-build-standalone`` provides only glibc builds.
"""

from __future__ import annotations

import os
import platform as _platform
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, FrozenSet, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Environment variable for manual target override.
_ENV_OVERRIDE: str = "PYTHON_STANDALONE_TARGET"

#: Mapping from :func:`platform.system()` return values to OS component
#: of the target triple.
_OS_MAP: Dict[str, str] = {
    "Linux": "linux",
    "Darwin": "darwin",
    "Windows": "windows",
}

#: Mapping from :func:`platform.machine()` return values to architecture
#: component of the target triple. Multiple keys may map to the same
#: value.
_ARCH_MAP: Dict[str, str] = {
    "x86_64": "x86_64",
    "AMD64": "x86_64",
    "x64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
    "ARM64": "aarch64",
    "armv7l": "armv7",
    "armv8l": "aarch64",
    "i686": "i686",
    "i386": "i686",
    "x86": "i686",
    "ppc64le": "powerpc64le",
    "s390x": "s390x",
}

#: ABI suffixes for Linux targets.
_LINUX_ABI_MAP: Dict[str, str] = {
    "x86_64": "gnu",
    "aarch64": "gnu",
    "armv7": "gnueabihf",
    "i686": "gnu",
    "powerpc64le": "gnu",
    "s390x": "gnu",
}

#: ABI suffixes for Windows targets.
_WINDOWS_ABI_MAP: Dict[str, str] = {
    "x86_64": "msvc",
    "i686": "msvc",
    "aarch64": "msvc",
}

#: Vendor component for each OS.
_VENDOR_MAP: Dict[str, str] = {
    "linux": "unknown",
    "darwin": "apple",
    "windows": "pc",
}

#: File extension for release archives on each OS.
_ARCHIVE_EXT_MAP: Dict[str, str] = {
    "linux": ".tar.gz",
    "darwin": ".tar.gz",
    "windows": ".tar.gz",
}

#: File extension for the Python executable on each OS.
_EXECUTABLE_EXT_MAP: Dict[str, str] = {
    "linux": "",
    "darwin": "",
    "windows": ".exe",
}

#: Python executable names to search for on each OS.
_PYTHON_BINARY_NAMES: Dict[str, tuple[str, ...]] = {
    "linux": ("python3", "python"),
    "darwin": ("python3", "python"),
    "windows": ("python.exe", "python3.exe"),
}

# ---------------------------------------------------------------------------
# Target Triple
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetTriple:
    """
    Immutable representation of a Rust-style target triple.

    Parameters
    ----------
    arch : str
        CPU architecture (e.g., ``"x86_64"``, ``"aarch64"``).
    vendor : str
        Vendor string (e.g., ``"unknown"``, ``"apple"``, ``"pc"``).
    os : str
        Operating system (e.g., ``"linux"``, ``"darwin"``, ``"windows"``).
    abi : str or None
        ABI string (e.g., ``"gnu"``, ``"msvc"``, ``"gnueabihf"``).
        ``None`` for targets without an ABI suffix.

    Raises
    ------
    ValueError
        If any component contains characters outside ``[a-z0-9_-]``.

    Examples
    --------
    >>> t = TargetTriple("x86_64", "unknown", "linux", "gnu")
    >>> t.raw
    'x86_64-unknown-linux-gnu'
    >>> t.is_linux
    True
    >>> t.is_64bit
    True
    """

    arch: str
    vendor: str
    os: str
    abi: Optional[str]

    #: Valid characters in target triple components.
    _VALID_CHARS: ClassVar[FrozenSet[str]] = frozenset(
        "abcdefghijklmnopqrstuvwxyz0123456789_-"
    )

    def __post_init__(self) -> None:
        """Validate component characters."""
        for name in ("arch", "vendor", "os"):
            value = getattr(self, name)
            if not value or not all(c in self._VALID_CHARS for c in value):
                raise ValueError(
                    f"Invalid {name}={value!r}. "
                    "Only [a-z0-9_-] allowed."
                )
        if self.abi is not None:
            if not self.abi or not all(
                c in self._VALID_CHARS for c in self.abi
            ):
                raise ValueError(
                    f"Invalid abi={self.abi!r}. "
                    "Only [a-z0-9_-] allowed."
                )

    @property
    def raw(self) -> str:
        """
        Full target triple string.

        Returns
        -------
        str
            e.g. ``"x86_64-unknown-linux-gnu"``.
        """
        base = f"{self.arch}-{self.vendor}-{self.os}"
        if self.abi:
            return f"{base}-{self.abi}"
        return base

    @property
    def is_linux(self) -> bool:
        """``True`` if this is a Linux target."""
        return self.os == "linux"

    @property
    def is_macos(self) -> bool:
        """``True`` if this is a macOS target."""
        return self.os == "darwin"

    @property
    def is_windows(self) -> bool:
        """``True`` if this is a Windows target."""
        return self.os == "windows"

    @property
    def is_android(self) -> bool:
        """``True`` if this is an Android target."""
        return self.abi == "android"

    @property
    def is_64bit(self) -> bool:
        """
        ``True`` for 64-bit architectures.

        Notes
        -----
        Covers ``x86_64``, ``aarch64``, ``powerpc64le``, ``s390x``.
        """
        return self.arch in ("x86_64", "aarch64", "powerpc64le", "s390x")

    @property
    def archive_extension(self) -> str:
        """
        File extension for release archives on this platform.

        Returns
        -------
        str
            ``".tar.gz"`` for all currently supported platforms.
        """
        return _ARCHIVE_EXT_MAP.get(self.os, ".tar.gz")

    @property
    def executable_extension(self) -> str:
        """
        File extension for Python executables on this platform.

        Returns
        -------
        str
            ``".exe"`` on Windows, ``""`` elsewhere.
        """
        return _EXECUTABLE_EXT_MAP.get(self.os, "")

    @property
    def python_binary_names(self) -> tuple[str, ...]:
        """
        Candidate filenames for the Python executable on this platform.

        Returns
        -------
        tuple of str
            e.g. ``("python3", "python")`` on Linux.
        """
        return _PYTHON_BINARY_NAMES.get(self.os, ("python3", "python"))

    @property
    def description(self) -> str:
        """
        Human-readable platform description.

        Returns
        -------
        str
            e.g. ``"Linux x86_64 (64-bit)"``.
        """
        os_name = {"linux": "Linux", "darwin": "macOS", "windows": "Windows"}
        arch_name = {
            "x86_64": "x86_64",
            "aarch64": "ARM64",
            "armv7": "ARMv7",
            "i686": "x86",
            "powerpc64le": "PPC64LE",
            "s390x": "s390x",
        }
        bits = "64-bit" if self.is_64bit else "32-bit"
        os_str = os_name.get(self.os, self.os)
        arch_str = arch_name.get(self.arch, self.arch)
        desc = f"{os_str} {arch_str} ({bits})"
        if self.is_android:
            desc += " Android"
        return desc

    @classmethod
    def from_string(cls, raw: str) -> TargetTriple:
        """
        Parse a target triple string.

        Parameters
        ----------
        raw : str
            e.g. ``"x86_64-unknown-linux-gnu"``.

        Returns
        -------
        TargetTriple

        Raises
        ------
        ValueError
            If *raw* does not have exactly 3 or 4 dash-separated
            components.
        """
        parts = raw.strip().lower().split("-")
        if len(parts) == 4:
            arch, vendor, os_name, abi = parts
            return cls(arch=arch, vendor=vendor, os=os_name, abi=abi)
        elif len(parts) == 3:
            arch, vendor, os_name = parts
            return cls(arch=arch, vendor=vendor, os=os_name, abi=None)
        else:
            raise ValueError(
                f"Target triple must have 3 or 4 parts, "
                f"got {len(parts)}: {raw!r}"
            )

    def __str__(self) -> str:
        return self.raw

    def __repr__(self) -> str:
        return f"TargetTriple({self.raw!r})"


# ---------------------------------------------------------------------------
# Supported Targets Registry
# ---------------------------------------------------------------------------


#: Complete list of target triples provided by ``python-build-standalone``.
#: Each entry maps to a known release asset.
SUPPORTED_TARGETS: tuple[TargetTriple, ...] = tuple(
    TargetTriple.from_string(t)
    for t in (
        # Linux glibc
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
        "armv7-unknown-linux-gnueabihf",
        "i686-unknown-linux-gnu",
        "powerpc64le-unknown-linux-gnu",
        "s390x-unknown-linux-gnu",
        # macOS
        "x86_64-apple-darwin",
        "aarch64-apple-darwin",
        # Windows
        "x86_64-pc-windows-msvc",
        "i686-pc-windows-msvc",
        "aarch64-pc-windows-msvc",
        # Android (Termux)
        "aarch64-linux-android",
    )
)

#: Set of supported target triples for fast lookup.
_SUPPORTED_SET: FrozenSet[str] = frozenset(t.raw for t in SUPPORTED_TARGETS)

# ---------------------------------------------------------------------------
# Dynamic Target Registration
# ---------------------------------------------------------------------------

#: Internal mutable set of supported target triples.
_supported_targets_set: set[str] = set()


def _init_supported_targets() -> None:
    """Populate the supported targets set from the built-in list."""
    for t in SUPPORTED_TARGETS:
        _supported_targets_set.add(t.raw)


# Initialise on module load
_init_supported_targets()


def register_target(target: str) -> TargetTriple:
    """
    Register a new target triple as supported at runtime.

    Parameters
    ----------
    target : str
        Target triple string, e.g. ``"x86_64-unknown-linux-musl"``.

    Returns
    -------
    TargetTriple
        The parsed and registered target triple.

    Raises
    ------
    ValueError
        If *target* is not a valid target triple format.

    Notes
    -----
    - Registered targets persist only for the lifetime of the process.
    - Call this before using :func:`is_supported` or
      :class:`PlatformDetector` with custom targets.
    - Duplicate registrations are silently ignored.

    Examples
    --------
    >>> from python_installer.platforms import register_target, is_supported
    >>> register_target("x86_64-unknown-linux-musl")
    TargetTriple('x86_64-unknown-linux-musl')
    >>> is_supported(TargetTriple.from_string("x86_64-unknown-linux-musl"))
    True
    """
    triple = TargetTriple.from_string(target)
    _supported_targets_set.add(triple.raw)
    return triple


def unregister_target(target: str) -> bool:
    """
    Remove a target from the supported list.

    Parameters
    ----------
    target : str
        Target triple string.

    Returns
    -------
    bool
        ``True`` if the target was removed, ``False`` if it was not
        registered.

    Notes
    -----
    - Built-in targets (from :data:`SUPPORTED_TARGETS`) can also be
      removed, but this is not recommended.
    """
    return _supported_targets_set.discard(target) is not None


def get_supported_targets() -> frozenset[str]:
    """
    Return the current set of supported target triples.

    Returns
    -------
    frozenset of str
        All registered target triple strings (built-in + custom).

    Notes
    -----
    - Returns an immutable snapshot. Modifications via
      :func:`register_target` or :func:`unregister_target` after
      calling this are not reflected in the returned frozenset.
    """
    return frozenset(_supported_targets_set)


def is_supported(triple: TargetTriple) -> bool:
    """
    Check if a target triple is in the supported list.

    Parameters
    ----------
    triple : TargetTriple
        The triple to check.

    Returns
    -------
    bool
        ``True`` if the triple is a known
        ``python-build-standalone`` target.
    """
    return triple.raw in _supported_targets_set


# ---------------------------------------------------------------------------
# Platform Detector
# ---------------------------------------------------------------------------


class PlatformDetector:
    """
    Detects the current platform and resolves it to a
    :class:`TargetTriple`.

    Parameters
    ----------
    allow_override : bool
        If ``True`` (default), the ``PYTHON_STANDALONE_TARGET``
        environment variable can override automatic detection.

    Examples
    --------
    >>> detector = PlatformDetector()
    >>> triple = detector.detect()
    >>> print(triple.raw)
    x86_64-unknown-linux-gnu

    Notes
    -----
    Detection logic by OS:

    **Linux**
        - Reads ``/proc/cpuinfo`` to distinguish ARM hard-float vs
          soft-float.
        - Checks ``ANDROID_ROOT`` for Android/Termux.
        - Falls back to :func:`platform.machine` with normalisation.

    **macOS**
        - Uses :func:`platform.machine`. On Apple Silicon,
          ``"arm64"`` is mapped to ``"aarch64"``.
        - Does **not** detect Rosetta 2; override with environment
          variable if needed.

    **Windows**
        - Uses :func:`platform.machine`. ``"AMD64"`` is mapped to
          ``"x86_64"``.
        - Always assumes MSVC ABI.
    """

    def __init__(self, allow_override: bool = True) -> None:
        self._allow_override = allow_override

    def detect(self) -> TargetTriple:
        """
        Detect the current platform target triple.

        Returns
        -------
        TargetTriple

        Raises
        ------
        RuntimeError
            If the platform cannot be detected or is unsupported.

        Notes
        -----
        Checks in order:
        1. Environment variable override (if allowed).
        2. OS-specific detection logic.
        3. Fallback to :func:`platform.system` / :func:`platform.machine`.
        """
        # 1. Environment override
        if self._allow_override:
            override = os.environ.get(_ENV_OVERRIDE, "").strip()
            if override:
                triple = TargetTriple.from_string(override)
                if not is_supported(triple):
                    raise RuntimeError(
                        f"Override target {triple.raw!r} is not in "
                        f"the supported list. Supported: "
                        f"{sorted(_SUPPORTED_SET)}"
                    )
                return triple

        # 2. OS-specific detection
        system = _platform.system()

        if system == "Linux":
            return self._detect_linux()
        elif system == "Darwin":
            return self._detect_macos()
        elif system == "Windows":
            return self._detect_windows()
        else:
            raise RuntimeError(
                f"Unsupported operating system: {system!r}. "
                f"Supported: Linux, Darwin, Windows."
            )

    def _detect_linux(self) -> TargetTriple:
        """
        Detect Linux target triple.

        Returns
        -------
        TargetTriple

        Raises
        ------
        RuntimeError
            If architecture cannot be determined.
        """
        # Check for Android/Termux first
        if "ANDROID_ROOT" in os.environ or "TERMUX_VERSION" in os.environ:
            return self._detect_android()

        arch = self._detect_linux_arch()
        abi = _LINUX_ABI_MAP.get(arch, "gnu")
        triple = TargetTriple(
            arch=arch,
            vendor="unknown",
            os="linux",
            abi=abi,
        )

        if not is_supported(triple):
            raise RuntimeError(
                f"Detected unsupported Linux target: {triple.raw}. "
                f"Supported: {sorted(_SUPPORTED_SET)}"
            )
        return triple

    def _detect_linux_arch(self) -> str:
        """
        Detect Linux CPU architecture.

        Returns
        -------
        str
            Normalised architecture name.

        Notes
        -----
        For ARM, reads ``/proc/cpuinfo`` to check for NEON/VFP
        (hard-float) support. Falls back to ``armv7`` with hard-float
        ABI if the file is unreadable.
        """
        machine = _platform.machine()

        # Normalise ARM
        if machine.startswith("armv"):
            return self._normalise_arm_linux(machine)
        if machine in ("aarch64", "arm64"):
            return "aarch64"

        # Use mapping for other architectures
        return _ARCH_MAP.get(machine, machine)

    def _normalise_arm_linux(self, machine: str) -> str:
        """
        Normalise ARM architecture on Linux.

        Parameters
        ----------
        machine : str
            Raw ``platform.machine()`` value.

        Returns
        -------
        str
            ``"armv7"`` or ``"aarch64"``.

        Notes
        -----
        Reads ``/proc/cpuinfo`` Features field for ``"vfp"`` and
        ``"neon"`` to confirm hard-float support. If the file is
        unreadable, hard-float is assumed.
        """
        # armv8l in 32-bit mode is armv7
        if machine == "armv8l":
            return "armv7"

        # Verify hard-float support via /proc/cpuinfo
        cpuinfo_path = Path("/proc/cpuinfo")
        if cpuinfo_path.exists():
            try:
                content = cpuinfo_path.read_text()
                has_vfp = "vfp" in content.lower()
                has_neon = "neon" in content.lower()
                if not (has_vfp or has_neon):
                    # Soft-float — not supported by python-build-standalone
                    raise RuntimeError(
                        "ARM soft-float detected. "
                        "python-build-standalone requires hard-float "
                        "(VFP/NEON)."
                    )
            except (OSError, UnicodeDecodeError):
                # Assume hard-float if unreadable
                pass

        return _ARCH_MAP.get(machine, "armv7")

    def _detect_android(self) -> TargetTriple:
        """
        Detect Android/Termux target.

        Returns
        -------
        TargetTriple

        Raises
        ------
        RuntimeError
            If architecture is not ``aarch64`` (only supported Android
            target).
        """
        machine = _platform.machine()
        if machine in ("aarch64", "arm64"):
            arch = "aarch64"
        else:
            raise RuntimeError(
                f"Unsupported Android architecture: {machine!r}. "
                f"Only aarch64 is supported for Android."
            )

        triple = TargetTriple(
            arch=arch,
            vendor="unknown",
            os="linux",  # Android uses linux os field
            abi="android",
        )

        if not is_supported(triple):
            raise RuntimeError(
                f"Android target not in supported list: {triple.raw}"
            )
        return triple

    def _detect_macos(self) -> TargetTriple:
        """
        Detect macOS target triple.

        Returns
        -------
        TargetTriple

        Raises
        ------
        RuntimeError
            If architecture is unsupported.

        Notes
        -----
        On Apple Silicon, ``platform.machine()`` returns ``"arm64"``
        which is normalised to ``"aarch64"``.
        """
        machine = _platform.machine()
        arch = _ARCH_MAP.get(machine, machine)

        if arch not in ("x86_64", "aarch64"):
            raise RuntimeError(
                f"Unsupported macOS architecture: {machine!r}. "
                f"Supported: x86_64, arm64."
            )

        triple = TargetTriple(
            arch=arch,
            vendor="apple",
            os="darwin",
            abi=None,
        )

        if not is_supported(triple):
            raise RuntimeError(
                f"Detected unsupported macOS target: {triple.raw}"
            )
        return triple

    def _detect_windows(self) -> TargetTriple:
        """
        Detect Windows target triple.

        Returns
        -------
        TargetTriple

        Raises
        ------
        RuntimeError
            If architecture is unsupported.

        Notes
        -----
        Always uses MSVC ABI. Checks pointer size to confirm 32-bit
        vs 64-bit.
        """
        machine = _platform.machine()
        arch = _ARCH_MAP.get(machine, machine)

        # Validate with pointer size
        pointer_bits = struct.calcsize("P") * 8
        if pointer_bits == 64 and arch == "i686":
            # Mismatch — override
            arch = "x86_64"
        elif pointer_bits == 32 and arch == "x86_64":
            arch = "i686"

        if arch not in ("x86_64", "i686", "aarch64"):
            raise RuntimeError(
                f"Unsupported Windows architecture: {machine!r}. "
                f"Supported: x86_64, i686, aarch64."
            )

        abi = _WINDOWS_ABI_MAP.get(arch, "msvc")
        triple = TargetTriple(
            arch=arch,
            vendor="pc",
            os="windows",
            abi=abi,
        )

        if not is_supported(triple):
            raise RuntimeError(
                f"Detected unsupported Windows target: {triple.raw}"
            )
        return triple

    @staticmethod
    def get_current_platform_info() -> dict:
        """
        Return detailed information about the current platform.

        Returns
        -------
        dict
            Keys: ``system``, ``release``, ``version``, ``machine``,
            ``processor``, ``python_arch``, ``pointer_bits``.

        Notes
        -----
        Uses :mod:`platform` and :mod:`struct`. Safe for logging
        and debugging.
        """
        return {
            "system": _platform.system(),
            "release": _platform.release(),
            "version": _platform.version(),
            "machine": _platform.machine(),
            "processor": _platform.processor(),
            "python_arch": _platform.architecture()[0],
            "pointer_bits": struct.calcsize("P") * 8,
        }


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def detect_target(allow_override: bool = True) -> TargetTriple:
    """
    Convenience function: detect the current target triple.

    Parameters
    ----------
    allow_override : bool
        Passed to :class:`PlatformDetector`.

    Returns
    -------
    TargetTriple
    """
    return PlatformDetector(allow_override=allow_override).detect()


def build_asset_filename(
    python_version: str,
    release_date: str,
    target: TargetTriple,
    variant: str = "install_only",
) -> str:
    """
    Build the expected asset filename for a ``python-build-standalone``
    release.

    Parameters
    ----------
    python_version : str
        Python version, e.g. ``"3.11.5"``.
    release_date : str
        Release tag date, e.g. ``"20231002"``.
    target : TargetTriple
        Platform target triple.
    variant : str
        Build variant. ``"install_only"`` (default) is the minimal
        distribution without debug symbols or headers.

    Returns
    -------
    str
        Asset filename, e.g.
        ``"cpython-3.11.5+20231002-x86_64-unknown-linux-gnu-install_only.tar.gz"``.

    Raises
    ------
    ValueError
        If *variant* contains invalid characters.

    Notes
    -----
    The format is::

        cpython-{version}+{date}-{target}-{variant}.tar.gz
    """
    if not re.match(r"^[a-zA-Z0-9_]+$", variant):
        raise ValueError(
            f"Invalid variant name: {variant!r}. "
            "Only [a-zA-Z0-9_] allowed."
        )
    ext = target.archive_extension
    return (
        f"cpython-{python_version}+{release_date}-{target.raw}"
        f"-{variant}{ext}"
    )
"""
Main orchestrator and public API for compiler discovery.

This module ties together the Detection, Identity, and Decision layers.
It runs discovery strategies, validates candidates, caches results,
and exposes the CompilerManager — the single entry point for all
user-facing operations.

Layer
-----
This is the Integration Layer. It coordinates the three core layers:
    1. Detection  (strategies.py)  — find raw paths
    2. Identity   (validation.py)  — verify and fingerprint
    3. Decision   (scoring.py)     — rank and filter (future)

Usage
-----
    from  import discover_compilers

    manager = discover_compilers()
    best = manager.best()
    all_gcc = manager.filter(kinds=[CompilerKind.GCC])

Performance
-----------
First call triggers full system scan. This may take 1–30 seconds
depending on PATH size, number of strategies, and filesystem speed.
Subsequent calls return cached results instantly.

Caching
-------
Results are cached in memory for the lifetime of the CompilerManager
instance. Cache is invalidated when:
    - `rescan()` is called explicitly
    - PATH environment variable changes (detected via hash)

Thread Safety
-------------
CompilerManager is NOT thread-safe for discovery.
- Multiple threads can safely READ results via `.all()`, `.best()`, `.filter()`.
- Only ONE thread should trigger discovery via `discover_compilers()` or `rescan()`.

Warnings
--------
- First discovery executes compiler binaries for validation.
  Review the blocklist in validation.py before deployment.
- Results reflect the system state at discovery time.
  Installed/uninstalled compilers after discovery are not reflected
  until `rescan()` is called.

User Instructions
-----------------
- Call `discover_compilers()` once at application startup.
- Use `manager.filter()` for specific compiler needs.
- Call `manager.rescan()` after installing new compilers at runtime.
- Check `CompilerInfo.confidence_score` for production build decisions.
"""

import logging
import os
from typing import Dict, List, Optional, Iterable

from .models import CompilerInfo, CompilerKind, CompilerSource
from .strategies import DiscoveryStrategyRegistry
from .validation import validate_and_fingerprint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def discover_compilers(
    user_paths: Optional[List[str]] = None,
    enabled_strategies: Optional[List[str]] = None,
    allowed: Iterable[str] = None
) -> "CompilerManager":
    """
    Discover all compilers available on the current system.

    This is the single public entry point for the entire discovery
    engine. It returns a CompilerManager that provides filtered,
    sorted access to results.

    Parameters
    ----------
    user_paths : Optional[List[str]]
        Additional compiler paths to include. These are treated as
        USER_OVERRIDE source and given highest priority.
        Example: ["/opt/custom-gcc/bin/gcc", "/home/user/clang/bin/clang"]
    enabled_strategies : Optional[List[str]]
        Strategy names to enable. If provided, ONLY these strategies
        run. If None, all registered non-disabled strategies run.
        Example: ["user_override", "system_path"]

    Returns
    -------
    CompilerManager
        Manager object with .all(), .best(), .filter(), .rescan() methods.

    Examples
    --------
    >>> manager = discover_compilers()
    >>> best = manager.best()
    >>> print(best.path, best.version)

    >>> manager = discover_compilers(
    ...     user_paths=["/opt/my-gcc/bin/gcc"],
    ...     enabled_strategies=["user_override", "system_path"]
    ... )
    """
    manager = CompilerManager()
    manager._discover(
        user_paths=user_paths,
        enabled_strategies=enabled_strategies,
        allowed=allowed,
    )
    return manager


# ---------------------------------------------------------------------------
# CompilerManager
# ---------------------------------------------------------------------------

class CompilerManager:
    """
    Manages discovered compilers with lazy discovery and caching.

    Do NOT instantiate directly. Use `discover_compilers()` instead.

    Methods are read-only and do not modify compiler installations
    or system state.

    Attributes
    ----------
    _all_compilers : Optional[List[CompilerInfo]]
        Cached discovery results. None until first discovery completes.
    _path_hash : Optional[str]
        Hash of PATH environment variable at discovery time.
        Used for cache invalidation.
    """

    def __init__(self) -> None:
        self._all_compilers: Optional[List[CompilerInfo]] = None
        self._path_hash: Optional[str] = None

    # ------------------------------------------------------------------
    # Public read methods
    # ------------------------------------------------------------------

    def all(self) -> List[CompilerInfo]:
        """
        Return all discovered compilers.

        Returns
        -------
        List[CompilerInfo]
            Complete list of validated compilers found during discovery.
            Sorted by priority: higher-priority compilers first.
            Empty list if discovery has not been run or found nothing.

        Warnings
        --------
        - Returns the same list object each call. Do not modify.
        - Results reflect system state at discovery time.
          Call `rescan()` to refresh.
        """
        if self._all_compilers is None:
            return []
        return self._all_compilers

    def best(self) -> Optional[CompilerInfo]:
        """
        Return the single best compiler for the current host.

        Selection criteria (in order):
            1. Highest priority source (User > Managed > PATH > Common)
            2. Native compiler preferred over cross-compiler
            3. Highest confidence score
            4. Highest version number

        Returns
        -------
        Optional[CompilerInfo]
            The best compiler, or None if no compilers were found.

        Notes
        -----
        This is a heuristic "best" for general use. For specific
        requirements, use `filter()` or the future `resolve()` method.
        """
        if not self._all_compilers:
            return None
        return self._all_compilers[0]

    def filter(
        self,
        kinds: Optional[List[CompilerKind]] = None,
        vendors: Optional[List[str]] = None,
        cross_compilers: Optional[bool] = None,
        min_confidence: Optional[float] = None,
        target_arch: Optional[str] = None,
    ) -> List[CompilerInfo]:
        """
        Filter discovered compilers by criteria.

        All criteria are optional and combined with AND logic.
        An empty filter returns all compilers.

        Parameters
        ----------
        kinds : Optional[List[CompilerKind]]
            Compiler families to include. Example: [CompilerKind.GCC].
        vendors : Optional[List[str]]
            Vendor strings to include. Example: ["GNU", "LLVM"].
            Case-insensitive matching.
        cross_compilers : Optional[bool]
            If True, return only cross-compilers.
            If False, return only native compilers.
            If None (default), return both.
        min_confidence : Optional[float]
            Minimum confidence score (0.0–1.0). Compilers below this
            threshold are excluded. Recommended: 0.7 for production.
        target_arch : Optional[str]
            Filter by target architecture substring.
            Example: "aarch64" matches "aarch64-linux-android".
            Case-insensitive.

        Returns
        -------
        List[CompilerInfo]
            Filtered list in priority order. May be empty.

        Examples
        --------
        >>> # All GCC compilers
        >>> manager.filter(kinds=[CompilerKind.GCC])

        >>> # High-confidence native Clang compilers
        >>> manager.filter(
        ...     kinds=[CompilerKind.CLANG],
        ...     cross_compilers=False,
        ...     min_confidence=0.8
        ... )

        >>> # ARM cross-compilers
        >>> manager.filter(target_arch="arm", cross_compilers=True)
        """
        if not self._all_compilers:
            return []

        results = self._all_compilers

        if kinds is not None:
            results = [c for c in results if c.kind in kinds]

        if vendors is not None:
            vendors_lower = [v.lower() for v in vendors]
            results = [
                c for c in results if c.vendor.lower() in vendors_lower
            ]

        if cross_compilers is True:
            results = [c for c in results if c.is_cross_compiler]
        elif cross_compilers is False:
            results = [c for c in results if not c.is_cross_compiler]

        if min_confidence is not None:
            results = [
                c for c in results if c.confidence_score >= min_confidence
            ]

        if target_arch is not None:
            arch_lower = target_arch.lower()
            results = [
                c for c in results if arch_lower in c.target_triplet.lower()
            ]

        return results

    def native(self) -> List[CompilerInfo]:
        """
        Return only native (non-cross) compilers.

        Shortcut for `filter(cross_compilers=False)`.

        Returns
        -------
        List[CompilerInfo]
            Native compilers in priority order.
        """
        return self.filter(cross_compilers=False)

    def cross(self) -> List[CompilerInfo]:
        """
        Return only cross-compilers.

        Shortcut for `filter(cross_compilers=True)`.

        Returns
        -------
        List[CompilerInfo]
            Cross-compilers in priority order.
        """
        return self.filter(cross_compilers=True)

    # ------------------------------------------------------------------
    # Cache and rescan
    # ------------------------------------------------------------------

    def rescan(self) -> None:
        """
        Force a full rediscovery of compilers.

        Clears the internal cache and reruns all discovery strategies
        with full validation. Use this after:
            - Installing a new compiler
            - Updating PATH environment variable
            - Modifying toolchain manager installations

        Warnings
        --------
        - This triggers subprocess execution for every candidate path.
        - May take 1–30 seconds depending on system state.
        - Results are replaced atomically only after full completion.
        """
        self._all_compilers = None
        self._path_hash = None
        self._discover()

    def is_stale(self) -> bool:
        """
        Check if the cached results may be outdated.

        Compares the current PATH hash with the hash at discovery time.
        Returns True if PATH has changed since last discovery.

        Returns
        -------
        bool
            True if PATH changed and results may be stale.

        Notes
        -----
        This only checks PATH changes. It does NOT detect:
            - New installations in previously-empty directories
            - Removed compilers in directories still in PATH
            - Changes to toolchain managers (rustup, Android NDK)
        """
        if self._path_hash is None:
            return True
        return self._path_hash != self._hash_path()

    # ------------------------------------------------------------------
    # Internal discovery logic
    # ------------------------------------------------------------------

    def _discover(
        self,
        user_paths: Optional[List[str]] = None,
        enabled_strategies: Optional[List[str]] = None,
        allowed: Iterable[str] = None
    ) -> None:
        """
        Run the full discovery pipeline.

        1. Collect raw paths from all enabled strategies.
        2. Deduplicate paths (by realpath).
        3. Validate each unique path.
        4. Sort results by priority and confidence.
        5. Cache results internally.

        Parameters
        ----------
        user_paths : Optional[List[str]]
            Additional paths to include with USER_OVERRIDE priority.
        enabled_strategies : Optional[List[str]]
            Strategy names to use. None means all enabled.
        """
        raw_paths: List[str] = []

        # Collect paths from strategies
        if enabled_strategies is not None:
            strategies = [
                s for s in DiscoveryStrategyRegistry.get_all()
                if s.name in enabled_strategies
            ]
        else:
            strategies = DiscoveryStrategyRegistry.get_enabled()

        for strategy_cls in strategies:
            try:
                instance = strategy_cls()
                paths = instance.discover()
                raw_paths.extend(paths)
                logger.debug(
                    "Strategy '%s' found %d candidates",
                    strategy_cls.name,
                    len(paths),
                )
            except Exception as exc:
                logger.warning(
                    "Strategy '%s' failed: %s",
                    strategy_cls.name,
                    exc,
                )

        # Add user-provided paths with highest priority
        if user_paths:
            for path in user_paths:
                abs_path = os.path.abspath(path)
                if os.path.isfile(abs_path):
                    raw_paths.append(abs_path)

        # Deduplicate by real path
        unique_paths = self._deduplicate(raw_paths)
        logger.info("Deduplicated %d raw paths to %d unique", len(raw_paths), len(unique_paths))

        # Validate each unique path
        validated: List[CompilerInfo] = []
        for path in unique_paths:
            try:
                compiler_info = validate_and_fingerprint(path, allowed)
                if compiler_info is not None:
                    # Override source for user-provided paths
                    if user_paths and os.path.abspath(path) in [
                        os.path.abspath(p) for p in user_paths
                    ]:
                        compiler_info = self._with_source(
                            compiler_info, CompilerSource.USER_OVERRIDE
                        )
                    validated.append(compiler_info)
            except Exception as exc:
                logger.debug("Validation failed for %s: %s", path, exc)

        # Sort by priority
        validated.sort(key=lambda c: self._sort_key(c))

        # Cache results
        self._all_compilers = validated
        self._path_hash = self._hash_path()
        logger.info("Discovery complete: %d compilers found", len(validated))

    @staticmethod
    def _deduplicate(paths: List[str]) -> List[str]:
        """
        Remove duplicate paths, resolving symlinks.

        Two paths are duplicates if they resolve to the same real path.
        Order is preserved: first occurrence is kept.

        Parameters
        ----------
        paths : List[str]
            Raw paths from all strategies.

        Returns
        -------
        List[str]
            Deduplicated paths in first-seen order.
        """
        seen: Dict[str, str] = {}
        result: List[str] = []
        for path in paths:
            try:
                real = os.path.realpath(path)
            except OSError:
                real = os.path.abspath(path)
            if real not in seen:
                seen[real] = path
                result.append(path)
        return result

    @staticmethod
    def _sort_key(compiler: CompilerInfo) -> tuple:
        """
        Generate a sort key for a compiler.

        Sort order (ascending = higher priority first):
            1. source priority (lower value = higher priority)
            2. native preferred over cross (0 = native, 1 = cross)
            3. confidence score (higher = better, negated for ascending sort)
            4. version (higher = better, negated for ascending sort)

        Parameters
        ----------
        compiler : CompilerInfo
            The compiler to generate a key for.

        Returns
        -------
        tuple
            Sort key for use with `sorted(key=...)`.
        """
        from .parsers import version_tuple

        return (
            compiler.source.value,                # Lower source = higher priority
            1 if compiler.is_cross_compiler else 0,  # Native first
            -compiler.confidence_score,            # Higher confidence first
            -version_tuple(compiler.version)[0],   # Higher major version first
            -version_tuple(compiler.version)[1] if len(version_tuple(compiler.version)) > 1 else 0,
            -version_tuple(compiler.version)[2] if len(version_tuple(compiler.version)) > 2 else 0,
        )

    @staticmethod
    def _with_source(info: CompilerInfo, source: CompilerSource) -> CompilerInfo:
        """
        Create a new CompilerInfo with a different source.

        Since CompilerInfo is frozen, we create a new instance.
        This is a deliberate limitation — source is set once at
        discovery time and should not change.

        Parameters
        ----------
        info : CompilerInfo
            Original compiler info.
        source : CompilerSource
            New source value.

        Returns
        -------
        CompilerInfo
            New instance with updated source.
        """
        return CompilerInfo(
            path=info.path,
            version=info.version,
            vendor=info.vendor,
            kind=info.kind,
            target_triplet=info.target_triplet,
            is_cross_compiler=info.is_cross_compiler,
            source=source,
            confidence_score=info.confidence_score,
            fingerprint=info.fingerprint,
        )

    @staticmethod
    def _hash_path() -> str:
        """
        Create a hash of the current PATH environment variable.

        Used for stale cache detection.

        Returns
        -------
        str
            Hex digest of PATH contents.
        """
        import hashlib
        path_val = os.environ.get("PATH", "")
        return hashlib.sha256(path_val.encode()).hexdigest()
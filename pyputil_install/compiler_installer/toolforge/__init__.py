"""
ToolForge
=========

Compiler discovery, validation, and selection engine.

Architecture
------------
The project is split into layers. Each layer has one responsibility.

Detection Layer
    Finds raw executable paths. No execution happens here.
    Module: strategies.py

Identity Layer
    Validates paths as real compilers. This is the only layer
    that executes binaries. Extracts version, vendor, target.
    Modules: validation.py, parsers.py

Decision Layer
    Converts compiler identities into numeric scores for ranking.
    Module: scoring.py

Integration
    Ties the three layers together and provides the public API.
    Module: discovery.py

Data
    Immutable types shared across all layers.
    Module: models.py

Storage
    Filesystem cache to avoid repeated system scans.
    Module: cache.py

Usage
-----
    from pyputil_install.compiler_installer import toolforge

    manager = toolforge.discover_compilers()

    # Single best native compiler
    best = manager.best()

    # Filter by family
    gcc_list = manager.filter(kinds=[toolforge.CompilerKind.GCC])

    # Force rescan after installing a new compiler
    manager.rescan()

Warnings
--------
- First discovery executes binaries for validation. See validation.py
  for the blocklist and security rules.
- Cached results reflect system state at discovery time. Call
  rescan() after installing or removing compilers.
- CompilerInfo.path is guaranteed valid only at discovery time.
  Network mounts or uninstalls may invalidate it later.

Environment Variables
---------------------
COMPILER_EXECUTION_TIMEOUT
    Subprocess timeout in seconds. Default: 5.
COMPILER_SEARCH_CANDIDATES
    Colon-separated list of executable names to search for.
    Default: gcc, g++, cc, c++, clang, clang++, cl.exe.
COMPILER_SEARCH_DIRS
    Colon-separated extra directories to scan.
TOOLFORGE_CACHE_DIR
    Directory for the cache file.
    Default: ~/.cache/toolforge (Unix) or %LOCALAPPDATA%/toolforge (Windows).
TOOLFORGE_CACHE_MAX_AGE
    Maximum cache age in seconds before considered stale. Default: 3600.
TOOLFORGE_CACHE_MONITOR_DIRS
    Colon-separated directories to watch for changes.
"""

# ---------------------------------------------------------------------------
# Public API — main entry points
# ---------------------------------------------------------------------------

from .discovery import CompilerManager, discover_compilers

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

from .models import CompilerInfo, CompilerKind, CompilerSource

# ---------------------------------------------------------------------------
# Validation (for advanced use: manual path validation)
# ---------------------------------------------------------------------------

from .validation import validate_and_fingerprint

# ---------------------------------------------------------------------------
# Parsers (for advanced use: parsing raw compiler output)
# ---------------------------------------------------------------------------

from .parsers import (
    parse_version_output,
    parse_kind_and_vendor,
    parse_target_triplet,
    split_target_triplet,
    normalize_output,
    version_tuple,
)

# ---------------------------------------------------------------------------
# Strategies (for advanced use: custom discovery plugins)
# ---------------------------------------------------------------------------

from .strategies import (
    DiscoveryStrategy,
    DiscoveryStrategyRegistry,
    UserOverrideStrategy,
    ManagedToolchainStrategy,
    PATHStrategy,
    ExtraDirsStrategy,
    CommonDirsStrategy,
)

# ---------------------------------------------------------------------------
# Scoring (for advanced use: custom ranking logic)
# ---------------------------------------------------------------------------

from .scoring import (
    score_compiler,
    rank_compilers,
    best_compiler,
    score_for_native_build,
    score_for_cross_compilation,
    explain_score,
    SCORING_WEIGHTS,
)

# ---------------------------------------------------------------------------
# Cache (for advanced use: direct cache control)
# ---------------------------------------------------------------------------

from .cache import CompilerCache

# ---------------------------------------------------------------------------
# What `import toolforge` exposes
# ---------------------------------------------------------------------------

__all__ = [
    # --- Public API ---
    "discover_compilers",
    "CompilerManager",
    # --- Data models ---
    "CompilerInfo",
    "CompilerKind",
    "CompilerSource",
    # --- Validation ---
    "validate_and_fingerprint",
    # --- Parsers ---
    "parse_version_output",
    "parse_kind_and_vendor",
    "parse_target_triplet",
    "split_target_triplet",
    "normalize_output",
    "version_tuple",
    # --- Strategies ---
    "DiscoveryStrategy",
    "DiscoveryStrategyRegistry",
    "UserOverrideStrategy",
    "ManagedToolchainStrategy",
    "PATHStrategy",
    "ExtraDirsStrategy",
    "CommonDirsStrategy",
    # --- Scoring ---
    "score_compiler",
    "rank_compilers",
    "best_compiler",
    "score_for_native_build",
    "score_for_cross_compilation",
    "explain_score",
    "SCORING_WEIGHTS",
    # --- Cache ---
    "CompilerCache",
]
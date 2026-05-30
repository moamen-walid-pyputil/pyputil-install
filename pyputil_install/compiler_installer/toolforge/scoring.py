"""
Compiler scoring and ranking utilities.

This module provides the scoring functions used to rank compilers
for specific tasks. It belongs to the Decision Layer — it receives
validated CompilerInfo objects and produces numeric scores that
determine which compiler is "best" for a given purpose.

Scope
-----
- Convert compiler attributes into numeric scores
- Apply penalties and bonuses based on task requirements
- Provide the default ranking used by CompilerManager

This module does NOT:
    - Discover or validate compilers (use strategies.py / validation.py)
    - Execute any binaries
    - Modify CompilerInfo objects

Scoring Philosophy
------------------
Simple sorting (by version, by path) is inadequate. Real compiler
selection requires a weighted scoring function that accounts for:
    - Discovery source priority
    - Native vs cross-compilation
    - Confidence in the compiler's identity
    - Version number
    - Vendor preferences
    - Known wrapper penalties

Usage
-----
    from .scoring import score_compiler, rank_compilers

    # Score a single compiler for general use
    score = score_compiler(compiler_info)

    # Rank all compilers for a specific task
    ranked = rank_compilers(compilers, prefer_native=True)

Warnings
--------
- Scores are relative within a single discovery run. Do not compare
  scores across different systems or discovery sessions.
- The scoring function is intentionally simple. Complex task-specific
  resolution will be added in a future `resolution.py` module.

User Instructions
-----------------
- Use `score_compiler()` directly only when building custom selection logic.
- For standard use, rely on `CompilerManager.best()` and `CompilerManager.filter()`.
- The scoring weights are tuned for general-purpose use. Adjust
  SCORING_WEIGHTS if your use case has different priorities.
"""

from typing import Dict, List, Optional

from .models import CompilerInfo, CompilerKind, CompilerSource
from .parsers import version_tuple


# ---------------------------------------------------------------------------
# Scoring weights — tunable for different use cases
# ---------------------------------------------------------------------------

SCORING_WEIGHTS: Dict[str, float] = {
    "source_bonus": 100.0,       # Base score per source priority level
    "native_bonus": 50.0,        # Bonus for native (non-cross) compilers
    "confidence_multiplier": 1.0, # Multiplier applied to confidence score
    "major_version_weight": 10.0, # Weight per major version number
    "minor_version_weight": 1.0,  # Weight per minor version number
    "patch_version_weight": 0.1,  # Weight per patch version number
    "gcc_bonus": 0.0,            # Neutral by default — no vendor preference
    "clang_bonus": 0.0,          # Neutral by default — no vendor preference
    "mingw_penalty": -15.0,      # MinGW wrappers are less reliable
    "unknown_vendor_penalty": -20.0,  # Unknown vendors are risky
    "low_confidence_penalty": -30.0,  # Applied when confidence < 0.7
}


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------

def score_compiler(
    compiler: CompilerInfo,
    prefer_native: bool = True,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate a numeric score for a compiler.

    Higher scores indicate better compilers. Scores are composed of
    bonuses, penalties, and multipliers.

    Parameters
    ----------
    compiler : CompilerInfo
        The compiler to score.
    prefer_native : bool
        If True, native compilers receive a bonus. Set to False when
        deliberately searching for cross-compilers.
    weights : Optional[Dict[str, float]]
        Custom scoring weights. If None, uses SCORING_WEIGHTS defaults.
        Provide a dict with any subset of keys to override only those values.

    Returns
    -------
    float
        Numeric score. Higher is better. May be negative for very
        low-confidence or penalized compilers.

    Score Components
    ----------------
    1. Source priority: higher-priority sources get higher base scores.
       Source priority order: USER_OVERRIDE > MANAGED_TOOLCHAIN >
       SYSTEM_PATH > COMMON_DIR > FALLBACK.
       Base score = source_bonus * (max_source_value - source_value)

    2. Native bonus: added when compiler is not a cross-compiler
       and prefer_native is True.

    3. Version score: major * major_weight + minor * minor_weight
       + patch * patch_weight.

    4. Confidence multiplier: total is multiplied by
       (confidence_score * confidence_multiplier).

    5. Vendor adjustments: GCC/Clang bonuses, MinGW penalty,
       unknown vendor penalty.

    6. Low confidence penalty: applied when confidence_score < 0.7.

    Examples
    --------
    >>> score_compiler(gcc_native)  # High score
    165.0
    >>> score_compiler(unknown_wrapper)  # Low or negative score
    -20.0
    """
    w = SCORING_WEIGHTS.copy()
    if weights:
        w.update(weights)

    score = 0.0

    # 1. Source priority — higher source = lower numeric value
    max_source = max(s.value for s in CompilerSource)
    score += w["source_bonus"] * (max_source - compiler.source.value)

    # 2. Native bonus
    if prefer_native and not compiler.is_cross_compiler:
        score += w["native_bonus"]

    # 3. Version score
    version_parts = version_tuple(compiler.version)
    if len(version_parts) >= 1:
        score += version_parts[0] * w["major_version_weight"]
    if len(version_parts) >= 2:
        score += version_parts[1] * w["minor_version_weight"]
    if len(version_parts) >= 3:
        score += version_parts[2] * w["patch_version_weight"]

    # 4. Vendor adjustments
    if compiler.kind == CompilerKind.GCC:
        score += w["gcc_bonus"]
    elif compiler.kind == CompilerKind.CLANG:
        score += w["clang_bonus"]

    if compiler.vendor == "MinGW":
        score += w["mingw_penalty"]
    elif compiler.vendor == "unknown":
        score += w["unknown_vendor_penalty"]

    # 5. Low confidence penalty
    if compiler.confidence_score < 0.7:
        score += w["low_confidence_penalty"]

    # 6. Confidence multiplier — applied last
    score *= compiler.confidence_score * w["confidence_multiplier"]

    return score


def rank_compilers(
    compilers: List[CompilerInfo],
    prefer_native: bool = True,
    weights: Optional[Dict[str, float]] = None,
) -> List[CompilerInfo]:
    """
    Sort a list of compilers by score, highest first.

    Parameters
    ----------
    compilers : List[CompilerInfo]
        Compilers to rank.
    prefer_native : bool
        Passed to score_compiler(). Set False when ranking cross-compilers.
    weights : Optional[Dict[str, float]]
        Passed to score_compiler().

    Returns
    -------
    List[CompilerInfo]
        New list sorted by score descending. Original list is unchanged.

    Warnings
    --------
    - Equal scores preserve original relative order (stable sort).
    - Do not mutate the returned list — it shares CompilerInfo objects
      with the input list.
    """
    return sorted(
        compilers,
        key=lambda c: score_compiler(c, prefer_native=prefer_native, weights=weights),
        reverse=True,
    )


def best_compiler(
    compilers: List[CompilerInfo],
    prefer_native: bool = True,
    weights: Optional[Dict[str, float]] = None,
) -> Optional[CompilerInfo]:
    """
    Return the single highest-scoring compiler.

    Parameters
    ----------
    compilers : List[CompilerInfo]
        Compilers to evaluate.
    prefer_native : bool
        Passed to score_compiler().
    weights : Optional[Dict[str, float]]
        Passed to score_compiler().

    Returns
    -------
    Optional[CompilerInfo]
        The highest-scoring compiler, or None if the list is empty.

    Notes
    -----
    This is equivalent to `rank_compilers(...)[0]` but slightly faster
    as it avoids a full sort.
    """
    if not compilers:
        return None

    return max(
        compilers,
        key=lambda c: score_compiler(c, prefer_native=prefer_native, weights=weights),
    )


# ---------------------------------------------------------------------------
# Task-specific scoring presets
# ---------------------------------------------------------------------------

def score_for_native_build(compiler: CompilerInfo) -> float:
    """
    Score optimized for native (host-targeting) builds.

    Heavily prefers native compilers over cross-compilers.
    Uses default weights with a higher native bonus.

    Parameters
    ----------
    compiler : CompilerInfo
        The compiler to score.

    Returns
    -------
    float
        Score with native_bonus doubled.
    """
    weights = SCORING_WEIGHTS.copy()
    weights["native_bonus"] = SCORING_WEIGHTS["native_bonus"] * 2.0
    return score_compiler(compiler, prefer_native=True, weights=weights)


def score_for_cross_compilation(compiler: CompilerInfo) -> float:
    """
    Score optimized for cross-compilation selection.

    Does not penalize cross-compilers. Prefers higher version numbers
    and higher confidence.

    Parameters
    ----------
    compiler : CompilerInfo
        The compiler to score.

    Returns
    -------
    float
        Score without native preference.
    """
    return score_compiler(compiler, prefer_native=False)


# ---------------------------------------------------------------------------
# Score explanation (for debugging)
# ---------------------------------------------------------------------------

def explain_score(compiler: CompilerInfo) -> Dict[str, float]:
    """
    Break down a compiler's score into its components.

    Useful for debugging why one compiler ranks above another.

    Parameters
    ----------
    compiler : CompilerInfo
        The compiler to analyze.

    Returns
    -------
    Dict[str, float]
        Component name → score contribution.
        Keys: source, native_bonus, version, vendor_adjustment,
              confidence_penalty, raw_total, final_score.

    Examples
    --------
    >>> explain_score(my_gcc)
    {
        "source": 300.0,
        "native_bonus": 50.0,
        "version": 130.0,
        "vendor_adjustment": 0.0,
        "confidence_penalty": 0.0,
        "raw_total": 480.0,
        "final_score": 470.4,
    }
    """
    w = SCORING_WEIGHTS
    breakdown: Dict[str, float] = {}

    # Source
    max_source = max(s.value for s in CompilerSource)
    breakdown["source"] = w["source_bonus"] * (max_source - compiler.source.value)

    # Native
    breakdown["native_bonus"] = w["native_bonus"] if not compiler.is_cross_compiler else 0.0

    # Version
    version_parts = version_tuple(compiler.version)
    version_score = 0.0
    if len(version_parts) >= 1:
        version_score += version_parts[0] * w["major_version_weight"]
    if len(version_parts) >= 2:
        version_score += version_parts[1] * w["minor_version_weight"]
    if len(version_parts) >= 3:
        version_score += version_parts[2] * w["patch_version_weight"]
    breakdown["version"] = version_score

    # Vendor
    vendor_adj = 0.0
    if compiler.vendor == "MinGW":
        vendor_adj = w["mingw_penalty"]
    elif compiler.vendor == "unknown":
        vendor_adj = w["unknown_vendor_penalty"]
    breakdown["vendor_adjustment"] = vendor_adj

    # Confidence penalty
    conf_penalty = w["low_confidence_penalty"] if compiler.confidence_score < 0.7 else 0.0
    breakdown["confidence_penalty"] = conf_penalty

    # Raw total before confidence multiplier
    raw = (
        breakdown["source"]
        + breakdown["native_bonus"]
        + breakdown["version"]
        + breakdown["vendor_adjustment"]
        + breakdown["confidence_penalty"]
    )
    breakdown["raw_total"] = raw

    # Final
    breakdown["final_score"] = raw * compiler.confidence_score * w["confidence_multiplier"]

    return breakdown
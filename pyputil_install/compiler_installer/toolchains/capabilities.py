"""
Toolchain capability detection.

Detects what language standards, features, and extensions a
toolchain supports by compiling small test programs and checking
the compiler exit code. Each function is independent and caches
results per compiler path to avoid redundant subprocess calls.

Design
------
This module is deliberately separate from the Toolchain classes.
A Toolchain represents what IS installed. This module determines
what the toolchain CAN DO. The separation allows adding new
capability checks without modifying toolchain implementations.

Each detection function follows the same pattern:
    1. Accept a Toolchain object or direct Path to a compiler.
    2. Build a minimal source string that uses the target feature.
    3. Write the source to a temporary file.
    4. Run the compiler with the appropriate flags.
    5. Check the exit code (0 = supported, non-zero = unsupported).
    6. Cache the result for this compiler path.
    7. Return True/False.

Temporary files are created with the appropriate extension (.c for
C, .cpp for C++) so the compiler can auto-detect the language.
Output files (.o) are created alongside and cleaned up after.

The cache is a module-level dictionary keyed by (compiler_path, flag).
It persists for the lifetime of the Python process. Call
clear_detection_cache() to reset it.

Environment
-----------
Detection uses the default environment inherited by the Python
process. If the compiler requires special environment variables
(e.g., MSVC's INCLUDE and LIB), they must be set before calling
these functions.

Subprocess calls use shell=False and have a 15-second timeout
per test. If a compiler hangs, the test returns False rather
than blocking indefinitely.

Usage
-----
    from pathlib import Path
    from pyputil_install.compiler_installer.toolchains.capabilities import (
        supports_cpp20,
        supports_openmp,
        detect_all,
        clear_detection_cache,
    )
    from toolforge.toolchains.gcc import GCCToolchain

    gcc = GCCToolchain(Path("/usr"))

    # Single capability check
    if supports_cpp20(gcc):
        print("C++20 is available")

    # Batch check — runs all known checks
    caps = detect_all(gcc)
    for feature, supported in caps.items():
        print(f"{feature}: {supported}")

    # Clear cache (e.g., after compiler upgrade)
    clear_detection_cache()

Warnings
--------
- Each check spawns a subprocess. Batch checks (detect_all)
  run sequentially and may take several seconds.
- Detection tests COMPILATION only, not linking or execution.
  A feature may compile but fail at runtime due to missing
  runtime libraries or kernel support (particularly sanitizers).
- Cross-compilers: features are detected for the TARGET platform
  as configured. The test program is compiled for the target,
  not the host.
- The cache is NOT persisted to disk. Restarting the process
  re-runs all checks.
- Test source strings use minimal feature usage. Some compilers
  may accept a feature flag but not fully implement it. These
  tests err on the side of false positives (reporting support
  when the implementation is buggy).

User Instructions
-----------------
- Use detect_all() for comprehensive profiling.
- Individual checks are available for build system integration.
- Clear the cache after upgrading or reconfiguring a toolchain.
- For MSVC, call msvc.apply_env() before running capability checks.
"""

import functools
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from .base import Toolchain

logger = logging.getLogger(__name__)

# ============================================================================
# Detection cache
# ============================================================================

# Maps (compiler_path, capability_key) -> bool
_detection_cache: Dict[tuple, bool] = {}


def clear_detection_cache() -> None:
    """
    Clear all cached capability detection results.

    Use this after upgrading or reconfiguring a compiler to force
    re-detection on the next capability check.

    Example
    -------
    >>> clear_detection_cache()
    """
    _detection_cache.clear()


def _cached(key: str) -> Callable:
    """
    Decorator that caches capability results per compiler path.

    The cache key is (compiler_path, capability_key). If the same
    compiler is queried for the same capability twice, the second
    call returns the cached result without spawning a subprocess.

    Parameters
    ----------
    key : str
        Unique identifier for this capability (e.g., "c++20", "lto").

    Returns
    -------
    Callable
        Wrapped function with caching behavior.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(toolchain: Union[Toolchain, Path], *args: Any, **kwargs: Any) -> bool:
            compiler_path = _get_compiler_path(toolchain)
            if compiler_path is None:
                return False

            cache_key = (str(compiler_path), key)
            if cache_key in _detection_cache:
                return _detection_cache[cache_key]

            result = func(toolchain, *args, **kwargs)
            _detection_cache[cache_key] = result
            return result
        return wrapper
    return decorator


# ============================================================================
# Internal: compile a test snippet and check exit code
# ============================================================================

def _compile_test(
    compiler: Path,
    source: str,
    flags: Optional[list] = None,
    language: str = "c++",
    timeout: int = 15,
) -> bool:
    """
    Compile a short source string and return True if exit code is 0.

    Creates a temporary source file with the correct extension,
    compiles it with the given flags, and checks the return code.
    All temporary files are cleaned up after, regardless of success
    or failure.

    Parameters
    ----------
    compiler : Path
        Absolute path to the compiler executable.
    source : str
        Complete source code as a string. Should be a minimal
        program that exercises exactly one feature.
    flags : Optional[list]
        Compiler flags to pass after the source file. Typically
        includes -std= flags and feature flags like -fopenmp.
    language : str
        "c" or "c++". Determines the source file extension and
        compiler invocation. Default "c++".
    timeout : int
        Maximum seconds to wait for compilation. Default 15.
        If exceeded, the test returns False.

    Returns
    -------
    bool
        True if the compiler exited with code 0, False for any
        other exit code, timeout, or subprocess error.
    """
    suffix = ".cpp" if language == "c++" else ".c"

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=suffix,
        delete=False,
    ) as src_file:
        src_file.write(source)
        src_path = src_file.name

    with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as out_file:
        out_path = out_file.name

    try:
        cmd = [str(compiler), "-c", src_path, "-o", out_path]
        if language == "c++":
            cmd.insert(1, "-x")
            cmd.insert(2, "c++")
        if flags:
            cmd.extend(flags)

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            shell=False,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.debug("Capability test timed out after %ds: %s", timeout, " ".join(cmd))
        return False
    except Exception as exc:
        logger.debug("Capability test failed: %s", exc)
        return False
    finally:
        try:
            Path(src_path).unlink(missing_ok=True)
        except OSError:
            pass
        try:
            Path(out_path).unlink(missing_ok=True)
        except OSError:
            pass


# ============================================================================
# C++ Standards
# ============================================================================

@_cached("c++20")
def supports_cpp20(toolchain: Union[Toolchain, Path]) -> bool:
    """
    Check if the toolchain supports the C++20 standard.

    Test: Compiles a program using the spaceship operator (<=>)
    and the <compare> header. This requires full C++20 support,
    not just partial implementation.

    Compiler flag used: -std=c++20

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain to check, or a direct path to a C++ compiler.

    Returns
    -------
    bool
        True if `compiler -std=c++20` compiles a three-way comparison
        test successfully.
    """
    source = """
    #include <compare>
    int main() {
        auto r = (1 <=> 2);
        return (r < 0) ? 0 : 1;
    }
    """
    compiler = _get_compiler_path(toolchain)
    if compiler is None:
        return False
    return _compile_test(compiler, source, ["-std=c++20"])


@_cached("c++17")
def supports_cpp17(toolchain: Union[Toolchain, Path]) -> bool:
    """
    Check if the toolchain supports the C++17 standard.

    Test: Compiles a program using `if constexpr`, a C++17
    feature that requires both parsing and semantic support.

    Compiler flag used: -std=c++17

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain to check.

    Returns
    -------
    bool
        True if the compiler accepts -std=c++17 with if constexpr.
    """
    source = """
    template<typename T>
    auto get_value(T t) {
        if constexpr (sizeof(T) > 4) return *t;
        else return t;
    }
    int main() { return 0; }
    """
    compiler = _get_compiler_path(toolchain)
    if compiler is None:
        return False
    return _compile_test(compiler, source, ["-std=c++17"])


@_cached("c++14")
def supports_cpp14(toolchain: Union[Toolchain, Path]) -> bool:
    """
    Check if the toolchain supports the C++14 standard.

    Test: Compiles a program using a generic lambda (auto parameter),
    introduced in C++14.

    Compiler flag used: -std=c++14

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain to check.

    Returns
    -------
    bool
        True if the compiler accepts -std=c++14 with generic lambdas.
    """
    source = """
    int main() {
        auto f = [](auto x) { return x + 1; };
        return f(1);
    }
    """
    compiler = _get_compiler_path(toolchain)
    if compiler is None:
        return False
    return _compile_test(compiler, source, ["-std=c++14"])


@_cached("c++11")
def supports_cpp11(toolchain: Union[Toolchain, Path]) -> bool:
    """
    Check if the toolchain supports the C++11 standard.

    Test: Compiles a program using `auto` for type deduction
    and range-based for loops.

    Compiler flag used: -std=c++11

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain to check.

    Returns
    -------
    bool
        True if the compiler accepts -std=c++11 with auto and
        range-for.
    """
    source = """
    int main() {
        auto x = 42;
        int arr[] = {1, 2, 3};
        for (auto i : arr) { x += i; }
        return x;
    }
    """
    compiler = _get_compiler_path(toolchain)
    if compiler is None:
        return False
    return _compile_test(compiler, source, ["-std=c++11"])


# ============================================================================
# C Standards
# ============================================================================

@_cached("c11")
def supports_c11(toolchain: Union[Toolchain, Path]) -> bool:
    """
    Check if the toolchain supports the C11 standard.

    Test: Compiles a program using `_Generic`, the C11 generic
    selection feature.

    Compiler flag used: -std=c11
    Language: C (not C++)

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain to check.

    Returns
    -------
    bool
        True if the compiler accepts -std=c11 with _Generic.
    """
    source = """
    #include <stdio.h>
    int main() {
        _Generic((1), int: printf("int\\n"), default: printf("?\\n"));
        return 0;
    }
    """
    compiler = _get_compiler_path(toolchain)
    if compiler is None:
        return False
    return _compile_test(compiler, source, ["-std=c11"], language="c")


# ============================================================================
# Extensions and sanitizers
# ============================================================================

@_cached("openmp")
def supports_openmp(toolchain: Union[Toolchain, Path]) -> bool:
    """
    Check if the toolchain supports OpenMP parallelization.

    Test: Compiles a program with `#pragma omp parallel` and
    includes <omp.h>. Requires both the compiler flag and the
    OpenMP runtime header.

    Compiler flag used: -fopenmp

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain to check.

    Returns
    -------
    bool
        True if the compiler accepts -fopenmp and finds <omp.h>.
    """
    source = """
    #include <omp.h>
    int main() {
        #pragma omp parallel
        { int x = 0; }
        return 0;
    }
    """
    compiler = _get_compiler_path(toolchain)
    if compiler is None:
        return False
    return _compile_test(compiler, source, ["-fopenmp"])


@_cached("lto")
def supports_lto(toolchain: Union[Toolchain, Path]) -> bool:
    """
    Check if the toolchain supports Link-Time Optimization.

    Test: Compiles a simple function with -flto. LTO requires
    the compiler to emit intermediate representation instead
    of native object code.

    Compiler flag used: -flto

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain to check.

    Returns
    -------
    bool
        True if the compiler accepts -flto.
    """
    source = "int f() { return 42; }"
    compiler = _get_compiler_path(toolchain)
    if compiler is None:
        return False
    return _compile_test(compiler, source, ["-flto"])


@_cached("asan")
def supports_asan(toolchain: Union[Toolchain, Path]) -> bool:
    """
    Check if the toolchain supports AddressSanitizer.

    Test: Compiles a trivial program with -fsanitize=address.
    AddressSanitizer detects memory errors at runtime. This
    test verifies the compiler accepts the flag and can link
    the sanitizer runtime (if linking were performed; this
    test is compile-only).

    Compiler flag used: -fsanitize=address

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain to check.

    Returns
    -------
    bool
        True if the compiler accepts -fsanitize=address.
    """
    source = "int main() { return 0; }"
    compiler = _get_compiler_path(toolchain)
    if compiler is None:
        return False
    return _compile_test(compiler, source, ["-fsanitize=address"])


@_cached("ubsan")
def supports_ubsan(toolchain: Union[Toolchain, Path]) -> bool:
    """
    Check if the toolchain supports UndefinedBehaviorSanitizer.

    Test: Compiles a trivial program with -fsanitize=undefined.
    UBSan detects undefined behavior at runtime.

    Compiler flag used: -fsanitize=undefined

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain to check.

    Returns
    -------
    bool
        True if the compiler accepts -fsanitize=undefined.
    """
    source = "int main() { return 0; }"
    compiler = _get_compiler_path(toolchain)
    if compiler is None:
        return False
    return _compile_test(compiler, source, ["-fsanitize=undefined"])


@_cached("thread_san")
def supports_thread_sanitizer(toolchain: Union[Toolchain, Path]) -> bool:
    """
    Check if the toolchain supports ThreadSanitizer.

    Test: Compiles a trivial program with -fsanitize=thread.
    ThreadSanitizer detects data races at runtime. This is
    typically only available on Linux and macOS.

    Compiler flag used: -fsanitize=thread

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain to check.

    Returns
    -------
    bool
        True if the compiler accepts -fsanitize=thread.
    """
    source = "int main() { return 0; }"
    compiler = _get_compiler_path(toolchain)
    if compiler is None:
        return False
    return _compile_test(compiler, source, ["-fsanitize=thread"])


@_cached("stack_protector")
def supports_stack_protector(toolchain: Union[Toolchain, Path]) -> bool:
    """
    Check if the toolchain supports stack protection.

    Test: Compiles a function with a local buffer using
    -fstack-protector-strong. This flag enables stack canaries
    to detect buffer overflows at runtime.

    Compiler flag used: -fstack-protector-strong

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain to check.

    Returns
    -------
    bool
        True if the compiler accepts -fstack-protector-strong.
    """
    source = "int main() { char buf[10]; return buf[0]; }"
    compiler = _get_compiler_path(toolchain)
    if compiler is None:
        return False
    return _compile_test(compiler, source, ["-fstack-protector-strong"])


@_cached("pic")
def supports_pic(toolchain: Union[Toolchain, Path]) -> bool:
    """
    Check if the toolchain supports Position-Independent Code.

    Test: Compiles a global variable definition with -fPIC.
    PIC is required for shared libraries on most platforms.

    Compiler flag used: -fPIC

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain to check.

    Returns
    -------
    bool
        True if the compiler accepts -fPIC.
    """
    source = "int x = 0;"
    compiler = _get_compiler_path(toolchain)
    if compiler is None:
        return False
    return _compile_test(compiler, source, ["-fPIC"])


@_cached("rtti")
def supports_rtti(toolchain: Union[Toolchain, Path]) -> bool:
    """
    Check if the toolchain supports RTTI (Run-Time Type Information).

    Test: Compiles a program using typeid() and the <typeinfo> header.
    RTTI is required for dynamic_cast and typeid.

    Compiler flag used: -frtti

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain to check.

    Returns
    -------
    bool
        True if the compiler accepts -frtti and <typeinfo> is available.
    """
    source = """
    #include <typeinfo>
    int main() {
        auto& ti = typeid(int);
        return 0;
    }
    """
    compiler = _get_compiler_path(toolchain)
    if compiler is None:
        return False
    return _compile_test(compiler, source, ["-frtti"])


@_cached("exceptions")
def supports_exceptions(toolchain: Union[Toolchain, Path]) -> bool:
    """
    Check if the toolchain supports C++ exception handling.

    Test: Compiles a program with a try/catch block that throws
    and catches an integer. Requires both compiler support and
    the exception handling runtime.

    Compiler flag used: -fexceptions

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain to check.

    Returns
    -------
    bool
        True if the compiler accepts -fexceptions and try/catch
        compiles successfully.
    """
    source = """
    int main() {
        try { throw 42; }
        catch (int) { return 0; }
    }
    """
    compiler = _get_compiler_path(toolchain)
    if compiler is None:
        return False
    return _compile_test(compiler, source, ["-fexceptions"])


# ============================================================================
# Batch detection — run all known checks at once
# ============================================================================

def detect_all(toolchain: Union[Toolchain, Path]) -> Dict[str, bool]:
    """
    Run all capability detection functions and return a dictionary.

    This is the recommended entry point for comprehensive toolchain
    profiling. It calls each detection function in sequence, caches
    individual results, and aggregates them into a single dict.

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain to profile.

    Returns
    -------
    Dict[str, bool]
        Mapping of capability name to boolean. Keys are:
        - c++20, c++17, c++14, c++11: C++ standard support
        - c11: C11 standard support
        - openmp: OpenMP parallelization
        - lto: Link-Time Optimization
        - asan: AddressSanitizer
        - ubsan: UndefinedBehaviorSanitizer
        - thread_sanitizer: ThreadSanitizer
        - stack_protector: Stack protector
        - pic: Position-Independent Code
        - rtti: Run-Time Type Information
        - exceptions: C++ exception handling

    Example
    -------
    >>> caps = detect_all(my_gcc)
    >>> for feature, supported in caps.items():
    ...     print(f"{feature}: {'yes' if supported else 'no'}")
    c++20: yes
    c++17: yes
    openmp: yes
    ...
    """
    checks: Dict[str, Callable] = {
        "c++20": supports_cpp20,
        "c++17": supports_cpp17,
        "c++14": supports_cpp14,
        "c++11": supports_cpp11,
        "c11": supports_c11,
        "openmp": supports_openmp,
        "lto": supports_lto,
        "asan": supports_asan,
        "ubsan": supports_ubsan,
        "thread_sanitizer": supports_thread_sanitizer,
        "stack_protector": supports_stack_protector,
        "pic": supports_pic,
        "rtti": supports_rtti,
        "exceptions": supports_exceptions,
    }

    results: Dict[str, bool] = {}
    for name, check_fn in checks.items():
        try:
            results[name] = check_fn(toolchain)
        except Exception as exc:
            logger.debug("Capability check '%s' raised: %s", name, exc)
            results[name] = False

    return results


# ============================================================================
# Helper
# ============================================================================

def _get_compiler_path(toolchain: Union[Toolchain, Path]) -> Optional[Path]:
    """
    Extract the C compiler path from a Toolchain or Path.

    If a Toolchain is provided, returns toolchain.c_compiler.
    If a Path is provided, returns it directly.
    Returns None if the Toolchain has no C compiler set.

    Parameters
    ----------
    toolchain : Toolchain or Path
        The toolchain or direct path.

    Returns
    -------
    Optional[Path]
        Path to the compiler, or None.
    """
    if isinstance(toolchain, Path):
        return toolchain
    if isinstance(toolchain, Toolchain):
        return toolchain.c_compiler
    return None
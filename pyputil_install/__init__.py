#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyPUtil Install - Python Package Utilities Installer

A collection of tools for managing Python installations, packages, compiler
toolchains, and development headers. Each subproject operates independently
and can be used as a standalone library or via CLI.

Author: Moamen Walid
License: MIT
Repository: https://github.com/moamen-walid-pyputil/pyputil-install
Issues: https://github.com/moamen-walid-pyputil/pyputil-install/issues
Email: pyputilframework@gmail.com
Version: 0.1.0

Project Structure
-----------------
pyputil_install/
├── auto_installer/      # Automatic package installation on import failure
├── stdlib_installer/    # Download stdlib modules from CPython GitHub
├── python_installer/    # Standalone Python version management
├── pyheaders_installer/ # Python C header installation
├── package_installer/   # Advanced package manager (pip wrapper)
├── pip_installer/       # Pip diagnosis, repair, and reinstallation
└── compiler_installer/  # C/C++ toolchain management
    ├── urls/            # URL resolution for compiler releases
    ├── installer/       # Installation, manifests, symlinks, activation
    ├── toolchains/      # Compiler implementations (GCC, Clang, Zig, etc.)
    └── toolforge/       # System compiler discovery and scoring

Subproject Descriptions
-----------------------
auto_installer
    Replaces builtins.__import__ to intercept ImportError and automatically
    install missing packages via pip. Supports synchronous (immediate) and
    asynchronous (deferred/concurrent) installation modes.

stdlib_installer
    Downloads pure-Python standard library modules from the CPython GitHub
    repository. Useful when system Python installations are minimal or missing
    certain stdlib components. Installs modules locally to a configurable
    directory and can add that directory to sys.path.

python_installer
    Downloads and installs standalone Python builds from the
    indygreg/python-build-standalone project. Manages multiple isolated Python
    versions in separate directories. Supports process switching (os.execve)
    and shell default configuration (PATH modification).

pyheaders_installer
    Downloads CPython source archives, extracts the Include directory, and
    installs C header files (Python.h, pyconfig.h) to system or user include
    directories. Supports atomic installation, backup/restore, and post-
    installation verification.

package_installer
    A comprehensive package manager that wraps pip with additional features:
    file-based caching with TTL and compression, retry logic with exponential
    backoff, cryptographic hash verification (SHA256, SHA512, BLAKE2b),
    trust verification (host allowlist, certificate pinning), automatic
    rollback on installation failure, and virtual environment management.

pip_installer
    A pip-specific rescue tool. Diagnoses pip's state (HEALTHY, MISSING,
    BROKEN, OUTDATED, BLOCKED) and reinstalls using the optimal strategy:
    ensurepip, get-pip.py, wheel installation, or manual zipimport extraction.
    Includes version compatibility checking against Python versions.

compiler_installer
    A complete toolchain management system for C/C++ compilers and related
    tools. Supports GCC (xPack), Clang/LLVM, MinGW-w64, ARM GNU Toolchain,
    RISC-V, Emscripten SDK, Zig, and Android NDK. Provides download URL
    resolution (with validation), extraction, manifest-based installation,
    symlink management (versioned and short), PATH activation (process stack),
    environment management (named compiler sets), and host system discovery
    with scoring.

Dependencies
------------
Required:
    - Python 3.9 or higher
    - No external dependencies for core functionality (uses only stdlib)

Optional:
    - aiohttp >= 3.8.0   : Async HTTP for compiler_installer downloads
    - colorama >= 0.4.6  : ANSI color output on Windows (CLI)
    - packaging >= 23.0  : PEP 440 version parsing (fallback included)

Environment Variables
---------------------
TOOLFORGE_HOME              : Compiler installation root (default: ~/.local/share/toolforge)
TOOLFORGE_BIN_DIR           : Centralized symlink directory for compilers
TOOLFORGE_TEMP_DIR          : Temporary staging directory for downloads
TOOLFORGE_GITHUB_TOKEN      : GitHub API token for higher rate limits
TOOLFORGE_SKIP_CHECKSUM     : Set "1" to skip checksum verification
TOOLFORGE_SKIP_VALIDATION   : Set "1" to skip pre-download URL validation
TOOLFORGE_NO_SYMLINKS       : Set "1" to skip symlink creation
TOOLFORGE_LINK_STYLE        : "both", "short", or "versioned"
TOOLFORGE_LINK_FORCE        : Set "1" to overwrite existing symlinks
TOOLFORGE_ACTIVE            : Colon-separated active toolchains (internal)
COMPILER_EXECUTION_TIMEOUT  : Subprocess timeout for compiler discovery (default: 5)
COMPILER_SEARCH_CANDIDATES  : Colon-separated executable names to search
COMPILER_SEARCH_DIRS        : Colon-separated extra directories to scan
TOOLFORGE_CACHE_DIR         : Cache directory for discovery results
TOOLFORGE_CACHE_MAX_AGE     : Cache max age in seconds (default: 3600)

CLI Entry Points
----------------
package-installer   : Main package manager (install, upgrade, search, freeze, venv)
pip-installer       : Pip diagnosis and repair
stdlib-installer    : Stdlib module management
python-installer    : Python version management
headers-installer   : Python header installation
pt / pyputil        : Aliases for package-installer

Exit Codes (package_installer CLI)
----------------------------------
0   : Success
1   : General error
2   : Argument error
3   : Download error
4   : Extraction error
5   : Installation error
6   : Verification error

Exit Codes (pip_installer CLI)
------------------------------
0   : Success
1   : General error
2   : Invalid arguments
3   : Network error
4   : Permission error
5   : Incompatible version
6   : Verification failed

Examples
--------
1. Install a Python package automatically on import:

    from pyputil_install.auto_installer import auto_install_sync
    auto_install_sync()
    import requests  # installs if missing

2. Install a missing stdlib module:

    from pyputil_install.stdlib_installer import install_stdlib, add_to_sys_path
    add_to_sys_path()
    install_stdlib("tomllib", version="3.13")
    import tomllib  # works even on older Python

3. Install a standalone Python version:

    from pyputil_install.python_installer.installer import PythonInstaller
    installer = PythonInstaller()
    result = installer.install("3.11.5")
    print(result.path)

4. Install Python C headers:

    from pyputil_install.pyheaders_installer import install_python_headers
    path = install_python_headers(version="3.11.0")
    print(path)  # /usr/include/python3.11

5. Install a package with hash verification:

    from pyputil_install.package_installer import PackageInstaller, InstallConfig
    config = InstallConfig(require_hashes=True, use_cache=True)
    installer = PackageInstaller("requests", config=config)
    result = installer.install()
    print(result.version_installed)

6. Repair a broken pip:

    from pyputil_install.pip_installer import repair
    result = repair(target_version="21.3.1", user_site=True)
    print(result.success, result.version_installed)

7. Install a GCC toolchain:

    import asyncio
    from pyputil_install.compiler_installer import install_toolchain
    result = await install_toolchain("gcc", "14.2.0-2", platform="linux", arch="x64")
    print(result.path)

8. Discover compilers on the system:

    from pyputil_install.compiler_installer.toolforge.discovery import discover_compilers
    manager = discover_compilers()
    best = manager.best()
    print(best.path, best.version, best.kind)

9. Activate a compiler in the current process:

    from pyputil_install.compiler_installer.activation import activate
    activate("gcc", "14.2.0-2")  # adds to PATH

10. Create and activate a named environment:

    from pyputil_install.compiler_installer.environments import EnvironmentStore
    store = EnvironmentStore()
    store.create("cpp20", {"CC": "gcc@14.2.0-2", "CXX": "g++@14.2.0-2"})
    store.activate("cpp20")

Module Exports
--------------
This __init__.py re-exports public APIs from all subprojects for convenience.
For detailed subproject APIs, import from the specific submodule.

Subproject Entry Points:
- auto_installer: auto_install_sync, auto_install_async, AsyncAutoInstaller, SyncAutoInstaller
- stdlib_installer: install_stdlib, remove_stdlib, list_installed_stdlib, add_to_sys_path
- python_installer: PythonInstaller, InstallResult as PythonInstallResult
- pyheaders_installer: install_python_headers, get_python_version, clean_cache
- package_installer: PackageInstaller, InstallConfig, InstallResult
- pip_installer: PipRescuer, repair, diagnose, check_compatibility
- compiler_installer: install_toolchain, activate, deactivate, discover_compilers.

Notes
-----

## Dependency Notice

This package is used in the second package of the PyPUtil Framework:

pyputil_cutil

So you may still need this package if you plan to install or use "pyputil_cutil".
Note: pyputil_cutil has not been released as a package yet, it is still under development. 
---

## Reporting Issues

If you encounter any problems, please report them in the GitHub issues section and include:

- The error/problem you faced
- How the issue occurred

### GitHub Issues:
https://github.com/moamen-walid-pyputil/pyputil-install/issues

---

## Assistance & Project Ideas

If you would like:

- Special assistance
- Help with a project
- To suggest a development idea

Feel free to contact me via email:

pyputilframework@gmail.com

---

## Documentation

There is currently (version '0.1.0') no official documentation website for this package.

However, one may be added in the future when time allows.

---

## Important Message

I LOVE YOU, MY USER!
"""

__version__ = "0.1.0"
__author__ = "Moamen Walid"
__license__ = "MIT"
__email__ = "pyputilframework@gmail.com"
__repository__ = "https://github.com/moamen-walid-pyputil/pyputil-install"
__issues__ = "https://github.com/moamen-walid-pyputil/pyputil-install/issues"
__all__ = [
    "python_installer",
    "pip_installer",
    "auto_installer",
    "pyheaders_installer",
    "stdlib_installer",
    "package_installer",
    "compiler_installer",
]


def get_doc() -> str:
    """
    Display the package documentation to help the user.
    """
    return __doc__



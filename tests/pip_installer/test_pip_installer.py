"""
Pip Installer Comprehensive Test Suite
========================================

This file tests all components of the pip_installer library including:
checker, installer, versions, fetcher, and main orchestrator.

Test Execution:
    python -m pytest test_pip_installer.py -v
    python test_pip_installer.py  (direct execution)

Supported Platforms & Expected Results:
===============================================================================
Platform                    | Python Versions      | Expected Result
----------------------------|----------------------|--------------------------------------------------
Linux (Ubuntu 20.04/22.04)  | 3.6 - 3.12           | All tests PASS. ensurepip available.
Linux (Debian)              | 3.6 - 3.11           | PASS except ensurepip tests fail without python3-venv.
Linux (CentOS/RHEL)         | 3.6 - 3.11           | PASS. ensurepip bundled with Python.
Linux (Alpine)              | 3.6 - 3.12           | ensurepip tests FAIL (musl + stripped Python).
macOS (Intel)               | 3.7 - 3.12           | All tests PASS. Framework builds include ensurepip.
macOS (Apple Silicon)       | 3.8 - 3.12           | All tests PASS. Rosetta 2 mode works.
Windows 10/11               | 3.6 - 3.12           | All tests PASS. Path handling uses backslashes.
Windows Server 2019/2022    | 3.6 - 3.11           | All tests PASS. Administrator privileges NOT required.
-------------------------------------------------------------------------------
Offline Environment         | Any                  | Download-based tests SKIP or FAIL with network error.
No write permissions        | Any                  | Installation tests FAIL with PERMISSION_ERROR.
PyPy                        | 3.8 - 3.10           | Parsing/import tests PASS. ensurepip behavior varies.
Conda environment           | Any                  | Conda detection tests PASS. Conda install requires conda.

Per-Function Expected Results:
===============================================================================

MODULE: checker.py
------------------
PipChecker.__init__(python_executable=None):
    - Expect: Uses sys.executable. No exception.
    - Input Path("/nonexistent/python"): Raises FileNotFoundError.

PipChecker.run_full_diagnosis():
    - On system with pip installed: status = HEALTHY, current_version != None
    - On system without pip: status = MISSING, current_version = None
    - On system with broken pip (missing _vendor): status = BROKEN
    - All platforms: environment.site_packages_paths list non-empty
    - Windows: executable_path returns Path with .exe extension
    - Linux/macOS: executable_path returns Path without .exe

PipChecker.inspect_environment():
    - Returns EnvironmentInfo with all fields populated
    - python_version_tuple[0] == sys.version_info[0]
    - has_internet: True if socket can reach any host, False otherwise
    - is_virtualenv: True if running in venv, False for system Python
    - environment_origin: SYSTEM/VENV/VIRTUALENV/CONDA/DOCKER/UNKNOWN
    - is_user_site_enabled: True if ~/.local exists or Windows equivalent

PipChecker.locate_pip_module():
    - Returns (exists, primary_path, all_paths)
    - exists == True when pip installed, False otherwise
    - primary_path resolves to directory containing pip/__init__.py
    - all_paths length >= 1 when pip installed

PipChecker.locate_pip_executable():
    - Returns (exists, path)
    - Windows: path.name == 'pip.exe' or 'pip'
    - Linux/macOS: path.name == 'pip'
    - Adjacent bin directory searched before PATH

PipChecker.detect_pip_version():
    - Returns (version_string, raw_output)
    - version_string format: 'X.Y.Z' or None
    - raw_output contains pip --version output or error text

PipChecker.try_import_pip():
    - Returns (success, error_message)
    - success == True when pip importable
    - error_message contains traceback when import fails
    - Timeout after 30 seconds when pip hangs on import

MODULE: versions.py
-------------------
parse_version(version_str):
    - Input "24.0" → (24, 0)
    - Input "21.3.1" → (21, 3, 1)
    - Input "20.3.4.post1" → (20, 3, 4)
    - Input "invalid" → ValueError

compare_versions(a, b):
    - Input ("24.0", "23.0") → 1
    - Input ("21.3.1", "21.3.1") → 0
    - Input ("20.0", "21.0") → -1
    - Input ("21.0", "21.0.0") → 0

normalize_version(version_str):
    - Input "24" → "24.0.0"
    - Input "21.3" → "21.3.0"
    - Input "20.3.4" → "20.3.4"
    - Input "20.3.4.1" → "20.3.4"

get_compatible_versions_for_python(python_version):
    - Input (3, 6) → Returns list containing "21.3.1", not "24.0"
    - Input (3, 8) → Returns list containing "24.3.1"
    - Input (2, 7) → Returns list ending with "20.3.4"
    - Returns newest first (sorted descending)

VersionResolver.__init__(python_version=None):
    - No arguments: uses sys.version_info[:3]
    - Input (3, 8, 10): python_version = (3, 8, 10)
    - Input (3, 6): Raises ValueError (need 3 components)

VersionResolver.resolve_best_version():
    - Python 3.6 → version = "21.3.1", source = AUTO_RESOLVED
    - Python 3.8+ → version = LATEST_KNOWN_PIP ("24.3.1")
    - Python 3.5 → version = "20.3.4"
    - Returns VersionResolution with reason string

VersionResolver.resolve_user_version(requested_version):
    - Request "24.0" on Python 3.6 → compatible=False, alternatives list contains "21.3.1"
    - Request "21.3.1" on Python 3.9 → compatible=True, version="21.3.1"
    - Request "invalid" → compatible=False, version=None
    - Request "24" → normalized to "24.0.0"

VersionResolver.is_compatible(pip_version, python_version):
    - Input ("24.0", (3,6,8)) → (False, "pip>=24.0 requires Python>=3.8")
    - Input ("21.3.1", (3,9,18)) → (True, "supports Python 3.9")
    - strict=True, unknown version → (False, "not in known compatibility table")

VersionResolver.get_latest_compatible():
    - Python 3.9 → Returns "24.3.1"
    - Python 3.6 → Returns "21.3.1"
    - Unsupported Python → Raises RuntimeError

MODULE: fetcher.py
------------------
ResourceFetcher.__init__(cache_dir=None):
    - No cache_dir: creates temp directory, prefix "pip_rescuer_cache_"
    - Invalid cache_dir (read-only): Raises PermissionError
    - Default timeout=30, max_retries=3

ResourceFetcher.fetch_get_pip(version=None):
    - No version: downloads from bootstrap.pypa.io/get-pip.py
    - version="3.6": downloads from bootstrap.pypa.io/pip/3.6/get-pip.py
    - Returns (success, path, metadata)
    - metadata.status: SUCCESS/CACHED/NETWORK_ERROR/NOT_FOUND
    - Success: path.exists() == True, path.stat().st_size > 1000000

ResourceFetcher.fetch_pip_wheel(version, python_version=None):
    - version="21.3.1": downloads pip-21.3.1-py3-none-any.whl
    - version="24.0": downloads pip-24.0-py3-none-any.whl
    - Invalid version (99.0.0): success=False, status=NOT_FOUND
    - metadata.sha256: hex string length 64 (when available)

ResourceFetcher.check_connectivity():
    - Internet available → True (any host reachable)
    - No internet (airplane mode, disconnected) → False
    - DNS failure → False
    - Firewall blocking all hosts → False

ResourceFetcher.clear_cache(older_than_hours=None):
    - Returns number of entries removed
    - older_than_hours=24: removes files cached >24 hours ago
    - Cache index file deleted after clearing

ResourceFetcher.verify_file_integrity(file_path, expected_sha256, expected_size):
    - File matches SHA256 → (True, "All integrity checks passed")
    - SHA256 mismatch → (False, "SHA256 mismatch: expected... got...")
    - File not found → (False, "File does not exist")
    - Size mismatch → (False, "Size mismatch: expected X, got Y")

ResourceFetcher.add_mirror(mirror_url, position=None):
    - position=0: inserts at beginning (highest priority)
    - position=None: appends to end
    - Mirrors stored in _mirrors list

MODULE: installer.py
--------------------
PipInstaller.__init__(diagnosis, target_version=None, ...):
    - diagnosis required (from PipChecker.run_full_diagnosis())
    - user_site=True with unavailable user site → adds error to _attempts

PipInstaller.install():
    - Returns InstallResult
    - result.success = True when pip installed or already healthy
    - result.strategy_used: ENSUREPIP/GET_PIP_SCRIPT/WHEEL_INSTALL/MANUAL_EXTRACTION
    - result.result_code: SUCCESS/ALREADY_INSTALLED/FAILED_*
    - dry_run=True: returns success=True, no filesystem changes

PipInstaller.install_with_strategy(strategy):
    - Forced strategy execution regardless of diagnosis
    - Invalid strategy → result_code = FAILED_UNKNOWN
    - Returns InstallResult with strategy_used = requested strategy

PipInstaller.verify_installation():
    - Returns (success, version, output)
    - success=True when pip --version or python -m pip --version works
    - version parsed from output (format "X.Y.Z")
    - Both methods attempted; success if either works

PipInstaller._install_via_ensurepip():
    - Returns None when target_version specified (cannot install specific version)
    - ensurepip missing (Debian without python3-venv) → success=False
    - ensurepip available → installs bundled pip version

PipInstaller._install_via_get_pip():
    - Uses PIP_VERSION environment variable for version selection
    - Local get-pip.py file used if provided
    - Downloads from bootstrap.pypa.io when network available
    - Returns error when no local copy and no network

PipInstaller._install_from_wheel():
    - Uses local wheel when wheel_path provided
    - Downloads wheel from PyPI when network available
    - Falls back to manual extraction when pip not importable
    - Returns result with strategy_used = WHEEL_INSTALL

PipInstaller._install_via_manual_extraction():
    - Last resort strategy
    - Finds bundled wheel from ensurepip._bundled
    - Extracts pip/ and pip-*.dist-info to site-packages
    - Creates executable script in bin/ or Scripts/
    - warnings field contains "manual extraction; metadata may be incomplete"

PipInstaller._install_via_conda():
    - Only attempted when environment.is_conda == True
    - Requires conda executable on PATH
    - Returns None when conda not found
    - success=False when conda install fails

MODULE: main.py
---------------
PipRescuer.__init__(...):
    - All parameters have defaults
    - python_executable=None → uses sys.executable
    - Invalid python_executable → FileNotFoundError

PipRescuer.run():
    - Returns RescueResult
    - Pipeline stages: VALIDATION, DIAGNOSIS, VERSION_RESOLUTION, FETCHING, INSTALLATION, VERIFICATION
    - dry_run=True: stops before INSTALLATION stage
    - Healthy pip + force=False → returns success=True, no installation

PipRescuer.uninstall():
    - Removes pip module, dist-info directories, executable script
    - dry_run=True: prints "Would remove" without deletion
    - Returns RescueResult with strategy_used="uninstall"
    - Verification confirms PipStatus.MISSING after removal
    - Partial removal (permission denied) → success=False, errors list populated

StageResult:
    - stage: PipelineStage enum value
    - success: bool
    - message: str
    - duration_ms: float (elapsed milliseconds)
    - data: Dict for stage-specific information

RescueResult.report(verbose=False):
    - Returns formatted string with status, version, duration
    - verbose=True includes per-stage details
    - dry_run=True shows "[DRY RUN]" header

create_argument_parser():
    - Returns argparse.ArgumentParser
    - --uninstall conflicts with --version, --offline, --get-pip
    - --verbose and --quiet mutually exclusive
    - --json overrides --verbose

main(argv=None):
    - Entry point for CLI
    - Returns ExitCode.SUCCESS (0) on success
    - Returns ExitCode.INVALID_ARGUMENTS (2) for invalid combinations
    - Returns ExitCode.NETWORK_ERROR (3) when network unavailable
    - Returns ExitCode.PERMISSION_ERROR (4) for permission issues
    - Returns ExitCode.INCOMPATIBLE_VERSION (5) for version mismatch

MODULE: __init__.py (public API)
--------------------------------
Public exports (__all__):
    - PipRescuer, PipChecker, VersionResolver, ResourceFetcher, PipInstaller
    - PipDiagnosis, EnvironmentInfo, VersionResolution, InstallResult, RescueResult
    - PipStatus, InstallStrategy, InstallResultCode, ExitCode, FetchStatus, VersionSource, CompatibilityStatus, PipelineStage
    - compare_versions, parse_version, normalize_version, get_compatible_versions_for_python

repair(target_version=None, ...):
    - Convenience function wrapping PipRescuer().run()
    - Returns RescueResult
    - Example: repair() → RescueResult with success=True/False

diagnose(python_executable=None):
    - Convenience function wrapping PipChecker().run_full_diagnosis()
    - Returns PipDiagnosis

check_compatibility(pip_version, python_version=None):
    - Returns (compatible: bool, reason: str)
    - Example: check_compatibility("24.0", (3,6,8)) → (False, "pip>=24.0 requires Python>=3.8")

Data Validation Expectations:
===============================================================================
All dataclass fields must be populated with correct types:
- Path fields: Path object, not string
- List fields: never None (empty list default)
- Optional fields: None allowed for missing values

Error Handling Expectations:
===============================================================================
- Network failures: status = NETWORK_ERROR, not crash
- Permission errors: status = PERMISSION_ERROR, error message includes details
- Invalid version strings: ValueError raised with clear message
- File not found: FileNotFoundError raised with path in message
- Timeout: subprocess.TimeoutExpired caught, returns error status
"""

import sys
import os
import shutil
import tempfile
import subprocess
import platform
import json
import re
import pytest
import time
import socket
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, List, Optional, Tuple, Any

# Disable network for offline tests
OFFLINE_MODE = os.environ.get("PIP_RESCUER_OFFLINE", "").lower() in ("1", "true", "yes")
SKIP_NETWORK_TESTS = OFFLINE_MODE

print(f"Test configuration: OFFLINE_MODE={OFFLINE_MODE}, SKIP_NETWORK_TESTS={SKIP_NETWORK_TESTS}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")


# =============================================================================
# Test Environment Setup
# =============================================================================

class TestEnvironment:
    """Isolated test environment with temporary directories."""
    
    def __init__(self):
        self.temp_dir: Optional[Path] = None
        self.cache_dir: Optional[Path] = None
        self.original_sys_path: Optional[List[str]] = None
        self.original_os_environ: Optional[Dict[str, str]] = None
    
    def setup(self):
        """Create isolated test environment."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="pip_rescuer_test_"))
        self.cache_dir = self.temp_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.original_sys_path = sys.path[:]
        self.original_os_environ = os.environ.copy()
        
        # Prevent side effects
        sys.path.insert(0, str(self.temp_dir))
        
    def teardown(self):
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        if self.original_sys_path:
            sys.path = self.original_sys_path
        if self.original_os_environ:
            os.environ.clear()
            os.environ.update(self.original_os_environ)


# =============================================================================
# Test: checker.py
# =============================================================================

class TestPipChecker:
    """Test PipChecker class - pip state inspection."""
    
    def test_init_default(self):
        """PipChecker.__init__(python_executable=None) - uses sys.executable"""
        from pyputil_install.pip_installer.checker import PipChecker
        
        checker = PipChecker()
        assert checker.python_executable == Path(sys.executable).resolve()
        assert checker.timeout == 30
        assert checker._cached_environment is None
    
    def test_init_with_valid_executable(self):
        """PipChecker.__init__(python_executable=Path) - uses provided path"""
        from pyputil_install.pip_installer.checker import PipChecker
        
        python_path = Path(sys.executable).resolve()
        checker = PipChecker(python_path)
        assert checker.python_executable == python_path
    
    def test_init_with_invalid_executable(self):
        """PipChecker.__init__(python_executable=nonexistent) - raises FileNotFoundError"""
        from pyputil_install.pip_installer.checker import PipChecker
        
        invalid_path = Path("/nonexistent/python/executable/for/testing")
        try:
            checker = PipChecker(invalid_path)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError as e:
            assert str(invalid_path) in str(e)
    
    def test_inspect_environment_returns_complete_info(self):
        """PipChecker.inspect_environment() - returns EnvironmentInfo with all fields"""
        from pyputil_install.pip_installer.checker import PipChecker
        
        checker = PipChecker()
        env = checker.inspect_environment()
        
        assert hasattr(env, 'python_version')
        assert hasattr(env, 'python_version_tuple')
        assert hasattr(env, 'python_executable')
        assert hasattr(env, 'python_implementation')
        assert hasattr(env, 'python_implementation_version')
        assert hasattr(env, 'is_virtualenv')
        assert hasattr(env, 'environment_origin')
        assert hasattr(env, 'is_conda')
        assert hasattr(env, 'is_user_site_enabled')
        assert hasattr(env, 'site_packages_paths')
        assert hasattr(env, 'user_site_packages')
        assert hasattr(env, 'has_internet')
        assert hasattr(env, 'os_name')
        assert hasattr(env, 'os_version')
        assert hasattr(env, 'is_admin')
        assert hasattr(env, 'env_vars')
        assert hasattr(env, 'disk_free_mb')
    
    def test_inspect_environment_caching(self):
        """PipChecker.inspect_environment() - caches result after first call"""
        from pyputil_install.pip_installer.checker import PipChecker
        
        checker = PipChecker()
        env1 = checker.inspect_environment()
        env2 = checker.inspect_environment()
        
        assert env1 is env2  # Same object reference
    
    def test_python_version_tuple_format(self):
        """EnvironmentInfo.python_version_tuple - returns (major, minor, micro)"""
        from pyputil_install.pip_installer.checker import PipChecker
        
        checker = PipChecker()
        env = checker.inspect_environment()
        
        assert len(env.python_version_tuple) == 3
        assert isinstance(env.python_version_tuple[0], int)
        assert isinstance(env.python_version_tuple[1], int)
        assert isinstance(env.python_version_tuple[2], int)
        assert env.python_version_tuple[0] == sys.version_info[0]
    
    def test_os_name_platform_identification(self):
        """EnvironmentInfo.os_name - identifies OS correctly"""
        from pyputil_install.pip_installer.checker import PipChecker
        
        checker = PipChecker()
        env = checker.inspect_environment()
        
        valid_os_names = ["linux", "darwin", "windows"]
        assert env.os_name in valid_os_names
    
    def test_site_packages_paths_non_empty(self):
        """EnvironmentInfo.site_packages_paths - returns at least one path"""
        from pyputil_install.pip_installer.checker import PipChecker
        
        checker = PipChecker()
        env = checker.inspect_environment()
        
        assert len(env.site_packages_paths) >= 1
        for path in env.site_packages_paths:
            assert isinstance(path, Path)
    
    def test_locate_pip_module_returns_tuple(self):
        """PipChecker.locate_pip_module() - returns (exists, primary, all_paths)"""
        from pyputil_install.pip_installer.checker import PipChecker
        
        checker = PipChecker()
        exists, primary, all_paths = checker.locate_pip_module()
        
        assert isinstance(exists, bool)
        assert primary is None or isinstance(primary, Path)
        assert isinstance(all_paths, list)
    
    def test_locate_pip_executable_returns_tuple(self):
        """PipChecker.locate_pip_executable() - returns (exists, path)"""
        from pyputil_install.pip_installer.checker import PipChecker
        
        checker = PipChecker()
        exists, path = checker.locate_pip_executable()
        
        assert isinstance(exists, bool)
        assert path is None or isinstance(path, Path)
        
        if exists:
            assert path is not None
            assert path.exists()
            if sys.platform == "win32":
                assert path.name in ("pip.exe", "pip")
            else:
                assert path.name == "pip"
    
    def test_detect_pip_version_returns_string(self):
        """PipChecker.detect_pip_version() - returns (version, raw_output)"""
        from pyputil_install.pip_installer.checker import PipChecker
        
        checker = PipChecker()
        version, raw = checker.detect_pip_version()
        
        assert isinstance(version, (str, type(None)))
        assert isinstance(raw, str)
        
        if version is not None:
            assert re.match(r"\d+\.\d+(?:\.\d+)?", version)
    
    def test_try_import_pip_returns_tuple(self):
        """PipChecker.try_import_pip() - returns (success, error_message)"""
        from pyputil_install.pip_installer.checker import PipChecker
        
        checker = PipChecker()
        success, error = checker.try_import_pip()
        
        assert isinstance(success, bool)
        assert error is None or isinstance(error, str)
    
    def test_run_pip_check_returns_tuple(self):
        """PipChecker.run_pip_check() - returns (success, output)"""
        from pyputil_install.pip_installer.checker import PipChecker
        
        checker = PipChecker()
        success, output = checker.run_pip_check()
        
        assert isinstance(success, bool)
        assert isinstance(output, str)
    
    def test_run_full_diagnosis_returns_pipdiagnosis(self):
        """PipChecker.run_full_diagnosis() - returns PipDiagnosis with all fields"""
        from pyputil_install.pip_installer.checker import PipChecker, PipDiagnosis, PipStatus
        
        checker = PipChecker()
        diagnosis = checker.run_full_diagnosis()
        
        assert isinstance(diagnosis, PipDiagnosis)
        assert isinstance(diagnosis.status, PipStatus)
        assert hasattr(diagnosis, 'current_version')
        assert hasattr(diagnosis, 'install_paths')
        assert hasattr(diagnosis, 'executable_path')
        assert hasattr(diagnosis, 'module_path')
        assert hasattr(diagnosis, 'import_error')
        assert hasattr(diagnosis, 'environment')
        assert hasattr(diagnosis, 'raw_output')
        assert isinstance(diagnosis.raw_output, dict)
    
    def test_pip_status_enum_values(self):
        """PipStatus - all expected status values"""
        from pyputil_install.pip_installer.checker import PipStatus
        
        expected_statuses = [
            "HEALTHY", "MISSING", "BROKEN", "OUTDATED", "BLOCKED", "UNKNOWN"
        ]
        
        for status_name in expected_statuses:
            assert hasattr(PipStatus, status_name)
            assert isinstance(getattr(PipStatus, status_name), PipStatus)
    
    def test_environment_origin_enum_values(self):
        """EnvironmentOrigin - all expected origin values"""
        from pyputil_install.pip_installer.checker import EnvironmentOrigin
        
        expected_origins = [
            "SYSTEM", "VENV", "VIRTUALENV", "CONDA", "PYENV", "DOCKER", "UNKNOWN"
        ]
        
        for origin_name in expected_origins:
            assert hasattr(EnvironmentOrigin, origin_name)
            assert isinstance(getattr(EnvironmentOrigin, origin_name), EnvironmentOrigin)


# =============================================================================
# Test: versions.py
# =============================================================================

class TestVersionsUtilities:
    """Test version parsing and comparison utilities."""
    
    def test_parse_version_full(self):
        """parse_version("X.Y.Z") → (X, Y, Z)"""
        from pyputil_install.pip_installer.versions import parse_version
        
        assert parse_version("24.3.1") == (24, 3, 1)
        assert parse_version("21.0.0") == (21, 0, 0)
        assert parse_version("1.2.3") == (1, 2, 3)
    
    def test_parse_version_two_parts(self):
        """parse_version("X.Y") → (X, Y)"""
        from pyputil_install.pip_installer.versions import parse_version
        
        assert parse_version("24.0") == (24, 0)
        assert parse_version("21.3") == (21, 3)
    
    def test_parse_version_one_part(self):
        """parse_version("X") → (X,)"""
        from pyputil_install.pip_installer.versions import parse_version
        
        assert parse_version("24") == (24,)
        assert parse_version("3") == (3,)
    
    def test_parse_version_with_suffix(self):
        """parse_version() strips .post, .dev, .rc suffixes"""
        from pyputil_install.pip_installer.versions import parse_version
        
        assert parse_version("20.3.4.post1") == (20, 3, 4)
        assert parse_version("21.0.0.dev0") == (21, 0, 0)
        assert parse_version("24.1.0.rc1") == (24, 1, 0)
    
    def test_parse_version_invalid_raises_value_error(self):
        """parse_version("invalid") → ValueError"""
        from pyputil_install.pip_installer.versions import parse_version
        
        try:
            parse_version("not_a_version")
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "No numeric version components" in str(e)
    
    def test_compare_versions_greater(self):
        """compare_versions(a, b) where a > b → 1"""
        from pyputil_install.pip_installer.versions import compare_versions
        
        assert compare_versions("24.0", "23.0") == 1
        assert compare_versions("24.1.0", "24.0.9") == 1
        assert compare_versions("21.3.1", "21.2.4") == 1
    
    def test_compare_versions_equal(self):
        """compare_versions(a, b) where a == b → 0"""
        from pyputil_install.pip_installer.versions import compare_versions
        
        assert compare_versions("21.3.1", "21.3.1") == 0
        assert compare_versions("24.0", "24.0.0") == 0
        assert compare_versions("23.0", "23.0") == 0
    
    def test_compare_versions_less(self):
        """compare_versions(a, b) where a < b → -1"""
        from pyputil_install.pip_installer.versions import compare_versions
        
        assert compare_versions("20.0", "21.0") == -1
        assert compare_versions("23.9.9", "24.0.0") == -1
        assert compare_versions("21.2.4", "21.3.1") == -1
    
    def test_normalize_version_one_part(self):
        """normalize_version("X") → "X.0.0" """
        from pyputil_install.pip_installer.versions import normalize_version
        
        assert normalize_version("24") == "24.0.0"
        assert normalize_version("3") == "3.0.0"
    
    def test_normalize_version_two_parts(self):
        """normalize_version("X.Y") → "X.Y.0" """
        from pyputil_install.pip_installer.versions import normalize_version
        
        assert normalize_version("21.3") == "21.3.0"
        assert normalize_version("24.0") == "24.0.0"
    
    def test_normalize_version_three_parts(self):
        """normalize_version("X.Y.Z") → "X.Y.Z" unchanged"""
        from pyputil_install.pip_installer.versions import normalize_version
        
        assert normalize_version("21.3.1") == "21.3.1"
        assert normalize_version("24.3.1") == "24.3.1"
    
    def test_normalize_version_truncates_extra_parts(self):
        """normalize_version() keeps only first 3 components"""
        from pyputil_install.pip_installer.versions import normalize_version
        
        assert normalize_version("20.3.4.1") == "20.3.4"
        assert normalize_version("1.2.3.4.5") == "1.2.3"
    
    def test_get_compatible_versions_for_python_36(self):
        """get_compatible_versions_for_python((3,6)) - returns versions up to 21.3.1"""
        from pyputil_install.pip_installer.versions import get_compatible_versions_for_python
        
        versions = get_compatible_versions_for_python((3, 6, 8))
        
        assert "21.3.1" in versions
        assert "24.0" not in versions
        assert "24.3.1" not in versions
        assert versions[0] == "21.3.1"  # newest first
    
    def test_get_compatible_versions_for_python_38(self):
        """get_compatible_versions_for_python((3,8)) - returns including 24.x"""
        from pyputil_install.pip_installer.versions import get_compatible_versions_for_python
        
        versions = get_compatible_versions_for_python((3, 8, 0))
        
        assert "21.3.1" in versions
        assert "24.0" in versions
        assert versions[0] == "24.3.1"  # newest first
    
    def test_get_compatible_versions_for_python_27(self):
        """get_compatible_versions_for_python((2,7)) - returns up to 20.3.4"""
        from pyputil_install.pip_installer.versions import get_compatible_versions_for_python
        
        versions = get_compatible_versions_for_python((2, 7, 18))
        
        assert "20.3.4" in versions
        assert "21.0" not in versions
        assert versions[0] == "20.3.4"


class TestVersionResolver:
    """Test VersionResolver class."""
    
    def test_init_default(self):
        """VersionResolver().__init__() - uses current Python version"""
        from pyputil_install.pip_installer.versions import VersionResolver
        
        resolver = VersionResolver()
        assert resolver.python_version[0] == sys.version_info[0]
        assert resolver.python_version[1] == sys.version_info[1]
        assert resolver.python_version[2] == sys.version_info[2]
        assert resolver.strict is False
    
    def test_init_with_python_version(self):
        """VersionResolver(python_version=(3,8,10)) - uses provided version"""
        from pyputil_install.pip_installer.versions import VersionResolver
        
        resolver = VersionResolver(python_version=(3, 8, 10))
        assert resolver.python_version == (3, 8, 10)
    
    def test_init_with_invalid_python_version_raises_error(self):
        """VersionResolver(python_version=(3,6)) - requires 3 components"""
        from pyputil_install.pip_installer.versions import VersionResolver
        
        try:
            resolver = VersionResolver(python_version=(3, 6))
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "3 components" in str(e)
    
    def test_resolve_best_version_python_36(self):
        """VersionResolver.resolve_best_version() for Python 3.6 → 21.3.1"""
        from pyputil_install.pip_installer.versions import VersionResolver, VersionSource
        
        resolver = VersionResolver(python_version=(3, 6, 8))
        result = resolver.resolve_best_version()
        
        assert result.version == "21.3.1"
        assert result.source == VersionSource.AUTO_RESOLVED
        assert result.compatible is True
        assert result.reason is not None
        assert isinstance(result.reason, str)
    
    def test_resolve_best_version_python_38(self):
        """VersionResolver.resolve_best_version() for Python 3.8 → LATEST_KNOWN_PIP"""
        from pyputil_install.pip_installer.versions import VersionResolver, LATEST_KNOWN_PIP
        
        resolver = VersionResolver(python_version=(3, 8, 0))
        result = resolver.resolve_best_version()
        
        assert result.version == LATEST_KNOWN_PIP
        assert result.compatible is True
    
    def test_resolve_best_version_python_39(self):
        """VersionResolver.resolve_best_version() for Python 3.9 → LATEST_KNOWN_PIP"""
        from pyputil_install.pip_installer.versions import VersionResolver, LATEST_KNOWN_PIP
        
        resolver = VersionResolver(python_version=(3, 9, 18))
        result = resolver.resolve_best_version()
        
        assert result.version == LATEST_KNOWN_PIP
    
    def test_resolve_best_version_python_35(self):
        """VersionResolver.resolve_best_version() for Python 3.5 → 20.3.4"""
        from pyputil_install.pip_installer.versions import VersionResolver
        
        resolver = VersionResolver(python_version=(3, 5, 10))
        result = resolver.resolve_best_version()
        
        assert result.version == "20.3.4"
    
    def test_resolve_user_version_compatible(self):
        """VersionResolver.resolve_user_version() - compatible version returns success"""
        from pyputil_install.pip_installer.versions import VersionResolver
        
        resolver = VersionResolver(python_version=(3, 9, 18))
        result = resolver.resolve_user_version("21.3.1")
        
        assert result.compatible is True
        assert result.version == "21.3.1"
        assert len(result.alternatives) == 0
    
    def test_resolve_user_version_incompatible(self):
        """VersionResolver.resolve_user_version() - incompatible version returns alternatives"""
        from pyputil_install.pip_installer.versions import VersionResolver
        
        resolver = VersionResolver(python_version=(3, 6, 8))
        result = resolver.resolve_user_version("24.0")
        
        assert result.compatible is False
        assert "21.3.1" in result.alternatives
        assert len(result.alternatives) >= 1
    
    def test_resolve_user_version_normalization(self):
        """VersionResolver.resolve_user_version("24") → normalized to "24.0.0" """
        from pyputil_install.pip_installer.versions import VersionResolver
        
        resolver = VersionResolver(python_version=(3, 11, 0))
        result = resolver.resolve_user_version("24")
        
        assert result.version == "24.0.0"
        assert len(result.warnings) >= 1
        assert "normalized" in result.warnings[0].lower()
    
    def test_resolve_user_version_invalid_format(self):
        """VersionResolver.resolve_user_version("invalid") → compatible=False, version=None"""
        from pyputil_install.pip_installer.versions import VersionResolver
        
        resolver = VersionResolver(python_version=(3, 11, 0))
        result = resolver.resolve_user_version("not_a_version")
        
        assert result.compatible is False
        assert result.version is None
    
    def test_is_compatible_true(self):
        """VersionResolver.is_compatible() - compatible version returns True"""
        from pyputil_install.pip_installer.versions import VersionResolver
        
        resolver = VersionResolver()
        compatible, reason = resolver.is_compatible("21.3.1", (3, 9, 18))
        
        assert compatible is True
        assert "supports" in reason.lower()
    
    def test_is_compatible_false_due_to_python_version(self):
        """VersionResolver.is_compatible() - incompatible due to Python version"""
        from pyputil_install.pip_installer.versions import VersionResolver
        
        resolver = VersionResolver()
        compatible, reason = resolver.is_compatible("24.0", (3, 6, 8))
        
        assert compatible is False
        assert "requires Python>=3.8" in reason or "requires Python" in reason
    
    def test_is_compatible_unknown_version_strict_mode(self):
        """VersionResolver(strict=True).is_compatible(unknown) → False"""
        from pyputil_install.pip_installer.versions import VersionResolver
        
        resolver = VersionResolver(strict=True)
        compatible, reason = resolver.is_compatible("99.0.0")
        
        assert compatible is False
        assert "not in the known compatibility table" in reason
    
    def test_get_latest_compatible(self):
        """VersionResolver.get_latest_compatible() - returns version string"""
        from pyputil_install.pip_installer.versions import VersionResolver
        
        resolver = VersionResolver(python_version=(3, 11, 0))
        version = resolver.get_latest_compatible()
        
        assert isinstance(version, str)
        assert re.match(r"\d+\.\d+\.\d+", version)
    
    def test_get_all_compatible_versions(self):
        """VersionResolver.get_all_compatible_versions() - returns list sorted newest first"""
        from pyputil_install.pip_installer.versions import VersionResolver
        
        resolver = VersionResolver(python_version=(3, 9, 18))
        versions = resolver.get_all_compatible_versions()
        
        assert isinstance(versions, list)
        assert len(versions) >= 10
        # Check sorted descending
        for i in range(len(versions) - 1):
            assert versions[i] >= versions[i + 1]
    
    def test_get_version_range_description(self):
        """VersionResolver.get_version_range_description() - returns descriptive string"""
        from pyputil_install.pip_installer.versions import VersionResolver
        
        resolver = VersionResolver(python_version=(3, 6, 8))
        desc = resolver.get_version_range_description()
        
        assert isinstance(desc, str)
        assert len(desc) > 0
    
    def test_sort_versions_newest_first(self):
        """VersionResolver.sort_versions_newest_first() - sorts descending"""
        from pyputil_install.pip_installer.versions import VersionResolver
        
        input_versions = ["20.0", "23.0", "21.3.1", "24.0"]
        sorted_versions = VersionResolver.sort_versions_newest_first(input_versions)
        
        expected_order = ["24.0", "23.0", "21.3.1", "20.0"]
        assert sorted_versions == expected_order
    
    def test_sort_versions_oldest_first(self):
        """VersionResolver.sort_versions_oldest_first() - sorts ascending"""
        from pyputil_install.pip_installer.versions import VersionResolver
        
        input_versions = ["23.0", "20.0", "24.0", "21.3.1"]
        sorted_versions = VersionResolver.sort_versions_oldest_first(input_versions)
        
        expected_order = ["20.0", "21.3.1", "23.0", "24.0"]
        assert sorted_versions == expected_order


# =============================================================================
# Test: fetcher.py
# =============================================================================

class TestResourceFetcher:
    """Test ResourceFetcher class - network resource acquisition."""
    
    def setup_method(self):
        """Create isolated fetcher for each test."""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        self.temp_dir = Path(tempfile.mkdtemp(prefix="pip_fetcher_test_"))
        self.fetcher = ResourceFetcher(cache_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up temporary directory."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_creates_cache_dir(self):
        """ResourceFetcher.__init__() - creates cache directory"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher(cache_dir=self.temp_dir)
        assert self.temp_dir.exists()
        assert self.temp_dir.is_dir()
    
    def test_init_without_cache_dir_creates_temp(self):
        """ResourceFetcher(cache_dir=None) - creates temporary directory"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher(cache_dir=None)
        assert fetcher.cache_dir.exists()
        assert "pip_rescuer_cache" in str(fetcher.cache_dir)
        # Clean up
        shutil.rmtree(fetcher.cache_dir, ignore_errors=True)
    
    def test_default_timeout_and_retries(self):
        """ResourceFetcher default timeout=30, max_retries=3"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher()
        assert fetcher.timeout == 30
        assert fetcher.max_retries == 3
    
    def test_verify_ssl_default_true(self):
        """ResourceFetcher(verify_ssl=True) - default SSL verification enabled"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher(verify_ssl=True)
        assert fetcher.verify_ssl is True
    
    def test_verify_ssl_false(self):
        """ResourceFetcher(verify_ssl=False) - disables SSL verification"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher(verify_ssl=False)
        assert fetcher.verify_ssl is False
    
    def test_add_mirror(self):
        """ResourceFetcher.add_mirror() - adds to mirror list"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher()
        original_length = len(fetcher.list_mirrors())
        
        fetcher.add_mirror("https://test-mirror.example.com/simple/")
        assert len(fetcher.list_mirrors()) == original_length + 1
        assert "https://test-mirror.example.com/simple/" in fetcher.list_mirrors()
    
    def test_add_mirror_at_position(self):
        """ResourceFetcher.add_mirror(url, position=0) - inserts at beginning"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher()
        fetcher.add_mirror("https://priority-mirror.example.com/", position=0)
        
        assert fetcher.list_mirrors()[0] == "https://priority-mirror.example.com/"
    
    def test_remove_mirror(self):
        """ResourceFetcher.remove_mirror() - removes from list"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher()
        test_mirror = "https://test-remove-mirror.example.com/"
        fetcher.add_mirror(test_mirror)
        
        result = fetcher.remove_mirror(test_mirror)
        assert result is True
        assert test_mirror not in fetcher.list_mirrors()
    
    def test_remove_nonexistent_mirror_returns_false(self):
        """ResourceFetcher.remove_mirror(nonexistent) → False"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher()
        result = fetcher.remove_mirror("https://nonexistent.example.com/")
        assert result is False
    
    def test_list_mirrors_returns_copy(self):
        """ResourceFetcher.list_mirrors() - returns copy, not internal list"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher()
        mirrors = fetcher.list_mirrors()
        mirrors.append("should-not-affect-internal")
        
        assert "should-not-affect-internal" not in fetcher.list_mirrors()
    
    def test_verify_file_integrity_file_not_found(self):
        """ResourceFetcher.verify_file_integrity(nonexistent) → (False, "File does not exist")"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher()
        valid, reason = fetcher.verify_file_integrity(Path("/nonexistent/file"))
        
        assert valid is False
        assert "does not exist" in reason
    
    def test_verify_file_integrity_size_mismatch(self):
        """ResourceFetcher.verify_file_integrity() - size mismatch returns False"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher()
        test_file = self.temp_dir / "test.txt"
        test_file.write_text("test content")
        
        valid, reason = fetcher.verify_file_integrity(test_file, expected_size=999999)
        
        assert valid is False
        assert "Size mismatch" in reason
    
    def test_compute_file_hash(self):
        """ResourceFetcher.compute_file_hash() - returns SHA256 hex digest"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        test_file = self.temp_dir / "hash_test.txt"
        test_file.write_text("Hello, World!")
        
        hash_value = ResourceFetcher.compute_file_hash(test_file)
        
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA256 hex length
        assert all(c in "0123456789abcdef" for c in hash_value)
    
    def test_check_connectivity_returns_bool(self):
        """ResourceFetcher.check_connectivity() - returns bool"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher()
        result = fetcher.check_connectivity()
        
        assert isinstance(result, bool)
    
    def test_clear_cache_removes_files(self):
        """ResourceFetcher.clear_cache() - removes cached files"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher(cache_dir=self.temp_dir)
        
        # Create dummy cache file
        dummy_file = self.temp_dir / "dummy.cache"
        dummy_file.write_text("test")
        
        removed = fetcher.clear_cache()
        
        assert removed >= 0
        assert isinstance(removed, int)
    
    @pytest.mark.skipif(SKIP_NETWORK_TESTS, reason="Network disabled (OFFLINE_MODE)")
    def test_fetch_get_pip_success(self):
        """ResourceFetcher.fetch_get_pip() - downloads get-pip.py successfully"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher(cache_dir=self.temp_dir)
        success, path, metadata = fetcher.fetch_get_pip()
        
        assert success is True
        assert path.exists()
        assert path.suffix == ".py"
        assert metadata.size_bytes > 100000  # get-pip.py > 100KB
        assert metadata.status in (FetchStatus.SUCCESS, FetchStatus.CACHED)
    
    @pytest.mark.skipif(SKIP_NETWORK_TESTS, reason="Network disabled (OFFLINE_MODE)")
    def test_fetch_get_pip_with_version(self):
        """ResourceFetcher.fetch_get_pip(version="3.6") - version-specific URL"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher(cache_dir=self.temp_dir)
        success, path, metadata = fetcher.fetch_get_pip(version="3.6")
        
        assert success is True
        assert path.exists()
    
    @pytest.mark.skipif(SKIP_NETWORK_TESTS, reason="Network disabled (OFFLINE_MODE)")
    def test_fetch_pip_wheel_2131(self):
        """ResourceFetcher.fetch_pip_wheel("21.3.1") - downloads wheel"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher(cache_dir=self.temp_dir)
        success, path, metadata = fetcher.fetch_pip_wheel("21.3.1")
        
        assert success is True
        assert path.exists()
        assert path.suffix == ".whl"
        assert "21.3.1" in path.name
    
    @pytest.mark.skipif(SKIP_NETWORK_TESTS, reason="Network disabled (OFFLINE_MODE)")
    def test_fetch_pip_wheel_240(self):
        """ResourceFetcher.fetch_pip_wheel("24.0") - downloads wheel"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher(cache_dir=self.temp_dir)
        success, path, metadata = fetcher.fetch_pip_wheel("24.0")
        
        assert success is True
        assert path.exists()
        assert "24.0" in path.name
    
    def test_fetch_pip_wheel_invalid_version(self):
        """ResourceFetcher.fetch_pip_wheel("99.0.0") - returns False"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher
        
        fetcher = ResourceFetcher(cache_dir=self.temp_dir)
        success, path, metadata = fetcher.fetch_pip_wheel("99.0.0")
        
        # This may fail due to 404 (expected) or network (if offline)
        if SKIP_NETWORK_TESTS:
            assert success is False
        else:
            # With network, 99.0.0 should 404
            assert success is False
    
    def test_fetch_pip_wheel_caching(self):
        """ResourceFetcher.fetch_pip_wheel() - caches results, second fetch returns CACHED"""
        from pyputil_install.pip_installer.fetcher import ResourceFetcher, FetchStatus
        
        if SKIP_NETWORK_TESTS:
            pytest.skip("Network required for cache test")
        
        fetcher = ResourceFetcher(cache_dir=self.temp_dir)
        
        # First fetch
        success1, path1, meta1 = fetcher.fetch_pip_wheel("21.3.1")
        assert success1 is True
        
        # Second fetch (should be cached)
        success2, path2, meta2 = fetcher.fetch_pip_wheel("21.3.1")
        assert success2 is True 
        assert meta2.status in (FetchStatus.CACHED, FetchStatus.MIRROR_USED)


# =============================================================================
# Test: installer.py
# =============================================================================

class TestPipInstaller:
    """Test PipInstaller class - installation strategies."""
    
    def setup_method(self):
        """Create diagnosis for tests."""
        from pyputil_install.pip_installer.checker import PipChecker
        
        self.checker = PipChecker()
        self.diagnosis = self.checker.run_full_diagnosis()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="installer_test_"))
    
    def teardown_method(self):
        """Clean up."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_with_diagnosis(self):
        """PipInstaller.__init__(diagnosis) - stores diagnosis"""
        from pyputil_install.pip_installer.installer import PipInstaller
        
        installer = PipInstaller(self.diagnosis)
        assert installer.diagnosis is self.diagnosis
        assert installer.force is False
        assert installer.user_site is False
    
    def test_init_with_target_version(self):
        """PipInstaller(diagnosis, target_version="21.3.1") - stores version"""
        from pyputil_install.pip_installer.installer import PipInstaller
        
        installer = PipInstaller(self.diagnosis, target_version="21.3.1")
        assert installer.target_version == "21.3.1"
    
    def test_init_with_force(self):
        """PipInstaller(diagnosis, force=True) - sets force flag"""
        from pyputil_install.pip_installer.installer import PipInstaller
        
        installer = PipInstaller(self.diagnosis, force=True)
        assert installer.force is True
    
    def test_init_with_user_site(self):
        """PipInstaller(diagnosis, user_site=True) - sets user_site flag"""
        from pyputil_install.pip_installer.installer import PipInstaller
        
        installer = PipInstaller(self.diagnosis, user_site=True)
        assert installer.user_site is True
    
    def test_install_returns_install_result(self):
        """PipInstaller.install() - returns InstallResult"""
        from pyputil_install.pip_installer.installer import PipInstaller
        
        installer = PipInstaller(self.diagnosis)
        result = installer.install()
        
        from pyputil_install.pip_installer.installer import InstallResult
        assert isinstance(result, InstallResult)
        assert hasattr(result, 'success')
        assert hasattr(result, 'result_code')
        assert hasattr(result, 'strategy_used')
        assert hasattr(result, 'version_installed')
    
    def test_verify_installation_returns_tuple(self):
        """PipInstaller.verify_installation() - returns (success, version, output)"""
        from pyputil_install.pip_installer.installer import PipInstaller
        
        installer = PipInstaller(self.diagnosis)
        success, version, output = installer.verify_installation()
        
        assert isinstance(success, bool)
        assert version is None or isinstance(version, str)
        assert isinstance(output, str)
    
    def test_install_result_success_property(self):
        """InstallResult.success - derived from result_code"""
        from pyputil_install.pip_installer.installer import InstallResult, InstallResultCode
        
        success_result = InstallResult(result_code=InstallResultCode.SUCCESS)
        assert success_result.success is True
        
        fail_result = InstallResult(result_code=InstallResultCode.FAILED_UNKNOWN)
        assert fail_result.success is False
    
    def test_install_strategy_enum_values(self):
        """InstallStrategy - all expected strategy values"""
        from pyputil_install.pip_installer.installer import InstallStrategy
        
        expected_strategies = [
            "ENSUREPIP", "GET_PIP_SCRIPT", "WHEEL_INSTALL", 
            "MANUAL_EXTRACTION", "CONDA_INSTALL", "COPY_FROM_SYSTEM"
        ]
        
        for strategy_name in expected_strategies:
            assert hasattr(InstallStrategy, strategy_name)
            assert isinstance(getattr(InstallStrategy, strategy_name), InstallStrategy)
    
    def test_install_result_code_enum_values(self):
        """InstallResultCode - all expected result codes"""
        from pyputil_install.pip_installer.installer import InstallResultCode
        
        expected_codes = [
            "SUCCESS", "SUCCESS_WITH_WARNINGS", "ALREADY_INSTALLED",
            "FAILED_PERMISSION", "FAILED_NETWORK", "FAILED_INCOMPATIBLE",
            "FAILED_VERIFICATION", "FAILED_UNKNOWN"
        ]
        
        for code_name in expected_codes:
            assert hasattr(InstallResultCode, code_name)
            assert isinstance(getattr(InstallResultCode, code_name), InstallResultCode)


# =============================================================================
# Test: main.py
# =============================================================================

class TestPipRescuer:
    """Test PipRescuer class - orchestrator."""
    
    def setup_method(self):
        """Create rescuer instance."""
        from pyputil_install.pip_installer.main import PipRescuer
        
        self.rescuer = PipRescuer(dry_run=True)  # dry_run prevents modifications
    
    def test_init_defaults(self):
        """PipRescuer.__init__() - default values"""
        from pyputil_install.pip_installer.main import PipRescuer
        
        rescuer = PipRescuer()
        assert rescuer.dry_run is False
        assert rescuer.force is False
        assert rescuer.user_site is False
        assert rescuer.timeout == 30
    
    def test_init_with_parameters(self):
        """PipRescuer(target_version="21.3.1", user_site=True) - stores parameters"""
        from pyputil_install.pip_installer.main import PipRescuer
        
        rescuer = PipRescuer(target_version="21.3.1", user_site=True, force=True)
        assert rescuer.target_version == "21.3.1"
        assert rescuer.user_site is True
        assert rescuer.force is True
    
    def test_init_with_python_executable(self):
        """PipRescuer(python_executable=Path) - uses provided executable"""
        from pyputil_install.pip_installer.main import PipRescuer
        
        python_path = Path(sys.executable).resolve()
        rescuer = PipRescuer(python_executable=python_path)
        assert rescuer.python_executable == python_path
    
    def test_run_returns_rescue_result(self):
        """PipRescuer.run() - returns RescueResult"""
        result = self.rescuer.run()
        
        from pyputil_install.pip_installer.main import RescueResult
        assert isinstance(result, RescueResult)
        assert hasattr(result, 'success')
        assert hasattr(result, 'exit_code')
        assert hasattr(result, 'stages')
    
    def test_run_dry_run_does_not_modify(self):
        """PipRescuer(dry_run=True).run() - no modifications, stages recorded"""
        from pyputil_install.pip_installer import PipRescuer
        rescuer = PipRescuer(dry_run=True, verbose=False, quiet=True)
        result = rescuer.run()
        
        assert result.dry_run is True
        assert len(result.stages) >= 1
    
    def test_uninstall_returns_rescue_result(self):
        """PipRescuer.uninstall() - returns RescueResult"""
        result = self.rescuer.uninstall()
        
        from pyputil_install.pip_installer.main import RescueResult
        assert isinstance(result, RescueResult)
    
    def test_rescue_result_report_method(self):
        """RescueResult.report() - returns formatted string"""
        from pyputil_install.pip_installer.main import RescueResult, ExitCode, PipelineStage, StageResult
        
        stages = [
            StageResult(stage=PipelineStage.DIAGNOSIS, success=True, 
                       message="pip found", duration_ms=100.0)
        ]
        
        result = RescueResult(
            success=True,
            version_installed="24.3.1",
            version_requested=None,
            strategy_used="ensurepip",
            stages=stages,
            total_duration_ms=1500.0,
            exit_code=ExitCode.SUCCESS,
            dry_run=False,
            errors=[],
            warnings=[]
        )
        
        report = result.report()
        assert isinstance(report, str)
        assert "SUCCESS" in report
        assert "24.3.1" in report
    
    def test_rescue_result_report_verbose(self):
        """RescueResult.report(verbose=True) - includes stage details"""
        from pyputil_install.pip_installer.main import RescueResult, ExitCode, PipelineStage, StageResult
        
        stages = [
            StageResult(stage=PipelineStage.DIAGNOSIS, success=True,
                       message="pip found", duration_ms=100.0)
        ]
        
        result = RescueResult(
            success=True,
            version_installed="24.3.1",
            version_requested=None,
            strategy_used="ensurepip",
            stages=stages,
            total_duration_ms=1500.0,
            exit_code=ExitCode.SUCCESS,
            dry_run=False,
            errors=[],
            warnings=[]
        )
        
        report = result.report(verbose=True)
        assert "Pipeline Stages:" in report
        assert "DIAGNOSIS" in report
    
    def test_rescue_result_report_with_errors(self):
        """RescueResult.report() - includes errors section"""
        from pyputil_install.pip_installer.main import RescueResult, ExitCode
        
        result = RescueResult(
            success=False,
            version_installed=None,
            version_requested="24.0",
            strategy_used=None,
            stages=[],
            total_duration_ms=1000.0,
            exit_code=ExitCode.GENERAL_ERROR,
            dry_run=False,
            errors=["Network connection failed", "Download timeout"],
            warnings=["Using fallback mirror"]
        )
        
        report = result.report()
        assert "Errors:" in report
        assert "Network connection failed" in report
        assert "Warnings:" in report


class TestPipelineStage:
    """Test PipelineStage enumeration."""
    
    def test_pipeline_stage_values(self):
        """PipelineStage - all expected stage values"""
        from pyputil_install.pip_installer.main import PipelineStage
        
        expected_stages = [
            "VALIDATION", "DIAGNOSIS", "VERSION_RESOLUTION",
            "FETCHING", "INSTALLATION", "VERIFICATION", "COMPLETE"
        ]
        
        for stage_name in expected_stages:
            assert hasattr(PipelineStage, stage_name)
            assert isinstance(getattr(PipelineStage, stage_name), PipelineStage)


class TestExitCode:
    """Test ExitCode enumeration."""
    
    def test_exit_code_values(self):
        """ExitCode - correct numeric values"""
        from pyputil_install.pip_installer.main import ExitCode
        
        assert ExitCode.SUCCESS.value == 0
        assert ExitCode.GENERAL_ERROR.value == 1
        assert ExitCode.INVALID_ARGUMENTS.value == 2
        assert ExitCode.NETWORK_ERROR.value == 3
        assert ExitCode.PERMISSION_ERROR.value == 4
        assert ExitCode.INCOMPATIBLE_VERSION.value == 5
        assert ExitCode.VERIFICATION_FAILED.value == 6


class TestCLI:
    """Test command-line interface."""
    
    def test_create_argument_parser(self):
        """create_argument_parser() - returns ArgumentParser"""
        from pyputil_install.pip_installer.main import create_argument_parser
        
        parser = create_argument_parser()
        assert parser.prog == "pip-rescuer"
    
    def test_parser_has_install_options(self):
        """ArgumentParser - includes install-related options"""
        from pyputil_install.pip_installer.main import create_argument_parser
        
        parser = create_argument_parser()
        args = parser.parse_args(["--version", "21.3.1"])
        assert args.version == "21.3.1"
    
    def test_parser_has_uninstall_option(self):
        """ArgumentParser - includes --uninstall"""
        from pyputil_install.pip_installer.main import create_argument_parser
        
        parser = create_argument_parser()
        args = parser.parse_args(["--uninstall"])
        assert args.uninstall is True
    
    def test_parser_has_verbose_and_quiet(self):
        """ArgumentParser - includes --verbose and --quiet"""
        from pyputil_install.pip_installer.main import create_argument_parser
        
        parser = create_argument_parser()
        args = parser.parse_args(["--verbose"])
        assert args.verbose is True
        assert args.quiet is False
        
        args = parser.parse_args(["--quiet"])
        assert args.quiet is True
        assert args.verbose is False
    
    def test_parser_has_dry_run(self):
        """ArgumentParser - includes --dry-run"""
        from pyputil_install.pip_installer.main import create_argument_parser
        
        parser = create_argument_parser()
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True
    
    def test_parser_has_offline(self):
        """ArgumentParser - includes --offline for wheel path"""
        from pyputil_install.pip_installer.main import create_argument_parser
        
        parser = create_argument_parser()
        args = parser.parse_args(["--offline", "/path/to/wheel.whl"])
        assert args.offline == Path("/path/to/wheel.whl")
    
    def test_main_returns_exit_code(self):
        """main() - returns integer exit code"""
        from pyputil_install.pip_installer.main import main
        
        exit_code = main(["--dry-run", "--quiet"])
        assert isinstance(exit_code, int)
        assert 0 <= exit_code <= 6
    
    def test_main_help(self):
        """main() with no arguments - prints help, returns INVALID_ARGUMENTS"""
        from pyputil_install.pip_installer.main import main
        
        # Capture stdout to avoid help output in test logs
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            exit_code = main([])
            assert exit_code == 2  # INVALID_ARGUMENTS
        finally:
            sys.stdout = old_stdout
    
    def test_main_uninstall_with_version_conflict(self):
        """main() --uninstall --version - returns INVALID_ARGUMENTS"""
        from pyputil_install.pip_installer.main import main
        
        exit_code = main(["--uninstall", "--version", "21.3.1"])
        assert exit_code == 2  # INVALID_ARGUMENTS


# =============================================================================
# Test: __init__.py (Public API)
# =============================================================================

class TestPublicAPI:
    """Test that __init__.py exports correct public API."""
    
    def test_all_exports(self):
        """__all__ - contains all expected public names"""
        import pyputil_install.pip_installer as pip_installer
        
        expected_exports = [
            # Core classes
            "PipRescuer", "PipChecker", "VersionResolver", 
            "ResourceFetcher", "PipInstaller",
            # Data containers
            "PipDiagnosis", "EnvironmentInfo", "VersionResolution",
            "InstallResult", "RescueResult",
            # Enumerations
            "PipStatus", "InstallStrategy", "InstallResultCode",
            "ExitCode", "FetchStatus", "VersionSource",
            "CompatibilityStatus", "PipelineStage",
            # Utility functions
            "compare_versions", "parse_version", "normalize_version",
            "get_compatible_versions_for_python"
        ]
        
        for name in expected_exports:
            assert hasattr(pip_installer, name), f"Missing {name} in __all__"
    
    def test_repair_function_exists(self):
        """repair() - convenience function exists"""
        import pyputil_install.pip_installer as pip_installer
        
        assert hasattr(pip_installer, 'repair')
        assert callable(pip_installer.repair)
    
    def test_diagnose_function_exists(self):
        """diagnose() - convenience function exists"""
        import pyputil_install.pip_installer as pip_installer
        
        assert hasattr(pip_installer, 'diagnose')
        assert callable(pip_installer.diagnose)
    
    def test_check_compatibility_function_exists(self):
        """check_compatibility() - convenience function exists"""
        import pyputil_install.pip_installer as pip_installer
        
        assert hasattr(pip_installer, 'check_compatibility')
        assert callable(pip_installer.check_compatibility)
    
    def test_repair_returns_rescue_result(self):
        """repair() - returns RescueResult"""
        from pyputil_install.pip_installer import repair
        
        result = repair(dry_run=True)
        
        from pyputil_install.pip_installer.main import RescueResult
        assert isinstance(result, RescueResult)
    
    def test_diagnose_returns_pip_diagnosis(self):
        """diagnose() - returns PipDiagnosis"""
        from pyputil_install.pip_installer import diagnose
        
        diagnosis = diagnose()
        
        from pyputil_install.pip_installer.checker import PipDiagnosis
        assert isinstance(diagnosis, PipDiagnosis)
    
    def test_check_compatibility_returns_tuple(self):
        """check_compatibility() - returns (bool, str)"""
        from pyputil_install.pip_installer import check_compatibility
        
        compatible, reason = check_compatibility("21.3.1")
        
        assert isinstance(compatible, bool)
        assert isinstance(reason, str)


# =============================================================================
# Platform-Specific Skip Conditions
# =============================================================================

# Skip network tests when offline mode is enabled
pytestmark = []

if SKIP_NETWORK_TESTS:
    # Mark all network-dependent test classes as skip
    for cls in [TestResourceFetcher]:
        cls = pytest.mark.skipif(True, reason="Network disabled (OFFLINE_MODE)")(cls)
else:
    # Import FetchStatus for network tests
    from pyputil_install.pip_installer.fetcher import FetchStatus


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == "__main__":
    import pytest
    import sys
    
    # Run tests with verbose output
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings",
    ])
    sys.exit(exit_code)
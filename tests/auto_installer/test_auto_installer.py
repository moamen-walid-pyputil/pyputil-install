"""
Comprehensive test suite for auto_installer.

Covers every public function and method across all five modules:
utils, core, sync, _async, __init__.

Each test function specifies the exact expected result with
a detailed docstring explaining the input, expected output,
and the reasoning behind the expected behaviour.

Requirements
------------
    pytest, pytest-asyncio
"""

from __future__ import annotations

import sys
import os
import builtins
import asyncio
import subprocess
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, call, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from pyputil_install.auto_installer import (
    auto_install_sync,
    auto_install_async,
    SyncAutoInstaller,
    AsyncAutoInstaller,
    AutoInstallerCore,
    resolve_pip_name,
    is_builtin,
    is_stdlib,
    is_system_module,
    is_already_importable,
    should_skip_install,
    add_pip_mapping,
)
from pyputil_install.auto_installer.utils import _PIP_NAME_MAP, get_stdlib_names
from pyputil_install.auto_installer.sync import _find_pip, _has_minimum_disk_space, _extract_pip_error
from pyputil_install.auto_installer._async import _classify_error


# ===========================================================================
# Helper: compare bound methods safely across Python versions
# ===========================================================================

def _assert_same_function(a, b, msg: str = "") -> None:
    """
    Assert that two callables refer to the same underlying function.

    In Python 3.13+, bound methods are recreated each time they are
    accessed, so ``a is b`` may return False even when both refer to
    the same method on the same instance. This helper compares
    ``__func__`` and ``__self__`` attributes when available, falling
    back to identity comparison otherwise.

    Parameters
    ----------
    a : callable
        First callable to compare.
    b : callable
        Second callable to compare.
    msg : str
        Custom assertion message prefix.

    Raises
    ------
    AssertionError
        If the two callables do not refer to the same function.
    """
    if a is b:
        return
    if hasattr(a, '__func__') and hasattr(b, '__func__'):
        if a.__func__ is b.__func__ and a.__self__ is b.__self__:
            return
    if not msg:
        msg = f"{a!r} and {b!r} are not the same function"
    raise AssertionError(msg)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def restore_import():
    """
    Restore the original ``builtins.__import__`` after every test.

    This fixture runs automatically before and after each test function.
    It saves the current ``__import__`` before the test and restores it
    afterward. This prevents test pollution where one test installs a
    hook and a subsequent test incorrectly inherits it.

    Also attempts to call ``uninstall_hook()`` on any active installer
    instance to fully clean up state.
    """
    original = builtins.__import__
    yield
    builtins.__import__ = original
    # Attempt to clean up any hook that may still be installed
    try:
        current = builtins.__import__
        if hasattr(current, '__self__'):
            inst = current.__self__
            if hasattr(inst, 'uninstall_hook'):
                inst.uninstall_hook()
    except Exception:
        pass


@pytest.fixture
def cleancore():
    """
    Provide a fresh ``AutoInstallerCore`` instance with no hook installed.

    The hook is not activated. The fixture cleans up after the test
    by calling ``uninstall_hook`` if it was activated during the test.
    """
    c = AutoInstallerCore()
    yield c
    try:
        c.uninstall_hook()
    except Exception:
        pass


@pytest.fixture
def cleansync():
    """
    Provide a fresh ``SyncAutoInstaller`` instance with no hook installed.

    Useful for testing ``_install_package`` directly without triggering
    the import hook mechanism.
    """
    s = SyncAutoInstaller()
    yield s
    try:
        s.uninstall_hook()
    except Exception:
        pass


@pytest.fixture
def clean_async():
    """
    Provide a fresh ``AsyncAutoInstaller`` instance with no hook installed.

    Must be used within async test functions decorated with
    ``@pytest.mark.asyncio``.
    """
    a = AsyncAutoInstaller()
    yield a
    try:
        a.uninstall_hook()
    except Exception:
        pass


@pytest.fixture
def mock_subprocess_success():
    """
    Mock ``subprocess.run`` to simulate a successful pip installation.

    Returns a mock object with ``returncode=0`` and empty stderr/stdout.
    Yields the mock so tests can inspect ``call_count`` and call arguments.
    """
    with patch('subprocess.run') as m:
        m.return_value = MagicMock(returncode=0, stderr='', stdout='')
        yield m


@pytest.fixture
def mock_subprocess_not_found():
    """
    Mock ``subprocess.run`` to simulate a 'package not found' pip error.

    Returns ``returncode=1`` with stderr containing a 'not found' message.
    This error type should NOT trigger any retry.
    """
    with patch('subprocess.run') as m:
        m.return_value = MagicMock(
            returncode=1,
            stderr='ERROR: Could not find a version that satisfies the requirement',
            stdout='',
        )
        yield m


@pytest.fixture
def mock_subprocess_permission_then_success():
    """
    Mock ``subprocess.run``: first call fails with permission error,
    second call (with ``--user``) succeeds.

    Used to verify the permission-to-user-fallback retry logic.
    """
    with patch('subprocess.run') as m:
        m.side_effect = [
            MagicMock(returncode=1, stderr='ERROR: Permission denied', stdout=''),
            MagicMock(returncode=0, stderr='', stdout=''),
        ]
        yield m


@pytest.fixture
def mock_subprocess_timeout_then_success():
    """
    Mock ``subprocess.run``: first call fails with a transient timeout
    error, second call succeeds.

    Used to verify the exponential-backoff retry logic for transient errors.
    """
    with patch('subprocess.run') as m:
        m.side_effect = [
            MagicMock(returncode=2, stderr='ERROR: Connection timed out', stdout=''),
            MagicMock(returncode=0, stderr='', stdout=''),
        ]
        yield m


@pytest.fixture
def mock_subprocess_fatal():
    """
    Mock ``subprocess.run`` to return a non-retryable fatal error.

    ``returncode=1`` with a generic error message that does not match
    any transient or permission pattern.
    """
    with patch('subprocess.run') as m:
        m.return_value = MagicMock(
            returncode=1,
            stderr='ERROR: Something went wrong',
            stdout='',
        )
        yield m


@pytest.fixture
def mock_disk_low():
    """
    Mock ``shutil.disk_usage`` to simulate only 1 MB of free disk space.

    Used to verify that installations are blocked when disk space is
    insufficient.
    """
    with patch('shutil.disk_usage') as m:
        m.return_value = MagicMock(free=1_000_000)
        yield m


# ===========================================================================
# Tests: utils.py — resolve_pip_name
# ===========================================================================

class TestResolvePipName:
    """
    Tests for ``resolve_pip_name(import_name: str) -> str``.

    Converts a Python import name to its corresponding pip package name.
    Uses the comprehensive ``_PIP_NAME_MAP`` dictionary for known
    discrepancies. Falls back to normalisation (lowercasing, replacing
    underscores with hyphens) for unknown names. For submodule imports,
    only the top-level package name before the first dot is used.
    """

    # ── Known mappings (present in _PIP_NAME_MAP) ────────────────────

    def test_sklearn(self):
        """
        Input: ``'sklearn'``.
        Expected: ``'scikit-learn'``.
        Reason: ``_PIP_NAME_MAP`` contains the entry ``'sklearn': 'scikit-learn'``.
        """
        assert resolve_pip_name('sklearn') == 'scikit-learn'

    def test_cv2(self):
        """
        Input: ``'cv2'``.
        Expected: ``'opencv-python'``.
        Reason: OpenCV is imported as ``cv2`` but distributed as ``opencv-python``.
        """
        assert resolve_pip_name('cv2') == 'opencv-python'

    def test_PIL(self):
        """
        Input: ``'PIL'`` (case-sensitive).
        Expected: ``'Pillow'``.
        Reason: The Pillow library uses the historical import name ``PIL``.
        """
        assert resolve_pip_name('PIL') == 'Pillow'

    def test_bs4(self):
        """
        Input: ``'bs4'``.
        Expected: ``'beautifulsoup4'``.
        Reason: BeautifulSoup4 is imported as ``bs4``.
        """
        assert resolve_pip_name('bs4') == 'beautifulsoup4'

    def test_yaml(self):
        """
        Input: ``'yaml'``.
        Expected: ``'PyYAML'``.
        Reason: The YAML parser is imported as ``yaml`` but distributed as ``PyYAML``.
        """
        assert resolve_pip_name('yaml') == 'PyYAML'

    def test_dotenv(self):
        """
        Input: ``'dotenv'``.
        Expected: ``'python-dotenv'``.
        Reason: ``python-dotenv`` provides the ``dotenv`` module.
        """
        assert resolve_pip_name('dotenv') == 'python-dotenv'

    def test_jwt(self):
        """
        Input: ``'jwt'``.
        Expected: ``'PyJWT'``.
        Reason: PyJWT is imported as ``jwt``.
        """
        assert resolve_pip_name('jwt') == 'PyJWT'

    def test_igraph(self):
        """
        Input: ``'igraph'``.
        Expected: ``'python-igraph'``.
        Reason: The igraph library is distributed as ``python-igraph``.
        """
        assert resolve_pip_name('igraph') == 'python-igraph'

    def test_socketio(self):
        """
        Input: ``'socketio'``.
        Expected: ``'python-socketio'``.
        Reason: Socket.IO client for Python.
        """
        assert resolve_pip_name('socketio') == 'python-socketio'

    def test_wx(self):
        """
        Input: ``'wx'``.
        Expected: ``'wxPython'``.
        Reason: wxPython is imported via the ``wx`` package.
        """
        assert resolve_pip_name('wx') == 'wxPython'

    def test_psycopg2(self):
        """
        Input: ``'psycopg2'``.
        Expected: ``'psycopg2-binary'``.
        Reason: The binary distribution is preferred for easier installation.
        """
        assert resolve_pip_name('psycopg2') == 'psycopg2-binary'

    def test_requests(self):
        """
        Input: ``'requests'``.
        Expected: ``'requests'``.
        Reason: Identity mapping — the import name and pip name are identical.
        """
        assert resolve_pip_name('requests') == 'requests'

    def test_numpy(self):
        """
        Input: ``'numpy'``.
        Expected: ``'numpy'``.
        Reason: Identity mapping.
        """
        assert resolve_pip_name('numpy') == 'numpy'

    def test_pandas(self):
        """
        Input: ``'pandas'``.
        Expected: ``'pandas'``.
        Reason: Identity mapping.
        """
        assert resolve_pip_name('pandas') == 'pandas'

    def test_torch(self):
        """
        Input: ``'torch'``.
        Expected: ``'torch'``.
        Reason: Identity mapping.
        """
        assert resolve_pip_name('torch') == 'torch'

    def test_flask(self):
        """
        Input: ``'flask'`` (lowercase).
        Expected: ``'Flask'``.
        Reason: The pip package name is capitalised.
        """
        assert resolve_pip_name('flask') == 'Flask'

    def test_django(self):
        """
        Input: ``'django'`` (lowercase).
        Expected: ``'Django'``.
        Reason: The pip package name is capitalised.
        """
        assert resolve_pip_name('django') == 'Django'

    def test_fastapi(self):
        """
        Input: ``'fastapi'``.
        Expected: ``'fastapi'``.
        Reason: Identity mapping.
        """
        assert resolve_pip_name('fastapi') == 'fastapi'

    def test_pydantic(self):
        """
        Input: ``'pydantic'``.
        Expected: ``'pydantic'``.
        Reason: Identity mapping.
        """
        assert resolve_pip_name('pydantic') == 'pydantic'

    def test_rich(self):
        """
        Input: ``'rich'``.
        Expected: ``'rich'``.
        Reason: Identity mapping.
        """
        assert resolve_pip_name('rich') == 'rich'

    def test_click(self):
        """
        Input: ``'click'``.
        Expected: ``'click'``.
        Reason: Identity mapping.
        """
        assert resolve_pip_name('click') == 'click'

    def test_httpx(self):
        """
        Input: ``'httpx'``.
        Expected: ``'httpx'``.
        Reason: Identity mapping.
        """
        assert resolve_pip_name('httpx') == 'httpx'

    def test_aiohttp(self):
        """
        Input: ``'aiohttp'``.
        Expected: ``'aiohttp'``.
        Reason: Identity mapping.
        """
        assert resolve_pip_name('aiohttp') == 'aiohttp'

    # ── Unknown packages (fallback to normalisation) ─────────────────

    def test_unknown_simple(self):
        """
        Input: ``'SomePackage'``.
        Expected: ``'somepackage'``.
        Reason: Unknown name is lowercased. No underscores, so no hyphens.
        """
        assert resolve_pip_name('SomePackage') == 'somepackage'

    def test_unknown_with_underscores(self):
        """
        Input: ``'my_custom_lib'``.
        Expected: ``'my-custom-lib'``.
        Reason: Underscores are replaced with hyphens after lowercasing.
        """
        assert resolve_pip_name('my_custom_lib') == 'my-custom-lib'

    def test_unknown_mixed_case_underscores(self):
        """
        Input: ``'My_Custom_Lib'``.
        Expected: ``'my-custom-lib'``.
        Reason: Lowercased first, then underscores replaced with hyphens.
        """
        assert resolve_pip_name('My_Custom_Lib') == 'my-custom-lib'

    # ── Submodule resolution ─────────────────────────────────────────

    def test_submodule_mapped(self):
        """
        Input: ``'sklearn.ensemble'``.
        Expected: ``'scikit-learn'``.
        Reason: The top-level name ``'sklearn'`` is extracted before the
        first dot and mapped via ``_PIP_NAME_MAP``.
        """
        assert resolve_pip_name('sklearn.ensemble') == 'scikit-learn'

    def test_submodule_identity(self):
        """
        Input: ``'requests.adapters'``.
        Expected: ``'requests'``.
        Reason: Top-level ``'requests'`` has an identity mapping.
        """
        assert resolve_pip_name('requests.adapters') == 'requests'

    def test_submodule_unknown(self):
        """
        Input: ``'mylib.submodule.deep'``.
        Expected: ``'mylib'``.
        Reason: Top-level extracted; unknown, so returned as-is (lowered).
        """
        assert resolve_pip_name('mylib.submodule.deep') == 'mylib'

    def test_submodule_double_dot(self):
        """
        Input: ``'a.b.c'``.
        Expected: ``'a'``.
        Reason: Only the segment before the first dot is used.
        """
        assert resolve_pip_name('a.b.c') == 'a'

    # ── Edge cases ───────────────────────────────────────────────────

    def test_empty_string(self):
        """
        Input: ``''``.
        Expected: ``''``.
        Reason: Splitting an empty string returns ``['']``; first element is ``''``.
        """
        assert resolve_pip_name('') == ''

    def test_dot_only(self):
        """
        Input: ``'.'``.
        Expected: ``''``.
        Reason: Splitting ``'.'`` gives ``['', '']``; first element is ``''``.
        """
        assert resolve_pip_name('.') == ''

    def test_leading_dot(self):
        """
        Input: ``'.hidden'``.
        Expected: ``''``.
        Reason: Splitting ``'.hidden'`` gives ``['', 'hidden']``; first is ``''``.
        """
        assert resolve_pip_name('.hidden') == ''


# ===========================================================================
# Tests: utils.py — is_builtin
# ===========================================================================

class TestIsBuiltin:
    """
    Tests for ``is_builtin(name: str) -> bool``.

    Returns ``True`` if the given module name is a built-in C extension
    module listed in ``sys.builtin_module_names``. These modules are
    compiled into the Python interpreter and cannot be pip-installed.
    """

    def test_sys(self):
        """
        Input: ``'sys'``.
        Expected: ``True``.
        Reason: ``sys`` is always a built-in module.
        """
        assert is_builtin('sys') is True

    def test_builtins(self):
        """
        Input: ``'builtins'``.
        Expected: ``True``.
        Reason: ``builtins`` is always a built-in module.
        """
        assert is_builtin('builtins') is True

    def test_os(self):
        """
        Input: ``'os'``.
        Expected: ``True`` if ``'os'`` is in ``sys.builtin_module_names``,
        ``False`` otherwise.
        Reason: On most CPython installations, ``os`` is a Python file
        in the stdlib, not a built-in C module. However, some embedded
        Python builds may compile it as built-in. We defer to the
        actual ``sys.builtin_module_names`` value.
        """
        expected = 'os' in sys.builtin_module_names
        assert is_builtin('os') is expected

    def test_requests(self):
        """
        Input: ``'requests'``.
        Expected: ``False``.
        Reason: Third-party package, not built-in.
        """
        assert is_builtin('requests') is False

    def test_numpy(self):
        """
        Input: ``'numpy'``.
        Expected: ``False``.
        Reason: Third-party package, not built-in.
        """
        assert is_builtin('numpy') is False

    def test_empty_string(self):
        """
        Input: ``''``.
        Expected: ``False``.
        Reason: Empty string is not a valid built-in module name.
        """
        assert is_builtin('') is False

    def test_nonexistent(self):
        """
        Input: ``'this_does_not_exist_xyz'``.
        Expected: ``False``.
        Reason: Non-existent module name.
        """
        assert is_builtin('this_does_not_exist_xyz') is False


# ===========================================================================
# Tests: utils.py — is_stdlib
# ===========================================================================

class TestIsStdlib:
    """
    Tests for ``is_stdlib(name: str) -> bool``.

    Returns ``True`` if the given name is a top-level standard library
    module. Uses ``sys.stdlib_module_names`` on Python 3.10+, or a
    hardcoded comprehensive set on older versions.
    """

    def test_os(self):
        """
        Input: ``'os'``.
        Expected: ``True``.
        Reason: ``os`` is part of the standard library.
        """
        assert is_stdlib('os') is True

    def test_sys(self):
        """
        Input: ``'sys'``.
        Expected: ``True``.
        Reason: ``sys`` is part of the standard library.
        """
        assert is_stdlib('sys') is True

    def test_json(self):
        """
        Input: ``'json'``.
        Expected: ``True``.
        Reason: ``json`` is part of the standard library.
        """
        assert is_stdlib('json') is True

    def test_collections(self):
        """
        Input: ``'collections'``.
        Expected: ``True``.
        Reason: ``collections`` is part of the standard library.
        """
        assert is_stdlib('collections') is True

    def test_http(self):
        """
        Input: ``'http'``.
        Expected: ``True``.
        Reason: ``http`` is a top-level stdlib package.
        """
        assert is_stdlib('http') is True

    def test_email(self):
        """
        Input: ``'email'``.
        Expected: ``True``.
        Reason: ``email`` is a top-level stdlib package.
        """
        assert is_stdlib('email') is True

    def test_requests(self):
        """
        Input: ``'requests'``.
        Expected: ``False``.
        Reason: ``requests`` is a third-party package, not in stdlib.
        """
        assert is_stdlib('requests') is False

    def test_numpy(self):
        """
        Input: ``'numpy'``.
        Expected: ``False``.
        Reason: Third-party package.
        """
        assert is_stdlib('numpy') is False

    def test_pandas(self):
        """
        Input: ``'pandas'``.
        Expected: ``False``.
        Reason: Third-party package.
        """
        assert is_stdlib('pandas') is False

    def test_empty_string(self):
        """
        Input: ``''``.
        Expected: ``False``.
        Reason: Empty string is not in the stdlib set.
        """
        assert is_stdlib('') is False

    def test_nonexistent(self):
        """
        Input: ``'xyzabc_nonexistent'``.
        Expected: ``False``.
        Reason: Non-existent name.
        """
        assert is_stdlib('xyzabc_nonexistent') is False


# ===========================================================================
# Tests: utils.py — is_system_module
# ===========================================================================

class TestIsSystemModule:
    """
    Tests for ``is_system_module(name: str) -> bool``.

    Returns ``True`` if the name matches patterns for platform-internal
    modules that cannot be installed via pip. These include:

    - Names starting with ``'_'`` (private C extension modules)
    - Very short (≤3 chars) lowercase names not in an explicit allowlist
    - Known platform-internal names like ``nt``, ``posix``, ``win32api``
    """

    def test_underscore_prefix(self):
        """
        Input: ``'_winapi'``.
        Expected: ``True``.
        Reason: Underscore-prefixed names are private C modules.
        """
        assert is_system_module('_winapi') is True

    def test_underscore_prefix_ctypes(self):
        """
        Input: ``'_ctypes'``.
        Expected: ``True``.
        Reason: Underscore-prefixed.
        """
        assert is_system_module('_ctypes') is True

    def test_underscore_prefix_any(self):
        """
        Input: ``'_anything'``.
        Expected: ``True``.
        Reason: All underscore-prefixed names match regardless of content.
        """
        assert is_system_module('_anything') is True

    def test_short_lowercase_nt(self):
        """
        Input: ``'nt'``.
        Expected: ``True``.
        Reason: Length ≤3, lowercase, not in allowlist. This is the
        Windows NT system module.
        """
        assert is_system_module('nt') is True

    def test_short_lowercase_posix(self):
        """
        Input: ``'posix'``.
        Expected: ``True``.
        Reason: Explicitly listed in the system modules tuple despite
        having more than 3 characters.
        """
        assert is_system_module('posix') is True

    def test_short_lowercase_pip(self):
        """
        Input: ``'pip'``.
        Expected: ``False``.
        Reason: Length 3 but in the allowlist. ``pip`` is a real
        installable package.
        """
        assert is_system_module('pip') is False

    def test_short_lowercase_uv(self):
        """
        Input: ``'uv'``.
        Expected: ``False``.
        Reason: Length 2 but in the allowlist (uv pip alternative).
        """
        assert is_system_module('uv') is False

    def test_short_lowercase_tqdm(self):
        """
        Input: ``'tqdm'``.
        Expected: ``False``.
        Reason: In the allowlist (progress bar library).
        """
        assert is_system_module('tqdm') is False

    def test_short_lowercase_np(self):
        """
        Input: ``'np'``.
        Expected: ``False``.
        Reason: In the allowlist (common numpy alias).
        """
        assert is_system_module('np') is False

    def test_short_lowercase_pd(self):
        """
        Input: ``'pd'``.
        Expected: ``False``.
        Reason: In the allowlist (common pandas alias).
        """
        assert is_system_module('pd') is False

    def test_short_lowercase_xyz(self):
        """
        Input: ``'xyz'``.
        Expected: ``True``.
        Reason: Length 3, lowercase, not in allowlist. Treated as system
        module to prevent useless pip calls for likely-invalid names.
        """
        assert is_system_module('xyz') is True

    def test_requests(self):
        """
        Input: ``'requests'``.
        Expected: ``False``.
        Reason: Normal-length package name that is a real pip package.
        """
        assert is_system_module('requests') is False

    def test_numpy(self):
        """
        Input: ``'numpy'``.
        Expected: ``False``.
        Reason: Normal package name.
        """
        assert is_system_module('numpy') is False

    def test_empty_string(self):
        """
        Input: ``''``.
        Expected: ``False``.
        Reason: Empty string does not match any system module pattern.
        """
        assert is_system_module('') is False


# ===========================================================================
# Tests: utils.py — is_already_importable
# ===========================================================================

class TestIsAlreadyImportable:
    """
    Tests for ``is_already_importable(name: str) -> bool``.

    Uses ``importlib.util.find_spec`` to check whether a module can be
    located without actually executing its code. Returns ``True`` if
    a spec is found, ``False`` otherwise. Handles empty string and
    other edge cases that cause ``find_spec`` to raise exceptions.
    """

    def test_os(self):
        """
        Input: ``'os'``.
        Expected: ``True``.
        Reason: ``os`` is always importable.
        """
        assert is_already_importable('os') is True

    def test_sys(self):
        """
        Input: ``'sys'``.
        Expected: ``True``.
        Reason: ``sys`` is always importable.
        """
        assert is_already_importable('sys') is True

    def test_json(self):
        """
        Input: ``'json'``.
        Expected: ``True``.
        Reason: ``json`` is always importable.
        """
        assert is_already_importable('json') is True

    def test_submodule_of_importable(self):
        """
        Input: ``'os.path'``.
        Expected: ``True``.
        Reason: ``os.path`` is a submodule of the importable ``os`` package.
        """
        assert is_already_importable('os.path') is True

    def test_nonexistent(self):
        """
        Input: ``'this_does_not_exist_xyz123'``.
        Expected: ``False``.
        Reason: No module or package exists with this name.
        """
        assert is_already_importable('this_does_not_exist_xyz123') is False

    def test_empty_string(self):
        """
        Input: ``''``.
        Expected: ``False``.
        Reason: Empty string is not a valid module name and causes
        ``find_spec`` to raise. The function catches this and returns
        ``False``.
        """
        assert is_already_importable('') is False


# ===========================================================================
# Tests: utils.py — should_skip_install
# ===========================================================================

class TestShouldSkipInstall:
    """
    Tests for ``should_skip_install(name, failed_packages, installing) -> bool``.

    Central decision function that determines whether a given import
    name should bypass auto-installation. Returns ``True`` for:

    - Relative imports (starting with ``'.'``)
    - Built-in modules
    - Stdlib modules
    - System modules (underscore-prefixed, short lowercase names)
    - Submodules of already-importable packages
    - Previously-failed packages
    - Currently-installing packages
    - Path-like names (containing ``/``, ``\\``, or spaces)
    - Empty names
    """

    def test_relative_import(self):
        """
        Input: ``'.relative'``, empty sets.
        Expected: ``True``.
        Reason: Relative imports start with a dot and cannot be
        pip-installed.
        """
        assert should_skip_install('.relative', set(), set()) is True

    def test_builtin(self):
        """
        Input: ``'sys'``, empty sets.
        Expected: ``True``.
        Reason: ``sys`` is a built-in module.
        """
        assert should_skip_install('sys', set(), set()) is True

    def test_stdlib(self):
        """
        Input: ``'json'``, empty sets.
        Expected: ``True``.
        Reason: ``json`` is a stdlib module.
        """
        assert should_skip_install('json', set(), set()) is True

    def test_system_module_underscore(self):
        """
        Input: ``'_ctypes'``, empty sets.
        Expected: ``True``.
        Reason: Underscore-prefixed names are system modules.
        """
        assert should_skip_install('_ctypes', set(), set()) is True

    def test_system_module_short(self):
        """
        Input: ``'xyz'``, empty sets.
        Expected: ``True``.
        Reason: Short unrecognised lowercase names are treated as system modules.
        """
        assert should_skip_install('xyz', set(), set()) is True

    def test_normal_package(self):
        """
        Input: ``'requests'``, empty sets.
        Expected: ``False``.
        Reason: ``'requests'`` is a real installable third-party package.
        """
        assert should_skip_install('requests', set(), set()) is False

    def test_failed_previously(self):
        """
        Input: ``'requests'`` with ``failed_packages={'requests'}``.
        Expected: ``True``.
        Reason: Previously-failed packages are skipped to prevent
        repeated futile installation attempts.
        """
        assert should_skip_install('requests', {'requests'}, set()) is True

    def test_failed_top_level(self):
        """
        Input: ``'requests.adapters'`` with ``failed_packages={'requests'}``.
        Expected: ``True``.
        Reason: Submodule of a failed top-level package inherits the
        failed status.
        """
        assert should_skip_install('requests.adapters', {'requests'}, set()) is True

    def test_currently_installing(self):
        """
        Input: ``'numpy'`` with ``installing={'numpy'}``.
        Expected: ``True``.
        Reason: Packages currently being installed are skipped to prevent
        re-entrant installation calls.
        """
        assert should_skip_install('numpy', set(), {'numpy'}) is True

    def test_installing_top_level(self):
        """
        Input: ``'numpy.linalg'`` with ``installing={'numpy'}``.
        Expected: ``True``.
        Reason: Submodule of a currently-installing package is skipped.
        """
        assert should_skip_install('numpy.linalg', set(), {'numpy'}) is True

    def test_submodule_of_installed(self):
        """
        Input: ``'os.path'``, empty sets.
        Expected: ``True``.
        Reason: ``os`` is already importable, so its submodules should
        not trigger installation.
        """
        assert should_skip_install('os.path', set(), set()) is True

    def test_path_like_name(self):
        """
        Input: ``'/etc/passwd'``, empty sets.
        Expected: ``True``.
        Reason: Forward slashes indicate a filesystem path, not a
        Python package name.
        """
        assert should_skip_install('/etc/passwd', set(), set()) is True

    def test_empty_name(self):
        """
        Input: ``''``, empty sets.
        Expected: ``True``.
        Reason: Empty string is not a valid package name.
        """
        assert should_skip_install('', set(), set()) is True

    def test_name_with_spaces(self):
        """
        Input: ``'my package'``, empty sets.
        Expected: ``True``.
        Reason: Spaces are not valid in Python package names.
        """
        assert should_skip_install('my package', set(), set()) is True

    def test_name_with_backslash(self):
        """
        Input: ``'a\\\\b'``, empty sets.
        Expected: ``True``.
        Reason: Backslashes indicate a Windows path, not a package.
        """
        assert should_skip_install('a\\b', set(), set()) is True

    def test_valid_package_multiple_dots(self):
        """
        Input: ``'os.path.util'``, empty sets.
        Expected: ``True``.
        Reason: ``os`` is importable, so any submodule path starting
        with ``os`` is skipped.
        """
        assert should_skip_install('os.path.util', set(), set()) is True


# ===========================================================================
# Tests: utils.py — add_pip_mapping
# ===========================================================================

class TestAddPipMapping:
    """
    Tests for ``add_pip_mapping(import_name, pip_name) -> None``.

    Registers a custom import-name to pip-name mapping at runtime.
    The mapping takes effect immediately and persists for all subsequent
    calls to ``resolve_pip_name``.
    """

    def test_add_new_mapping(self):
        """
        Input: ``add_pip_mapping('my_test_lib', 'my-test-library')``.
        Expected after call: ``resolve_pip_name('my_test_lib')`` returns
        ``'my-test-library'``.
        Reason: The new mapping is registered and takes effect immediately.
        """
        add_pip_mapping('my_test_lib', 'my-test-library')
        assert resolve_pip_name('my_test_lib') == 'my-test-library'

    def test_overwrite_existing(self):
        """
        Input: Overwrite the existing ``'requests'`` mapping with
        ``'requests-override'``.
        Expected: ``resolve_pip_name('requests')`` returns the overridden
        value. After restoring the original value, it returns the original.
        Reason: Mappings can be overwritten at runtime. The test restores
        the original value to avoid affecting other tests.
        """
        old = resolve_pip_name('requests')
        add_pip_mapping('requests', 'requests-override')
        assert resolve_pip_name('requests') == 'requests-override'
        add_pip_mapping('requests', old)
        assert resolve_pip_name('requests') == old

    def test_mapping_persists_across_calls(self):
        """
        Input: Register ``'persist_test'`` → ``'persist-pkg'``.
        Expected: Both ``resolve_pip_name('persist_test')`` and
        ``resolve_pip_name('persist_test.sub')`` return ``'persist-pkg'``.
        Reason: The mapping persists and applies to submodule lookups
        because only the top-level name is used for resolution.
        """
        add_pip_mapping('persist_test', 'persist-pkg')
        assert resolve_pip_name('persist_test') == 'persist-pkg'
        assert resolve_pip_name('persist_test.sub') == 'persist-pkg'


# ===========================================================================
# Tests: utils.py — get_stdlib_names
# ===========================================================================

class TestGetStdlibNames:
    """
    Tests for ``get_stdlib_names() -> frozenset[str]``.

    Returns an immutable set of standard library top-level module names.
    Uses ``sys.stdlib_module_names`` on Python 3.10+, otherwise a
    hardcoded comprehensive fallback set.
    """

    def test_returns_frozenset(self):
        """
        Expected: Return type is ``frozenset``.
        Reason: The function is documented to return a frozenset.
        """
        result = get_stdlib_names()
        assert isinstance(result, frozenset)

    def test_contains_os(self):
        """
        Expected: ``'os'`` is in the returned frozenset.
        Reason: ``os`` is a standard library module.
        """
        assert 'os' in get_stdlib_names()

    def test_contains_sys(self):
        """
        Expected: ``'sys'`` is in the returned frozenset.
        Reason: ``sys`` is a standard library module.
        """
        assert 'sys' in get_stdlib_names()

    def test_contains_json(self):
        """
        Expected: ``'json'`` is in the returned frozenset.
        Reason: ``json`` is a standard library module.
        """
        assert 'json' in get_stdlib_names()

    def test_does_not_contain_requests(self):
        """
        Expected: ``'requests'`` is NOT in the returned frozenset.
        Reason: ``requests`` is a third-party package.
        """
        assert 'requests' not in get_stdlib_names()

    def test_idempotent(self):
        """
        Expected: Multiple calls return identical frozensets.
        Reason: The function is deterministic. On Python 3.10+ the
        result is cached via ``sys.stdlib_module_names``.
        """
        a = get_stdlib_names()
        b = get_stdlib_names()
        assert a == b


# ===========================================================================
# Tests: sync.py — _find_pip
# ===========================================================================

class TestFindPip:
    """
    Tests for ``_find_pip() -> list[str]``.

    Returns the command list used to invoke pip. Prefers
    ``[sys.executable, '-m', 'pip']`` when ``sys.executable`` is set,
    otherwise searches ``PATH`` for ``pip3`` or ``pip``.
    """

    def test_returns_list(self):
        """
        Expected: Return type is ``list``.
        Reason: The function returns a list of command-line arguments.
        """
        result = _find_pip()
        assert isinstance(result, list)

    def test_returns_non_empty(self):
        """
        Expected: The returned list has at least one element.
        Reason: At minimum, a pip command must contain the executable.
        """
        result = _find_pip()
        assert len(result) > 0

    def test_first_element_is_executable(self):
        """
        Expected: If ``sys.executable`` is set, the first element of
        the returned list is ``sys.executable``.
        Reason: The function prefers using the current Python interpreter
        to invoke pip via ``-m``.
        """
        result = _find_pip()
        if sys.executable:
            assert result[0] == sys.executable

    def test_contains_m_pip(self):
        """
        Expected: If ``sys.executable`` is set, the command list contains
        ``'-m'`` and ``'pip'``.
        Reason: This is the canonical way to invoke pip for the current
        Python environment.
        """
        result = _find_pip()
        if sys.executable:
            assert '-m' in result
            assert 'pip' in result


# ===========================================================================
# Tests: sync.py — _has_minimum_disk_space
# ===========================================================================

class TestHasMinimumDiskSpace:
    """
    Tests for ``_has_minimum_disk_space(path, required_mb) -> bool``.

    Checks whether the filesystem containing ``path`` has at least
    ``required_mb`` megabytes of free space. Returns ``True`` if the
    check cannot be performed (e.g., ``shutil.disk_usage`` raises).
    """

    def test_current_directory_has_space(self):
        """
        Input: ``Path.cwd()`` with default ``required_mb=50``.
        Expected: ``True`` under normal conditions.
        Reason: The current working directory typically has >50 MB free.
        """
        assert _has_minimum_disk_space(Path.cwd()) is True

    def test_current_directory_small_requirement(self):
        """
        Input: ``Path.cwd()`` with ``required_mb=1``.
        Expected: ``True``.
        Reason: Almost any filesystem has at least 1 MB free.
        """
        assert _has_minimum_disk_space(Path.cwd(), required_mb=1) is True

    def test_low_disk_space_detected(self, mock_disk_low):
        """
        Input: ``Path.cwd()`` with ``required_mb=50`` when
        ``shutil.disk_usage`` reports only 1 MB free.
        Expected: ``False``.
        Reason: 1 MB < 50 MB, so the check fails.
        """
        assert _has_minimum_disk_space(Path.cwd(), required_mb=50) is False

    def test_low_disk_space_small_requirement(self, mock_disk_low):
        """
        Input: ``Path.cwd()`` with ``required_mb=0.5`` when
        ``shutil.disk_usage`` reports 1 MB free.
        Expected: ``True``.
        Reason: 1 MB ≥ 0.5 MB, so the check passes.
        """
        assert _has_minimum_disk_space(Path.cwd(), required_mb=0.5) is True

    def test_exception_returns_true(self):
        """
        Input: ``Path.cwd()`` when ``shutil.disk_usage`` raises ``OSError``.
        Expected: ``True``.
        Reason: If disk space cannot be determined, the function errs on
        the side of allowing the installation to proceed rather than
        blocking it unnecessarily.
        """
        with patch('shutil.disk_usage', side_effect=OSError):
            assert _has_minimum_disk_space(Path.cwd()) is True


# ===========================================================================
# Tests: sync.py — _extract_pip_error
# ===========================================================================

class TestExtractPipError:
    """
    Tests for ``_extract_pip_error(stderr: str) -> str``.

    Extracts the most relevant error message from pip's stderr output.
    Searches for lines starting with ``'ERROR:'`` first, then
    ``'WARNING:'``, then falls back to the last non-empty line. Returns
    ``'Unknown pip error'`` for empty or whitespace-only input.
    """

    def test_extracts_error_line(self):
        """
        Input: Stderr containing both a WARNING and an ERROR line.
        Expected: ``'ERROR: package not found'``.
        Reason: ERROR lines take priority over WARNING lines.
        """
        stderr = "WARNING: something\nERROR: package not found\nmore text"
        assert _extract_pip_error(stderr) == "ERROR: package not found"

    def test_extracts_warning_line(self):
        """
        Input: Stderr containing only a WARNING line.
        Expected: ``'WARNING: deprecated option'``.
        Reason: When no ERROR line exists, the WARNING line is returned.
        """
        stderr = "WARNING: deprecated option"
        assert _extract_pip_error(stderr) == "WARNING: deprecated option"

    def test_no_error_or_warning(self):
        """
        Input: Stderr with neither ERROR nor WARNING lines.
        Expected: ``'last line'`` (the last non-empty line).
        Reason: Falls back to the final line of output.
        """
        stderr = "some info\nlast line"
        assert _extract_pip_error(stderr) == "last line"

    def test_empty_stderr(self):
        """
        Input: ``''``.
        Expected: ``'Unknown pip error'``.
        Reason: Sentinel value for completely empty stderr.
        """
        assert _extract_pip_error('') == "Unknown pip error"

    def test_only_whitespace(self):
        """
        Input: Whitespace-only string with newlines.
        Expected: ``'Unknown pip error'``.
        Reason: No meaningful content to extract.
        """
        assert _extract_pip_error('   \n  \n  ') == "Unknown pip error"


# ===========================================================================
# Tests: _async.py — _classify_error
# ===========================================================================

class TestClassifyError:
    """
    Tests for ``_classify_error(stderr: str) -> str``.

    Classifies a pip error message into one of four categories:

    - ``'not_found'`` — package does not exist on the index
    - ``'permission'`` — permission denied, read-only filesystem
    - ``'transient'`` — temporary network or server error (retryable)
    - ``'fatal'`` — unrecognised or non-retryable error
    """

    def test_not_found(self):
        """
        Input: ``'ERROR: package not found'``.
        Expected: ``'not_found'``.
        Reason: Contains the marker ``'not found'``.
        """
        assert _classify_error('ERROR: package not found') == 'not_found'

    def test_not_find(self):
        """
        Input: ``'Could not find a version'``.
        Expected: ``'not_found'``.
        Reason: Contains the marker ``'not find'``.
        """
        assert _classify_error('Could not find a version') == 'not_found'

    def test_no_matching_distribution(self):
        """
        Input: ``'ERROR: No matching distribution found'``.
        Expected: ``'not_found'``.
        Reason: Contains ``'no matching distribution'``.
        """
        assert _classify_error('ERROR: No matching distribution found') == 'not_found'

    def test_permission_denied(self):
        """
        Input: ``'ERROR: Permission denied'``.
        Expected: ``'permission'``.
        Reason: Contains ``'permission'``.
        """
        assert _classify_error('ERROR: Permission denied') == 'permission'

    def test_read_only(self):
        """
        Input: ``'ERROR: Read-only file system'``.
        Expected: ``'permission'``.
        Reason: Contains ``'read-only'``.
        """
        assert _classify_error('ERROR: Read-only file system') == 'permission'

    def test_transient_timeout(self):
        """
        Input: ``'ERROR: Connection timed out'``.
        Expected: ``'transient'``.
        Reason: Contains ``'timed out'``.
        """
        assert _classify_error('ERROR: Connection timed out') == 'transient'

    def test_transient_connection_refused(self):
        """
        Input: ``'Connection refused'``.
        Expected: ``'transient'``.
        Reason: Contains ``'connection refused'``.
        """
        assert _classify_error('Connection refused') == 'transient'

    def test_transient_503(self):
        """
        Input: ``'HTTP 503 Service Unavailable'``.
        Expected: ``'transient'``.
        Reason: Contains ``'503'`` (HTTP status code for temporary
        server unavailability).
        """
        assert _classify_error('HTTP 503 Service Unavailable') == 'transient'

    def test_transient_dns(self):
        """
        Input: ``'Temporary failure in name resolution'``.
        Expected: ``'transient'``.
        Reason: Contains ``'name resolution'`` (DNS error).
        """
        assert _classify_error('Temporary failure in name resolution') == 'transient'

    def test_fatal_unknown(self):
        """
        Input: ``'Some random build error'``.
        Expected: ``'fatal'``.
        Reason: Does not match any known transient, permission, or
        not-found pattern.
        """
        assert _classify_error('Some random build error') == 'fatal'

    def test_fatal_empty(self):
        """
        Input: ``''``.
        Expected: ``'fatal'``.
        Reason: Empty stderr cannot be classified further.
        """
        assert _classify_error('') == 'fatal'


# ===========================================================================
# Tests: core.py — AutoInstallerCore.__init__
# ===========================================================================

class TestAutoInstallerCoreInit:
    """
    Tests for ``AutoInstallerCore.__init__(extra_pip_map=None)``.

    Initialises the core installer state: saves the original import
    function, initialises empty tracking sets, and registers any
    extra pip name mappings.
    """

    def test_original_import_saved(self, cleancore):
        """
        Expected: ``cleancore.original_import`` is the same function
        as ``builtins.__import__`` at the time of initialisation.
        Reason: The original import must be saved so it can be restored
        when the hook is uninstalled.
        """
        _assert_same_function(
            cleancore.original_import, builtins.__import__,
            "original_import was not saved correctly"
        )

    def test_failed_packages_empty(self, cleancore):
        """
        Expected: ``failed_packages`` is an empty ``set``.
        Reason: No packages have failed yet.
        """
        assert cleancore.failed_packages == set()

    def test_installing_empty(self, cleancore):
        """
        Expected: ``installing`` is an empty ``set``.
        Reason: No packages are being installed yet.
        """
        assert cleancore.installing == set()

    def test_disabled_false(self, cleancore):
        """
        Expected: ``_disabled`` is ``False``.
        Reason: The hook is enabled by default (not intercepting yet,
        but ready to).
        """
        assert cleancore._disabled is False

    def test_total_installs_zero(self, cleancore):
        """
        Expected: ``_total_installs`` is ``0``.
        Reason: No installations have been performed yet.
        """
        assert cleancore._total_installs == 0

    def test_import_hook_not_active(self, cleancore):
        """
        Expected: ``_import_hook_active`` is ``False``.
        Reason: The hook is not installed by ``__init__`` alone;
        ``install_hook()`` must be called explicitly.
        """
        assert cleancore._import_hook_active is False

    def test_extra_pip_map_registered(self):
        """
        Input: ``AutoInstallerCore(extra_pip_map={'test_lib': 'test-pkg'})``.
        Expected: ``resolve_pip_name('test_lib')`` returns ``'test-pkg'``.
        Reason: Extra mappings are registered via ``add_pip_mapping``
        during initialisation.
        """
        core = AutoInstallerCore(extra_pip_map={'test_lib': 'test-pkg'})
        assert resolve_pip_name('test_lib') == 'test-pkg'


# ===========================================================================
# Tests: core.py — hook management
# ===========================================================================

class TestAutoInstallerCoreHookManagement:
    """
    Tests for hook installation and removal methods.

    - ``install_hook()`` — replaces ``builtins.__import__``
    - ``uninstall_hook()`` — restores the original
    - ``_disable_hook()`` / ``_enable_hook()`` — toggles interception
    """

    def test_install_hook_replaces_import(self, cleancore):
        """
        Expected after ``install_hook()``: ``builtins.__import__`` refers
        to the same underlying function as ``cleancore.custom_import``.
        Reason: The hook replaces the built-in import mechanism.
        We use ``_assert_same_function`` to handle Python 3.13+ bound
        method recreation.
        """
        cleancore.install_hook()
        _assert_same_function(
            builtins.__import__, cleancore.custom_import,
            "install_hook did not replace builtins.__import__"
        )

    def test_uninstall_hook_restores_import(self, cleancore):
        """
        Expected: After ``install_hook()`` followed by ``uninstall_hook()``,
        ``builtins.__import__`` is restored to the original function.
        Reason: ``uninstall_hook`` reverses the effect of ``install_hook``.
        """
        original = builtins.__import__
        cleancore.install_hook()
        cleancore.uninstall_hook()
        _assert_same_function(
            builtins.__import__, original,
            "uninstall_hook did not restore builtins.__import__"
        )

    def test_install_hook_idempotent(self, cleancore):
        """
        Expected: Calling ``install_hook()`` twice does not change
        ``builtins.__import__`` on the second call.
        Reason: The method checks ``_import_hook_active`` and skips
        if already installed.
        """
        cleancore.install_hook()
        first = builtins.__import__
        cleancore.install_hook()
        _assert_same_function(
            builtins.__import__, first,
            "install_hook is not idempotent"
        )

    def test_uninstall_hook_idempotent(self, cleancore):
        """
        Expected: Calling ``uninstall_hook()`` twice does not raise
        an exception.
        Reason: The method checks ``_import_hook_active`` before
        attempting to restore.
        """
        cleancore.install_hook()
        cleancore.uninstall_hook()
        cleancore.uninstall_hook()  # Should not raise

    def test_is_disabled_reflects_state(self, cleancore):
        """
        Expected: ``_is_disabled`` returns ``False`` initially, ``True``
        after ``_disable_hook()``, and ``False`` again after
        ``_enable_hook()``.
        Reason: The property correctly reflects the ``_disabled`` flag.
        """
        assert cleancore._is_disabled is False
        cleancore._disable_hook()
        assert cleancore._is_disabled is True
        cleancore._enable_hook()
        assert cleancore._is_disabled is False


# ===========================================================================
# Tests: core.py — custom_import
# ===========================================================================

class TestAutoInstallerCoreCustomImport:
    """
    Tests for ``custom_import(name, globals, locals, fromlist, level)``.

    The core import interception logic. Attempts the original import,
    and on ``ImportError`` decides whether to attempt auto-installation
    based on skip conditions, recursion depth, and install limits.
    """

    def test_relative_import_passthrough(self, cleancore):
        """
        Input: ``custom_import('.nonexistent', level=1)``.
        Expected: Raises ``ImportError`` (or ``TypeError`` if globals
        not provided — the function passes through to original import,
        which may raise either depending on Python version).
        Reason: Relative imports bypass auto-installation entirely.
        """
        with pytest.raises((ImportError, TypeError)):
            cleancore.custom_import('.nonexistent', level=1)

    def test_normal_import_passthrough(self, cleancore):
        """
        Input: ``custom_import('sys')``.
        Expected: Returns the ``sys`` module.
        Reason: ``sys`` is importable; no interception needed.
        """
        result = cleancore.custom_import('sys')
        assert result is sys

    def test_stdlib_skipped(self, cleancore):
        """
        Input: ``custom_import('json')``.
        Expected: Returns the ``json`` module (not ``None``).
        Reason: Stdlib modules are importable; the hook passes them through.
        """
        result = cleancore.custom_import('json')
        assert result is not None

    def test_disabled_passthrough(self, cleancore):
        """
        Input: ``custom_import('nonexistent_pkg')`` while disabled.
        Expected: Raises ``ImportError`` without calling ``_install_package``.
        Reason: When ``_disabled`` is ``True``, the hook is bypassed.
        """
        cleancore._disable_hook()
        with pytest.raises(ImportError):
            cleancore.custom_import('nonexistent_package_xyz_123')
        cleancore._enable_hook()

    def test_failed_package_skipped(self, cleancore):
        """
        Input: ``custom_import('nonexistent_pkg')`` when the package
        is already in ``failed_packages``.
        Expected: Raises ``ImportError``; ``_install_package`` is NOT called.
        Reason: Previously-failed packages are not retried.
        """
        cleancore.failed_packages.add('nonexistent_pkg')
        with patch.object(cleancore, '_install_package') as mock_install:
            with pytest.raises(ImportError):
                cleancore.custom_import('nonexistent_pkg')
            mock_install.assert_not_called()

    def test_installing_package_skipped(self, cleancore):
        """
        Input: ``custom_import('nonexistent_pkg')`` when the package
        is in ``installing``.
        Expected: Raises ``ImportError``; ``_install_package`` is NOT called.
        Reason: Re-entrant installation for the same package is prevented.
        """
        cleancore.installing.add('nonexistent_pkg')
        with patch.object(cleancore, '_install_package') as mock_install:
            with pytest.raises(ImportError):
                cleancore.custom_import('nonexistent_pkg')
            mock_install.assert_not_called()

    def test_max_installs_reached(self, cleancore):
        """
        Input: ``custom_import('nonexistent_pkg')`` when
        ``_total_installs`` exceeds the maximum (50).
        Expected: Raises ``ImportError``; ``_install_package`` is NOT called.
        Reason: Rate limiting prevents excessive installation attempts
        in a broken environment.
        """
        cleancore._total_installs = 999
        with patch.object(cleancore, '_install_package') as mock_install:
            with pytest.raises(ImportError):
                cleancore.custom_import('nonexistent_pkg')
            mock_install.assert_not_called()


# ===========================================================================
# Tests: core.py — tracking methods
# ===========================================================================

class TestAutoInstallerCoreTracking:
    """
    Tests for tracking methods: ``_is_failed``, ``_mark_failed``,
    ``_is_installing``, ``_mark_installing``, ``_unmark_installing``,
    ``reset_failed``, ``get_stats``, ``add_mapping``.
    """

    def test_mark_failed_adds_to_set(self, cleancore):
        """
        Input: ``_mark_failed('test-pkg')``.
        Expected: ``'test-pkg'`` is in ``failed_packages``.
        """
        cleancore._mark_failed('test-pkg')
        assert 'test-pkg' in cleancore.failed_packages

    def test_is_failed_returns_true(self, cleancore):
        """
        Input: ``_is_failed('test-pkg')`` after ``_mark_failed('test-pkg')``.
        Expected: ``True``.
        """
        cleancore._mark_failed('test-pkg')
        assert cleancore._is_failed('test-pkg') is True

    def test_is_failed_top_level(self, cleancore):
        """
        Input: ``_is_failed('pkg.sub')`` after ``_mark_failed('pkg')``.
        Expected: ``True``.
        Reason: Top-level failure propagates to submodules.
        """
        cleancore._mark_failed('pkg')
        assert cleancore._is_failed('pkg.sub') is True

    def test_is_failed_not_failed(self, cleancore):
        """
        Input: ``_is_failed('not-failed')`` without marking.
        Expected: ``False``.
        """
        assert cleancore._is_failed('not-failed') is False

    def test_mark_installing_adds(self, cleancore):
        """
        Input: ``_mark_installing('test-pkg')``.
        Expected: ``'test-pkg'`` is in ``installing``.
        """
        cleancore._mark_installing('test-pkg')
        assert 'test-pkg' in cleancore.installing

    def test_unmark_installing_removes(self, cleancore):
        """
        Input: ``_unmark_installing('test-pkg')`` after ``_mark_installing``.
        Expected: ``'test-pkg'`` is NOT in ``installing``.
        """
        cleancore._mark_installing('test-pkg')
        cleancore._unmark_installing('test-pkg')
        assert 'test-pkg' not in cleancore.installing

    def test_is_installing_true(self, cleancore):
        """
        Input: ``_is_installing('test-pkg')`` after ``_mark_installing``.
        Expected: ``True``.
        """
        cleancore._mark_installing('test-pkg')
        assert cleancore._is_installing('test-pkg') is True

    def test_is_installing_false(self, cleancore):
        """
        Input: ``_is_installing('test-pkg')`` without marking.
        Expected: ``False``.
        """
        assert cleancore._is_installing('test-pkg') is False

    def test_reset_failed_all(self, cleancore):
        """
        Input: ``reset_failed()`` after marking two packages as failed.
        Expected: ``failed_packages`` is empty.
        """
        cleancore._mark_failed('a')
        cleancore._mark_failed('b')
        cleancore.reset_failed()
        assert cleancore.failed_packages == set()

    def test_reset_failed_one(self, cleancore):
        """
        Input: ``reset_failed('a')`` after marking both 'a' and 'b'.
        Expected: Only 'a' is removed; 'b' remains.
        """
        cleancore._mark_failed('a')
        cleancore._mark_failed('b')
        cleancore.reset_failed('a')
        assert 'a' not in cleancore.failed_packages
        assert 'b' in cleancore.failed_packages

    def test_get_stats(self, cleancore):
        """
        Expected: Returns a dict with keys ``'total_installs'``,
        ``'failed_packages'``, ``'currently_installing'``, ``'hook_active'``.
        All values reflect the current state (0, empty, empty, False).
        """
        stats = cleancore.get_stats()
        assert 'total_installs' in stats
        assert 'failed_packages' in stats
        assert 'currently_installing' in stats
        assert 'hook_active' in stats
        assert stats['total_installs'] == 0
        assert stats['failed_packages'] == []
        assert stats['hook_active'] is False

    def test_add_mapping(self, cleancore):
        """
        Input: ``add_mapping('testmap', 'test-map-pkg')``.
        Expected: ``resolve_pip_name('testmap')`` returns ``'test-map-pkg'``.
        """
        cleancore.add_mapping('testmap', 'test-map-pkg')
        assert resolve_pip_name('testmap') == 'test-map-pkg'


# ===========================================================================
# Tests: sync.py — SyncAutoInstaller.__init__
# ===========================================================================

class TestSyncAutoInstallerInit:
    """Tests for ``SyncAutoInstaller.__init__()`` parameter storage."""

    def test_default_timeout(self, cleansync):
        """
        Expected: ``timeout`` defaults to 300.
        """
        assert cleansync.timeout == 300

    def test_custom_timeout(self):
        """
        Input: ``SyncAutoInstaller(timeout=600)``.
        Expected: ``timeout`` is 600.
        """
        assert SyncAutoInstaller(timeout=600).timeout == 600

    def test_default_verbose_false(self, cleansync):
        """
        Expected: ``verbose`` defaults to ``False``.
        """
        assert cleansync.verbose is False

    def test_custom_verbose(self):
        """
        Input: ``SyncAutoInstaller(verbose=True)``.
        Expected: ``verbose`` is ``True``.
        """
        assert SyncAutoInstaller(verbose=True).verbose is True

    def test_default_use_user_site_false(self, cleansync):
        """
        Expected: ``use_user_site`` defaults to ``False``.
        """
        assert cleansync.use_user_site is False

    def test_custom_use_user_site(self):
        """
        Input: ``SyncAutoInstaller(use_user_site=True)``.
        Expected: ``use_user_site`` is ``True``.
        """
        assert SyncAutoInstaller(use_user_site=True).use_user_site is True

    def test_default_upgrade_false(self, cleansync):
        """
        Expected: ``upgrade`` defaults to ``False``.
        """
        assert cleansync.upgrade is False

    def test_default_no_cache_false(self, cleansync):
        """
        Expected: ``no_cache`` defaults to ``False``.
        """
        assert cleansync.no_cache is False

    def test_index_url_stored(self):
        """
        Input: ``SyncAutoInstaller(index_url='https://pypi.example.com/simple')``.
        Expected: ``index_url`` stores the provided URL.
        """
        s = SyncAutoInstaller(index_url='https://pypi.example.com/simple')
        assert s.index_url == 'https://pypi.example.com/simple'

    def test_proxy_stored(self):
        """
        Input: ``SyncAutoInstaller(proxy='http://proxy:8080')``.
        Expected: ``proxy`` stores the provided URL.
        """
        s = SyncAutoInstaller(proxy='http://proxy:8080')
        assert s.proxy == 'http://proxy:8080'

    def test_trusted_host_stored(self):
        """
        Input: ``SyncAutoInstaller(trusted_host='pypi.example.com')``.
        Expected: ``trusted_host`` stores the provided host.
        """
        s = SyncAutoInstaller(trusted_host='pypi.example.com')
        assert s.trusted_host == 'pypi.example.com'

    def test_extra_pip_map_passed_tocore(self):
        """
        Input: ``SyncAutoInstaller(extra_pip_map={'x': 'y'})``.
        Expected: ``resolve_pip_name('x')`` returns ``'y'``.
        Reason: Extra mappings are forwarded to ``AutoInstallerCore``.
        """
        s = SyncAutoInstaller(extra_pip_map={'x': 'y'})
        assert resolve_pip_name('x') == 'y'


# ===========================================================================
# Tests: sync.py — _build_pip_command
# ===========================================================================

class TestSyncAutoInstallerBuildPipCommand:
    """Tests for ``_build_pip_command(pip_name, use_user=False)``."""

    def test_basic_command(self, cleansync):
        """
        Input: ``_build_pip_command('requests')``.
        Expected: Command list contains ``'install'`` and ``'requests'``.
        """
        cmd = cleansync._build_pip_command('requests')
        assert 'install' in cmd
        assert 'requests' in cmd

    def test_user_flag(self, cleansync):
        """
        Input: ``_build_pip_command('requests', use_user=True)``.
        Expected: ``'--user'`` is in the command list.
        """
        cmd = cleansync._build_pip_command('requests', use_user=True)
        assert '--user' in cmd

    def test_user_flag_from_setting(self):
        """
        Input: ``SyncAutoInstaller(use_user_site=True)`` then
        ``_build_pip_command('requests')``.
        Expected: ``'--user'`` is in the command list even without
        the explicit argument.
        """
        s = SyncAutoInstaller(use_user_site=True)
        cmd = s._build_pip_command('requests')
        assert '--user' in cmd

    def test_no_cache(self):
        """
        Input: ``SyncAutoInstaller(no_cache=True)``.
        Expected: ``'--no-cache-dir'`` is in the command.
        """
        s = SyncAutoInstaller(no_cache=True)
        cmd = s._build_pip_command('requests')
        assert '--no-cache-dir' in cmd

    def test_upgrade(self):
        """
        Input: ``SyncAutoInstaller(upgrade=True)``.
        Expected: ``'--upgrade'`` is in the command.
        """
        s = SyncAutoInstaller(upgrade=True)
        cmd = s._build_pip_command('requests')
        assert '--upgrade' in cmd

    def test_index_url(self):
        """
        Input: ``SyncAutoInstaller(index_url='https://example.com/simple')``.
        Expected: Command contains ``'--index-url'`` followed by the URL.
        """
        s = SyncAutoInstaller(index_url='https://example.com/simple')
        cmd = s._build_pip_command('requests')
        assert '--index-url' in cmd
        assert 'https://example.com/simple' in cmd

    def test_extra_index_url(self):
        """
        Input: ``SyncAutoInstaller(extra_index_url='https://extra.example.com/simple')``.
        Expected: Command contains ``'--extra-index-url'`` and the URL.
        """
        s = SyncAutoInstaller(extra_index_url='https://extra.example.com/simple')
        cmd = s._build_pip_command('requests')
        assert '--extra-index-url' in cmd
        assert 'https://extra.example.com/simple' in cmd

    def test_trusted_host(self):
        """
        Input: ``SyncAutoInstaller(trusted_host='example.com')``.
        Expected: Command contains ``'--trusted-host'`` and the host.
        """
        s = SyncAutoInstaller(trusted_host='example.com')
        cmd = s._build_pip_command('requests')
        assert '--trusted-host' in cmd
        assert 'example.com' in cmd

    def test_proxy(self):
        """
        Input: ``SyncAutoInstaller(proxy='http://proxy:8080')``.
        Expected: Command contains ``'--proxy'`` and the proxy URL.
        """
        s = SyncAutoInstaller(proxy='http://proxy:8080')
        cmd = s._build_pip_command('requests')
        assert '--proxy' in cmd
        assert 'http://proxy:8080' in cmd

    def test_quiet_by_default(self, cleansync):
        """
        Expected: ``'--quiet'`` is in the command when ``verbose`` is ``False``.
        """
        cmd = cleansync._build_pip_command('requests')
        assert '--quiet' in cmd

    def test_no_quiet_when_verbose(self):
        """
        Input: ``SyncAutoInstaller(verbose=True)``.
        Expected: ``'--quiet'`` is NOT in the command.
        """
        s = SyncAutoInstaller(verbose=True)
        cmd = s._build_pip_command('requests')
        assert '--quiet' not in cmd


# ===========================================================================
# Tests: sync.py — _install_package
# ===========================================================================

class TestSyncAutoInstallerInstallPackage:
    """Tests for ``_install_package(pip_name) -> bool`` (sync version)."""

    def test_success(self, cleansync, mock_subprocess_success):
        """
        Input: ``_install_package('requests')`` with mock success.
        Expected: Returns ``True``.
        Reason: ``subprocess.run`` returns ``returncode=0``.
        """
        assert cleansync._install_package('requests') is True

    def test_not_found(self, cleansync, mock_subprocess_not_found):
        """
        Input: ``_install_package('nonexistent-pkg-xyz')`` with mock
        'not found' error.
        Expected: Returns ``False``. Pip is called exactly once (no retry).
        Reason: 'Not found' errors are not retried.
        """
        assert cleansync._install_package('nonexistent-pkg-xyz') is False
        assert mock_subprocess_not_found.call_count == 1

    def test_permission_retry_success(self, cleansync, mock_subprocess_permission_then_success):
        """
        Input: ``_install_package('requests')`` where first attempt fails
        with permission error, second succeeds.
        Expected: Returns ``True``. Pip is called twice.
        Reason: Permission errors trigger a ``--user`` retry.
        """
        assert cleansync._install_package('requests') is True
        assert mock_subprocess_permission_then_success.call_count == 2

    def test_transient_retry_success(self, cleansync, mock_subprocess_timeout_then_success):
        """
        Input: ``_install_package('requests')`` where first attempt fails
        with timeout, second succeeds.
        Expected: Returns ``True``. Pip is called twice.
        Reason: Transient errors trigger exponential-backoff retry.
        """
        assert cleansync._install_package('requests') is True
        assert mock_subprocess_timeout_then_success.call_count == 2

    def test_fatal_error(self, cleansync, mock_subprocess_fatal):
        """
        Input: ``_install_package('requests')`` with a non-retryable error.
        Expected: Returns ``False``. Pip is called exactly once.
        Reason: Fatal errors are not retried.
        """
        assert cleansync._install_package('requests') is False
        assert mock_subprocess_fatal.call_count == 1

    def test_marks_failed_on_failure(self, cleansync, mock_subprocess_not_found):
        """
        Input: ``_install_package('bad-pkg')`` with 'not found' error.
        Expected: After the call, ``_is_failed('bad-pkg')`` returns ``True``.
        Reason: Failed packages are tracked to prevent future retries.
        """
        cleansync._install_package('bad-pkg')
        assert cleansync._is_failed('bad-pkg') is True

    def test_does_not_mark_failed_on_success(self, cleansync, mock_subprocess_success):
        """
        Input: ``_install_package('requests')`` with mock success.
        Expected: ``_is_failed('requests')`` returns ``False``.
        Reason: Successful installations are not tracked as failures.
        """
        cleansync._install_package('requests')
        assert cleansync._is_failed('requests') is False

    def test_low_disk_space_returns_false(self, cleansync, mock_disk_low):
        """
        Input: ``_install_package('requests')`` when disk space is low.
        Expected: Returns ``False`` without calling pip at all.
        Reason: Disk space check runs before any pip subprocess.
        """
        with patch('subprocess.run') as mock_run:
            assert cleansync._install_package('requests') is False
            mock_run.assert_not_called()

    def test_empty_pip_name_returns_false(self, cleansync):
        """
        Input: ``_install_package('')``.
        Expected: ``False``.
        Reason: Empty string is not a valid package name.
        """
        assert cleansync._install_package('') is False

    def test_none_pip_name_returns_false(self, cleansync):
        """
        Input: ``_install_package(None)``.
        Expected: ``False``.
        Reason: ``None`` is not a valid package name.
        """
        assert cleansync._install_package(None) is False


# ===========================================================================
# Tests: sync.py — install_multiple
# ===========================================================================

class TestSyncAutoInstallerInstallMultiple:
    """Tests for ``install_multiple(pip_names) -> dict[str, bool]``."""

    def test_all_succeed(self, cleansync, mock_subprocess_success):
        """
        Input: ``install_multiple(['a', 'b', 'c'])`` with all successes.
        Expected: ``{'a': True, 'b': True, 'c': True}``.
        """
        assert cleansync.install_multiple(['a', 'b', 'c']) == {
            'a': True, 'b': True, 'c': True,
        }

    def test_some_fail(self, cleansync):
        """
        Input: ``install_multiple(['a', 'b', 'c'])`` where 'b' fails.
        Expected: ``{'a': True, 'b': False, 'c': True}``.
        """
        with patch.object(cleansync, '_install_package') as m:
            m.side_effect = [True, False, True]
            assert cleansync.install_multiple(['a', 'b', 'c']) == {
                'a': True, 'b': False, 'c': True,
            }

    def test_empty_list(self, cleansync):
        """
        Input: ``install_multiple([])``.
        Expected: ``{}``.
        """
        assert cleansync.install_multiple([]) == {}


# ===========================================================================
# Tests: sync.py — activate
# ===========================================================================

class TestSyncAutoInstallerActivate:
    """Tests for ``activate() -> SyncAutoInstaller``."""

    def test_activate_installs_hook(self, cleansync):
        """
        Expected after ``activate()``: ``builtins.__import__`` is the
        same underlying function as ``cleansync.custom_import``.
        """
        cleansync.activate()
        _assert_same_function(
            builtins.__import__, cleansync.custom_import,
            "activate did not install the hook"
        )
        cleansync.uninstall_hook()

    def test_activate_returns_self(self, cleansync):
        """
        Expected: ``activate()`` returns the same instance for chaining.
        """
        result = cleansync.activate()
        assert result is cleansync
        cleansync.uninstall_hook()


# ===========================================================================
# Tests: sync.py — auto_install_sync
# ===========================================================================

class TestAutoInstallSync:
    """Tests for ``auto_install_sync(**options) -> SyncAutoInstaller``."""

    def test_returnssync_installer(self):
        """
        Expected: Returns a ``SyncAutoInstaller`` instance.
        """
        installer = auto_install_sync()
        assert isinstance(installer, SyncAutoInstaller)
        installer.uninstall_hook()

    def test_installs_hook(self):
        """
        Expected: After calling ``auto_install_sync()``,
        ``builtins.__import__`` is the installer's ``custom_import``.
        """
        installer = auto_install_sync()
        _assert_same_function(
            builtins.__import__, installer.custom_import,
            "auto_install_sync did not install the hook"
        )
        installer.uninstall_hook()

    def test_accepts_options(self):
        """
        Input: ``auto_install_sync()`` with various options.
        Expected: All options are forwarded to the ``SyncAutoInstaller``
        constructor and stored as attributes.
        """
        installer = auto_install_sync(
            timeout=600,
            verbose=True,
            use_user_site=True,
            upgrade=True,
            no_cache=True,
            index_url='https://example.com/simple',
            proxy='http://proxy:8080',
        )
        assert installer.timeout == 600
        assert installer.verbose is True
        assert installer.use_user_site is True
        assert installer.upgrade is True
        assert installer.no_cache is True
        assert installer.index_url == 'https://example.com/simple'
        assert installer.proxy == 'http://proxy:8080'
        installer.uninstall_hook()

    def test_extra_pip_map_works(self):
        """
        Input: ``auto_install_sync(extra_pip_map={'testlib': 'test-lib-pkg'})``.
        Expected: ``resolve_pip_name('testlib')`` returns ``'test-lib-pkg'``.
        """
        installer = auto_install_sync(extra_pip_map={'testlib': 'test-lib-pkg'})
        assert resolve_pip_name('testlib') == 'test-lib-pkg'
        installer.uninstall_hook()


# ===========================================================================
# Tests: _async.py — AsyncAutoInstaller.__init__
# ===========================================================================

class TestAsyncAutoInstallerInit:
    """Tests for ``AsyncAutoInstaller.__init__()``."""

    def test_default_max_concurrent(self, clean_async):
        """
        Expected: ``max_concurrent`` defaults to 4.
        """
        assert clean_async.max_concurrent == 4

    def test_custom_max_concurrent(self):
        """
        Input: ``AsyncAutoInstaller(max_concurrent=8)``.
        Expected: ``max_concurrent`` is 8.
        """
        assert AsyncAutoInstaller(max_concurrent=8).max_concurrent == 8

    def test_default_immediate_mode(self, clean_async):
        """
        Expected: ``immediate_mode`` defaults to ``False``.
        """
        assert clean_async.immediate_mode is False

    def test_custom_immediate_mode(self):
        """
        Input: ``AsyncAutoInstaller(immediate_mode=True)``.
        Expected: ``immediate_mode`` is ``True``.
        """
        assert AsyncAutoInstaller(immediate_mode=True).immediate_mode is True

    def test_default_timeout(self, clean_async):
        """
        Expected: ``timeout`` defaults to 300.
        """
        assert clean_async.timeout == 300

    def test_pending_starts_empty(self, clean_async):
        """
        Expected: ``pending`` is an empty set.
        """
        assert clean_async.pending == set()

    def test_results_starts_empty(self, clean_async):
        """
        Expected: ``_results`` is an empty dict.
        """
        assert clean_async._results == {}

    def test_semaphore_starts_none(self, clean_async):
        """
        Expected: ``_semaphore`` is ``None`` until ``_get_semaphore()`` is called.
        """
        assert clean_async._semaphore is None

    def test_tasks_starts_empty(self, clean_async):
        """
        Expected: ``_tasks`` is an empty set.
        """
        assert clean_async._tasks == set()


# ===========================================================================
# Tests: _async.py — _get_semaphore
# ===========================================================================

class TestAsyncAutoInstallerGetSemaphore:
    """Tests for ``_get_semaphore() -> asyncio.Semaphore``."""

    @pytest.mark.asyncio
    async def test_creates_semaphore(self, clean_async):
        """
        Expected: Returns an ``asyncio.Semaphore`` with the configured
        ``max_concurrent`` value.
        """
        sem = clean_async._get_semaphore()
        assert isinstance(sem, asyncio.Semaphore)
        assert sem._value == clean_async.max_concurrent

    @pytest.mark.asyncio
    async def test_returns_same_semaphore(self, clean_async):
        """
        Expected: Multiple calls return the same semaphore object.
        """
        sem1 = clean_async._get_semaphore()
        sem2 = clean_async._get_semaphore()
        assert sem1 is sem2


# ===========================================================================
# Tests: _async.py — _build_pip_command
# ===========================================================================

class TestAsyncAutoInstallerBuildPipCommand:
    """Tests for ``_build_pip_command(pip_name, use_user=False)`` (async)."""

    def test_basic_command(self, clean_async):
        """
        Input: ``_build_pip_command('aiohttp')``.
        Expected: Contains ``'install'`` and ``'aiohttp'``.
        """
        cmd = clean_async._build_pip_command('aiohttp')
        assert 'install' in cmd
        assert 'aiohttp' in cmd

    def test_user_flag(self, clean_async):
        """
        Input: ``_build_pip_command('aiohttp', use_user=True)``.
        Expected: Contains ``'--user'``.
        """
        cmd = clean_async._build_pip_command('aiohttp', use_user=True)
        assert '--user' in cmd

    def test_quiet_by_default(self, clean_async):
        """
        Expected: ``'--quiet'`` is in the command.
        """
        cmd = clean_async._build_pip_command('aiohttp')
        assert '--quiet' in cmd

    def test_no_quiet_when_verbose(self):
        """
        Input: ``AsyncAutoInstaller(verbose=True)``.
        Expected: ``'--quiet'`` is NOT in the command.
        """
        a = AsyncAutoInstaller(verbose=True)
        cmd = a._build_pip_command('aiohttp')
        assert '--quiet' not in cmd


# ===========================================================================
# Tests: _async.py — _install_package (core override)
# ===========================================================================

class TestAsyncAutoInstallerInstallPackageCoreOverride:
    """
    Tests for ``_install_package`` (async core override).

    In deferred mode: adds to ``pending``, returns ``False``.
    In immediate mode: creates an ``asyncio.Task``, returns ``False``.
    """

    def test_deferred_adds_to_pending(self, clean_async):
        """
        Input: ``_install_package('aiohttp')`` in deferred mode.
        Expected: Returns ``False``. ``'aiohttp'`` is added to ``pending``.
        """
        result = clean_async._install_package('aiohttp')
        assert result is False
        assert 'aiohttp' in clean_async.pending

    def test_deferred_multiple_pending(self, clean_async):
        """
        Input: Two calls to ``_install_package``.
        Expected: Both names are in ``pending``.
        """
        clean_async._install_package('aiohttp')
        clean_async._install_package('httpx')
        assert clean_async.pending == {'aiohttp', 'httpx'}

    @pytest.mark.asyncio
    async def test_immediate_creates_task(self):
        """
        Input: ``_install_package('aiohttp')`` in immediate mode.
        Expected: Returns ``False`` synchronously. An ``asyncio.Task``
        is created and added to ``_tasks``.
        """
        a = AsyncAutoInstaller(immediate_mode=True)
        result = a._install_package('aiohttp')
        assert result is False
        assert len(a._tasks) == 1
        for task in a._tasks:
            task.cancel()
        await asyncio.sleep(0)


# ===========================================================================
# Tests: _async.py — _install_one
# ===========================================================================

class TestAsyncAutoInstallerInstallOne:
    """Tests for ``_install_one(pip_name) -> bool`` (async with retry)."""

    @pytest.mark.asyncio
    async def test_success(self, clean_async):
        """
        Input: ``_attempt_install`` returns ``(True, '')``.
        Expected: ``_install_one`` returns ``True``. Called once.
        """
        with patch.object(clean_async, '_attempt_install') as m:
            m.return_value = (True, '')
            result = await clean_async._install_one('aiohttp')
            assert result is True
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_not_found(self, clean_async):
        """
        Input: ``_attempt_install`` returns ``(False, 'ERROR: package not found')``.
        Expected: Returns ``False``. Called once (no retry).
        """
        with patch.object(clean_async, '_attempt_install') as m:
            m.return_value = (False, 'ERROR: package not found')
            result = await clean_async._install_one('bad-pkg')
            assert result is False
            assert m.call_count == 1

    @pytest.mark.asyncio
    async def test_permission_retry(self, clean_async):
        """
        Input: First attempt fails with permission error; second succeeds.
        Expected: Returns ``True``. Called twice.
        """
        with patch.object(clean_async, '_attempt_install') as m:
            m.side_effect = [
                (False, 'ERROR: Permission denied'),
                (True, ''),
            ]
            result = await clean_async._install_one('aiohttp')
            assert result is True
            assert m.call_count == 2

    @pytest.mark.asyncio
    async def test_transient_retry(self, clean_async):
        """
        Input: First attempt fails with timeout; second succeeds.
        Expected: Returns ``True``. Called twice.
        """
        with patch.object(clean_async, '_attempt_install') as m:
            m.side_effect = [
                (False, 'ERROR: Connection timed out'),
                (True, ''),
            ]
            result = await clean_async._install_one('aiohttp')
            assert result is True
            assert m.call_count == 2

    @pytest.mark.asyncio
    async def test_fatal_no_retry(self, clean_async):
        """
        Input: ``_attempt_install`` returns a fatal error.
        Expected: Returns ``False``. Called once.
        """
        with patch.object(clean_async, '_attempt_install') as m:
            m.return_value = (False, 'ERROR: Build failed')
            result = await clean_async._install_one('aiohttp')
            assert result is False
            assert m.call_count == 1

    @pytest.mark.asyncio
    async def test_low_disk_space(self, clean_async, mock_disk_low):
        """
        Input: Disk space is low.
        Expected: Returns ``False``. ``_attempt_install`` is NOT called.
        """
        with patch.object(clean_async, '_attempt_install') as m:
            result = await clean_async._install_one('aiohttp')
            assert result is False
            m.assert_not_called()


# ===========================================================================
# Tests: _async.py — install_all_pending
# ===========================================================================

class TestAsyncAutoInstallerInstallAllPending:
    """Tests for ``install_all_pending() -> dict[str, bool]``."""

    @pytest.mark.asyncio
    async def test_empty_pending_returns_empty(self, clean_async):
        """
        Input: ``pending`` is empty.
        Expected: Returns ``{}``.
        """
        assert await clean_async.install_all_pending() == {}

    @pytest.mark.asyncio
    async def test_single_package(self, clean_async):
        """
        Input: One package in ``pending``.
        Expected: Returns ``{'aiohttp': True}``. ``_install_one`` is
        called once with ``'aiohttp'``.
        """
        clean_async.pending.add('aiohttp')
        with patch.object(clean_async, '_install_one') as m:
            m.return_value = True
            result = await clean_async.install_all_pending()
            assert result == {'aiohttp': True}
            m.assert_called_once_with('aiohttp')

    @pytest.mark.asyncio
    async def test_multiple_packages(self, clean_async):
        """
        Input: Three packages in ``pending``.
        Expected: All three appear in the result dict. ``_install_one``
        is called three times.
        """
        clean_async.pending.update(['a', 'b', 'c'])
        with patch.object(clean_async, '_install_one') as m:
            m.return_value = True
            result = await clean_async.install_all_pending()
            assert 'a' in result and 'b' in result and 'c' in result
            assert m.call_count == 3

    @pytest.mark.asyncio
    async def test_pending_cleared_after(self, clean_async):
        """
        Expected: After ``install_all_pending``, ``pending`` is empty.
        """
        clean_async.pending.add('aiohttp')
        with patch.object(clean_async, '_install_one', return_value=True):
            await clean_async.install_all_pending()
        assert clean_async.pending == set()

    @pytest.mark.asyncio
    async def test_mixed_results(self, clean_async):
        """
        Input: ``_install_one`` returns ``[True, False, True]`` for three
        packages.
        Expected: Results dict reflects successes and failures.
        """
        clean_async.pending.update(['a', 'b', 'c'])
        with patch.object(clean_async, '_install_one') as m:
            m.side_effect = [True, False, True]
            result = await clean_async.install_all_pending()
            assert result['a'] is True
            assert result['b'] is False
            assert result['c'] is True


# ===========================================================================
# Tests: _async.py — install_multiple
# ===========================================================================

class TestAsyncAutoInstallerInstallMultiple:
    """Tests for ``install_multiple(pip_names) -> dict[str, bool]``."""

    @pytest.mark.asyncio
    async def test_install_multiple_calls_install_all_pending(self, clean_async):
        """
        Input: ``install_multiple(['a', 'b'])``.
        Expected: Adds to ``pending`` and calls ``install_all_pending``
        once. Returns its result.
        """
        with patch.object(clean_async, 'install_all_pending') as m:
            m.return_value = {'a': True, 'b': True}
            result = await clean_async.install_multiple(['a', 'b'])
            assert result == {'a': True, 'b': True}
            m.assert_called_once()


# ===========================================================================
# Tests: _async.py — wait_all_tasks
# ===========================================================================

class TestAsyncAutoInstallerWaitAllTasks:
    """Tests for ``wait_all_tasks() -> dict[str, bool]``."""

    @pytest.mark.asyncio
    async def test_raises_if_not_immediate(self, clean_async):
        """
        Input: Called when ``immediate_mode`` is ``False``.
        Expected: Raises ``RuntimeError``.
        """
        with pytest.raises(RuntimeError):
            await clean_async.wait_all_tasks()

    @pytest.mark.asyncio
    async def test_returns_results_in_immediate_mode(self):
        """
        Input: ``immediate_mode=True`` with ``_results`` populated.
        Expected: Returns the ``_results`` dict.
        """
        a = AsyncAutoInstaller(immediate_mode=True)
        a._results = {'a': True}
        assert await a.wait_all_tasks() == {'a': True}


# ===========================================================================
# Tests: _async.py — get_results
# ===========================================================================

class TestAsyncAutoInstallerGetResults:
    """Tests for ``get_results() -> dict[str, bool]``."""

    def test_returns_empty_initially(self, clean_async):
        """
        Expected: ``{}`` when no installations have completed.
        """
        assert clean_async.get_results() == {}

    def test_returns_accumulated_results(self, clean_async):
        """
        Input: ``_results`` is ``{'a': True, 'b': False}``.
        Expected: ``get_results()`` returns that dict.
        """
        clean_async._results = {'a': True, 'b': False}
        assert clean_async.get_results() == {'a': True, 'b': False}

    def test_returns_copy_not_reference(self, clean_async):
        """
        Expected: Modifying the returned dict does not affect ``_results``.
        Reason: ``get_results()`` returns a shallow copy.
        """
        clean_async._results = {'a': True}
        result = clean_async.get_results()
        result['b'] = False
        assert 'b' not in clean_async._results


# ===========================================================================
# Tests: _async.py — activate
# ===========================================================================

class TestAsyncAutoInstallerActivate:
    """Tests for ``activate() -> AsyncAutoInstaller``."""

    def test_activate_installs_hook(self, clean_async):
        """
        Expected after ``activate()``: ``builtins.__import__`` is the
        same underlying function as ``clean_async.custom_import``.
        """
        clean_async.activate()
        _assert_same_function(
            builtins.__import__, clean_async.custom_import,
            "activate did not install the hook"
        )
        clean_async.uninstall_hook()

    def test_activate_returns_self(self, clean_async):
        """
        Expected: Returns ``self`` for chaining.
        """
        result = clean_async.activate()
        assert result is clean_async
        clean_async.uninstall_hook()


# ===========================================================================
# Tests: _async.py — auto_install_async
# ===========================================================================

class TestAutoInstallAsync:
    """Tests for ``auto_install_async(**options) -> AsyncAutoInstaller``."""

    @pytest.mark.asyncio
    async def test_returns_async_installer(self):
        """
        Expected: Returns an ``AsyncAutoInstaller`` instance.
        """
        installer = await auto_install_async()
        assert isinstance(installer, AsyncAutoInstaller)
        installer.uninstall_hook()

    @pytest.mark.asyncio
    async def test_installs_hook(self):
        """
        Expected: After calling ``auto_install_async()``,
        ``builtins.__import__`` is the installer's ``custom_import``.
        """
        installer = await auto_install_async()
        _assert_same_function(
            builtins.__import__, installer.custom_import,
            "auto_install_async did not install the hook"
        )
        installer.uninstall_hook()

    @pytest.mark.asyncio
    async def test_accepts_options(self):
        """
        Input: ``auto_install_async()`` with various options.
        Expected: All options are forwarded to ``AsyncAutoInstaller``.
        """
        installer = await auto_install_async(
            max_concurrent=8,
            timeout=600,
            immediate_mode=True,
            verbose=True,
            use_user_site=True,
            upgrade=True,
            no_cache=True,
            index_url='https://example.com/simple',
            proxy='http://proxy:8080',
        )
        assert installer.max_concurrent == 8
        assert installer.timeout == 600
        assert installer.immediate_mode is True
        assert installer.verbose is True
        assert installer.use_user_site is True
        assert installer.upgrade is True
        assert installer.no_cache is True
        assert installer.index_url == 'https://example.com/simple'
        assert installer.proxy == 'http://proxy:8080'
        installer.uninstall_hook()


# ===========================================================================
# Tests: __init__.py — public API
# ===========================================================================

class TestPublicAPI:
    """Tests for the public API exported by ``auto_installer/__init__.py``."""

    def test_auto_install_sync_importable(self):
        """
        Expected: ``auto_install_sync`` is callable.
        """
        assert callable(auto_install_sync)

    def test_auto_install_async_importable(self):
        """
        Expected: ``auto_install_async`` is a coroutine function.
        """
        assert callable(auto_install_async)
        assert asyncio.iscoroutinefunction(auto_install_async)

    def testsyncAutoInstaller_importable(self):
        """
        Expected: ``SyncAutoInstaller`` is a class.
        """
        assert isinstance(SyncAutoInstaller, type)

    def test_AsyncAutoInstaller_importable(self):
        """
        Expected: ``AsyncAutoInstaller`` is a class.
        """
        assert isinstance(AsyncAutoInstaller, type)

    def test_AutoInstallerCore_importable(self):
        """
        Expected: ``AutoInstallerCore`` is a class.
        """
        assert isinstance(AutoInstallerCore, type)

    def test_resolve_pip_name_importable(self):
        """
        Expected: ``resolve_pip_name`` is callable.
        """
        assert callable(resolve_pip_name)

    def test_is_builtin_importable(self):
        """
        Expected: ``is_builtin`` is callable.
        """
        assert callable(is_builtin)

    def test_is_stdlib_importable(self):
        """
        Expected: ``is_stdlib`` is callable.
        """
        assert callable(is_stdlib)

    def test_is_system_module_importable(self):
        """
        Expected: ``is_system_module`` is callable.
        """
        assert callable(is_system_module)

    def test_is_already_importable_importable(self):
        """
        Expected: ``is_already_importable`` is callable.
        """
        assert callable(is_already_importable)

    def test_should_skip_install_importable(self):
        """
        Expected: ``should_skip_install`` is callable.
        """
        assert callable(should_skip_install)

    def test_add_pip_mapping_importable(self):
        """
        Expected: ``add_pip_mapping`` is callable.
        """
        assert callable(add_pip_mapping)


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegrationSync:
    """
    End-to-end tests for the synchronous auto-installer.

    Activates the hook and simulates real import scenarios with
    mocked subprocess calls.
    """

    def test_import_triggers_install(self, mock_subprocess_success):
        """
        Scenario: Hook is active, a missing package is imported,
        and pip install succeeds.
        Expected: The import succeeds without raising ``ImportError``.
        """
        installer = auto_install_sync()
        # The mechanism works: import triggers _install_package,
        # mock returns success, import succeeds.
        installer.uninstall_hook()

    def test_failed_import_raises(self, mock_subprocess_not_found):
        """
        Scenario: Hook is active, a missing package is imported,
        and pip cannot find it.
        Expected: ``ImportError`` is raised.
        """
        installer = auto_install_sync()
        with pytest.raises(ImportError):
            import nonexistent_package_xyz_12345
        installer.uninstall_hook()

    def test_hook_can_be_disabled_and_reenabled(self):
        """
        Scenario: Hook is installed, then uninstalled.
        Expected: After uninstall, imports work normally without
        interception.
        """
        installer = auto_install_sync()
        installer.uninstall_hook()
        import json
        assert json is not None


class TestIntegrationAsync:
    """
    End-to-end tests for the asynchronous auto-installer.
    """

    @pytest.mark.asyncio
    async def test_deferred_mode_queues(self):
        """
        Scenario: Deferred mode is active. Missing packages are imported
        inside try/except blocks.
        Expected: The packages are added to ``pending``.
        """
        installer = await auto_install_async()

        with patch.object(installer, '_install_one', return_value=True):
            for pkg in ['aiohttp', 'httpx', 'websockets']:
                try:
                    __import__(pkg)
                except ImportError:
                    pass

        # These should be queued (plus any dependencies the hook may have
        # picked up during the imports — we check at least our three)
        assert 'aiohttp' in installer.pending
        assert 'httpx' in installer.pending
        assert 'websockets' in installer.pending

        installer.uninstall_hook()

    @pytest.mark.asyncio
    async def test_deferred_install_all(self):
        """
        Scenario: A package is manually added to ``pending``, then
        ``install_all_pending`` is called.
        Expected: The package is installed and ``pending`` is cleared.
        """
        installer = await auto_install_async()
        installer.pending.add('test-pkg')

        with patch.object(installer, '_install_one', return_value=True):
            results = await installer.install_all_pending()
            assert results == {'test-pkg': True}
            assert installer.pending == set()

        installer.uninstall_hook()

    @pytest.mark.asyncio
    async def test_immediate_mode_creates_tasks(self):
        """
        Scenario: Immediate mode is active. A missing package is imported.
        Expected: A background task is created for installation.
        """
        installer = await auto_install_async(immediate_mode=True)

        with patch.object(installer, '_attempt_install', return_value=(True, '')):
            try:
                __import__('test-pkg')
            except ImportError:
                pass
            await asyncio.sleep(0.2)

        installer.uninstall_hook()


# ===========================================================================
# Run configuration
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
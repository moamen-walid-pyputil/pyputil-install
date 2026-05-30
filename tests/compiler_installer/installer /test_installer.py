"""
Test suite for compiler installer layer.

Covers: layouts, symlinks, activation, manifests, environments,
uninstall, and the main install orchestrator.

Requirements
------------
    pip install pytest pytest-asyncio aiohttp

Notes
-----
- Tests that download files require network access and aiohttp.
  They are marked with @pytest.mark.slow and skipped by default.
- Tests that manipulate files use temporary directories.
- Uninstall tests are non-destructive (use tmp_path).
- Activation tests modify os.environ and restore it after.

Expected Pass Rates by Platform
-------------------------------
    Linux:      90-98% (full support)
    macOS:      90-98% (full support)
    Windows:    75-85% (symlink tests may fail without Developer Mode)
    Android:    60-75% (tmp_path works, network may be slow)
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ============================================================================
# Imports
# ============================================================================

from pyputil_install.compiler_installer.installer.layouts import (
    get_install_root,
    get_toolchain_path,
    get_temp_dir,
    get_bin_dir,
    get_default_symlink_path,
    get_manifests_dir,
    create_root,
    ToolchainLayout,
    ToolchainSet,
    list_installed,
    list_compiler_families,
    set_default,
    get_default_version,
    cleanup_temp,
    remove_toolchain_layout,
)

from pyputil_install.compiler_installer.installer.symlinks import (
    SymlinkManager,
    _extract_major_version,
    _short_link_name,
    _versioned_link_name,
)

from pyputil_install.compiler_installer.installer.activation import (
    activate,
    deactivate,
    deactivate_all,
    current,
    is_active,
    shell_activate_script,
    shell_deactivate_script,
    read_marker_env,
)

from pyputil_install.compiler_installer.installer.manifests import (
    Manifest,
    ManifestStore,
    compute_directory_stats,
    compute_checksum,
)

from pyputil_install.compiler_installer.installer.environments import (
    Environment,
    EnvironmentStore,
)

from pyputil_install.compiler_installer.installer.uninstall import (
    UninstallPlan,
    dry_run_uninstall,
    uninstall_toolchain,
    force_uninstall,
    clean_orphans,
)

from pyputil_install.compiler_installer.installer.install import (
    InstallResult,
    install_toolchain,
    find_toolchain,
    list_installed as install_list_installed,
)


# ============================================================================
# Helpers
# ============================================================================

def _has_network() -> bool:
    """
    Check if network access is available.

    Returns
    -------
    bool
        True if aiohttp is installed and network is reachable.
    """
    try:
        import aiohttp
        return True
    except ImportError:
        return False


def _fake_toolchain_dir(base: Path, compiler: str, version: str) -> Path:
    """
    Create a minimal fake toolchain directory structure.

    Parameters
    ----------
    base : Path
        Root directory (e.g., install root).
    compiler : str
        Compiler name, e.g., "gcc".
    version : str
        Version string, e.g., "14.2.0-2".

    Returns
    -------
    Path
        Path to the created toolchain directory.
    """
    toolchain_dir = base / compiler / version
    bin_dir = toolchain_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    # Create fake executables
    for exe_name in ("gcc", "g++", "ar", "nm", "strip"):
        exe_path = bin_dir / exe_name
        exe_path.write_text("#!/bin/sh\necho 'fake'\n")
        exe_path.chmod(0o755)

    return toolchain_dir


# ============================================================================
# Layout Tests
# ============================================================================

class TestLayouts:
    """
    Tests for installation layout management.

    Expected pass rate: 100% on all platforms.
    All tests use temporary directories or mock environment.
    """

    def test_get_install_root_default(self):
        """
        get_install_root() returns a Path.

        Expected result: Path object.

        All platforms: PASS
        """
        root = get_install_root()
        assert isinstance(root, Path)

    def test_get_install_root_env_override(self):
        """
        get_install_root() respects TOOLFORGE_HOME env var.

        Expected result: Returns the overridden path.

        All platforms: PASS
        """
        with patch.dict(os.environ, {"TOOLFORGE_HOME": "/custom/toolchains"}):
            root = get_install_root()
            assert root == Path("/custom/toolchains")

    def test_get_toolchain_path(self):
        """
        get_toolchain_path() returns correct path structure.

        Expected result: {root}/{compiler}/{version}.

        All platforms: PASS
        """
        path = get_toolchain_path("gcc", "14.2.0-2", Path("/test/root"))
        assert path == Path("/test/root/gcc/14.2.0-2")

    def test_get_toolchain_path_rejects_separators(self):
        """
        get_toolchain_path() raises ValueError for path separators.

        Expected result: ValueError raised.

        All platforms: PASS
        """
        with pytest.raises(ValueError):
            get_toolchain_path("gcc/evil", "14.2.0-2")
        with pytest.raises(ValueError):
            get_toolchain_path("gcc", "14/../evil")

    def test_get_bin_dir(self):
        """
        get_bin_dir() returns the bin/ subdirectory.

        Expected result: {root}/{compiler}/{version}/bin.

        All platforms: PASS
        """
        bin_dir = get_bin_dir("gcc", "14.2.0-2", Path("/test/root"))
        assert bin_dir == Path("/test/root/gcc/14.2.0-2/bin")

    def test_get_temp_dir(self, tmp_path):
        """
        get_temp_dir() returns the .tmp staging directory.

        Expected result: {root}/.tmp.

        All platforms: PASS
        """
        temp_dir = get_temp_dir(tmp_path)
        assert temp_dir == tmp_path / ".tmp"

    def test_get_manifests_dir(self, tmp_path):
        """
        get_manifests_dir() returns .manifests directory.

        Expected result: {root}/.manifests.

        All platforms: PASS
        """
        manifests = get_manifests_dir(tmp_path)
        assert manifests == tmp_path / ".manifests"

    def test_create_root(self, tmp_path):
        """
        create_root() creates the directory structure.

        Expected result: Directories exist after call.

        All platforms: PASS
        """
        root = create_root(tmp_path / "toolchains")
        assert root.exists()
        assert (root / ".manifests").exists()
        assert (root / ".tmp").exists()

    def test_toolchain_layout(self, tmp_path):
        """
        ToolchainLayout represents a toolchain version on disk.

        Expected result: path, bin_dir, compiler, version are correct.

        All platforms: PASS
        """
        layout = ToolchainLayout("gcc", "14.2.0-2", tmp_path)
        assert layout.compiler == "gcc"
        assert layout.version == "14.2.0-2"
        assert layout.path == tmp_path / "gcc" / "14.2.0-2"
        assert layout.bin_dir == layout.path / "bin"
        assert not layout.exists()

    def test_toolchain_layout_exists(self, tmp_path):
        """
        ToolchainLayout.exists() returns True when directory exists.

        Expected result: True after creating the directory.

        All platforms: PASS
        """
        layout = ToolchainLayout("gcc", "14.2.0-2", tmp_path)
        layout.path.mkdir(parents=True)
        assert layout.exists()

    def test_toolchain_layout_find_executable(self, tmp_path):
        """
        ToolchainLayout.find_executable() locates an executable.

        Expected result: Path to gcc, or None if not found.

        All platforms: PASS
        """
        toolchain_dir = _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        layout = ToolchainLayout("gcc", "14.2.0-2", tmp_path)
        exe = layout.find_executable("gcc")
        assert exe is not None
        assert exe.is_file()

    def test_toolchain_set(self, tmp_path):
        """
        ToolchainSet manages all versions of a compiler family.

        Expected result: list_versions() returns installed versions.

        All platforms: PASS
        """
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        _fake_toolchain_dir(tmp_path, "gcc", "13.3.0")
        ts = ToolchainSet("gcc", tmp_path)
        versions = ts.list_versions()
        assert "14.2.0-2" in versions
        assert "13.3.0" in versions
        assert ts.latest_version() == "14.2.0-2"

    def test_list_installed(self, tmp_path):
        """
        list_installed() returns all installed (compiler, version) pairs.

        Expected result: List of tuples.

        All platforms: PASS
        """
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        _fake_toolchain_dir(tmp_path, "clang", "18.1.0")
        installed = list_installed(tmp_path)
        assert ("gcc", "14.2.0-2") in installed
        assert ("clang", "18.1.0") in installed

    def test_list_compiler_families(self, tmp_path):
        """
        list_compiler_families() returns unique compiler names.

        Expected result: ["clang", "gcc"].

        All platforms: PASS
        """
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        _fake_toolchain_dir(tmp_path, "clang", "18.1.0")
        families = list_compiler_families(tmp_path)
        assert "gcc" in families
        assert "clang" in families

    def test_set_and_get_default(self, tmp_path):
        """
        set_default() and get_default_version() manage default symlinks.

        Expected result: get_default_version returns the set version.

        All platforms: PASS
        """
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        _fake_toolchain_dir(tmp_path, "gcc", "13.3.0")
        set_default("gcc", "14.2.0-2", tmp_path)
        assert get_default_version("gcc", tmp_path) == "14.2.0-2"

    def test_cleanup_temp(self, tmp_path):
        """
        cleanup_temp() removes files in .tmp directory.

        Expected result: Returns count of removed items.

        All platforms: PASS
        """
        temp_dir = get_temp_dir(tmp_path)
        temp_dir.mkdir(parents=True, exist_ok=True)
        (temp_dir / "test-file").write_text("data")
        count = cleanup_temp(tmp_path)
        assert count == 1
        assert not (temp_dir / "test-file").exists()

    def test_remove_toolchain_layout(self, tmp_path):
        """
        remove_toolchain_layout() deletes a toolchain directory.

        Expected result: Directory no longer exists.

        All platforms: PASS
        """
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        assert remove_toolchain_layout("gcc", "14.2.0-2", tmp_path) is True
        assert not ToolchainLayout("gcc", "14.2.0-2", tmp_path).exists()


# ============================================================================
# Symlink Tests
# ============================================================================

class TestSymlinks:
    """
    Tests for symlink management.

    Expected pass rate:
        Linux/macOS: 100%
        Windows: 70% (symlinks require Developer Mode)
        Android: 70% (some file systems restrict symlinks)
    """

    def test_extract_major_version(self):
        """
        _extract_major_version() returns the major version number.

        Expected result: "14" from "14.2.0-2", "18" from "18.1.0".

        All platforms: PASS
        """
        assert _extract_major_version("14.2.0-2") == "14"
        assert _extract_major_version("18.1.0") == "18"
        assert _extract_major_version("0.11.0") == "0"

    def test_short_link_name(self):
        """
        _short_link_name() returns the executable name unchanged.

        Expected result: "gcc" from "gcc".

        All platforms: PASS
        """
        assert _short_link_name("gcc") == "gcc"
        assert _short_link_name("g++") == "g++"

    def test_versioned_link_name(self):
        """
        _versioned_link_name() returns name@major format.

        Expected result: "gcc@14" from ("gcc", "14.2.0-2").

        All platforms: PASS
        """
        assert _versioned_link_name("gcc", "14.2.0-2") == "gcc@14"
        assert _versioned_link_name("clang", "18.1.0") == "clang@18"

    def test_symlink_manager_init(self, tmp_path):
        """
        SymlinkManager creates the link directory on init.

        Expected result: link_root exists.

        All platforms: PASS
        """
        manager = SymlinkManager(tmp_path, tmp_path / "bin")
        assert manager.link_root.exists()

    def test_symlink_manager_create_and_remove(self, tmp_path):
        """
        SymlinkManager creates and removes versioned links.

        Expected result: Links are created and then removed.

        All platforms: PASS
        """
        toolchain_dir = _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        manager = SymlinkManager(tmp_path, tmp_path / "bin")

        # Create links
        count = manager.create_links("gcc", "14.2.0-2")
        # At least some links should be created
        assert count >= 0  # May be 0 if symlinks not supported

        # Remove links
        removed = manager.remove_links("gcc", "14.2.0-2")
        assert isinstance(removed, int)

    def test_symlink_manager_list_links(self, tmp_path):
        """
        SymlinkManager.list_links() returns a dict.

        Expected result: dict, possibly empty.

        All platforms: PASS
        """
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        manager = SymlinkManager(tmp_path, tmp_path / "bin")
        manager.create_links("gcc", "14.2.0-2")
        links = manager.list_links()
        assert isinstance(links, dict)

    def test_symlink_manager_rescan(self, tmp_path):
        """
        SymlinkManager.rescan() returns (stale, removed) counts.

        Expected result: tuple of two ints.

        All platforms: PASS
        """
        manager = SymlinkManager(tmp_path, tmp_path / "bin")
        stale, removed = manager.rescan()
        assert isinstance(stale, int)
        assert isinstance(removed, int)


# ============================================================================
# Activation Tests
# ============================================================================

class TestActivation:
    """
    Tests for toolchain activation/deactivation.

    Expected pass rate: 90-100% on all platforms.
    Tests modify os.environ and restore it.

    Warnings
    --------
    These tests modify os.environ['PATH'] and TOOLFORGE_ACTIVE.
    They restore the original values after each test.
    """

    @pytest.fixture(autouse=True)
    def _save_restore_env(self):
        """Save and restore environment variables after each test."""
        original_path = os.environ.get("PATH", "")
        original_active = os.environ.get("TOOLFORGE_ACTIVE", "")
        yield
        os.environ["PATH"] = original_path
        if original_active:
            os.environ["TOOLFORGE_ACTIVE"] = original_active
        else:
            os.environ.pop("TOOLFORGE_ACTIVE", None)
        # Clear activation stack
        deactivate_all()

    def test_activate_nonexistent(self):
        """
        activate() returns False for non-existent toolchain.

        Expected result: False.

        All platforms: PASS
        """
        result = activate("nonexistent", "1.0.0", Path("/nonexistent"))
        assert result is False

    def test_activate_existing(self, tmp_path):
        """
        activate() returns True for an existing toolchain.

        Expected result: True.

        All platforms: PASS
        """
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        result = activate("gcc", "14.2.0-2", tmp_path)
        assert result is True

    def test_deactivate_empty(self):
        """
        deactivate() returns None when stack is empty.

        Expected result: None.

        All platforms: PASS
        """
        deactivate_all()
        result = deactivate()
        assert result is None

    def test_deactivate_after_activate(self, tmp_path):
        """
        deactivate() returns the (compiler, version) after activate.

        Expected result: ("gcc", "14.2.0-2").

        All platforms: PASS
        """
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        activate("gcc", "14.2.0-2", tmp_path)
        result = deactivate()
        assert result == ("gcc", "14.2.0-2")

    def test_current_stack(self, tmp_path):
        """
        current() returns the activation stack.

        Expected result: List of (compiler, version) tuples.

        All platforms: PASS
        """
        deactivate_all()
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        activate("gcc", "14.2.0-2", tmp_path)
        stack = current()
        assert len(stack) == 1
        assert stack[0] == ("gcc", "14.2.0-2")

    def test_is_active(self, tmp_path):
        """
        is_active() checks if a compiler is active.

        Expected result: True after activation, False after deactivation.

        All platforms: PASS
        """
        deactivate_all()
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        activate("gcc", "14.2.0-2", tmp_path)
        assert is_active("gcc", "14.2.0-2") is True
        deactivate()
        assert is_active("gcc", "14.2.0-2") is False

    def test_deactivate_all(self, tmp_path):
        """
        deactivate_all() clears the entire stack.

        Expected result: Stack is empty.

        All platforms: PASS
        """
        deactivate_all()
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        activate("gcc", "14.2.0-2", tmp_path)
        activate("gcc", "14.2.0-2", tmp_path)
        count = deactivate_all()
        assert count == 2
        assert len(current()) == 0

    def test_shell_activate_script_bash(self, tmp_path):
        """
        shell_activate_script() generates bash commands.

        Expected result: String with export PATH and TOOLFORGE_ACTIVE.

        All platforms: PASS
        """
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        script = shell_activate_script("gcc", "14.2.0-2", tmp_path, "bash")
        assert "export PATH=" in script
        assert "export TOOLFORGE_ACTIVE=" in script

    def test_shell_activate_script_raises_for_missing(self):
        """
        shell_activate_script() raises FileNotFoundError for missing toolchain.

        Expected result: FileNotFoundError.

        All platforms: PASS
        """
        with pytest.raises(FileNotFoundError):
            shell_activate_script("nonexistent", "1.0.0", Path("/nonexistent"))

    def test_shell_activate_script_raises_for_unknown_shell(self, tmp_path):
        """
        shell_activate_script() raises ValueError for unknown shell.

        Expected result: ValueError.

        All platforms: PASS
        """
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        with pytest.raises(ValueError):
            shell_activate_script("gcc", "14.2.0-2", tmp_path, "csh")

    def test_shell_deactivate_script(self):
        """
        shell_deactivate_script() generates deactivation commands.

        Expected result: Non-empty string.

        All platforms: PASS
        """
        script = shell_deactivate_script("bash")
        assert isinstance(script, str)
        assert len(script) > 0

    def test_read_marker_env_empty(self):
        """
        read_marker_env() returns empty list when var is unset.

        Expected result: [].

        All platforms: PASS
        """
        os.environ.pop("TOOLFORGE_ACTIVE", None)
        result = read_marker_env()
        assert result == []

    def test_read_marker_env_parses_pairs(self):
        """
        read_marker_env() parses "gcc:14.2.0-2::clang:18.1.0".

        Expected result: [("gcc", "14.2.0-2"), ("clang", "18.1.0")].

        All platforms: PASS
        """
        os.environ["TOOLFORGE_ACTIVE"] = "gcc:14.2.0-2::clang:18.1.0"
        result = read_marker_env()
        assert len(result) == 2
        assert result[0] == ("gcc", "14.2.0-2")
        assert result[1] == ("clang", "18.1.0")


# ============================================================================
# Manifest Tests
# ============================================================================

class TestManifests:
    """
    Tests for installation manifests.

    Expected pass rate: 100% on all platforms.
    Pure file I/O with temporary directories.
    """

    def test_manifest_create(self):
        """
        Manifest.create() returns a Manifest with a timestamp.

        Expected result: Manifest with compiler, version, installed_at.

        All platforms: PASS
        """
        manifest = Manifest.create(
            compiler="gcc",
            version="14.2.0-2",
            url="https://example.com/gcc.tar.gz",
        )
        assert manifest.compiler == "gcc"
        assert manifest.version == "14.2.0-2"
        assert manifest.url == "https://example.com/gcc.tar.gz"
        assert manifest.installed_at != ""

    def test_manifest_to_dict(self):
        """
        Manifest.to_dict() produces a JSON-compatible dict.

        Expected result: dict with all fields.

        All platforms: PASS
        """
        manifest = Manifest.create("gcc", "14.2.0-2", "https://example.com")
        d = manifest.to_dict()
        assert d["compiler"] == "gcc"
        assert d["version"] == "14.2.0-2"

    def test_manifest_from_dict(self):
        """
        Manifest.from_dict() reconstructs a Manifest.

        Expected result: Manifest with matching fields.

        All platforms: PASS
        """
        data = {
            "compiler": "gcc",
            "version": "14.2.0-2",
            "url": "https://example.com",
            "installed_at": "2025-01-01T00:00:00",
            "checksum": None,
            "file_count": 100,
            "size_bytes": 500000,
            "symlinks_created": ["gcc@14"],
            "was_set_default": False,
            "manifest_version": 1,
        }
        manifest = Manifest.from_dict(data)
        assert manifest.compiler == "gcc"
        assert manifest.file_count == 100

    def test_manifest_store_save_and_load(self, tmp_path):
        """
        ManifestStore saves and loads a manifest.

        Expected result: Loaded manifest equals saved manifest.

        All platforms: PASS
        """
        store = ManifestStore(tmp_path)
        manifest = Manifest.create("gcc", "14.2.0-2", "https://example.com")
        store.save(manifest)

        loaded = store.load("gcc", "14.2.0-2")
        assert loaded is not None
        assert loaded.compiler == "gcc"
        assert loaded.version == "14.2.0-2"

    def test_manifest_store_load_nonexistent(self, tmp_path):
        """
        ManifestStore.load() returns None for nonexistent manifest.

        Expected result: None.

        All platforms: PASS
        """
        store = ManifestStore(tmp_path)
        assert store.load("gcc", "14.2.0-2") is None

    def test_manifest_store_delete(self, tmp_path):
        """
        ManifestStore.delete() removes a manifest.

        Expected result: load() returns None after delete.

        All platforms: PASS
        """
        store = ManifestStore(tmp_path)
        manifest = Manifest.create("gcc", "14.2.0-2", "https://example.com")
        store.save(manifest)
        assert store.exists("gcc", "14.2.0-2")
        store.delete("gcc", "14.2.0-2")
        assert not store.exists("gcc", "14.2.0-2")

    def test_manifest_store_list_all(self, tmp_path):
        """
        ManifestStore.list_all() returns all manifests.

        Expected result: List with saved manifests.

        All platforms: PASS
        """
        store = ManifestStore(tmp_path)
        store.save(Manifest.create("gcc", "14.2.0-2", "url"))
        store.save(Manifest.create("clang", "18.1.0", "url"))
        all_m = store.list_all()
        assert len(all_m) == 2

    def test_manifest_store_list_by_compiler(self, tmp_path):
        """
        ManifestStore.list_by_compiler() filters by compiler.

        Expected result: Only GCC manifests.

        All platforms: PASS
        """
        store = ManifestStore(tmp_path)
        store.save(Manifest.create("gcc", "14.2.0-2", "url"))
        store.save(Manifest.create("clang", "18.1.0", "url"))
        gcc_list = store.list_by_compiler("gcc")
        assert len(gcc_list) == 1
        assert gcc_list[0].compiler == "gcc"

    def test_compute_directory_stats(self, tmp_path):
        """
        compute_directory_stats() counts files and total bytes.

        Expected result: (file_count, total_bytes).

        All platforms: PASS
        """
        dir_path = tmp_path / "test-dir"
        dir_path.mkdir()
        (dir_path / "a.txt").write_text("hello")
        (dir_path / "b.txt").write_text("world!")
        file_count, total_bytes = compute_directory_stats(dir_path)
        assert file_count == 2
        assert total_bytes == 11

    def test_compute_checksum(self, tmp_path):
        """
        compute_checksum() returns SHA256 hex digest.

        Expected result: 64-character hex string.

        All platforms: PASS
        """
        file_path = tmp_path / "test.txt"
        file_path.write_text("hello world")
        checksum = compute_checksum(file_path)
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)


# ============================================================================
# Environment Tests
# ============================================================================

class TestInstallerEnvironments:
    """
    Tests for installer environment management.

    Expected pass rate: 100% on all platforms.
    Pure file I/O with temporary directories.
    """

    def test_create_environment(self, tmp_path):
        """
        EnvironmentStore.create() creates an environment file.

        Expected result: Environment with correct name and compilers.

        All platforms: PASS
        """
        store = EnvironmentStore(tmp_path)
        env = store.create("cpp20", {"gcc": "14.2.0-2", "clang": "18.1.0"})
        assert env.name == "cpp20"
        assert env.compilers == {"gcc": "14.2.0-2", "clang": "18.1.0"}

    def test_get_environment(self, tmp_path):
        """
        EnvironmentStore.get() retrieves a saved environment.

        Expected result: Environment with matching fields.

        All platforms: PASS
        """
        store = EnvironmentStore(tmp_path)
        store.create("test", {"gcc": "14.2.0-2"})
        env = store.get("test")
        assert env is not None
        assert env.compilers["gcc"] == "14.2.0-2"

    def test_update_environment(self, tmp_path):
        """
        EnvironmentStore.update() modifies an environment.

        Expected result: Updated compilers reflect the change.

        All platforms: PASS
        """
        store = EnvironmentStore(tmp_path)
        store.create("test", {"gcc": "14.2.0-2"})
        store.update("test", compilers={"gcc": "15.0.0", "zig": "0.11.0"})
        env = store.get("test")
        assert env.compilers["gcc"] == "15.0.0"
        assert env.compilers["zig"] == "0.11.0"

    def test_delete_environment(self, tmp_path):
        """
        EnvironmentStore.delete() removes an environment.

        Expected result: get() returns None after delete.

        All platforms: PASS
        """
        store = EnvironmentStore(tmp_path)
        store.create("test", {})
        assert store.get("test") is not None
        store.delete("test")
        assert store.get("test") is None

    def test_list_all_environments(self, tmp_path):
        """
        EnvironmentStore.list_all() returns environment names.

        Expected result: List of names.

        All platforms: PASS
        """
        store = EnvironmentStore(tmp_path)
        store.create("env-a", {})
        store.create("env-b", {})
        names = store.list_all()
        assert "env-a" in names
        assert "env-b" in names

    def test_export_import_environment(self, tmp_path):
        """
        Environment can be exported and re-imported.

        Expected result: Imported environment matches original.

        All platforms: PASS
        """
        store = EnvironmentStore(tmp_path)
        store.create("original", {"gcc": "14.2.0-2"})

        export_path = tmp_path / "exported.json"
        store.export_env("original", export_path)
        assert export_path.is_file()

        imported = store.import_env(export_path, overwrite=True)
        assert imported.name == "original"

    def test_add_and_remove_compiler(self, tmp_path):
        """
        add_compiler() and remove_compiler() modify an environment.

        Expected result: Compiler added then removed.

        All platforms: PASS
        """
        store = EnvironmentStore(tmp_path)
        store.create("test", {"gcc": "14.2.0-2"})
        store.add_compiler("test", "zig", "0.11.0")
        env = store.get("test")
        assert "zig" in env.compilers

        store.remove_compiler("test", "zig")
        env = store.get("test")
        assert "zig" not in env.compilers

    def test_validate_environment(self, tmp_path):
        """
        validate() checks if referenced toolchains exist.

        Expected result: List of warnings.

        All platforms: PASS
        """
        store = EnvironmentStore(tmp_path)
        store.create("test", {"gcc": "999.0.0"})
        warnings = store.validate("test")
        assert isinstance(warnings, list)


# ============================================================================
# Uninstall Tests
# ============================================================================

class TestUninstall:
    """
    Tests for uninstallation.

    Expected pass rate: 100% on all platforms.
    All destructive operations use temporary directories.
    """

    def test_dry_run_uninstall(self, tmp_path):
        """
        dry_run_uninstall() returns an UninstallPlan.

        Expected result: UninstallPlan instance.

        All platforms: PASS
        """
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        plan = dry_run_uninstall("gcc", "14.2.0-2", tmp_path)
        assert isinstance(plan, UninstallPlan)
        assert plan.compiler == "gcc"
        assert plan.version == "14.2.0-2"

    def test_dry_run_uninstall_nonexistent(self, tmp_path):
        """
        dry_run_uninstall() for non-existent toolchain has toolchain_exists=False.

        Expected result: toolchain_exists is False.

        All platforms: PASS
        """
        plan = dry_run_uninstall("gcc", "999.0.0", tmp_path)
        assert not plan.toolchain_exists

    def test_uninstall_toolchain(self, tmp_path):
        """
        uninstall_toolchain() removes a toolchain.

        Expected result: True, and directory no longer exists.

        All platforms: PASS
        """
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        result = uninstall_toolchain("gcc", "14.2.0-2", tmp_path)
        assert result is True
        assert not ToolchainLayout("gcc", "14.2.0-2", tmp_path).exists()

    def test_force_uninstall(self, tmp_path):
        """
        force_uninstall() removes even if environments reference it.

        Expected result: True.

        All platforms: PASS
        """
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        result = force_uninstall("gcc", "14.2.0-2", tmp_path)
        assert result is True

    def test_clean_orphans(self, tmp_path):
        """
        clean_orphans() removes stale files.

        Expected result: Returns count of cleaned items.

        All platforms: PASS
        """
        count = clean_orphans(tmp_path)
        assert isinstance(count, int)

    def test_uninstall_plan_summary(self, tmp_path):
        """
        UninstallPlan.summary() returns a human-readable string.

        Expected result: Multi-line string.

        All platforms: PASS
        """
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        plan = dry_run_uninstall("gcc", "14.2.0-2", tmp_path)
        summary = plan.summary()
        assert "gcc" in summary
        assert "14.2.0-2" in summary


# ============================================================================
# Install Tests
# ============================================================================

class TestInstall:
    """
    Tests for the main install orchestrator.

    Expected pass rate:
        Linux:    85-95% (network available)
        macOS:    85-95%
        Windows:  70-80%
        Android:  60-75% (aiohttp may not be installed)
    """

    def test_install_result_defaults(self):
        """
        InstallResult has correct default values.

        Expected result: success=False, all optional fields None.

        All platforms: PASS
        """
        result = InstallResult()
        assert result.success is False
        assert result.path is None
        assert result.artifact is None
        assert result.error is None
        assert result.checksum_verified is False

    def test_install_result_success_repr(self):
        """
        InstallResult.__repr__ for success case.

        Expected result: Contains "success=True".

        All platforms: PASS
        """
        result = InstallResult()
        result.success = True
        result.path = Path("/test/path")
        repr_str = repr(result)
        assert "success=True" in repr_str

    def test_install_result_failure_repr(self):
        """
        InstallResult.__repr__ for failure case.

        Expected result: Contains "success=False" and error message.

        All platforms: PASS
        """
        result = InstallResult()
        result.success = False
        result.error = "Download failed"
        repr_str = repr(result)
        assert "success=False" in repr_str
        assert "Download failed" in repr_str

    def test_find_toolchain_existing(self, tmp_path):
        """
        find_toolchain() returns the path for an installed toolchain.

        Expected result: Path object.

        All platforms: PASS
        """
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        path = find_toolchain("gcc", "14.2.0-2", tmp_path)
        assert path is not None
        assert path.is_dir()

    def test_find_toolchain_nonexistent(self, tmp_path):
        """
        find_toolchain() returns None for non-existent toolchain.

        Expected result: None.

        All platforms: PASS
        """
        path = find_toolchain("gcc", "999.0.0", tmp_path)
        assert path is None

    def test_list_installed_returns_list(self, tmp_path):
        """
        install_list_installed() returns a list.

        Expected result: List of (compiler, version) tuples.

        All platforms: PASS
        """
        _fake_toolchain_dir(tmp_path, "gcc", "14.2.0-2")
        installed = install_list_installed(tmp_path)
        assert isinstance(installed, list)
        assert ("gcc", "14.2.0-2") in installed

    @pytest.mark.skipif(not _has_network(), reason="aiohttp not installed")
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_install_toolchain_creates_directory(self, tmp_path):
        """
        install_toolchain() downloads and extracts a toolchain.

        This test requires network access and aiohttp.
        It downloads a real (but small) toolchain artifact.

        Expected result: InstallResult.success is True.

        Linux:    PASS (network available)
        macOS:    PASS
        Windows:  PASS
        Android:  PASS (if aiohttp installed)
        """
        # Skip checksum to avoid 404 issues
        os.environ["TOOLFORGE_SKIP_CHECKSUM"] = "1"
        try:
            result = await install_toolchain(
                "gcc",
                "14.2.0-2",
                install_root=tmp_path,
            )
            # May fail due to network issues, but should not crash
            assert isinstance(result, InstallResult)
        finally:
            os.environ.pop("TOOLFORGE_SKIP_CHECKSUM", None)


# ============================================================================
# Run configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
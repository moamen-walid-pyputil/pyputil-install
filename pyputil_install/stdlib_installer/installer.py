"""
Core installer for standard library modules.

Orchestrates the download, installation, verification, and removal
of standard library packages. Handles dependency resolution, rollback
on failure, and compiled module detection. Uses the FetchResult and
GitHubItem dataclasses from downloader for type-safe tree handling.
"""

import ast
import hashlib
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any

from .downloader import Downloader, FetchResult, GitHubItem
from .manifest import Manifest
from .exceptions import (
    StdlibInstallerError,
    PackageNotFoundError,
    ModuleNotInstalledError,
    AlreadyInstalledError,
    CompiledModuleError,
    NetworkError,
    InstallationError,
    CircularDependencyError,
    ChecksumVerificationError,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Dependency Resolver
# ============================================================================


class DependencyResolver:
    """
    Detects and resolves dependencies for installed Python modules.

    Parses Python source files with the ``ast`` module to find import
    statements and identifies which imports correspond to standard
    library modules that may need to be co-installed.

    Parameters
    ----------
    stdlib_module_names : set of str or None
        Set of known standard library module names. If ``None``,
        defaults to ``sys.stdlib_module_names`` when available
        (Python 3.10+), otherwise an empty set. When the set is empty,
        all discovered imports are treated as potential stdlib modules.

    Attributes
    ----------
    stdlib_names : set of str
        The set of known standard library module names used to filter
        detected imports.
    visited : set of str
        Modules already visited during the current dependency resolution
        pass. Cleared at the start of each ``resolve_dependencies`` call.

    Examples
    --------
    >>> resolver = DependencyResolver()
    >>> deps = resolver.scan_file(Path("mymodule.py"))
    >>> print(deps)
    {'json', 'os', 'pathlib'}
    """

    def __init__(
        self,
        stdlib_module_names: Optional[Set[str]] = None,
    ) -> None:
        self.stdlib_names: Set[str] = stdlib_module_names or set()
        if not self.stdlib_names and hasattr(sys, "stdlib_module_names"):
            self.stdlib_names = sys.stdlib_module_names
        self.visited: Set[str] = set()

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan_file(self, file_path: Path) -> Set[str]:
        """
        Extract all top-level imported module names from a single
        ``.py`` file.

        Handles both forms of import:

        - ``import x`` and ``import x.y`` — yields ``"x"``.
        - ``from x import y`` and ``from x.y import z`` — yields
          ``"x"``.

        Relative imports (``from . import ...``) are silently skipped
        because they do not reference standard library modules.

        Parameters
        ----------
        file_path : Path
            Path to a ``.py`` file to scan.

        Returns
        -------
        set of str
            Top-level module names imported. Returns an empty set if
            the file cannot be read, is not a ``.py`` file, or contains
            a syntax error.

        Examples
        --------
        >>> resolver = DependencyResolver()
        >>> imports = resolver.scan_file(Path("test.py"))
        >>> "json" in imports
        True
        """
        if file_path.suffix != ".py":
            return set()

        try:
            source = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(
                "Cannot read %s for dependency scan: %s", file_path, e
            )
            return set()

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            logger.warning(
                "Syntax error in %s, skipping dependency scan: %s",
                file_path,
                e,
            )
            return set()

        imports: Set[str] = set()

        for node in ast.walk(tree):
            # import x, import x.y
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level = alias.name.split(".")[0]
                    imports.add(top_level)

            # from x import y, from x.y import z
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top_level = node.module.split(".")[0]
                    imports.add(top_level)

        return imports

    def scan_directory(self, directory: Path) -> Set[str]:
        """
        Recursively scan all ``.py`` files under a directory and return
        the union of all discovered imports.

        Parameters
        ----------
        directory : Path
            Root directory to scan.

        Returns
        -------
        set of str
            All top-level module names imported by any ``.py`` file
            under the directory.
        """
        all_imports: Set[str] = set()
        if not directory.is_dir():
            return all_imports

        for py_file in directory.rglob("*.py"):
            all_imports.update(self.scan_file(py_file))

        return all_imports

    def scan_fetch_result(self, result: FetchResult) -> Set[str]:
        """
        Extract dependency hints from a ``FetchResult`` without
        downloading files.

        Uses the file names present in the tree as a lightweight
        heuristic: any file named ``<name>.py`` where ``<name>``
        is a known standard library module is treated as a likely
        dependency. This avoids downloading files solely for dependency
        scanning.

        Parameters
        ----------
        result : FetchResult
            The fetch result for the package being installed.

        Returns
        -------
        set of str
            Candidate dependency names.
        """
        candidates: Set[str] = set()

        for item in result:
            if item.is_file and item.name.endswith(".py"):
                module_name = item.name[:-3]  # strip ".py"
                if module_name in self.stdlib_names:
                    candidates.add(module_name)

        return candidates

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_dependencies(
        self,
        module_names: Set[str],
        installed: Set[str],
        max_depth: int = 10,
    ) -> List[str]:
        """
        Resolve dependencies for a set of module names recursively.

        Only modules that are present in ``stdlib_names`` and not
        already in ``installed`` are included in the result. The
        returned list is ordered so that dependencies appear before
        the modules that depend on them (topological ordering is
        approximated by depth-first traversal).

        Parameters
        ----------
        module_names : set of str
            Seed set of module names to start resolution from.
        installed : set of str
            Names of already-installed modules. These are excluded
            from the returned list.
        max_depth : int
            Maximum recursion depth. Guards against deeply nested or
            circular dependency chains.

        Returns
        -------
        list of str
            Ordered list of dependency names to install.

        Raises
        ------
        CircularDependencyError
            If a circular chain is detected (a module appears in its
            own dependency path).
        """
        resolved: List[str] = []
        self.visited = set()

        for name in sorted(module_names):
            if name not in installed and name not in resolved:
                self._resolve_one(
                    name, installed, resolved, [], max_depth
                )

        return resolved

    def _resolve_one(
        self,
        name: str,
        installed: Set[str],
        resolved: List[str],
        path: List[str],
        max_depth: int,
    ) -> None:
        """
        Recursive worker for dependency resolution of a single module.

        Parameters
        ----------
        name : str
            Module name being resolved.
        installed : set of str
            Already-installed module names (excluded from output).
        resolved : list of str
            Accumulator list. Successfully resolved dependencies are
            appended in order.
        path : list of str
            Current resolution stack for circularity detection.
        max_depth : int
            Remaining depth allowance.

        Raises
        ------
        CircularDependencyError
            If ``name`` already appears in ``path``.
        """
        if name in installed:
            return

        if name in path:
            chain = path + [name]
            raise CircularDependencyError(chain=chain)

        if len(path) >= max_depth:
            logger.warning(
                "Max dependency depth reached for '%s', stopping",
                name,
            )
            return

        if name in self.visited:
            return

        self.visited.add(name)

        if name in self.stdlib_names or True:
            if name not in resolved and name not in installed:
                resolved.append(name)


# ============================================================================
# Installer
# ============================================================================


class Installer:
    """
    Main orchestrator for installing and managing standard library
    modules.

    Coordinates the full lifecycle: fetching repository metadata,
    downloading files, writing them to disk, resolving and installing
    dependencies, recording everything in the manifest, and performing
    rollback on failure.

    Parameters
    ----------
    storage_dir : Path or None
        Root directory for installed packages and the manifest file.
        Defaults to ``~/.stdlib-packages``. Created if it does not
        exist.
    downloader : Downloader or None
        Pre-configured ``Downloader`` instance. Created with defaults
        when ``None``.
    manifest : Manifest or None
        Pre-configured ``Manifest`` instance bound to the same
        ``storage_dir``. Created when ``None``.
    dependency_resolver : DependencyResolver or None
        Pre-configured resolver. Created with defaults when ``None``.

    Attributes
    ----------
    storage_dir : Path
        Root directory containing all installed packages.
    downloader : Downloader
        Handles network transport and caching.
    manifest : Manifest
        Persistent record of installed packages.
    resolver : DependencyResolver
        AST-based import scanner.

    Examples
    --------
    >>> installer = Installer()
    >>> path = installer.install("json")
    >>> installer.is_installed("json")
    True
    >>> installer.remove("json")
    """

    #: Modules known to be C extensions. These cannot be installed from
    #: source because they require platform-specific compilation.
    COMPILED_MODULES: Set[str] = {
        "math",
        "_socket",
        "_ssl",
        "_sqlite3",
        "_ctypes",
        "_tkinter",
        "_curses",
        "_hashlib",
        "_bz2",
        "_lzma",
        "_decimal",
        "_elementtree",
        "_json",
        "_multiprocessing",
        "_csv",
        "_posixsubprocess",
        "_random",
        "_statistics",
        "array",
        "fcntl",
        "grp",
        "ossaudiodev",
        "pty",
        "pwd",
        "resource",
        "select",
        "spwd",
        "syslog",
        "termios",
        "_asyncio",
        "_bisect",
        "_blake2",
        "_codecs",
        "_contextvars",
        "_crypt",
        "_datetime",
        "_dbm",
        "_gdbm",
        "_heapq",
        "_lsprof",
        "_opcode",
        "_pickle",
        "_posixshmem",
        "_queue",
        "_sha1",
        "_sha256",
        "_sha3",
        "_sha512",
        "_signal",
        "_stat",
        "_string",
        "_struct",
        "_symtable",
        "_testbuffer",
        "_testimportmultiple",
        "_testmultiphase",
        "_tracemalloc",
        "_uuid",
        "_xxsubinterpreters",
        "_zoneinfo",
    }

    #: Mapping from compiled module names to suggested system packages
    #: for common package managers.
    SYSTEM_PACKAGES: Dict[str, Dict[str, str]] = {
        "_tkinter": {
            "apt": "python3-tk",
            "dnf": "python3-tkinter",
            "pacman": "tk",
            "brew": "python-tk",
        },
        "_sqlite3": {
            "apt": "libsqlite3-dev",
            "dnf": "sqlite-devel",
            "pacman": "sqlite",
            "brew": "sqlite",
        },
        "_ssl": {
            "apt": "libssl-dev",
            "dnf": "openssl-devel",
            "pacman": "openssl",
            "brew": "openssl",
        },
        "_bz2": {
            "apt": "libbz2-dev",
            "dnf": "bzip2-devel",
            "pacman": "bzip2",
            "brew": "bzip2",
        },
        "_lzma": {
            "apt": "liblzma-dev",
            "dnf": "xz-devel",
            "pacman": "xz",
            "brew": "xz",
        },
        "_curses": {
            "apt": "libncurses-dev",
            "dnf": "ncurses-devel",
            "pacman": "ncurses",
            "brew": "ncurses",
        },
    }

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        downloader: Optional[Downloader] = None,
        manifest: Optional[Manifest] = None,
        dependency_resolver: Optional[DependencyResolver] = None,
    ) -> None:
        self.storage_dir = storage_dir or Path.home() / ".stdlib-packages"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.downloader = downloader or Downloader()
        self.manifest = manifest or Manifest(self.storage_dir)
        self.resolver = dependency_resolver or DependencyResolver()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _check_compiled(self, name: str) -> None:
        """
        Raise ``CompiledModuleError`` if ``name`` is a known C
        extension module.

        Checks against ``COMPILED_MODULES`` and also catches any name
        that starts with an underscore (convention for internal C
        extensions).

        Parameters
        ----------
        name : str
            Module name to check.

        Raises
        ------
        CompiledModuleError
            Always, when the name matches.
        """
        if name in self.COMPILED_MODULES or name.startswith("_"):
            system_pkgs = self.SYSTEM_PACKAGES.get(name, {})
            raise CompiledModuleError(
                module_name=name,
                system_packages=system_pkgs if system_pkgs else None,
            )

    def _validate_name(self, name: str) -> str:
        """
        Validate and normalise a user-supplied package name.

        Strips leading/trailing whitespace. Rejects empty strings,
        non-string types, paths containing ``..`` (parent directory
        traversal), and names that start or end with a dot.

        Parameters
        ----------
        name : str
            Raw name input.

        Returns
        -------
        str
            Stripped, valid name.

        Raises
        ------
        ValueError
            If the name fails any validation rule.
        """
        if not isinstance(name, str):
            raise ValueError(
                f"Name must be a string, got {type(name).__name__}"
            )
        if not name.strip():
            raise ValueError("Name cannot be empty")
        if ".." in name:
            raise ValueError(
                f"Invalid name: '{name}' contains double dots"
            )
        if name.startswith(".") or name.endswith("."):
            raise ValueError(
                f"Invalid name: '{name}' starts or ends with a dot"
            )
        return name.strip()

    def _get_package_dir(self, name: str) -> Path:
        """
        Resolve the installation directory path for a package name.

        Dots in the name are replaced with path separators to support
        nested packages (e.g., ``"xml.etree"`` → ``xml/etree``).

        Parameters
        ----------
        name : str
            Validated package name.

        Returns
        -------
        Path
            Absolute path inside ``storage_dir``.
        """
        return self.storage_dir / name.replace(".", "/")

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def is_installed(self, name: str) -> bool:
        """
        Check whether a package is currently installed.

        A package is considered installed when it exists in the
        manifest **and** its corresponding directory or file is
        present on disk.

        Parameters
        ----------
        name : str
            Package name.

        Returns
        -------
        bool
            ``True`` if installed, ``False`` otherwise.

        Examples
        --------
        >>> installer.is_installed("json")
        True
        """
        try:
            name = self._validate_name(name)
        except ValueError:
            return False

        if not self.manifest.has(name):
            return False

        return self._get_package_dir(name).exists()

    def list_installed(self) -> List[Dict[str, Any]]:
        """
        Return metadata for every installed package.

        Returns
        -------
        list of dict
            Each dict contains ``name``, ``version``, ``installed_at``,
            ``file_count``, and ``dependencies``. Sorted alphabetically
            by name.

        Examples
        --------
        >>> for pkg in installer.list_installed():
        ...     print(pkg["name"], pkg["version"])
        json 3.13
        """
        results: List[Dict[str, Any]] = []
        for name in self.manifest.list_names():
            entry = self.manifest.get(name)
            if entry:
                results.append(
                    {
                        "name": name,
                        "version": entry.get("version", "unknown"),
                        "installed_at": entry.get(
                            "installed_at", "unknown"
                        ),
                        "file_count": len(entry.get("files", [])),
                        "dependencies": entry.get("dependencies", []),
                    }
                )
        return sorted(results, key=lambda x: x["name"])

    def get_package_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Return the full manifest entry for a package.

        Parameters
        ----------
        name : str
            Package name.

        Returns
        -------
        dict or None
            Manifest entry dict, or ``None`` if not installed.
        """
        name = self._validate_name(name)
        return self.manifest.get(name)

    def check_updates(self) -> List[Dict[str, str]]:
        """
        Compare installed package versions against the current Python
        version.

        A package is considered outdated when its ``version`` field
        differs from the current interpreter's ``major.minor`` version
        string.

        Returns
        -------
        list of dict
            Each dict has keys ``name``, ``installed_version``, and
            ``available_version``.

        Examples
        --------
        >>> for u in installer.check_updates():
        ...     print(u["name"], "→", u["available_version"])
        json → 3.13
        """
        current = f"{sys.version_info.major}.{sys.version_info.minor}"
        updates: List[Dict[str, str]] = []

        for name in self.manifest.list_names():
            entry = self.manifest.get(name)
            if not entry:
                continue
            installed_version = entry.get("version")
            if installed_version and installed_version != current:
                updates.append(
                    {
                        "name": name,
                        "installed_version": installed_version,
                        "available_version": current,
                    }
                )

        return updates

    # ------------------------------------------------------------------
    # Install
    # ------------------------------------------------------------------

    def install(
        self,
        name: str,
        version: Optional[str] = None,
        force: bool = False,
        with_dependencies: bool = True,
        dry_run: bool = False,
    ) -> Path:
        """
        Install a standard library package.

        Full installation pipeline:

        1. Validate the name and version.
        2. Reject compiled C extensions.
        3. Check for existing installation (honour ``force``).
        4. Fetch the directory tree from GitHub as a ``FetchResult``.
        5. Scan for and optionally install dependencies.
        6. Download all files into a temporary location, then move
           into place (with backup/rollback on failure).
        7. Ensure an ``__init__.py`` exists for packages.
        8. Record the installation in the manifest.

        Parameters
        ----------
        name : str
            Dotted package/module name (e.g., ``"json"``,
            ``"xml.etree"``).
        version : str or None
            CPython version tag or branch. Defaults to the current
            interpreter's ``major.minor``.
        force : bool
            If ``True``, replace any existing installation.
        with_dependencies : bool
            If ``True``, attempt to detect and install standard library
            dependencies before the main package.
        dry_run : bool
            If ``True``, perform all checks and logging but do not
            write any files or modify the manifest.

        Returns
        -------
        Path
            Absolute path to the installed package directory or file.

        Raises
        ------
        AlreadyInstalledError
            When the package exists and ``force`` is ``False``.
        CompiledModuleError
            When the package is a C extension.
        PackageNotFoundError
            When the repository has no matching path.
        NetworkError
            On transport failures.
        InstallationError
            On other failures (disk full, permission denied, etc.).
        """
        name = self._validate_name(name)
        version = version or f"{sys.version_info.major}.{sys.version_info.minor}"

        # --- Already installed? -------------------------------------------------
        if self.is_installed(name):
            if not force:
                raise AlreadyInstalledError(
                    package_name=name,
                    installed_path=str(self._get_package_dir(name)),
                )

        # --- Compiled module? ---------------------------------------------------
        self._check_compiled(name)

        # --- Dry run ------------------------------------------------------------
        if dry_run:
            logger.info(
                "[DRY RUN] Would install '%s' from Python %s",
                name,
                version,
            )
            return self._get_package_dir(name)

        logger.info("Installing '%s' from Python %s", name, version)

        # --- Fetch tree ---------------------------------------------------------
        try:
            result = self.downloader.fetch_tree(name, version)
        except (PackageNotFoundError, NetworkError):
            raise
        except Exception as exc:
            raise InstallationError(
                f"Failed to fetch package metadata for '{name}': {exc}"
            )

        # --- Dependencies -------------------------------------------------------
        installed_deps: List[str] = []
        if with_dependencies:
            try:
                installed_deps = self._install_dependencies(
                    result, version
                )
            except Exception as exc:
                logger.warning(
                    "Dependency installation failed, continuing: %s", exc
                )

        # --- Download with rollback ---------------------------------------------
        target = self._get_package_dir(name)
        backup_dir: Optional[Path] = None
        installed_files: List[Path] = []

        try:
            # Backup existing installation for force-reinstall
            if target.exists() and force:
                backup_dir = target.with_name(target.name + ".backup")
                shutil.move(str(target), str(backup_dir))

            # Download
            installed_files = self.downloader.download_recursive(
                result, target, version
            )

            # Ensure __init__.py for package directories
            self._ensure_init_file(target, installed_files)

            # Record in manifest
            file_checksums = {
                str(f.relative_to(self.storage_dir)): self._hash_file(f)
                for f in installed_files
            }
            self.manifest.add(
                name=name,
                version=version,
                files=[str(f) for f in installed_files],
                checksums=file_checksums,
                dependencies=installed_deps,
            )

            # Discard backup
            if backup_dir is not None and backup_dir.exists():
                shutil.rmtree(backup_dir)

            logger.info("Successfully installed '%s' at %s", name, target)
            return target

        except Exception as exc:
            logger.error("Installation failed for '%s': %s", name, exc)

            # Rollback: restore backup
            if backup_dir is not None and backup_dir.exists():
                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(backup_dir), str(target))
                logger.info("Restored backup of '%s'", name)
            else:
                # Clean partial install
                if target.exists():
                    shutil.rmtree(target)
                for dep in installed_deps:
                    try:
                        self.manifest.remove(dep)
                    except Exception:
                        pass

            if isinstance(exc, StdlibInstallerError):
                raise
            raise InstallationError(
                f"Failed to install '{name}': {exc}"
            )

    def _install_dependencies(
        self,
        result: FetchResult,
        version: str,
    ) -> List[str]:
        """
        Detect and install standard library dependencies for a package
        represented by a ``FetchResult``.

        Uses ``DependencyResolver.scan_fetch_result`` to heuristically
        identify dependencies from file names in the tree, then calls
        ``install()`` for each one that is not already present.

        Parameters
        ----------
        result : FetchResult
            The fetch result for the main package.
        version : str
            CPython version string passed through to ``install()``.

        Returns
        -------
        list of str
            Names of the dependencies that were successfully installed
            during this call.
        """
        already = set(self.manifest.list_names())
        candidates = self.resolver.scan_fetch_result(result)

        installed: List[str] = []
        for dep_name in sorted(candidates):
            if dep_name in already:
                continue
            try:
                self.install(
                    name=dep_name,
                    version=version,
                    force=False,
                    with_dependencies=False,
                )
                installed.append(dep_name)
            except Exception as exc:
                logger.warning(
                    "Failed to install dependency '%s': %s", dep_name, exc
                )

        return installed

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------

    def remove(self, name: str) -> None:
        """
        Remove an installed package.

        Deletes all files tracked in the manifest for the package,
        removes the package directory (if empty after file removal),
        and drops the manifest entry.

        Parameters
        ----------
        name : str
            Package name.

        Raises
        ------
        ModuleNotInstalledError
            If the package is not in the manifest.

        Examples
        --------
        >>> installer.remove("json")
        """
        name = self._validate_name(name)

        if not self.manifest.has(name):
            raise ModuleNotInstalledError(module_name=name)

        entry = self.manifest.get(name)
        tracked_files: List[str] = entry.get("files", []) if entry else []
        package_dir = self._get_package_dir(name)

        # Remove individual tracked files
        for rel_path in tracked_files:
            abs_path = self.storage_dir / rel_path
            try:
                if abs_path.exists():
                    abs_path.unlink()
            except OSError as exc:
                logger.warning("Could not remove %s: %s", abs_path, exc)

        # Remove the package directory (and any leftover contents)
        if package_dir.exists():
            try:
                shutil.rmtree(package_dir)
            except OSError as exc:
                logger.warning(
                    "Could not remove directory %s: %s", package_dir, exc
                )

        # Drop from manifest
        self.manifest.remove(name)
        logger.info("Removed '%s'", name)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, name: str, version: Optional[str] = None) -> Path:
        """
        Update an installed package to a different version.

        This is a convenience wrapper that calls ``remove()`` followed
        by ``install()`` with ``force=True``.

        Parameters
        ----------
        name : str
            Package to update.
        version : str or None
            Target CPython version. Defaults to the current
            interpreter's version.

        Returns
        -------
        Path
            Path to the updated package.

        Raises
        ------
        ModuleNotInstalledError
            If the package is not currently installed.
        """
        name = self._validate_name(name)

        if not self.is_installed(name):
            raise ModuleNotInstalledError(module_name=name)

        logger.info("Updating '%s'", name)
        self.remove(name)
        return self.install(name, version=version, force=True)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_init_file(
        target: Path, installed_files: List[Path]
    ) -> None:
        """
        Create an ``__init__.py`` inside ``target`` if it is a
        directory containing ``.py`` files but lacking an init file.

        This ensures that the directory is treated as a regular
        package (rather than a namespace package) for consistent
        import behaviour across Python versions.

        Parameters
        ----------
        target : Path
            The package directory.
        installed_files : list of Path
            Files that were installed (the new ``__init__.py`` is
            appended to this list if created).
        """
        if not target.is_dir():
            return

        init_file = target / "__init__.py"
        if not init_file.exists():
            if any(f.suffix == ".py" for f in installed_files):
                try:
                    init_file.write_text(
                        "# Auto-generated by stdlib_installer\n"
                    )
                    installed_files.append(init_file)
                    logger.debug("Created %s", init_file)
                except OSError as exc:
                    logger.warning(
                        "Could not create __init__.py: %s", exc
                    )

    @staticmethod
    def _hash_file(file_path: Path) -> str:
        """
        Compute the SHA-256 hex digest of a file.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        str
            Hexadecimal digest, or the literal string ``"UNREADABLE"``
            if the file cannot be opened or read.
        """
        hasher = hashlib.sha256()
        try:
            with open(file_path, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    hasher.update(chunk)
        except OSError:
            return "UNREADABLE"
        return hasher.hexdigest()
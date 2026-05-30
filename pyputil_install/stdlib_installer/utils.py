"""
High-level convenience functions for stdlib_installer.

Provides simple one-call functions for common operations without
needing to instantiate Installer or manage paths manually.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Union

from .installer import Installer
from .exceptions import StdlibInstallerError

logger = logging.getLogger(__name__)

# Cache the installer instance so repeated calls reuse the same
# manifest and downloader (with their internal caches).
_installer: Optional[Installer] = None


def _get_installer(storage_dir: Optional[Path] = None) -> Installer:
    """
    Return a cached Installer instance, creating one if needed.

    Parameters
    ----------
    storage_dir : Path or None
        Custom storage directory. If None, uses the default
        ~/.stdlib-packages.

    Returns
    -------
    Installer
        Shared or newly-created installer instance.
    """
    global _installer

    if storage_dir is not None:
        # Custom path: always create a fresh installer so the caller
        # gets exactly what they asked for.
        return Installer(storage_dir=storage_dir)

    if _installer is None:
        _installer = Installer()

    return _installer


# ============================================================================
# Installation
# ============================================================================


def install_stdlib(
    name: str,
    version: Optional[str] = None,
    force: bool = False,
    with_dependencies: bool = True,
    storage_dir: Optional[Path] = None,
) -> Path:
    """
    Install a standard library module or package.

    Downloads the module from the CPython GitHub repository and
    stores it in the local package directory.

    Parameters
    ----------
    name : str
        Module or package name (e.g., ``"json"``, ``"csv"``,
        ``"xml.etree"``).
    version : str or None
        CPython version to download from. Defaults to the current
        Python version (``major.minor``).
    force : bool
        If ``True``, reinstall even if already installed.
    with_dependencies : bool
        If ``True``, attempt to detect and install stdlib dependencies.
    storage_dir : Path or None
        Custom storage directory. Defaults to ``~/.stdlib-packages``.

    Returns
    -------
    Path
        Absolute path to the installed package.

    Raises
    ------
    AlreadyInstalledError
        If the package exists and ``force`` is ``False``.
    CompiledModuleError
        If the package is a C extension.
    PackageNotFoundError
        If the package is not found in the repository.
    NetworkError
        On network failures.

    Examples
    --------
    >>> from stdlib_installer import install_stdlib
    >>> path = install_stdlib("json")
    >>> print(path)
    /home/user/.stdlib-packages/json

    >>> install_stdlib("csv", version="3.11", force=True)
    PosixPath('/home/user/.stdlib-packages/csv.py')
    """
    installer = _get_installer(storage_dir)
    return installer.install(
        name=name,
        version=version,
        force=force,
        with_dependencies=with_dependencies,
    )


def install_many_stdlib(
    names: List[str],
    version: Optional[str] = None,
    force: bool = False,
    with_dependencies: bool = True,
    storage_dir: Optional[Path] = None,
    stop_on_first_error: bool = False,
) -> Dict[str, Union[Path, Exception]]:
    """
    Install multiple standard library packages at once.

    Parameters
    ----------
    names : list of str
        Package names to install.
    version : str or None
        CPython version to download from.
    force : bool
        If ``True``, reinstall even if already installed.
    with_dependencies : bool
        If ``True``, resolve and install dependencies.
    storage_dir : Path or None
        Custom storage directory.
    stop_on_first_error : bool
        If ``True``, stop immediately on the first failure. If
        ``False`` (default), continue installing remaining packages
        and return a dict of results.

    Returns
    -------
    dict
        Mapping of package name to ``Path`` (success) or ``Exception``
        (failure).

    Examples
    --------
    >>> from stdlib_installer import install_many_stdlib
    >>> results = install_many_stdlib(["json", "csv", "datetime"])
    >>> for name, result in results.items():
    ...     if isinstance(result, Path):
    ...         print(f"{name}: installed at {result}")
    ...     else:
    ...         print(f"{name}: failed - {result}")
    """
    installer = _get_installer(storage_dir)
    results: Dict[str, Union[Path, Exception]] = {}

    for name in names:
        try:
            path = installer.install(
                name=name,
                version=version,
                force=force,
                with_dependencies=with_dependencies,
            )
            results[name] = path
        except Exception as exc:
            results[name] = exc
            if stop_on_first_error:
                break

    return results


# ============================================================================
# Removal
# ============================================================================


def remove_stdlib(
    name: str,
    storage_dir: Optional[Path] = None,
) -> None:
    """
    Remove an installed standard library package.

    Parameters
    ----------
    name : str
        Package name to remove.
    storage_dir : Path or None
        Custom storage directory.

    Raises
    ------
    ModuleNotInstalledError
        If the package is not installed.

    Examples
    --------
    >>> from stdlib_installer import remove_stdlib
    >>> remove_stdlib("json")
    """
    installer = _get_installer(storage_dir)
    installer.remove(name)


def remove_many_stdlib(
    names: List[str],
    storage_dir: Optional[Path] = None,
    ignore_missing: bool = True,
) -> List[str]:
    """
    Remove multiple installed packages.

    Parameters
    ----------
    names : list of str
        Package names to remove.
    storage_dir : Path or None
        Custom storage directory.
    ignore_missing : bool
        If ``True`` (default), skip packages that are not installed
        without raising an error.

    Returns
    -------
    list of str
        Names of packages that were successfully removed.

    Examples
    --------
    >>> from stdlib_installer import remove_many_stdlib
    >>> removed = remove_many_stdlib(["json", "csv", "xml"])
    >>> print(f"Removed: {removed}")
    """
    installer = _get_installer(storage_dir)
    removed: List[str] = []

    for name in names:
        try:
            installer.remove(name)
            removed.append(name)
        except StdlibInstallerError:
            if not ignore_missing:
                raise

    return removed


# ============================================================================
# Query
# ============================================================================


def is_installed_stdlib(
    name: str,
    storage_dir: Optional[Path] = None,
) -> bool:
    """
    Check if a standard library package is installed.

    Parameters
    ----------
    name : str
        Package name.
    storage_dir : Path or None
        Custom storage directory.

    Returns
    -------
    bool
        ``True`` if installed.

    Examples
    --------
    >>> from stdlib_installer import is_installed_stdlib
    >>> is_installed_stdlib("json")
    True
    """
    installer = _get_installer(storage_dir)
    return installer.is_installed(name)


def list_installed_stdlib(
    storage_dir: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """
    List all installed standard library packages.

    Parameters
    ----------
    storage_dir : Path or None
        Custom storage directory.

    Returns
    -------
    list of dict
        Each dict has keys: ``name``, ``version``, ``installed_at``,
        ``file_count``, ``dependencies``.

    Examples
    --------
    >>> from stdlib_installer import list_installed_stdlib
    >>> for pkg in list_installed_stdlib():
    ...     print(pkg["name"], pkg["version"])
    csv 3.13
    json 3.11
    """
    installer = _get_installer(storage_dir)
    return installer.list_installed()


def get_stdlib_info(
    name: str,
    storage_dir: Optional[Path] = None,
) -> Optional[Dict]:
    """
    Get detailed metadata for an installed package.

    Parameters
    ----------
    name : str
        Package name.
    storage_dir : Path or None
        Custom storage directory.

    Returns
    -------
    dict or None
        Full manifest entry, or ``None`` if not installed.

    Examples
    --------
    >>> from stdlib_installer import get_stdlib_info
    >>> info = get_stdlib_info("json")
    >>> print(info["version"])
    3.13
    >>> print(len(info["files"]), "files")
    5 files
    """
    installer = _get_installer(storage_dir)
    return installer.get_package_info(name)


# ============================================================================
# Update
# ============================================================================


def update_stdlib(
    name: str,
    version: Optional[str] = None,
    storage_dir: Optional[Path] = None,
) -> Path:
    """
    Update an installed package to a newer version.

    Parameters
    ----------
    name : str
        Package to update.
    version : str or None
        Target CPython version. Defaults to current interpreter.
    storage_dir : Path or None
        Custom storage directory.

    Returns
    -------
    Path
        Path to the updated package.

    Raises
    ------
    ModuleNotInstalledError
        If the package is not installed.

    Examples
    --------
    >>> from stdlib_installer import update_stdlib
    >>> update_stdlib("json", version="3.13")
    """
    installer = _get_installer(storage_dir)
    return installer.update(name, version=version)


def update_all_stdlib(
    storage_dir: Optional[Path] = None,
) -> Dict[str, Union[Path, Exception]]:
    """
    Update all outdated packages to the current Python version.

    Parameters
    ----------
    storage_dir : Path or None
        Custom storage directory.

    Returns
    -------
    dict
        Mapping of package name to ``Path`` (success) or ``Exception``
        (failure).

    Examples
    --------
    >>> from stdlib_installer import update_all_stdlib
    >>> results = update_all_stdlib()
    >>> for name, result in results.items():
    ...     if isinstance(result, Path):
    ...         print(f"{name}: updated")
    ...     else:
    ...         print(f"{name}: failed")
    """
    installer = _get_installer(storage_dir)
    updates = installer.check_updates()
    results: Dict[str, Union[Path, Exception]] = {}

    for entry in updates:
        name = entry["name"]
        try:
            path = installer.update(name)
            results[name] = path
        except Exception as exc:
            results[name] = exc

    return results


def check_updates_stdlib(
    storage_dir: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """
    Check which installed packages have newer versions available.

    Parameters
    ----------
    storage_dir : Path or None
        Custom storage directory.

    Returns
    -------
    list of dict
        Each dict has ``name``, ``installed_version``, and
        ``available_version``.

    Examples
    --------
    >>> from stdlib_installer import check_updates_stdlib
    >>> for update in check_updates_stdlib():
    ...     print(f"{update['name']}: {update['installed_version']} -> {update['available_version']}")
    """
    installer = _get_installer(storage_dir)
    return installer.check_updates()


# ============================================================================
# Path registration
# ============================================================================


def register_stdlib(
    source_dir: str,
    storage_dir: Optional[Path] = None,
) -> List[str]:
    """
    Register an existing local directory as a stdlib package so it
    appears in ``sys.path`` and can be imported.

    This adds the storage directory to ``sys.path`` if it is not
    already present, and optionally symlinks or notes the source
    directory so the installer is aware of it.

    Parameters
    ----------
    source_dir : str
        Path to the directory containing the stdlib modules. This
        directory should have the same structure as the CPython
        ``Lib/`` folder (e.g., ``json/__init__.py``, ``csv.py``).
    storage_dir : Path or None
        Custom storage directory. If None, uses the default.

    Returns
    -------
    list of str
        Names of packages found and registered.

    Raises
    ------
    FileNotFoundError
        If ``source_dir`` does not exist.
    NotADirectoryError
        If ``source_dir`` is not a directory.

    Examples
    --------
    >>> from stdlib_installer import register_stdlib
    >>> registered = register_stdlib(Path("/path/to/cpython/Lib"))
    >>> print(f"Registered: {registered}")
    Registered: ['json', 'csv', 'xml']
    """
    source_dir = Path(source_dir).resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {source_dir}")

    installer = _get_installer(storage_dir)

    # Add storage dir to sys.path so imports work
    if str(installer.storage_dir) not in sys.path:
        sys.path.insert(0, str(installer.storage_dir))
        logger.info(
            "Added %s to sys.path", installer.storage_dir
        )

    # Discover packages in the source directory
    registered: List[str] = []
    for item in sorted(source_dir.iterdir()):
        # Skip private/special names
        if item.name.startswith("_") or item.name.startswith("."):
            continue

        if item.is_dir() and (item / "__init__.py").exists():
            # It's a package
            registered.append(item.name)
        elif item.is_file() and item.suffix == ".py":
            # It's a single-file module
            registered.append(item.stem)

    logger.info(
        "Found %d packages in %s", len(registered), source_dir
    )
    return registered


def add_to_sys_path(
    storage_dir: Optional[Path] = None,
    position: int = 0,
) -> Path:
    """
    Ensure the stdlib storage directory is in ``sys.path``.

    This allows installed stdlib modules to be imported with the
    standard ``import`` statement.

    Parameters
    ----------
    storage_dir : Path or None
        Custom storage directory. Defaults to the installer default.
    position : int
        Position in ``sys.path`` to insert at. Default is 0 (highest
        priority, checked before standard library).

    Returns
    -------
    Path
        The path that was added to ``sys.path``.

    Examples
    --------
    >>> from stdlib_installer import add_to_sys_path, install_stdlib
    >>> add_to_sys_path()
    >>> install_stdlib("json")
    >>> import json  # Uses the installed version if not in system
    """
    installer = _get_installer(storage_dir)
    path_str = str(installer.storage_dir)

    if path_str not in sys.path:
        sys.path.insert(position, path_str)
        logger.info("Added %s to sys.path[%d]", path_str, position)
    else:
        logger.debug("%s already in sys.path", path_str)

    return installer.storage_dir


def remove_from_sys_path(
    storage_dir: Optional[Path] = None,
) -> bool:
    """
    Remove the stdlib storage directory from ``sys.path``.

    Parameters
    ----------
    storage_dir : Path or None
        Custom storage directory.

    Returns
    -------
    bool
        ``True`` if the path was removed, ``False`` if it was not
        present.

    Examples
    --------
    >>> from stdlib_installer import remove_from_sys_path
    >>> remove_from_sys_path()
    True
    """
    installer = _get_installer(storage_dir)
    path_str = str(installer.storage_dir)

    if path_str in sys.path:
        sys.path.remove(path_str)
        logger.info("Removed %s from sys.path", path_str)
        return True

    return False


# ============================================================================
# Cache management
# ============================================================================


def clear_cache(
    expired_only: bool = True,
) -> Dict:
    """
    Clear the download cache.

    Parameters
    ----------
    expired_only : bool
        If ``True`` (default), only remove expired entries. If
        ``False``, remove all entries.

    Returns
    -------
    dict
        Cache statistics after cleaning.

    Examples
    --------
    >>> from stdlib_installer import clear_cache
    >>> stats = clear_cache(expired_only=False)
    >>> print(f"Cache size: {stats['total_size_mb']} MB")
    """
    installer = _get_installer()
    cache = installer.downloader.cache

    if expired_only:
        removed = cache.clear_expired()
        logger.info("Removed %d expired cache entries", removed)
    else:
        removed = cache.clear()
        logger.info("Removed all %d cache entries", removed)

    return cache.stats()


def get_cache_stats() -> Dict:
    """
    Return current cache statistics.

    Returns
    -------
    dict
        Keys: ``entry_count``, ``total_size_bytes``, ``total_size_mb``,
        ``cache_dir``, ``ttl_seconds``.

    Examples
    --------
    >>> from stdlib_installer import get_cache_stats
    >>> stats = get_cache_stats()
    >>> print(f"{stats['entry_count']} entries, {stats['total_size_mb']} MB")
    """
    installer = _get_installer()
    return installer.downloader.cache.stats()
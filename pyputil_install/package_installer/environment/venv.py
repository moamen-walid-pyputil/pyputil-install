#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Virtual environment creation and management.

This module provides utilities for creating, detecting, activating,
and managing Python virtual environments. It wraps Python's built-in
``venv`` module with additional features for package installation
within isolated environments.

Classes
-------
EnvironmentInfo
    Information about a Python environment.
VirtualEnvironment
    Manages a virtual environment lifecycle.

Examples
--------
>>> venv = VirtualEnvironment("/tmp/my-venv")
>>> venv.create()
>>> venv.is_active()
True
>>> venv.install_package("requests")
True
"""

import os
import sys
import shutil
import logging
import subprocess
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime

from ..exceptions import (
    PackageInstallerError,
    PackageInstallError,
)

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentInfo:
    """
    Information about a Python environment.

    Parameters
    ----------
    path : Path
        Path to the environment root directory.
    python_version : str
        Python version string (e.g., ``3.10.12``).
    python_executable : Path
        Path to the Python executable.
    pip_executable : Path
        Path to the pip executable.
    site_packages : Path
        Path to the site-packages directory.
    is_virtual_env : bool
        Whether this is a virtual environment.
    is_active : bool
        Whether this environment is currently active.
    created_at : str, optional
        ISO format timestamp of creation time.
    packages : list of str, optional
        List of installed package names.

    Examples
    --------
    >>> info = EnvironmentInfo(
    ...     path=Path("/tmp/venv"),
    ...     python_version="3.10.12",
    ...     python_executable=Path("/tmp/venv/bin/python"),
    ...     pip_executable=Path("/tmp/venv/bin/pip"),
    ...     site_packages=Path("/tmp/venv/lib/python3.10/site-packages"),
    ...     is_virtual_env=True,
    ...     is_active=False,
    ... )
    >>> info.python_version
    '3.10.12'
    """

    path: Path
    python_version: str
    python_executable: Path
    pip_executable: Path
    site_packages: Path
    is_virtual_env: bool
    is_active: bool
    created_at: Optional[str] = None
    packages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert environment info to a dictionary.

        Returns
        -------
        dict
            Dictionary with all environment information fields.
        """
        return {
            "path": str(self.path),
            "python_version": self.python_version,
            "python_executable": str(self.python_executable),
            "pip_executable": str(self.pip_executable),
            "site_packages": str(self.site_packages),
            "is_virtual_env": self.is_virtual_env,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "package_count": len(self.packages),
        }


class VirtualEnvironment:
    """
    Manages a Python virtual environment lifecycle.

    This class provides methods to create, activate, deactivate,
    and destroy virtual environments. It also supports installing
    packages within the environment using pip.

    Parameters
    ----------
    path : str or Path
        Path where the virtual environment will be or is located.
    python_executable : str or Path, optional
        Python interpreter to use for creating the environment.
        If None, uses the current interpreter (``sys.executable``).
    system_site_packages : bool, default=False
        If True, give the virtual environment access to system
        site-packages.
    clear : bool, default=False
        If True, clear the target directory before creating
        the environment.
    symlinks : bool, default=False
        If True, use symlinks instead of copies when possible.
    upgrade : bool, default=False
        If True, upgrade the environment's pip to the latest version
        after creation.
    with_pip : bool, default=True
        If True, install pip in the virtual environment.

    Attributes
    ----------
    path : Path
        Path to the virtual environment.
    env_dir : Path
        Alias for path.
    bin_dir : Path
        Path to the bin/Scripts directory.
    python_path : Path
        Path to the Python executable in the environment.
    pip_path : Path
        Path to the pip executable in the environment.

    Notes
    -----
    On Windows, executables are located in ``Scripts\\`` instead of
    ``bin/``. This class handles platform differences automatically.

    Creating a virtual environment with ``system_site_packages=True``
    allows access to globally installed packages but may cause
    dependency conflicts.

    Examples
    --------
    >>> venv = VirtualEnvironment("/tmp/test-env")
    >>> venv.create()
    >>> venv.python_path.exists()
    True
    >>> venv.install_package("requests")
    True
    >>> venv.destroy()
    True

    With custom Python::

    >>> venv = VirtualEnvironment(
    ...     "/tmp/py39-env",
    ...     python_executable="/usr/bin/python3.9",
    ... )
    """

    def __init__(
        self,
        path: Union[str, Path],
        python_executable: Optional[Union[str, Path]] = None,
        system_site_packages: bool = False,
        clear: bool = False,
        symlinks: bool = False,
        upgrade: bool = False,
        with_pip: bool = True,
    ) -> None:
        self.path = Path(path).resolve()
        self.env_dir = self.path

        if python_executable is None:
            self._source_python = Path(sys.executable)
        else:
            self._source_python = Path(python_executable)

        self._system_site_packages = system_site_packages
        self._clear = clear
        self._symlinks = symlinks
        self._upgrade = upgrade
        self._with_pip = with_pip

        self._bin_dir_name = "Scripts" if sys.platform == "win32" else "bin"

        self.bin_dir = self.path / self._bin_dir_name
        self.python_path = self.bin_dir / (
            "python.exe" if sys.platform == "win32" else "python"
        )
        self.pip_path = self.bin_dir / (
            "pip.exe" if sys.platform == "win32" else "pip"
        )

        logger.debug(
            f"VirtualEnvironment initialized at {self.path}"
        )

    def exists(self) -> bool:
        """
        Check if the virtual environment directory exists.

        Returns
        -------
        bool
            True if the environment directory exists and contains
            a Python executable.

        Examples
        --------
        >>> venv = VirtualEnvironment("/tmp/check-env")
        >>> venv.exists()
        False
        """
        return self.path.exists() and self.python_path.exists()

    def create(
        self,
        prompt: Optional[str] = None,
    ) -> bool:
        """
        Create the virtual environment.

        Parameters
        ----------
        prompt : str, optional
            Custom prompt prefix for the activated environment.
            If None, uses the directory name.

        Returns
        -------
        bool
            True if the environment was created successfully.

        Raises
        ------
        PackageInstallerError
            If environment creation fails or the path is not writable.

        Notes
        -----
        If the environment already exists and ``clear`` was not
        specified in the constructor, this method does nothing
        and returns True.

        Examples
        --------
        >>> venv = VirtualEnvironment("/tmp/new-env")
        >>> venv.create(prompt="my-project")
        True
        """
        if self.exists():
            if not self._clear:
                logger.info(
                    f"Virtual environment already exists at {self.path}"
                )
                return True
            else:
                logger.info(
                    f"Clearing existing environment at {self.path}"
                )
                self.destroy()

        self.path.mkdir(parents=True, exist_ok=True)

        try:
            import venv

            builder_args = {
                "system_site_packages": self._system_site_packages,
                "clear": self._clear,
                "symlinks": self._symlinks,
                "with_pip": self._with_pip,
            }

            if prompt is not None:
                builder_args["prompt"] = prompt

            builder = venv.EnvBuilder(**builder_args)
            builder.create(self.path)

            logger.info(
                f"Created virtual environment at {self.path}"
            )

        except Exception as e:
            raise PackageInstallerError(
                f"Failed to create virtual environment at "
                f"{self.path}: {e}"
            ) from e

        if self._upgrade and self._with_pip:
            self._upgrade_pip()

        return True

    def _upgrade_pip(self) -> bool:
        """
        Upgrade pip to the latest version in the environment.

        Returns
        -------
        bool
            True if upgrade succeeded.

        Notes
        -----
        This is called automatically if ``upgrade=True`` was
        specified in the constructor.
        """
        try:
            result = subprocess.run(
                [str(self.python_path), "-m", "pip", "install", "--upgrade", "pip"],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )

            if result.returncode != 0:
                logger.warning(
                    f"Failed to upgrade pip: {result.stderr.strip()}"
                )
                return False

            logger.info("Upgraded pip to latest version")
            return True

        except subprocess.TimeoutExpired:
            logger.warning("pip upgrade timed out")
            return False
        except Exception as e:
            logger.warning(f"pip upgrade failed: {e}")
            return False

    def is_active(self) -> bool:
        """
        Check if this virtual environment is the currently active one.

        Returns
        -------
        bool
            True if this environment's Python executable matches
            the current ``sys.executable``.

        Notes
        -----
        This checks if the current process is running inside this
        virtual environment by comparing executable paths.

        Examples
        --------
        >>> venv = VirtualEnvironment("/tmp/active-check")
        >>> venv.is_active()
        False
        """
        try:
            return self.python_path.resolve() == Path(sys.executable).resolve()
        except (OSError, ValueError):
            return False

    def get_python_version(self) -> str:
        """
        Get the Python version of the environment.

        Returns
        -------
        str
            Python version string (e.g., ``3.10.12``).

        Raises
        ------
        PackageInstallerError
            If the environment does not exist or Python cannot
            be executed.

        Examples
        --------
        >>> venv = VirtualEnvironment("/tmp/version-check")
        >>> venv.create()
        >>> version = venv.get_python_version()
        >>> version.startswith("3.")
        True
        """
        if not self.exists():
            raise PackageInstallerError(
                f"Virtual environment does not exist at {self.path}"
            )

        try:
            result = subprocess.run(
                [
                    str(self.python_path),
                    "-c",
                    "import sys; print(f'{sys.version_info.major}."
                    f"{sys.version_info.minor}."
                    f"{sys.version_info.micro}')",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise PackageInstallerError(
                f"Failed to get Python version: {e.stderr.strip()}"
            ) from e
        except subprocess.TimeoutExpired:
            raise PackageInstallerError(
                "Timed out getting Python version"
            )

    def get_pip_version(self) -> str:
        """
        Get the pip version installed in the environment.

        Returns
        -------
        str
            pip version string (e.g., ``23.1.2``).

        Raises
        ------
        PackageInstallerError
            If pip is not installed or cannot be executed.
        """
        if not self.exists():
            raise PackageInstallerError(
                f"Virtual environment does not exist at {self.path}"
            )

        try:
            result = subprocess.run(
                [str(self.pip_path), "--version"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            version_str = result.stdout.strip()
            parts = version_str.split()
            if len(parts) >= 2:
                return parts[1]
            return version_str
        except subprocess.CalledProcessError as e:
            raise PackageInstallerError(
                f"Failed to get pip version: {e.stderr.strip()}"
            ) from e

    def install_package(
        self,
        package_name: str,
        version: Optional[str] = None,
        upgrade: bool = False,
        extra_args: Optional[List[str]] = None,
    ) -> bool:
        """
        Install a package inside the virtual environment.

        Parameters
        ----------
        package_name : str
            Name of the package to install.
        version : str, optional
            Specific version to install (e.g., ``1.2.3``).
        upgrade : bool, default=False
            If True, upgrade the package if already installed.
        extra_args : list of str, optional
            Additional arguments to pass to pip.

        Returns
        -------
        bool
            True if installation succeeded.

        Raises
        ------
        PackageInstallError
            If installation fails.
        PackageInstallerError
            If the environment does not exist.

        Examples
        --------
        >>> venv = VirtualEnvironment("/tmp/install-test")
        >>> venv.create()
        >>> venv.install_package("requests", version="2.28.0")
        True
        """
        if not self.exists():
            raise PackageInstallerError(
                f"Virtual environment does not exist at {self.path}"
            )

        pkg_spec = package_name
        if version:
            pkg_spec = f"{package_name}=={version}"

        cmd = [
            str(self.python_path),
            "-m",
            "pip",
            "install",
            pkg_spec,
        ]

        if upgrade:
            cmd.append("--upgrade")

        if extra_args:
            cmd.extend(extra_args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or "Unknown error"
                raise PackageInstallError(
                    package_name,
                    message=f"Installation failed: {error_msg}",
                    stderr=error_msg,
                )

            logger.info(
                f"Installed {pkg_spec} in environment at {self.path}"
            )
            return True

        except subprocess.TimeoutExpired:
            raise PackageInstallError(
                package_name,
                message="Installation timed out",
            )
        except PackageInstallError:
            raise
        except Exception as e:
            raise PackageInstallError(
                package_name,
                message=f"Unexpected error: {e}",
            ) from e

    def uninstall_package(
        self,
        package_name: str,
        confirm: bool = False,
    ) -> bool:
        """
        Uninstall a package from the virtual environment.

        Parameters
        ----------
        package_name : str
            Name of the package to uninstall.
        confirm : bool, default=False
            If False, skip confirmation prompt (``-y`` flag).

        Returns
        -------
        bool
            True if uninstallation succeeded.

        Raises
        ------
        PackageInstallerError
            If uninstallation fails.
        """
        if not self.exists():
            raise PackageInstallerError(
                f"Virtual environment does not exist at {self.path}"
            )

        cmd = [
            str(self.python_path),
            "-m",
            "pip",
            "uninstall",
            package_name,
        ]

        if not confirm:
            cmd.append("-y")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )

            if result.returncode != 0:
                logger.warning(
                    f"Uninstallation warning: {result.stderr.strip()}"
                )

            return True

        except Exception as e:
            raise PackageInstallerError(
                f"Failed to uninstall {package_name}: {e}"
            ) from e

    def get_installed_packages(self) -> List[str]:
        """
        Get a list of packages installed in the environment.

        Returns
        -------
        list of str
            List of package names (lowercase).

        Raises
        ------
        PackageInstallerError
            If the environment does not exist or pip list fails.

        Examples
        --------
        >>> venv = VirtualEnvironment("/tmp/list-test")
        >>> venv.create()
        >>> venv.install_package("six")
        >>> "six" in venv.get_installed_packages()
        True
        """
        if not self.exists():
            raise PackageInstallerError(
                f"Virtual environment does not exist at {self.path}"
            )

        try:
            result = subprocess.run(
                [str(self.python_path), "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )

            packages = json.loads(result.stdout)
            return [pkg["name"].lower() for pkg in packages]

        except json.JSONDecodeError:
            logger.error("Failed to parse pip list output")
            return []
        except Exception as e:
            raise PackageInstallerError(
                f"Failed to list installed packages: {e}"
            ) from e

    def install_requirements(
        self,
        requirements_file: Union[str, Path],
    ) -> bool:
        """
        Install packages from a requirements file.

        Parameters
        ----------
        requirements_file : str or Path
            Path to the requirements file.

        Returns
        -------
        bool
            True if all packages were installed.

        Raises
        ------
        PackageInstallError
            If installation fails.
        """
        if not self.exists():
            raise PackageInstallerError(
                f"Virtual environment does not exist at {self.path}"
            )

        req_path = Path(requirements_file)
        if not req_path.exists():
            raise PackageInstallError(
                "requirements",
                message=f"Requirements file not found: {req_path}",
            )

        cmd = [
            str(self.python_path),
            "-m",
            "pip",
            "install",
            "-r",
            str(req_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )

            if result.returncode != 0:
                raise PackageInstallError(
                    "requirements",
                    message=f"Installation failed: {result.stderr.strip()}",
                    stderr=result.stderr.strip(),
                )

            logger.info(
                f"Installed requirements from {req_path} "
                f"in environment at {self.path}"
            )
            return True

        except PackageInstallError:
            raise
        except Exception as e:
            raise PackageInstallError(
                "requirements",
                message=f"Failed to install requirements: {e}",
            ) from e

    def freeze(self, output_file: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a pip freeze output for the environment.

        Parameters
        ----------
        output_file : str or Path, optional
            If provided, write the freeze output to this file.

        Returns
        -------
        str
            The pip freeze output string.

        Raises
        ------
        PackageInstallerError
            If the environment does not exist or freeze fails.

        Examples
        --------
        >>> venv = VirtualEnvironment("/tmp/freeze-test")
        >>> venv.create()
        >>> freeze_output = venv.freeze()
        >>> "pip==" in freeze_output
        True
        """
        if not self.exists():
            raise PackageInstallerError(
                f"Virtual environment does not exist at {self.path}"
            )

        try:
            result = subprocess.run(
                [str(self.python_path), "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )

            output = result.stdout.strip()

            if output_file is not None:
                out_path = Path(output_file)
                out_path.write_text(output + "\n", encoding="utf-8")
                logger.info(f"Freeze output written to {out_path}")

            return output

        except Exception as e:
            raise PackageInstallerError(
                f"Failed to freeze environment: {e}"
            ) from e

    def get_info(self) -> EnvironmentInfo:
        """
        Get detailed information about the environment.

        Returns
        -------
        EnvironmentInfo
            Dataclass with environment information.

        Notes
        -----
        The ``is_active`` field indicates whether this environment
        is the one currently running the process.

        Examples
        --------
        >>> venv = VirtualEnvironment("/tmp/info-test")
        >>> venv.create()
        >>> info = venv.get_info()
        >>> info.is_virtual_env
        True
        """
        python_version = "unknown"
        try:
            python_version = self.get_python_version()
        except PackageInstallerError:
            pass

        packages = []
        try:
            packages = self.get_installed_packages()
        except PackageInstallerError:
            pass

        site_packages = self.path / "lib" / f"python{python_version}" / "site-packages"
        if sys.platform == "win32":
            site_packages = self.path / "Lib" / "site-packages"

        created_at = None
        if self.path.exists():
            try:
                created_at = datetime.fromtimestamp(
                    self.path.stat().st_ctime
                ).isoformat()
            except OSError:
                pass

        return EnvironmentInfo(
            path=self.path,
            python_version=python_version,
            python_executable=self.python_path,
            pip_executable=self.pip_path,
            site_packages=site_packages,
            is_virtual_env=self.exists(),
            is_active=self.is_active(),
            created_at=created_at,
            packages=packages,
        )

    def destroy(self) -> bool:
        """
        Remove the virtual environment directory entirely.

        Returns
        -------
        bool
            True if the environment was destroyed, False if it
            did not exist.

        Raises
        ------
        PackageInstallerError
            If the directory exists but cannot be removed.

        Warnings
        --------
        This operation is irreversible. All installed packages
        and configuration in the environment are permanently
        deleted.

        Examples
        --------
        >>> venv = VirtualEnvironment("/tmp/destroy-test")
        >>> venv.create()
        >>> venv.destroy()
        True
        >>> venv.exists()
        False
        """
        if not self.path.exists():
            return False

        try:
            shutil.rmtree(self.path)
            logger.info(f"Destroyed virtual environment at {self.path}")
            return True
        except OSError as e:
            raise PackageInstallerError(
                f"Failed to destroy virtual environment at "
                f"{self.path}: {e}"
            ) from e

    def run_module(
        self,
        module: str,
        args: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> subprocess.CompletedProcess:
        """
        Run a Python module inside the virtual environment.

        Parameters
        ----------
        module : str
            Module name to run (e.g., ``pip``, ``http.server``).
        args : list of str, optional
            Arguments to pass to the module.
        timeout : float, optional
            Timeout in seconds for the subprocess.

        Returns
        -------
        subprocess.CompletedProcess
            The subprocess result.

        Raises
        ------
        PackageInstallerError
            If the environment does not exist or execution fails.

        Examples
        --------
        >>> venv = VirtualEnvironment("/tmp/run-test")
        >>> venv.create()
        >>> result = venv.run_module("json.tool", args=["--help"])
        >>> result.returncode
        0
        """
        if not self.exists():
            raise PackageInstallerError(
                f"Virtual environment does not exist at {self.path}"
            )

        cmd = [str(self.python_path), "-m", module]
        if args:
            cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return result
        except subprocess.TimeoutExpired as e:
            raise PackageInstallerError(
                f"Command timed out: {' '.join(cmd)}"
            ) from e
        except Exception as e:
            raise PackageInstallerError(
                f"Failed to run module {module}: {e}"
            ) from e

    def run_script(
        self,
        script_path: Union[str, Path],
        args: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> subprocess.CompletedProcess:
        """
        Run a Python script inside the virtual environment.

        Parameters
        ----------
        script_path : str or Path
            Path to the Python script.
        args : list of str, optional
            Arguments to pass to the script.
        timeout : float, optional
            Timeout in seconds.

        Returns
        -------
        subprocess.CompletedProcess
            The subprocess result.

        Raises
        ------
        PackageInstallerError
            If the script does not exist or execution fails.

        Examples
        --------
        >>> venv = VirtualEnvironment("/tmp/script-test")
        >>> venv.create()
        >>> # Write a script and run it
        """
        script = Path(script_path)
        if not script.exists():
            raise PackageInstallerError(
                f"Script not found: {script}"
            )

        cmd = [str(self.python_path), str(script)]
        if args:
            cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return result
        except subprocess.TimeoutExpired as e:
            raise PackageInstallerError(
                f"Script timed out: {script}"
            ) from e
        except Exception as e:
            raise PackageInstallerError(
                f"Failed to run script {script}: {e}"
            ) from e

    def check_package(self, package_name: str) -> bool:
        """
        Check if a package is installed in the environment.

        Parameters
        ----------
        package_name : str
            Name of the package to check.

        Returns
        -------
        bool
            True if the package is installed.

        Examples
        --------
        >>> venv = VirtualEnvironment("/tmp/check-pkg")
        >>> venv.create()
        >>> venv.install_package("click")
        >>> venv.check_package("click")
        True
        """
        try:
            result = subprocess.run(
                [
                    str(self.python_path),
                    "-c",
                    f"import {package_name.replace('-', '_')}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def __repr__(self) -> str:
        """String representation of the environment."""
        status = "active" if self.is_active() else "inactive"
        exists = "exists" if self.exists() else "not created"
        return (
            f"VirtualEnvironment("
            f"path={self.path}, "
            f"{status}, "
            f"{exists})"
        )

    def __enter__(self) -> "VirtualEnvironment":
        """
        Context manager entry.

        Creates the environment if it does not exist.

        Returns
        -------
        VirtualEnvironment
            Self reference.
        """
        if not self.exists():
            self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Context manager exit.

        Does not destroy the environment. Use ``destroy()``
        explicitly if cleanup is needed.
        """
        pass
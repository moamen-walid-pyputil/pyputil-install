"""
Named toolchain environments for project-specific compiler sets.

An environment is a named collection of compiler toolchains at
specific versions. It allows users to define, activate, and share
consistent compiler configurations across projects and machines.

Design
------
Environments are stored as JSON files in `{root}/environments/`.
Each environment is a simple mapping of compiler name → version.

    {root}/
        environments/
            cpp20.json    → {"gcc": "14.2.0-2", "cmake": "3.28.1"}
            embedded.json → {"arm-gnu": "13.2.0", "zig": "0.11.0"}
            default.json  → {"gcc": "14.2.0-2", "clang": "18.1.0"}

Activating an environment calls `activation.activate()` for each
toolchain in the environment, in definition order.

Deactivating an environment calls `activation.deactivate()` for
each toolchain in reverse order, restoring the previous PATH state.

Usage
-----
    from pyputil_install.compiler_installer.environments import EnvironmentStore

    store = EnvironmentStore()

    # Create an environment
    store.create("cpp20", {"gcc": "14.2.0-2", "zig": "0.11.0"})

    # Activate it in the current process
    store.activate("cpp20")

    # Deactivate it
    store.deactivate("cpp20")

    # Generate shell activation script
    script = store.shell_activate_script("cpp20", shell="bash")
    print(script)

    # List all environments
    for name in store.list_all():
        print(name, store.get(name))

Warnings
--------
- Environments do NOT validate that the referenced toolchains are
  installed. Use `store.validate()` to check.
- Activating an environment that references a non-existent toolchain
  will raise FileNotFoundError. Install missing toolchains first.
- Environment files are plain JSON. They are NOT encrypted.
- The "default" environment name is reserved. It is activated
  automatically by `install.py` when no specific environment is given.
- Concurrent writes to the same environment file are NOT safe.
  Use external locking if multiple processes manage environments.

User Instructions
-----------------
- Create an environment per project to pin compiler versions.
- Use `store.export_env(name)` to generate a portable JSON file
  that can be shared via version control.
- Use `store.import_env(path)` to load an environment file from
  any location on disk.
- The activation stack is managed by `activation.py`. Environments
  push onto the same stack — you can mix environment and manual
  activation calls.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from .layouts import get_install_root
from .manifests import ManifestStore

logger = logging.getLogger(__name__)

# Current environment schema version.
ENV_SCHEMA_VERSION = 1


# ============================================================================
# Environment definition
# ============================================================================

@dataclass(frozen=True)
class Environment:
    """
    Immutable definition of a named toolchain environment.

    Attributes
    ----------
    name : str
        Environment name. Must be filesystem-safe (alphanumeric,
        hyphens, underscores). Reserved names: "default".
    compilers : Dict[str, str]
        Mapping of compiler name → version string.
        Example: {"gcc": "14.2.0-2", "zig": "0.11.0"}.
    created_at : str
        ISO 8601 timestamp of creation (UTC).
    updated_at : str
        ISO 8601 timestamp of last update (UTC).
    description : str
        Optional human-readable description.
    schema_version : int
        Schema version for forward compatibility. Currently 1.

    Methods
    -------
    to_dict() -> Dict
        Serialize to JSON-compatible dictionary.
    from_dict(data: Dict) -> Environment
        Deserialize from JSON-compatible dictionary.
    """

    name: str
    compilers: Dict[str, str]
    created_at: str = ""
    updated_at: str = ""
    description: str = ""
    schema_version: int = ENV_SCHEMA_VERSION

    def to_dict(self) -> Dict:
        """
        Serialize to a JSON-compatible dictionary.

        Returns
        -------
        Dict
            Dictionary with all environment fields.
        """
        return {
            "name": self.name,
            "compilers": self.compilers,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "description": self.description,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Environment":
        """
        Deserialize from a JSON-compatible dictionary.

        Parameters
        ----------
        data : Dict
            Raw dictionary from JSON.

        Returns
        -------
        Environment
            Deserialized environment.
        """
        return cls(
            name=data.get("name", ""),
            compilers=data.get("compilers", {}),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            description=data.get("description", ""),
            schema_version=data.get("schema_version", 1),
        )

    def compiler_list(self) -> List[Tuple[str, str]]:
        """
        Return compilers as a list of (name, version) tuples.

        Returns
        -------
        List[Tuple[str, str]]
            Compiler entries in insertion order.
        """
        return list(self.compilers.items())

    def has_compiler(self, compiler: str) -> bool:
        """
        Check if a compiler is defined in this environment.

        Parameters
        ----------
        compiler : str
            Compiler name.

        Returns
        -------
        bool
            True if the compiler is in the environment.
        """
        return compiler in self.compilers

    def get_version(self, compiler: str) -> Optional[str]:
        """
        Get the version of a compiler in this environment.

        Parameters
        ----------
        compiler : str
            Compiler name.

        Returns
        -------
        Optional[str]
            Version string, or None if not in environment.
        """
        return self.compilers.get(compiler)


# ============================================================================
# EnvironmentStore — CRUD for environment files
# ============================================================================

class EnvironmentStore:
    """
    Manages named toolchain environment files on disk.

    Parameters
    ----------
    install_root : Optional[Path]
        Root directory for toolchain installations.
        If None, `get_install_root()` is called.

    Attributes
    ----------
    env_dir : Path
        Path to the `environments` directory.
    """

    def __init__(self, install_root: Optional[Path] = None) -> None:
        self._root = install_root or get_install_root()
        self.env_dir = self._root / "environments"

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _env_path(self, name: str) -> Path:
        """
        Return the path to an environment file.

        Parameters
        ----------
        name : str
            Environment name.

        Returns
        -------
        Path
            Path to `{env_dir}/{name}.json`.
        """
        return self.env_dir / f"{name}.json"

    @staticmethod
    def _validate_name(name: str) -> None:
        """
        Validate an environment name.

        Parameters
        ----------
        name : str
            Proposed environment name.

        Raises
        ------
        ValueError
            If the name contains invalid characters or is empty.
        """
        if not name:
            raise ValueError("Environment name must not be empty")
        if not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                f"Environment name contains invalid characters: {name!r}. "
                f"Use only letters, numbers, hyphens, and underscores."
            )

    # ------------------------------------------------------------------
    # Create / Update
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        compilers: Dict[str, str],
        description: str = "",
        overwrite: bool = False,
    ) -> Environment:
        """
        Create a new environment or overwrite an existing one.

        Parameters
        ----------
        name : str
            Environment name. Must be filesystem-safe.
        compilers : Dict[str, str]
            Mapping of compiler name → version.
        description : str
            Optional description.
        overwrite : bool
            If True, overwrites an existing environment of the same name.
            If False, raises FileExistsError if the environment exists.

        Returns
        -------
        Environment
            The created environment object.

        Raises
        ------
        ValueError
            If the name is invalid.
        FileExistsError
            If the environment exists and overwrite is False.
        """
        self._validate_name(name)

        env_path = self._env_path(name)
        if env_path.exists() and not overwrite:
            raise FileExistsError(
                f"Environment {name!r} already exists. "
                f"Use overwrite=True to replace it."
            )

        now = datetime.now(timezone.utc).isoformat()

        # If updating, preserve original created_at
        created_at = now
        if env_path.exists():
            existing = self.get(name)
            if existing:
                created_at = existing.created_at

        env = Environment(
            name=name,
            compilers=compilers,
            created_at=created_at,
            updated_at=now,
            description=description,
        )

        self._write(env)
        logger.info("Created environment %r with %d compiler(s)", name, len(compilers))
        return env

    def update(
        self,
        name: str,
        compilers: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> Environment:
        """
        Update an existing environment.

        Parameters
        ----------
        name : str
            Environment name.
        compilers : Optional[Dict[str, str]]
            New compiler mapping. If None, existing compilers are kept.
        description : Optional[str]
            New description. If None, existing description is kept.

        Returns
        -------
        Environment
            The updated environment.

        Raises
        ------
        FileNotFoundError
            If the environment does not exist.
        """
        existing = self.get_or_raise(name)
        now = datetime.now(timezone.utc).isoformat()

        env = Environment(
            name=name,
            compilers=compilers if compilers is not None else existing.compilers,
            created_at=existing.created_at,
            updated_at=now,
            description=description if description is not None else existing.description,
        )

        self._write(env)
        logger.info("Updated environment %r", name)
        return env

    def add_compiler(self, name: str, compiler: str, version: str) -> Environment:
        """
        Add or update a single compiler in an environment.

        Parameters
        ----------
        name : str
            Environment name.
        compiler : str
            Compiler name to add.
        version : str
            Version string.

        Returns
        -------
        Environment
            Updated environment.
        """
        existing = self.get_or_raise(name)
        new_compilers = dict(existing.compilers)
        new_compilers[compiler] = version
        return self.update(name, compilers=new_compilers)

    def remove_compiler(self, name: str, compiler: str) -> Environment:
        """
        Remove a compiler from an environment.

        Parameters
        ----------
        name : str
            Environment name.
        compiler : str
            Compiler name to remove.

        Returns
        -------
        Environment
            Updated environment.

        Raises
        ------
        ValueError
            If the compiler is not in the environment.
        """
        existing = self.get_or_raise(name)
        if compiler not in existing.compilers:
            raise ValueError(
                f"Compiler {compiler!r} is not in environment {name!r}"
            )
        new_compilers = dict(existing.compilers)
        del new_compilers[compiler]
        return self.update(name, compilers=new_compilers)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[Environment]:
        """
        Load an environment by name.

        Parameters
        ----------
        name : str
            Environment name.

        Returns
        -------
        Optional[Environment]
            The loaded environment, or None if it does not exist
            or is corrupted.
        """
        env_path = self._env_path(name)
        if not env_path.is_file():
            return None

        try:
            data = json.loads(env_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupted environment %r: %s", name, exc)
            return None

        return Environment.from_dict(data)

    def get_or_raise(self, name: str) -> Environment:
        """
        Load an environment, raising if not found.

        Parameters
        ----------
        name : str
            Environment name.

        Returns
        -------
        Environment

        Raises
        ------
        FileNotFoundError
            If the environment does not exist.
        ValueError
            If the environment file is corrupted.
        """
        env = self.get(name)
        if env is None:
            env_path = self._env_path(name)
            if not env_path.is_file():
                raise FileNotFoundError(f"Environment {name!r} not found")
            raise ValueError(f"Environment {name!r} is corrupted")
        return env

    def get_default(self) -> Optional[Environment]:
        """
        Load the default environment.

        Returns
        -------
        Optional[Environment]
            The default environment, or None.
        """
        return self.get("default")

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, name: str) -> bool:
        """
        Delete an environment file.

        Parameters
        ----------
        name : str
            Environment name.

        Returns
        -------
        bool
            True if deleted, False if it did not exist.
        """
        env_path = self._env_path(name)
        if not env_path.is_file():
            return False

        try:
            env_path.unlink()
            logger.info("Deleted environment %r", name)
            return True
        except OSError as exc:
            logger.warning("Failed to delete environment %r: %s", name, exc)
            return False

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_all(self) -> List[str]:
        """
        List all environment names.

        Returns
        -------
        List[str]
            Sorted list of environment names.
        """
        if not self.env_dir.exists():
            return []

        names = []
        for entry in sorted(self.env_dir.iterdir()):
            if entry.suffix == ".json":
                names.append(entry.stem)

        return names

    def list_all_environments(self) -> List[Environment]:
        """
        List all environments with full data.

        Returns
        -------
        List[Environment]
            All valid environments, sorted by name.
        """
        environments = []
        for name in self.list_all():
            env = self.get(name)
            if env is not None:
                environments.append(env)
        return environments

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def export_env(self, name: str, dest: Path) -> bool:
        """
        Export an environment to a portable JSON file.

        Parameters
        ----------
        name : str
            Environment name.
        dest : Path
            Destination file path.

        Returns
        -------
        bool
            True on success.

        Raises
        ------
        FileNotFoundError
            If the environment does not exist.
        """
        env = self.get_or_raise(name)

        try:
            dest.write_text(
                json.dumps(env.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Exported environment %r to %s", name, dest)
            return True
        except OSError as exc:
            logger.error("Failed to export environment %r: %s", name, exc)
            return False

    def import_env(self, source: Path, overwrite: bool = False) -> Environment:
        """
        Import an environment from a JSON file.

        Parameters
        ----------
        source : Path
            Path to a JSON environment file.
        overwrite : bool
            If True, overwrites an existing environment of the same name.

        Returns
        -------
        Environment
            The imported environment.

        Raises
        ------
        FileNotFoundError
            If the source file does not exist.
        ValueError
            If the source file is not valid JSON or missing required fields.
        FileExistsError
            If the environment name already exists and overwrite is False.
        """
        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")

        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {source}: {exc}")

        if "name" not in data or "compilers" not in data:
            raise ValueError(
                f"Environment file missing required fields 'name' and 'compilers': {source}"
            )

        return self.create(
            name=data["name"],
            compilers=data["compilers"],
            description=data.get("description", ""),
            overwrite=overwrite,
        )

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def activate(self, name: str) -> int:
        """
        Activate all compilers in an environment in the current process.

        Calls `activation.activate()` for each compiler in the
        environment, in definition order.

        Parameters
        ----------
        name : str
            Environment name.

        Returns
        -------
        int
            Number of compilers activated.

        Raises
        ------
        FileNotFoundError
            If the environment does not exist.
        """
        from toolforge.installer.activation import activate

        env = self.get_or_raise(name)
        count = 0

        for compiler, version in env.compiler_list():
            if activate(compiler, version, self._root):
                count += 1

        logger.info("Activated environment %r: %d compiler(s)", name, count)
        return count

    def deactivate(self, name: str) -> int:
        """
        Deactivate all compilers in an environment.

        Calls `activation.deactivate()` for each compiler in
        reverse definition order.

        Parameters
        ----------
        name : str
            Environment name.

        Returns
        -------
        int
            Number of compilers deactivated.
        """
        from toolforge.installer.activation import deactivate

        env = self.get_or_raise(name)
        count = 0

        # Deactivate in reverse order
        for compiler, version in reversed(env.compiler_list()):
            result = deactivate()
            if result is not None:
                count += 1

        logger.info("Deactivated environment %r: %d compiler(s)", name, count)
        return count

    def shell_activate_script(
        self,
        name: str,
        shell: str = "bash",
    ) -> str:
        """
        Generate shell commands to activate an entire environment.

        Parameters
        ----------
        name : str
            Environment name.
        shell : str
            Target shell: "bash", "zsh", "fish", "cmd", "powershell".

        Returns
        -------
        str
            Shell commands to activate all compilers in the environment.

        Raises
        ------
        FileNotFoundError
            If the environment does not exist.
        """
        from toolforge.installer.activation import shell_activate_script

        env = self.get_or_raise(name)
        lines = [f"# Activating environment: {name}"]
        if env.description:
            lines.append(f"# {env.description}")
        lines.append("")

        for compiler, version in env.compiler_list():
            lines.append(f"# {compiler}@{version}")
            lines.append(
                shell_activate_script(compiler, version, self._root, shell)
            )
            lines.append("")

        return "\n".join(lines)

    def shell_deactivate_script(self, shell: str = "bash") -> str:
        """
        Generate shell commands to deactivate the current environment.

        Parameters
        ----------
        shell : str
            Target shell.

        Returns
        -------
        str
            Shell commands to deactivate.
        """
        from toolforge.installer.activation import shell_deactivate_script

        return shell_deactivate_script(shell)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, name: str) -> List[str]:
        """
        Check that all compilers in an environment are installed.

        Parameters
        ----------
        name : str
            Environment name.

        Returns
        -------
        List[str]
            Human-readable warnings for missing toolchains.
            Empty list if all are installed.
        """
        env = self.get_or_raise(name)
        store = ManifestStore(self._root)
        warnings = []

        for compiler, version in env.compiler_list():
            if not store.exists(compiler, version):
                warnings.append(
                    f"Compiler {compiler}@{version} is not installed. "
                    f"Install it first: install {compiler} {version}"
                )

        return warnings

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, env: Environment) -> None:
        """
        Write an environment to disk atomically.

        Parameters
        ----------
        env : Environment
            The environment to persist.
        """
        self.env_dir.mkdir(parents=True, exist_ok=True)
        env_path = self._env_path(env.name)

        fd, temp_path = tempfile.mkstemp(
            suffix=".json",
            prefix=".tmp-",
            dir=str(self.env_dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(
                    env.to_dict(),
                    f,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
            os.replace(temp_path, env_path)
        except Exception:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            raise
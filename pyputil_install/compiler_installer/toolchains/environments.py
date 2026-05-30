"""
Toolchain environment management.

Manages named sets of toolchains that can be activated together.
An environment is a mapping of compiler names to specific versions
and target configurations. Environments enable reproducible builds
by pinning exact toolchain versions per project.

This module is distinct from `installer/environments.py` which
manages environments for the download/install layer. This module
manages environments of already-installed, locally-detected
toolchains that can be activated as a group.

Design
------
An environment is a JSON-serializable dictionary stored in
`{toolforge_data}/environments/`. Each environment specifies:
    - Which compiler to use for C, C++, and other languages
    - The toolchain version or path
    - Target architecture and ABI preferences
    - Environment variables to set on activation

Environments can be activated in-process (modifying os.environ)
or exported as shell scripts for eval.

Usage
-----
    from pyputil_install.compiler_installer.toolchains.environments import EnvironmentStore

    store = EnvironmentStore()

    # Create an environment for a C++20 project
    store.create("cpp20-project", {
        "CC": "gcc@14",
        "CXX": "g++@14",
        "CFLAGS": "-O2 -march=native",
        "CXXFLAGS": "-std=c++20 -O2",
    })

    # Activate it
    store.activate("cpp20-project")

    # List all environments
    for name in store.list_all():
        print(name, store.get(name))

Warnings
--------
- Environments reference toolchains by name and version.
  The toolchains must be installed and detected by `detection.py`
  before they can be used in an environment.
- Activating an environment modifies PATH and other environment
  variables in the current process. This is not thread-safe.
- The "default" environment name is reserved. It is activated
  automatically by some ToolForge commands when present.

User Instructions
-----------------
- Create one environment per project to pin compiler versions.
- Use `store.export_env()` to share environments via version control.
- Use `store.import_env()` to load environments from files.
- Activate an environment before building: `store.activate("myproject")`.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .base import Toolchain
from .detection import detect_all_toolchains, detect_best_toolchain
from ..installer.activation import activate as activate_toolchain
from ..installer.activation import deactivate as deactivate_toolchain
from ..installer.activation import deactivate_all as deactivate_all_toolchains

logger = logging.getLogger(__name__)

# Current environment schema version
ENV_SCHEMA_VERSION = 1

# Reserved environment name for the default
DEFAULT_ENV_NAME = "default"


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
        Unique environment name. Filesystem-safe characters only.
    compilers : Dict[str, str]
        Mapping of role (CC, CXX, FC, etc.) to toolchain spec.
        A spec is either "name@version" (e.g., "gcc@14.2.0-2")
        or a direct path to a compiler executable.
    flags : Dict[str, str]
        Default compiler flags for each role.
        Example: {"CFLAGS": "-O2", "CXXFLAGS": "-std=c++20 -O2"}.
    env_vars : Dict[str, str]
        Additional environment variables to set on activation.
    description : str
        Human-readable description of the environment.
    created_at : str
        ISO 8601 creation timestamp.
    updated_at : str
        ISO 8601 last-modification timestamp.
    schema_version : int
        Schema version for forward compatibility.
    """

    name: str
    compilers: Dict[str, str] = field(default_factory=dict)
    flags: Dict[str, str] = field(default_factory=dict)
    env_vars: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    schema_version: int = ENV_SCHEMA_VERSION

    def to_dict(self) -> Dict:
        """Serialize to JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Environment":
        """Deserialize from JSON-compatible dictionary."""
        return cls(
            name=data.get("name", ""),
            compilers=data.get("compilers", {}),
            flags=data.get("flags", {}),
            env_vars=data.get("env_vars", {}),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            schema_version=data.get("schema_version", 1),
        )

    def has_compiler(self, role: str) -> bool:
        """Check if a compiler role is defined."""
        return role in self.compilers

    def get_compiler_spec(self, role: str) -> Optional[str]:
        """Return the compiler spec for a role."""
        return self.compilers.get(role)

    def get_flags(self, role: str) -> Optional[str]:
        """Return default flags for a role."""
        return self.flags.get(role)


# ============================================================================
# EnvironmentStore
# ============================================================================


class EnvironmentStore:
    """
    Manages named toolchain environment files.

    Parameters
    ----------
    data_dir : Optional[Path]
        Directory for environment files. Defaults to
        `{toolforge_data}/environments/`.

    Attributes
    ----------
    env_dir : Path
        Path to the environments directory.
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        if data_dir is None:
            xdg = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
            data_dir = Path(xdg) / "toolforge"

        self.env_dir = data_dir / "environments"
        self.env_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _env_path(self, name: str) -> Path:
        """Return the path to an environment file."""
        return self.env_dir / f"{name}.json"

    @staticmethod
    def _validate_name(name: str) -> None:
        """
        Validate an environment name.

        Parameters
        ----------
        name : str
            Proposed name.

        Raises
        ------
        ValueError
            If the name is empty or contains invalid characters.
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
        compilers: Optional[Dict[str, str]] = None,
        flags: Optional[Dict[str, str]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        description: str = "",
        overwrite: bool = False,
    ) -> Environment:
        """
        Create a new environment.

        Parameters
        ----------
        name : str
            Environment name.
        compilers : Optional[Dict[str, str]]
            Compiler specs by role.
        flags : Optional[Dict[str, str]]
            Default compiler flags.
        env_vars : Optional[Dict[str, str]]
            Extra environment variables.
        description : str
            Description.
        overwrite : bool
            If True, overwrites existing environment.

        Returns
        -------
        Environment
            Created environment.

        Raises
        ------
        FileExistsError
            If the environment exists and overwrite is False.
        """
        self._validate_name(name)

        env_path = self._env_path(name)
        if env_path.exists() and not overwrite:
            raise FileExistsError(
                f"Environment {name!r} already exists. Use overwrite=True to replace."
            )

        now = datetime.now(timezone.utc).isoformat()
        created_at = now
        if env_path.exists():
            existing = self.get(name)
            if existing:
                created_at = existing.created_at

        env = Environment(
            name=name,
            compilers=compilers or {},
            flags=flags or {},
            env_vars=env_vars or {},
            description=description,
            created_at=created_at,
            updated_at=now,
        )

        self._write(env)
        logger.info("Created environment %r", name)
        return env

    def update(
        self,
        name: str,
        compilers: Optional[Dict[str, str]] = None,
        flags: Optional[Dict[str, str]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> Environment:
        """
        Update an existing environment.

        Parameters
        ----------
        name : str
            Environment name.
        compilers, flags, env_vars, description : Optional
            New values. If None, existing value is kept.

        Returns
        -------
        Environment
            Updated environment.
        """
        existing = self.get_or_raise(name)
        now = datetime.now(timezone.utc).isoformat()

        env = Environment(
            name=name,
            compilers=compilers if compilers is not None else existing.compilers,
            flags=flags if flags is not None else existing.flags,
            env_vars=env_vars if env_vars is not None else existing.env_vars,
            description=description if description is not None else existing.description,
            created_at=existing.created_at,
            updated_at=now,
        )

        self._write(env)
        logger.info("Updated environment %r", name)
        return env

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
            Loaded environment, or None.
        """
        env_path = self._env_path(name)
        if not env_path.is_file():
            return None

        try:
            data = json.loads(env_path.read_text(encoding="utf-8"))
            return Environment.from_dict(data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupted environment %r: %s", name, exc)
            return None

    def get_or_raise(self, name: str) -> Environment:
        """
        Load an environment, raising if not found.

        Raises
        ------
        FileNotFoundError
            If not found.
        ValueError
            If corrupted.
        """
        env = self.get(name)
        if env is None:
            env_path = self._env_path(name)
            if not env_path.is_file():
                raise FileNotFoundError(f"Environment {name!r} not found")
            raise ValueError(f"Environment {name!r} is corrupted")
        return env

    def get_default(self) -> Optional[Environment]:
        """Load the default environment."""
        return self.get(DEFAULT_ENV_NAME)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, name: str) -> bool:
        """
        Delete an environment.

        Parameters
        ----------
        name : str
            Environment name.

        Returns
        -------
        bool
            True if deleted.
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
        """List all environment names."""
        if not self.env_dir.exists():
            return []
        return sorted([
            entry.stem for entry in self.env_dir.iterdir()
            if entry.suffix == ".json"
        ])

    def list_all_environments(self) -> List[Environment]:
        """List all environments with full data."""
        result = []
        for name in self.list_all():
            env = self.get(name)
            if env is not None:
                result.append(env)
        return result

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def export_env(self, name: str, dest: Path) -> bool:
        """
        Export an environment to a JSON file.

        Parameters
        ----------
        name : str
            Environment name.
        dest : Path
            Destination path.

        Returns
        -------
        bool
            True on success.
        """
        env = self.get_or_raise(name)
        try:
            dest.write_text(
                json.dumps(env.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
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
            Source file.
        overwrite : bool
            Overwrite existing.

        Returns
        -------
        Environment
            Imported environment.
        """
        if not source.is_file():
            raise FileNotFoundError(f"Source not found: {source}")

        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}")

        if "name" not in data:
            raise ValueError("Environment file missing 'name' field")

        return self.create(
            name=data["name"],
            compilers=data.get("compilers", {}),
            flags=data.get("flags", {}),
            env_vars=data.get("env_vars", {}),
            description=data.get("description", ""),
            overwrite=overwrite,
        )

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def activate(self, name: str) -> bool:
        """
        Activate an environment in the current process.

        Sets environment variables (CC, CXX, CFLAGS, etc.) and
        activates each toolchain via activation.activate().

        Parameters
        ----------
        name : str
            Environment name.

        Returns
        -------
        bool
            True if activated successfully.
        """
        env = self.get_or_raise(name)
        logger.info("Activating environment %r", name)

        # Set compiler environment variables
        for role, spec in env.compilers.items():
            os.environ[role] = spec

        # Set flags
        for flag_role, flag_value in env.flags.items():
            os.environ[flag_role] = flag_value

        # Set extra vars
        for var_name, var_value in env.env_vars.items():
            os.environ[var_name] = var_value

        # Activate toolchains by spec
        for role, spec in env.compilers.items():
            if "@" in spec and "/" not in spec:
                compiler_name, _, version = spec.partition("@")
                try:
                    activate_toolchain(compiler_name, version)
                except Exception as exc:
                    logger.warning("Failed to activate %s: %s", spec, exc)
            elif Path(spec).is_file():
                # Direct path to compiler — add its directory to PATH
                bin_dir = str(Path(spec).parent)
                current_path = os.environ.get("PATH", "")
                if bin_dir not in current_path.split(os.pathsep):
                    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{current_path}"

        return True

    def deactivate(self, name: str) -> bool:
        """
        Deactivate an environment.

        Removes environment-specific variables and deactivates
        the most recently activated toolchain for each compiler
        in the environment.

        Parameters
        ----------
        name : str
            Environment name.

        Returns
        -------
        bool
            True if deactivated.
        """
        env = self.get_or_raise(name)
        logger.info("Deactivating environment %r", name)

        # Remove environment variables set by this environment
        for role in env.compilers:
            os.environ.pop(role, None)
        for flag_role in env.flags:
            os.environ.pop(flag_role, None)
        for var_name in env.env_vars:
            os.environ.pop(var_name, None)

        # Deactivate each toolchain
        for _ in env.compilers:
            try:
                deactivate_toolchain()
            except Exception:
                pass

        return True

    def shell_activate_script(self, name: str, shell: str = "bash") -> str:
        """
        Generate shell commands to activate an environment.

        Parameters
        ----------
        name : str
            Environment name.
        shell : str
            Target shell: "bash", "zsh", "fish", "cmd", "powershell".

        Returns
        -------
        str
            Shell commands.
        """
        env = self.get_or_raise(name)

        generators = {
            "bash": self._script_bash,
            "zsh": self._script_bash,
            "fish": self._script_fish,
            "cmd": self._script_cmd,
            "powershell": self._script_powershell,
        }

        gen = generators.get(shell.lower(), self._script_bash)
        return gen(env)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, name: str) -> List[str]:
        """
        Check that all toolchains referenced by an environment
        are available and installed.

        Parameters
        ----------
        name : str
            Environment name.

        Returns
        -------
        List[str]
            Warnings for missing toolchains.
        """
        env = self.get_or_raise(name)
        warnings = []
        all_tcs = detect_all_toolchains()

        for role, spec in env.compilers.items():
            if "@" in spec and "/" not in spec:
                compiler_name, _, version = spec.partition("@")
                found = any(
                    tc.kind.name.lower() == compiler_name.lower()
                    and tc.version == version
                    for tc in all_tcs
                )
                if not found:
                    warnings.append(
                        f"{role}: {spec} not found among detected toolchains"
                    )
            elif "/" in spec:
                if not Path(spec).is_file():
                    warnings.append(f"{role}: path not found — {spec}")

        return warnings

    # ------------------------------------------------------------------
    # Shell script generators
    # ------------------------------------------------------------------

    def _script_bash(self, env: Environment) -> str:
        """Generate bash/zsh activation script."""
        lines = [f"# Environment: {env.name}", ""]
        for role, spec in env.compilers.items():
            lines.append(f"export {role}={spec}")
        for flag_role, flag_value in env.flags.items():
            lines.append(f"export {flag_role}={flag_value}")
        for var_name, var_value in env.env_vars.items():
            lines.append(f"export {var_name}={var_value}")
        return "\n".join(lines)

    def _script_fish(self, env: Environment) -> str:
        """Generate fish activation script."""
        lines = [f"# Environment: {env.name}", ""]
        for role, spec in env.compilers.items():
            lines.append(f"set -gx {role} {spec}")
        for flag_role, flag_value in env.flags.items():
            lines.append(f"set -gx {flag_role} {flag_value}")
        for var_name, var_value in env.env_vars.items():
            lines.append(f"set -gx {var_name} {var_value}")
        return "\n".join(lines)

    def _script_cmd(self, env: Environment) -> str:
        """Generate cmd.exe activation script."""
        lines = [f"@rem Environment: {env.name}", ""]
        for role, spec in env.compilers.items():
            lines.append(f'@set "{role}={spec}"')
        for flag_role, flag_value in env.flags.items():
            lines.append(f'@set "{flag_role}={flag_value}"')
        for var_name, var_value in env.env_vars.items():
            lines.append(f'@set "{var_name}={var_value}"')
        return "\n".join(lines)

    def _script_powershell(self, env: Environment) -> str:
        """Generate PowerShell activation script."""
        lines = [f"# Environment: {env.name}", ""]
        for role, spec in env.compilers.items():
            lines.append(f'$env:{role} = "{spec}"')
        for flag_role, flag_value in env.flags.items():
            lines.append(f'$env:{flag_role} = "{flag_value}"')
        for var_name, var_value in env.env_vars.items():
            lines.append(f'$env:{var_name} = "{var_value}"')
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, env: Environment) -> None:
        """Write an environment to disk atomically."""
        env_path = self._env_path(env.name)
        fd, tmp = tempfile.mkstemp(suffix=".json", prefix=".tmp-", dir=str(self.env_dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(env.to_dict(), f, indent=2, sort_keys=True, ensure_ascii=False)
            os.replace(tmp, env_path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
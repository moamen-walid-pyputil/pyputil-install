"""
Toolchain activation and deactivation for shell sessions.

Provides temporary (per-session) and persistent environment
modifications so that installed toolchains appear on PATH.

Two modes of operation:
    1. Session activation — modifies os.environ in the current
       Python process. The effect lasts only until the process exits.
    2. Shell script generation — prints shell commands that the
       user can `eval` in their shell to activate a toolchain
       in that shell session.

Design
------
Activation is a stack. Each call to `activate()` pushes a toolchain
onto the activation stack. `deactivate()` pops the most recent one.

    activate("gcc", "14.2.0-2")    # gcc@14 now on PATH
    activate("zig", "0.11.0")       # gcc@14 + zig@0.11 on PATH
    deactivate()                     # zig@0.11 removed
    deactivate()                     # gcc@14 removed

The activation state is stored in a per-process singleton and is
NOT persisted across Python process restarts.

Environment Variables
---------------------
TOOLFORGE_ACTIVE
    Colon-separated list of active toolchain identifiers in order.
    Set during session activation. Read by subprocesses to inherit
    the active toolchain state.

TOOLFORGE_PATH_PREPEND
    If set to "1" (default), toolchain bin directories are prepended
    to PATH. If "0", they are appended.

Usage
-----
    from toolforge.installer.activation import activate, deactivate, current

    activate("gcc", "14.2.0-2")
    print(current())         # [("gcc", "14.2.0-2")]
    activate("zig", "0.11.0")
    print(current())         # [("gcc", "14.2.0-2"), ("zig", "0.11.0")]
    deactivate()
    print(current())         # [("gcc", "14.2.0-2")]

Shell Script Generation
-----------------------
    from pyputil_install.compiler_installer.activation import shell_activate_script

    script = shell_activate_script("gcc", "14.2.0-2", shell="bash")
    print(script)
    # export PATH="/home/user/.local/share/toolforge/toolchains/gcc/14.2.0-2/bin:$PATH"
    # export TOOLFORGE_ACTIVE="gcc:14.2.0-2"

Warnings
--------
- Session activation modifies os.environ directly. This is NOT
  thread-safe. Use in single-threaded contexts or guard with a lock.
- The activation stack is process-global. Calling activate() in
  one module affects all modules in the same process.
- Activation does NOT validate that toolchain executables work.
  It only adjusts PATH. Run `gcc --version` to verify.
- Shell scripts are NOT evaluated automatically. The user must
  `eval` them: `eval "$(python -m toolforge activate gcc@14)"`

User Instructions
-----------------
- Use `activate()` / `deactivate()` in Python scripts that need
  to temporarily use a specific toolchain.
- Use `shell_activate_script()` to generate shell commands for
  interactive terminal sessions.
- Call `deactivate_all()` at the end of a script to restore the
  original PATH.
- The activation stack is available via `current()`.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .layouts import ToolchainLayout, get_install_root

logger = logging.getLogger(__name__)

# ============================================================================
# Per-process activation stack
# ============================================================================

# _activation_stack stores (compiler, version, old_path_entry)
# The old_path_entry is the PATH segment that was added; used to
# restore PATH exactly on deactivation.
_activation_stack: List[Tuple[str, str, str]] = []

# _original_path is saved once before the first activation.
# Used by deactivate_all() to fully restore.
_original_path: Optional[str] = None


# ============================================================================
# PATH manipulation
# ============================================================================

def _prepend_to_path(new_entry: str) -> str:
    """
    Prepend a directory to the PATH-like string.

    Parameters
    ----------
    new_entry : str
        Directory path to add.

    Returns
    -------
    str
        New PATH string with the entry prepended.
        Duplicate entries are removed first.
    """
    path = os.environ.get("PATH", "")
    entries = path.split(os.pathsep)

    # Remove duplicate if present
    entries = [e for e in entries if e != new_entry]

    return os.pathsep.join([new_entry] + entries)


def _remove_from_path(old_entry: str) -> str:
    """
    Remove a directory from the PATH-like string.

    Parameters
    ----------
    old_entry : str
        Directory path to remove.

    Returns
    -------
    str
        New PATH string without the entry.
    """
    path = os.environ.get("PATH", "")
    entries = path.split(os.pathsep)
    entries = [e for e in entries if e != old_entry]
    return os.pathsep.join(entries)


def _path_prepend_enabled() -> bool:
    """
    Check whether toolchain directories should be prepended.

    Returns
    -------
    bool
        True if TOOLFORGE_PATH_PREPEND is unset or "1".
    """
    return os.environ.get("TOOLFORGE_PATH_PREPEND", "1") == "1"


# ============================================================================
# Session activation (in-process)
# ============================================================================

def activate(
    compiler: str,
    version: str,
    install_root: Optional[Path] = None,
) -> bool:
    """
    Activate a toolchain version in the current process.

    Prepends (or appends) the toolchain's `bin` directory to PATH
    and pushes the activation onto the internal stack.

    Parameters
    ----------
    compiler : str
        Compiler name, e.g., "gcc".
    version : str
        Version string, e.g., "14.2.0-2".
    install_root : Optional[Path]
        Root directory. If None, `get_install_root()` is called.

    Returns
    -------
    bool
        True if the toolchain exists and was activated, False if
        the toolchain directory does not exist.

    Warnings
    --------
    - This modifies os.environ['PATH'] directly.
    - Calling activate() for an already-active version will activate
      it again, duplicating the PATH entry. Use `is_active()` to
      check first.
    """
    global _original_path, _activation_stack

    layout = ToolchainLayout(compiler, version, install_root)
    if not layout.exists():
        logger.warning("Toolchain not found: %s", layout.path)
        return False

    bin_dir = str(layout.bin_dir)

    # Save original PATH on first activation
    if _original_path is None:
        _original_path = os.environ.get("PATH", "")

    # Modify PATH
    if _path_prepend_enabled():
        new_path = _prepend_to_path(bin_dir)
    else:
        path = os.environ.get("PATH", "")
        entries = path.split(os.pathsep)
        entries = [e for e in entries if e != bin_dir]
        entries.append(bin_dir)
        new_path = os.pathsep.join(entries)

    os.environ["PATH"] = new_path

    # Update the activation stack and marker env var
    _activation_stack.append((compiler, version, bin_dir))
    _update_marker_env()

    logger.info("Activated %s@%s (%s)", compiler, version, bin_dir)
    return True


def deactivate() -> Optional[Tuple[str, str]]:
    """
    Deactivate the most recently activated toolchain.

    Removes its bin directory from PATH and pops it from the stack.

    Returns
    -------
    Optional[Tuple[str, str]]
        The (compiler, version) that was deactivated, or None if
        the stack was empty.

    Warnings
    --------
    - If PATH was modified by other code between activate() and
      deactivate(), the restoration may be incomplete.
    """
    global _activation_stack

    if not _activation_stack:
        logger.warning("No toolchains are active")
        return None

    compiler, version, bin_dir = _activation_stack.pop()
    new_path = _remove_from_path(bin_dir)
    os.environ["PATH"] = new_path
    _update_marker_env()

    logger.info("Deactivated %s@%s", compiler, version)
    return compiler, version


def deactivate_all() -> int:
    """
    Deactivate all active toolchains, restoring the original PATH.

    Returns
    -------
    int
        Number of toolchains deactivated.
    """
    global _activation_stack, _original_path

    count = len(_activation_stack)

    if _original_path is not None:
        os.environ["PATH"] = _original_path
    else:
        # No original saved — just remove all known entries
        for _, _, bin_dir in _activation_stack:
            os.environ["PATH"] = _remove_from_path(bin_dir)

    _activation_stack = []
    _update_marker_env()

    if count > 0:
        logger.info("Deactivated %d toolchain(s)", count)
    return count


def current() -> List[Tuple[str, str]]:
    """
    Return the current activation stack.

    Returns
    -------
    List[Tuple[str, str]]
        List of (compiler, version) tuples in activation order.
        First item is the oldest activation, last is the most recent.
    """
    return [(c, v) for c, v, _ in _activation_stack]


def is_active(compiler: str, version: Optional[str] = None) -> bool:
    """
    Check if a specific compiler (and optionally version) is active.

    Parameters
    ----------
    compiler : str
        Compiler name to check.
    version : Optional[str]
        If provided, also checks the version. If None, any version
        of this compiler matches.

    Returns
    -------
    bool
        True if the compiler is currently active.
    """
    for c, v, _ in _activation_stack:
        if c == compiler:
            if version is None or v == version:
                return True
    return False


# ============================================================================
# Environment marker
# ============================================================================

def _update_marker_env() -> None:
    """
    Update the TOOLFORGE_ACTIVE environment variable.

    Sets it to a double-colon-separated list of "compiler:version"
    pairs. The double colon "::" is used as the entry separator
    because version strings may contain single colons.
    """
    if not _activation_stack:
        os.environ.pop("TOOLFORGE_ACTIVE", None)
        return

    pairs = [f"{c}:{v}" for c, v, _ in _activation_stack]
    os.environ["TOOLFORGE_ACTIVE"] = "::".join(pairs)


def read_marker_env() -> List[Tuple[str, str]]:
    """
    Parse the TOOLFORGE_ACTIVE environment variable.

    The variable contains colon-separated entries. Each entry is
    formatted as "compiler:version". The version may itself contain
    colons (e.g., ARM GNU toolchains use versions like "13.2.Rel1").
    To handle this, entries are separated by a double colon "::".

    If the variable is set by a newer ToolForge, the format is:
        gcc:14.2.0-2::clang:18.1.0::zig:0.11.0

    If the variable was set by an older ToolForge (single colon
    separator), we fall back to the old format:
        gcc:14.2.0-2:clang:18.1.0

    Returns
    -------
    List[Tuple[str, str]]
        List of (compiler, version) pairs in activation order.
        Returns an empty list if the variable is not set or empty.

    Examples
    --------
    >>> os.environ["TOOLFORGE_ACTIVE"] = "gcc:14.2.0-2::clang:18.1.0"
    >>> read_marker_env()
    [('gcc', '14.2.0-2'), ('clang', '18.1.0')]

    >>> os.environ["TOOLFORGE_ACTIVE"] = ""
    >>> read_marker_env()
    []
    """
    raw = os.environ.get("TOOLFORGE_ACTIVE", "")
    if not raw or not raw.strip():
        return []

    result: List[Tuple[str, str]] = []

    # New format: double colon separates entries
    if "::" in raw:
        entries = raw.split("::")
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(":", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                result.append((parts[0].strip(), parts[1].strip()))
        return result

    # Old format: single colon separates everything
    # Format: compiler:version:compiler:version...
    # We need to pair them up: even indices = compiler, odd = version
    tokens = raw.split(":")
    if len(tokens) >= 2:
        for i in range(0, len(tokens) - 1, 2):
            compiler = tokens[i].strip()
            version = tokens[i + 1].strip()
            if compiler and version:
                result.append((compiler, version))

    return result


# ============================================================================
# Shell script generation
# ============================================================================

def shell_activate_script(
    compiler: str,
    version: str,
    install_root: Optional[Path] = None,
    shell: str = "bash",
) -> str:
    """
    Generate shell commands to activate a toolchain.

    The output is intended to be `eval`ed in a shell:
        eval "$(python -m toolforge activate gcc@14)"

    Parameters
    ----------
    compiler : str
        Compiler name.
    version : str
        Version string.
    install_root : Optional[Path]
        Root directory. If None, `get_install_root()` is called.
    shell : str
        Target shell: "bash", "zsh", "fish", "cmd", "powershell".
        Default is "bash" (also works for zsh).

    Returns
    -------
    str
        Shell commands to activate the toolchain.

    Raises
    ------
    ValueError
        If the shell type is not supported.
    FileNotFoundError
        If the toolchain is not installed.

    Examples
    --------
    >>> print(shell_activate_script("gcc", "14.2.0-2"))
    export PATH="/home/user/.../gcc/14.2.0-2/bin:$PATH"
    export TOOLFORGE_ACTIVE="gcc:14.2.0-2"
    """
    layout = ToolchainLayout(compiler, version, install_root)
    if not layout.exists():
        raise FileNotFoundError(f"Toolchain not found: {layout.path}")

    bin_dir = str(layout.bin_dir)
    marker = f"{compiler}:{version}"

    generators = {
        "bash": _script_bash,
        "zsh": _script_bash,   # Same syntax as bash
        "fish": _script_fish,
        "cmd": _script_cmd,
        "powershell": _script_powershell,
    }

    generator = generators.get(shell.lower())
    if generator is None:
        raise ValueError(
            f"Unsupported shell: {shell!r}. "
            f"Supported: {list(generators.keys())}"
        )

    return generator(bin_dir, marker)


def shell_deactivate_script(shell: str = "bash") -> str:
    """
    Generate shell commands to deactivate the most recent toolchain.

    Parameters
    ----------
    shell : str
        Target shell. Default "bash".

    Returns
    -------
    str
        Shell commands. If no toolchain is active, prints a message
        to stderr and does nothing.
    """
    generators = {
        "bash": _deactivate_script_bash,
        "zsh": _deactivate_script_bash,
        "fish": _deactivate_script_fish,
        "cmd": _deactivate_script_cmd,
        "powershell": _deactivate_script_powershell,
    }

    generator = generators.get(shell.lower())
    if generator is None:
        raise ValueError(f"Unsupported shell: {shell!r}")

    return generator()


# --------------------------------------------------------------------------
# Shell-specific generators
# --------------------------------------------------------------------------

def _script_bash(bin_dir: str, marker: str) -> str:
    """Generate bash/zsh activation script."""
    return (
        f'export PATH="{bin_dir}:$PATH"\n'
        f'export TOOLFORGE_ACTIVE="{marker}"\n'
    )


def _script_fish(bin_dir: str, marker: str) -> str:
    """Generate fish activation script."""
    return (
        f'set -gx PATH "{bin_dir}" $PATH\n'
        f'set -gx TOOLFORGE_ACTIVE "{marker}"\n'
    )


def _script_cmd(bin_dir: str, marker: str) -> str:
    """Generate cmd.exe activation script."""
    return (
        f'@set "PATH={bin_dir};%PATH%"\n'
        f'@set "TOOLFORGE_ACTIVE={marker}"\n'
    )


def _script_powershell(bin_dir: str, marker: str) -> str:
    """Generate PowerShell activation script."""
    return (
        f'$env:PATH = "{bin_dir};" + $env:PATH\n'
        f'$env:TOOLFORGE_ACTIVE = "{marker}"\n'
    )


def _deactivate_script_bash() -> str:
    """Generate bash/zsh deactivation script."""
    return (
        'if [ -n "$TOOLFORGE_ACTIVE" ]; then\n'
        '  echo "Deactivated ${TOOLFORGE_ACTIVE%%:*}"\n'
        '  unset TOOLFORGE_ACTIVE\n'
        'else\n'
        '  echo "No toolchain is active" >&2\n'
        'fi\n'
    )


def _deactivate_script_fish() -> str:
    """Generate fish deactivation script."""
    return (
        'if set -q TOOLFORGE_ACTIVE\n'
        '  echo "Deactivated (fish does not track PATH changes automatically)"\n'
        '  set -e TOOLFORGE_ACTIVE\n'
        'else\n'
        '  echo "No toolchain is active" >&2\n'
        'end\n'
    )


def _deactivate_script_cmd() -> str:
    """Generate cmd.exe deactivation script."""
    return (
        '@if defined TOOLFORGE_ACTIVE (\n'
        '  @echo Deactivated\n'
        '  @set "TOOLFORGE_ACTIVE="\n'
        ') else (\n'
        '  @echo No toolchain is active >&2\n'
        ')\n'
    )


def _deactivate_script_powershell() -> str:
    """Generate PowerShell deactivation script."""
    return (
        'if ($env:TOOLFORGE_ACTIVE) {\n'
        '  Write-Host "Deactivated"\n'
        '  Remove-Item Env:TOOLFORGE_ACTIVE\n'
        '} else {\n'
        '  Write-Error "No toolchain is active"\n'
        '}\n'
    )
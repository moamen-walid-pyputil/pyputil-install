"""
Compiler discovery strategies.

Each strategy is a self-contained plugin that finds raw compiler paths.
Strategies MUST NOT execute or validate discovered binaries — that is
the exclusive responsibility of the validation layer.

Layer
-----
This belongs to the Detection Layer. It finds paths and returns them.
No subprocess execution on compiler binaries happens here.

Strategy Contract
-----------------
Every strategy must:
    1. Subclass DiscoveryStrategy
    2. Set a unique `name` class attribute
    3. Set an integer `priority` (lower = runs earlier, ranks higher)
    4. Implement `discover() -> List[str]`
    5. Handle ALL internal exceptions — never let them propagate
    6. Return only absolute paths (use os.path.abspath)

Adding Custom Strategies
------------------------
    from toolforge.strategies import DiscoveryStrategy, DiscoveryStrategyRegistry

    class MyCustomStrategy(DiscoveryStrategy):
        name = "my_custom"
        priority = 25

        def discover(self) -> List[str]:
            paths = []
            # Your discovery logic here
            return paths

    DiscoveryStrategyRegistry.register(MyCustomStrategy)

Usage
-----
    from toolforge.strategies import DiscoveryStrategyRegistry

    # Get all enabled strategies sorted by priority
    strategies = DiscoveryStrategyRegistry.get_enabled()

    # Disable a problematic strategy
    DiscoveryStrategyRegistry.disable("common_dirs")

    # Run all strategies
    all_paths = []
    for strategy_cls in strategies:
        instance = strategy_cls()
        all_paths.extend(instance.discover())

Warnings
--------
- Strategies run in the main process by default. The orchestrator
  in `discovery.py` handles parallel execution when appropriate.
- Returned paths may be broken symlinks, non-compiler executables,
  or completely invalid — the caller MUST validate them.
- Strategy `discover()` is called once per discovery cycle.
  Do not cache state inside strategy instances.

User Instructions
-----------------
- Set COMPILER_SEARCH_CANDIDATES env var to override default
  candidate executable names (colon-separated on Unix, semicolon on Windows).
- Set COMPILER_SEARCH_DIRS to add extra directories to scan
  (colon-separated on Unix, semicolon on Windows).
- Disable slow strategies via `DiscoveryStrategyRegistry.disable(name)`.
"""

import abc
import os
import platform
from typing import List, Type, Dict, Set, ClassVar, Optional


# ---------------------------------------------------------------------------
# Candidate executable names
# ---------------------------------------------------------------------------

def _get_candidates() -> List[str]:
    """
    Get the list of executable names to search for.

    Reads COMPILER_SEARCH_CANDIDATES env var if set.
    Default candidates cover GCC, Clang, MSVC, and generic `cc`.

    Returns
    -------
    List[str]
        Executable names in lowercase.
    """
    env_val = os.environ.get("COMPILER_SEARCH_CANDIDATES", "")
    if env_val:
        separator = ";" if platform.system() == "Windows" else ":"
        return [name.strip().lower() for name in env_val.split(separator) if name.strip()]

    return [
        "gcc", "g++", "cc", "c++",
        "clang", "clang++",
        "cl.exe",
    ]


# ---------------------------------------------------------------------------
# Strategy base class
# ---------------------------------------------------------------------------

class DiscoveryStrategy(abc.ABC):
    """
    Abstract base class for all compiler discovery strategies.

    Subclasses must set `name` and `priority` as class attributes
    and implement the `discover()` method.

    Class Attributes
    ----------------
    name : str
        Unique strategy identifier. Used for enable/disable/reorder.
        Must be set by subclasses. No default value.
    priority : int
        Execution order and ranking priority. Lower numbers execute
        first and have higher weight in the final compiler ordering.
        Default is 100 if not overridden.
    """

    name: ClassVar[str] = ""
    priority: ClassVar[int] = 100

    @abc.abstractmethod
    def discover(self) -> List[str]:
        """
        Find potential compiler executable paths.

        Returns
        -------
        List[str]
            Absolute paths to files that MAY be compilers.
            Return empty list if nothing found.
            Paths are NOT validated — they may be broken, non-executable,
            or completely unrelated binaries.

        Warnings
        --------
        - Must handle all internal exceptions. Never let exceptions
          propagate to the caller.
        - Must be stateless. Multiple calls should return consistent results
          based on the current system state.
        """
        ...


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------

class UserOverrideStrategy(DiscoveryStrategy):
    """
    Use compiler paths explicitly provided by the user.

    Reads these environment variables in order:
        CC, CXX, C_COMPILER, CXX_COMPILER

    Also checks for an explicit path passed via the API
    (handled by the orchestrator, not here).

    Priority: 0 (always runs first and ranks highest)

    Warnings
    --------
    - Paths are trusted if they exist as files. No validation here.
    - A typo in CC=/usr/bin/gccc will still be returned as a path.
      Validation will reject it later.
    """

    name: ClassVar[str] = "user_override"
    priority: ClassVar[int] = 0

    def discover(self) -> List[str]:
        """
        Read compiler paths from environment variables.

        Returns
        -------
        List[str]
            Absolute paths from CC, CXX, C_COMPILER, CXX_COMPILER.
            Skips variables that are unset or point to missing files.
        """
        paths: List[str] = []
        env_vars = ["CC", "CXX", "C_COMPILER", "CXX_COMPILER"]

        for var in env_vars:
            try:
                value = os.environ.get(var)
                if value and os.path.isfile(value):
                    paths.append(os.path.abspath(value))
            except OSError:
                continue

        return paths


class ManagedToolchainStrategy(DiscoveryStrategy):
    """
    Search for compilers installed by toolchain managers.

    Checks known installation directories for:
        - Rustup (~/.rustup/toolchains/*/bin)
        - Android NDK ($ANDROID_NDK_HOME/toolchains/*/prebuilt/*/bin)
        - SDKMAN (~/.sdkman/candidates/java/*/bin) — for completeness
        - Emscripten ($EMSDK/upstream/bin)

    Priority: 5 (runs after user override, before PATH)

    Warnings
    --------
    - Directory structures are hardcoded and may change with future
      toolchain manager versions.
    - ANDROID_NDK_HOME and EMSDK env vars must be set for those checks.
    - This strategy does NOT download or install anything.
    """

    name: ClassVar[str] = "managed_toolchains"
    priority: ClassVar[int] = 5

    def discover(self) -> List[str]:
        """
        Scan known toolchain manager directories for compilers.

        Returns
        -------
        List[str]
            Paths found under known toolchain directories.
        """
        paths: List[str] = []
        paths.extend(self._scan_rustup())
        paths.extend(self._scan_android_ndk())
        paths.extend(self._scan_emscripten())
        return paths

    def _scan_rustup(self) -> List[str]:
        """Scan Rustup toolchain directories for cc/gcc symlinks."""
        paths: List[str] = []
        rustup_home = os.environ.get(
            "RUSTUP_HOME", os.path.expanduser("~/.rustup")
        )
        toolchains_dir = os.path.join(rustup_home, "toolchains")

        if not os.path.isdir(toolchains_dir):
            return paths

        candidates = _get_candidates()
        try:
            for toolchain in os.listdir(toolchains_dir):
                bin_dir = os.path.join(toolchains_dir, toolchain, "bin")
                if not os.path.isdir(bin_dir):
                    continue
                for candidate in candidates:
                    full_path = os.path.join(bin_dir, candidate)
                    if os.path.isfile(full_path):
                        paths.append(full_path)
        except PermissionError:
            pass

        return paths

    def _scan_android_ndk(self) -> List[str]:
        """Scan Android NDK toolchain directories."""
        paths: List[str] = []
        ndk_home = os.environ.get("ANDROID_NDK_HOME", "")

        if not ndk_home or not os.path.isdir(ndk_home):
            return paths

        toolchains_dir = os.path.join(ndk_home, "toolchains")
        if not os.path.isdir(toolchains_dir):
            return paths

        candidates = _get_candidates()
        try:
            for arch_dir in os.listdir(toolchains_dir):
                prebuilt_dir = os.path.join(toolchains_dir, arch_dir, "prebuilt")
                if not os.path.isdir(prebuilt_dir):
                    continue
                for host_dir in os.listdir(prebuilt_dir):
                    bin_dir = os.path.join(prebuilt_dir, host_dir, "bin")
                    if not os.path.isdir(bin_dir):
                        continue
                    for candidate in candidates:
                        full_path = os.path.join(bin_dir, candidate)
                        if os.path.isfile(full_path):
                            paths.append(full_path)
                        # Also check for prefixed compilers: arm-linux-androideabi-gcc
                        for entry in os.listdir(bin_dir):
                            if entry.endswith(f"-{candidate}"):
                                full_path = os.path.join(bin_dir, entry)
                                if os.path.isfile(full_path):
                                    paths.append(full_path)
        except PermissionError:
            pass

        return paths

    def _scan_emscripten(self) -> List[str]:
        """Scan Emscripten SDK for emcc."""
        paths: List[str] = []
        emsdk = os.environ.get("EMSDK", "")

        if not emsdk or not os.path.isdir(emsdk):
            return paths

        emcc_path = os.path.join(emsdk, "upstream", "bin", "emcc")
        if os.path.isfile(emcc_path):
            paths.append(emcc_path)

        return paths


class PATHStrategy(DiscoveryStrategy):
    """
    Search for compilers in directories listed in the PATH environment variable.

    Iterates every directory in PATH and checks for files matching
    the candidate compiler names.

    Priority: 10 (standard system search)

    Warnings
    --------
    - Slow on systems with many PATH entries or network-mounted directories.
    - May find ccache, distcc, or other compiler wrappers.
      These are NOT real compilers and will be caught by validation.
    - On Windows, PATH may contain invalid or inaccessible entries.
      This strategy handles those gracefully.
    """

    name: ClassVar[str] = "system_path"
    priority: ClassVar[int] = 10

    def discover(self) -> List[str]:
        """
        Scan PATH directories for compiler executables.

        Returns
        -------
        List[str]
            Paths to files with compiler-like names found in PATH.
        """
        paths: List[str] = []
        path_env = os.environ.get("PATH", "")
        candidates = _get_candidates()

        for directory in path_env.split(os.pathsep):
            directory = directory.strip()
            if not directory or not os.path.isdir(directory):
                continue

            try:
                for entry in os.listdir(directory):
                    if entry.lower() in candidates:
                        full_path = os.path.join(directory, entry)
                        try:
                            if os.path.isfile(full_path):
                                paths.append(full_path)
                        except OSError:
                            continue
            except (PermissionError, OSError):
                continue

        return paths


class ExtraDirsStrategy(DiscoveryStrategy):
    """
    Search user-specified extra directories from environment variable.

    Reads COMPILER_SEARCH_DIRS env var (colon-separated on Unix,
    semicolon on Windows).

    Priority: 15 (runs between PATH and common dirs)

    User Instructions
    -----------------
    Set COMPILER_SEARCH_DIRS=/opt/custom/bin:/home/user/tools
    to add custom search paths without modifying PATH.
    """

    name: ClassVar[str] = "extra_dirs"
    priority: ClassVar[int] = 15

    def discover(self) -> List[str]:
        """
        Scan user-specified directories for compilers.

        Returns
        -------
        List[str]
            Paths found in COMPILER_SEARCH_DIRS directories.
        """
        paths: List[str] = []
        env_val = os.environ.get("COMPILER_SEARCH_DIRS", "")

        if not env_val:
            return paths

        separator = ";" if platform.system() == "Windows" else ":"
        dirs = [d.strip() for d in env_val.split(separator) if d.strip()]
        candidates = _get_candidates()

        for directory in dirs:
            if not os.path.isdir(directory):
                continue
            try:
                for candidate in candidates:
                    full_path = os.path.join(directory, candidate)
                    if os.path.isfile(full_path):
                        paths.append(os.path.abspath(full_path))
            except PermissionError:
                continue

        return paths


class CommonDirsStrategy(DiscoveryStrategy):
    """
    Search platform-specific standard installation directories.

    Checks directories where compilers are commonly installed
    but may not be in PATH.

    Directories checked by platform:
        Linux   : /usr/bin, /usr/local/bin, /opt/*
        macOS   : /usr/bin, /usr/local/bin, /opt/homebrew/bin,
                  /usr/local/opt/*/bin, /Applications/Xcode.app/*
        Windows : C:\\msys64\\mingw64\\bin, C:\\msys64\\ucrt64\\bin,
                  C:\\mingw-w64, C:\\msys2, Scoop/Chocolatey dirs

    Priority: 20 (runs after PATH)

    Warnings
    --------
    - Directory scanning can be slow, especially /opt on Linux
      and /usr/local/opt on macOS with many installed packages.
    - Some directories may not exist; this is handled silently.
    - This strategy is disabled by default on Windows due to
      potential performance impact. Enable explicitly if needed.
    """

    name: ClassVar[str] = "common_dirs"
    priority: ClassVar[int] = 20

    def discover(self) -> List[str]:
        """
        Scan common compiler installation directories.

        Returns
        -------
        List[str]
            Paths found in platform-specific standard directories.
        """
        system = platform.system()
        if system == "Linux":
            return self._scan_linux()
        elif system == "Darwin":
            return self._scan_macos()
        elif system == "Windows":
            return self._scan_windows()
        return []

    def _scan_linux(self) -> List[str]:
        """Scan common Linux compiler directories."""
        paths: List[str] = []
        candidates = _get_candidates()
        dirs_to_scan = ["/usr/bin", "/usr/local/bin"]

        # Add /opt subdirectories
        if os.path.isdir("/opt"):
            try:
                for entry in os.listdir("/opt"):
                    opt_subdir = os.path.join("/opt", entry, "bin")
                    if os.path.isdir(opt_subdir):
                        dirs_to_scan.append(opt_subdir)
            except PermissionError:
                pass

        for directory in dirs_to_scan:
            if not os.path.isdir(directory):
                continue
            try:
                for candidate in candidates:
                    full_path = os.path.join(directory, candidate)
                    if os.path.isfile(full_path):
                        paths.append(full_path)
            except PermissionError:
                continue

        return paths

    def _scan_macos(self) -> List[str]:
        """Scan common macOS compiler directories."""
        paths: List[str] = []
        candidates = _get_candidates()
        dirs_to_scan = [
            "/usr/bin",
            "/usr/local/bin",
            "/opt/homebrew/bin",
        ]

        # Homebrew opt directories
        homebrew_opt = "/usr/local/opt"
        if os.path.isdir(homebrew_opt):
            try:
                for entry in os.listdir(homebrew_opt):
                    opt_bin = os.path.join(homebrew_opt, entry, "bin")
                    if os.path.isdir(opt_bin):
                        dirs_to_scan.append(opt_bin)
            except PermissionError:
                pass

        # Xcode toolchains
        xcode_toolchains = "/Applications/Xcode.app/Contents/Developer/Toolchains"
        if os.path.isdir(xcode_toolchains):
            try:
                for toolchain in os.listdir(xcode_toolchains):
                    bin_dir = os.path.join(
                        xcode_toolchains, toolchain, "usr", "bin"
                    )
                    if os.path.isdir(bin_dir):
                        dirs_to_scan.append(bin_dir)
            except PermissionError:
                pass

        for directory in dirs_to_scan:
            if not os.path.isdir(directory):
                continue
            try:
                for candidate in candidates:
                    full_path = os.path.join(directory, candidate)
                    if os.path.isfile(full_path):
                        paths.append(full_path)
            except PermissionError:
                continue

        return paths

    def _scan_windows(self) -> List[str]:
        """
        Scan common Windows compiler directories.

        Checks MSYS2, MinGW-w64, and other common installation paths.

        Warnings
        --------
        This strategy performs filesystem scans on Windows which may
        be slow on some systems. It returns empty by default.
        Override in a subclass to enable for specific environments.
        """
        paths: List[str] = []
        candidates = _get_candidates()
        # Use double backslashes to avoid Unicode escape errors in string literals
        base_dirs = [
            r"C:\\msys64\\mingw64\\bin",
            r"C:\\msys64\\ucrt64\\bin",
            r"C:\\msys64\\clang64\\bin",
            r"C:\\mingw-w64",
            r"C:\\msys2",
        ]

        for base in base_dirs:
            if not os.path.isdir(base):
                continue
            bin_dir = base if base.endswith("bin") else os.path.join(base, "mingw64", "bin")
            if not os.path.isdir(bin_dir):
                # Try without mingw64 subdirectory
                for root, dirs, _ in os.walk(base):
                    for d in dirs:
                        if d == "bin":
                            bin_dir = os.path.join(root, d)
                            break
                    if os.path.isdir(bin_dir):
                        break
                else:
                    continue

            try:
                for candidate in candidates:
                    full_path = os.path.join(bin_dir, candidate)
                    exe_path = full_path + ".exe"
                    if os.path.isfile(full_path):
                        paths.append(full_path)
                    elif os.path.isfile(exe_path):
                        paths.append(exe_path)
            except PermissionError:
                continue

        return paths


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

class DiscoveryStrategyRegistry:
    """
    Plugin registry for discovery strategies.

    Stores all registered strategy classes and provides methods
    for enabling, disabling, and reordering them.

    Class Methods
    -------------
    register(strategy_cls) : Register a new strategy class
    get_enabled() : Get enabled strategies sorted by priority
    get_all() : Get all registered strategies regardless of state
    disable(name) : Disable a strategy by name
    enable(name) : Re-enable a disabled strategy
    set_order(names) : Reorder strategies by name list

    Usage
    -----
        # Register a custom strategy
        DiscoveryStrategyRegistry.register(MyStrategy)

        # Disable the slow common_dirs strategy
        DiscoveryStrategyRegistry.disable("common_dirs")

        # Override the execution order
        DiscoveryStrategyRegistry.set_order([
            "user_override",
            "system_path",
            "common_dirs",
        ])
    """

    _strategies: ClassVar[List[Type[DiscoveryStrategy]]] = []
    _disabled: ClassVar[Set[str]] = set()

    @classmethod
    def register(cls, strategy_cls: Type[DiscoveryStrategy]) -> None:
        """
        Register a strategy class for use in discovery.

        Parameters
        ----------
        strategy_cls : Type[DiscoveryStrategy]
            A concrete subclass of DiscoveryStrategy.

        Raises
        ------
        ValueError
            If a strategy with the same name is already registered.
        """
        if not strategy_cls.name:
            raise ValueError(
                f"Strategy {strategy_cls.__name__} must define a 'name' class attribute"
            )

        existing_names = [s.name for s in cls._strategies]
        if strategy_cls.name in existing_names:
            raise ValueError(
                f"Strategy with name '{strategy_cls.name}' is already registered"
            )

        cls._strategies.append(strategy_cls)

    @classmethod
    def get_enabled(cls) -> List[Type[DiscoveryStrategy]]:
        """
        Get all enabled strategies sorted by priority.

        Returns
        -------
        List[Type[DiscoveryStrategy]]
            Strategy classes that are not disabled, sorted by priority
            (lower number first).
        """
        enabled = [s for s in cls._strategies if s.name not in cls._disabled]
        enabled.sort(key=lambda s: s.priority)
        return enabled

    @classmethod
    def get_all(cls) -> List[Type[DiscoveryStrategy]]:
        """
        Get all registered strategies including disabled ones.

        Returns
        -------
        List[Type[DiscoveryStrategy]]
            All registered strategy classes in registration order.
        """
        return list(cls._strategies)

    @classmethod
    def disable(cls, name: str) -> None:
        """
        Disable a strategy by name.

        Disabled strategies are skipped during discovery.
        Does nothing if the strategy is already disabled.

        Parameters
        ----------
        name : str
            The strategy name to disable.
        """
        cls._disabled.add(name)

    @classmethod
    def enable(cls, name: str) -> None:
        """
        Re-enable a previously disabled strategy.

        Does nothing if the strategy is not disabled.

        Parameters
        ----------
        name : str
            The strategy name to enable.
        """
        cls._disabled.discard(name)

    @classmethod
    def set_order(cls, names: List[str]) -> None:
        """
        Reorder strategies by assigning sequential priorities.

        The first name in the list gets priority 0, second gets 1, etc.
        Strategies not in the list keep their current priority.

        Parameters
        ----------
        names : List[str]
            Strategy names in desired execution order.

        Warnings
        --------
        - Names that do not match any registered strategy are silently ignored.
        - This changes the priority class attribute permanently.
        """
        for index, name in enumerate(names):
            for strategy_cls in cls._strategies:
                if strategy_cls.name == name:
                    strategy_cls.priority = index
                    break


# ---------------------------------------------------------------------------
# Register default strategies
# ---------------------------------------------------------------------------

DiscoveryStrategyRegistry.register(UserOverrideStrategy)
DiscoveryStrategyRegistry.register(ManagedToolchainStrategy)
DiscoveryStrategyRegistry.register(PATHStrategy)
DiscoveryStrategyRegistry.register(ExtraDirsStrategy)
DiscoveryStrategyRegistry.register(CommonDirsStrategy)
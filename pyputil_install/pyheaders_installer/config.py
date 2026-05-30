#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration Management for Python Headers Installer.

This module provides configuration dataclasses and validation logic
for controlling all aspects of the header installation process.
It centralizes all configurable parameters and provides methods
for loading configurations from files, environment variables,
and command-line arguments.

The configuration system is designed to be:
- Type-safe: All parameters have explicit types
- Validatable: Each parameter can be validated independently
- Serializable: Configurations can be saved and loaded
- Immutable: Frozen dataclasses prevent accidental modification
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from enum import Enum, auto
import platform
import os
import json
import sysconfig


class DownloadSource(str, Enum):
    """
    Supported download sources for Python source code.
    
    Each source provides different URL patterns and archive formats.
    """
    GITHUB = "github"
    PYTHON_ORG = "python.org"
    CUSTOM = "custom"
    
    def get_default_url_template(self) -> str:
        """
        Get the default URL template for this source.
        
        Returns
        -------
        str
            URL template with {version} placeholder
        """
        templates = {
            DownloadSource.GITHUB: "https://github.com/python/cpython/archive/refs/tags/v{version}.zip",
            DownloadSource.PYTHON_ORG: "https://www.python.org/ftp/python/{version}/Python-{version}.tar.xz",
        }
        return templates.get(self, "")
    
    def get_archive_extension(self) -> str:
        """
        Get the expected archive extension for this source.
        
        Returns
        -------
        str
            Archive file extension
        """
        extensions = {
            DownloadSource.GITHUB: ".zip",
            DownloadSource.PYTHON_ORG: ".tar.xz",
        }
        return extensions.get(self, "")


class HashAlgorithm(str, Enum):
    """
    Supported hash algorithms for file verification.
    """
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    MD5 = "md5"
    BLAKE2B = "blake2b"
    
    def get_hash_function(self):
        """
        Get the corresponding hashlib function.
        
        Returns
        -------
        callable
            Hash function from hashlib
        """
        import hashlib
        mapping = {
            HashAlgorithm.SHA256: hashlib.sha256,
            HashAlgorithm.SHA384: hashlib.sha384,
            HashAlgorithm.SHA512: hashlib.sha512,
            HashAlgorithm.MD5: hashlib.md5,
            HashAlgorithm.BLAKE2B: hashlib.blake2b,
        }
        return mapping[self]


class LogLevel(str, Enum):
    """
    Log levels matching the standard logging module.
    """
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    def to_logging_level(self) -> int:
        """
        Convert to logging module integer level.
        
        Returns
        -------
        int
            Logging level constant
        """
        import logging
        mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }
        return mapping[self]


class PlatformPreset(str, Enum):
    """
    Predefined platform configurations for header paths.
    """
    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "macos"
    TERMUX = "termux"
    AUTO = "auto"
    
    @classmethod
    def detect(cls) -> "PlatformPreset":
        """
        Auto-detect the current platform.
        
        Returns
        -------
        PlatformPreset
            Detected platform preset
        
        Notes
        -----
        Detects Termux by checking for the TERMUX_VERSION environment variable.
        """
        system = platform.system().lower()
        
        # Check for Termux first (runs on Android/Linux)
        if "termux" in os.environ.get("PREFIX", "").lower() or \
           "termux" in os.environ.get("HOME", "").lower():
            return cls.TERMUX
        
        mapping = {
            "linux": cls.LINUX,
            "windows": cls.WINDOWS,
            "darwin": cls.MACOS,
        }
        return mapping.get(system, cls.LINUX)


@dataclass(frozen=True)
class NetworkConfig:
    """
    Configuration for network operations.
    
    Controls all aspects of downloading files from remote sources
    including retry behavior, timeouts, and connection settings.
    
    Attributes
    ----------
    retries : int
        Maximum number of download retry attempts
    retry_delay : float
        Base delay between retries in seconds (exponential backoff applied)
    timeout : float
        Connection timeout in seconds
    chunk_size : int
        Download chunk size in bytes for streaming
    max_redirects : int
        Maximum number of HTTP redirects to follow
    user_agent : str
        User-Agent header for HTTP requests
    """
    retries: int = 3
    retry_delay: float = 2.0
    timeout: float = 30.0
    chunk_size: int = 8192
    max_redirects: int = 5
    user_agent: str = "PythonHeadersInstaller/2.0"
    
    def __post_init__(self) -> None:
        """Validate network configuration parameters."""
        if self.retries < 0:
            raise ValueError(f"retries must be >= 0, got {self.retries}")
        if self.retry_delay < 0:
            raise ValueError(f"retry_delay must be >= 0, got {self.retry_delay}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {self.timeout}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {self.chunk_size}")


@dataclass(frozen=True)
class VerificationConfig:
    """
    Configuration for file verification.
    
    Controls how downloaded files are verified for integrity
    and authenticity.
    
    Attributes
    ----------
    enabled : bool
        Whether verification is performed
    algorithm : HashAlgorithm
        Hash algorithm to use
    hash_url : Optional[str]
        URL to fetch expected hashes from (with {version} placeholder)
    skip_on_failure : bool
        If True, continue installation even if verification fails
    """
    enabled: bool = False
    algorithm: HashAlgorithm = HashAlgorithm.SHA256
    hash_url: Optional[str] = None
    skip_on_failure: bool = False
    
    def __post_init__(self) -> None:
        """Validate verification configuration."""
        if self.enabled and self.hash_url is None:
            # Default hash URL for Python releases
            object.__setattr__(
                self,
                'hash_url',
                "https://www.python.org/ftp/python/{version}/Python-{version}.tar.xz.sha256"
            )


@dataclass(frozen=True)
class BackupConfig:
    """
    Configuration for backup operations.
    
    Controls backup creation, rotation, and restoration behavior.
    
    Attributes
    ----------
    enabled : bool
        Whether backups are created
    suffix : str
        Suffix appended to backup directory names
    max_backups : int
        Maximum number of backups to keep (0 = unlimited)
    compress : bool
        Whether to compress backup archives
    include_timestamp : bool
        Whether to include timestamp in backup name
    """
    enabled: bool = True
    suffix: str = ".backup"
    max_backups: int = 5
    compress: bool = False
    include_timestamp: bool = False
    
    def __post_init__(self) -> None:
        """Validate backup configuration."""
        if self.max_backups < 0:
            raise ValueError(f"max_backups must be >= 0, got {self.max_backups}")


@dataclass(frozen=True)
class PathConfig:
    """
    Configuration for file system paths.
    
    Centralizes all path-related settings including target directories
    and temporary storage locations.
    
    Attributes
    ----------
    target_include_dir : Optional[Path]
        Target directory for header installation (auto-detect if None)
    temp_dir : Optional[Path]
        Directory for temporary files (system default if None)
    platform_preset : PlatformPreset
        Platform preset for path resolution
    """
    target_include_dir: Optional[Path] = None
    temp_dir: Optional[Path] = None
    platform_preset: PlatformPreset = PlatformPreset.AUTO
    
    def __post_init__(self) -> None:
        """Resolve auto-detected paths."""
        if self.platform_preset == PlatformPreset.AUTO:
            object.__setattr__(self, 'platform_preset', PlatformPreset.detect())
    
    def resolve_target_dir(self) -> Path:
        """
        Resolve the target include directory.
        
        Returns
        -------
        Path
            Resolved target directory path
        
        Notes
        -----
        Path resolution follows this priority:
        1. Explicitly set target_include_dir
        2. sysconfig.get_paths()["include"] for standard platforms
        3. Platform-specific defaults for non-standard environments
        """
        if self.target_include_dir is not None:
            return self.target_include_dir
        
        # Try sysconfig first
        try:
            include_path = sysconfig.get_paths().get("include")
            if include_path:
                return Path(include_path)
        except Exception:
            pass
        
        # Platform-specific fallbacks
        if self.platform_preset == PlatformPreset.TERMUX:
            return Path(os.environ.get("PREFIX", "/data/data/com.termux/files/usr")) / "include"
        elif self.platform_preset == PlatformPreset.WINDOWS:
            return Path(sys.prefix) / "include"
        
        # Final fallback
        return Path("/usr/include")
    
    def resolve_temp_dir(self) -> Path:
        """
        Resolve the temporary directory.
        
        Returns
        -------
        Path
            Resolved temporary directory path
        """
        if self.temp_dir is not None:
            return self.temp_dir
        return Path(__import__('tempfile').gettempdir())


@dataclass(frozen=True)
class ExtractionConfig:
    """
    Configuration for archive extraction.
    
    Controls how source archives are extracted and processed.
    
    Attributes
    ----------
    preserve_permissions : bool
        Whether to preserve file permissions during extraction
    filter_patterns : List[str]
        Glob patterns for files to include (empty = all files)
    max_file_size : Optional[int]
        Maximum file size to extract in bytes (None = unlimited)
    """
    preserve_permissions: bool = True
    filter_patterns: List[str] = field(default_factory=list)
    max_file_size: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate extraction configuration."""
        if self.max_file_size is not None and self.max_file_size <= 0:
            raise ValueError(f"max_file_size must be > 0, got {self.max_file_size}")


@dataclass(frozen=True)
class InstallConfig:
    """
    Master configuration for header installation.
    
    Aggregates all sub-configurations into a single configuration object
    that controls the entire installation process.
    
    Attributes
    ----------
    version : Optional[str]
        Python version to install headers for (None = current version)
    source : DownloadSource
        Source repository for downloading Python source
    custom_url : Optional[str]
        Custom URL template (required if source is CUSTOM)
    clean_existing : bool
        Whether to remove existing headers before installation
    include_subdirs : bool
        Whether to include subdirectories when copying headers
    verbose : bool
        Enable verbose output
    network : NetworkConfig
        Network operation configuration
    verification : VerificationConfig
        File verification configuration
    backup : BackupConfig
        Backup operation configuration
    paths : PathConfig
        File system path configuration
    extraction : ExtractionConfig
        Archive extraction configuration
    log_level : LogLevel
        Logging verbosity level
    """
    version: Optional[str] = None
    source: DownloadSource = DownloadSource.GITHUB
    custom_url: Optional[str] = None
    clean_existing: bool = False
    include_subdirs: bool = True
    verbose: bool = False
    network: NetworkConfig = field(default_factory=NetworkConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    log_level: LogLevel = LogLevel.INFO
    
    def __post_init__(self) -> None:
        """Validate master configuration."""
        if self.source == DownloadSource.CUSTOM and not self.custom_url:
            raise ValueError("custom_url is required when source is CUSTOM")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a nested dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the configuration
        """
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if hasattr(value, '__dataclass_fields__'):
                result[field_name] = asdict(value)
            elif isinstance(value, Enum):
                result[field_name] = value.value
            elif isinstance(value, Path):
                result[field_name] = str(value)
            else:
                result[field_name] = value
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """
        Serialize configuration to JSON string.
        
        Parameters
        ----------
        indent : int
            JSON indentation level
        
        Returns
        -------
        str
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> "InstallConfig":
        """
        Create configuration from JSON string.
        
        Parameters
        ----------
        json_str : str
            JSON string containing configuration
        
        Returns
        -------
        InstallConfig
            Parsed configuration object
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstallConfig":
        """
        Create configuration from dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing configuration values
        
        Returns
        -------
        InstallConfig
            Constructed configuration object
        
        Notes
        -----
        Handles nested configurations and converts string values
        to appropriate enum types where necessary.
        """
        # Parse enums
        if 'source' in data and isinstance(data['source'], str):
            data['source'] = DownloadSource(data['source'])
        if 'log_level' in data and isinstance(data['log_level'], str):
            data['log_level'] = LogLevel(data['log_level'])
        
        # Parse nested configs
        nested_configs = {
            'network': NetworkConfig,
            'verification': VerificationConfig,
            'backup': BackupConfig,
            'paths': PathConfig,
            'extraction': ExtractionConfig,
        }
        
        for key, config_class in nested_configs.items():
            if key in data and isinstance(data[key], dict):
                # Parse enums within nested configs
                if key == 'paths' and 'platform_preset' in data[key]:
                    if isinstance(data[key]['platform_preset'], str):
                        data[key]['platform_preset'] = PlatformPreset(data[key]['platform_preset'])
                if key == 'verification' and 'algorithm' in data[key]:
                    if isinstance(data[key]['algorithm'], str):
                        data[key]['algorithm'] = HashAlgorithm(data[key]['algorithm'])
                if key == 'verification' and 'hash_url' in data[key]:
                    data[key]['hash_url'] = data[key].get('hash_url') or None
                
                data[key] = config_class(**data[key])
        
        return cls(**data)


@dataclass(frozen=True)
class SystemInfo:
    """
    Immutable snapshot of system information.
    
    Captures relevant system details at runtime for logging
    and debugging purposes.
    
    Attributes
    ----------
    os_name : str
        Operating system name
    os_version : str
        Operating system version
    architecture : str
        Machine architecture
    python_version : str
        Python interpreter version
    python_implementation : str
        Python implementation name
    python_path : Path
        Path to Python executable
    """
    os_name: str = field(default_factory=lambda: platform.system())
    os_version: str = field(default_factory=lambda: platform.release())
    architecture: str = field(default_factory=lambda: platform.machine())
    python_version: str = field(default_factory=lambda: platform.python_version())
    python_implementation: str = field(default_factory=lambda: platform.python_implementation())
    python_path: Path = field(default_factory=lambda: Path(sys.executable))
    
    def to_log_string(self) -> str:
        """
        Format system information for logging.
        
        Returns
        -------
        str
            Formatted system information string
        """
        lines = [
            f"System Information:",
            f"  OS: {self.os_name} {self.os_version}",
            f"  Architecture: {self.architecture}",
            f"  Python: {self.python_implementation} {self.python_version}",
            f"  Executable: {self.python_path}",
        ]
        return "\n".join(lines)
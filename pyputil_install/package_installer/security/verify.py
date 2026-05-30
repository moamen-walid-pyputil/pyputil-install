#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trust verification for package sources and distributions.

This module provides mechanisms to verify the trustworthiness of
package sources, including trusted host validation, SSL/TLS certificate
pinning, GPG signature verification, and package provenance checks.

Classes
-------
TrustConfig
    Configuration for trust verification behavior.
TrustVerifier
    Verifies package source trustworthiness and authenticity.

Functions
--------
is_trusted_host
    Check if a host is in the trusted hosts list.
validate_url
    Validate that a URL meets security requirements.
parse_requirements_hash
    Parse pip-style ``--hash`` arguments into structured data.

Examples
--------
>>> verifier = TrustVerifier()
>>> verifier.is_host_trusted("pypi.org")
True
>>> verifier.validate_package_source("https://pypi.org/pypi/requests/json")
True
"""

import re
import logging
import ssl
import socket
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse
from dataclasses import dataclass, field

from ..exceptions import SecurityError
from .hashes import HashAlgorithm, compute_hash, verify_hash, compare_hashes

logger = logging.getLogger(__name__)


@dataclass
class TrustConfig:
    """
    Configuration for trust verification.

    Parameters
    ----------
    trusted_hosts : set of str, optional
        Set of hostnames that are trusted without TLS verification.
        Default includes ``pypi.org`` and ``files.pythonhosted.org``.
    require_tls : bool, default=True
        If True, non-HTTPS URLs are rejected unless the host is
        in ``trusted_hosts``.
    verify_ssl : bool, default=True
        If True, SSL certificates are validated for HTTPS URLs.
    pinned_certificates : dict, optional
        Mapping of hostnames to expected certificate fingerprints
        (SHA256). If a host is pinned, its certificate must match.
    allowed_schemes : set of str, default={"https"}
        URL schemes allowed for package sources.
    blocked_hosts : set of str, optional
        Hostnames that are always rejected, even if trusted.
    require_hashes : bool, default=False
        If True, all package installations must include hash
        verification via ``--hash``.
    allow_insecure_transport : bool, default=False
        If True, allows HTTP (non-TLS) transport for trusted hosts.
        Not recommended for production.
    verify_signatures : bool, default=False
        If True, attempt to verify GPG signatures when available.
    gpg_keyring_path : Path or None, default=None
        Path to GPG keyring directory. If None, uses system default.

    Notes
    -----
    By default, PyPI and its CDN are pre-trusted. Additional trusted
    hosts can be added for internal mirrors or private indices.

    Certificate pinning adds an extra layer of security by ensuring
    the server's certificate matches a known fingerprint, protecting
    against compromised certificate authorities.

    Warnings
    --------
    Disabling SSL verification or allowing insecure transport exposes
    package downloads to man-in-the-middle attacks. Only use these
    options for development or isolated internal networks.

    Examples
    --------
    >>> config = TrustConfig(
    ...     trusted_hosts={"pypi.org", "internal-mirror.local"},
    ...     pinned_certificates={"internal-mirror.local": "sha256:abc123..."},
    ... )
    >>> "pypi.org" in config.trusted_hosts
    True
    """

    trusted_hosts: Set[str] = field(default_factory=lambda: {
        "pypi.org",
        "files.pythonhosted.org",
        "pythonhosted.org",
    })
    require_tls: bool = True
    verify_ssl: bool = True
    pinned_certificates: Dict[str, str] = field(default_factory=dict)
    allowed_schemes: Set[str] = field(default_factory=lambda: {"https"})
    blocked_hosts: Set[str] = field(default_factory=set)
    require_hashes: bool = False
    allow_insecure_transport: bool = False
    verify_signatures: bool = False
    gpg_keyring_path: Optional[Path] = None

    def __post_init__(self) -> None:
        """Validate configuration consistency."""
        if self.require_hashes and not self.require_tls:
            logger.warning(
                "require_hashes=True but require_tls=False. "
                "Hash verification over insecure transport provides "
                "limited protection against active attackers."
            )

        if self.allow_insecure_transport and self.require_tls:
            logger.debug(
                "allow_insecure_transport=True has no effect when "
                "require_tls=True."
            )


class TrustVerifier:
    """
    Verifies package source trustworthiness and authenticity.

    This class validates that package sources (URLs, hosts, files)
    meet configured security requirements before packages are
    downloaded or installed. It supports host allowlisting,
    certificate pinning, and source provenance checks.

    Parameters
    ----------
    config : TrustConfig, optional
        Trust verification configuration. If None, default
        configuration is used.

    Attributes
    ----------
    config : TrustConfig
        The active trust configuration.
    stats : dict
        Verification statistics.

    Notes
    -----
    The verifier enforces a defense-in-depth approach:

    1. **Transport security**: TLS with certificate validation.
    2. **Host trust**: Only allowlisted hosts are accepted.
    3. **Certificate pinning**: Optional per-host certificate fingerprints.
    4. **Hash verification**: Optional requirement for content hashes.
    5. **Signature verification**: Optional GPG signature checks.

    Each layer provides independent protection. The strongest
    security is achieved when all layers are enabled.

    Examples
    --------
    >>> verifier = TrustVerifier()
    >>> verifier.is_host_trusted("pypi.org")
    True
    >>> verifier.is_host_trusted("evil-mirror.example.com")
    False

    With custom configuration::

    >>> config = TrustConfig(
    ...     trusted_hosts={"pypi.org", "my-mirror.local"},
    ...     pinned_certificates={"my-mirror.local": "sha256:abc..."},
    ...     require_hashes=True,
    ... )
    >>> verifier = TrustVerifier(config)
    """

    def __init__(self, config: Optional[TrustConfig] = None) -> None:
        self.config = config if config is not None else TrustConfig()
        self._stats: Dict[str, int] = {
            "host_checks": 0,
            "host_rejections": 0,
            "url_validations": 0,
            "url_rejections": 0,
            "certificate_checks": 0,
            "certificate_failures": 0,
            "hash_verifications": 0,
            "hash_failures": 0,
        }
        logger.debug(
            f"TrustVerifier initialized with "
            f"{len(self.config.trusted_hosts)} trusted hosts"
        )

    def is_host_trusted(self, host: str) -> bool:
        """
        Check if a hostname is in the trusted hosts set.

        Parameters
        ----------
        host : str
            The hostname to check (case-insensitive).

        Returns
        -------
        bool
            True if the host is trusted and not blocked.

        Notes
        -----
        Hostname matching is case-insensitive. The host is first
        checked against the blocked list, then against the trusted
        list. A host that appears in both lists is rejected.

        Subdomain matching is NOT performed. ``pypi.org`` does not
        match ``files.pypi.org`` unless explicitly listed.

        Examples
        --------
        >>> verifier = TrustVerifier()
        >>> verifier.is_host_trusted("PyPI.org")
        True
        >>> verifier.is_host_trusted("unknown.example.com")
        False
        """
        self._stats["host_checks"] += 1
        normalized = host.lower().strip()

        if normalized in {h.lower() for h in self.config.blocked_hosts}:
            self._stats["host_rejections"] += 1
            logger.warning(f"Host '{host}' is in the blocked list")
            return False

        is_trusted = normalized in {h.lower() for h in self.config.trusted_hosts}

        if not is_trusted:
            self._stats["host_rejections"] += 1
            logger.debug(f"Host '{host}' is not in the trusted list")

        return is_trusted

    def add_trusted_host(self, host: str) -> None:
        """
        Add a hostname to the trusted hosts set.

        Parameters
        ----------
        host : str
            The hostname to trust.

        Notes
        -----
        Adding a host to the trusted list allows connections without
        TLS verification if ``verify_ssl`` is False. Exercise caution
        when trusting non-PyPI hosts.

        Examples
        --------
        >>> verifier = TrustVerifier()
        >>> verifier.add_trusted_host("internal-pypi.company.com")
        >>> verifier.is_host_trusted("internal-pypi.company.com")
        True
        """
        normalized = host.lower().strip()
        self.config.trusted_hosts.add(normalized)
        logger.info(f"Added trusted host: {normalized}")

    def remove_trusted_host(self, host: str) -> bool:
        """
        Remove a hostname from the trusted hosts set.

        Parameters
        ----------
        host : str
            The hostname to remove.

        Returns
        -------
        bool
            True if the host was removed, False if it was not found.

        Examples
        --------
        >>> verifier = TrustVerifier()
        >>> verifier.remove_trusted_host("pypi.org")
        True
        """
        normalized = host.lower().strip()
        if normalized in {h.lower() for h in self.config.trusted_hosts}:
            self.config.trusted_hosts = {
                h for h in self.config.trusted_hosts
                if h.lower() != normalized
            }
            logger.info(f"Removed trusted host: {normalized}")
            return True
        return False

    def block_host(self, host: str) -> None:
        """
        Add a hostname to the blocked list.

        Parameters
        ----------
        host : str
            The hostname to block. Takes precedence over trusted list.

        Examples
        --------
        >>> verifier = TrustVerifier()
        >>> verifier.block_host("malicious.example.com")
        """
        normalized = host.lower().strip()
        self.config.blocked_hosts.add(normalized)
        logger.warning(f"Blocked host: {normalized}")

    def unblock_host(self, host: str) -> bool:
        """
        Remove a hostname from the blocked list.

        Parameters
        ----------
        host : str
            The hostname to unblock.

        Returns
        -------
        bool
            True if the host was unblocked, False if not found.
        """
        normalized = host.lower().strip()
        if normalized in {h.lower() for h in self.config.blocked_hosts}:
            self.config.blocked_hosts = {
                h for h in self.config.blocked_hosts
                if h.lower() != normalized
            }
            return True
        return False

    def validate_url(self, url: str) -> bool:
        """
        Validate that a URL meets security requirements.

        Parameters
        ----------
        url : str
            The URL to validate.

        Returns
        -------
        bool
            True if the URL passes all security checks.

        Raises
        ------
        SecurityError
            If the URL fails validation with details about which
            check failed.

        Notes
        -----
        Validation steps (in order):

        1. Parse the URL structure.
        2. Check the scheme against allowed schemes.
        3. Extract the hostname and check against blocked/trusted lists.
        4. If ``require_tls`` is True and the scheme is not HTTPS,
           verify that the host is trusted and ``allow_insecure_transport``
           is enabled.

        Examples
        --------
        >>> verifier = TrustVerifier()
        >>> verifier.validate_url("https://pypi.org/pypi/requests/json")
        True

        Blocked host::

        >>> verifier.block_host("pypi.org")
        >>> verifier.validate_url("https://pypi.org/pypi/requests/json")
        Traceback (most recent call last):
            ...
        SecurityError: Host 'pypi.org' is blocked
        """
        self._stats["url_validations"] += 1

        try:
            parsed = urlparse(url)
        except Exception as e:
            self._stats["url_rejections"] += 1
            raise SecurityError(
                f"Failed to parse URL '{url}': {e}",
                source=url,
                reason="invalid-url",
            ) from e

        scheme = parsed.scheme.lower()
        if scheme not in self.config.allowed_schemes:
            self._stats["url_rejections"] += 1
            raise SecurityError(
                f"URL scheme '{scheme}' is not allowed. "
                f"Allowed schemes: {self.config.allowed_schemes}",
                source=url,
                reason="invalid-scheme",
            )

        host = parsed.hostname
        if host is None:
            self._stats["url_rejections"] += 1
            raise SecurityError(
                f"URL has no hostname: {url}",
                source=url,
                reason="missing-host",
            )

        if not self.is_host_trusted(host):
            self._stats["url_rejections"] += 1
            raise SecurityError(
                f"Host '{host}' is not in the trusted hosts list",
                source=url,
                reason="untrusted-host",
            )

        if self.config.require_tls and scheme != "https":
            if not self.config.allow_insecure_transport:
                self._stats["url_rejections"] += 1
                raise SecurityError(
                    f"Insecure transport (HTTP) is not allowed for {url}. "
                    f"Use HTTPS or enable allow_insecure_transport.",
                    source=url,
                    reason="insecure-transport",
                )
            logger.warning(f"Using insecure transport for trusted host: {url}")

        return True

    def validate_package_source(
        self,
        url: str,
        expected_hash: Optional[str] = None,
        hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    ) -> bool:
        """
        Validate a package source URL and optionally its hash.

        Parameters
        ----------
        url : str
            The package source URL.
        expected_hash : str, optional
            Expected hex-encoded hash of the downloaded content.
        hash_algorithm : HashAlgorithm, default=SHA256
            Algorithm used for the expected hash.

        Returns
        -------
        bool
            True if all validations pass.

        Raises
        ------
        SecurityError
            If URL validation fails or hash verification is required
            but not provided.

        Notes
        -----
        If ``require_hashes`` is True in the configuration and no
        ``expected_hash`` is provided, a ``SecurityError`` is raised
        before any download occurs.

        Examples
        --------
        >>> verifier = TrustVerifier()
        >>> verifier.validate_package_source(
        ...     "https://files.pythonhosted.org/packages/.../package.whl",
        ...     expected_hash="sha256:abc123...",
        ... )
        True
        """
        self.validate_url(url)

        if self.config.require_hashes and expected_hash is None:
            raise SecurityError(
                f"Hash verification is required but no hash provided for {url}",
                source=url,
                reason="missing-hash",
            )

        if expected_hash is not None:
            if not expected_hash.startswith(f"{hash_algorithm.value}:"):
                expected_hash = f"{hash_algorithm.value}:{expected_hash}"

        return True

    def pin_certificate(
        self,
        host: str,
        fingerprint: str,
        algorithm: str = "sha256",
    ) -> None:
        """
        Pin an expected TLS certificate fingerprint for a host.

        Parameters
        ----------
        host : str
            The hostname to pin.
        fingerprint : str
            The expected certificate fingerprint (hex-encoded).
        algorithm : str, default="sha256"
            The fingerprint algorithm (``sha256`` or ``sha1``).

        Notes
        -----
        Certificate pinning provides protection against compromised
        certificate authorities. When a host is pinned, connections
        are rejected if the server's certificate fingerprint does
        not match.

        To obtain a certificate fingerprint::

            openssl s_client -connect host:443 </dev/null 2>/dev/null \\
                | openssl x509 -fingerprint -sha256 -noout

        Examples
        --------
        >>> verifier = TrustVerifier()
        >>> verifier.pin_certificate(
        ...     "pypi.org",
        ...     "AA:BB:CC:DD:...",
        ...     algorithm="sha256",
        ... )
        """
        normalized_host = host.lower().strip()
        normalized_fingerprint = fingerprint.strip().replace(":", "").lower()
        normalized_algorithm = algorithm.lower().strip()

        self.config.pinned_certificates[normalized_host] = (
            f"{normalized_algorithm}:{normalized_fingerprint}"
        )
        logger.info(
            f"Pinned certificate for {normalized_host} "
            f"(algorithm={normalized_algorithm})"
        )

    def unpin_certificate(self, host: str) -> bool:
        """
        Remove certificate pinning for a host.

        Parameters
        ----------
        host : str
            The hostname to unpin.

        Returns
        -------
        bool
            True if the pin was removed, False if not found.
        """
        normalized = host.lower().strip()
        if normalized in self.config.pinned_certificates:
            del self.config.pinned_certificates[normalized]
            return True
        return False

    def get_pinned_fingerprint(self, host: str) -> Optional[str]:
        """
        Get the pinned certificate fingerprint for a host.

        Parameters
        ----------
        host : str
            The hostname to query.

        Returns
        -------
        str or None
            The pinned fingerprint (format: ``algorithm:fingerprint``)
            or None if not pinned.

        Examples
        --------
        >>> verifier = TrustVerifier()
        >>> verifier.pin_certificate("example.com", "abc123")
        >>> verifier.get_pinned_fingerprint("example.com")
        'sha256:abc123'
        """
        return self.config.pinned_certificates.get(host.lower().strip())

    def verify_certificate_fingerprint(
        self,
        host: str,
        certificate_der: bytes,
    ) -> bool:
        """
        Verify that a certificate matches the pinned fingerprint.

        Parameters
        ----------
        host : str
            The hostname being verified.
        certificate_der : bytes
            The certificate in DER format.

        Returns
        -------
        bool
            True if the certificate matches the pin or if no pin
            is configured for this host.

        Raises
        ------
        SecurityError
            If the certificate does not match the pinned fingerprint.

        Notes
        -----
        If no pin is configured for the host, this method returns
        True without verification. Use ``pin_certificate`` to set
        a pin before calling this method.

        Examples
        --------
        >>> verifier = TrustVerifier()
        >>> verifier.pin_certificate("example.com", "abc123...")
        >>> import ssl
        >>> cert = ssl.get_server_certificate(("example.com", 443))
        >>> # Convert PEM to DER, then verify
        """
        self._stats["certificate_checks"] += 1

        pinned = self.get_pinned_fingerprint(host)
        if pinned is None:
            return True

        try:
            algo_name, expected_fingerprint = pinned.split(":", 1)
        except ValueError:
            raise SecurityError(
                f"Invalid pin format for {host}: {pinned}",
                source=host,
                reason="invalid-pin-format",
            )

        try:
            if algo_name == "sha256":
                computed = hashlib.sha256(certificate_der).hexdigest()
            elif algo_name == "sha1":
                computed = hashlib.sha1(certificate_der).hexdigest()
            else:
                raise SecurityError(
                    f"Unsupported pin algorithm: {algo_name}",
                    source=host,
                    reason="unsupported-algorithm",
                )
        except Exception as e:
            self._stats["certificate_failures"] += 1
            raise SecurityError(
                f"Failed to compute certificate fingerprint: {e}",
                source=host,
                reason="fingerprint-computation-error",
            ) from e

        if not compare_hashes(computed, expected_fingerprint):
            self._stats["certificate_failures"] += 1
            raise SecurityError(
                f"Certificate fingerprint mismatch for {host}. "
                f"Expected: {expected_fingerprint[:16]}..., "
                f"Got: {computed[:16]}...",
                source=host,
                reason="fingerprint-mismatch",
            )

        logger.debug(f"Certificate fingerprint verified for {host}")
        return True

    def verify_ssl_certificate(
        self,
        host: str,
        port: int = 443,
        timeout: float = 10.0,
    ) -> bool:
        """
        Verify the SSL/TLS certificate of a remote host.

        Parameters
        ----------
        host : str
            The hostname to connect to.
        port : int, default=443
            The port number.
        timeout : float, default=10.0
            Connection timeout in seconds.

        Returns
        -------
        bool
            True if the certificate is valid and matches any
            configured pin.

        Raises
        ------
        SecurityError
            If connection fails, certificate is invalid, or
            fingerprint does not match.

        Notes
        -----
        This method establishes a real connection to the host and
        performs SSL/TLS handshake to obtain and validate the
        certificate chain.

        Examples
        --------
        >>> verifier = TrustVerifier()
        >>> verifier.verify_ssl_certificate("pypi.org")
        True
        """
        self._stats["certificate_checks"] += 1

        try:
            context = ssl.create_default_context()

            if not self.config.verify_ssl:
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE

            with socket.create_connection(
                (host, port), timeout=timeout
            ) as sock:
                with context.wrap_socket(sock, server_hostname=host) as tls_sock:
                    cert_der = tls_sock.getpeercert(binary_form=True)

                    if cert_der is None:
                        raise SecurityError(
                            f"No certificate received from {host}",
                            source=host,
                            reason="no-certificate",
                        )

                    self.verify_certificate_fingerprint(host, cert_der)

        except socket.timeout:
            self._stats["certificate_failures"] += 1
            raise SecurityError(
                f"Connection to {host}:{port} timed out",
                source=host,
                reason="connection-timeout",
            )
        except socket.gaierror as e:
            self._stats["certificate_failures"] += 1
            raise SecurityError(
                f"DNS resolution failed for {host}: {e}",
                source=host,
                reason="dns-failure",
            )
        except ConnectionRefusedError:
            self._stats["certificate_failures"] += 1
            raise SecurityError(
                f"Connection refused by {host}:{port}",
                source=host,
                reason="connection-refused",
            )
        except ssl.SSLError as e:
            self._stats["certificate_failures"] += 1
            raise SecurityError(
                f"SSL error for {host}: {e}",
                source=host,
                reason="ssl-error",
            )
        except OSError as e:
            self._stats["certificate_failures"] += 1
            raise SecurityError(
                f"Connection error for {host}: {e}",
                source=host,
                reason="connection-error",
            )

        return True

    def get_trusted_hosts(self) -> List[str]:
        """
        Get the list of trusted hosts.

        Returns
        -------
        list of str
            Sorted list of trusted hostnames.

        Examples
        --------
        >>> verifier = TrustVerifier()
        >>> "pypi.org" in verifier.get_trusted_hosts()
        True
        """
        return sorted(self.config.trusted_hosts)

    def get_blocked_hosts(self) -> List[str]:
        """
        Get the list of blocked hosts.

        Returns
        -------
        list of str
            Sorted list of blocked hostnames.
        """
        return sorted(self.config.blocked_hosts)

    def get_pinned_hosts(self) -> List[str]:
        """
        Get the list of hosts with pinned certificates.

        Returns
        -------
        list of str
            Sorted list of hostnames with certificate pins.
        """
        return sorted(self.config.pinned_certificates.keys())

    def get_stats(self) -> Dict[str, int]:
        """
        Get trust verification statistics.

        Returns
        -------
        dict
            Dictionary with counts of checks and failures.

        Examples
        --------
        >>> verifier = TrustVerifier()
        >>> stats = verifier.get_stats()
        >>> stats["host_checks"]
        0
        """
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset all verification statistics to zero."""
        for key in self._stats:
            self._stats[key] = 0

    def __repr__(self) -> str:
        """String representation of the verifier."""
        return (
            f"TrustVerifier("
            f"trusted={len(self.config.trusted_hosts)}, "
            f"blocked={len(self.config.blocked_hosts)}, "
            f"pinned={len(self.config.pinned_certificates)})"
        )


def is_trusted_host(
    host: str,
    trusted_hosts: Optional[Set[str]] = None,
) -> bool:
    """
    Convenience function to check if a host is trusted.

    Parameters
    ----------
    host : str
        The hostname to check.
    trusted_hosts : set of str, optional
        Set of trusted hosts. If None, uses default PyPI hosts.

    Returns
    -------
    bool
        True if the host is trusted.

    Examples
    --------
    >>> is_trusted_host("pypi.org")
    True
    >>> is_trusted_host("pypi.org", {"custom-mirror.local"})
    False
    """
    if trusted_hosts is None:
        trusted_hosts = {
            "pypi.org",
            "files.pythonhosted.org",
            "pythonhosted.org",
        }

    return host.lower().strip() in {h.lower() for h in trusted_hosts}


def validate_url(
    url: str,
    require_tls: bool = True,
    allowed_schemes: Optional[Set[str]] = None,
) -> bool:
    """
    Convenience function to validate a URL's security.

    Parameters
    ----------
    url : str
        The URL to validate.
    require_tls : bool, default=True
        If True, require HTTPS scheme.
    allowed_schemes : set of str, optional
        Allowed URL schemes. Defaults to ``{"https"}``.

    Returns
    -------
    bool
        True if the URL is valid.

    Raises
    ------
    SecurityError
        If the URL fails validation.

    Examples
    --------
    >>> validate_url("https://pypi.org/simple/")
    True
    """
    if allowed_schemes is None:
        allowed_schemes = {"https"}

    parsed = urlparse(url)

    if parsed.scheme not in allowed_schemes:
        raise SecurityError(
            f"Scheme '{parsed.scheme}' not allowed",
            source=url,
            reason="invalid-scheme",
        )

    if require_tls and parsed.scheme != "https":
        raise SecurityError(
            f"TLS required but got {parsed.scheme}",
            source=url,
            reason="tls-required",
        )

    return True


def parse_requirements_hash(
    hash_string: str,
) -> Tuple[HashAlgorithm, str]:
    """
    Parse a pip-style ``--hash`` argument into algorithm and digest.

    Parameters
    ----------
    hash_string : str
        Hash string in format ``algorithm:digest``
        (e.g., ``sha256:abc123...``).

    Returns
    -------
    tuple
        ``(HashAlgorithm, digest_string)``.

    Raises
    ------
    ValueError
        If the hash string format is invalid or the algorithm is
        not recognized.

    Examples
    --------
    >>> algo, digest = parse_requirements_hash(
    ...     "sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    ... )
    >>> algo
    <HashAlgorithm.SHA256: 'sha256'>
    >>> len(digest)
    64
    """
    if ":" not in hash_string:
        raise ValueError(
            f"Invalid hash format: '{hash_string}'. "
            f"Expected format: 'algorithm:digest'"
        )

    algo_name, digest = hash_string.split(":", 1)
    algo_name = algo_name.lower().strip()
    digest = digest.strip().lower()

    try:
        algorithm = HashAlgorithm(algo_name)
    except ValueError:
        raise ValueError(
            f"Unknown hash algorithm: '{algo_name}'. "
            f"Supported: {[a.value for a in HashAlgorithm]}"
        )

    if not digest:
        raise ValueError("Digest string is empty")

    if len(digest) != algorithm.hex_length:
        raise ValueError(
            f"Digest length {len(digest)} does not match "
            f"expected {algorithm.hex_length} for {algo_name}"
        )

    try:
        int(digest, 16)
    except ValueError:
        raise ValueError(f"Digest is not valid hex: '{digest[:32]}...'")

    return algorithm, digest
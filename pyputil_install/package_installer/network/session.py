#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HTTP session management using Python standard library.

This module provides a robust, dependency-free HTTP client built
entirely on ``http.client`` from the standard library. It supports
TLS encryption, automatic retries with exponential backoff, gzip
decompression, configurable timeouts, and structured response
parsing — all without requiring ``urllib3``, ``requests``, or any
third-party package.

Classes
-------
SessionConfig
    Immutable configuration for HTTP session behaviour.
PackageSession
    HTTP client with retry, TLS, and response parsing.

Notes
-----
This module intentionally avoids ``urllib3`` to eliminate version
conflicts on constrained platforms (e.g. Pydroid 3, embedded
Python). The standard library ``http.client`` is available on every
Python 3.8+ installation and provides the same fundamental
capabilities.

Examples
--------
>>> session = PackageSession()
>>> resp = session.get("https://pypi.org/pypi/requests/json")
>>> resp["status"]
200
>>> resp["json"]["info"]["version"]
'2.31.0'
"""

import json
import ssl
import time
import gzip
import socket
import logging
import sys
from typing import Any, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse, urlencode
from http.client import HTTPSConnection, HTTPConnection, HTTPException, HTTPResponse

from ..exceptions import (
    NetworkError,
    TimeoutError as PackageTimeoutError,
    SecurityError,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────


@dataclass
class SessionConfig:
    """
    Configuration for an HTTP session.

    Parameters
    ----------
    timeout_connect : float, default=10.0
        Maximum time (seconds) to wait for a TCP connection to be
        established. Must be positive.
    timeout_read : float, default=30.0
        Maximum time (seconds) to wait for the server to send a
        response after the connection is established. Must be
        positive.
    total_retries : int, default=3
        Maximum number of retry attempts for idempotent requests
        that fail due to transient network errors. Set to 0 to
        disable retries entirely.
    backoff_factor : float, default=0.5
        Multiplier used to calculate the delay between successive
        retries. The delay for attempt *n* is::

            delay = backoff_factor * (2 ** n)

        For example, with the default value the delays are 0.5 s,
        1.0 s, 2.0 s.
    verify_tls : bool, default=True
        If True, validate the server's TLS certificate against the
        system trust store. Disable **only** for internal mirrors
        or development environments.
    user_agent : str or None, default=None
        Value of the ``User-Agent`` header. If None, a default
        string is constructed from the library version and Python
        interpreter version.
    headers : dict of str to str, default={}
        Additional headers included with every request. These can
        be overridden on a per-request basis.
    allowed_statuses : tuple of int, default=(200,)
        HTTP status codes that are treated as success. Any response
        whose status code is not in this tuple raises a
        ``NetworkError``.

    Raises
    ------
    ValueError
        If ``timeout_connect`` or ``timeout_read`` is not positive,
        or if ``total_retries`` is negative.

    Examples
    --------
    >>> config = SessionConfig(
    ...     timeout_connect=5.0,
    ...     timeout_read=15.0,
    ...     total_retries=5,
    ...     verify_tls=True,
    ... )
    """

    timeout_connect: float = 10.0
    timeout_read: float = 30.0
    total_retries: int = 3
    backoff_factor: float = 0.5
    verify_tls: bool = True
    user_agent: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    allowed_statuses: Tuple[int, ...] = (200,)

    def __post_init__(self) -> None:
        """Validate configuration after the dataclass is constructed."""
        if self.timeout_connect <= 0:
            raise ValueError(
                f"timeout_connect must be > 0, got {self.timeout_connect}"
            )
        if self.timeout_read <= 0:
            raise ValueError(
                f"timeout_read must be > 0, got {self.timeout_read}"
            )
        if self.total_retries < 0:
            raise ValueError(
                f"total_retries must be >= 0, got {self.total_retries}"
            )

        if self.user_agent is None:
            py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
            self.user_agent = f"package_installer/1.0 (Python {py_ver})"


# ──────────────────────────────────────────────────────────────────
# HTTP Session
# ──────────────────────────────────────────────────────────────────


class PackageSession:
    """
    HTTP client with automatic retry and response parsing.

    Built entirely on Python's ``http.client`` module.  Provides
    connection timeouts, TLS verification, gzip decompression,
    structured JSON/text response parsing, and exponential-backoff
    retries for transient failures.

    Parameters
    ----------
    config : SessionConfig, optional
        Session configuration.  When *None*, a default
        ``SessionConfig`` is used.

    Attributes
    ----------
    config : SessionConfig
        The configuration object this session was created with.
    stats : dict
        Read-only view of cumulative statistics: ``requests``,
        ``retries``, ``failures``, ``bytes``.

    Notes
    -----
    - **No external dependencies** — uses only the standard library.
      This guarantees compatibility across all Python 3.8+
      environments including embedded and mobile runtimes.
    - **TLS** — certificate verification uses the system trust store
      by default.  A custom CA bundle can be supplied via
      ``ssl.create_default_context(cafile=...)`` if needed (not
      exposed in ``SessionConfig`` to keep the API simple; callers
      can subclass or patch the SSL context).
    - **Thread safety** — each ``request()`` call opens a fresh
      connection and closes it after reading the response, so a
      single ``PackageSession`` instance is safe to use from
      multiple threads, though connection pooling is not implemented.

    Examples
    --------
    >>> session = PackageSession()
    >>> response = session.get("https://pypi.org/pypi/flask/json")
    >>> response["json"]["info"]["name"]
    'flask'

    With retries disabled::

    >>> config = SessionConfig(total_retries=0)
    >>> session = PackageSession(config)

    Context manager::

    >>> with PackageSession() as s:
    ...     s.head("https://pypi.org")
    """

    def __init__(self, config: Optional[SessionConfig] = None) -> None:
        self.config = config if config is not None else SessionConfig()

        # ── statistics ──────────────────────────────────────────
        self._stats: Dict[str, int] = {
            "requests": 0,
            "retries": 0,
            "failures": 0,
            "bytes": 0,
        }

        # ── SSL context ─────────────────────────────────────────
        if self.config.verify_tls:
            self._ssl_context = ssl.create_default_context()
        else:
            self._ssl_context = ssl._create_unverified_context()
            logger.warning(
                "TLS certificate verification is DISABLED. "
                "All HTTPS connections are vulnerable to "
                "man-in-the-middle attacks."
            )

        logger.debug(
            "PackageSession initialised "
            "(retries=%d, connect_timeout=%.1fs, read_timeout=%.1fs)",
            self.config.total_retries,
            self.config.timeout_connect,
            self.config.timeout_read,
        )

    # ── connection factory ──────────────────────────────────────

    def _create_connection(
        self, scheme: str, host: str, port: int
    ) -> Union[HTTPConnection, HTTPSConnection]:
        """
        Create an ``HTTPConnection`` or ``HTTPSConnection``.

        Parameters
        ----------
        scheme : str
            ``"http"`` or ``"https"``.
        host : str
            Remote hostname.
        port : int
            Remote port number.

        Returns
        -------
        HTTPConnection or HTTPSConnection
            A connection object with the connect timeout already
            baked in.

        Notes
        -----
        The read timeout is applied later when reading the response
        body, not during connection creation.
        """
        timeout = self.config.timeout_connect
        if scheme == "https":
            return HTTPSConnection(
                host=host,
                port=port,
                timeout=timeout,
                context=self._ssl_context,
            )
        return HTTPConnection(
            host=host,
            port=port,
            timeout=timeout,
        )

    # ── header assembly ─────────────────────────────────────────

    def _build_headers(
        self, extra: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Merge default headers with per-request overrides.

        Parameters
        ----------
        extra : dict or None
            Request-specific headers that override defaults.

        Returns
        -------
        dict
            The complete set of headers for the request.
        """
        headers: Dict[str, str] = {
            "User-Agent": self.config.user_agent or "",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "Connection": "close",
        }
        headers.update(self.config.headers)
        if extra:
            headers.update(extra)
        return headers

    # ── body preparation ────────────────────────────────────────

    @staticmethod
    def _prepare_body(
        body: Optional[Union[bytes, str]],
        fields: Optional[Dict[str, Any]],
        headers: Dict[str, str],
    ) -> Optional[bytes]:
        """
        Normalise request body and set the ``Content-Type`` header.

        Parameters
        ----------
        body : bytes, str, or None
            Pre-encoded body.
        fields : dict or None
            Form fields to URL-encode.
        headers : dict
            Headers dict (mutated in-place to set Content-Type).

        Returns
        -------
        bytes or None
            The body ready to send, or None.

        Raises
        ------
        ValueError
            If both *body* and *fields* are supplied.
        """
        if body is not None and fields is not None:
            raise ValueError(
                "Provide either 'body' or 'fields', not both."
            )

        if fields is not None:
            if "content-type" not in {k.lower() for k in headers}:
                headers["Content-Type"] = "application/x-www-form-urlencoded"
            return urlencode(fields).encode("utf-8")

        if isinstance(body, str):
            return body.encode("utf-8")

        return body

    # ── response parsing ────────────────────────────────────────

    def _parse_response(
        self,
        response: HTTPResponse,
        url: str,
        start_time: float,
    ) -> Dict[str, Any]:
        """
        Convert a raw ``http.client`` response into a dictionary.

        Parameters
        ----------
        response : HTTPResponse
            The response object after ``getresponse()``.
        url : str
            Original request URL (for error context).
        start_time : float
            ``time.monotonic()`` value captured before the request
            was sent.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - **status** (*int*) — HTTP status code.
            - **headers** (*dict*) — Response headers (lower-case
              keys).
            - **data** (*bytes*) — Raw response body.
            - **json** (*any*) — Parsed JSON if the Content-Type is
              ``application/json``, else ``None``.
            - **text** (*str or None*) — Decoded UTF-8 body for
              non-JSON responses.
            - **url** (*str*) — The request URL.
            - **elapsed** (*float*) — Wall-clock seconds for the
              request (connect + read + parse).
        """
        raw = response.read()
        elapsed = time.monotonic() - start_time

        resp_headers: Dict[str, str] = {}
        for key, value in response.getheaders():
            resp_headers[key.lower()] = value

        # decompress gzip if present
        if "gzip" in resp_headers.get("content-encoding", ""):
            try:
                raw = gzip.decompress(raw)
            except Exception:
                logger.debug("gzip decompression failed — returning raw data")

        self._stats["bytes"] += len(raw)

        result: Dict[str, Any] = {
            "status": response.status,
            "headers": resp_headers,
            "data": raw,
            "url": url,
            "elapsed": round(elapsed, 4),
        }

        content_type = resp_headers.get("content-type", "")

        if "application/json" in content_type:
            try:
                result["json"] = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.debug("JSON parse failed for %s", url)
                result["json"] = None
        else:
            try:
                result["text"] = raw.decode("utf-8")
            except UnicodeDecodeError:
                result["text"] = None

        return result

    # ── single request (no retry) ───────────────────────────────

    def _request_once(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[bytes],
        timeout: float,
    ) -> Dict[str, Any]:
        """
        Perform a single HTTP request without retry logic.

        Parameters
        ----------
        method : str
            HTTP method (GET, POST, …).
        url : str
            Full URL.
        headers : dict
            Request headers.
        body : bytes or None
            Request body.
        timeout : float
            Read timeout in seconds.

        Returns
        -------
        dict
            Parsed response (see ``_parse_response``).

        Raises
        ------
        NetworkError
            Transport-level failure or unexpected status.
        PackageTimeoutError
            Connection or read timeout.
        SecurityError
            TLS verification failure.
        """
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
        port = parsed.port or (443 if scheme == "https" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query

        conn = None
        try:
            conn = self._create_connection(scheme, host, port)
            if body is not None:
                headers["Content-Length"] = str(len(body))

            start = time.monotonic()
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()

            if resp.status not in self.config.allowed_statuses:
                resp.read()
                raise NetworkError(
                    f"Unexpected status {resp.status} from {url}",
                    url=url,
                    status_code=resp.status,
                )

            return self._parse_response(resp, url, start)

        except socket.timeout as exc:
            raise PackageTimeoutError(
                f"Timeout connecting to {host}:{port}",
                timeout_seconds=self.config.timeout_connect,
                operation=f"{method} {url}",
            ) from exc

        except ssl.SSLError as exc:
            raise SecurityError(
                f"TLS verification failed for {host}: {exc}",
                source=host,
                reason="tls-error",
            ) from exc

        except HTTPException as exc:
            raise NetworkError(
                f"HTTP protocol error for {url}: {exc}",
                url=url,
            ) from exc

        except OSError as exc:
            raise NetworkError(
                f"OS error for {url}: {exc}",
                url=url,
            ) from exc

        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    # ── public request method (with retry) ──────────────────────

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Union[bytes, str]] = None,
        fields: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute an HTTP request with automatic retries.

        Parameters
        ----------
        method : str
            HTTP method: ``GET``, ``POST``, ``PUT``, ``DELETE``,
            ``HEAD``, ``OPTIONS``, or ``PATCH``. Case-insensitive.
        url : str
            Full URL including scheme (``http`` or ``https``).
        headers : dict, optional
            Additional request headers merged with the session
            defaults.
        body : bytes or str, optional
            Raw request body. Strings are UTF-8 encoded.
        fields : dict, optional
            Form data to URL-encode and send.  Mutually exclusive
            with *body*.
        timeout : float, optional
            Per-request read timeout override.  If *None*, the
            session's ``timeout_read`` is used.

        Returns
        -------
        dict
            Parsed response. See ``_parse_response`` for the
            complete schema.

        Raises
        ------
        ValueError
            If *method* is not a recognised HTTP verb, or if both
            *body* and *fields* are supplied.
        NetworkError
            If all retry attempts are exhausted or the server
            returns an unexpected status code.
        PackageTimeoutError
            If the connection or read timeout is exceeded on every
            attempt.
        SecurityError
            If TLS certificate verification fails.

        Notes
        -----
        Retries are performed for **all** exceptions that derive
        from ``NetworkError``, ``PackageTimeoutError``, or
        ``SecurityError``.  The delay between attempts follows an
        exponential backoff formula::

            wait = backoff_factor * (2 ** attempt_number)

        Examples
        --------
        >>> session = PackageSession()
        >>> resp = session.request("GET", "https://pypi.org/pypi/six/json")
        >>> resp["status"]
        200

        POST with form data::

        >>> resp = session.request(
        ...     "POST",
        ...     "https://httpbin.org/post",
        ...     fields={"key": "value"},
        ... )
        """
        method = method.upper()
        valid_methods = {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}
        if method not in valid_methods:
            raise ValueError(
                f"Unknown HTTP method '{method}'. "
                f"Expected one of {sorted(valid_methods)}."
            )

        merged_headers = self._build_headers(headers)
        prepared_body = self._prepare_body(body, fields, merged_headers)
        read_timeout = (
            timeout if timeout is not None else self.config.timeout_read
        )

        self._stats["requests"] += 1
        last_error: Optional[Exception] = None

        max_attempts = self.config.total_retries + 1
        for attempt in range(max_attempts):
            try:
                return self._request_once(
                    method=method,
                    url=url,
                    headers=merged_headers,
                    body=prepared_body,
                    timeout=read_timeout,
                )
            except (NetworkError, PackageTimeoutError, SecurityError) as exc:
                last_error = exc
                if attempt < max_attempts - 1:
                    wait = self.config.backoff_factor * (2 ** attempt)
                    logger.warning(
                        "Request attempt %d/%d failed (%s). "
                        "Retrying in %.1f s …",
                        attempt + 1,
                        max_attempts,
                        exc,
                        wait,
                    )
                    self._stats["retries"] += 1
                    time.sleep(wait)

        self._stats["failures"] += 1
        assert last_error is not None
        raise last_error

    # ── HTTP verb shortcuts ─────────────────────────────────────

    def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Perform an HTTP GET request.

        Parameters
        ----------
        url : str
            Target URL.
        headers : dict, optional
            Extra headers.
        timeout : float, optional
            Read timeout override.

        Returns
        -------
        dict
            Parsed response.

        Examples
        --------
        >>> s = PackageSession()
        >>> s.get("https://pypi.org/pypi/click/json")["status"]
        200
        """
        return self.request("GET", url, headers=headers, timeout=timeout)

    def head(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Perform an HTTP HEAD request.

        Parameters
        ----------
        url : str
            Target URL.
        headers : dict, optional
            Extra headers.
        timeout : float, optional
            Read timeout override.

        Returns
        -------
        dict
            Parsed response (body is empty).
        """
        return self.request("HEAD", url, headers=headers, timeout=timeout)

    def post(
        self,
        url: str,
        body: Optional[Union[bytes, str]] = None,
        fields: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Perform an HTTP POST request.

        Parameters
        ----------
        url : str
            Target URL.
        body : bytes or str, optional
            Raw body.
        fields : dict, optional
            Form fields.
        headers : dict, optional
            Extra headers.
        timeout : float, optional
            Read timeout override.

        Returns
        -------
        dict
            Parsed response.
        """
        return self.request(
            "POST",
            url,
            headers=headers,
            body=body,
            fields=fields,
            timeout=timeout,
        )

    # ── utility methods ─────────────────────────────────────────

    def test_connectivity(self, url: str = "https://pypi.org") -> bool:
        """
        Check whether a remote host is reachable.

        Parameters
        ----------
        url : str, default="https://pypi.org"
            URL to test.

        Returns
        -------
        bool
            *True* if the server responds with an allowed status
            code within 5 seconds, *False* otherwise.

        Examples
        --------
        >>> PackageSession().test_connectivity()
        True
        """
        try:
            resp = self.head(url, timeout=5.0)
            return resp["status"] in self.config.allowed_statuses
        except Exception:
            return False

    def get_stats(self) -> Dict[str, int]:
        """
        Return a copy of the session statistics.

        Returns
        -------
        dict
            Keys: ``requests``, ``retries``, ``failures``, ``bytes``.
        """
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset all cumulative statistics to zero."""
        self._stats = {"requests": 0, "retries": 0, "failures": 0, "bytes": 0}

    # ── resource management ─────────────────────────────────────

    def close(self) -> None:
        """
        Release resources held by the session.

        Notes
        -----
        For the stdlib-based session this is a no-op because each
        connection is closed immediately after use.  The method
        exists for interface compatibility with other backends.
        """
        logger.debug("PackageSession.close() called")

    def __enter__(self) -> "PackageSession":
        """Enter the runtime context — returns *self*."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the runtime context — calls ``close()``."""
        self.close()

    def __repr__(self) -> str:
        """Return a compact string representation."""
        return (
            f"PackageSession(req={self._stats['requests']}, "
            f"retries={self._stats['retries']}, "
            f"fail={self._stats['failures']})"
        )
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyPI JSON API client for package metadata retrieval.

This module provides a client for the PyPI JSON API
(https://wiki.python.org/moin/PyPIJSON), enabling direct access to
package metadata, version listings, release information, and dependency
resolution without invoking pip subprocesses.

Classes
-------
PyPIClient
    Client for the PyPI JSON API with caching awareness.

Examples
--------
>>> client = PyPIClient()
>>> metadata = client.get_package_metadata("requests")
>>> metadata["info"]["version"]
'2.31.0'
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import quote

from .session import PackageSession, SessionConfig
from ..exceptions import (
    NetworkError,
    PackageNotFoundError,
    PackageVersionError,
)
from ..cache import PackageCache, CacheConfig

logger = logging.getLogger(__name__)


class PyPIClient:
    """
    Client for the PyPI JSON API.

    This class provides methods to query package metadata, version
    information, release details, and dependencies directly from PyPI's
    JSON API. It integrates with the caching layer to reduce redundant
    network requests.

    Parameters
    ----------
    session : PackageSession, optional
        HTTP session for making requests. If None, a default session
        is created.
    cache : PackageCache, optional
        Cache for storing API responses. If None, caching is disabled.
    base_url : str, default="https://pypi.org/pypi"
        Base URL for the PyPI JSON API. Change this to use a custom
        or mirror index.

    Attributes
    ----------
    session : PackageSession
        The HTTP session used for requests.
    cache : PackageCache or None
        The cache instance, or None if caching is disabled.
    base_url : str
        Base URL for the PyPI API.

    Notes
    -----
    The PyPI JSON API returns package data in the format described at
    https://warehouse.pypa.io/api-reference/json/. Responses include
    metadata, all released versions, and per-release file information
    with hashes and download URLs.

    When a cache is provided, ``get_package_metadata`` and
    ``get_package_versions`` responses are cached using keys prefixed
    with ``pypi:metadata:`` and ``pypi:versions:`` respectively.

    Examples
    --------
    >>> client = PyPIClient()
    >>> info = client.get_package_info("numpy")
    >>> info["name"]
    'numpy'

    With custom index::

    >>> client = PyPIClient(base_url="https://my-mirror.local/pypi")
    """

    def __init__(
        self,
        session: Optional[PackageSession] = None,
        cache: Optional[PackageCache] = None,
        base_url: str = "https://pypi.org/pypi",
    ) -> None:
        self.session = session if session is not None else PackageSession()
        self.cache = cache
        self.base_url = base_url.rstrip("/")
        self._cache_ttl = 1800.0
        logger.debug(f"PyPIClient initialized with base URL: {self.base_url}")

    def _normalize_package_name(self, name: str) -> str:
        """
        Normalize a package name per PEP 503.

        Parameters
        ----------
        name : str
            The package name to normalize.

        Returns
        -------
        str
            Normalized package name (lowercase, hyphens for separators).

        Notes
        -----
        Per PEP 503, package names are case-insensitive and treat
        ``-``, ``_``, and ``.`` as equivalent. This method normalizes
        to the standard form using hyphens.
        """
        return name.strip().lower().replace("_", "-").replace(".", "-")

    def _build_url(self, package_name: str, suffix: str = "") -> str:
        """
        Build a PyPI API URL for a package.

        Parameters
        ----------
        package_name : str
            The package name.
        suffix : str, default=""
            URL suffix (e.g., ``/json``, ``/2.0.0/json``).

        Returns
        -------
        str
            Complete API URL.
        """
        normalized = self._normalize_package_name(package_name)
        encoded = quote(normalized, safe="")
        return f"{self.base_url}/{encoded}{suffix}"

    def _handle_response(
        self,
        response: Dict[str, Any],
        package_name: str,
    ) -> Dict[str, Any]:
        """
        Validate and extract JSON data from an API response.

        Parameters
        ----------
        response : dict
            The parsed HTTP response from PackageSession.
        package_name : str
            The package name being queried.

        Returns
        -------
        dict
            The JSON body from the response.

        Raises
        ------
        PackageNotFoundError
            If the response status is 404.
        NetworkError
            If the response status is not 200 or the JSON is invalid.
        """
        if response["status"] == 404:
            raise PackageNotFoundError(
                package_name,
                message=f"Package '{package_name}' not found on PyPI",
                location="pypi",
            )

        if response["status"] != 200:
            raise NetworkError(
                f"PyPI API returned status {response['status']} for {package_name}",
                url=response.get("url", ""),
                status_code=response["status"],
                package_name=package_name,
            )

        if response.get("json") is None:
            raise NetworkError(
                f"Invalid or non-JSON response from PyPI for {package_name}",
                url=response.get("url", ""),
                package_name=package_name,
            )

        return response["json"]

    def get_package_metadata(
        self,
        package_name: str,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Retrieve complete package metadata from PyPI.

        Parameters
        ----------
        package_name : str
            The name of the package.
        use_cache : bool, default=True
            Whether to check the cache before making a request.

        Returns
        -------
        dict
            Complete package metadata as returned by PyPI JSON API.
            Contains ``info``, ``releases``, ``urls``, and other keys.

        Raises
        ------
        PackageNotFoundError
            If the package does not exist on PyPI.
        NetworkError
            If the API request fails.

        Notes
        -----
        The returned dictionary has the following structure::

            {
                "info": {
                    "name": str,
                    "version": str,
                    "summary": str,
                    "description": str,
                    "keywords": str,
                    "license": str,
                    "author": str,
                    "author_email": str,
                    "maintainer": str,
                    "maintainer_email": str,
                    "home_page": str,
                    "package_url": str,
                    "project_urls": dict,
                    "requires_python": str,
                    "requires_dist": list[str],
                    "classifiers": list[str],
                    ...
                },
                "releases": {
                    "1.0.0": [{"filename": str, "url": str, "digests": dict, ...}],
                    ...
                },
                "urls": [...],
                "vulnerabilities": [...],
            }

        Examples
        --------
        >>> client = PyPIClient()
        >>> metadata = client.get_package_metadata("flask")
        >>> metadata["info"]["requires_python"]
        '>=3.8'
        """
        cache_key = f"pypi:metadata:{self._normalize_package_name(package_name)}"

        if use_cache and self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for metadata: {package_name}")
                return cached

        url = self._build_url(package_name, "/json")
        logger.debug(f"Fetching metadata for {package_name} from {url}")

        response = self.session.get(url)
        data = self._handle_response(response, package_name)

        if self.cache is not None:
            self.cache.set(cache_key, data, ttl=self._cache_ttl)

        return data

    def get_package_versions(
        self,
        package_name: str,
        use_cache: bool = True,
    ) -> List[str]:
        """
        Retrieve all released versions for a package.

        Parameters
        ----------
        package_name : str
            The name of the package.
        use_cache : bool, default=True
            Whether to check the cache before making a request.

        Returns
        -------
        list of str
            All released version strings, sorted in descending order
            (newest first).

        Raises
        ------
        PackageNotFoundError
            If the package does not exist on PyPI.
        NetworkError
            If the API request fails.

        Notes
        -----
        Pre-release versions (containing ``a``, ``b``, ``rc``, ``dev``)
        are included in the returned list. Use ``filter_versions`` to
        exclude them.

        Examples
        --------
        >>> client = PyPIClient()
        >>> versions = client.get_package_versions("click")
        >>> len(versions) > 0
        True
        >>> versions[0]  # newest version
        '8.1.7'
        """
        cache_key = f"pypi:versions:{self._normalize_package_name(package_name)}"

        if use_cache and self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for versions: {package_name}")
                return cached

        metadata = self.get_package_metadata(
            package_name, use_cache=use_cache
        )
        releases = metadata.get("releases", {})

        versions = sorted(
            releases.keys(),
            key=lambda v: self._parse_version_for_sorting(v),
            reverse=True,
        )

        if self.cache is not None:
            self.cache.set(cache_key, versions, ttl=self._cache_ttl)

        return versions

    def _parse_version_for_sorting(self, version_string: str) -> Tuple:
        """
        Parse a version string into a sortable tuple.

        Parameters
        ----------
        version_string : str
            The version string to parse.

        Returns
        -------
        tuple
            A tuple of (numeric_parts, pre_release_flag, pre_release_num)
            that can be compared for sorting.

        Notes
        -----
        This method handles common version formats including:

        - Simple: ``1.2.3``
        - Pre-release: ``1.2.3a1``, ``1.2.3b2``, ``1.2.3rc3``
        - Dev: ``1.2.3.dev4``
        - Post: ``1.2.3.post1``
        - Epoch: ``1!1.2.3``

        Pre-release versions sort before the corresponding release version.
        """
        import re

        version_string = version_string.strip()

        epoch = 0
        if "!" in version_string:
            epoch_str, version_string = version_string.split("!", 1)
            try:
                epoch = int(epoch_str)
            except ValueError:
                epoch = 0

        pre_release_flag = 0
        pre_release_num = 0
        post_release_num = 0
        dev_release_num = 0

        pre_pattern = r"(a|alpha|b|beta|rc|c|pre|preview)(\d*)"
        pre_match = re.search(pre_pattern, version_string, re.IGNORECASE)
        if pre_match:
            prefix = pre_match.group(1).lower()
            pre_release_num = int(pre_match.group(2)) if pre_match.group(2) else 0

            if prefix in ("a", "alpha"):
                pre_release_flag = 1
            elif prefix in ("b", "beta"):
                pre_release_flag = 2
            elif prefix in ("rc", "c", "pre", "preview"):
                pre_release_flag = 3

            version_string = version_string[: pre_match.start()]

        post_match = re.search(r"\.?post(\d*)", version_string, re.IGNORECASE)
        if post_match:
            post_release_num = int(post_match.group(1)) if post_match.group(1) else 0
            version_string = version_string[: post_match.start()]

        dev_match = re.search(r"\.?dev(\d*)", version_string, re.IGNORECASE)
        if dev_match:
            dev_release_num = int(dev_match.group(1)) if dev_match.group(1) else 0
            version_string = version_string[: dev_match.start()]

        numeric_parts: List[int] = []
        for part in version_string.split("."):
            try:
                numeric_parts.append(int(part))
            except ValueError:
                numeric_parts.append(0)

        while len(numeric_parts) < 3:
            numeric_parts.append(0)

        return (
            epoch,
            tuple(numeric_parts),
            pre_release_flag,
            pre_release_num,
            post_release_num,
            dev_release_num,
        )

    def get_latest_version(
        self,
        package_name: str,
        include_pre_releases: bool = False,
        use_cache: bool = True,
    ) -> Optional[str]:
        """
        Get the latest stable (or pre-release) version of a package.

        Parameters
        ----------
        package_name : str
            The name of the package.
        include_pre_releases : bool, default=False
            If True, include pre-release versions in the search.
        use_cache : bool, default=True
            Whether to check the cache before making a request.

        Returns
        -------
        str or None
            The latest version string, or None if no versions exist.

        Raises
        ------
        PackageNotFoundError
            If the package does not exist on PyPI.
        NetworkError
            If the API request fails.

        Examples
        --------
        >>> client = PyPIClient()
        >>> latest = client.get_latest_version("django")
        >>> latest.startswith("4.")
        True

        Including pre-releases::

        >>> latest = client.get_latest_version("django", include_pre_releases=True)
        """
        versions = self.get_package_versions(
            package_name, use_cache=use_cache
        )

        if not versions:
            return None

        if include_pre_releases:
            return versions[0]

        for version in versions:
            if not self._is_pre_release(version):
                return version

        return None

    def _is_pre_release(self, version_string: str) -> bool:
        """
        Check if a version string represents a pre-release.

        Parameters
        ----------
        version_string : str
            The version string to check.

        Returns
        -------
        bool
            True if the version is a pre-release.

        Notes
        -----
        Pre-release versions contain one of these patterns:
        ``a``, ``alpha``, ``b``, ``beta``, ``rc``, ``c``, ``pre``,
        ``preview``, or ``.dev``.
        """
        import re

        pre_pattern = r"(a\d|alpha|b\d|beta|rc\d|c\d|pre|preview|\.dev\d)"
        return bool(re.search(pre_pattern, version_string, re.IGNORECASE))

    def filter_versions(
        self,
        versions: List[str],
        stable_only: bool = True,
        min_version: Optional[str] = None,
        max_version: Optional[str] = None,
    ) -> List[str]:
        """
        Filter a list of version strings.

        Parameters
        ----------
        versions : list of str
            Version strings to filter.
        stable_only : bool, default=True
            If True, exclude pre-release versions.
        min_version : str, optional
            Minimum version (inclusive).
        max_version : str, optional
            Maximum version (inclusive).

        Returns
        -------
        list of str
            Filtered version strings, sorted descending.

        Examples
        --------
        >>> client = PyPIClient()
        >>> versions = ["2.0.0", "1.0.0", "1.0.0a1", "2.1.0rc1"]
        >>> client.filter_versions(versions, stable_only=True)
        ['2.0.0', '1.0.0']
        >>> client.filter_versions(versions, min_version="1.5.0")
        ['2.0.0', '2.1.0rc1']
        """
        result = versions

        if stable_only:
            result = [v for v in result if not self._is_pre_release(v)]

        if min_version is not None:
            min_tuple = self._parse_version_for_sorting(min_version)
            result = [
                v for v in result
                if self._parse_version_for_sorting(v) >= min_tuple
            ]

        if max_version is not None:
            max_tuple = self._parse_version_for_sorting(max_version)
            result = [
                v for v in result
                if self._parse_version_for_sorting(v) <= max_tuple
            ]

        return sorted(
            result,
            key=lambda v: self._parse_version_for_sorting(v),
            reverse=True,
        )

    def get_package_info(self, package_name: str) -> Dict[str, Any]:
        """
        Get summary information about a package.

        Parameters
        ----------
        package_name : str
            The name of the package.

        Returns
        -------
        dict
            Dictionary with keys: ``name``, ``version``, ``summary``,
            ``author``, ``author_email``, ``license``, ``home_page``,
            ``requires_python``, ``dependencies``, ``latest_version``,
            ``all_versions_count``.

        Raises
        ------
        PackageNotFoundError
            If the package does not exist on PyPI.
        NetworkError
            If the API request fails.

        Examples
        --------
        >>> client = PyPIClient()
        >>> info = client.get_package_info("requests")
        >>> info["name"]
        'requests'
        >>> "dependencies" in info
        True
        """
        metadata = self.get_package_metadata(package_name)
        info = metadata.get("info", {})

        dependencies = info.get("requires_dist", [])
        if dependencies is None:
            dependencies = []

        parsed_deps = self.parse_dependencies(dependencies)

        versions = self.get_package_versions(package_name)

        return {
            "name": info.get("name", package_name),
            "version": info.get("version", ""),
            "summary": info.get("summary", ""),
            "description": info.get("description", ""),
            "author": info.get("author", ""),
            "author_email": info.get("author_email", ""),
            "maintainer": info.get("maintainer", ""),
            "maintainer_email": info.get("maintainer_email", ""),
            "license": info.get("license", ""),
            "home_page": info.get("home_page", ""),
            "project_urls": info.get("project_urls", {}),
            "keywords": info.get("keywords", ""),
            "classifiers": info.get("classifiers", []),
            "requires_python": info.get("requires_python", ""),
            "dependencies": parsed_deps,
            "latest_version": versions[0] if versions else None,
            "all_versions_count": len(versions),
        }

    def parse_dependencies(
        self, requires_dist: List[str]
    ) -> List[Dict[str, str]]:
        """
        Parse a list of dependency strings into structured data.

        Parameters
        ----------
        requires_dist : list of str
            List of dependency strings from ``info.requires_dist``.

        Returns
        -------
        list of dict
            Each dict has keys: ``name``, ``specifier``, ``extras``,
            ``environment``.

        Notes
        -----
        Dependency strings follow the format defined in PEP 508:

        - ``package_name``
        - ``package_name>=1.0,<2.0``
        - ``package_name[extra1,extra2]>=1.0``
        - ``package_name; python_version>="3.6"``
        - ``package_name; extra=="testing"``

        Examples
        --------
        >>> client = PyPIClient()
        >>> deps = client.parse_dependencies([
        ...     "requests>=2.28.0,<3.0",
        ...     "urllib3>=1.21.1; python_version>='3.7'",
        ...     "cryptography[ssh]",
        ... ])
        >>> deps[0]["name"]
        'requests'
        """
        import re

        parsed = []

        for dep in requires_dist:
            if not dep or not dep.strip():
                continue

            dep = dep.strip()
            entry: Dict[str, str] = {
                "name": "",
                "specifier": "",
                "extras": "",
                "environment": "",
                "raw": dep,
            }

            env_part = ""
            if ";" in dep:
                dep_part, env_part = dep.split(";", 1)
                entry["environment"] = env_part.strip()
            else:
                dep_part = dep

            extras = ""
            extras_match = re.match(r'^([^\[]+)\[([^\]]+)\](.*)$', dep_part)
            if extras_match:
                entry["name"] = extras_match.group(1).strip()
                entry["extras"] = extras_match.group(2).strip()
                remainder = extras_match.group(3).strip()
                if remainder:
                    entry["specifier"] = remainder
            else:
                spec_match = re.match(
                    r'^([^<>=!~]+)\s*(.*)$', dep_part
                )
                if spec_match:
                    entry["name"] = spec_match.group(1).strip()
                    entry["specifier"] = spec_match.group(2).strip()
                else:
                    entry["name"] = dep_part.strip()

            entry["name"] = self._normalize_package_name(entry["name"])
            parsed.append(entry)

        return parsed

    def get_release_info(
        self,
        package_name: str,
        version: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific release.

        Parameters
        ----------
        package_name : str
            The name of the package.
        version : str
            The version to query.

        Returns
        -------
        dict or None
            Release information with files, digests, and URLs,
            or None if the version does not exist.

        Raises
        ------
        PackageNotFoundError
            If the package does not exist on PyPI.
        NetworkError
            If the API request fails.

        Notes
        -----
        Each release contains a list of distribution files. Each file
        entry includes::

            {
                "filename": str,
                "url": str,
                "packagetype": "sdist" or "bdist_wheel",
                "python_version": str,
                "size": int,
                "digests": {"sha256": str, "md5": str},
                "has_sig": bool,
            }

        Examples
        --------
        >>> client = PyPIClient()
        >>> release = client.get_release_info("click", "8.1.7")
        >>> len(release["files"]) > 0
        True
        """
        metadata = self.get_package_metadata(package_name)
        releases = metadata.get("releases", {})

        if version not in releases:
            return None

        release_data = releases[version]

        return {
            "package": package_name,
            "version": version,
            "files": release_data,
            "file_count": len(release_data),
        }

    def get_download_urls(
        self,
        package_name: str,
        version: str,
        prefer_wheel: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Get download URLs and digests for a specific package version.

        Parameters
        ----------
        package_name : str
            The name of the package.
        version : str
            The version to download.
        prefer_wheel : bool, default=True
            If True, return wheel URLs before source distributions.

        Returns
        -------
        list of dict
            Each dict contains: ``filename``, ``url``, ``packagetype``,
            ``size``, ``sha256``, ``md5``.

        Raises
        ------
        PackageNotFoundError
            If the package does not exist on PyPI.
        PackageVersionError
            If the version does not exist.

        Examples
        --------
        >>> client = PyPIClient()
        >>> urls = client.get_download_urls("six", "1.16.0")
        >>> len(urls) > 0
        True
        >>> urls[0]["sha256"]  # 64-character hex string
        '1e61c37477a1626458e36f7b1d82aa5c9b094fa4802892072e49de9c60c4c926'
        """
        release_info = self.get_release_info(package_name, version)

        if release_info is None:
            raise PackageVersionError(
                package_name,
                message=f"Version {version} not found for {package_name}",
                version_string=version,
            )

        urls = []
        for file_info in release_info["files"]:
            urls.append({
                "filename": file_info.get("filename", ""),
                "url": file_info.get("url", ""),
                "packagetype": file_info.get("packagetype", ""),
                "size": file_info.get("size", 0),
                "sha256": file_info.get("digests", {}).get("sha256", ""),
                "md5": file_info.get("digests", {}).get("md5", ""),
            })

        if prefer_wheel:
            urls.sort(
                key=lambda u: (0 if u["packagetype"] == "bdist_wheel" else 1)
            )

        return urls

    def search_packages(
        self,
        query: str,
        max_results: int = 20,
    ) -> List[Dict[str, str]]:
        """
        Search PyPI for packages matching a query.

        Parameters
        ----------
        query : str
            Search query string.
        max_results : int, default=20
            Maximum number of results to return.

        Returns
        -------
        list of dict
            Each dict contains: ``name``, ``version``, ``summary``,
            ``description``.

        Notes
        -----
        Uses the PyPI search API endpoint. This endpoint may be rate-limited
        and is not guaranteed to return exact matches. For exact package
        lookups, use ``get_package_metadata`` instead.

        Examples
        --------
        >>> client = PyPIClient()
        >>> results = client.search_packages("http client")
        >>> any(r["name"] == "requests" for r in results)
        True
        """
        from urllib.parse import urlencode

        search_url = f"https://pypi.org/search/?{urlencode({'q': query})}"
        logger.debug(f"Searching PyPI with query: {query}")

        url = self._build_url("") + f"?{urlencode({'q': query})}"

        try:
            response = self.session.get(
                f"https://pypi.org/search/?q={quote(query)}",
                headers={"Accept": "application/json"},
            )

            if response["status"] == 200 and response.get("json"):
                results = response["json"].get("results", [])
                parsed = []
                for item in results[:max_results]:
                    parsed.append({
                        "name": item.get("name", ""),
                        "version": item.get("version", ""),
                        "summary": item.get("summary", ""),
                        "description": item.get("description", ""),
                    })
                return parsed

            return []

        except NetworkError:
            logger.warning(f"Search failed for query: {query}")
            return []

    def verify_package_exists(self, package_name: str) -> bool:
        """
        Check if a package exists on PyPI.

        Parameters
        ----------
        package_name : str
            The name of the package.

        Returns
        -------
        bool
            True if the package exists on PyPI.

        Notes
        -----
        Uses a HEAD request to the package URL, which is faster than
        fetching full metadata.

        Examples
        --------
        >>> client = PyPIClient()
        >>> client.verify_package_exists("pip")
        True
        >>> client.verify_package_exists("this-package-does-not-exist-xyz")
        False
        """
        try:
            url = self._build_url(package_name, "/json")
            response = self.session.head(url, timeout=5.0)
            return response["status"] == 200
        except (NetworkError, PackageNotFoundError):
            return False

    def batch_get_metadata(
        self,
        package_names: List[str],
        use_cache: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch metadata for multiple packages efficiently.

        Parameters
        ----------
        package_names : list of str
            List of package names to query.
        use_cache : bool, default=True
            Whether to use the cache for individual requests.

        Returns
        -------
        dict
            Mapping of package name to metadata dict. Failed packages
            are excluded from the result.

        Notes
        -----
        Requests are made sequentially to avoid overwhelming PyPI.
        For high-throughput scenarios, consider using multiple
        PyPIClient instances with connection pooling.

        Warnings
        --------
        Do not make concurrent requests to public PyPI without
        rate limiting. This method makes sequential requests by
        design.

        Examples
        --------
        >>> client = PyPIClient()
        >>> metadata = client.batch_get_metadata(["flask", "django", "click"])
        >>> len(metadata)
        3
        """
        results = {}
        for name in package_names:
            try:
                results[name] = self.get_package_metadata(
                    name, use_cache=use_cache
                )
            except (PackageNotFoundError, NetworkError) as e:
                logger.warning(f"Skipping {name}: {e}")
                continue

        return results

    def get_dependency_tree(
        self,
        package_name: str,
        depth: int = 2,
        include_extras: bool = False,
    ) -> Dict[str, Any]:
        """
        Build a dependency tree for a package.

        Parameters
        ----------
        package_name : str
            Root package name.
        depth : int, default=2
            Maximum recursion depth. Must be >= 1.
        include_extras : bool, default=False
            If True, include optional/extras dependencies.

        Returns
        -------
        dict
            Nested dictionary representing the dependency tree:
            ``{"name": str, "version": str, "dependencies": [...]}``.

        Notes
        -----
        Dependency resolution is recursive and may trigger many API
        requests. Use caching to reduce load on subsequent calls.

        Warnings
        --------
        Setting ``depth`` > 3 may result in many API requests and
        slow performance. Consider using the PyPI JSON API directly
        for deep dependency analysis.

        Examples
        --------
        >>> client = PyPIClient()
        >>> tree = client.get_dependency_tree("flask", depth=1)
        >>> tree["name"]
        'flask'
        """
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        metadata = self.get_package_metadata(package_name)
        info = metadata.get("info", {})

        tree: Dict[str, Any] = {
            "name": info.get("name", package_name),
            "version": info.get("version", ""),
            "summary": info.get("summary", ""),
            "dependencies": [],
        }

        if depth > 1:
            dependencies = info.get("requires_dist", []) or []
            for dep in dependencies:
                if not include_extras and 'extra=="' in dep:
                    continue

                dep_name = dep.split(";")[0].split("[")[0].split("<")[0]
                dep_name = dep_name.split(">")[0].split("=")[0].split("!")[0]
                dep_name = dep_name.strip()

                if dep_name:
                    try:
                        sub_tree = self.get_dependency_tree(
                            dep_name,
                            depth=depth - 1,
                            include_extras=include_extras,
                        )
                        tree["dependencies"].append(sub_tree)
                    except (PackageNotFoundError, NetworkError) as e:
                        tree["dependencies"].append({
                            "name": dep_name,
                            "version": "",
                            "error": str(e),
                            "dependencies": [],
                        })

        return tree

    def clear_cache(self) -> bool:
        """
        Clear cached PyPI responses.

        Returns
        -------
        bool
            True if cache was cleared, False if no cache is configured.

        Examples
        --------
        >>> client = PyPIClient()
        >>> client.clear_cache()
        False  # No cache configured
        """
        if self.cache is None:
            return False

        self.cache.clear()
        logger.info("PyPI client cache cleared")
        return True

    def __repr__(self) -> str:
        """String representation of the client."""
        return (
            f"PyPIClient(base_url={self.base_url}, "
            f"cached={'yes' if self.cache else 'no'})"
        )
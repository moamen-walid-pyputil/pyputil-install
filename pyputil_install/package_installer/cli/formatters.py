#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Output formatters for CLI results.

This module provides formatting utilities for displaying package
information, installation results, and error messages in various
output formats (table, JSON, plain text).

Classes
-------
OutputFormatter
    Base class for output formatters.
TableFormatter
    Formats output as aligned tables for terminal display.
JsonFormatter
    Formats output as JSON for programmatic consumption.
SimpleFormatter
    Formats output as plain text with minimal structure.
"""

import json
import sys
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


class OutputFormatter:
    """
    Base class for output formatters.

    Parameters
    ----------
    stream : file-like object, optional
        Output stream. Defaults to ``sys.stdout``.
    color : bool, default=True
        Whether to use ANSI color codes in output.

    Notes
    -----
    Subclasses must implement ``format_result``, ``format_error``,
    and ``format_info`` methods.
    """

    def __init__(self, stream=None, color: bool = True) -> None:
        self.stream = stream or sys.stdout
        self.color = color and stream is None

    def format_result(self, data: Dict[str, Any]) -> str:
        """
        Format a successful operation result.

        Parameters
        ----------
        data : dict
            Result data to format.

        Returns
        -------
        str
            Formatted string.
        """
        raise NotImplementedError

    def format_error(self, message: str, details: Optional[Dict[str, Any]] = None) -> str:
        """
        Format an error message.

        Parameters
        ----------
        message : str
            Error message.
        details : dict, optional
            Additional error details.

        Returns
        -------
        str
            Formatted error string.
        """
        raise NotImplementedError

    def format_info(self, data: Dict[str, Any]) -> str:
        """
        Format informational output.

        Parameters
        ----------
        data : dict
            Information to format.

        Returns
        -------
        str
            Formatted string.
        """
        raise NotImplementedError

    def write(self, text: str) -> None:
        """
        Write formatted text to the output stream.

        Parameters
        ----------
        text : str
            Text to write.
        """
        self.stream.write(text + "\n")
        self.stream.flush()


class TableFormatter(OutputFormatter):
    """
    Formats output as aligned tables for terminal display.

    Parameters
    ----------
    stream : file-like object, optional
        Output stream.
    color : bool, default=True
        Whether to use ANSI colors.
    max_width : int, default=120
        Maximum width for table rows before wrapping.

    Examples
    --------
    >>> fmt = TableFormatter()
    >>> result = {"package": "requests", "version": "2.31.0", "status": "installed"}
    >>> print(fmt.format_result(result))
    """

    def __init__(
        self,
        stream=None,
        color: bool = True,
        max_width: int = 120,
    ) -> None:
        super().__init__(stream=stream, color=color)
        self.max_width = max_width

    def _colorize(self, text: str, code: str) -> str:
        """
        Apply ANSI color to text if color is enabled.

        Parameters
        ----------
        text : str
            Text to colorize.
        code : str
            ANSI color code (e.g., '32' for green, '31' for red).

        Returns
        -------
        str
            Colorized text or original if color is disabled.
        """
        if not self.color:
            return text

        colors = {
            "green": "32",
            "red": "31",
            "yellow": "33",
            "blue": "34",
            "cyan": "36",
            "bold": "1",
            "reset": "0",
        }
        color_code = colors.get(code, "0")
        return f"\033[{color_code}m{text}\033[0m"

    def _format_table(
        self,
        headers: List[str],
        rows: List[List[str]],
    ) -> str:
        """
        Format data as an aligned table.

        Parameters
        ----------
        headers : list of str
            Column headers.
        rows : list of list of str
            Row data.

        Returns
        -------
        str
            Formatted table string.
        """
        if not rows:
            return ""

        col_widths = [
            max(len(str(row[i])) for row in rows + [headers])
            for i in range(len(headers))
        ]

        total_width = sum(col_widths) + (len(headers) * 3) + 1
        if total_width > self.max_width:
            for i in range(len(col_widths)):
                col_widths[i] = min(col_widths[i], self.max_width // len(headers))

        separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

        lines = [separator]

        header_line = "|"
        for i, header in enumerate(headers):
            header_line += f" {self._colorize(header.upper(), 'bold'):<{col_widths[i]}} |"
        lines.append(header_line)
        lines.append(separator)

        for row in rows:
            row_line = "|"
            for i, cell in enumerate(row):
                cell_str = str(cell)
                if len(cell_str) > col_widths[i]:
                    cell_str = cell_str[:col_widths[i] - 3] + "..."
                row_line += f" {cell_str:<{col_widths[i]}} |"
            lines.append(row_line)

        lines.append(separator)
        return "\n".join(lines)

    def format_result(self, data: Dict[str, Any]) -> str:
        """
        Format an installation or operation result.

        Parameters
        ----------
        data : dict
            Result data with keys like ``package``, ``version``,
            ``status``, ``duration``.

        Returns
        -------
        str
            Formatted table showing the result.
        """
        success = data.get("success", False)
        status_text = self._colorize(
            "✓ SUCCESS" if success else "✗ FAILED",
            "green" if success else "red",
        )

        lines = [
            f"\n{self._colorize('Operation Result', 'bold')}",
            f"{'─' * 50}",
            f"  Package:  {data.get('package_name', data.get('package', 'N/A'))}",
            f"  Action:   {data.get('action', 'N/A')}",
            f"  Status:   {status_text}",
        ]

        if data.get("version_installed"):
            lines.append(f"  Version:  {data['version_installed']}")

        if data.get("version_previous"):
            lines.append(f"  Previous: {data['version_previous']}")

        if data.get("duration_seconds"):
            lines.append(f"  Duration: {data['duration_seconds']:.2f}s")

        if data.get("rolled_back"):
            lines.append(
                f"  {self._colorize('⚠ Rolled back', 'yellow')}"
            )

        if data.get("cache_used"):
            lines.append(f"  Cache:    used")

        warnings = data.get("warnings", [])
        if warnings:
            lines.append(f"\n{self._colorize('Warnings:', 'yellow')}")
            for w in warnings:
                lines.append(f"  ⚠ {w}")

        errors = data.get("errors", [])
        if errors:
            lines.append(f"\n{self._colorize('Errors:', 'red')}")
            for e in errors:
                lines.append(f"  ✗ {e}")

        return "\n".join(lines)

    def format_error(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format an error message.

        Parameters
        ----------
        message : str
            Error message.
        details : dict, optional
            Additional error details.

        Returns
        -------
        str
            Formatted error string.
        """
        lines = [
            f"\n{self._colorize('✗ Error', 'red')}",
            f"{'─' * 50}",
            f"  {message}",
        ]

        if details:
            for key, value in details.items():
                if value is not None:
                    lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def format_info(self, data: Dict[str, Any]) -> str:
        """
        Format package information.

        Parameters
        ----------
        data : dict
            Package metadata dictionary.

        Returns
        -------
        str
            Formatted information string.
        """
        package_name = data.get("name", "N/A") 
        lines = [
            f"\n{self._colorize(f'Package: {package_name})', 'bold')}",
            f"{'─' * 50}",
        ]

        summary = data.get("summary", "")
        if summary:
            lines.append(f"  Summary:     {summary}")

        version = data.get("version") or data.get("latest_version") or "N/A"
        lines.append(f"  Version:     {version}")

        author = data.get("author", "")
        if author:
            email = data.get("author_email", "")
            lines.append(f"  Author:      {author} {f'<{email}>' if email else ''}")

        license_info = data.get("license", "")
        if license_info:
            lines.append(f"  License:     {license_info}")

        home = data.get("home_page", "")
        if home:
            lines.append(f"  Homepage:    {home}")

        requires_python = data.get("requires_python", "")
        if requires_python:
            lines.append(f"  Python:      {requires_python}")

        deps = data.get("dependencies", [])
        if deps:
            lines.append(f"\n{self._colorize('Dependencies:', 'cyan')}")
            for dep in deps:
                if isinstance(dep, dict):
                    dep_str = dep.get("name", "")
                    if dep.get("specifier"):
                        dep_str += f" {dep['specifier']}"
                    if dep.get("environment"):
                        dep_str += f"  [{dep['environment']}]"
                    lines.append(f"  • {dep_str}")
                else:
                    lines.append(f"  • {dep}")

        versions_count = data.get("all_versions_count", 0)
        if versions_count:
            lines.append(f"\n  Total versions: {versions_count}")

        return "\n".join(lines)

    def format_list(self, items: List[str], title: str = "") -> str:
        """
        Format a list of items.

        Parameters
        ----------
        items : list of str
            Items to format.
        title : str, optional
            Title for the list.

        Returns
        -------
        str
            Formatted list string.
        """
        if not items:
            return f"\n{self._colorize('(empty)', 'yellow')}"

        lines = []
        if title:
            lines.append(f"\n{self._colorize(title, 'bold')}")
            lines.append(f"{'─' * 50}")

        for item in items:
            lines.append(f"  • {item}")

        lines.append(f"\n  Total: {len(items)}")
        return "\n".join(lines)


class JsonFormatter(OutputFormatter):
    """
    Formats output as JSON for programmatic consumption.

    Parameters
    ----------
    stream : file-like object, optional
        Output stream.
    indent : int, default=2
        JSON indentation level.
    sort_keys : bool, default=False
        Whether to sort JSON keys alphabetically.

    Examples
    --------
    >>> fmt = JsonFormatter()
    >>> result = {"package": "requests", "success": True}
    >>> print(fmt.format_result(result))
    {
      "package": "requests",
      "success": true
    }
    """

    def __init__(
        self,
        stream=None,
        indent: int = 2,
        sort_keys: bool = False,
    ) -> None:
        super().__init__(stream=stream, color=False)
        self.indent = indent
        self.sort_keys = sort_keys

    def _to_json(self, data: Dict[str, Any]) -> str:
        """
        Convert data to formatted JSON string.

        Parameters
        ----------
        data : dict
            Data to serialize.

        Returns
        -------
        str
            JSON string.
        """
        return json.dumps(
            data,
            indent=self.indent,
            sort_keys=self.sort_keys,
            ensure_ascii=False,
            default=str,
        )

    def format_result(self, data: Dict[str, Any]) -> str:
        """Format result as JSON."""
        return self._to_json(data)

    def format_error(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format error as JSON."""
        error_data = {"error": message}
        if details:
            error_data["details"] = details
        return self._to_json(error_data)

    def format_info(self, data: Dict[str, Any]) -> str:
        """Format info as JSON."""
        return self._to_json(data)

    def format_list(self, items: List[str], title: str = "") -> str:
        """Format list as JSON array."""
        return self._to_json({"items": items, "count": len(items)})


class SimpleFormatter(OutputFormatter):
    """
    Formats output as plain text with minimal structure.

    Examples
    --------
    >>> fmt = SimpleFormatter()
    >>> result = {"package": "requests", "success": True}
    >>> print(fmt.format_result(result))
    Package: requests
    Status: SUCCESS
    """

    def format_result(self, data: Dict[str, Any]) -> str:
        """Format result as plain text."""
        lines = []
        success = data.get("success", False)
        lines.append(f"Package: {data.get('package_name', data.get('package', 'N/A'))}")
        lines.append(f"Action:  {data.get('action', 'N/A')}")
        lines.append(f"Status:  {'SUCCESS' if success else 'FAILED'}")

        if data.get("version_installed"):
            lines.append(f"Version: {data['version_installed']}")

        if data.get("duration_seconds"):
            lines.append(f"Time:    {data['duration_seconds']:.2f}s")

        return "\n".join(lines)

    def format_error(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format error as plain text."""
        lines = [f"ERROR: {message}"]
        if details:
            for key, value in details.items():
                if value is not None:
                    lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def format_info(self, data: Dict[str, Any]) -> str:
        """Format info as plain text."""
        lines = []
        for key, value in data.items():
            if value is not None and value != "":
                if isinstance(value, list):
                    lines.append(f"{key}:")
                    for item in value:
                        if isinstance(item, dict):
                            lines.append(f"  - {item.get('name', item)}")
                        else:
                            lines.append(f"  - {item}")
                else:
                    lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def format_list(self, items: List[str], title: str = "") -> str:
        """Format list as plain text."""
        if not items:
            return "(empty)"
        lines = []
        if title:
            lines.append(f"{title}:")
        for item in items:
            lines.append(f"  {item}")
        return "\n".join(lines)
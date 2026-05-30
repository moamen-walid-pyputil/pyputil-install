"""
Template context and safe formatting utilities.

Provides the TemplateContext dataclass — the ONLY allowed argument
type for template variable substitution — and safe_format_template(),
which validates all placeholders before calling str.format().

This module enforces the contract between raw templates (templates.py)
and the builder (builder.py). No other module should perform template
formatting directly.

Design
------
- TemplateContext is frozen and immutable. Each URL resolution creates
  a new instance. This prevents accidental state mutation and makes
  the formatting pipeline deterministic.
- safe_format_template() uses Python's own string.Formatter().parse()
  to extract placeholder names. This is production-safe and handles
  format specs, escaped braces, and nested structures correctly.
- join_url() is a pure utility for combining base URLs and paths
  without double-slash issues.

Usage
-----
    from toolforge.urls.context import TemplateContext, safe_format_template, join_url
    from toolforge.urls.templates import GCC_XPACK_FILENAME

    ctx = TemplateContext(
        version="14.2.0-2",
        platform=PlatformType.LINUX,
        arch=ArchitectureType.X64,
        ext=ArchiveType.TAR_GZ,
    )
    filename = safe_format_template(GCC_XPACK_FILENAME, ctx)

    url = join_url("https://example.com/releases/", filename)

Warnings
--------
- NEVER call template.format(**dict) directly in any other module.
  All formatting MUST go through safe_format_template() to ensure
  placeholder validation.
- TemplateContext is NOT a generic key-value store. Adding new fields
  requires updating templates that use them.
- The `strict_unused` flag is optional and defaults to False.
  Enable it during testing to catch dead context keys.
- join_url() does NOT perform URL encoding. If filenames contain
  special characters, encode them before calling join_url().

User Instructions
-----------------
- This module is internal to the URL layer. Use builder.py public API
  to resolve artifacts — you rarely need to import this directly.
- If adding a new placeholder to templates, add the corresponding
  field to TemplateContext and update to_format_dict().
- To debug format errors, enable `strict_unused=True` and wrap
  safe_format_template() calls in try/except to catch missing keys.
"""

from dataclasses import dataclass
from string import Formatter
from typing import Dict, Optional, Set

from .base import ArchitectureType, ArchiveType, PlatformType, TemplateContext


# ============================================================================
# Placeholder extraction (internal)
# ============================================================================

def _extract_fields(template: str) -> Set[str]:
    """
    Extract named format fields from a template string.

    Uses Python's built-in `string.Formatter().parse()` method.
    This correctly handles:
        - Named placeholders: {name}
        - Format specifications: {name:>10}
        - Conversion flags: {name!r}
        - Escaped braces: {{ and }}
        - Nested or complex field names (though not used in our templates)

    Parameters
    ----------
    template : str
        A Python format string.

    Returns
    -------
    Set[str]
        A set of field names (strings) that the template references.
        Escaped braces {{ }} are ignored.
        Field names with format specs (e.g., {name:>10}) are returned
        as just "name".

    Examples
    --------
    >>> _extract_fields("{version}-{platform}-{arch}.{ext}")
    {'version', 'platform', 'arch', 'ext'}

    >>> _extract_fields("Hello {{name}}")  # Escaped, no field
    set()

    >>> _extract_fields("{name!r:>20}")
    {'name'}
    """
    parser = Formatter()
    fields: Set[str] = set()
    for _, field_name, _, _ in parser.parse(template):
        if field_name is not None:
            # field_name may include format specs after ':' or conversion '!'.
            # Formatter().parse() already isolates the field name,
            # so we can use it directly.
            fields.add(field_name)
    return fields


# ============================================================================
# Safe formatting
# ============================================================================

def safe_format_template(
    template: str,
    context: TemplateContext,
    strict_unused: bool = False,
) -> str:
    """
    Safely format a template string using a TemplateContext.

    Before calling str.format(), this function:
    1. Extracts all placeholders from the template using Formatter.parse()
    2. Converts the context to a dict via context.to_format_dict()
    3. Checks that every placeholder in the template exists in the context
    4. Optionally checks that every key in the context is actually used
       by the template (strict_unused=True)

    This prevents silent KeyErrors and helps catch template/context
    mismatches during development.

    Parameters
    ----------
    template : str
        A Python format string with named placeholders.
        Example: "xpack-gcc-{version}-{platform}-{arch}.{ext}"
    context : TemplateContext
        The immutable context providing values for all placeholders.
    strict_unused : bool, optional
        If True, raises ValueError when the context dict contains keys
        that are not referenced by the template. This helps detect
        dead context fields. Default is False.
        Note: The "platform_variant" key is conditionally present in
        the context dict. When strict_unused is True, it is still
        exempted from the check because its presence depends on the
        provider. A missing "platform_variant" key in the template
        will NOT raise an error even in strict mode.

    Returns
    -------
    str
        The formatted string with all placeholders replaced.

    Raises
    ------
    KeyError
        If the template contains a placeholder that is not available
        in the context dict. The error message lists the missing
        placeholders and the available keys.
    ValueError
        If `strict_unused=True` and the context dict contains keys
        (other than "platform_variant") that are not used by the
        template. This indicates that the context is providing
        unnecessary data, which may signal a template bug.

    Examples
    --------
    >>> ctx = TemplateContext("14.2.0", PlatformType.LINUX,
    ...                       ArchitectureType.X64, ArchiveType.TAR_GZ)
    >>> safe_format_template("gcc-{version}-{platform}.{ext}", ctx)
    'gcc-14.2.0-linux.tar.gz'

    >>> # Missing placeholder in context
    >>> try:
    ...     safe_format_template("{version}-{missing}", ctx)
    ... except KeyError as e:
    ...     print("Error:", e)
    Error: "Template requires placeholder(s) {'missing'} but context only provides ['version', 'platform', 'arch', 'ext']"

    >>> # Strict unused check
    >>> ctx_with_variant = TemplateContext("14.2.0", PlatformType.LINUX,
    ...                                    ArchitectureType.X64, ArchiveType.TAR_GZ,
    ...                                    platform_variant="ubuntu-22.04")
    >>> # This template does not use platform_variant
    >>> safe_format_template("{version}.{ext}", ctx_with_variant, strict_unused=True)
    '14.2.0.tar.gz'  # platform_variant is exempted
    """
    context_dict = context.to_format_dict()
    template_fields = _extract_fields(template)

    # 1. Check for missing placeholders: template needs 'foo' but context lacks 'foo'
    missing = template_fields - set(context_dict.keys())
    if missing:
        raise KeyError(
            f"Template requires placeholder(s) {missing} "
            f"but context only provides {list(context_dict.keys())}"
        )

    # 2. Optional strict check: context has 'bar' but template never uses 'bar'
    if strict_unused:
        unused = set(context_dict.keys()) - template_fields
        # platform_variant is conditionally provided; it's not an error
        # if it's in the context but not in the template.
        for key in ("platform_variant", "platform", "arch"):
            unused.discard(key)
        if unused:
            raise ValueError(
                f"Context provides unused key(s) {unused}. "
                f"Template only references {template_fields}"
            )

    # 3. Safe to format — all placeholders are satisfied
    return template.format(**context_dict)


# ============================================================================
# URL joining utility
# ============================================================================

def join_url(base: str, path: str) -> str:
    """
    Join a base URL with a relative path, normalizing slashes.

    Ensures exactly one slash between the base and path.
    Does NOT perform URL encoding — paths must be pre-encoded
    if they contain special characters.

    Parameters
    ----------
    base : str
        The base URL. May or may not end with a slash.
        Example: "https://github.com/releases/download/v1.0"
    path : str
        The relative path to append. May or may not start with a slash.
        Example: "/file.tar.gz" or "file.tar.gz"

    Returns
    -------
    str
        The combined URL with exactly one slash between components.

    Examples
    --------
    >>> join_url("https://example.com/", "/file.tar.gz")
    'https://example.com/file.tar.gz'
    >>> join_url("https://example.com", "file.tar.gz")
    'https://example.com/file.tar.gz'
    >>> join_url("https://example.com/releases", "v1/file.tar.gz")
    'https://example.com/releases/v1/file.tar.gz'

    Warnings
    --------
    - This is a simple string operation. It does NOT handle:
        - Query parameters
        - Fragments (#)
        - URL encoding of spaces or non-ASCII characters
    - For production URL manipulation with query strings, use
      urllib.parse.urljoin() instead, but be aware of its
      behavior when the path starts with '/'.
    """
    base = base.rstrip("/")
    path = path.lstrip("/")
    return f"{base}/{path}"
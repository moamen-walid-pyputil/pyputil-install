#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Retry mechanism with exponential backoff and jitter.

This module provides decorators and context managers for automatically
retrying operations that may fail transiently. It implements exponential
backoff with full jitter to prevent thundering herd problems and includes
support for conditional retries based on exception types.

Classes
-------
RetryConfig
    Configuration object controlling retry behavior.
RetryManager
    Context manager for retry logic with detailed statistics.

Functions
--------
retry
    Decorator that adds retry behavior to a function.

Examples
--------
>>> from packman.exceptions import NetworkError
>>> @retry(max_attempts=3, retryable_exceptions=(NetworkError,))
... def fetch_package(name):
...     # network operation
...     pass
"""

import time
import random
import logging
import functools
from typing import Type, Tuple, Optional, Callable, Any, Dict, List, Union
from dataclasses import dataclass, field
from collections import defaultdict

from .exceptions import NetworkError, TimeoutError

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.

    Parameters
    ----------
    max_attempts : int, default=3
        Maximum number of attempts including the initial call.
    min_wait : float, default=1.0
        Minimum wait time between attempts in seconds.
    max_wait : float, default=60.0
        Maximum wait time between attempts in seconds.
    backoff_factor : float, default=2.0
        Multiplicative factor for exponential backoff.
    jitter : bool, default=True
        Whether to add random jitter to wait times. Uses full jitter
        when True, which is optimal for preventing thundering herd.
    retryable_exceptions : tuple of exception types, optional
        Exception types that trigger a retry. If None, retries on
        any Exception. Subclasses of specified types are also matched.
    non_retryable_exceptions : tuple of exception types, optional
        Exception types that immediately abort retries. These take
        precedence over retryable_exceptions.

    Notes
    -----
    The wait time calculation with full jitter is::

        wait = random.uniform(0, min(max_wait, min_wait * backoff_factor ** attempt))

    This is based on the "full jitter" algorithm from AWS Architecture
    Blog, which provides optimal distribution of retry attempts in
    distributed systems.

    Examples
    --------
    >>> config = RetryConfig(max_attempts=5, min_wait=0.5, max_wait=30.0)
    >>> config.max_attempts
    5
    """

    max_attempts: int = 3
    min_wait: float = 1.0
    max_wait: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
    non_retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if self.max_attempts < 1:
            raise ValueError(
                f"max_attempts must be >= 1, got {self.max_attempts}"
            )
        if self.min_wait < 0:
            raise ValueError(
                f"min_wait must be >= 0, got {self.min_wait}"
            )
        if self.max_wait < self.min_wait:
            raise ValueError(
                f"max_wait ({self.max_wait}) must be >= min_wait ({self.min_wait})"
            )
        if self.backoff_factor < 1:
            raise ValueError(
                f"backoff_factor must be >= 1, got {self.backoff_factor}"
            )

    def calculate_wait(self, attempt: int) -> float:
        """
        Calculate wait time for a given attempt number.

        Parameters
        ----------
        attempt : int
            Zero-based attempt index. Attempt 0 is the first retry
            after the initial call.

        Returns
        -------
        float
            Wait time in seconds before the next attempt.

        Notes
        -----
        Implements the "full jitter" algorithm when `jitter` is True.
        Without jitter, uses standard exponential backoff capped at
        max_wait.

        Raises
        ------
        ValueError
            If attempt is negative.
        """
        if attempt < 0:
            raise ValueError(f"attempt must be >= 0, got {attempt}")

        raw_wait = self.min_wait * (self.backoff_factor ** attempt)
        capped_wait = min(raw_wait, self.max_wait)

        if self.jitter:
            return random.uniform(0, capped_wait)
        return capped_wait


class _RetryState:
    """
    Internal state tracker for retry operations.

    Parameters
    ----------
    config : RetryConfig
        The retry configuration being used.

    Attributes
    ----------
    attempts : int
        Total number of attempts made so far.
    exceptions : list
        List of exceptions caught during retries.
    total_wait : float
        Cumulative wait time across all retries.
    """

    def __init__(self, config: RetryConfig) -> None:
        self.config = config
        self.attempts: int = 0
        self.exceptions: List[Exception] = []
        self.total_wait: float = 0.0
        self._start_time: float = 0.0

    def record_attempt(self) -> None:
        """Increment attempt counter."""
        self.attempts += 1

    def record_exception(self, exc: Exception) -> None:
        """
        Record an exception that triggered a retry.

        Parameters
        ----------
        exc : Exception
            The exception to record.
        """
        self.exceptions.append(exc)

    def record_wait(self, wait_seconds: float) -> None:
        """
        Record wait time spent before a retry.

        Parameters
        ----------
        wait_seconds : float
            Time waited in seconds.
        """
        self.total_wait += wait_seconds

    @property
    def elapsed(self) -> float:
        """Total elapsed time since first attempt."""
        return time.monotonic() - self._start_time

    def start(self) -> None:
        """Mark the start time of the operation."""
        self._start_time = time.monotonic()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to a dictionary for logging or debugging.

        Returns
        -------
        dict
            Dictionary with attempts, exceptions, wait times, and elapsed.
        """
        return {
            "attempts": self.attempts,
            "exceptions": [str(e) for e in self.exceptions],
            "total_wait": round(self.total_wait, 3),
            "elapsed": round(self.elapsed, 3),
        }


def _should_retry(
    exc: Exception,
    config: RetryConfig,
) -> bool:
    """
    Determine if an exception should trigger a retry.

    Parameters
    ----------
    exc : Exception
        The exception to evaluate.
    config : RetryConfig
        Configuration specifying retryable and non-retryable exceptions.

    Returns
    -------
    bool
        True if the exception warrants a retry.

    Notes
    -----
    Non-retryable exceptions take precedence: if the exception matches
    a type in `non_retryable_exceptions`, this returns False regardless
    of `retryable_exceptions`.
    """
    if config.non_retryable_exceptions is not None:
        if isinstance(exc, config.non_retryable_exceptions):
            logger.debug(
                f"Exception {type(exc).__name__} is non-retryable, aborting"
            )
            return False

    if config.retryable_exceptions is not None:
        return isinstance(exc, config.retryable_exceptions)

    return True


class RetryManager:
    """
    Context manager providing retry logic with detailed statistics.

    This context manager wraps a block of code and retries it on failure
    according to the provided configuration. It collects statistics
    about the retry operation for inspection after completion.

    Parameters
    ----------
    config : RetryConfig, optional
        Retry configuration. If not provided, default configuration is used.
    on_retry : callable, optional
        Callback invoked before each retry attempt with signature
        ``on_retry(exception, attempt, wait_seconds)``.

    Attributes
    ----------
    state : _RetryState or None
        The state of the retry operation. None before the context
        manager is entered.

    Examples
    --------
    >>> config = RetryConfig(max_attempts=3)
    >>> with RetryManager(config) as rm:
    ...     # perform operation
    ...     result = risky_operation()
    >>> print(rm.state.attempts)
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        on_retry: Optional[
            Callable[[Exception, int, float], None]
        ] = None,
    ) -> None:
        self.config = config if config is not None else RetryConfig()
        self.on_retry = on_retry
        self.state: Optional[_RetryState] = None

    def __enter__(self) -> "RetryManager":
        """Initialize retry state and return self."""
        self.state = _RetryState(self.config)
        self.state.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """
        Handle context exit, triggering retries if necessary.

        Parameters
        ----------
        exc_type : type or None
            Exception type if an exception occurred.
        exc_val : Exception or None
            Exception instance if an exception occurred.
        exc_tb : traceback or None
            Traceback if an exception occurred.

        Returns
        -------
        bool
            True if the exception was handled (retries exhausted
            gracefully), False to propagate.

        Notes
        -----
        This method implements the retry loop internally. It is not
        designed to be called directly by users.
        """
        if exc_val is None:
            return False

        if not isinstance(exc_val, Exception):
            return False

        while self.state.attempts < self.config.max_attempts:
            self.state.record_attempt()
            self.state.record_exception(exc_val)

            if not _should_retry(exc_val, self.config):
                logger.error(
                    f"Exception {type(exc_val).__name__} not retryable, "
                    f"aborting after {self.state.attempts} attempt(s)"
                )
                return False

            if self.state.attempts >= self.config.max_attempts:
                break

            wait = self.config.calculate_wait(self.state.attempts - 1)
            self.state.record_wait(wait)

            logger.warning(
                f"Attempt {self.state.attempts}/{self.config.max_attempts} "
                f"failed with {type(exc_val).__name__}: {exc_val}. "
                f"Retrying in {wait:.2f}s..."
            )

            if self.on_retry is not None:
                try:
                    self.on_retry(exc_val, self.state.attempts, wait)
                except Exception as callback_error:
                    logger.error(
                        f"on_retry callback failed: {callback_error}"
                    )

            time.sleep(wait)
            return True

        logger.error(
            f"All {self.config.max_attempts} attempts exhausted. "
            f"Last error: {type(exc_val).__name__}: {exc_val}"
        )
        return False


def retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[
        Union[Type[Exception], Tuple[Type[Exception], ...]]
    ] = None,
    non_retryable_exceptions: Optional[
        Union[Type[Exception], Tuple[Type[Exception], ...]]
    ] = None,
    on_retry: Optional[
        Callable[[Exception, int, float], None]
    ] = None,
) -> Callable:
    """
    Decorator that adds retry behavior to a function.

    Parameters
    ----------
    max_attempts : int, default=3
        Maximum number of attempts including the initial call.
    min_wait : float, default=1.0
        Minimum wait time between retries in seconds.
    max_wait : float, default=60.0
        Maximum wait time between retries in seconds.
    backoff_factor : float, default=2.0
        Multiplicative factor for exponential backoff.
    jitter : bool, default=True
        Whether to apply random jitter to wait times.
    retryable_exceptions : exception type or tuple, optional
        Exception type(s) that should trigger a retry. If None,
        all exceptions are retryable.
    non_retryable_exceptions : exception type or tuple, optional
        Exception type(s) that should immediately abort retries.
    on_retry : callable, optional
        Callback invoked before each retry with signature
        ``on_retry(exception, attempt_number, wait_seconds)``.

    Returns
    -------
    callable
        Decorated function with retry behavior.

    Raises
    ------
    ValueError
        If max_attempts is less than 1 or timing parameters are invalid.

    Notes
    -----
    The decorated function preserves its metadata (name, docstring,
    etc.) via `functools.wraps`.

    The decorator can be used with or without parentheses::

        @retry
        def func():
            pass

        @retry(max_attempts=5, min_wait=2.0)
        def func():
            pass

    Examples
    --------
    >>> from packman.exceptions import NetworkError
    >>> @retry(
    ...     max_attempts=3,
    ...     retryable_exceptions=(NetworkError, TimeoutError),
    ...     min_wait=1.0,
    ...     max_wait=10.0,
    ... )
    ... def download_package(url: str) -> bytes:
    ...     # network operation that may fail transiently
    ...     pass

    With custom retry callback::

    >>> def log_retry(exc, attempt, wait):
    ...     print(f"Retry {attempt} after {wait:.1f}s: {exc}")
    >>> @retry(on_retry=log_retry)
    ... def unstable_operation():
    ...     pass
    """
    if isinstance(retryable_exceptions, type) and issubclass(
        retryable_exceptions, Exception
    ):
        retryable_exceptions = (retryable_exceptions,)

    if isinstance(non_retryable_exceptions, type) and issubclass(
        non_retryable_exceptions, Exception
    ):
        non_retryable_exceptions = (non_retryable_exceptions,)

    config = RetryConfig(
        max_attempts=max_attempts,
        min_wait=min_wait,
        max_wait=max_wait,
        backoff_factor=backoff_factor,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions,
        non_retryable_exceptions=non_retryable_exceptions,
    )

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            state = _RetryState(config)
            state.start()

            for attempt in range(config.max_attempts):
                try:
                    state.record_attempt()
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(
                            f"{func.__name__} succeeded on attempt "
                            f"{state.attempts}/{config.max_attempts}"
                        )
                    return result
                except Exception as exc:
                    state.record_exception(exc)

                    if not _should_retry(exc, config):
                        logger.error(
                            f"{func.__name__} failed with non-retryable "
                            f"{type(exc).__name__}: {exc}"
                        )
                        raise

                    if attempt == config.max_attempts - 1:
                        logger.error(
                            f"{func.__name__} failed after "
                            f"{config.max_attempts} attempt(s). "
                            f"State: {state.to_dict()}"
                        )
                        raise

                    wait = config.calculate_wait(attempt)
                    state.record_wait(wait)

                    logger.warning(
                        f"{func.__name__} attempt {state.attempts}/"
                        f"{config.max_attempts} failed: "
                        f"{type(exc).__name__}: {exc}. "
                        f"Retrying in {wait:.2f}s..."
                    )

                    if on_retry is not None:
                        try:
                            on_retry(exc, state.attempts, wait)
                        except Exception as callback_error:
                            logger.error(
                                f"on_retry callback failed: {callback_error}"
                            )

                    time.sleep(wait)

            raise RuntimeError("Unreachable: retry loop exhausted")

        wrapper._retry_config = config
        return wrapper

    return decorator


def is_retryable_exception(
    exc: Exception,
    retryable_types: Optional[Tuple[Type[Exception], ...]] = None,
) -> bool:
    """
    Check if an exception is considered retryable.

    Parameters
    ----------
    exc : Exception
        The exception to check.
    retryable_types : tuple of exception types, optional
        Exception types considered retryable. If None, checks against
        default retryable types (NetworkError, TimeoutError).

    Returns
    -------
    bool
        True if the exception is retryable.

    Notes
    -----
    This utility function is useful for manual retry logic outside
    of the decorator or context manager.

    Examples
    --------
    >>> from packman.exceptions import NetworkError, PackageInstallError
    >>> is_retryable_exception(NetworkError("timeout", url="http://example.com"))
    True
    >>> is_retryable_exception(PackageInstallError("requests", message="syntax error"))
    False
    """
    if retryable_types is None:
        retryable_types = (NetworkError, TimeoutError)
    return isinstance(exc, retryable_types)


def combine_retry_configs(
    *configs: RetryConfig,
) -> RetryConfig:
    """
    Merge multiple RetryConfig instances, taking the most conservative values.

    Parameters
    ----------
    *configs : RetryConfig
        Variable number of RetryConfig instances to merge.

    Returns
    -------
    RetryConfig
        A new configuration using max of max_attempts and max_wait,
        and min of min_wait from all inputs.

    Notes
    -----
    For retryable_exceptions and non_retryable_exceptions, the union
    of all specified types is used.

    Examples
    --------
    >>> c1 = RetryConfig(max_attempts=3, min_wait=1.0)
    >>> c2 = RetryConfig(max_attempts=5, min_wait=0.5)
    >>> combined = combine_retry_configs(c1, c2)
    >>> combined.max_attempts
    5
    >>> combined.min_wait
    0.5
    """
    if not configs:
        return RetryConfig()

    max_attempts = max(c.max_attempts for c in configs)
    min_wait = min(c.min_wait for c in configs)
    max_wait = max(c.max_wait for c in configs)
    backoff_factor = max(c.backoff_factor for c in configs)
    jitter = any(c.jitter for c in configs)

    retryable_types: List[Type[Exception]] = []
    non_retryable_types: List[Type[Exception]] = []

    for c in configs:
        if c.retryable_exceptions:
            retryable_types.extend(c.retryable_exceptions)
        if c.non_retryable_exceptions:
            non_retryable_types.extend(c.non_retryable_exceptions)

    return RetryConfig(
        max_attempts=max_attempts,
        min_wait=min_wait,
        max_wait=max_wait,
        backoff_factor=backoff_factor,
        jitter=jitter,
        retryable_exceptions=(
            tuple(set(retryable_types)) if retryable_types else None
        ),
        non_retryable_exceptions=(
            tuple(set(non_retryable_types)) if non_retryable_types else None
        ),
    )


class _ExponentialBackoffIterator:
    """
    Iterator yielding wait times for exponential backoff with jitter.

    Parameters
    ----------
    config : RetryConfig
        Configuration for backoff behavior.

    Yields
    ------
    float
        Wait time in seconds for the next retry attempt.

    Notes
    -----
    This is an internal utility class used when manual iteration
    over backoff times is needed instead of the decorator or
    context manager approach.
    """

    def __init__(self, config: RetryConfig) -> None:
        self._config = config
        self._attempt: int = 0

    def __iter__(self) -> "_ExponentialBackoffIterator":
        return self

    def __next__(self) -> float:
        if self._attempt >= self._config.max_attempts:
            raise StopIteration
        wait = self._config.calculate_wait(self._attempt)
        self._attempt += 1
        return wait
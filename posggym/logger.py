"""posggym specific logging.

Adapted from Farama Foundation Gymnasium:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/logger.py

"""
import sys
import warnings
from typing import Optional, Type

from gymnasium.utils import colorize


DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50

min_level = 30


def set_level(level: int):
    """Set logging threshold on current logger."""
    global min_level
    min_level = level


def debug(msg: str, *args):
    """Log debug message."""
    if min_level <= DEBUG:
        print(f"DEBUG: {msg % args}", file=sys.stderr)


def info(msg: str, *args):
    """Log info message."""
    if min_level <= INFO:
        print(f"INFO: {msg % args}", file=sys.stderr)


def warn(
    msg: str,
    *args: object,
    category: Optional[Type[Warning]] = None,
    stacklevel: int = 1,
):
    """Raises a warning to the user if the min_level <= WARN.

    Arguments
    ---------
    msg: str
        The message to warn the user
    *args: object
        Additional information to warn the user
    category: Type[Warning], Optional
        The category of warning (default=None)
    stacklevel: int, Optional
        The stack level to raise to (default=1)

    """
    if min_level <= WARN:
        warnings.warn(
            colorize(f"WARN: {msg % args}", "yellow"),
            category=category,
            stacklevel=stacklevel + 1,
        )


def deprecation(msg: str, *args: object):
    """Logs a deprecation warning to users."""
    warn(msg, *args, category=DeprecationWarning, stacklevel=2)


def error(msg: str, *args):
    """Log error message."""
    if min_level <= ERROR:
        print(colorize(f"ERROR: {msg % args}", "red"), file=sys.stderr)

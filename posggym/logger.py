"""posggym specific logging.

Adapted from OPEN AI gym:
https://github.com/openai/gym/blob/v0.21.0/gym/logger.py

"""
import warnings

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50

MIN_LEVEL = 30


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(string, color, bold=False, highlight=False):
    """Return string surrounded by terminal color codes to print colorized text.

    Valid colors: gray, red, green, yellow, blue, magenta, cyan, white, crimson

    Ref: https://github.com/openai/gym/blob/v0.21.0/gym/utils/colorize.py

    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    attrs = ";".join(attr)
    return "\x1b[%sm%s\x1b[0m" % (attrs, string)


def set_level(level):
    """Set logging threshold on current logger."""
    global MIN_LEVEL
    MIN_LEVEL = level


def debug(msg, *args):
    """Log debug message."""
    if MIN_LEVEL <= DEBUG:
        print("%s: %s" % ("DEBUG", msg % args))


def info(msg, *args):
    """Log info message."""
    if MIN_LEVEL <= INFO:
        print("%s: %s" % ("INFO", msg % args))


def warn(msg, *args):
    """Log warning message."""
    if MIN_LEVEL <= WARN:
        warnings.warn(colorize("%s: %s" % ("WARN", msg % args), "yellow"))


def error(msg, *args):
    """Log error message."""
    if MIN_LEVEL <= ERROR:
        print(colorize("%s: %s" % ("ERROR", msg % args), "red"))

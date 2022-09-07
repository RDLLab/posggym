"""posggym specific errors.

Adapted from OPEN AI gym:
https://github.com/openai/gym/blob/v0.21.0/gym/error.py

"""


class Error(Exception):
    """Base posggym error."""
    pass


class Unregistered(Error):
    """Raised when user requests item from registry that doesn't exist."""
    pass


class UnregisteredEnv(Unregistered):
    """Raised when user requests an env from registry that doesn't exist."""
    pass


# Video errors

class VideoRecorderError(Error):
    """Video recorder error."""
    pass


class InvalidFrame(Error):
    """Invalid video frame error."""
    pass

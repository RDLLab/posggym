"""posggym specific errors.

Adapted from Farama Foundation gymnasium, copied here to so that error source path is
reported correctly so as to avoid any confusion.
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/error.py

"""


class Error(Exception):
    """Base posggym error."""


class Unregistered(Error):
    """Raised when user requests item from registry that doesn't exist."""


class UnregisteredEnv(Unregistered):
    """Raised when user requests env from registry that doesn't exist."""


class NamespaceNotFound(UnregisteredEnv):
    """Raised when user requests env from registry where namespace doesn't exist."""


class NameNotFound(UnregisteredEnv):
    """Raised when user requests env from registry where name doesn't exist."""


class VersionNotFound(UnregisteredEnv):
    """Raised when user requests env from registry where version doesn't exist."""


class DeprecatedEnv(Error):
    """Raised when user requests env from registry with old version.

    I.e. if the version number is older than the latest version env with the same
    name.
    """


class RegistrationError(Error):
    """Raised when the user attempts to register an invalid env.

    For example, an unversioned env when a versioned env exists.
    """


class UnsupportedMode(Error):
    """Raised when user requests rendering mode not supported by the environment."""


class InvalidMetadata(Error):
    """Raised when the metadata of an environment is not valid."""


# Video errors


class VideoRecorderError(Error):
    """Video recorder error."""

    pass


class InvalidFrame(Error):
    """Invalid video frame error."""

    pass

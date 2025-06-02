"""posggym specific errors.

Adapted from Farama Foundation gymnasium, copied here to so that error source path is
reported correctly so as to avoid any confusion.
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/error.py

"""


class Error(Exception):
    """Base posggym error."""


class UnregisteredError(Error):
    """Raised when user requests item from registry that doesn't exist."""


class UnregisteredEnvError(UnregisteredError):
    """Raised when user requests env from registry that doesn't exist."""


class NamespaceNotFoundError(UnregisteredEnvError):
    """Raised when user requests env from registry where namespace doesn't exist."""


class NameNotFoundError(UnregisteredEnvError):
    """Raised when user requests env from registry where name doesn't exist."""


class VersionNotFoundError(UnregisteredEnvError):
    """Raised when user requests env from registry where version doesn't exist."""


class DeprecatedEnvError(Error):
    """Raised when user requests env from registry with old version.

    I.e. if the version number is older than the latest version env with the same
    name.
    """


class RegistrationError(Error):
    """Raised when the user attempts to register an invalid env.

    For example, an unversioned env when a versioned env exists.
    """


class UnseedableEnvError(Error):
    """Raised when the user tries to seed an env that does not support seeding."""


class DependencyNotInstalledError(Error):
    """Raised when the user has not installed a dependency."""


class UnsupportedModeError(Error):
    """Raised when user requests rendering mode not supported by the environment."""


class InvalidMetadataError(Error):
    """Raised when the metadata of an environment is not valid."""


class ResetNeededError(Error):
    """Raised when the user attempts to step environment before a reset."""


class ResetNotAllowedError(Error):
    """Raised when user tries to reset an environment that's not done.

    Applicable when monitor is active.
    """


class InvalidActionError(Error):
    """Raised when the user performs an action not contained within the action space."""


class MissingArgumentError(Error):
    """Raised when a required argument in the initializer is missing."""


class InvalidProbabilityError(Error):
    """Raised when given an invalid value for a probability."""


class InvalidBoundError(Error):
    """Raised when the clipping an array with invalid upper and/or lower bound."""


# Video errors


class VideoRecorderErrorError(Error):
    """Video recorder error."""


class InvalidFrameError(Error):
    """Invalid video frame error."""


# posggym.agent specific errors


class UnregisteredPolicyError(UnregisteredError):
    """Raised when user requests policy from registry that doesn't exist."""


class PolicyEnvIDNotFoundError(UnregisteredPolicyError):
    """Raised when user requests policy from registry with env-id that doesn't exist."""


class PolicyEnvArgsIDNotFoundError(UnregisteredPolicyError):
    """Raised when user requests policy from registry with env-args that don't exist."""


class PolicyNameNotFoundError(UnregisteredPolicyError):
    """Raised when user requests policy from registry where name doesn't exist."""


class PolicyVersionNotFoundError(UnregisteredPolicyError):
    """Raised when user requests policy from registry where version doesn't exist."""


class DeprecatedPolicyError(Error):
    """Raised when user requests policy from registry with old version.

    I.e. if the version number is older than the latest version env with the same
    name.
    """


class PolicyRegistrationError(Error):
    """Raised when the user attempts to register an invalid policy.

    For example, an unversioned policy when a versioned env exists.
    """


class UnseedablePolicyError(Error):
    """Raised when the user tries to seed an policy that does not support seeding."""


class InvalidFileError(Error):
    """Raised when trying to access and invalid posggym file."""


class DownloadError(Error):
    """Raised when error occurred while trying to download posggym file."""

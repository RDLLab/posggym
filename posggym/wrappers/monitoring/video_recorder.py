"""Classes for recording videos of episodes.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/wrappers/monitoring/video_recorder.py

"""
import json
import os
import os.path
import tempfile
from typing import Dict, Optional, List

from posggym import Env, error, logger


class VideoRecorder:
    """VideoRecorder renders a nice movie of a rollout, frame by frame.

    It comes with an `enabled` option so you can still use the same code on
    episodes where you don't want to record video.

    This implementation is the same as the gymnasium VideoRecorder class but
    makes it compatible with posggym.env.render function which returns rgb
    arrays for the whole environment as well as (optionally) each agent.

    Note
    ----
    You are responsible for calling `close` on a created VideoRecorder, or else
    you may leak an encoder process.

    """

    combatible_render_modes = ["rgb_array", "rgb_array_dict"]

    def __init__(
        self,
        env: Env,
        path: Optional[str] = None,
        metadata: Optional[Dict] = None,
        enabled: bool = True,
        base_path: Optional[str] = None,
        disable_logger: bool = False,
    ):
        """Video recorder renders a nice movie of a rollout, frame by frame.

        Arguments
        ---------
        env: Environment to take video of.
        path: Path to the video file; will be randomly chosen if omitted.
        metadata: Contents to save to the metadata file.
        enabled: Whether to actually record video, or just no-op (for convenience)
        base_path: Alternatively, path to the video file without extension, which will
            be added.
        disable_logger: Whether to disable moviepy logger or not.

        """
        try:
            # check that moviepy is now installed
            import moviepy  # noqa
        except ImportError as e:
            raise error.DependencyNotInstalled(
                "MoviePy is not installed, run `pip install moviepy`"
            ) from e

        self._async = env.metadata.get("semantics.async")
        self.enabled = enabled
        self.disable_logger = disable_logger
        self._closed = False

        self.render_history: List = []
        self.env = env

        self.render_mode = env.render_mode

        if self.render_mode not in self.combatible_render_modes:
            logger.warn(
                f"Disabling video recorder because environment {env} was not "
                "initialized with any compatible video modes in "
                f"{self.combatible_render_modes}."
            )
            # Disable since the env not initialized with a compatible `render_mode`
            self.enabled = False

        # Don't bother setting anything else if not enabled
        if not self.enabled:
            return

        if path is not None and base_path is not None:
            raise error.Error("You can pass at most one of `path` or `base_path`.")

        self.last_frame = None
        self.env = env

        required_ext = ".mp4"
        if path is None:
            if base_path is not None:
                # Base path given, append ext
                path = base_path + required_ext
            else:
                # Otherwise, just generate a unique filename
                with tempfile.NamedTemporaryFile(suffix=required_ext) as f:
                    path = f.name
        self.path = path

        path_base, actual_ext = os.path.splitext(self.path)

        if actual_ext != required_ext:
            raise error.Error(
                f"Invalid path given: {self.path} -- must have file extension "
                f"{required_ext}."
            )

        self.frames_per_sec = env.metadata.get("render_fps", 30)

        self.broken = False

        # Dump metadata
        self.metadata = metadata or {}
        self.metadata["content_type"] = "video/mp4"
        self.metadata_path = f"{path_base}.meta.json"
        self.write_metadata()

        logger.info("Starting new video recorder writing to %s", self.path)
        self.recorded_frames: List = []

    @property
    def functional(self) -> bool:
        """Returns if the video recorder is functional, is enabled and not broken."""
        return self.enabled and not self.broken

    def capture_frame(self):
        """Render the given `env` and add the resulting frame to the video.

        Note, this function has been changed from the source implementation.
        """
        if not self.functional:
            return

        frame = self.env.render()
        if isinstance(frame, list):
            self.render_history += frame
            frame = frame[-1]
        elif isinstance(frame, dict):
            if "env" not in frame:
                logger.warn(
                    "The video recorder expects an entry with the key `env` when "
                    "trying to record an environment that is using the "
                    "`rgb_array_dict` render mode."
                )
                self.broken = True
                return
            frame = frame["env"]

        if self._closed:
            logger.warn(
                "The video recorder has been closed and no frames will be "
                "captured anymore."
            )
            return
        logger.debug("Capturing video frame: path=%s", self.path)

        if frame is None:
            if self._async:
                return
            else:
                # Indicates a bug in the environment: don't want to raise
                # an error here.
                logger.warn(
                    "Env returned None on `render()`. Disabling further rendering for "
                    f"video recorder by marking as disabled: path={self.path} "
                    f"metadata_path={self.metadata_path}"
                )
                self.broken = True
        else:
            self.recorded_frames.append(frame)

    def close(self):
        """Flush all data to disk and close any open frame encoders."""
        if not self.enabled or self._closed:
            # mark as closed in case it is broken to save future warnings
            self._closed = True
            return

        if len(self.recorded_frames) > 0:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    "MoviePy is not installed, run `pip install moviepy`"
                ) from e

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            moviepy_logger = None if self.disable_logger else "bar"
            clip.write_videofile(self.path, logger=moviepy_logger)
        else:
            # No frames captured. Set metadata.
            if self.metadata is None:
                self.metadata = {}
            self.metadata["empty"] = True

        self.write_metadata()

        # Stop tracking this for autoclose
        self._closed = True

    def write_metadata(self):
        """Writes metadata to metadata path."""
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def __del__(self):
        """Closes the environment correctly when the recorder is deleted."""
        # Make sure we've closed up shop when garbage collecting
        if not self._closed:
            logger.warn("Unable to save last video! Did you call close()?")

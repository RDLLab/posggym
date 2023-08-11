"""Wrapper for recording videos of an environment.

Ref
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/wrappers/record_video.py

"""
import os
from typing import Callable, Optional

from posggym import logger
from posggym.core import Env, Wrapper
from posggym.wrappers.monitoring.video_recorder import VideoRecorder


def capped_cubic_video_schedule(episode_id: int) -> bool:
    """Get cubic schedule.

    Triggers recording of episodes numbers that are perfect cubes :math:`k^3` up to
    the 1000th episode, then every 1000 episodes after that:
    0, 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000, 2000, 3000, ...

    Arguments
    ---------
    episode_id: int
      The episode number

    Returns
    -------
    bool
      Whether to record episode or not.

    """
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0


class RecordVideo(Wrapper):
    """Wrapper for recording videos of rollouts.

    Arguments
    ---------
    env: posggym.Env
        The environment that will be wrapped
    video_folder: str
        The folder where the recordings will be stored
    episode_trigger: Optional[Callable[[int], bool]]
        Function that accepts an integer and returns ``True`` iff a recording should be
        started at this episode
    step_trigger: Optional[Callable[[int], bool]]
        Function that accepts an integer and returns ``True`` iff a recording should be
        started at this step
    video_length: int
        The length of recorded episodes. If 0, entire episodes are recorded. Otherwise,
        snippets of the specified length are captured
    name_prefix: str, optional
        Will be prepended to the filename of the recordings
    disable_logger: bool
        Whether to disable moviepy logger or not.

    Note
    ----
    This implementation is based on the gymnasium.wrappers.RecordVideo (version
    gymnasium 0.27) wrapper, adapted here to work with posggym's multiagent environment:
    https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/wrappers/record_video.py

    """

    def __init__(
        self,
        env: Env,
        video_folder: str,
        episode_trigger: Optional[Callable[[int], bool]] = None,
        step_trigger: Optional[Callable[[int], bool]] = None,
        video_length: int = 0,
        name_prefix: str = "posggym-video",
        disable_logger: bool = False,
    ):
        super().__init__(env)

        if episode_trigger is None and step_trigger is None:
            episode_trigger = capped_cubic_video_schedule

        trigger_count = sum(x is not None for x in [episode_trigger, step_trigger])
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.video_recorder: Optional[VideoRecorder] = None
        self.disable_logger = disable_logger

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder (try "
                "specifying a different `video_folder` for the `RecordVideo` wrapper "
                "if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.video_length = video_length
        self.step_id = 0
        self.episode_id = 0

        self.recording = False
        self.episode_done = False
        self.recorded_frames = 0
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_done = False
        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.recorded_frames = []
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.video_length > 0 and self.recorded_frames > self.video_length:
                self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder()
        return observations

    def step(self, actions):
        obs, rewards, terminated, truncated, all_done, info = super().step(actions)

        if not self.episode_done:
            self.episode_done = (not self.is_vector_env and all_done) or (
                self.is_vector_env and all_done[0]
            )

            # increment steps and episodes
            self.step_id += 1
            if self.episode_done:
                self.episode_id += 1

            if self.recording:
                assert self.video_recorder is not None
                self.video_recorder.capture_frame()
                self.recorded_frames += 1
                if self.episode_done or (
                    self.video_length > 0 and self.recorded_frames > self.video_length
                ):
                    self.close_video_recorder()
            elif self._video_enabled():
                self.start_video_recorder()

        return obs, rewards, terminated, truncated, all_done, info

    def render(self, *args, **kwargs):
        """Compute the render frames as specified by environment's render_mode."""
        if self.video_recorder is None or not self.video_recorder.enabled:
            return super().render(*args, **kwargs)

        if len(self.video_recorder.render_history) > 0:
            recorded_frames = [
                self.video_recorder.render_history.pop()
                for _ in range(len(self.video_recorder.render_history))
            ]
            if self.recording:
                return recorded_frames
            else:
                render_output = super().render(*args, **kwargs)
                if isinstance(render_output, dict):
                    if "env" not in render_output:
                        return recorded_frames
                    return recorded_frames + render_output.get("env", [])
                return recorded_frames + render_output
        else:
            return super().render(*args, **kwargs)

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        assert self.episode_trigger is not None
        return self.episode_trigger(self.episode_id)

    def start_video_recorder(self):
        """Starts video recorder."""
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = VideoRecorder(
            env=self.env,
            base_path=base_path,
            metadata={"step_id": self.step_id, "episode_id": self.episode_id},
            disable_logger=self.disable_logger,
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def close_video_recorder(self):
        """Closes the video recorder if currently recording."""
        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 1

    def close(self):
        super().close()
        self.close_video_recorder()

    def __del__(self):
        self.close_video_recorder()

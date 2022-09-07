import os
from typing import Callable, Optional, Tuple, Dict

import posggym.model as M
from posggym import logger
from posggym.core import Env, Wrapper
from posggym.wrappers.monitoring.video_recorder import VideoRecorder


def capped_cubic_video_schedule(episode_id: int) -> bool:
    """Get cubic schedule."""
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0


class RecordVideo(Wrapper):
    """Wrapper for recording videos of rollouts.

    It is based on the gym.wrappers.RecordVideo (version gym>0.22) wrapper.
    Adapted here to work with posggym's multiagent environment.

    https://github.com/openai/gym/blob/0.22.0/gym/wrappers/record_video.py

    """

    def __init__(self,
                 env: Env,
                 video_folder: str,
                 episode_trigger: Optional[Callable[[int], bool]] = None,
                 step_trigger: Optional[Callable[[int], bool]] = None,
                 video_length: int = 0,
                 name_prefix: str = "posggym-video"):
        super().__init__(env)

        if episode_trigger is None and step_trigger is None:
            episode_trigger = capped_cubic_video_schedule

        trigger_count = sum(
            x is not None for x in [episode_trigger, step_trigger]
        )
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.video_recorder = None

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                "(try specifying a different `video_folder` for the "
                "`RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.episode_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs) -> M.JointObservation:
        observations = super().reset(**kwargs)
        if not self.recording and self._video_enabled():
            self.start_video_recorder()
        return observations

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
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
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def step(self,
             actions: Tuple[M.Action, ...]
             ) -> Tuple[M.JointObservation, M.JointReward, bool, Dict]:
        observations, rewards, dones, infos = super().step(actions)

        # increment steps and episodes
        self.step_id += 1
        if not self.is_vector_env:
            if dones:
                self.episode_id += 1
        elif dones[0]:
            self.episode_id += 1

        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames > self.video_length:
                    self.close_video_recorder()
            else:
                if not self.is_vector_env:
                    if dones:
                        self.close_video_recorder()
                elif dones[0]:
                    self.close_video_recorder()

        elif self._video_enabled():
            self.start_video_recorder()

        return observations, rewards, dones, infos

    def close_video_recorder(self):
        """Closes the video recorder if currently recording."""
        if self.recording:
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 1

    def close(self):
        super().close()
        self.close_video_recorder()

    def __del__(self):
        self.close_video_recorder()

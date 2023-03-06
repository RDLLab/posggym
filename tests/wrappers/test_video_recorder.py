"""Tests the VideoRecorder class.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/wrappers/test_video_recorder.py

"""
import os
import re

import pytest

import posggym
from posggym.wrappers.monitoring.video_recorder import VideoRecorder


class BrokenRecordableEnv(posggym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode="rgb_array"):
        self.render_mode = render_mode

    def render(self):
        pass

    def step(self, actions):
        return {}, {}, {}, {}, False, {}

    @property
    def state(self):
        return None


class UnrecordableEnv(posggym.Env):
    metadata = {"render_modes": [None]}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

    def render(self):
        pass

    def step(self, actions):
        return {}, {}, {}, {}, False, {}

    @property
    def state(self):
        return None


def test_record_simple():
    env = posggym.make(
        "Driving-7x7RoundAbout-n2-v0",
        render_mode="rgb_array",
        disable_env_checker=True,
    )
    rec = VideoRecorder(env)
    env.reset()
    rec.capture_frame()
    rec.close()

    assert not rec.broken
    assert os.path.exists(rec.path)
    with open(rec.path) as f:
        assert os.fstat(f.fileno()).st_size > 100


def test_no_frames():
    env = BrokenRecordableEnv()
    rec = VideoRecorder(env)
    rec.close()
    assert rec.functional
    assert not os.path.exists(rec.path)


def test_record_unrecordable_method():
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "\x1b[33mWARN: Disabling video recorder because environment "
            "<UnrecordableEnv instance> was not initialized with any compatible video "
            "modes in"
        ),
    ):
        env = UnrecordableEnv()
        rec = VideoRecorder(env)
        assert not rec.enabled
        rec.close()


def test_record_breaking_render_method():
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "\x1b[33mWARN: Env returned None on `render()`. Disabling further "
            "rendering for video recorder by marking as disabled:"
        ),
    ):
        env = BrokenRecordableEnv()
        rec = VideoRecorder(env)
        rec.capture_frame()
        rec.close()
        assert rec.broken
        assert not os.path.exists(rec.path)


# def test_text_envs():
#     env = posggym.make(
#         "MultiAgentTiger-v0", render_mode="ansi", disable_env_checker=True
#     )
#     video = VideoRecorder(env)
#     try:
#         env.reset()
#         video.capture_frame()
#         video.close()
#     finally:
#         os.remove(video.path)

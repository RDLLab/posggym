"""Test the RecordVideo wrapper.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/wrappers/test_record_video.py

"""
import shutil
from pathlib import Path
import posggym
from posggym.wrappers.record_video import RecordVideo, capped_cubic_video_schedule


def test_record_video_using_default_trigger():
    env = posggym.make(
        "Driving-v1",
        disable_env_checker=True,
        render_mode="rgb_array",
        num_agents=2,
        grid="7x7RoundAbout",
    )
    env = RecordVideo(env, "videos")

    env.reset()
    for _ in range(199):
        actions = {i: act_space.sample() for i, act_space in env.action_spaces.items()}
        _, _, _, _, all_done, _ = env.step(actions)
        if all_done:
            env.reset()
    env.close()
    video_path = Path("videos")
    assert video_path.is_dir()
    mp4_files = [f.name for f in Path("videos").glob("*.mp4")]
    assert len(mp4_files) == sum(
        capped_cubic_video_schedule(i) for i in range(env.episode_id + 1)
    )
    shutil.rmtree("videos")


def test_record_video_reset():
    env = posggym.make(
        "Driving-v1",
        disable_env_checker=True,
        render_mode="rgb_array",
        num_agents=2,
        grid="7x7RoundAbout",
    )
    env = RecordVideo(env, "videos", step_trigger=lambda x: x % 100 == 0)

    obs, info = env.reset()
    env.close()
    assert Path("videos").is_dir()
    shutil.rmtree("videos")
    for i, obs_i in obs.items():
        assert env.observation_spaces[i].contains(obs_i)
    assert isinstance(info, dict)


def test_record_video_step_trigger():
    env = posggym.make(
        "Driving-v1",
        disable_env_checker=True,
        render_mode="rgb_array",
        max_episode_steps=20,
        num_agents=2,
        grid="7x7RoundAbout",
    )
    env = RecordVideo(env, "videos", step_trigger=lambda x: x % 100 == 0)

    env.reset()
    for _ in range(199):
        actions = {i: act_space.sample() for i, act_space in env.action_spaces.items()}
        _, _, _, _, all_done, _ = env.step(actions)
        if all_done:
            env.reset()
    env.close()

    video_path = Path("videos")
    assert video_path.is_dir()
    mp4_files = [f.name for f in Path("videos").glob("*.mp4")]
    assert len(mp4_files) == 2
    shutil.rmtree("videos")

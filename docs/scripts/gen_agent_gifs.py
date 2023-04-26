"""Generates gifs of agents.

Adapted from Gymnasium:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/docs/scripts/gen_gifs.py

"""
import os
import os.path as osp
import re
import argparse
from typing import List

from PIL import Image

import posggym
import posggym.agents as pga

DOCS_DIR = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), os.pardir))

# snake to camel case:
# https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
pattern = re.compile(r"(?<!^)(?=[A-Z])")
# how many steps to record an env for
LENGTH = 300
# height of GIF in pixels, width will be scaled to ensure correct aspec ratio
HEIGHT = 256


def gen_gif(
    env_id: str,
    policy_ids: List[str],
    ignore_existing: bool = False,
    custom_env: bool = False,
    resize: bool = False,
):
    """Gen gif for env."""
    print(env_id)
    for pi_id in policy_ids:
        print(pi_id)

    policy_specs = []
    env_args, env_args_id = None, None
    for policy_id in policy_ids:
        try:
            pi_spec = pga.spec(policy_id)
        except posggym.error.NameNotFound as e:
            if "/" not in policy_id:
                # try prepending env id
                policy_id = f"{env_id}/{policy_id}"
                pi_spec = pga.spec(policy_id)
            else:
                raise e

        if env_args is None and pi_spec.env_args is not None:
            env_args, env_args_id = pi_spec.env_args, pi_spec.env_args_id
        elif pi_spec.env_args is not None:
            assert pi_spec.env_args_id == env_args_id
        policy_specs.append(pi_spec)

    env_args = {} if env_args is None else env_args
    env = posggym.make(
        env_id, disable_env_checker=True, render_mode="rgb_array", **env_args
    )

    policies = {}
    for idx, spec in enumerate(policy_specs):
        agent_id = env.possible_agents[idx]
        policies[agent_id] = pga.make(spec.id, env.model, agent_id)

    # extract env name/type from class
    split = str(type(env.unwrapped)).split(".")
    # get the env type (e.g. grid_world, continuous)
    env_type = "custom_env" if custom_env else split[2]

    # get rid of version info
    env_name = env_id.split("-")[0]
    # convert NameLikeThis to name_like_this
    env_name = pattern.sub("_", env_name).lower()

    # path for saving video
    v_dir_path = os.path.join(DOCS_DIR, "_static", "videos", "agents", env_type)
    os.makedirs(v_dir_path, exist_ok=True)
    v_file_path = os.path.join(v_dir_path, env_name + ".gif")

    if os.path.exists(v_file_path) and not ignore_existing:
        # don't overwrite existing video
        print(
            f"GIF already exists for {env_name} so skipping (Use `--ignore-existing` "
            "to overwrite existing files."
        )
        return

    # obtain and save LENGTH frames worth of steps
    frames: List[Image] = []
    while True:
        obs, _ = env.reset()
        for policy in policies.values():
            policy.reset()
        all_done = False
        while not all_done and len(frames) <= LENGTH:
            frame = env.render()  # type: ignore
            repeat = (
                int(60 / env.metadata["render_fps"]) if env_type == "classic" else 1
            )
            for _ in range(repeat):
                frames.append(Image.fromarray(frame))

            actions = {}
            for i in env.agents:
                if policies[i].observes_state:
                    actions[i] = policies[i].step(env.state)
                else:
                    actions[i] = policies[i].step(obs[i])
            obs, _, _, _, all_done, _ = env.step(actions)

        if len(frames) > LENGTH:
            break

    env.close()

    if resize:
        for idx, img in enumerate(frames):
            # h / w = H / w'
            # w' = Hw/h
            resized_img = img.resize((HEIGHT, int(HEIGHT * img.width / img.height)))
            frames[idx] = resized_img

    frames[0].save(
        os.path.join(v_file_path),
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
    )
    print(f"Saved: {env_name} to {v_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env_id",
        type=str,
        help="ID of environment to run, if None then runs all registered envs.",
    )
    parser.add_argument(
        "-pids",
        "--policy_ids",
        type=str,
        nargs="+",
        help=(
            "List of IDs of policies to compare, one for each agent. Policy IDs should "
            "be provided in order of env.possible_agents (i.e. the first policy ID "
            "will be assigned to the 0-index policy in env.possible_agent, etc.)."
        ),
    )
    parser.add_argument(
        "--ignore-existing",
        action="store_true",
        help="Overwrite existing GIF if it exists.",
    )
    args = parser.parse_args()

    gen_gif(args.env_id, args.policy_ids, args.ignore_existing)
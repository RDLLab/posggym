"""Generates gifs of environments.

Copied and adapted from Gymnasium:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/docs/scripts/gen_gifs.py

"""

import argparse
import re
from pathlib import Path

import posggym
from PIL import Image
from tqdm import tqdm

from utils import kill_strs


DOCS_DIR = Path(__file__).resolve().parent.parent

# snake to camel case:
# https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
pattern = re.compile(r"(?<!^)(?=[A-Z])")
# how many steps to perform in env before saving image
STEPS = 50
# height of PNG in pixels, width will be scaled to ensure correct aspec ratio
HEIGHT = 1024


def gen_img(
    env_id: str,
    ignore_existing: bool = False,
    custom_env: bool = False,
    resize: bool = False,
):
    """Generate image for env."""
    print(env_id)
    env = posggym.make(env_id, disable_env_checker=True, render_mode="rgb_array")

    # extract env name/type from class path
    split = str(type(env.unwrapped)).split(".")
    # get the env type (e.g. Box2D)
    env_type = "custom_env" if custom_env else split[2]

    # get rid of version info
    env_name = env_id.split("-")[0]
    # convert NameLikeThis to name_like_this
    env_name = pattern.sub("_", env_name).lower()

    # path for saving video
    v_dir_path = DOCS_DIR / "_static" / "images" / env_type
    # create dir if it doesn't exist
    v_dir_path.mkdir(exist_ok=True)
    v_file_path = v_dir_path / (env_name + ".png")

    if v_file_path.exists() and not ignore_existing:
        # don't overwrite existing video
        print(
            f"PNG image already exists for {env_name} so skipping (Use "
            "`--ignore-existing` to overwrite existing files."
        )
        return

    # obtain and save STEPS frames worth of steps
    frames = []
    while True:
        env.reset()
        done = False
        while not done and len(frames) <= STEPS:
            rgb_frame = env.render()  # type: ignore
            frames.append(Image.fromarray(rgb_frame))
            action = {i: env.action_spaces[i].sample() for i in env.agents}
            _, _, _, _, done, _ = env.step(action)

        if len(frames) > STEPS:
            break

    env.close()

    frame = frames[-1]
    if resize:
        # h / w = H / w'
        # w' = Hw/h
        frame = frame.resize((HEIGHT, int(HEIGHT * frame.width / frame.height)))

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#png
    frame.save(
        v_file_path,
        dpi=(120, 120),
    )
    print(f"Saved: {env_name} to {v_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default=None,
        help="ID of environment to run, if None then runs all registered envs.",
    )
    parser.add_argument(
        "--ignore_existing",
        action="store_true",
        help="Overwrite existing imange if it exists.",
    )
    args = parser.parse_args()

    if args.env_id is None:
        # iterate through all envspecs
        for env_spec in tqdm(posggym.envs.registry.values()):
            if any(x in str(env_spec.id) for x in kill_strs):
                continue
            # try catch in case missing some installs
            try:
                env = posggym.make(env_spec.id, disable_env_checker=True)
                # the gymnasium needs to be rgb renderable
                if "rgb_array" not in env.metadata["render_modes"]:
                    continue
                gen_img(env_spec.id, args.ignore_existing)
            except BaseException as e:
                print(f"{env_spec.id} ERROR", e)
                continue
    else:
        gen_img(args.env_id, args.ignore_existing)

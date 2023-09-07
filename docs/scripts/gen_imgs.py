"""Generates gifs of environments.

Copied and adapted from Gymnasium:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/docs/scripts/gen_gifs.py

"""
import os
import os.path as osp
import re
from typing import List
from typing_extensions import Annotated

import typer
from PIL import Image
from tqdm import tqdm

import posggym
from utils import kill_strs

DOCS_DIR = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), os.pardir))

# snake to camel case:
# https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
pattern = re.compile(r"(?<!^)(?=[A-Z])")
# how many steps to perform in env before saving image
STEPS = 50
# height of PNG in pixels, width will be scaled to ensure correct aspec ratio
HEIGHT = 1024

app = typer.Typer()


@app.command()
def generate_all_imgs(
    ignore_existing: Annotated[
        bool, typer.Option(help="Overwrite existing GIF if it exists.")
    ] = False,
    custom_env: Annotated[bool, typer.Option()] = False,
    resize: Annotated[bool, typer.Option()] = False,
):
    """Gen image for all envs."""
    for env_spec in tqdm(posggym.envs.registry.values()):
        if any(x in str(env_spec.id) for x in kill_strs):
            continue
        # try catch in case missing some installs
        try:
            env = posggym.make(env_spec.id, disable_env_checker=True)
            # the gymnasium needs to be rgb renderable
            if "rgb_array" not in env.metadata["render_modes"]:
                continue
            gen_img(env_spec.id, ignore_existing, custom_env, resize)
        except BaseException as e:
            print(f"{env_spec.id} ERROR", e)
            continue


@app.command()
def gen_img(
    env_id: Annotated[
        str,
        typer.Option(
            help="ID of environment to run, if None then runs all registered envs."
        ),
    ],
    ignore_existing: Annotated[
        bool, typer.Option(help="Overwrite existing GIF if it exists.")
    ] = False,
    custom_env: Annotated[bool, typer.Option()] = False,
    resize: Annotated[bool, typer.Option()] = False,
):
    """Gen image for env."""
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
    v_dir_path = os.path.join(DOCS_DIR, "_static", "images", env_type)
    # create dir if it doesn't exist
    os.makedirs(v_dir_path, exist_ok=True)
    v_file_path = os.path.join(v_dir_path, env_name + ".png")

    if os.path.exists(v_file_path) and not ignore_existing:
        # don't overwrite existing video
        print(
            f"PNG image already exists for {env_name} so skipping (Use "
            "`--ignore-existing` to overwrite existing files."
        )
        return

    # obtain and save STEPS frames worth of steps
    frames: List[Image] = []
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
        os.path.join(v_file_path),
        dpi=(120, 120),
    )
    print(f"Saved: {env_name} to {v_file_path}")


if __name__ == "__main__":
    app()

"""Generates gifs of environments.

Copied and adapted from Gymnasium:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/docs/scripts/gen_gifs.py

"""
import re
from typing import List
from typing_extensions import Annotated
from pathlib import Path

import typer
from PIL import Image
from tqdm import tqdm

import posggym
from utils import kill_strs

DOCS_DIR = Path(__file__).resolve().parent.parent

# snake to camel case:
# https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
pattern = re.compile(r"(?<!^)(?=[A-Z])")
# how many steps to record an env for
LENGTH = 300
# height of GIF in pixels, width will be scaled to ensure correct aspec ratio
HEIGHT = 256

app = typer.Typer()


@app.command()
def generate_all_gifs(
    ignore_existing: Annotated[
        bool, typer.Option(help="Overwrite existing GIF if it exists.")
    ] = False,
    custom_env: Annotated[bool, typer.Option()] = False,
    resize: Annotated[bool, typer.Option()] = False,
):
    "Generate gif for every environment with an RGB render mode"
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
            gen_gif(env_spec.id, ignore_existing, custom_env, resize)
        except BaseException as e:
            print(f"{env_spec.id} ERROR", e)
            continue


@app.command()
def gen_gif(
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
    """Gen gif for env."""
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
    v_dir_path = DOCS_DIR / "_static" / "videos" / env_type
    # create dir if it doesn't exist
    v_dir_path.mkdir(exist_ok=True)
    v_file_path = v_dir_path / env_name + ".gif"

    if v_file_path.exist() and not ignore_existing:
        # don't overwrite existing video
        print(
            f"GIF already exists for {env_name} so skipping (Use `--ignore-existing` "
            "to overwrite existing files."
        )
        return

    # obtain and save LENGTH frames worth of steps
    frames: List[Image] = []
    while True:
        env.reset()
        done = False
        while not done and len(frames) <= LENGTH:
            frame = env.render()  # type: ignore
            repeat = (
                int(60 / env.metadata["render_fps"]) if env_type == "classic" else 1
            )
            for i in range(repeat):
                frames.append(Image.fromarray(frame))
            action = {i: env.action_spaces[i].sample() for i in env.agents}
            _, _, _, _, done, _ = env.step(action)

        if len(frames) > LENGTH:
            break

    env.close()

    if resize:
        for i, img in enumerate(frames):
            # h / w = H / w'
            # w' = Hw/h
            resized_img = img.resize((HEIGHT, int(HEIGHT * img.width / img.height)))
            frames[i] = resized_img

    frames[0].save(
        v_file_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
    )
    print(f"Saved: {env_name} to {v_file_path}")


if __name__ == "__main__":
    app()

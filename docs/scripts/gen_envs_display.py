"""Generates page with list of environments with images.

Copied and adapted from Gymnasium:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/docs/scripts/gen_envs_display.py

"""
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent

all_envs = [
    {
        "id": "classic",
        "list": [
            "multi_access_broadcast_channel",
            "multi_agent_tiger",
            "rock_paper_scissors",
        ],
    },
    {
        "id": "grid_world",
        "list": [
            "driving",
            "driving_gen",
            "level_based_foraging",
            "predator_prey",
            "pursuit_evasion",
            "two_paths",
            "u_a_v",
        ],
    },
]


def create_grid_cell(type_id, env_id, base_path):
    """Gen page cell for specific env ID."""
    video_path = Path(DOCS_DIR) / "_static" / "videos" / type_id / (env_id + ".gif")
    if video_path.exists():
        return f"""
            <a href="{base_path}{env_id}">
                <div class="env-grid__cell">
                    <div class="cell__image-container">
                        <img src="../../_static/videos/{type_id}/{env_id}.gif">
                    </div>
                    <div class="cell__title">
                        <span>{' '.join(env_id.split('_')).title()}</span>
                    </div>
                </div>
            </a>
        """
    else:
        return f"""
            <a href="{base_path}{env_id}">
                <div class="env-grid__cell">
                    <div class="cell__title">
                        <span>{' '.join(env_id.split('_')).title()}</span>
                    </div>
                </div>
            </a>
        """


def generate_page(env, limit=-1, base_path=""):
    """Gen page for env type."""
    env_type_id = env["id"]
    env_list = env["list"]
    cells = [create_grid_cell(env_type_id, env_id, base_path) for env_id in env_list]
    non_limited_page = limit == -1 or limit >= len(cells)
    cells = "\n".join(cells) if non_limited_page else "\n".join(cells[:limit])

    more_btn = (
        """
<a href="./complete_list">
    <button class="more-btn">
        See More Environments
    </button>
</a>
"""
        if not non_limited_page
        else ""
    )
    return f"""
<div class="env-grid">
    {cells}
</div>
{more_btn}
    """


if __name__ == "__main__":
    """
    python gen_envs_display [ env_type ]
    """

    type_dict_arr = []
    type_arg = ""

    if len(sys.argv) > 1:
        type_arg = sys.argv[1]

    for env in all_envs:
        if type_arg == env["id"] or type_arg == "":
            type_dict_arr.append(env)

    for type_dict in type_dict_arr:
        type_id: str = type_dict["id"]  # type: ignore
        envs_path = DOCS_DIR / "environments" / type_id
        if len(type_dict["list"]) > 20:
            page = generate_page(type_dict, limit=8)
            with open(envs_path / "list.html", "w", encoding="utf-8") as fp:
                fp.write(page)

            page = generate_page(type_dict, base_path="../")
            with open(
                envs_path / "complete_list.html",
                "w",
                encoding="utf-8",
            ) as fp:
                fp.write(page)

            with open(
                envs_path / "complete_list.md",
                "w",
                encoding="utf-8",
            ) as fp:
                env_name = " ".join(type_id.split("_")).title()
                fp.write(
                    f"# Complete List - {env_name}\n\n"
                    + "```{raw} html\n:file: complete_list.html\n```"
                )
        else:
            page = generate_page(type_dict)
            with open(
                envs_path / "list.html",
                "w",
                encoding="utf-8",
            ) as fp:
                fp.write(page)

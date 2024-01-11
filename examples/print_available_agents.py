"""Print IDs pf available agents of posggym.agents.

Example, print all available agents for PursuitEvasion-v0 environment:

    python print_available_agents.py --env_id PursuitEvasion-v0


Example, print all available agents for all environments:

    python print_available_agents.py

"""
from typing_extensions import Annotated
from typing import Optional

import typer
import posggym.agents as pga

app = typer.Typer()


@app.command()
def print_agents(env_id: Annotated[Optional[str], typer.Argument()] = None):
    if env_id is None:
        pga.pprint_registry()
    else:
        pga.pprint_registry(include_env_ids=[env_id])


if __name__ == "__main__":
    app()

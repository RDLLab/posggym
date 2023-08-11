"""Code and functions for running pairwise comparison of policies."""
import copy
from itertools import product
from pprint import pprint
from typing import Any, Dict, List, Optional

import posggym
from posggym import logger
from posggym.agents.evaluation.exp import ExpParams, run_experiments
from posggym.agents.evaluation.render import EpisodeRenderer, Renderer
from posggym.agents.policy import PolicyID
from posggym.agents.registration import get_env_agent_policies, get_env_args_id


def _renderer_fn() -> List[Renderer]:
    return [EpisodeRenderer()]


def get_pairwise_exp_params(
    env_id: str,
    env_args: Optional[Dict[str, Any]],
    env_args_id: Optional[str],
    policy_ids: Dict[str, List[str]],
    init_seed: int,
    num_seeds: int,
    num_episodes: int,
    time_limit: Optional[int] = None,
    exp_id_init: int = 0,
    render_mode: Optional[str] = None,
    record_env: bool = False,
) -> List[ExpParams]:
    """Get params for individual experiments from high level parameters.

    - Requires a list of policy_ids for each agent in the environment
    - Will create an experiment for every possible pairing of policy ids.
    """
    assert (record_env and render_mode in (None, "rgb_array")) or not record_env
    if record_env:
        render_mode = "rgb_array"
    render = not record_env and render_mode is not None

    env = posggym.make(env_id) if env_args is None else posggym.make(env_id, **env_args)
    assert len(policy_ids) == len(env.possible_agents)
    assert all(len(p) > 0 for p in policy_ids)

    exp_params_list = []
    agent_ids = list(policy_ids)
    agent_policies = [policy_ids[i] for i in agent_ids]
    for i, (exp_seed, policies) in enumerate(
        product(range(num_seeds), product(*agent_policies))
    ):
        exp_params = ExpParams(
            exp_id=exp_id_init + i,
            env_id=env_id,
            env_args=copy.deepcopy(env_args),
            env_args_id=env_args_id,
            policy_ids={agent_ids[i]: pi for i, pi in enumerate(policies)},
            seed=init_seed + exp_seed,
            num_episodes=num_episodes,
            time_limit=time_limit,
            tracker_fn=None,
            renderer_fn=_renderer_fn if render else None,
            render_mode=render_mode,
            record_env=record_env,
            record_env_freq=None,
        )
        exp_params_list.append(exp_params)

    env.close()
    return exp_params_list


def run_pairwise_experiments(
    env_id: str,
    env_args: Optional[Dict[str, Any]],
    policy_ids: Optional[Dict[str, List[PolicyID]]] = None,
    init_seed: int = 0,
    num_seeds: int = 1,
    num_episodes: int = 1000,
    time_limit: Optional[int] = None,
    exp_id_init: int = 0,
    render_mode: Optional[str] = None,
    record_env: bool = False,
    exp_log_level: int = logger.INFO + 1,
    n_procs: Optional[int] = None,
    root_save_dir: Optional[str] = None,
):
    print("\n== Running Pairwise Experiments ==")
    pprint(locals())
    exp_args = locals()

    if policy_ids is None:
        print("== Getting Policies ==")
        policy_specs = get_env_agent_policies(
            env_id, env_args, include_generic_policies=True
        )
        policy_ids = {
            i: [spec.id for spec in agent_specs]
            for i, agent_specs in policy_specs.items()
        }

    print("== Creating Experiments ==")
    exp_params_list = get_pairwise_exp_params(
        env_id=env_id,
        env_args=env_args,
        env_args_id=None if env_args is None else get_env_args_id(env_args),
        policy_ids=policy_ids,
        init_seed=init_seed,
        num_seeds=num_seeds,
        num_episodes=num_episodes,
        time_limit=time_limit,
        exp_id_init=exp_id_init,
        render_mode=render_mode,
        record_env=record_env,
    )

    exp_name = f"pairwise_initseed{init_seed}_numseeds{num_seeds}"

    print(f"== Running {len(exp_params_list)} Experiments ==")
    print(f"== Using {n_procs} CPUs ==")

    run_experiments(
        exp_name,
        exp_params_list=exp_params_list,
        exp_log_level=exp_log_level,
        n_procs=n_procs,
        exp_args=exp_args,
        root_save_dir=root_save_dir,
    )

    print("== All done ==")

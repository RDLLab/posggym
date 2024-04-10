"""Runs episodes with chosen posggym.agents policies.

The script takes an environment ID and a list of policy ids as arguments.
It then runs and (optionally) renders episodes.

Example, to run 10 episodes of the `Driving-v1` environment with two random policies
and with `human` rendering mode, run:

    python run_agents.py \
        --env_id Driving-v1 \
        --policy_ids Random-v0 Random-v0 \
        --num_episodes 10 \
        --render_mode human

"""

import argparse
from typing import Dict, List, Optional

import posggym
import posggym.agents as pga


def run_agents(
    env_id: str,
    policy_ids: List[str],
    num_episodes: int,
    seed: Optional[int] = None,
    render_mode: Optional[str] = "human",
):
    """Run agents."""
    print("\n== Running Agents ==")
    policy_specs = []
    env_args, env_args_id = None, None
    for i, policy_id in enumerate(policy_ids):
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

    if env_args:
        env = posggym.make(env_id, render_mode=render_mode, **env_args)
    else:
        env = posggym.make(env_id, render_mode=render_mode)

    assert len(policy_specs) == len(env.possible_agents)

    policies = {}
    for idx, spec in enumerate(policy_specs):
        agent_id = env.possible_agents[idx]
        policies[agent_id] = pga.make(spec.id, env.model, agent_id)

    if seed is not None:
        env.reset(seed=seed)
        for i, policy in enumerate(policies.values()):
            policy.reset(seed=seed + i)

    episode_steps = []
    episode_rewards: Dict[str, List[float]] = {i: [] for i in env.possible_agents}
    for ep_num in range(num_episodes):
        obs, _ = env.reset()
        env.render()
        for policy in policies.values():
            policy.reset()

        t = 0
        all_done = False
        rewards = {i: 0.0 for i in env.possible_agents}
        while not all_done:
            actions = {i: policies[i].step(obs[i]) for i in env.agents}
            obs, rews, _, _, all_done, _ = env.step(actions)
            env.render()

            t += 1
            for agent_id, r_i in rews.items():
                rewards[agent_id] += r_i

        episode_steps.append(t)
        for j in env.possible_agents:
            episode_rewards[j].append(rewards[j])

    env.close()
    for policy in policies.values():
        policy.close()

    print("\n== All done ==")
    mean_steps = sum(episode_steps) / len(episode_steps)
    print(f"Mean episode steps = {mean_steps:.2f}")
    mean_returns = {i: sum(r) / len(r) for i, r in episode_rewards.items()}
    print(f"Mean Episode returns {mean_returns}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env_id",
        type=str,
        required=True,
        help="ID of the environment to run.",
    )
    parser.add_argument(
        "-pids",
        "--policy_ids",
        type=str,
        required=True,
        nargs="+",
        help=(
            "List of IDs of policies to compare, one for each agent. Policy IDs should "
            "be provided in order of env.possible_agents (i.e. the first policy ID "
            "will be assigned to the 0-index policy in env.possible_agent, etc.)."
        ),
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes per experiment.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument(
        "--render_mode", type=str, default=None, help="The render mode to use."
    )
    args = parser.parse_args()
    run_agents(**vars(args))

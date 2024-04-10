"""Script for running each available agent for different environments.

Useful for visually inspecting all the different agents.

To see available arguments, run:

    python run_all_agents.py --help


Example 1. To run 10 episodes of every agent for every environment:

    python run_all_agents.py --num-episodes 10


Example 2. To run 10 episodes of every agent that is compatible with the `Driving-v1`
environment:

    python run_all_agents.py --env-id-prefix Driving-v1 --num-episodes 10


Example 3. To run 10 episodes of every agent that is compatible with the
`PursuitEvasion-v0` environment with specific argument, e.g. grid="8x8" run:

    python run_all_agents.py \
        --env_id_prefix PursuitEvasion-v0/grid=8x8 \
        --num_episodes 10

    # Note: the argument `env_id_prefix` is a prefix of the environment id, so the
    # following would also work to show all agents for the 8x8 version of
    # the PursuitEvasion-v0 environment
    python run_all_agents.py \
        --env_id_prefix PursuitEvasion-v0/grid=8 \
        --num_episodes 10

"""

import argparse
from typing import Dict, Optional, Tuple

import posggym
import posggym.agents as pga
from posggym.agents.registration import PolicySpec


def try_make_policy(
    spec: PolicySpec, render_mode: Optional[str]
) -> Tuple[Optional[posggym.Env], Optional[Dict[str, pga.Policy]]]:
    """Tries to make the policy showing if it is possible."""
    try:
        env_id = "Driving-v1" if spec.env_id is None else spec.env_id
        env_args = {} if spec.env_args is None else spec.env_args
        env = posggym.make(env_id, render_mode=render_mode, **env_args)

        policies = {}
        for i in env.possible_agents:
            if spec.valid_agent_ids is None or i in spec.valid_agent_ids:
                policies[i] = pga.make(spec, env.model, i)
            else:
                print(f"Using random policy for agent '{i}'")
                policies[i] = pga.make("Random-v0", env.model, i)

        return env, policies
    except (
        ImportError,
        posggym.error.DependencyNotInstalled,
        posggym.error.MissingArgument,
    ) as e:
        posggym.logger.warn(
            f"Not testing posggym.agents policy spec `{spec.id}` due to error: {e}"
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Error trying to make posggym.agents policy spec `{spec.id}`."
        ) from e
    return None, None


def run_policy(
    spec: PolicySpec,
    num_episodes: int,
    seed: Optional[int],
    render_mode: Optional[str] = "human",
):
    """Run a posggym.policy."""
    print(f"Running policy={spec.id}")
    env, policies = try_make_policy(spec, render_mode)

    if policies is None:
        return
    assert env is not None

    env.reset(seed=seed)
    for i, pi in enumerate(policies.values()):
        pi.reset(seed=seed if seed is None else seed + i)

    episode_rewards = {i: [] for i in env.possible_agents}
    for _ in range(num_episodes):
        obs, _ = env.reset()
        for pi in policies.values():
            pi.reset()
        env.render()

        all_done = False
        rewards = {i: 0.0 for i in env.possible_agents}
        while not all_done:
            joint_action = {}
            for i in env.agents:
                pi = policies[i]
                a = pi.step(env.state) if pi.observes_state else pi.step(obs[i])
                joint_action[i] = a

            obs, rews, _, _, all_done, _ = env.step(joint_action)
            env.render()

            for agent_id, r_i in rews.items():
                rewards[agent_id] += r_i

        for j in env.possible_agents:
            episode_rewards[j].append(rewards[j])

    env.close()
    for pi in policies.values():
        pi.close()

    mean_returns = {i: f"{sum(r) / len(r):.2f}" for i, r in episode_rewards.items()}
    print(f"  Mean Episode returns {mean_returns}")


def run_all_agents(
    env_id_prefix: Optional[str],
    num_episodes: int,
    seed: Optional[int],
    render_mode: str = "human",
):
    """Run all agents."""
    if render_mode.lower() == "none":
        render_mode = None

    for policy_spec in pga.registry.values():
        if env_id_prefix is None or policy_spec.id.startswith(env_id_prefix):
            run_policy(policy_spec, num_episodes, seed, render_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env_id_prefix",
        type=str,
        default=None,
        help=(
            "Prefix of environment ID to run agents for. If 'None' runs all agents "
            "for all environments. Otherwise only runs agents whose ID starts with "
            "this prefix."
        ),
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes per experiment.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
        help="The render mode to use. For no rendering use 'None'.",
    )
    args = parser.parse_args()
    run_all_agents(**vars(args))

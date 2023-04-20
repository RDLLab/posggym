"""Script for running each available agent in posggym.agents.

Useful for visually inspecting all the different agents.
"""
import argparse
from typing import Optional, Tuple

import posggym
import posggym.agents as pga
from posggym.agents.registration import PolicySpec


def try_make_policy(
    spec: PolicySpec, render_mode: str
) -> Tuple[Optional[posggym.Env], Optional[pga.Policy]]:
    """Tries to make the policy showing if it is possible."""
    try:
        env_id = "Driving-v0" if spec.env_id is None else spec.env_id
        env_args = {} if spec.env_args is None else spec.env_args
        env = posggym.make(env_id, render_mode=render_mode, **env_args)

        if spec.valid_agent_ids:
            agent_id = spec.valid_agent_ids[0]
        else:
            agent_id = env.possible_agents[0]

        return env, pga.make(spec, env.model, agent_id)
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
    spec: PolicySpec, num_episodes: int, seed: Optional[int], render_mode: str = "human"
):
    """Run a posggym.policy."""
    env, test_policy = try_make_policy(spec, render_mode)

    if test_policy is None:
        return
    assert env is not None

    print(f"Running policy={spec.id}")

    env.reset(seed=seed)
    test_policy.reset(seed=seed)

    for ep_num in range(num_episodes):
        obs, _ = env.reset()
        test_policy.reset()
        env.render()

        all_done = False
        while not all_done:
            joint_action = {}
            for i in env.agents:
                if i == test_policy.agent_id and test_policy.observes_state:
                    a = test_policy.step(env.state)
                elif i == test_policy.agent_id:
                    a = test_policy.step(obs[i])
                else:
                    a = env.action_spaces[i].sample()
                joint_action[i] = a

            obs, _, _, _, all_done, _ = env.step(joint_action)
            env.render()

    env.close()
    test_policy.close()


def run_all_agents(
    env_id_prefix: Optional[str],
    num_episodes: int,
    seed: Optional[int],
    render_mode: str = "human",
):
    """Run all agents."""
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
        help="Prefix of environment ID to run agents for. If None runs all agents.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes per experiment.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument(
        "--render_mode", type=str, default="human", help="The render mode to use."
    )
    args = parser.parse_args()
    run_all_agents(**vars(args))

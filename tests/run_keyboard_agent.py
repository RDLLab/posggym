"""Run a keyboard agent on an environment."""
from typing import Optional, List
from argparse import ArgumentParser

from gym import spaces

import posggym
import posggym.model as M


def _random_action_selection(action_space):
    return action_space.sample()


def _keyboard_action_selection(agent_id, action_space):
    while True:
        try:
            a = input(
                f"Select action for agent {agent_id} (0, {action_space.n-1}): "
            )
            return int(a)
        except ValueError:
            print("Invalid selection. Try again.")


def _get_action(env, keyboard_agent_ids):
    action_list = []
    for agent_id in range(env.n_agents):
        if agent_id in keyboard_agent_ids:
            agent_a = _keyboard_action_selection(
                agent_id, env.action_spaces[agent_id]
            )
        else:
            agent_a = _random_action_selection(env.action_spaces[agent_id])
        action_list.append(agent_a)
    return tuple(action_list)


def _get_outcome_counts(episode_outcomes, n_agents):
    outcome_counts = {k: [0 for _ in range(n_agents)] for k in M.Outcome}
    for outcome in episode_outcomes:
        if outcome is None:
            outcome = tuple(M.Outcome.NA for _ in range(n_agents))
        for i in range(n_agents):
            outcome_counts[outcome[i]][i] += 1
    return outcome_counts


def main(env_name: str,
         keyboard_agent_ids: List[int],
         num_episodes: int,
         episode_step_limit: int,
         seed: Optional[int],
         render: bool,
         render_mode: str,
         pause_each_step: bool):
    """Run keyboard agents."""
    env = posggym.make(env_name)
    action_spaces = env.action_spaces

    for agent_id in keyboard_agent_ids:
        assert isinstance(action_spaces[agent_id], spaces.Discrete)

    # set random seeds
    if seed is not None:
        env.reset(seed=seed)
        for i in range(len(action_spaces)):
            action_spaces[i].seed(seed+1+i)

    dones = 0
    episode_steps = []
    episode_rewards = [[] for _ in range(env.n_agents)]    # type: ignore
    episode_outcomes = []   # type: ignore
    for i in range(num_episodes):

        env.reset()

        if render:
            env.render(render_mode)

        if pause_each_step:
            input("Press any key")

        t = 0
        done = False
        rewards = [0.0] * env.n_agents
        while episode_step_limit is None or t < episode_step_limit:
            a = _get_action(env, keyboard_agent_ids)
            _, r, done, info = env.step(a)
            print(f"{r=}")
            print(f"{done=}")
            t += 1

            for j in range(env.n_agents):
                rewards[j] += r[j]

            if render:
                env.render(render_mode)

            if pause_each_step:
                input("Press any key")

            if done:
                print(f"End episode {i}")
                break

        dones += int(done)
        episode_steps.append(t)
        episode_outcomes.append(info.get("outcome", None))

        for j in range(env.n_agents):
            episode_rewards[j].append(rewards[j])

    env.close()

    print("All episodes finished")
    print(f"Episodes ending with 'done=True' = {dones} out of {num_episodes}")
    mean_steps = sum(episode_steps) / len(episode_steps)
    step_limit = env.spec.max_episode_steps      # type: ignore
    if episode_step_limit is not None:
        if step_limit is None:
            step_limit = episode_step_limit
        else:
            step_limit = min(step_limit, episode_step_limit)
    print(f"Mean episode steps = {mean_steps:.2f} out of max {step_limit}")
    mean_returns = [sum(r) / len(r) for r in episode_rewards]
    print(f"Mean Episode returns {mean_returns}")

    outcome_counts = _get_outcome_counts(episode_outcomes, env.n_agents)
    print("Outcomes")
    for k, v in outcome_counts.items():
        print(f"{k} = {v}")


if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')
    parser.add_argument(
        "env_name", type=str,
        help="Name of environment to run"
    )
    parser.add_argument(
        "keyboard_agent_ids", type=int, nargs="+",
        help="IDs of agents to run as keyboard agents."
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1,
        help="The number of episodes to run (default=1)"
    )
    parser.add_argument(
        "--episode_step_limit", type=int, default=None,
        help="Max number of steps to run each epsiode for (default=None)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random Seed (default=None)"
    )
    parser.add_argument(
        "--render", action='store_true',
        help="Render environment steps"
    )
    parser.add_argument(
        "--render_mode", type=str, default='human',
        help="Mode to use for renderering, if rendering (default='human')"
    )
    parser.add_argument(
        "--pause_each_step", action='store_true',
        help="Pause execution after each step"
    )
    args = parser.parse_args()
    main(**vars(args))

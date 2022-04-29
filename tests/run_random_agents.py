"""Run a random agent on an environment."""
from argparse import ArgumentParser
from typing import Optional

import posggym


def main(env_name: str,
         num_episodes: int,
         episode_step_limit: int,
         seed: Optional[int],
         render: bool,
         render_mode: str,
         pause_each_step: bool):
    """Run random agents."""
    env = posggym.make(env_name)
    action_spaces = env.action_spaces
    # set random seeds
    if seed is not None:
        env.reset(seed=seed)
        for i in range(len(action_spaces)):
            action_spaces[i].seed(seed+1+i)

    for i in range(num_episodes):

        env.reset()

        if render:
            env.render(render_mode)

        if pause_each_step:
            input("Press any key")

        t = 0
        while episode_step_limit is None or t < episode_step_limit:
            a = tuple(a.sample() for a in action_spaces)
            _, _, done, _ = env.step(a)

            if render:
                env.render(render_mode)

            if pause_each_step:
                input("Press any key")

            if done:
                print(f"End episode {i}")
                break
            t += 1

    env.close()


if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')
    parser.add_argument(
        "env_name", type=str,
        help="Name of environment to run"
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

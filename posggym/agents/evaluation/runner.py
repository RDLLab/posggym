"""Functions and classes for running episodes."""
import logging
import time
from typing import Any, Dict, Iterable, List, NamedTuple, Optional

import posggym
import posggym.agents.evaluation.render as render_lib
import posggym.agents.evaluation.stats as stats_lib
import posggym.agents.evaluation.writer as writer_lib
import posggym.model as M
from posggym.agents.policy import Policy


LINE_BREAK = "-" * 60
MAJOR_LINE_BREAK = "=" * 60


class EpisodeLoopStep(NamedTuple):
    """Output for a single episode step."""

    env: posggym.Env
    timestep: M.JointTimestep
    actions: Dict[str, Any]
    policies: Dict[str, Policy]
    done: bool


def run_episode_loop(
    env: posggym.Env,
    policies: Dict[str, Policy],
) -> Iterable[EpisodeLoopStep]:
    """Run policies in environment."""
    assert len(policies) == len(
        env.agents
    ), f"{len(policies)} policies supplied for env with {len(env.agents)} agents."

    observations, infos = env.reset()
    joint_timestep = M.JointTimestep(
        state=env.state,
        observations=observations,
        rewards={i: 0.0 for i in env.agents},
        terminations={i: False for i in env.agents},
        truncations={i: False for i in env.agents},
        all_done=False,
        infos=infos,
    )

    init_action = {i: None for i in env.agents}
    yield EpisodeLoopStep(env, joint_timestep, init_action, policies, False)

    all_done = False
    steps = 0
    while not all_done:
        actions = {}
        for i in env.agents:
            actions[i] = policies[i].step(observations[i])

        observations, rewards, terminations, truncations, all_done, infos = env.step(
            actions
        )
        joint_timestep = M.JointTimestep(
            state=env.state,
            observations=observations,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            all_done=all_done,
            infos=infos,
        )
        steps += 1

        yield EpisodeLoopStep(env, joint_timestep, actions, policies, all_done)


def run_episode(
    env: posggym.Env,
    policies: Dict[str, Policy],
    num_episodes: int,
    trackers: List[stats_lib.Tracker],
    renderers: List[render_lib.Renderer],
    time_limit: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    writer: Optional[writer_lib.Writer] = None,
) -> stats_lib.AgentStatisticsMap:
    """Run Episode simulations for given env and policies."""
    logger = logging.getLogger() if logger is None else logger
    writer = writer_lib.NullWriter() if writer is None else writer

    logger.info(
        "%s\nRunning %d episodes with Time Limit = %s s\n%s",
        MAJOR_LINE_BREAK,
        num_episodes,
        str(time_limit),
        MAJOR_LINE_BREAK,
    )

    episode_num = 0
    progress_display_freq = max(1, num_episodes // 10)
    time_limit_reached = False
    run_start_time = time.time()

    for tracker in trackers:
        tracker.reset()

    while episode_num < num_episodes and not time_limit_reached:
        logger.log(
            logging.INFO - 1,
            "%s\nEpisode %d Start\n%s",
            MAJOR_LINE_BREAK,
            episode_num,
            MAJOR_LINE_BREAK,
        )

        for tracker in trackers:
            tracker.reset_episode()

        for policy in policies.values():
            policy.reset()

        timestep_sequence = run_episode_loop(env, policies)
        for t, loop_step in enumerate(timestep_sequence):
            for tracker in trackers:
                tracker.step(t, *loop_step)
            render_lib.generate_renders(renderers, t, *loop_step)

        if len(trackers):
            episode_statistics = stats_lib.generate_episode_statistics(trackers)
            writer.write_episode(episode_statistics)
            logger.log(
                logging.INFO - 1,
                "%s\nEpisode %d Complete\n%s",
                LINE_BREAK,
                episode_num,
                writer_lib.format_as_table(episode_statistics),
            )
        else:
            logger.log(logging.INFO - 1, "\nEpisode %d Complete\n", episode_num)

        if (episode_num + 1) % progress_display_freq == 0:
            logger.info("Episode %d / %d complete", episode_num + 1, num_episodes)

        episode_num += 1

        if time_limit is not None and time.time() - run_start_time > time_limit:
            time_limit_reached = True
            logger.info(
                "%s\nTime limit of %d s reached after %d episodes",
                MAJOR_LINE_BREAK,
                time_limit,
                episode_num,
            )

    if len(trackers):
        statistics = stats_lib.generate_statistics(trackers)
        logger.info(
            "%s\nSimulations Complete\n%s\n%s",
            MAJOR_LINE_BREAK,
            writer_lib.format_as_table(statistics),
            MAJOR_LINE_BREAK,
        )
    else:
        statistics = {}
        logger.info(
            "%s\nSimulations Complete\n%s",
            MAJOR_LINE_BREAK,
            MAJOR_LINE_BREAK,
        )

    return statistics

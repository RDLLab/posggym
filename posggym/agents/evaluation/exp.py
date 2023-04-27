"""Functions and data structures for running experiments."""
import argparse
import json
import logging
import multiprocessing as mp
import os
import pathlib
import tempfile
import time
from datetime import datetime
from pprint import pformat
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import posggym
import posggym.agents.evaluation.render as render_lib
import posggym.agents.evaluation.stats as stats_lib
import posggym.agents.evaluation.writer as writer_lib
from posggym import wrappers
from posggym.agents import make
from posggym.agents.evaluation import runner
from posggym.config import BASE_RESULTS_DIR


LINE_BREAK = "-" * 60
EXP_ARG_FILE_NAME = "exp_args.json"


# A global lock used for controlling when processes print to stdout
# This helps keep top level stdout readable
LOCK = mp.Lock()


def _init_lock(lck):
    # pylint: disable=[global-statement]
    global LOCK
    LOCK = lck


class ExpParams(NamedTuple):
    """Params for a single experiment run."""

    exp_id: int
    env_id: str
    env_args: Optional[Dict[str, Any]]
    env_args_id: Optional[str]
    policy_ids: Dict[str, str]
    seed: int
    num_episodes: int
    time_limit: Optional[int] = None
    tracker_fn: Optional[Callable[[], List[stats_lib.Tracker]]] = None
    renderer_fn: Optional[Callable[[], List[render_lib.Renderer]]] = None
    render_mode: Optional[str] = None
    stream_log_level: int = logging.INFO
    file_log_level: int = logging.DEBUG
    record_env: bool = False
    # If None then uses the default cubic frequency
    record_env_freq: Optional[int] = None


def get_exp_parser() -> argparse.ArgumentParser:
    """Get command line argument parser with default experiment args."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Number of episodes per experiment.",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=None,
        help="Experiment time limit, in seconds.",
    )
    parser.add_argument(
        "--n_procs",
        type=int,
        default=1,
        help="Number of processors/experiments to run in parallel.",
    )
    parser.add_argument(
        "--log_level", type=int, default=21, help="Experiment log level."
    )
    parser.add_argument(
        "--root_save_dir",
        type=str,
        default=None,
        help=(
            "Optional directory to save results in. If supplied then it must "
            "be an existing directory. If None the default "
            "~/posggym_agents_results/<env_id>/results/ dir is used root "
            "results dir."
        ),
    )
    return parser


def make_exp_result_dir(
    exp_name: str,
    env_id: str,
    env_args_id: Optional[str],
    root_save_dir: Optional[str] = None,
) -> str:
    """Make a directory for experiment results."""
    time_str = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    if root_save_dir is None and env_args_id is not None:
        root_save_dir = os.path.join(BASE_RESULTS_DIR, env_id, env_args_id)
    elif root_save_dir is None:
        root_save_dir = os.path.join(BASE_RESULTS_DIR, env_id)
    pathlib.Path(root_save_dir).mkdir(parents=True, exist_ok=True)
    result_dir = tempfile.mkdtemp(prefix=f"{exp_name}_{time_str}", dir=root_save_dir)
    return result_dir


def _log_exp_start(params: ExpParams, result_dir: str, logger: logging.Logger):
    LOCK.acquire()
    try:
        logger.info(LINE_BREAK)
        logger.info("Running with:")
        logger.info(pformat(params))
        logger.info(f"Result dir = {result_dir}")
        logger.info(LINE_BREAK)
    finally:
        LOCK.release()


def _log_exp_end(
    params: ExpParams, result_dir: str, logger: logging.Logger, exp_time: float
):
    LOCK.acquire()
    try:
        logger.info(LINE_BREAK)
        logger.info(f"Finished exp num {params.exp_id}")
        logger.info(f"Result dir = {result_dir}")
        logger.info(f"Experiment Run time {exp_time:.2f} seconds")
        logger.info(LINE_BREAK)
    finally:
        LOCK.release()


def get_exp_run_logger(
    exp_id: int,
    result_dir: str,
    stream_log_level: int = logging.INFO,
    file_log_level: int = logging.DEBUG,
) -> logging.Logger:
    """Get the logger for a single experiment run."""
    logger_name = f"exp_{exp_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(min(stream_log_level, file_log_level))

    fname = f"exp_{exp_id}.log"
    log_file = os.path.join(result_dir, fname)
    file_formatter = logging.Formatter(
        # [Day-Month Hour-Minute-Second] exp_x Message
        "[%(asctime)s] %(levelname)s %(message)s",
        "%d-%m %H:%M:%S",
    )

    filehandler = logging.FileHandler(log_file)
    filehandler.setFormatter(file_formatter)
    filehandler.setLevel(file_log_level)

    stream_formatter = logging.Formatter(
        # [Day-Month Hour-Minute-Second] exp_x Message
        "[%(asctime)s] %(name)s %(message)s",
        "%d-%m %H:%M:%S",
    )
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(stream_formatter)
    streamhandler.setLevel(stream_log_level)

    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.propagate = False

    return logger


def _get_exp_statistics(params: ExpParams) -> stats_lib.AgentStatisticsMap:
    stats = {}
    for i in params.policy_ids:
        stats[i] = {
            "exp_id": params.exp_id,
            "agent_id": i,
            "env_id": params.env_id,
            "env_args_id": params.env_args_id,
            "policy_id": params.policy_ids[i],
            "exp_seed": params.seed,
            "num_episodes": params.num_episodes,
            "time_limit": params.time_limit if params.time_limit else "None",
        }
    return stats


def _get_linear_episode_trigger(freq: int) -> Callable[[int], bool]:
    return lambda t: t % freq == 0


def run_single_experiment(args: Tuple[ExpParams, str]):
    """Run a single experiment and write results to a file."""
    params, result_dir = args
    exp_start_time = time.time()

    exp_logger = get_exp_run_logger(
        params.exp_id, result_dir, params.stream_log_level, params.file_log_level
    )
    _log_exp_start(params, result_dir, exp_logger)

    if params.env_args is not None:
        env = posggym.make(
            params.env_id, render_mode=params.render_mode, **params.env_args
        )
    else:
        env = posggym.make(params.env_id, render_mode=params.render_mode)
    assert len(params.policy_ids) == len(env.possible_agents), (
        f"Experiment env '{env}' has {len(env.possible_agents)} possible agents, but "
        f"only {len(params.policy_ids)} policies supplied."
    )

    if params.record_env:
        video_folder = os.path.join(result_dir, f"exp_{params.exp_id}_video")
        episode_trigger = None
        if params.record_env_freq:
            episode_trigger = _get_linear_episode_trigger(params.record_env_freq)
        env = wrappers.RecordVideo(env, video_folder, episode_trigger=episode_trigger)

    policies = {}
    for i, policy_id in params.policy_ids.items():
        policies[i] = make(policy_id, env.model, i)

    if params.tracker_fn:
        trackers = params.tracker_fn()
    else:
        trackers = stats_lib.get_default_trackers()

    renderers = params.renderer_fn() if params.renderer_fn else []
    writer = writer_lib.ExperimentWriter(
        params.exp_id, result_dir, _get_exp_statistics(params)
    )

    if params.seed is not None:
        env.reset(seed=params.seed)
        for idx, policy in enumerate(policies.values()):
            policy.reset(seed=params.seed + idx)

    try:
        statistics = runner.run_episode(
            env,
            policies,
            params.num_episodes,
            trackers,
            renderers,
            time_limit=params.time_limit,
            logger=exp_logger,
            writer=writer,
        )
        writer.write(statistics)

    except Exception as ex:
        exp_logger.exception("Exception occurred: %s", str(ex))
        exp_logger.error(pformat(locals()))
        raise ex
    finally:
        _log_exp_end(params, result_dir, exp_logger, time.time() - exp_start_time)
        env.close()
        for policy in policies.values():
            policy.close()
        for h in exp_logger.handlers:
            h.close()


def run_experiments(
    exp_name: str,
    exp_params_list: List[ExpParams],
    exp_log_level: int = logging.INFO + 1,
    n_procs: Optional[int] = None,
    exp_args: Optional[Dict] = None,
    root_save_dir: Optional[str] = None,
) -> str:
    """Run series of experiments.

    If exp_args is not None then will write to file in the result dir.
    """
    exp_start_time = time.time()
    logging.basicConfig(
        level=exp_log_level,
        # [Day-Month Hour-Minute-Second] Message
        format="[%(asctime)s] %(message)s",
        datefmt="%d-%m %H:%M:%S",
    )

    num_exps = len(exp_params_list)
    logging.log(exp_log_level, "Running %d experiments", num_exps)

    result_dir = make_exp_result_dir(
        exp_name,
        exp_params_list[0].env_id,
        exp_params_list[0].env_args_id,
        root_save_dir,
    )
    logging.log(exp_log_level, "Saving results to dir=%s", result_dir)

    if exp_args:
        write_experiment_arguments(exp_args, result_dir)

    if n_procs is None:
        n_procs = os.cpu_count()
    logging.log(exp_log_level, "Running %d processes", n_procs)

    mp_lock = mp.Lock()

    def _initializer(init_args):
        proc_lock = init_args
        _init_lock(proc_lock)

    if n_procs == 1:
        _initializer(mp_lock)
        for params in exp_params_list:
            run_single_experiment((params, result_dir))
    else:
        args_list = [(params, result_dir) for params in exp_params_list]
        with mp.Pool(n_procs, initializer=_initializer, initargs=(mp_lock,)) as p:
            p.map(run_single_experiment, args_list, 1)

    logging.log(exp_log_level, "Compiling results")
    writer_lib.compile_results(result_dir)

    logging.log(
        exp_log_level, "Experiment Run time %.2f seconds", time.time() - exp_start_time
    )

    return result_dir


def write_experiment_arguments(args: Dict[str, Any], result_dir: str) -> str:
    """Write experiment arguments to file."""
    arg_file = os.path.join(result_dir, EXP_ARG_FILE_NAME)
    with open(arg_file, "w", encoding="utf-8") as fout:
        json.dump(args, fout)
    return arg_file

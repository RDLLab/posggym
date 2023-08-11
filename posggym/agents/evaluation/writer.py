"""Functions and classes for writing results to file."""
import abc
import csv
import os
import pathlib
from datetime import datetime
from typing import Any, List, Optional, Sequence

import pandas as pd
from prettytable import PrettyTable

from posggym.agents.evaluation.stats import AgentStatisticsMap, combine_statistics
from posggym.config import BASE_RESULTS_DIR


COMPILED_RESULTS_FNAME = "compiled_results.csv"


def make_dir(exp_name: str) -> str:
    """Make a new experiment results directory at."""
    result_dir = os.path.join(BASE_RESULTS_DIR, f"{exp_name}_{datetime.now()}")
    pathlib.Path(result_dir).mkdir(exist_ok=True)
    return result_dir


def format_as_table(values: AgentStatisticsMap) -> str:
    """Format values as a table."""
    table = PrettyTable()

    agent_ids = list(values)
    table.field_names = ["AgentID"] + [str(i) for i in agent_ids]

    for row_name in list(values[agent_ids[0]].keys()):
        row = [row_name]
        for i in agent_ids:
            agent_row_value = values[i][row_name]
            if isinstance(agent_row_value, float):
                row.append(f"{agent_row_value:.4f}")
            else:
                row.append(str(agent_row_value))
        table.add_row(row)

    table.align = "r"
    table.align["AgentID"] = "l"  # type: ignore
    return table.get_string()


def compile_result_files(
    save_dir: str, result_filepaths: List[str], extra_output_dir: Optional[str] = None
) -> str:
    """Compile list of results files into a single file."""
    concat_resultspath = os.path.join(save_dir, COMPILED_RESULTS_FNAME)

    dfs = list(map(pd.read_csv, result_filepaths))

    def do_concat_df(df0, df1):
        exp_ids0 = df0["exp_id"].unique().tolist()
        exp_ids1 = df1["exp_id"].unique().tolist()
        if len(set(exp_ids0).intersection(exp_ids1)) > 0:
            df1["exp_id"] += max(exp_ids0) + 1
        return pd.concat([df0, df1], ignore_index=True)

    concat_df = dfs[0]
    for df_i in dfs[1:]:
        concat_df = do_concat_df(concat_df, df_i)

    concat_df.to_csv(concat_resultspath, index=False)

    if extra_output_dir:
        extra_results_filepath = os.path.join(extra_output_dir, COMPILED_RESULTS_FNAME)
        concat_df.to_csv(extra_results_filepath, index=False)

    return concat_resultspath


def compile_results(result_dir: str, extra_output_dir: Optional[str] = None) -> str:
    """Compile all .csv results files in a directory into a single file.

    If extra_output_dir is provided then will additionally compile_result to
    the extra_output_dir.

    If handle_duplicate_exp_ids is True, then function will assign new unique
    exp_ids to entries that have duplicate exp_ids.
    """
    result_filepaths = [
        os.path.join(result_dir, f)
        for f in os.listdir(result_dir)
        if (
            os.path.isfile(os.path.join(result_dir, f))
            and ExperimentWriter.is_results_file(f)
            and not f.startswith(COMPILED_RESULTS_FNAME)
        )
    ]

    concat_resultspath = compile_result_files(
        result_dir, result_filepaths, extra_output_dir
    )
    return concat_resultspath


class Writer(abc.ABC):
    """Abstract logging object for writing results to some destination.

    Each 'write()' and 'write_episode()' takes an 'OrderedDict'
    """

    @abc.abstractmethod
    def write(self, statistics: AgentStatisticsMap):
        """Write statistics to destination.."""

    @abc.abstractmethod
    def write_episode(self, statistics: AgentStatisticsMap):
        """Write episode statistics to destination.."""

    @abc.abstractmethod
    def close(self):
        """Close the Writer."""


class NullWriter(Writer):
    """Placeholder Writer class that does nothing."""

    def write(self, statistics: AgentStatisticsMap):
        return

    def write_episode(self, statistics: AgentStatisticsMap):
        return

    def close(self):
        return


class CSVWriter(Writer):
    """A logging object to write to CSV files.

    Each 'write()' takes an 'OrderedDict', creating one column in the CSV file
    for each dictionary key on the first call. Subsequent calls to 'write()'
    must contain the same dictionary keys.

    Inspired by:
    https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/parts.py

    Does not support the 'write_episode()' function. Or rather it does nothing.
    """

    DEFAULT_RESULTS_FILENAME = "results.csv"

    def __init__(self, filepath: Optional[str] = None, dirpath: Optional[str] = None):
        if filepath is not None and dirpath is None:
            dirpath = os.path.dirname(filepath)
        elif filepath is None and dirpath is not None:
            filepath = os.path.join(dirpath, self.DEFAULT_RESULTS_FILENAME)
        else:
            raise AssertionError("Expects filepath or dirpath, not both")

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        self._filepath = filepath
        self._header_written = False
        self._fieldnames: Sequence[Any] = []

    def write(self, statistics: AgentStatisticsMap):
        """Append given statistics as new rows to CSV file.

        1 row per agent entry in the AgentStatisticsMap.
        Assumes all agent's statistics maps share the same headers
        """
        agent_ids = list(statistics)
        if self._fieldnames == []:
            self._fieldnames = list(statistics[agent_ids[0]].keys())

        # Open a file in 'append' mode, so we can continue logging safely to
        # the same file if needed.
        with open(self._filepath, "a") as fout:
            # Always use same fieldnames to create writer, this way a
            # consistency check is performed automatically on each write.
            writer = csv.DictWriter(fout, fieldnames=self._fieldnames)
            # Write a header if this is the very first write.
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            for i in agent_ids:
                writer.writerow(statistics[i])

    def write_episode(self, statistics: AgentStatisticsMap):
        return

    def close(self):
        return


class ExperimentWriter(Writer):
    """A logging object for writing results during experiments.

    This logger handles storing of results after each episode of an experiment
    as well as the final summarized results.

    The results are stored in two separate files:
    - "exp_<exp_id>_episodes.csv": stores results for each episode
    - "exp_<exp_id>.csv": stores summary results for experiment

    Includes an additional function "checkpoint" for checkpointing results
    during experiments. This function takes a list of Tracker objects as input
    and writes a summary of the results so far to the summary results file.
    This function is useful for experiments that may take a long time to run or
    could be interrupted early.

    """

    def __init__(self, exp_id: int, dirpath: str, exp_params: AgentStatisticsMap):
        self._episode_filepath = os.path.join(dirpath, f"exp_{exp_id}_episodes.csv")
        self._filepath = os.path.join(dirpath, f"exp_{exp_id}.csv")

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        self._exp_params = exp_params

        self._episode_header_written = False
        self._episode_fieldnames: Sequence[Any] = []
        self._header_written = False
        self._fieldnames: Sequence[Any] = []

    def write_episode(self, statistics: AgentStatisticsMap):
        """Append given statistics as new rows to episode results CSV file.

        1 row per agent entry in the AgentStatisticsMap.
        Assumes all agent's statistics maps share the same headers

        Will handle adding experiment parameters to result rows.

        """
        agent_ids = list(statistics)
        statistics = combine_statistics([statistics, self._exp_params])

        if self._episode_fieldnames == []:
            self._episode_fieldnames = list(statistics[agent_ids[0]].keys())

        # Open in 'append' mode to add to results file
        with open(self._episode_filepath, "a") as fout:
            writer = csv.DictWriter(fout, fieldnames=self._episode_fieldnames)
            if not self._episode_header_written:
                writer.writeheader()
                self._episode_header_written = True
            for i in agent_ids:
                writer.writerow(statistics[i])

    def write(self, statistics: AgentStatisticsMap):
        """Write results summary to results summary CSV file."""
        agent_ids = list(statistics)
        statistics = combine_statistics([statistics, self._exp_params])

        if self._fieldnames == []:
            self._fieldnames = list(statistics[agent_ids[0]].keys())

        # Open in 'write' mode to overwrite any previous summary results
        with open(self._filepath, "w") as fout:
            writer = csv.DictWriter(fout, fieldnames=self._fieldnames)
            writer.writeheader()
            for i in agent_ids:
                writer.writerow(statistics[i])

    def close(self):
        """Close the `ExperimentWriter`."""

    @staticmethod
    def is_results_file(filename: str) -> bool:
        """Check if filename is for an experiment summary results file."""
        if not filename.endswith(".csv"):
            return False
        filename = filename.replace(".csv", "")
        tokens = filename.split("_")
        if len(tokens) != 2 or tokens[0] != "exp":
            return False
        try:
            int(tokens[1])
            return True
        except ValueError:
            return False

    @staticmethod
    def is_episodes_results_file(filename: str) -> bool:
        """Check if filename is for an experiment episode results file."""
        if not filename.endswith(".csv"):
            return False
        filename = filename.replace(".csv", "")
        tokens = filename.split("_")
        if len(tokens) != 3 or tokens[0] != "exp" or tokens[2] != "episodes":
            return False
        try:
            int(tokens[1])
            return True
        except ValueError:
            return False

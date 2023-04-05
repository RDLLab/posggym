"""Sub-package containing code for running evaluation experiments on policies."""
from posggym.agents.evaluation.exp import ExpParams, get_exp_parser, run_experiments
from posggym.agents.evaluation.pairwise import run_pairwise_experiments
from posggym.agents.evaluation.render import EpisodeRenderer, Renderer
from posggym.agents.evaluation.runner import run_episode
from posggym.agents.evaluation.stats import get_default_trackers

"""Tests for posggym.agent.torch_policy module."""
import pickle

import posggym
import posggym.agents as pga
import posggym.agents.continuous.driving_continuous as driving_continuous_agents
import posggym.agents.grid_world.driving as driving_agents
from posggym.agents.torch_policy import (
    PPOLSTMModel,
    PPOPolicy,
    PPOTorchModelSaveFileFormat,
)
from posggym.agents.utils import download, processors


# Discrete test policy
TEST_DISCRETE_ENV_ID = "Driving-v1"
TEST_DISCRETE_ENV_ARGS_ID = "grid=14x14RoundAbout-num_agents=2"
TEST_DISCRETE_ENV_ARGS = {
    "grid": "14x14RoundAbout",
    "num_agents": 2,
    "obs_dim": (3, 1, 1),
}
TEST_DISCRETE_POLICY_FILE_NAME = "RL1.pkl"
TEST_DISCRETE_POLICY_NAME = TEST_DISCRETE_POLICY_FILE_NAME.split(".")[0]
TEST_DISCRETE_POLICY_VERSION = 0
TEST_DISCRETE_POLICY_FILE = (
    driving_agents.agent_model_dir
    / TEST_DISCRETE_ENV_ARGS_ID
    / TEST_DISCRETE_POLICY_FILE_NAME
)

# Continuous test policy
TEST_CONTINUOUS_ENV_ID = driving_continuous_agents.ENV_ID
TEST_CONTINUOUS_ENV_ARGS_ID = (
    "world=14x14RoundAbout-num_agents=2-obs_dist=5-n_sensors=16"
)
TEST_CONTINUOUS_ENV_ARGS = {
    "world": "14x14RoundAbout",
    "num_agents": 2,
    "obs_dist": 5,
    "n_sensors": 16,
}
TEST_CONTINUOUS_POLICY_FILE_NAME = "sp_seed0.pkl"
TEST_CONTINUOUS_POLICY_NAME = TEST_CONTINUOUS_POLICY_FILE_NAME.split(".")[0]
TEST_CONTINUOUS_POLICY_VERSION = 0
TEST_CONTINUOUS_POLICY_FILE = (
    driving_continuous_agents.agent_model_dir
    / TEST_CONTINUOUS_ENV_ARGS_ID
    / TEST_CONTINUOUS_POLICY_FILE_NAME
)


def test_load_discrete_ppo_policy_model():
    if not TEST_DISCRETE_POLICY_FILE.exists():
        # ensure file is already downloaded
        download.download_from_repo(TEST_DISCRETE_POLICY_FILE)

    env = posggym.make(TEST_DISCRETE_ENV_ID, **TEST_DISCRETE_ENV_ARGS)
    agent_id = env.possible_agents[0]
    action_space = env.action_spaces[agent_id]
    obs_space = env.observation_spaces[agent_id]

    obs_processor = processors.FlattenProcessor(obs_space)

    with open(TEST_DISCRETE_POLICY_FILE, "rb") as f:
        data = PPOTorchModelSaveFileFormat(**pickle.load(f))

    model = PPOLSTMModel(
        obs_space=obs_processor.get_processed_space(),
        action_space=action_space,
        trunk_sizes=data.trunk_sizes,
        lstm_size=data.lstm_size,
        lstm_layers=data.lstm_layers,
        head_sizes=data.head_sizes,
        activation=data.activation,
        lstm_use_prev_action=data.lstm_use_prev_action,
        lstm_use_prev_reward=data.lstm_use_prev_reward,
    )
    model.load_state_dict(data.weights)

    init_state = model.get_initial_state()
    obs, _ = env.reset(seed=42)
    action, lstm_state, value, pi = model.get_action_and_value(
        obs_processor(obs[agent_id]), init_state, None, None, False
    )


def test_load_continuous_ppo_policy_model():
    if not TEST_CONTINUOUS_POLICY_FILE.exists():
        # ensure file is already downloaded
        download.download_from_repo(TEST_CONTINUOUS_POLICY_FILE)

    env = posggym.make(TEST_CONTINUOUS_ENV_ID, **TEST_CONTINUOUS_ENV_ARGS)
    agent_id = env.possible_agents[0]
    action_space = env.action_spaces[agent_id]
    obs_space = env.observation_spaces[agent_id]

    obs_processor = processors.RescaleProcessor(obs_space, -1.0, 1.0)

    with open(TEST_CONTINUOUS_POLICY_FILE, "rb") as f:
        data = PPOTorchModelSaveFileFormat(**pickle.load(f))

    model = PPOLSTMModel(
        obs_space=obs_processor.get_processed_space(),
        action_space=action_space,
        trunk_sizes=data.trunk_sizes,
        lstm_size=data.lstm_size,
        lstm_layers=data.lstm_layers,
        head_sizes=data.head_sizes,
        activation=data.activation,
        lstm_use_prev_action=data.lstm_use_prev_action,
        lstm_use_prev_reward=data.lstm_use_prev_reward,
    )
    model.load_state_dict(data.weights)

    init_state = model.get_initial_state()
    obs, _ = env.reset(seed=42)
    action, lstm_state, value, pi = model.get_action_and_value(
        obs_processor(obs[agent_id]), init_state, None, None, False
    )


def test_discrete_ppo_get_spec_from_path():
    """Test can load PPO policy spec from path for discrete Policy."""
    spec = PPOPolicy.get_spec_from_path(
        env_id=TEST_DISCRETE_ENV_ID,
        env_args_id=TEST_DISCRETE_ENV_ARGS_ID,
        env_args=TEST_DISCRETE_ENV_ARGS,
        policy_file_path=TEST_DISCRETE_POLICY_FILE,
        version=TEST_DISCRETE_POLICY_VERSION,
        valid_agent_ids=None,
        nondeterministic=False,
        deterministic=False,
        obs_processor_cls=processors.FlattenProcessor,
        obs_processor_config=None,
        action_processor_cls=None,
        action_processor_config=None,
    )
    env = posggym.make(TEST_DISCRETE_ENV_ID, **TEST_DISCRETE_ENV_ARGS)
    agent_id = env.possible_agents[0]
    pga.make(spec, env.model, agent_id)


def test_continuous_ppo_get_spec_from_path():
    """Test can load PPO policy spec from path for continuous Policy."""
    spec = PPOPolicy.get_spec_from_path(
        env_id=TEST_CONTINUOUS_ENV_ID,
        env_args_id=TEST_CONTINUOUS_ENV_ARGS_ID,
        env_args=TEST_CONTINUOUS_ENV_ARGS,
        policy_file_path=TEST_CONTINUOUS_POLICY_FILE,
        version=TEST_CONTINUOUS_POLICY_VERSION,
        valid_agent_ids=None,
        nondeterministic=False,
        deterministic=False,
        obs_processor_cls=processors.RescaleProcessor,
        obs_processor_config={"min_val": -1, "max_val": 1},
        action_processor_cls=processors.RescaleProcessor,
        action_processor_config={"min_val": -1, "max_val": 1},
    )
    env = posggym.make(TEST_CONTINUOUS_ENV_ID, **TEST_CONTINUOUS_ENV_ARGS)
    agent_id = env.possible_agents[0]
    pga.make(spec, env.model, agent_id)


if __name__ == "__main__":
    test_load_discrete_ppo_policy_model()
    test_load_continuous_ppo_policy_model()
    test_discrete_ppo_get_spec_from_path()
    test_continuous_ppo_get_spec_from_path()

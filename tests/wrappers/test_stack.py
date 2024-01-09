"""Test for Stack Wrapper."""
import numpy as np
import posggym
from posggym.vector import SyncVectorEnv
from posggym.wrappers import FlattenObservations, StackEnv

from tests.wrappers.utils import assert_equals


def test_stack():
    env = posggym.make(
        "Driving-v1", disable_env_checker=True, num_agents=2, grid="7x7RoundAbout"
    )
    # need to convert to Box obs for it to work with StackEnv
    env = FlattenObservations(env)

    env2 = posggym.make(
        "Driving-v1", disable_env_checker=True, num_agents=2, grid="7x7RoundAbout"
    )
    wrapped_env = StackEnv(FlattenObservations(env2))

    assert wrapped_env.possible_agents == env.possible_agents

    num_agents = len(env.possible_agents)
    unwrapped_obs_space = env.observation_spaces[env.possible_agents[0]]
    assert unwrapped_obs_space.shape is not None

    obs, infos = env.reset(seed=55)
    w_obs, w_infos = wrapped_env.reset(seed=55)

    assert isinstance(w_obs, np.ndarray)
    assert w_obs.shape == (num_agents, *unwrapped_obs_space.shape)
    assert_equals(infos, w_infos)

    for idx, i in enumerate(env.possible_agents):
        assert np.allclose(obs[i], w_obs[idx])

    action = {agent: env.action_spaces[agent].sample() for agent in env.possible_agents}
    stacked_action = np.stack(list(action.values()), dtype=int)

    obs, rewards, terms, truncs, dones, infos = env.step(action)
    w_obs, w_rewards, w_terms, w_truncs, w_dones, w_infos = wrapped_env.step(
        stacked_action
    )

    assert isinstance(w_obs, np.ndarray)
    assert w_obs.shape == (num_agents, *unwrapped_obs_space.shape)
    assert isinstance(w_rewards, np.ndarray)
    assert w_rewards.shape == (num_agents,)
    assert isinstance(w_terms, np.ndarray)
    assert w_terms.shape == (num_agents,)
    assert isinstance(w_truncs, np.ndarray)
    assert w_truncs.shape == (num_agents,)
    assert isinstance(w_dones, np.ndarray)
    assert w_dones.shape == (1,)
    assert isinstance(w_infos, dict)
    assert len(w_infos) == num_agents

    for idx, i in enumerate(env.possible_agents):
        assert np.allclose(obs[i], w_obs[idx])
        assert np.allclose(rewards[i], w_rewards[idx])
        assert terms[i] == w_terms[idx]
        assert truncs[i] == w_truncs[idx]
        assert dones == w_dones
    assert_equals(infos, w_infos)


def test_stack_vec_env():
    num_envs = 4

    def thunk():
        env = posggym.make(
            "Driving-v1", disable_env_checker=True, num_agents=2, grid="7x7RoundAbout"
        )
        # need to convert to Box obs for it to work with StackEnv
        env = FlattenObservations(env)
        return env

    vec_env = SyncVectorEnv([thunk for _ in range(num_envs)])

    wrapped_env = StackEnv(SyncVectorEnv([thunk for _ in range(num_envs)]))
    possible_agents = wrapped_env.possible_agents
    agent_0 = possible_agents[0]
    assert wrapped_env.num_envs == num_envs
    assert wrapped_env.is_vector_env
    assert len(wrapped_env.possible_agents) == 2
    assert all(
        wrapped_env.observation_spaces[i] == vec_env.observation_spaces[i]
        for i in possible_agents
    )

    num_agents = len(possible_agents)
    single_observation_space = wrapped_env.single_observation_spaces[agent_0]

    v_obs, v_infos = vec_env.reset(seed=77)
    w_obs, w_infos = wrapped_env.reset(seed=77)
    assert isinstance(w_obs, np.ndarray)
    assert w_obs.shape == (num_agents * num_envs, *single_observation_space.shape)
    for idx, i in enumerate(possible_agents):
        assert np.allclose(v_obs[i], w_obs[idx :: len(possible_agents)])
    assert_equals(w_infos, v_infos)

    action = {i: vec_env.action_spaces[i].sample() for i in possible_agents}
    stacked_action = np.empty((num_envs * num_agents,), dtype=int)
    for idx, i in enumerate(action):
        stacked_action[idx::num_agents] = action[i]

    v_obs, v_rewards, v_terms, v_truncs, v_dones, v_infos = vec_env.step(action)
    w_obs, w_rewards, w_terms, w_truncs, w_dones, w_infos = wrapped_env.step(
        stacked_action
    )

    assert isinstance(w_obs, np.ndarray)
    assert w_obs.shape == (num_agents * num_envs, *single_observation_space.shape)
    assert isinstance(w_rewards, np.ndarray)
    assert w_rewards.shape == (num_agents * num_envs,)
    assert isinstance(w_terms, np.ndarray)
    assert w_terms.shape == (num_agents * num_envs,)
    assert isinstance(w_truncs, np.ndarray)
    assert w_truncs.shape == (num_agents * num_envs,)
    assert isinstance(w_dones, np.ndarray)
    assert w_dones.shape == (num_envs,)
    assert isinstance(w_infos, dict)
    assert len(w_infos) == num_agents

    for idx, i in enumerate(possible_agents):
        assert np.allclose(v_obs[i], w_obs[idx :: len(possible_agents)])
        assert np.allclose(v_rewards[i], w_rewards[idx :: len(possible_agents)])
        assert np.allclose(v_terms[i], w_terms[idx :: len(possible_agents)])
        assert np.allclose(v_truncs[i], w_truncs[idx :: len(possible_agents)])
        assert np.allclose(v_dones, w_dones)

    assert_equals(w_infos, v_infos)


if __name__ == "__main__":
    test_stack()
    test_stack_vec_env()

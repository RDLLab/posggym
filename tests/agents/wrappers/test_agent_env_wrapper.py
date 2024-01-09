"""Tests for the AgentEnvWrapper class."""

import posggym
import posggym.agents as pga
from posggym.agents.wrappers.agent_env import AgentEnvWrapper


def test_agent_env_wrapper():
    env_id = "MultiAccessBroadcastChannel-v0"
    env = posggym.make(env_id, disable_env_checker=True)
    controlled_agent_id = env.agents[0]

    def agent_fn(model: posggym.POSGModel):
        return {controlled_agent_id: pga.make("Random-v0", model, controlled_agent_id)}

    wrapped_env = AgentEnvWrapper(env, agent_fn)

    assert wrapped_env.controlled_agents == [controlled_agent_id]
    assert len(wrapped_env.policies) == 1
    assert controlled_agent_id in wrapped_env.policies
    assert wrapped_env.agents == [i for i in env.agents if i != controlled_agent_id]
    assert wrapped_env.possible_agents == tuple(
        i for i in env.possible_agents if i != controlled_agent_id
    )
    assert wrapped_env.action_spaces == {
        i: act_space
        for i, act_space in env.action_spaces.items()
        if i != controlled_agent_id
    }
    assert wrapped_env.observation_spaces == {
        i: obs_space
        for i, obs_space in env.observation_spaces.items()
        if i != controlled_agent_id
    }
    assert wrapped_env.reward_ranges == {
        i: reward_range
        for i, reward_range in env.reward_ranges.items()
        if i != controlled_agent_id
    }

    obs, info = wrapped_env.reset()
    assert controlled_agent_id not in obs
    assert len(obs) == len(env.agents) - 1
    assert controlled_agent_id not in info
    assert len(info) == len(env.agents) - 1

    action = {i: wrapped_env.action_spaces[i].sample() for i in wrapped_env.agents}
    obs, reward, terminated, info, done, infos = wrapped_env.step(action)
    assert controlled_agent_id not in obs
    assert len(obs) == len(env.agents) - 1
    assert controlled_agent_id not in reward
    assert len(reward) == len(env.agents) - 1
    assert controlled_agent_id not in terminated
    assert len(terminated) == len(env.agents) - 1
    assert controlled_agent_id not in infos
    assert len(infos) == len(env.agents) - 1

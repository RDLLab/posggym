"""Base class for policies."""
import abc

import posggym.model as M


class Policy(abc.ABC):
    """Interface for an agent policy that can act in a POSG environment.

    This interface defines an API for an policy to interact within an episode
    loop where each Episode is of the form:


        # reset policies and env for a new episode observation.
        policy.reset()
        obs = env.reset()

        done = False
        while not done:

            # update policy and get next action
            policy_action = policy.step(obs[policy.agent_id])

            # get actions for other agents and combine into a joint action
            actions = ...

            obs, rewards, done, info = env.step(actions)


    Note, it is also possible to perform updates and get policy action steps
    seperately:

        # reset policies and env for a new episode and get initial  obs
        policy.reset()
        obs = env.reset()

        # update policy with initial observation
        policy.update(obs(policy.agent_id])

        done = False
        while not done:

            # get next action
            policy_action = policy.get_action()

            # get actions for other agents and combine into a joint action
            actions = ...

            obs, rewards, done, info = env.step(actions)

            # update policy with latest obs
            policy.update(obs(policy.agent_id])

    """

    @abc.abstractmethod
    @property
    def agent_id(self) -> M.AgentID:
        """Get the ID of the agent this policy is for."""

    @abc.abstractmethod
    def step(self, obs: M.Observation) -> M.Action:
        """Execute a single policy step.

        This involves:
        1. a updating policy with last action and given observation
        2. next action using updated policy
        """

    @abc.abstractmethod
    def get_action(self) -> M.Action:
        """Get next action."""

    @abc.abstractmethod
    def update(self, action: M.Action, obs: M.Observation) -> None:
        """Update policy history given action and observation."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the policy."""

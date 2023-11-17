"""PyTorch Policies."""
from __future__ import annotations

import os
import pickle
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from torch.distributions import Categorical, Normal

from posggym import logger
from posggym.agents.policy import ActType, ObsType, Policy, PolicyID, PolicyState
from posggym.agents.registration import PolicySpec
from posggym.agents.utils import action_distributions, processors
from posggym.agents.utils.download import download_from_repo
from posggym.utils import seeding


if TYPE_CHECKING:
    import posggym.model as M


class PPOTorchModelSaveFileFormat(NamedTuple):
    """Format for saving and loading POSGGym PPOLSTMModel."""

    weights: Dict[str, Any]
    trunk_sizes: List[int]
    lstm_size: int
    lstm_layers: int
    head_sizes: List[int]
    activation: str
    lstm_use_prev_action: bool
    lstm_use_prev_reward: bool


class PPOLSTMModel(nn.Module):
    """A PPO LSTM Model.

    Follows architecture used by Rllib, namely:

      Fully Connected Layers -> LSTM Layer -> policy head (Fully connected layer)
                                           -> value head (Fully connected layer)

    Note, this model is designed for the intended use of having it's weights loaded
    from a file, and not for training. Thus it makes no effort to do things like using
    good weight initialization, rather it just aims to match the architecture of the
    saved model's weights.

    Additionally, it's forward function is designed to take in a single step of
    observations (along with lstm state and optional action and reward) and output
    next actions. This could be a single observation or a batch. But does
    not include logic for unrolling a sequence of observations.

    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        trunk_sizes: List[int],
        lstm_size: int,
        lstm_layers: int,
        head_sizes: List[int],
        activation: str,
        lstm_use_prev_action: bool,
        lstm_use_prev_reward: bool,
    ):
        assert isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 1, (
            "Only 1D Box observation spaces are supported for PPO PyTorch Policy "
            "models. Look into using `gymansium.spaces.flatten_space` to flatten your "
            "observation space, along with using the `FlattenPreprocessor` from "
            f"`posggym.agents.utils.preprocessors`. Got {obs_space}."
        )
        super().__init__()
        self.obs_space = obs_space
        self.action_space = action_space

        self.use_prev_action = lstm_use_prev_action
        self.use_prev_reward = lstm_use_prev_reward

        if isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
        elif isinstance(action_space, spaces.MultiDiscrete):
            # Output is a one hot encoding for each action dimension
            # All one hot encodings concatenated together
            # Hence why we sum the action space nvec
            self.action_dim = np.sum(action_space.nvec)
        elif isinstance(action_space, spaces.Box) and action_space.shape is not None:
            # Output is mean and log_std for each action dimension
            # Hence why we multiply by 2
            # First half of output is mean, second half is log_std
            self.action_dim = np.prod(action_space.shape, dtype=np.int32) * 2
        else:
            raise ValueError(
                f"Unsupported action space for PPOLSTMModel `{type(action_space)}`. "
                "Expected either Discrete, MultiDiscrete, or Box."
            )

        activation_fn: Optional[Callable] = None
        if activation == "tanh":
            activation_fn = nn.Tanh
        elif activation == "relu":
            activation_fn = nn.ReLU

        # Fully connected trunk
        prev_size = int(np.product(obs_space.shape))
        trunk = []
        for size in trunk_sizes:
            trunk.append(nn.Linear(prev_size, size))
            if activation_fn:
                trunk.append(activation_fn())
            prev_size = size
        self.trunk = nn.Sequential(*trunk)

        # LSTM Layer
        lstm_input_size = prev_size
        if self.use_prev_action:
            lstm_input_size += self.action_dim
        if self.use_prev_reward:
            lstm_input_size += 1

        self.lstm = nn.LSTM(
            lstm_input_size, lstm_size, num_layers=lstm_layers, batch_first=False
        )
        prev_size = lstm_size

        # Fully connected actor and critic heads
        actor = []
        critic = []
        for size in head_sizes:
            actor.append(nn.Linear(prev_size, size))
            critic.append(nn.Linear(prev_size, size))
            if activation_fn:
                actor.append(activation_fn())
                critic.append(activation_fn())
            prev_size = size

        actor.append(nn.Linear(prev_size, self.action_dim))
        critic.append(nn.Linear(prev_size, 1))
        self.actor = nn.Sequential(*actor)
        self.critic = nn.Sequential(*critic)

        # Assume model is used for evaluation only
        self.eval()

    @property
    def device(self) -> torch.device:
        """The device this model is on."""
        return next(self.parameters()).device

    def get_next_state(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        lstm_state: Tuple[torch.Tensor, torch.Tensor],
        prev_action: Optional[Union[np.ndarray, torch.Tensor]] = None,
        prev_reward: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get next lstm output and state.

        If obs is not batched, adds batch dimension with batch size of 1.

        Arguments
        ---------
        obs: the observation, shape=(batch_size, obs_size) | (obs_size, )
        lstm_state: the lstm state, this is a tuple of two tensors, each with
            shape=(num_layers, batch_size, cell_size)
        prev_action: the previous actions, shape=(batch_size, action_size) |
            (action_size, )
        prev_reward: the previous reward, shape=(batch_size, 1) | (1, )

        Returns
        -------
        lstm_output: the lstm output, shape=(batch_size, cell_size)
        next_lstm_state: the next lstm state, this is a tuple of two tensors, each with
            shape=(num_layers, batch_size, cell_size)
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        if len(obs.shape) == 1:
            # Single observation
            # Add batch and sequence length dimensions
            obs = obs.reshape(1, 1, -1)
        elif len(obs.shape) == 2:
            # Batch of observations
            # Add sequence length dimension
            obs = obs.unsqueeze(1)

        hidden = self.trunk(obs)

        prev_action_reward = []
        if self.use_prev_action:
            if isinstance(self.action_space, (spaces.Discrete, spaces.MultiDiscrete)):
                # One-hot encode discrete actions
                prev_action = F.one_hot(
                    torch.tensor(prev_action, dtype=torch.int64),
                    self.action_dim,
                ).float()
            else:
                prev_action = torch.tensor(prev_action, dtype=torch.float32)
            prev_action_reward.append(prev_action)
        if self.use_prev_reward:
            raise ValueError(
                "Using previous reward as input to LSTM layer not currently supported."
            )

        if len(prev_action_reward) > 0:
            hidden = torch.cat([hidden] + prev_action_reward, dim=-1)

        lstm_output, lstm_state = self.lstm(hidden, lstm_state)
        # remove sequence length dimension
        lstm_output = lstm_output.squeeze(0)
        return lstm_output, lstm_state

    def get_value(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        lstm_state: Tuple[torch.Tensor, torch.Tensor],
        prev_action: Optional[Union[np.ndarray, torch.Tensor]] = None,
        prev_reward: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Get value function output.

        If input is not batched, then adds batch dimension with batch size of 1.

        Arguments
        ---------
        obs: the observation, shape=(batch_size, obs_size) | (obs_size, )
        lstm_state: the lstm state, this is a tuple of two tensors, each with
            shape=(num_layers, batch_size, cell_size)
        prev_action: the previous actions, shape=(batch_size, action_size) |
            (action_size, )
        prev_reward: the previous reward, shape=(batch_size, 1) | (1, )

        Returns
        -------
        value: output of value function, shape=(batch_size, 1)

        """
        hidden_state, _ = self.get_next_state(obs, lstm_state, prev_action, prev_reward)
        return self.critic(hidden_state)

    def get_action_and_value(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        lstm_state: Tuple[torch.Tensor, torch.Tensor],
        prev_action: Optional[Union[np.ndarray, torch.Tensor]] = None,
        prev_reward: Optional[Union[np.ndarray, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor
    ]:
        """Get next action and value.

        If input is not batched, then adds batch dimension with batch size of 1.

        Arguments
        ---------
        obs: the observation, shape=(batch_size, obs_size) | (obs_size, )
        lstm_state: the lstm state, this is a tuple of two tensors, each
            with shape=(num_layers, batch_size, cell_size)
        prev_action: the previous actions, shape=(batch_size, action_size) |
            (action_size, )
        prev_reward: the previous reward, shape=(batch_size, 1) | (1, )
        deterministic: whether to sample action from action distribution or
            deterministicly select action with highest probability.

        Returns
        -------
        action: next action, shape=(batch_size, action_size)
        next_lstm_state: state of LSTM layer after processing input, this is a tuple of
            two tensors, each with shape=(num_layers, batch_size, cell_size)
        value: output of value function, shape=(batch_size, 1)
        action_dist: the policy action distribution, shape will differ depending on
            the action space.

        """
        with torch.no_grad():
            hidden_state, next_lstm_state = self.get_next_state(
                obs, lstm_state, prev_action, prev_reward
            )
            value = self.critic(hidden_state)
            logits = self.actor(hidden_state)
            if isinstance(self.action_space, spaces.Discrete):
                action_dist = Categorical(logits=logits)
                if deterministic:
                    action = action_dist.probs.argmax(dim=1)
                else:
                    action = action_dist.sample()
                probs = action_dist.probs
            elif isinstance(self.action_space, spaces.MultiDiscrete):
                logit_splits = logits.split(self.action_space.nvec.tolist(), dim=1)
                action_dists = [Categorical(logits=split) for split in logit_splits]
                if deterministic:
                    actions = [dist.probs.argmax(dim=1) for dist in action_dists]
                else:
                    actions = [dist.sample() for dist in action_dists]
                action = torch.stack(actions, dim=1)
                probs = torch.stack([dist.probs for dist in action_dists], dim=1)
            else:
                # Box - continuous action space
                mean, log_std = logits.chunk(2, dim=1)
                action_dist = Normal(mean, log_std.exp())
                action = action_dist.mean if deterministic else action_dist.sample()
                probs = torch.stack([action_dist.mean, action_dist.stddev])

        return action, next_lstm_state, value, probs

    def get_initial_state(
        self, batch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the initial LSTM state.

        Arguments
        ---------
        batch_size: the batch size of the LSTM state

        Returns
        -------
        initial_state: the initial LSTM state, this is a tuple of two tensors, each
            with shape=(num_layers, batch_size, cell_size)

        """
        return (
            torch.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size)),
            torch.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size)),
        )


class PPOPolicy(Policy[ActType, ObsType]):
    """A PyTorch PPO Policy.

    Arguments
    ---------
    model: the model of the environment
    agent_id: ID of the agent in the environment the policy is for
    policy_id: ID of the policy
    policy_model: the underlying PyTorch policy model
    obs_processor: the observation processor to use for processing observations before
        they are passed into the policy model. If None, then an identity processor is
        used.
    action_processor: the action processor to use for processing actions before they
        are passed into the policy model, and for unprocessing actions sampled from the
        policy model. If None, then an identity processor is used.
    deterministic: whether to sample actions from the policy model stochastically or
        deterministically. If True, then actions are sampled deterministically.

    """

    def __init__(
        self,
        model: M.POSGModel,
        agent_id: str,
        policy_id: PolicyID,
        policy_model: PPOLSTMModel,
        obs_processor: processors.Processor | None = None,
        action_processor: processors.Processor | None = None,
        deterministic: bool = False,
    ):
        self.policy_model = policy_model
        self.deterministic = deterministic
        self.action_space = model.action_spaces[agent_id]

        if obs_processor is None:
            obs_processor = processors.IdentityProcessor(
                model.observation_spaces[agent_id]
            )
        self.obs_processor = obs_processor

        if action_processor is None:
            action_processor = processors.IdentityProcessor(
                model.action_spaces[agent_id]
            )
        self.action_processor = action_processor

        super().__init__(model, agent_id, policy_id)

        # RNG for sampling actions
        self._rng, _ = seeding.np_random()

    def reset(self, *, seed: int | None = None):
        super().reset(seed=seed)
        if seed is not None:
            # RNG for sampling actions
            self._rng, _ = seeding.np_random(seed=seed)

            import torch

            # https://pytorch.org/docs/stable/notes/randomness.html
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)

            if (
                hasattr(self.policy_model, "device")
                and str(self.policy_model.device) != "cpu"
            ):
                torch.backends.cudnn.benchmark = False
                # See https://github.com/pytorch/pytorch/issues/47672.
                # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
                cuda_version = torch.version.cuda
                if (
                    cuda_version is not None
                    and float(cuda_version) >= 10.2
                    and (
                        "CUBLAS_WORKSPACE_CONFIG" not in os.environ
                        or (
                            os.environ["CUBLAS_WORKSPACE_CONFIG"]
                            not in (":16:8", ":4096:2")
                        )
                    )
                ):
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

    def get_initial_state(self) -> PolicyState:
        state = super().get_initial_state()
        state["obs"] = None
        state["prev_action"] = None
        state["prev_reward"] = None
        state["lstm_state"] = self.policy_model.get_initial_state(batch_size=1)
        state["action_probs"] = None
        state["value"] = 0.0
        return state

    def get_next_state(
        self, action: ActType | None, obs: ObsType, state: PolicyState
    ) -> PolicyState:
        obs = self.obs_processor(obs)

        if action is not None:
            action = self.action_processor(action)

        (
            _,
            lstm_state,
            value,
            action_probs,
        ) = self.policy_model.get_action_and_value(
            obs,
            state["lstm_state"],
            prev_action=action,
            prev_reward=None,
            deterministic=self.deterministic,
        )

        return {
            "obs": obs,
            "prev_action": action,
            "prev_reward": None,
            "lstm_state": lstm_state,
            "action_probs": action_probs.numpy().squeeze(),
            "value": value[0],
        }

    def sample_action(self, state: PolicyState) -> ActType:
        pi = self.get_pi(state)
        action = pi.sample()
        return self.action_processor.unprocess(action)

    def get_value(self, state: PolicyState) -> float:
        return state["value"]

    def get_pi(self, state: PolicyState) -> action_distributions.ActionDistribution:
        processed_action_space = self.action_processor.input_space
        if isinstance(processed_action_space, spaces.Discrete):
            probs = {
                a: state["action_probs"][a] for a in range(processed_action_space.n)
            }
            return action_distributions.DiscreteActionDistribution(probs, self._rng)
        if isinstance(processed_action_space, spaces.MultiDiscrete):
            probs = [
                {
                    a: state["action_probs"][i][a]
                    for a in range(processed_action_space.nvec[i])
                }
                for i in range(len(processed_action_space))
            ]
            return action_distributions.MultiDiscreteActionDistribution(
                probs, self._rng
            )
        return action_distributions.NormalActionDistribution(
            state["action_probs"][0], state["action_probs"][1], self._rng
        )

    @staticmethod
    def load_from_path(
        model: M.POSGModel,
        agent_id: str,
        policy_id: str,
        policy_file_path: str,
        deterministic: bool = False,
        obs_processor_cls: Type[processors.Processor] | None = None,
        obs_processor_config: Dict[str, Any] | None = None,
        action_processor_cls: Type[processors.Processor] | None = None,
        action_processor_config: Dict[str, Any] | None = None,
    ) -> PPOPolicy:
        if not os.path.exists(policy_file_path):
            logger.info(
                f"Local copy of policy file for policy `{policy_id}` not found, so "
                "downloading it from posggym-agents repo and storing local copy for "
                "future use."
            )
            download_from_repo(policy_file_path)

        obs_space = model.observation_spaces[agent_id]
        if obs_processor_cls is None:
            if isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 1:
                obs_processor_cls = processors.IdentityProcessor
                obs_processor_config = None
            else:
                obs_processor_cls = processors.FlattenProcessor
                obs_processor_config = None

        if obs_processor_config is not None:
            obs_processor: processors.Processor = obs_processor_cls(
                obs_space, **obs_processor_config
            )
        else:
            obs_processor = obs_processor_cls(obs_space)

        action_space = model.action_spaces[agent_id]
        if action_processor_cls is None:
            action_processor_cls = processors.IdentityProcessor
            action_processor_config = None

        if action_processor_config is not None:
            action_processor: processors.Processor = action_processor_cls(
                action_space, **action_processor_config
            )
        else:
            action_processor = action_processor_cls(action_space)

        with open(policy_file_path, "rb") as f:
            data = PPOTorchModelSaveFileFormat(**pickle.load(f))

        policy_model = PPOLSTMModel(
            obs_space=obs_processor.get_processed_space(),
            action_space=action_processor.get_processed_space(),
            trunk_sizes=data.trunk_sizes,
            lstm_size=data.lstm_size,
            lstm_layers=data.lstm_layers,
            head_sizes=data.head_sizes,
            activation=data.activation,
            lstm_use_prev_action=data.lstm_use_prev_action,
            lstm_use_prev_reward=data.lstm_use_prev_reward,
        )
        policy_model.load_state_dict(data.weights)

        return PPOPolicy(
            model=model,
            agent_id=agent_id,
            policy_id=policy_id,
            policy_model=policy_model,
            obs_processor=obs_processor,
            action_processor=action_processor,
            deterministic=deterministic,
        )

    @staticmethod
    def get_spec_from_path(
        policy_file_path: str,
        env_id: str,
        env_args: Dict[str, Any] | None,
        env_args_id: str | None = None,
        version: int = 0,
        valid_agent_ids: List[str] | None = None,
        nondeterministic: bool = False,
        description: str | None = None,
        **kwargs,
    ) -> PolicySpec:
        """Load PPO policy spec from policy file.

        Arguments
        ---------
        policy_file_path: path to the policy file.
        env_id: ID of the posggym environment that the policy is for.
        env_args: Optional keywords arguments for the environment that the policy is
            for (if it is a environment specific policy). If None then assumes policy
            can be used for the environment with any arguments.
        env_args_id: Optional ID for the environment arguments. If None then an ID will
            be generated automatically from the env_args.
        version: the policy version
        valid_agent_ids: Optional AgentIDs for agents in environment that policy is
            compatible with. If None then assumes policy can be used for any agent in
            the environment.
        nondeterministic: Whether this policy is non-deterministic even after seeding.
        kwargs: Additional kwargs, if any, to pass to the agent initializing function
        description: Optional description of the policy.

        Returns
        -------
        spec: Policy specs for PPO Policy loaded from policy file.

        """
        # remove file extension
        policy_name = os.path.basename(policy_file_path).split(".")[0]
        kwargs = kwargs.copy()
        kwargs["policy_file_path"] = policy_file_path
        return PolicySpec(
            policy_name=policy_name,
            entry_point=PPOPolicy.load_from_path,  # type: ignore
            version=version,
            env_id=env_id,
            env_args=env_args,
            env_args_id=env_args_id,
            valid_agent_ids=valid_agent_ids,
            nondeterministic=nondeterministic,
            kwargs=kwargs,
            description=description,
        )

from ctypes import byref
from functools import partial
from itertools import product
from typing import NamedTuple, cast

import numpy as np
import torch
from gymnasium import spaces
from vmas.simulator.core import Agent, EntityState, Line, Sphere, World
from vmas.simulator.dynamics.common import Dynamics
from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.utils import (
    Color,
    TorchUtils,
    X,
    Y,
)

import posggym.model as M
from posggym.core import DefaultEnv
from posggym.envs.differentiable.utils import (
    AgentStateWrapper,
    POSGGymLandmark,
    POSGGymLidar,
    POSGGymSensor,
    TensorJointTimestep,
    clip_actions,
    clone_state,
)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class PPState(NamedTuple):
    """A state in the Continuous Predator-Prey Environment."""

    predator_states: dict[str, AgentStateWrapper]
    prey_states: dict[str, AgentStateWrapper]
    prey_caught: torch.Tensor


class PPAgent(Agent):
    def __init__(self, batch_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.caught = torch.zeros(batch_size, 1, dtype=torch.bool)

        self._state = AgentStateWrapper()
        self.rew = torch.Tensor()

    def set_caught(self, caught):
        self.caught = caught

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove lambda function or other unpicklable attributes
        if "_collision_filter" in state:
            del state["_collision_filter"]
        return state

    # Optional: Override __setstate__ to restore state
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Optionally re-create the lambda function after unpickling
        self._collision_filter = lambda _: True

    @property
    def state(self) -> AgentStateWrapper:
        return self._state

    @property
    def sensors(self) -> list[POSGGymSensor]:
        return self._sensors

    def set_all_pos(self, pos: torch.Tensor):
        self._set_all_state_property(EntityState.pos, self.state, pos)

    def set_all_vel(self, vel: torch.Tensor):
        self._set_all_state_property(EntityState.vel, self.state, vel)

    def set_all_rot(self, rot: torch.Tensor):
        self._set_all_state_property(EntityState.rot, self.state, rot)

    def set_all_ang_vel(self, ang_vel: torch.Tensor):
        self._set_all_state_property(EntityState.ang_vel, self.state, ang_vel)

    def _set_all_state_property(self, prop, entity: EntityState, new: torch.Tensor):
        value = prop.fget(entity)
        value[:, ...] = new
        self.notify_observers()


class PPWorld(World):
    def __init__(
        self,
        bound: float,
        blocks: list[tuple[tuple[float, float, float], float]] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, x_semidim=bound, y_semidim=bound, **kwargs)
        self.bound = bound
        self._agents: list[PPAgent] = []

        for i in range(4):
            self.add_landmark(
                POSGGymLandmark(
                    name=f"landmark-wall{i}",
                    collide=True,
                    shape=Line(length=self.bound * 2),
                    color=Color.WHITE,
                )
            )
        # Add landmarks
        if blocks is not None:
            for _, r in blocks:
                landmark = POSGGymLandmark(
                    name=f"landmark {i}",
                    collide=True,
                    shape=Sphere(radius=r),
                    color=Color.BLACK,
                )
                self.add_landmark(landmark)

        all_wall_pos: list[tuple[list[float], float]] = [
            ([0, self.bound], 0),
            ([0, -self.bound], 0),
            ([-self.bound, 0], np.pi / 2),
            ([self.bound, 0], np.pi / 2),
        ]

        for idx, (pos, rot) in enumerate(all_wall_pos):
            self.landmarks[idx].set_pos(
                torch.tensor(
                    pos,
                    device=self.device,
                ),
                batch_index=None,  # type: ignore
            )
            self.landmarks[idx].set_rot(
                torch.tensor(
                    [rot],
                    device=self.device,
                ),
                batch_index=None,  # type: ignore
            )
        if blocks is not None:
            for ((x, y, _), _), landmark in zip(
                blocks, self.landmarks[4:], strict=False
            ):
                landmark.set_pos(
                    torch.ones(
                        (self.batch_dim, self.dim_p),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    * torch.tensor([x, y], device=self.device, dtype=torch.float32),
                    batch_index=None,  # type: ignore
                )

    def add_agent(self, agent: PPAgent):
        super().add_agent(agent)

    def update_state(self, state: PPState):
        for a in self.agents:
            if a.name.startswith("adversary"):
                a_state = state.prey_states[a.name]
            else:
                a_state = state.predator_states[a.name]

            a.set_all_pos(a_state.pos_safe.clone())
            a.set_all_vel(a_state.vel_safe.clone())
            a.set_all_rot(a_state.rot_safe.clone())
            a.set_all_ang_vel(a_state.ang_vel_safe.clone())

    def get_state(self) -> PPState:
        return PPState(
            {x.name: clone_state(x.state) for x in self.predator},
            {x.name: clone_state(x.state) for x in self.prey},
            torch.cat([x.caught for x in self.prey], dim=1),
        )

    @property
    def agents(self) -> list[PPAgent]:
        return self._agents

    @property
    def prey(self) -> list[PPAgent]:
        return [x for x in self.agents if x.name.startswith("adversary")]

    @property
    def predator(self) -> list[PPAgent]:
        return [x for x in self.agents if x.name.startswith("agent")]


class PredatorPreyDiffModel(M.POSGModel[PPState, torch.Tensor, torch.Tensor]):
    R_MAX = 2
    MAX_AGENTS = 8

    def __init__(
        self,
        world: partial[PPWorld],
        num_predators: int = 5,
        num_prey: int = 8,
        cooperative: bool = False,
        prey_strength: int | None = None,
        obs_dist: float = 10,
        n_sensors: int = 8,
        batch_size=4,
        device: str = "cpu",
    ) -> None:
        assert 1 < num_predators <= self.MAX_AGENTS
        assert num_prey > 0
        assert obs_dist > 0

        self._world = world

        self.num_predators = num_predators
        self.num_prey = num_prey
        self.num_landmarks = 2
        self.num_agents = self.num_predators + self.num_predators
        self.cooperative = cooperative
        self.prey_strength = prey_strength
        self.obs_dist = obs_dist
        self.n_sensors = n_sensors
        self.prey_obs_dist = 1.0
        self.adversaries_share_rew = True
        self.shape_agent_rew = True
        self.shape_adversary_rew = True
        self.agents_share_rew = False
        self.prey_share_rew = True
        self.observe_same_team = True
        self.observe_pos = True
        self.observe_vel = True
        self.bound = None
        self.respawn_at_catch = False
        self.per_prey_reward = self.R_MAX / self.num_prey
        self.prey_capture_dist = 0.1
        self.batch_size = batch_size
        self.is_symmetric = True
        self.device = device

        self.action_spaces = {
            i: spaces.Box(np.array([-1, -1]), np.array([1, 1]), seed=42 + idx)
            for idx, i in enumerate(self.possible_agents)
        }

        self.observation_spaces = {
            i: spaces.Box(
                low=np.array([0.0] * self.n_sensors * 3),
                high=np.array([self.obs_dist] * self.n_sensors * 3),
            )
            for i in self.possible_agents
        }
        self.initialise()

    @property
    def reward_ranges(self) -> dict[str, tuple[float, float]]:
        return {i: (-12, self.R_MAX) for i in self.possible_agents}

    def get_agents(self, state: PPState) -> list[str]:
        return list(self.possible_agents)

    def sample_initial_state(self) -> PPState:
        return self.reset_world_at()

    def gen_dynamics(self) -> Dynamics:
        idx = torch.randint(0, 2, (1,), generator=self.rng, device=self.device).item()
        return [Holonomic(), DiffDrive(self.world, integration="rk4")][
            idx
        ]  # type: ignore

    def initialise(self) -> PPState:
        self.world = self._world(
            batch_dim=self.batch_size,
            device=self.device,
            substeps=10,
            collision_force=500,
        )

        self.bound = self.world.bound

        # set any world properties first
        num_agents = self.num_predators + self.num_predators
        self.adversary_radius = 0.075

        # Add agents
        for i in range(num_agents):
            adversary = i < self.num_predators
            name = f"adversary_{i}" if adversary else f"agent_{i - self.num_predators}"
            agent = PPAgent(
                batch_size=self.batch_size,
                name=name,
                collide=True,
                shape=Sphere(radius=self.adversary_radius if adversary else 0.05),
                u_multiplier=3.0 if adversary else 4.0,
                max_speed=1.0 if adversary else 1.3,
                color=Color.BLUE if adversary else Color.GREEN,
                adversary=adversary,
                dynamics=Holonomic() if adversary else self.gen_dynamics(),
                sensors=(
                    [
                        POSGGymLidar(
                            self.world,
                            entity_name="landmark",
                            n_rays=self.n_sensors,
                            max_range=self.obs_dist,
                            render_color=Color.GREEN,
                            angle_start=0.05,
                            angle_end=(2 * torch.pi) + 0.05,
                        ),
                        POSGGymLidar(
                            self.world,
                            entity_name="adversary",
                            n_rays=self.n_sensors,
                            max_range=self.obs_dist,
                            render_color=Color.RED,
                            angle_start=0.05,
                            angle_end=(2 * torch.pi) + 0.05,
                        ),
                        POSGGymLidar(
                            self.world,
                            entity_name="agent",
                            n_rays=self.n_sensors,
                            max_range=self.obs_dist,
                            render_color=Color.BLUE,
                            angle_start=0.05,
                            angle_end=(2 * torch.pi) + 0.05,
                        ),
                    ]
                    if not adversary
                    else []
                ),
            )
            self.world.add_agent(agent)

        return self.reset_world_at()

    def reset_world_at(self) -> PPState:
        assert self.bound is not None

        predator_states, prey_states, prey_caught = (
            {},
            {},
            torch.zeros(self.batch_size, self.num_prey, dtype=torch.bool),
        )
        for p in self.world.predator:
            state = AgentStateWrapper()
            state.batch_dim = self.world._batch_dim  # pyright: ignore
            state.device = self.world._device  # pyright: ignore

            state.pos = torch.zeros(
                (self.world.batch_dim, self.world.dim_p),
                device=self.device,
                dtype=torch.float32,
            ).uniform_(-self.bound, self.bound, generator=self.rng)
            state.pos.requires_grad = True
            state.vel = torch.zeros(
                (self.world.batch_dim, self.world.dim_p),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            state.rot = torch.zeros(
                self.world.batch_dim,
                1,
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            state.ang_vel = torch.zeros(
                self.world.batch_dim,
                1,
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            predator_states[p.name] = state

        for p in self.world.prey:
            state = AgentStateWrapper()
            state.batch_dim = self.world._batch_dim  # pyright: ignore
            state.device = self.world._device  # pyright: ignore

            state.pos = torch.zeros(
                (self.world.batch_dim, self.world.dim_p),
                device=self.device,
                dtype=torch.float32,
            ).uniform_(-self.bound, self.bound, generator=self.rng)
            state.pos.requires_grad = True
            state.vel = torch.zeros(
                (self.world.batch_dim, self.world.dim_p),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            state.rot = torch.zeros(
                self.world.batch_dim,
                1,
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            state.ang_vel = torch.zeros(
                self.world.batch_dim,
                1,
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            prey_states[p.name] = state

        return PPState(predator_states, prey_states, prey_caught)

    def is_collision(self, agent1: Agent, agent2: Agent):
        delta_pos = agent1.state.pos - agent2.state.pos  # type: ignore
        dist = torch.linalg.vector_norm(delta_pos, dim=-1)
        dist_min = agent1.shape.radius + agent2.shape.radius  # type: ignore
        return dist < dist_min

    # return all adversarial agents
    def prey(self, world: PPWorld):
        return [agent for agent in world.agents if agent.adversary]

    def _get_prey_move_angles(self, state: PPState) -> torch.Tensor:
        pred_states = torch.stack(
            [x.pos_safe for x in state.predator_states.values()], dim=1
        )
        prey_states = torch.stack(
            [x.pos_safe for x in state.prey_states.values()], dim=1
        )

        pred_dists = torch.linalg.norm(prey_states - pred_states, axis=-1)
        prey_dists = torch.linalg.norm(
            prey_states.unsqueeze(2) - prey_states.unsqueeze(1), dim=-1
        )

        prey_actions = -torch.ones(state.prey_caught.shape)

        a = []

        for i, prey in enumerate(state.prey_states.values()):
            prey_actions[state.prey_caught[:, i], i] = 0

            pred_dists = torch.linalg.norm(
                prey.pos_safe[:, None, :] - pred_states, axis=1
            )

            min_pred_dist, pred_idx = pred_dists.min(dim=1)

            expanded_idx = pred_idx.view(-1, 1, 1).expand(-1, 1, 2)
            gathered_values = torch.gather(pred_states, 1, expanded_idx)
            pred_influence_angle = torch.atan2(
                prey.pos_safe[:, 1] - gathered_values[:, 0, 1],
                prey.pos_safe[:, 0] - gathered_values[:, 0, 0],
            )

            not_current_mask = torch.ones(len(state.prey_states), dtype=torch.bool)
            not_current_mask[i] = False

            # Compute distances
            prey_dists = torch.linalg.norm(
                prey_states[:, not_current_mask, :] - prey_states[:, i : i + 1, :],
                dim=2,
            )
            min_prey_dist, prey_idx = prey_dists.min(dim=1)

            prey_influence_strength = torch.clamp(
                1.0 - prey_dists, min=0.0
            )  # Clamp to ensure no negative values
            strength_sum = prey_influence_strength.sum(dim=1, keepdim=True)
            normalized_strength = prey_influence_strength / (
                strength_sum + 1e-6
            )  # Avoid division by zero

            expanded_idx = prey_idx.view(-1, 1, 1).expand(-1, 1, 2)
            gathered_values = torch.gather(pred_states, 1, expanded_idx)
            prey_influence_angle = torch.atan2(
                prey.pos_safe[:, 1] - gathered_values[:, 0, 1],
                prey.pos_safe[:, 0] - gathered_values[:, 0, 0],
            )

            prey_influence_strength = torch.clamp(
                1.0 - min_prey_dist, min=0.0
            )  # Prey influence strength
            pred_influence_strength = torch.clamp(
                1.0 - min_pred_dist, min=0.0
            )  # Predator influence strength

            # Step 2: Combine the influence strengths
            total_influence_strength = torch.stack(
                (prey_influence_strength, pred_influence_strength), dim=0
            )
            row_sums = total_influence_strength.sum(dim=0, keepdim=True)
            normalized_strength = total_influence_strength / (row_sums + 1e-6)
            zero_rows = (normalized_strength.sum(dim=0, keepdim=True) == 0).float()
            normalized_strength += zero_rows / normalized_strength.size(1)

            angles = torch.stack((prey_influence_angle, pred_influence_angle), dim=0)
            a.append((angles * normalized_strength).sum(dim=0, keepdim=True).T)
        return torch.stack(a, dim=1).squeeze(2)

    def reward(self, agent: PPAgent):
        is_first = agent == self.world.predator[0]

        if is_first:
            for a in self.world.predator:
                a.rew = self.agent_reward(a)

            self.agents_rew = torch.stack(
                [a.rew for a in self.world.predator], dim=-1
            ).sum(-1)

        if self.agents_share_rew:
            return self.agents_rew
        else:
            return agent.rew

    def agent_reward(self, agent: PPAgent):
        # Agents are negatively rewarded if caught by adversaries
        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        adversaries = self.world.prey
        if self.shape_agent_rew:
            # reward can optionally be shaped
            # (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * torch.linalg.vector_norm(
                    agent.state.pos_safe - adv.state.pos, dim=-1
                )
        if agent.collide:
            for a in adversaries:
                # pass
                rew[self.is_collision(a, agent)] -= 10 / len(adversaries)

        return rew

    def adversary_reward(self, agent: PPAgent):
        # Adversaries are rewarded for collisions with agents
        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        agents = self.world.predator
        if self.shape_adversary_rew:  # reward can optionally be shaped
            # (decreased reward for increased distance from agents)
            rew -= (
                0.1
                * torch.min(
                    torch.stack(
                        [
                            torch.linalg.vector_norm(
                                a.state.pos_safe - agent.state.pos,
                                dim=-1,
                            )
                            for a in agents
                        ],
                        dim=-1,
                    ),
                    dim=-1,
                )[0]
            )
        if agent.collide:
            for ag in agents:
                rew[self.is_collision(ag, agent)] += 10
        return rew

    def observation(self, name: str):
        world_agent = next(x for x in self.world.agents if name == x.name)
        lidar_1_measures = torch.stack(
            tuple(s.measure(self.world) for s in world_agent.sensors)
        )
        return lidar_1_measures.reshape(self.batch_size, -1)

    def sample_initial_obs(self, state: PPState) -> dict[str, torch.Tensor]:
        obs = {}
        for name, _agent in state.predator_states.items():
            observation = TorchUtils.recursive_clone(self.observation(name))
            obs.update({name: observation})
        return obs

    def info(self, world: PPWorld, agent: Agent):
        return {}

    def done(self, world: PPWorld):
        agents: list[PPAgent] = world.agents  # type: ignore
        return torch.Tensor([x.caught for x in agents])

    def get_from_scenario(
        self,
    ):
        obs, rewards, infos, dones = {}, {}, {}, {}

        for agent in self.world.agents:
            if agent.name.startswith("agent"):
                observation = TorchUtils.recursive_clone(self.observation(agent.name))
                obs.update({agent.name: observation})

        for agent in self.world.predator:
            reward = self.reward(agent).clone()
            rewards.update({agent.name: reward})

        for agent in self.world.predator:
            info = TorchUtils.recursive_clone(self.info(self.world, agent))
            infos.update({agent.name: info})

        dones = {
            i: torch.zeros((self.batch_size), dtype=torch.bool, device=self.device)
            for i in self.possible_agents
        }
        truncated = {
            i: torch.zeros((self.batch_size), dtype=torch.bool, device=self.device)
            for i in self.possible_agents
        }

        return [obs, rewards, dones, truncated, infos]

    @property
    def possible_agents(self):
        return tuple(f"agent_{x}" for x in range(self.num_predators))

    def render(self):
        pass

    def step(
        self, state: PPState, actions: dict[str, torch.Tensor]
    ) -> TensorJointTimestep:
        self.world.update_state(state)

        prey_actions = self._get_prey_move_angles(state)
        prey_actions_ = [
            torch.stack([torch.cos(angle), torch.sin(angle)]).detach()
            for angle in prey_actions
        ]
        prey_actions_ = torch.stack(
            [torch.cos(prey_actions), torch.sin(prey_actions)], dim=-1
        ).permute(1, 0, 2)

        # clip actions
        clipped_actions = clip_actions(actions, self.action_spaces)

        for idx, agent in enumerate(self.world.predator):
            action = clipped_actions[f"agent_{idx}"]
            agent.action.u = action
            agent.dynamics.process_action()

        for act, agent in zip(prey_actions_, self.world.prey, strict=False):
            agent.action.u = act
            agent.state.force = agent.action.u
        self.world.step()

        next_state = self.world.get_state()

        obs, rewards, terminated, truncated, infos = self.get_from_scenario()
        all_done = torch.stack(tuple(terminated.values())).transpose(1, 0).all(dim=1)

        return TensorJointTimestep(
            next_state, obs, rewards, terminated, truncated, all_done, infos
        )

    @property
    def rng(self) -> torch.Generator:
        if self._rng is None:
            self._rng = torch.Generator(device=self.device)

        return self._rng


class PredatorPreyDiff(DefaultEnv[PPState, torch.Tensor, torch.Tensor]):
    def __init__(
        self,
        world: partial[PPWorld] | str,
        num_predators: int = 4,
        num_prey: int = 8,
        cooperative: bool = False,
        prey_strength: int | None = None,
        obs_dist: float = 10,
        n_sensors: int = 32,
        batch_size: int = 4,
        device: str | None = None,
        render_mode: str = "human",
    ) -> None:
        if isinstance(world, str):
            assert world in SUPPORTED_WORLDS, (
                f"Unsupported world name '{world}'. World name must be one of: "
                f"{list(SUPPORTED_WORLDS)}."
            )
            world = SUPPORTED_WORLDS[world]()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model = PredatorPreyDiffModel(
            world,
            num_predators,
            num_prey,
            cooperative,
            prey_strength,
            obs_dist,
            n_sensors,
            batch_size,
            self.device,
        )
        self.batch_size = batch_size
        self.viewer = None
        self.visible_display = None

        super().__init__(model)

    def render(
        self,
        mode="human",
        env_index=0,
        agent_index_focus: int | None = None,
        visualize_when_rgb: bool = False,
    ):
        """Render function for environment using pyglet
        From VMAS.
        """
        viewer_size = (700, 700)

        model = cast(PredatorPreyDiffModel, self.model)

        shared_viewer = agent_index_focus is None
        aspect_ratio = viewer_size[0] / viewer_size[1]

        headless = mode == "rgb_array" and not visualize_when_rgb
        # First time rendering
        if self.visible_display is None:
            self.visible_display = not headless
            self.headless = headless
        # All other times headless should be the same
        else:
            assert self.visible_display is not headless

        # First time rendering
        if self.viewer is None:
            try:
                import pyglet
            except ImportError as err:
                raise ImportError(
                    "Cannot import pyglet: you can install"
                    "pyglet directly via 'pip install pyglet'."
                ) from err

            try:
                # Try to use EGL
                pyglet.lib.load_library("EGL")

                # Only if we have GPUs
                from pyglet.libs.egl import egl, eglext

                num_devices = egl.EGLint()
                eglext.eglQueryDevicesEXT(0, None, byref(num_devices))
                assert num_devices.value > 0

            except (ImportError, AssertionError):
                self.headless = False
            pyglet.options["headless"] = self.headless

            self._init_rendering()

        zoom = 1.2

        if aspect_ratio < 1:
            cam_range = torch.tensor([zoom, zoom / aspect_ratio], device=self.device)
        else:
            cam_range = torch.tensor([zoom * aspect_ratio, zoom], device=self.device)

        if shared_viewer:
            # zoom out to fit everyone
            all_poses = torch.stack(
                [
                    agent.state.pos[env_index]  # type: ignore
                    for agent in model.world.agents + model.world.landmarks
                ],
                dim=0,
            )
            max_agent_radius = max(
                [agent.shape.circumscribed_radius() for agent in model.world.agents]
            )
            viewer_size_fit = (
                torch.stack(
                    [
                        torch.max(torch.abs(all_poses[:, X] - 0)),
                        torch.max(torch.abs(all_poses[:, Y] - 0)),
                    ]
                )
                + 2 * max_agent_radius
            )

            viewer_size = torch.maximum(
                viewer_size_fit / cam_range,
                torch.tensor(zoom, device=self.device),
            )
            cam_range *= torch.max(viewer_size)
            assert self.viewer is not None

            self.viewer.set_bounds(
                -cam_range[X] + 0,
                cam_range[X] + 0,
                -cam_range[Y] + 0,
                cam_range[Y] + 0,
            )

        for entity in model.world.entities:
            assert self.viewer is not None
            self.viewer.add_onetime_list(entity.render(env_index=env_index))

        # render to display or array
        assert self.viewer is not None

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def _init_rendering(self):
        from vmas.simulator import rendering

        self.viewer = rendering.Viewer(
            *(700, 700), visible=self.visible_display or False
        )
        model = cast(PredatorPreyDiffModel, self.model)

        self.text_lines = []
        idx = 0
        if model.world.dim_c > 0:
            for agent in model.world.agents:
                if not agent.silent:
                    text_line = rendering.TextLine(y=idx * 40)
                    self.viewer.geoms.append(text_line)
                    self.text_lines.append(text_line)
                    idx += 1


def get_5x5_world() -> partial[PPWorld]:
    """Generate 5x5 world layou`t."""
    return get_default_world(5, include_blocks=False)


def get_5x5_blocks_world() -> partial[PPWorld]:
    """Generate 5x5 Blocks world layout."""
    return get_default_world(5, include_blocks=True)


def get_10x10_world() -> partial[PPWorld]:
    """Generate 10x10 world layou`t."""
    return get_default_world(10, include_blocks=False)


def get_10x10_blocks_world() -> partial[PPWorld]:
    """Generate 10x10 Blocks world layout."""
    return get_default_world(10, include_blocks=True)


def get_15x15_world() -> partial[PPWorld]:
    """Generate 15x15 world layou`t."""
    return get_default_world(15, include_blocks=False)


def get_15x15_blocks_world() -> partial[PPWorld]:
    """Generate 15x15 Blocks world layout."""
    return get_default_world(15, include_blocks=True)


def get_20x20_world() -> partial[PPWorld]:
    """Generate 20x20 world layout."""
    return get_default_world(20, include_blocks=False)


def get_20x20_blocks_world() -> partial[PPWorld]:
    """Generate 20x20 Blocks world layout."""
    return get_default_world(20, include_blocks=True)


def get_default_world(size: int, include_blocks: bool) -> partial[PPWorld]:
    """Get function for generaing default world with given size.

    If `include_blocks=True` then world will contain blocks with the following layout:

    .....
    .#.#.
    .....
    .#.#.
    .....

    Where `#` are the blocks, which will be represented as a single circle.
    """
    bound = size / 10

    r = float(bound / 10)
    if include_blocks:
        blocks = [
            ((x, y, 0.0), r)
            for x, y in product([-3 * bound / 5, 3 * bound / 5], repeat=2)
        ]
    else:
        blocks = []
    return partial(PPWorld, bound=bound, blocks=blocks)


SUPPORTED_WORLDS = {
    "5x5": get_5x5_world,
    "5x5Blocks": get_5x5_blocks_world,
    "10x10": get_10x10_world,
    "10x10Blocks": get_10x10_blocks_world,
    "15x15": get_15x15_world,
    "15x15Blocks": get_15x15_blocks_world,
    "20x20": get_20x20_world,
    "20x20Blocks": get_20x20_blocks_world,
}

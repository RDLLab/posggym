import dataclasses
from abc import abstractmethod
from collections.abc import Callable

import numpy as np
import torch
from gymnasium import spaces
from vmas.simulator.core import AgentState, Box, Entity, Landmark, Line, Sphere, World
from vmas.simulator.sensors import Lidar, Sensor
from vmas.simulator.utils import TorchUtils, X, Y

import posggym.model as M


ZERO = 0.0
ABOVE_VALUE = 0.5
BELOW_VALUE = -0.5


@dataclasses.dataclass(order=True)
class TensorJointTimestep(M.JointTimestep):
    """The result of a single step in the model.

    Supports iteration.

    A dataclass is used instead of a Namedtuple so that generic typing is seamlessly
    supported.

    """

    terminations: dict[str, torch.Tensor]
    truncations: dict[str, torch.Tensor]
    all_done: torch.Tensor
    infos: dict[str, dict]

    def __iter__(self):
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)


def clone_tensors(obj: AgentState):
    cloned_attrs = {}
    for attr_name, attr_value in obj.__dict__.items():
        if isinstance(attr_value, torch.Tensor):
            cloned_attrs[attr_name] = attr_value.clone()
        else:
            cloned_attrs[attr_name] = attr_value
    return cloned_attrs


class AgentStateWrapper(AgentState):
    @property
    def pos_safe(self) -> torch.Tensor:
        if self._pos is None:
            raise AttributeError("pos is none")

        return self._pos

    @property
    def rot_safe(self) -> torch.Tensor:
        if self._rot is None:
            raise AttributeError("rot is none")

        return self._rot

    @property
    def vel_safe(self) -> torch.Tensor:
        if self._vel is None:
            raise AttributeError("vel is none")

        return self._vel

    @property
    def ang_vel_safe(self) -> torch.Tensor:
        if self._ang_vel is None:
            raise AttributeError("ang_vel is none")

        return self._ang_vel

    def __eq__(self, value) -> bool:
        if not isinstance(value, AgentStateWrapper):
            return False

        return (
            torch.equal(self.pos_safe, value.pos_safe)
            and torch.equal(self.rot_safe, value.rot_safe)
            and torch.equal(self.vel_safe, value.vel_safe)
            and torch.equal(self.ang_vel_safe, value.ang_vel_safe)
        )


def clone_state(state: AgentStateWrapper):
    a_s = AgentStateWrapper()
    t = clone_tensors(state)
    a_s._batch_dim = t["_batch_dim"]  # pyright: ignore
    a_s._device = t["_device"]  # pyright: ignore
    a_s.pos = t["_pos"]
    a_s.ang_vel = t["_ang_vel"]
    a_s.force = t["_force"]
    a_s.pos = t["_pos"]
    a_s.rot = t["_rot"]
    a_s.torque = t["_torque"]
    a_s.vel = t["_vel"]

    return a_s


def clip_actions(
    actions: dict[str, torch.Tensor], action_spaces: dict[str, spaces.Space]
) -> dict[str, torch.Tensor]:
    """Clip continuous actions so they are within the agents action space dims."""
    clipped_actions = {}
    for i, a in actions.items():
        a_space = action_spaces[i]
        assert isinstance(a_space, spaces.Box)
        if isinstance(a, torch.Tensor):
            clipped_actions[i] = torch.clip(
                a, torch.from_numpy(a_space.low), torch.from_numpy(a_space.high)
            )
        else:
            clipped_actions[i] = torch.from_numpy(np.clip(a, a_space.low, a_space.high))

    return clipped_actions


class POSGGymLandmark(Landmark):
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


class POSGGymSensor(Sensor):
    @abstractmethod
    def measure(self, world: World):
        raise NotImplementedError


class POSGGymLidar(Lidar, POSGGymSensor):
    def __init__(self, world: World, entity_name: str, **kwargs) -> None:
        self.entity_name = entity_name
        super().__init__(world, **kwargs)

    def entity_filter(self, e: Entity) -> bool:
        return e.name.startswith(self.entity_name)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove lambda function or other unpicklable attributes
        if "_entity_filter" in state:
            del state["_entity_filter"]
        return state

    # Optional: Override __setstate__ to restore state
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Optionally re-create the lambda function after unpickling
        self._entity_filter = lambda _: True

    def measure(self, world: World):
        assert self.agent is not None
        dists = cast_ray(
            self.agent,
            world.entities,
            self._angles,
            max_range=self._max_range,
            entity_filter=self.entity_filter,
            batch_dim=world.batch_dim,
            device=world.device,
        )
        self._last_measurement = dists.swapaxes(1, 0)
        return torch.clip(dists, 0, self._max_range)


# @torch.compile
def cast_ray(
    entity: Entity,
    entities: list[Entity],
    angles: torch.Tensor,
    max_range: float,
    entity_filter: Callable[[Entity], bool] = lambda _: False,
    batch_dim: int = 0,
    device: torch.device | None = None,
):
    if device is None:
        device = torch.device("cuda")

    pos = entity.state.pos

    # Initialize with full max_range to avoid
    # dists being empty when all entities are filtered
    dists = [
        torch.full((batch_dim, angles.shape[0]), fill_value=max_range, device=device)
    ]

    for e in entities:
        if entity is e or not entity_filter(e):
            continue
        assert e.collides(entity) and entity.collides(
            e
        ), "Rays are only casted among collidables"
        if isinstance(e.shape, Box):
            d = _cast_ray_to_box(e, pos, angles.T, max_range)
        elif isinstance(e.shape, Sphere):
            d = _cast_ray_to_sphere(e, pos, angles.T, max_range)
        elif isinstance(e.shape, Line):
            d = _cast_ray_to_line(e, pos, angles.T, max_range)
        else:
            raise RuntimeError(f"Shape {e.shape} currently not handled by cast_ray")
        dists.append(d)
    dist, _ = torch.min(torch.stack(dists, dim=-1), dim=-1)
    return dist


VECTOR_SIZE = 2


def rotate_vector(vector: torch.Tensor, angle: torch.Tensor):
    if len(angle.shape) == len(vector.shape):
        angle = angle.squeeze(-1)

    if angle.ndim < vector.ndim:
        angle = angle.view(*([1] * (vector.ndim - angle.ndim)), *angle.shape)

    assert vector.shape[-1] == VECTOR_SIZE

    cos = torch.cos(angle)
    sin = torch.sin(angle)

    return torch.stack(
        [
            vector[..., 0] * cos - vector[..., 1] * sin,
            vector[..., 0] * sin + vector[..., 1] * cos,
        ],
        dim=-1,
    )


# @torch.compile
def _cast_ray_to_sphere(
    sphere: Entity,
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
    max_range: float,
):
    ray_dir_world = torch.stack(
        [torch.cos(ray_direction), torch.sin(ray_direction)], dim=-1
    )
    assert sphere.state.pos is not None

    test_point_pos = sphere.state.pos[:, None, :].repeat(1, ray_dir_world.shape[1], 1)
    line_rot = ray_direction
    line_length = max_range
    ray_origin_ = ray_origin[:, None, :].repeat(1, ray_dir_world.shape[1], 1)
    line_pos = ray_origin_ + ray_dir_world * (line_length / 2)

    closest_point = _get_closest_point_line(
        line_pos,
        line_rot.unsqueeze(-1),
        line_length,
        test_point_pos,
        limit_to_line_length=False,
    )

    d = test_point_pos - closest_point
    d_norm = torch.linalg.vector_norm(d, dim=2)
    ray_intersects = d_norm < sphere.shape.radius
    a = sphere.shape.radius**2 - d_norm**2
    m = torch.sqrt(torch.where(a > 0, a, 1e-8))

    u = test_point_pos - ray_origin_
    u1 = closest_point - ray_origin_

    # Dot product of u and u1
    u_dot_ray = (u * ray_dir_world).sum(-1)
    sphere_is_in_front = u_dot_ray > ZERO
    dist = torch.linalg.vector_norm(u1, dim=2) - m
    dist[~(ray_intersects & sphere_is_in_front)] = max_range

    return dist


def cross(vector_a: torch.Tensor, vector_b: torch.Tensor):
    # Ensure vector_a is broadcasted to match the shape of vector_b
    vector_a_expanded = vector_a.unsqueeze(1).expand(-1, vector_b.size(1), -1)

    return (
        vector_a_expanded[..., 0] * vector_b[..., 1]
        - vector_a_expanded[..., 1] * vector_b[..., 0]
    ).unsqueeze(-1)


# @torch.compile
def _cast_ray_to_line(
    line: Entity,
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
    max_range: float,
):
    """Inspired by:
    https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
    Computes distance of ray originating from pos at angle to a line an
    sets distance to max_range if there is no intersection.
    """
    assert isinstance(line.shape, Line)

    assert line.state.rot is not None

    p = line.state.pos
    r = (
        torch.stack(
            [
                torch.cos(line.state.rot.squeeze(1)),
                torch.sin(line.state.rot.squeeze(1)),
            ],
            dim=-1,
        )
        * line.shape.length
    )

    q = ray_origin
    s = torch.stack(
        [
            torch.cos(ray_direction),
            torch.sin(ray_direction),
        ],
        dim=-1,
    )

    r = r.unsqueeze(1)  # Shape becomes [2, 1, 2]
    r = r.expand(-1, s.shape[1], -1)  # Shape becomes [2, 32, 2]

    rxs = TorchUtils.cross(r, s)
    rxs[rxs == ZERO] = 1e-10

    t = cross(q - p, s / rxs)
    u = cross(q - p, r / rxs)

    d = torch.linalg.norm(u * s, dim=-1)

    perpendicular = rxs == ZERO
    above_line = t > ABOVE_VALUE
    below_line = t < BELOW_VALUE
    behind_line = u < ZERO

    new_d = d.clone()
    new_d[perpendicular.squeeze(-1)] = max_range
    new_d[above_line.squeeze(-1)] = max_range
    new_d[below_line.squeeze(-1)] = max_range
    new_d[behind_line.squeeze(-1)] = max_range

    return new_d


# @torch.compile
def _get_closest_point_line(
    line_pos,
    line_rot,
    line_length,
    test_point_pos,
    limit_to_line_length: bool = True,
):
    if not isinstance(line_length, torch.Tensor):
        line_length = torch.tensor(
            line_length, dtype=torch.float32, device=line_pos.device
        ).expand(line_pos.shape[0])
    # Rotate it by the angle of the line
    rotated_vector = torch.cat([line_rot.cos(), line_rot.sin()], dim=-1)
    # Get distance between line and sphere
    delta_pos = line_pos - test_point_pos
    # Dot product of distance and line vector
    dot_p = (delta_pos * rotated_vector).sum(-1).unsqueeze(-1)
    # Coordinates of the closes point
    sign = torch.sign(dot_p)
    distance_from_line_center = (
        torch.minimum(
            torch.abs(dot_p),
            (line_length / 2).view(dot_p.shape),
        )
        if limit_to_line_length
        else torch.abs(dot_p)
    )
    closest_point = line_pos - sign * distance_from_line_center * rotated_vector
    return closest_point


# @torch.compile
def _cast_ray_to_box(
    box: Entity,
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
    max_range: float,
):
    """Inspired from https://tavianator.com/2011/ray_box.html
    Computes distance of ray originating from pos at angle to a box and sets distance to
    max_range if there is no intersection.
    """
    assert isinstance(box.shape, Box)

    pos_origin = ray_origin - box.state.pos
    pos_aabb = rotate_vector(pos_origin, -box.state.rot)[:, :, None, :].repeat(
        1, 1, ray_direction.shape[1], 1
    )
    ray_dir_world = torch.stack(
        [torch.cos(ray_direction), torch.sin(ray_direction)], dim=-1
    )

    ray_dir_aabb = rotate_vector(ray_dir_world, -box.state.rot)

    tx1 = (-box.shape.length / 2 - pos_aabb[..., X]) / ray_dir_aabb[..., X]
    tx2 = (box.shape.length / 2 - pos_aabb[..., X]) / ray_dir_aabb[..., X]
    tx = torch.stack([tx1, tx2], dim=-1)
    tmin, _ = torch.min(tx, dim=-1)
    tmax, _ = torch.max(tx, dim=-1)

    ty1 = (-box.shape.width / 2 - pos_aabb[..., Y]) / ray_dir_aabb[..., Y]

    ty2 = (box.shape.width / 2 - pos_aabb[..., Y]) / ray_dir_aabb[..., Y]

    ty = torch.stack([ty1, ty2], dim=-1)
    tymin, _ = torch.min(ty, dim=-1)
    tymax, _ = torch.max(ty, dim=-1)
    tmin, _ = torch.max(torch.stack([tmin, tymin], dim=-1), dim=-1)
    tmax, _ = torch.min(torch.stack([tmax, tymax], dim=-1), dim=-1)

    intersect_aabb = tmin.unsqueeze(tmin.ndim) * ray_dir_aabb + pos_aabb

    assert box.state.pos is not None
    assert box.state.rot is not None

    intersect_world = rotate_vector(
        intersect_aabb, box.state.rot.reshape(1, box.state.rot.shape[0], 1, 1)
    ) + box.state.pos.reshape(1, box.state.pos.shape[0], 1, box.state.pos.shape[1])

    collision = (tmax >= tmin) & (tmin > ZERO)

    dist = torch.linalg.norm(
        ray_origin[:, None, :].repeat(1, ray_direction.shape[1], 1) - intersect_world,
        dim=-1,
    )

    new_dist = dist.clone().squeeze(0).squeeze(0)

    new_dist[~collision.squeeze(0)] = max_range

    return new_dist

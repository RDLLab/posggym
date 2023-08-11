"""Core functionality for continuous environments."""
from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from itertools import product
from queue import PriorityQueue
from typing import Dict, Iterable, List, NamedTuple, Set, Tuple, Union

import numpy as np
from gymnasium import spaces

from posggym.error import DependencyNotInstalled

try:
    import pymunk
    from pymunk import Vec2d
except ImportError as e:
    raise DependencyNotInstalled(
        "pymunk is not installed, run `pip install posggym[continuous]`"
    ) from e


# (x, y) coord = (col, row) coord
Coord = Tuple[int, int]
FloatCoord = Tuple[float, float]
# (x, y, yaw) in continuous world
Position = Tuple[float, float, float]
Location = Union[Coord, FloatCoord, Position, np.ndarray]
# Position, radius
CircleEntity = Tuple[Position, float]
# start (x, y), end (x, y)
Line = Tuple[FloatCoord, FloatCoord]
IntLine = Tuple[Tuple[int, int], Tuple[int, int]]

# (Main, Alternative) agent colors (from pygame.colordict.THECOLORS)
AGENT_COLORS = [
    ((255, 0, 0, 255), (205, 0, 0, 255)),  # red, red3
    ((0, 0, 255, 255), (0, 0, 205, 255)),  # blue, blue3
    ((0, 255, 0, 255), (0, 205, 0, 255)),  # green, green3
    ((160, 32, 240, 255), (125, 38, 205, 255)),  # purple, purple3
    ((255, 255, 0, 255), (205, 205, 0, 255)),  # yellow, yellow3
    ((0, 255, 255, 255), (0, 205, 205, 255)),  # cyan, cyan3
    ((255, 105, 180, 255), (205, 96, 144, 255)),  # hotpink, hotpink3
    ((255, 165, 0, 255), (205, 133, 0, 255)),  # orange, orange3
]


class CollisionType(Enum):
    """Type of collision in world."""

    NONE = 0
    AGENT = 1
    BLOCK = 2
    BORDER = 3
    INTERIOR_WALL = 4


class PMBodyState(NamedTuple):
    """State of a Pymunk Body."""

    x: float
    y: float
    angle: float  # in radians
    vx: float
    vy: float
    vangle: float  # in radians/s

    @staticmethod
    def num_features() -> int:
        """Get the number of features of a pymunk body state."""
        return 6

    @staticmethod
    def get_space(world_size: float) -> spaces.Box:
        """Get the space for a pymunk body's state."""
        # x, y, angle, vx, vy, vangle
        # shape = (1, 6)
        size, angle = world_size, 2 * math.pi
        low = np.array([-1, -1, -angle, -1, -1, -angle], dtype=np.float32)
        high = np.array(
            [size, size, angle, 1.0, 1.0, angle],
            dtype=np.float32,
        )
        return spaces.Box(low=low, high=high)


# This function needs to be in global scope or we get pickle errors
def ignore_collisions(arbiter, space, data):
    """Pymunk collision handler which ignores collisions."""
    return False


def clip_actions(
    actions: Dict[str, np.ndarray], action_spaces: Dict[str, spaces.Space]
) -> Dict[str, np.ndarray]:
    """Clip continuous actions so they are within the agents action space dims."""
    clipped_actions = {}
    for i, a in actions.items():
        a_space = action_spaces[i]
        assert isinstance(a_space, spaces.Box)
        clipped_actions[i] = np.clip(a, a_space.low, a_space.high)
    return clipped_actions


class AbstractContinuousWorld(ABC):
    """A continuous 2D world with a rectangular border."""

    WALL_COLOR = (0, 0, 0, 255)  # Black
    BLOCK_COLOR = (50, 50, 50, 255)  # Black

    def __init__(
        self,
        size: float,
        blocks: List[CircleEntity] | None = None,
        interior_walls: List[Line] | None = None,
        agent_radius: float = 0.5,
        border_thickness: float = 0.1,
        enable_agent_collisions: bool = True,
    ):
        self.size = size
        self.blocks = blocks or []
        self.interior_walls = interior_walls or []
        self.agent_radius = agent_radius
        self.border_thickness = border_thickness
        # access via blocked_coords property
        self._blocked_coords: Set[Coord] | None = None

        self.collision_id = 0
        self.enable_agent_collisions = enable_agent_collisions

        # 2D physics stuff
        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0.0, 0.0)

        if not enable_agent_collisions:
            # Turn off all collisions
            self.space.add_collision_handler(0, 0).pre_solve = ignore_collisions

        self.add_border_to_space(size)
        self.add_interior_walls_to_space(self.interior_walls)

        for pos, radius in self.blocks:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            shape = pymunk.Circle(body, radius)
            body.position = Vec2d(pos[0], pos[1])
            shape.elasticity = 0.0  # no bouncing
            shape.color = self.BLOCK_COLOR
            shape.collision_type = self.get_collision_id()
            self.space.add(body, shape)

        # moveable entities in the world
        self.entities: Dict[str, Tuple[pymunk.Body, pymunk.Circle]] = {}

    @abstractmethod
    def add_border_to_space(self, size: float):
        """Adds solid border to the world physics space."""
        pass

    @abstractmethod
    def check_border_collisions(
        self, ray_start_coords: np.ndarray, ray_end_coords: np.ndarray
    ) -> np.ndarray:
        """Check for collision between rays and world border."""
        pass

    @abstractmethod
    def clip_position(self, position: Vec2d) -> Vec2d:
        """Clip the position of an agent to be inside the border"""
        pass

    @abstractmethod
    def copy(self) -> "AbstractContinuousWorld":
        """Get a deep copy of this world."""
        pass

    def simulate(
        self,
        dt: float = 1.0 / 10,
        t: int = 10,
        normalize_angles: bool = True,
    ):
        """Simulate the world, updating all entities.

        As per pymunk.Space.step documentation, using a fixed time step `dt` is
        recommended and improves efficiency.

        Also performing multiple steps `t` with a smaller `dt` creates a more stable
        and accurate simulation.

        Arguments
        ---------
        dt: the step size
        t: the number of steps
        normalize_angles: whether to normalize angle of each entity to be in [0, 2*pi]
            at the end of the simulation.

        Reference
        ---------
        https://www.pymunk.org/en/latest/pymunk.html#pymunk.Space.step

        """
        for _ in range(t):
            self.space.step(dt)

        if normalize_angles:
            for body, _ in self.entities.values():
                body.angle = self.convert_angle_to_0_2pi_interval(body.angle)
                body.angular_velocity = self.convert_angle_to_0_2pi_interval(
                    body.angular_velocity
                )

    def add_entity(
        self,
        id: str,
        radius: float | None,
        color: Tuple[int, int, int, int] | None,
        is_static: bool = False,
    ) -> Tuple[pymunk.Body, pymunk.Circle]:
        """Add moveable entity to the world.

        Arguments
        ---------
        id: the unique ID of the entity
        radius: the radius of the entity. If none uses `self.agent_radius`
        color: optional color for the entity. This only impacts rendering of the world.

        Returns
        -------
        body: underlying physics Body of the entity
        shape: the shape of the entity

        """
        if radius is None:
            radius = self.agent_radius
        mass = 1.0
        inertia = pymunk.moment_for_circle(mass, 0.0, radius)
        body_type = pymunk.Body.STATIC if is_static else pymunk.Body.DYNAMIC
        body = pymunk.Body(mass, inertia, body_type=body_type)
        shape = pymunk.Circle(body, radius)

        shape.collision_type = self.get_collision_id()

        shape.elasticity = 0.0  # no bouncing
        shape.collision_type = self.get_collision_id()
        if color is not None:
            shape.color = color

        self.space.add(body, shape)
        self.entities[id] = (body, shape)
        return body, shape

    def remove_entity(self, id: str):
        """Remove entity from the world."""
        body, shape = self.entities[id]
        self.space.remove(body, shape)
        del self.entities[id]

    def add_interior_walls_to_space(self, walls: List[Line]):
        """Adds interior walls to the world physics space."""
        self.interior_walls_array = (
            np.array([ln[0] for ln in walls], dtype=np.float32),
            np.array([ln[1] for ln in walls], dtype=np.float32),
        )

        for w_start, w_end in walls:
            wall = pymunk.Segment(
                self.space.static_body,
                w_start,
                w_end,
                self.border_thickness,
            )
            wall.friction = 1.0
            wall.collision_type = self.get_collision_id() + 1
            wall.color = self.WALL_COLOR
            self.space.add(wall)

    def get_collision_id(self):
        return self.collision_id

    def get_entity_state(self, id: str) -> PMBodyState:
        """Get underlying state of an entity in the world."""
        body, _ = self.entities[id]
        x, y = body.position
        vx, vy = body.velocity
        return PMBodyState(x, y, body.angle, vx, vy, body.angular_velocity)

    def set_entity_state(self, id: str, state: PMBodyState | np.ndarray):
        """Set the state of an entity.

        If state is a np.ndarray, then it should be 1D with values ordered the same as
        the PMBodyState namedtuple.
        """
        body, _ = self.entities[id]
        body.position = Vec2d(state[0], state[1])
        body.angle = state[2]
        body.velocity = Vec2d(state[3], state[4])
        body.angular_velocity = state[5]

    def update_entity_state(
        self,
        id: str,
        *,
        coord: FloatCoord | List[float] | np.ndarray | Vec2d | None = None,
        angle: float | None = None,
        vel: FloatCoord | List[float] | np.ndarray | Vec2d | None = None,
        vangle: float | None = None,
    ):
        """Update the state of an entity.

        Can be used to update specific parts of the entity state, while leaving the
        other components unchanged.

        """
        body, _ = self.entities[id]
        if coord is not None:
            body.position = Vec2d(coord[0], coord[1])

        if angle is not None:
            body.angle = angle

        if vel is not None:
            body.velocity = Vec2d(vel[0], vel[1])

        if vangle is not None:
            body.angular_velocity = vangle

    def get_bounds(self) -> Tuple[FloatCoord, FloatCoord]:
        """Get  (min x, max_x), (min y, max y) bounds of the world."""
        return (0, self.size), (0, self.size)

    @property
    def blocked_coords(self) -> Set[Coord]:
        """The set of all integer coordinates that contain at least part of a block."""
        if self._blocked_coords is None:
            self._blocked_coords = set()
            # offset from coord, to middle of cell
            offset = 0.5
            # to handle any weird rounding
            max_coord = math.ceil(round(self.size, 6))
            for x, y in product(list(range(max_coord)), repeat=2):
                offset_coord = (x + offset, y + offset)

                # check if cell contains any block
                for pos, r in self.blocks:
                    if self.euclidean_dist(offset_coord, pos) < (offset + r):
                        self._blocked_coords.add((x, y))
                        break

                # check if cell contains any wall
                if len(self.interior_walls):
                    dists = self.check_circle_line_intersection(
                        np.array([[x, y]]),
                        np.array([offset]),
                        *self.interior_walls_array,
                    )
                    if np.any(np.where(dists < offset, True, False)):
                        self._blocked_coords.add((x, y))
                        break

        return self._blocked_coords

    @staticmethod
    def manhattan_dist(loc1: Location, loc2: Location) -> float:
        """Get manhattan distance between two positions in the world."""
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    @staticmethod
    def euclidean_dist(loc1: Location, loc2: Location) -> float:
        """Get Euclidean distance between two positions on the grid."""
        return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

    @staticmethod
    def squared_euclidean_dist(loc1: Location, loc2: Location) -> float:
        """Get Squared Euclidean distance between two positions on the grid."""
        return (loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2

    @staticmethod
    def convert_angle_to_0_2pi_interval(angle: float) -> float:
        return angle % (2 * math.pi)

    @staticmethod
    def convert_angle_to_negpi_pi_interval(angle: float) -> float:
        """Convert angle in radians to be in (-pi, pi] interval."""
        angle = angle % (2 * math.pi)
        if angle > math.pi:
            angle -= 2 * math.pi
        return angle

    @staticmethod
    def array_to_position(arr: np.ndarray) -> Position:
        """Convert from numpy array to tuple representation of a Position."""
        assert arr.shape[0] >= 3
        return (arr[0], arr[1], arr[2])

    @staticmethod
    def linear_to_xy_velocity(linear_vel: float, angle: float) -> Vec2d:
        """Convert from linear velocity to velocity along x and y axis."""
        return linear_vel * Vec2d(1, 0).rotated(angle)

    @staticmethod
    def rotate_vector(vx: float, vy: float, angle: float) -> Vec2d:
        """Rotate a 2D vector by given angle."""
        return Vec2d(vx, vy).rotated(angle)

    @staticmethod
    def clamp_norm(vx: float, vy: float, norm_max: float) -> Tuple[float, float]:
        """Clamp x, y vector to within a given max norm."""
        if vx == 0.0 and vy == 0.0:
            return vx, vy
        norm = math.sqrt(vx**2 + vy**2)
        f = min(norm, norm_max) / norm
        return f * vx, f * vy

    @staticmethod
    def convert_into_interval(
        x: float | np.ndarray,
        x_min: float | np.ndarray,
        x_max: float | np.ndarray,
        new_min: float | np.ndarray,
        new_max: float | np.ndarray,
        clip: bool = False,
    ) -> float | np.ndarray:
        """Convert variable from [x_min, x_max] to [new_min, new_max] interval."""
        new_x = (x - x_min) / (x_max - x_min) * (new_max - new_min) + new_min
        if clip:
            return np.clip(new_x, new_min, new_max)
        return new_x

    def agents_collide(self, loc1: Location, loc2: Location) -> bool:
        """Get whether two agents have collided or not.

        Assumes agents are circles with radius `self.agent_radius`.
        """
        dist = self.squared_euclidean_dist(loc1, loc2)
        return 0 <= dist <= (self.agent_radius + self.agent_radius) ** 2

    def check_circle_line_intersection(
        self,
        circle_coords: np.ndarray,
        circle_radii: np.ndarray,
        lines_start_coords: np.ndarray,
        lines_end_coords: np.ndarray,
    ) -> np.ndarray:
        """Check if lines intersect circles.

        Arguments
        ---------
        circle_coords: array containing the `(x, y)` of the center of each circle.
            Should have shape `(n_circles, 2)`
        circle_radii: array containing the radius of each circle. Should have shape
            `(n_circles,)`
        lines_start_coords: array containing the `(x, y)` coords of the start of each of
            the lines. Should have shape `(n_lines, 2)`
        lines_end_coords: array containing the `(x, y)` coords of the end of of each of
            the lines. Should have shape `(n_lines, 2)`

        Returns
        -------
        distances: An array containing the euclidean distance from each lines start to
            the first point of intersection with the corresponding circle. If the line
            does not intersect the circle, its distance will be `np.nan`.
            Should have shape `(n_circles, n_lines)`

        """
        starts = lines_start_coords[None, :, :] - circle_coords[:, None, :]
        ends = lines_end_coords[None, :, :] - circle_coords[:, None, :]
        v = ends - starts

        a = (v**2).sum(axis=2)
        b = 2 * (starts * v).sum(axis=2)
        c = (starts**2).sum(axis=2) - circle_radii[:, None] ** 2
        disc = b**2 - 4 * a * c

        with np.errstate(invalid="ignore"):
            # this will output np.nan for any negative discriminants, which is what we
            # want, but will throw a runtime warning which we want to ignore
            sqrtdisc = np.sqrt(disc)

        t1 = (-b - sqrtdisc) / (2 * a)
        t2 = (-b + sqrtdisc) / (2 * a)
        t1 = np.where(((t1 >= 0.0) & (t1 <= 1.0)), t1, np.nan)
        t2 = np.where(((t2 >= 0.0) & (t2 <= 1.0)), t2, np.nan)
        t = np.where(t1 <= t2, t1, t2)

        t = np.expand_dims(t, axis=-1)
        return np.sqrt(((t * v) ** 2).sum(axis=2))  # type: ignore

    def check_line_line_intersection(
        self,
        l1_start_coords: np.ndarray,
        l1_end_coords: np.ndarray,
        l2_start_coords: np.ndarray,
        l2_end_coords: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Check if lines intersect.

        Checks for each line in `l1` if it intersects with any line in `l2`.

        Arguments
        ---------
        l1_start_coords: array with shape `(n_lines1, 2)` containing the (x, y) coord
          for the start of each of the first set of lines.
        l1_end_coords: array with shape `(n_lines1, 2)` containing the (x, y) coord
          for the end of each of the first set of lines.
        l2_start_coords: array with shape `(n_lines2, 2)` containing the (x, y) coord
          for the start of each of the second set of lines.
        l2_end_coords: array with shape `(n_lines2, 2)` containing the (x, y) coord
          for the end of each of the second set of lines.

        Returns
        -------
        intersection_coords: array with shape `(n_lines1, n_lines2, 2)` containing the
           (x, y) coords for the point of intersection between each pair of line
           segments from `l1` and `l2`. If a pair of lines does not intersect, the
           x and y coordinate values will be `np.nan`.
        distances: array with shape `(n_lines1, n_lines2)` containing the distances
            between the intersection point (if exists) and the corresponding point in
            l1_start_cooords.  If a pair of lines does not intersect, the distance value
            will be `np.nan`.

        """
        # https://stackoverflow.com/questions/3252194/numpy-and-line-intersections?noredirect=1&lq=1
        # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        dl1 = l1_end_coords - l1_start_coords  # (n_lines1, 2)
        dl2 = l2_end_coords - l2_start_coords  # (n_lines2, 2)

        # diff between each l1 with each l2 along x and y directions
        # (n_lines1, :, 2) - (:, n_lines2, 2) = (n_lines1, n_lines2, 2)
        dl1l2 = l1_start_coords[:, np.newaxis, :] - l2_start_coords[np.newaxis, :, :]
        # needed for matmul (n_lines1, 2, n_lines2)
        dl1l2_T = dl1l2.transpose(0, 2, 1)
        # needed for matmul (n_lines2, 2, n_lines1)
        dl1l2_T2 = dl1l2.transpose(1, 2, 0)

        # line perpendicular to each l1 line
        dl1p = np.empty_like(dl1)  # (n_lines1, 2)
        dl1p[:, 0] = -dl1[:, 1]
        dl1p[:, 1] = dl1[:, 0]

        # line perpendicular to each l2 line
        dl2p = np.empty_like(dl2)  # (n_lines1, 2)
        dl2p[:, 0] = -dl2[:, 1]
        dl2p[:, 1] = dl2[:, 0]

        # mult (n_lines1, 2) @ (n_lines1, 2, nlines2) = (n_lines1, n_lines2)
        # each i in n_lines1 is multiplied with one of the n_lines1 matrices in dl1l2
        # l1[i] @ (l1[i] - l2[j]) for i in [0, n_lines1], j in [0, n_lines2]
        u_num = np.stack([np.matmul(dl1p[i], dl1l2_T[i]) for i in range(dl1p.shape[0])])

        # mult (n_lines2, 2) @ (n_lines2, 2, nlines1) = (n_lines2, n_lines1)
        # same as above except for l2 lines
        t_num = np.stack(
            [np.matmul(dl2p[j], dl1l2_T2[j]) for j in range(dl2p.shape[0])]
        )

        # mult (n_lines1, 2) @ (2, n_lines2) = (n_lines1, n_lines2)
        # get l1[i] dot l2[j] for i in [0, n_lines1], j in [0, n_lines2]
        # but using perpendicular lines to l1,
        denom = np.matmul(dl1p, dl2.transpose())
        # handle case where lines are parallel/colinear, leading to denom being zero
        denom = np.where(np.isclose(denom, 0.0), np.nan, denom)

        # from wiki, lines only intersect if 0 <= u <= 1
        u = u_num / denom  # (n_lines1, n_lines2)
        t = t_num.transpose() / denom  # (n_lines1, n_lines2)

        # segments intersect when 0 <= u <= 1 and 0 <= t <= 1
        u = np.where(((u >= 0) & (u <= 1) & (t >= 0) & (t <= 1)), u, np.nan)
        u = u[:, :, np.newaxis]  # (n_lines1, n_lines2, 1)

        intersection_coords = u * dl2 + l2_start_coords
        intersection_distances = np.linalg.norm(
            intersection_coords - l1_start_coords[:, np.newaxis], axis=2
        )

        return intersection_coords, intersection_distances

    def check_ray_collisions(
        self,
        ray_start_coords: np.ndarray,
        ray_end_coords: np.ndarray,
        max_ray_distance: float,
        other_agents: np.ndarray | None = None,
        include_blocks: bool = True,
        check_walls: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Check for collision along rays.

        Arguments
        ---------
        ray_start_coords: start coords of rays. Should be 2D array with shape
           `(n_rays, 2`),
        ray_end_coords: end coords of rays. Should be 2D array with shape
           `(n_rays, 2`),
        max_ray_distance: maximum ray distance
        other_agents: `(x, y)` coordinates of any other agents to check for collisions
           with. Should be a 2D array with shape `(n_agents, 2)`.
        include_blocks: whether to check for collisions with blocks in the world
        check_walls: whether to check for collisions with the world border.

        Returns
        -------
        distances: the distance each ray extends sway from the origin, up to a max of
            `max_ray_distance`. Array will have shape `(n_rays,)`.
        collision_types: the type of collision for each ray (see CollisionType), if any.
            Will have shape `(n_rays,)`.

        """
        n_rays = len(ray_start_coords)
        closest_distances = np.full(n_rays, max_ray_distance, dtype=np.float32)
        collision_types = np.full(n_rays, CollisionType.NONE.value, dtype=np.uint8)

        if other_agents is not None and len(other_agents):
            radii = np.array([self.agent_radius] * len(other_agents))
            dists = self.check_circle_line_intersection(
                other_agents, radii, ray_start_coords, ray_end_coords
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                min_dists = np.nanmin(dists, axis=0)

            collision_types[min_dists < closest_distances] = CollisionType.AGENT.value
            np.fmin(closest_distances, min_dists, out=closest_distances)

        if include_blocks and len(self.blocks):
            radii = np.array([s for _, s in self.blocks])
            block_array = np.array([[pos[0], pos[1]] for pos, _ in self.blocks])

            dists = self.check_circle_line_intersection(
                block_array, radii, ray_start_coords, ray_end_coords
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                min_dists = np.nanmin(dists, axis=0)

            collision_types[min_dists < closest_distances] = CollisionType.BLOCK.value
            np.fmin(closest_distances, min_dists, out=closest_distances)

        if check_walls:
            # shape = (n_lines, walls, 2)
            wall_intersect_coords = self.check_border_collisions(
                ray_start_coords, ray_end_coords
            )

            # Need to get coords of intersected walls, each ray can intersect a max of
            # of 1 wall, so we just find the minimum non nan coords
            # shape = (n_lines, 2)
            with warnings.catch_warnings():
                # if no wall intersected, we take min of all NaN which throws a warning
                # but this is acceptable behevaiour, so we suppress the warning
                warnings.simplefilter("ignore")
                wall_intersect_coords = np.nanmin(wall_intersect_coords, axis=1)

            dists = np.sqrt(
                ((wall_intersect_coords - ray_start_coords) ** 2).sum(axis=1)
            )

            collision_types[dists < closest_distances] = CollisionType.BORDER.value
            np.fmin(closest_distances, dists, out=closest_distances)

        if check_walls and len(self.interior_walls):
            _, distances = self.check_line_line_intersection(
                ray_start_coords, ray_end_coords, *self.interior_walls_array
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dists = np.nanmin(distances, axis=1)

            collision_types[
                dists < closest_distances
            ] = CollisionType.INTERIOR_WALL.value
            np.fmin(closest_distances, dists, out=closest_distances)

        return closest_distances, collision_types

    def check_collision_circular_rays(
        self,
        origin: Position,
        ray_distance: float,
        n_rays: int,
        other_agents: np.ndarray | None = None,
        include_blocks: bool = True,
        check_walls: bool = True,
        use_relative_angle: bool = True,
        angle_bounds: Tuple[float, float] = (0.0, 2 * np.pi),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Check for collision along rays that radiate away from the origin.

        Rays are evenly spaced around the origin, with the number of rays controlled
        by the `n_rays` arguments.

        Arguments
        ---------
        origin: the origin position
        ray_distance: how far each ray extends from the origin in a straight line
        n_rays: the number of rays
        other_agents: `(x, y)` coordinates of any other agents to check for collisions
           with. Should be a 2D array with shape `(n_agents, 2)`.
        include_blocks: whether to check for collisions with blocks in the world
        check_walls: whether to check for collisions with the world border.
        use_relative_angle: whether ray angles should be relative to the origin
            position's angle. Otherwise line angle is treated as absolute (i.e.
            relative to angle of 0). This controls the ordering of the rays in the
            output.
        angle_bounds: The maximum and minimum of the FOV as a tuple. By default
            the agent will have a full FOV as a circle around them. This will
            be between 0 and 2π. This can be decreased as needed.

        Returns
        -------
        distances: the distance each ray extends sway from the origin, up to a max of
            `ray_distance`. Array will have shape `(n_rays,)`.

        """
        x, y, rel_angle = origin
        if not use_relative_angle:
            rel_angle = 0.0

        angles = np.linspace(
            angle_bounds[0], angle_bounds[1], n_rays, endpoint=False, dtype=np.float32
        )

        ray_end_xs = x + ray_distance * np.cos(angles + rel_angle)
        ray_end_ys = y + ray_distance * np.sin(angles + rel_angle)

        ray_start_coords = np.tile((x, y), (n_rays, 1))
        ray_end_coords = np.stack([ray_end_xs, ray_end_ys], axis=1)

        return self.check_ray_collisions(
            ray_start_coords,
            ray_end_coords,
            max_ray_distance=ray_distance,
            other_agents=other_agents,
            include_blocks=include_blocks,
            check_walls=check_walls,
        )

    def get_all_shortest_paths(
        self, origins: Iterable[FloatCoord | Coord | Position]
    ) -> Dict[Tuple[int, int], Dict[Tuple[int, int], int]]:
        """Get shortest path distance from every origin to all other coords."""
        src_dists = {}
        for origin in origins:
            origin_coord = self.convert_to_coord(origin)
            src_dists[origin_coord] = self.dijkstra(origin)
        return src_dists

    def convert_to_coord(self, origin: FloatCoord | Coord | Position) -> Coord:
        """Convert a position/float coord to a integer coords."""
        return (math.floor(origin[0]), math.floor(origin[1]))

    def dijkstra(self, origin: FloatCoord | Coord | Position) -> Dict[Coord, int]:
        """Get shortest path distance between origin and all other coords."""
        coord_origin = self.convert_to_coord(origin)

        dist = {coord_origin: 0}
        pq: PriorityQueue[Tuple[int, Coord]] = PriorityQueue()
        pq.put((dist[coord_origin], coord_origin))

        visited = {coord_origin}

        while not pq.empty():
            _, coord = pq.get()
            for adj_coord in self.get_cardinal_neighbours_coords(
                coord, ignore_blocks=False
            ):
                if dist[coord] + 1 < dist.get(adj_coord, float("inf")):
                    dist[adj_coord] = dist[coord] + 1

                    if adj_coord not in visited:
                        pq.put((dist[adj_coord], adj_coord))
                        visited.add(adj_coord)
        return dist

    def get_cardinal_neighbours_coords(
        self,
        coord: Coord,
        ignore_blocks: bool = False,
        include_out_of_bounds: bool = False,
    ) -> List[Coord]:
        """Get set of adjacent non-blocked coords."""
        (min_x, max_x), (min_y, max_y) = self.get_bounds()
        neighbours = []

        if coord[1] > min_y or include_out_of_bounds:
            neighbours.append((coord[0], coord[1] - 1))  # N
        if coord[0] < max_x - 1 or include_out_of_bounds:
            neighbours.append((coord[0] + 1, coord[1]))  # E
        if coord[1] < max_y or include_out_of_bounds:
            neighbours.append((coord[0], coord[1] + 1))  # S
        if coord[0] > min_x or include_out_of_bounds:
            neighbours.append((coord[0] - 1, coord[1]))  # W

        if ignore_blocks:
            return neighbours
        for i in range(len(neighbours), 0, -1):
            if neighbours[i - 1] in self.blocked_coords:
                neighbours.pop(i - 1)

        return neighbours


class SquareContinuousWorld(AbstractContinuousWorld):
    """A continuous world with a square border."""

    def copy(self) -> "SquareContinuousWorld":
        world = SquareContinuousWorld(
            size=self.size,
            blocks=self.blocks,
            interior_walls=self.interior_walls,
            agent_radius=self.agent_radius,
            border_thickness=self.border_thickness,
        )

        for id, (body, shape) in self.entities.items():
            # make copies of each entity, and ensure the copies are linked correctly
            # and added to the new world and world space
            body = body.copy()
            shape = shape.copy()
            shape.body = body

            world.space.add(body, shape)
            world.entities[id] = (body, shape)
        return world

    def add_border_to_space(self, size: float):
        # world border lines (start coords, end coords)
        # bottom, left, top, right
        self.border = (
            np.array([[0, 0], [0, 0], [0, size], [size, 0]], dtype=np.float32),
            np.array(
                [[size, 0], [0, size], [size, size], [size, size]],
                dtype=np.float32,
            ),
        )

        for w_start, w_end in zip(*self.border):
            wall = pymunk.Segment(
                self.space.static_body,
                (w_start[0], w_start[1]),
                (w_end[0], w_end[1]),
                self.border_thickness,
            )
            wall.friction = 1.0
            wall.collision_type = self.get_collision_id() + 1
            wall.color = self.WALL_COLOR
            self.space.add(wall)

    def clip_position(self, position: Vec2d) -> Vec2d:
        return Vec2d(
            *np.clip([position[0], position[1]], [0, 0], [self.size, self.size])
        )

    def check_border_collisions(
        self, ray_start_coords: np.ndarray, ray_end_coords: np.ndarray
    ) -> np.ndarray:
        return self.check_line_line_intersection(
            ray_start_coords, ray_end_coords, *self.border
        )[0]


class CircularContinuousWorld(AbstractContinuousWorld):
    """A 2D continuous world with a circular border."""

    def copy(self) -> "CircularContinuousWorld":
        world = CircularContinuousWorld(
            size=self.size,
            blocks=self.blocks,
            interior_walls=self.interior_walls,
            agent_radius=self.agent_radius,
            border_thickness=self.border_thickness,
        )
        for id, (body, shape) in self.entities.items():
            # make copies of each entity, and ensure the copies are linked correctly
            # and added to the new world and world space
            body = body.copy()
            shape = shape.copy()
            shape.body = body
            world.space.add(body, shape)
            world.entities[id] = (body, shape)
        return world

    def add_border_to_space(self, size: float):
        num_segments = 180
        radius = size / 2
        segment_angle_inc = 2 * math.pi / num_segments

        self.border = []
        for i in range(num_segments):
            prev_angle = i * segment_angle_inc
            angle = (i + 1) * segment_angle_inc
            x1 = radius * math.cos(prev_angle) + radius
            y1 = radius * math.sin(prev_angle) + radius
            x2 = radius * math.cos(angle) + radius
            y2 = radius * math.sin(angle) + radius
            wall = pymunk.Segment(
                self.space.static_body,
                (x1, y1),
                (x2, y2),
                self.border_thickness,
            )
            wall.friction = 0.0
            wall.color = self.WALL_COLOR
            wall.collision_type = self.get_collision_id() + 1
            self.border.append(wall)

        for wall in self.border:
            self.space.add(wall)

        self.size = size

    def check_border_collisions(
        self, ray_start_coords: np.ndarray, ray_end_coords: np.ndarray
    ) -> np.ndarray:
        center = np.array([0, 0]).reshape(1, 2)
        return self.check_circle_line_intersection(
            center, np.array([self.size]), ray_start_coords, ray_end_coords
        )

    def clip_position(self, position: Vec2d) -> Vec2d:
        x, y = position.x, position.y
        r_max = self.size / 2
        radius = np.linalg.norm([x - r_max, y - r_max])
        gamma = math.atan2(y, x)
        if radius > r_max:
            x = r_max * (1 + math.cos(gamma))
            y = r_max * (1 + math.sin(gamma))

        return Vec2d(x, y)


def generate_interior_walls(
    width: int, height: int, blocked_coords: Iterable[Coord]
) -> List[Line]:
    """Generate interior walls for rectangular world based on blocked coordinates."""
    # dx, dy
    # north, east, south, west
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    line_offsets = [
        ((0, 0), (1, 0)),
        ((1, 0), (1, 1)),
        ((0, 1), (1, 1)),
        ((0, 0), (0, 1)),
    ]

    # get line for each block face adjacent to empty cell
    lines_map: Dict[Coord, Set[Coord]] = {}
    lines: Set[IntLine] = set()
    for x, y in blocked_coords:
        for (dx, dy), line_offset in zip(directions, line_offsets):
            if (x + dx, y + dy) in blocked_coords:
                # adjacent cell blocked
                continue

            s_offset, e_offset = line_offset
            l_start = (x + s_offset[0], y + s_offset[1])
            l_end = (x + e_offset[0], y + e_offset[1])
            if (
                (l_start[0] == 0 and l_end[0] == 0)
                or (l_start[0] == width and l_end[0] == width)
                or (l_start[1] == 0 and l_end[1] == 0)
                or (l_start[1] == height and l_end[1] == height)
            ):
                # line is along border
                continue

            if l_start not in lines_map:
                lines_map[l_start] = set()
            lines_map[l_start].add(l_end)
            lines.add((l_start, l_end))

    # merge lines
    merged_lines: List[IntLine] = []

    # lines l1 and l2 can merge if
    # 1. l1[1] == l2[0] and (l1[0][0] == l2[1][0] or l1[0][1] == l2[1][1]
    # or vice versa with l1 and l2 swapped
    # Assumes lines are parallel to either x or y axis

    # Basically a DFS
    stack = list(lines)
    stack.sort(reverse=True)

    visited: Set[IntLine] = set()
    while len(stack):
        line = stack.pop()
        if line in visited:
            continue

        visited.add(line)
        l_start, l_end = line
        line_extended = True
        while l_end in lines_map and line_extended:
            line_extended = False
            for l_end_next in lines_map[l_end]:
                if l_end_next != l_start and (
                    l_start[0] == l_end_next[0] or l_start[1] == l_end_next[1]
                ):
                    # merge
                    visited.add((l_end, l_end_next))
                    visited.add((l_end_next, l_end))
                    l_end = l_end_next
                    line_extended = True

        merged_lines.append((l_start, l_end))

    return [
        ((float(ln[0][0]), float(ln[0][1])), (float(ln[1][0]), float(ln[1][1])))
        for ln in merged_lines
    ]

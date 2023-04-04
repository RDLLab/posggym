"""Core functionality for continuous environments."""
from __future__ import annotations

from itertools import product
import math
from queue import PriorityQueue
import warnings
from typing import (
    Dict,
    Iterable,
    List,
    NamedTuple,
    Set,
    Tuple,
    Union,
    TYPE_CHECKING,
    Optional,
)

import numpy as np
from gymnasium import spaces

from posggym.error import DependencyNotInstalled

if TYPE_CHECKING:
    import posggym.model as M

try:
    import pymunk
    from pymunk import Vec2d
except ImportError as e:
    raise DependencyNotInstalled(
        "pymunk is not installed, run `pip install posggym[continuous]`"
    ) from e

from enum import Enum


class CollisionType(Enum):
    NO_COLLISION = 0
    AGENT_COLLISION = 1
    BLOCK_COLLISION = 2
    BORDER_COLLISION = 3


# (x, y) coord = (col, row) coord
Coord = Tuple[int, int]
# (x, y, yaw) in continuous world
Position = Tuple[float, float, float]
Location = Union[Coord, Position, np.ndarray, Tuple[float, float]]
# Position, radius
CircleEntity = Tuple[Position, float]


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


def clip_actions(
    actions: Dict[M.AgentID, np.ndarray], action_spaces: Dict[M.AgentID, spaces.Space]
) -> Dict[M.AgentID, np.ndarray]:
    """Clip continuous actions so they are within the agents action space dims."""
    clipped_actions = {}
    for i, a in actions.items():
        a_space = action_spaces[i]
        assert isinstance(a_space, spaces.Box)
        clipped_actions[i] = np.clip(a, a_space.low, a_space.high)

    return clipped_actions


class SquareContinuousWorld:
    """A continuous 2D world with a rectangular border."""

    WALL_COLOR = (0, 0, 0, 255)  # Black
    BLOCK_COLOR = (0, 0, 0, 255)  # Black

    def __init__(
        self,
        size: float,
        blocks: Optional[List[CircleEntity]] = None,
        agent_radius: float = 0.5,
        border_thickness=0.1,
    ):
        self.size = size
        self.blocks = [] if blocks is None else blocks
        self.agent_radius = agent_radius
        self.border_thickness = border_thickness
        # access via blocked_coords property
        self._blocked_coords: Optional[Set[Coord]] = None

        # world border lines (start coords, end coords)
        # bottom, left, top, right
        self.walls = (
            np.array([[0, 0], [0, 0], [0, size], [size, 0]], dtype=np.float32),
            np.array(
                [[size, 0], [0, size], [size, size], [size, size]],
                dtype=np.float32,
            ),
        )

        # 2D physics stuff
        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0.0, 0.0)

        for w_start, w_end in zip(*self.walls):
            wall = pymunk.Segment(
                self.space.static_body,
                (w_start[0], w_start[1]),
                (w_end[0], w_end[1]),
                self.border_thickness,
            )
            wall.friction = 1.0
            wall.collision_type = 1
            wall.color = self.WALL_COLOR
            self.space.add(wall)

        for pos, radius in self.blocks:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            shape = pymunk.Circle(body, radius)
            body.position = Vec2d(pos[0], pos[1])
            shape.elasticity = 0.0  # no bouncing
            shape.color = self.BLOCK_COLOR
            self.space.add(body, shape)

        # moveable entities in the world
        self.entities: Dict[str, Tuple[pymunk.Body, pymunk.Circle]] = {}

    def simulate(self, dt: float = 1.0 / 10, t: int = 10):
        """Simulate the world, updating all entities.

        As per pymunk.Space.step documentation, using a fixed time step `dt` is
        recommended and improves efficiency.

        Also performing multiple steps `t` with a smaller `dt` creates a more stable
        and accurate simulation.

        Arguments
        ---------
        dt: the step size
        t: the number of steps

        Reference
        ---------
        https://www.pymunk.org/en/latest/pymunk.html#pymunk.Space.step

        """
        for _ in range(t):
            self.space.step(dt)

    def get_copy_of_pymunk_space(self) -> pymunk.Space:
        """Get a copy of this world's underlying 2D physisc space.

        This returns a deepcopy, so can be used to simulate actions, etc, without
        affecting the original space.
        """
        return self.space.copy()

    def copy(self) -> "SquareContinuousWorld":
        """Get a deep copy of this world."""
        world = SquareContinuousWorld(
            self.size, self.blocks, self.agent_radius, self.border_thickness
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

    def add_entity(
        self,
        id: str,
        radius: Optional[float],
        color: Optional[Tuple[int, int, int, int]],
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
        shape.elasticity = 0.0  # no bouncing
        if color is not None:
            shape.color = color

        self.space.add(body, shape)
        self.entities[id] = (body, shape)
        return body, shape

    def get_entity_state(self, id: str) -> PMBodyState:
        """Get underlying state of an entity in the world."""
        body, _ = self.entities[id]
        x, y = body.position
        vx, vy = body.velocity
        return PMBodyState(x, y, body.angle, vx, vy, body.angular_velocity)

    def set_entity_state(self, id: str, state: Union[PMBodyState, np.ndarray]):
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
        coord: Optional[
            Union[Tuple[float, float], List[float], np.ndarray, Vec2d]
        ] = None,
        angle: Optional[float] = None,
        vel: Optional[
            Union[Tuple[float, float], List[float], np.ndarray, Vec2d]
        ] = None,
        vangle: Optional[float] = None,
        acceleration: Union[Tuple[float, float], List[float], np.ndarray, Vec2d]
        | None = None,
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

        if acceleration is not None:
            linear_acc = acceleration[0]
            angular_acc = acceleration[1]
            body.apply_force_at_local_point(
                Vec2d(
                    linear_acc * body.mass * math.cos(body.angle),
                    linear_acc * body.mass * math.sin(body.angle),
                ),
                Vec2d(0, 0),
            )
            body.torque = angular_acc * body.moment

    def get_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
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
                for pos, r in self.blocks:
                    if self.euclidean_dist(offset_coord, pos) < (offset + r):
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
    ) -> np.ndarray:
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

        return u * dl2 + l2_start_coords

    @staticmethod
    def convert_angle_to_0_2pi_interval(angle):
        new_angle = np.arctan2(np.sin(angle), np.cos(angle))
        if new_angle < 0:
            new_angle = abs(new_angle) + 2 * (np.pi - abs(new_angle))
        return new_angle

    def _check_collision_circular_rays(
        self,
        origin: Position,
        ray_distance: float,
        n_rays: int,
        other_agents: Optional[np.ndarray] = None,
        include_blocks: bool = True,
        check_walls: bool = True,
        use_relative_angle: bool = True,
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

        Returns
        -------
        distances: the distance each ray extends sway from the origin, up to a max of
            `ray_distance`. Array will have shape `(n_rays,)`.

        """
        x, y, rel_angle = origin
        if not use_relative_angle:
            rel_angle = 0.0

        angles = np.linspace(0.0, 2 * math.pi, n_rays, endpoint=False, dtype=np.float32)

        ray_end_xs = x + ray_distance * np.cos(angles + rel_angle)
        ray_end_ys = y + ray_distance * np.sin(angles + rel_angle)

        ray_start_coords = np.tile((x, y), (n_rays, 1))
        ray_end_coords = np.stack([ray_end_xs, ray_end_ys], axis=1)

        closest_distances = np.full_like(angles, ray_distance)
        collision_types = np.full_like(
            angles, CollisionType.NO_COLLISION.value, dtype=np.uint8
        )

        if other_agents is not None:
            radii = np.array([self.agent_radius] * len(other_agents))
            dists = self.check_circle_line_intersection(
                other_agents, radii, ray_start_coords, ray_end_coords
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                min_dists = np.nanmin(dists, axis=0)

            collision_types[
                min_dists < closest_distances
            ] = CollisionType.AGENT_COLLISION.value
            np.fmin(closest_distances, min_dists, out=closest_distances)

        if include_blocks:
            radii = np.array([s for _, s in self.blocks])
            block_array = np.array([[pos[0], pos[1]] for pos, _ in self.blocks])

            dists = self.check_circle_line_intersection(
                block_array, radii, ray_start_coords, ray_end_coords
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                min_dists = np.nanmin(dists, axis=0)

            collision_types[
                min_dists < closest_distances
            ] = CollisionType.BLOCK_COLLISION.value
            np.fmin(closest_distances, min_dists, out=closest_distances)

        if check_walls:
            # shape = (n_lines, walls, 2)
            wall_intersect_coords = self.check_line_line_intersection(
                ray_start_coords, ray_end_coords, *self.walls
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

            collision_types[
                dists < closest_distances
            ] = CollisionType.BORDER_COLLISION.value
            np.fmin(closest_distances, dists, out=closest_distances)

        return closest_distances, collision_types

    def check_collision_circular_rays(
        self,
        origin: Position,
        ray_distance: float,
        n_rays: int,
        other_agents: Optional[np.ndarray] = None,
        include_blocks: bool = True,
        check_walls: bool = True,
        use_relative_angle: bool = True,
    ) -> np.ndarray:
        return self._check_collision_circular_rays(
            origin,
            ray_distance,
            n_rays,
            other_agents,
            include_blocks,
            check_walls,
            use_relative_angle,
        )[0]

    def get_all_shortest_paths(
        self, origins: Iterable[Position]
    ) -> Dict[Tuple[int, int], Dict[Tuple[int, int], int]]:
        """Get shortest path distance from every origin to all other coords."""
        src_dists = {}
        for origin in origins:
            origin_coord = self.convert_position_to_coord(origin)
            src_dists[origin_coord] = self.dijkstra(origin)
        return src_dists

    def convert_position_to_coord(self, origin: Position) -> Tuple[int, int]:
        """Convert a position to a integer coords."""
        return (math.floor(origin[0]), math.floor(origin[1]))

    def dijkstra(self, origin: Position) -> Dict[Tuple[int, int], int]:
        """Get shortest path distance between origin and all other coords."""
        coord_origin = self.convert_position_to_coord(origin)

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
        ignore_blocks=False,
        include_out_of_bounds=False,
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


def single_item_to_position(coords: np.ndarray) -> Position:
    assert coords.shape[0] >= 3
    return tuple(coords[:3])  # type: ignore

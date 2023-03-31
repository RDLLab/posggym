"""General grid world problem utility functions and classes."""
import math
from abc import ABC, abstractmethod
from enum import Enum
from itertools import product
from queue import PriorityQueue
from typing import Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import numpy as np
from gymnasium import spaces

import posggym.model as M


# (x, y) coord = (col, row) coord
Coord = Tuple[float, float]

# (x, y, yaw) in continuous world
Position = Tuple[float, float, float]

# Position, radius
Object = Tuple[Position, float]


class ArenaTypes(Enum):
    Square = 0
    Circle = 1


T = TypeVar("T")  # Declare type variable


def asserts_exist(x: Optional[T]) -> T:
    assert x
    return x


class ContinuousWorld(ABC):
    """A continuous 2D world."""

    EPS = 1e-4

    def __init__(
        self,
        world_type: ArenaTypes,
        agent_size: float = 0.5,
        yaw_limit: float = math.pi / 10,
        block_coords: Optional[List[Object]] = None,
    ):
        self.agent_size = agent_size
        self.block_coords = [] if block_coords is None else block_coords
        self.yaw_limit = yaw_limit

    @staticmethod
    def manhattan_dist(
        coord1: Union[Coord, Position], coord2: Union[Coord, Position]
    ) -> float:
        """Get manhattan distance between two positions on the grid."""
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

    @staticmethod
    def euclidean_dist(
        coord1: Union[Coord, Position], coord2: Union[Coord, Position]
    ) -> float:
        """Get Euclidean distance between two positions on the grid."""
        return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    @staticmethod
    def squared_euclidean_dist(
        coord1: Union[Coord, Position], coord2: Union[Coord, Position]
    ) -> float:
        """Get Squared Euclidean distance between two positions on the grid."""
        return (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2

    @abstractmethod
    def coord_in_bounds(self, coord: Position) -> bool:
        """Return whether a coordinate is inside the grid or not."""
        pass

    @abstractmethod
    def clamp_coords(self, coord: Position) -> Position:
        """Return closest position that is inside the grid."""
        pass

    @abstractmethod
    def check_collision_wall(
        self, coord: Position, line_distance: float, angle: float
    ) -> Tuple[bool, Optional[float]]:
        """Check for collision with a wall."""
        pass

    def get_all_shortest_paths(
        self, origins: Iterable[Position]
    ) -> Dict[Tuple[int, int], Dict[Tuple[int, int], int]]:
        """Get shortest path distance from every origin to all other coords."""
        src_dists = {}
        for origin in origins:
            origin_coord = self.convert_position_to_coordinate(origin)
            src_dists[origin_coord] = self.dijkstra(origin)
        return src_dists

    def convert_position_to_coordinate(self, origin: Position) -> Tuple[int, int]:
        return (int(math.floor(origin[0])), int(math.floor(origin[1])))

    def dijkstra(self, origin: Position) -> Dict[Tuple[int, int], int]:
        """Get shortest path distance between origin and all other coords."""
        coord_origin = self.convert_position_to_coordinate(origin)

        dist = {coord_origin: 0}
        pq = PriorityQueue()  # type: ignore
        pq.put((dist[coord_origin], coord_origin))

        visited = {coord_origin}

        while not pq.empty():
            _, coord = pq.get()
            coord_c = self.convert_position_to_coordinate(coord)
            for adj_coord in self.get_cardinal_neighbours(coord, ignore_blocks=False):
                adj_coord_c = self.convert_position_to_coordinate(adj_coord)
                if dist[coord_c] + 1 < dist.get(adj_coord_c, float("inf")):
                    dist[adj_coord_c] = dist[coord_c] + 1

                    if adj_coord not in visited:
                        pq.put((dist[adj_coord_c], adj_coord))
                        visited.add(adj_coord_c)
        return dist

    def get_next_coord(
        self,
        coord: Position,
        action: Union[np.ndarray, List[float]],
        ignore_blocks: bool = False,
    ) -> Position:
        return self._get_next_coord(coord, action, ignore_blocks)[0]

    def _non_holonomic_model(
        self,
        coord: Position,
        action: Union[np.ndarray, List[float]],
        ignore_blocks: bool = False,
        include_out_of_bounds: bool = True,
    ) -> Tuple[Position, bool]:
        if len(action) == 1:
            delta_yaw = action[0]
            velocity = 0.25
        else:
            delta_yaw, velocity = action

        x, y, yaw = coord
        new_yaw = yaw + delta_yaw

        new_yaw = new_yaw % (2 * math.pi)  # To avoid bigger than 360 degrees
        if new_yaw > (math.pi):
            new_yaw = new_yaw - 2 * math.pi

        x += velocity * math.cos(new_yaw)
        y += velocity * math.sin(new_yaw)

        new_coord = self.clamp_coords((x, y, new_yaw))

        if not ignore_blocks and self.check_collision((new_coord, self.agent_size)):
            return (coord, False)

        # Out of bounds is unsuccessful
        if not include_out_of_bounds and new_coord != (x, y, new_yaw):
            return (new_coord, False)

        return (new_coord, True)

    def _holonomic_model(
        self,
        coord: Position,
        action: Union[np.ndarray, List[float]],
        ignore_blocks: bool = False,
        include_out_of_bounds: bool = True,
    ) -> Tuple[Position, bool]:
        x, y, yaw = coord
        delta_x, delta_y = action

        x += delta_x
        y += delta_y

        new_coord = self.clamp_coords((x, y, yaw))

        if not ignore_blocks and self.check_collision((new_coord, self.agent_size)):
            return (coord, False)

        if not include_out_of_bounds and new_coord != (x, y, yaw):
            return (new_coord, False)

        return (new_coord, True)

    def _get_next_coord(
        self,
        coord: Position,
        action: Union[np.ndarray, List[float]],
        ignore_blocks: bool = False,
        use_holonomic_model: bool = False,
    ) -> Tuple[Position, bool]:
        """Get next position given loc and movement direction.

        If new position is outside of the grid boundary, or (ignore_blocks is
        False and new coordinate is a block) then returns the original
        coordinate.
        """
        if use_holonomic_model:
            return self._holonomic_model(coord, action, ignore_blocks)
        else:
            return self._non_holonomic_model(coord, action, ignore_blocks)

    def clamp(self, min_bound, max_bound, value):
        return min(max_bound, max(value, min_bound))

    def rotate_point_around_origin(
        self, point: Tuple[float, float], angle: float
    ) -> Tuple[float, float]:
        """Rotate a point around the origin by a given angle in radians.

        Args:
            point (Tuple[float, float]): The (x, y) coordinates of the point to rotate.
            angle (float): The angle in radians by which to rotate the point.

        Returns:
            Tuple[float, float]: The rotated (x', y') coordinates of the point.
        """
        x, y = point
        x_prime = x * math.cos(angle) - y * math.sin(angle)
        y_prime = x * math.sin(angle) + y * math.cos(angle)
        return x_prime, y_prime

    def check_circle_line_intersection(
        self,
        x: float,
        y: float,
        r: float,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
    ) -> Optional[float]:
        """Check if line intersects circle at (x, y) with radius r.

        Line starts at (start_x, start_y) and ends at (end_x, end_y).

        If line intersects circle, returns distance till intersection, otherwise returns
        None.
        """
        # https://math.stackexchange.com/questions/275529/check-if-line-intersects-with-circles-perimeter
        start_x -= x
        start_y -= y
        end_x -= x
        end_y -= y
        dx, dy = (end_x - start_x), (end_y - start_y)

        a = dx**2 + dy**2
        b = 2 * (start_x * dx + start_y * dy)
        c = start_x**2 + start_y**2 - r**2
        disc = b**2 - 4 * a * c

        if disc <= 0:
            return None

        sqrtdisc = math.sqrt(disc)
        t1 = (-b + sqrtdisc) / (2 * a)
        t2 = (-b - sqrtdisc) / (2 * a)
        if 0 <= t1 <= t2:
            t = t1
        elif 0 <= t2 <= t1:
            t = t2
        else:
            # collision only possible moving backwards
            return None
        # distance till intersection
        return math.sqrt((t * dx) ** 2 + (t * dy) ** 2)

    def check_circle_circle_intersection(
        self,
        circle_1: Union[Coord, Position],
        circle_2: Union[Coord, Position],
        velocity_1: Tuple[float, float],
        velocity_2: Tuple[float, float],
        radius_1: float,
        radius_2: float,
    ) -> Optional[float]:
        """Check if circle collides with another circle.

        `circle_1` and `circle_2` are the start positions of each circle's center.
        `velocity_1` and `velocity_2` are the velocities of each circle in the x and y
        directions.
        `radius_1` and `radius_2` are the radius of each circle.

        If objects will collide, returns distance till collision, otherwise returns
        None.

        """
        # https://physics.stackexchange.com/questions/221895/time-until-next-collision
        (vx1, vy1), (vx2, vy2) = velocity_1, velocity_2
        vx, vy = (vx1 - vx2), (vy1 - vy2)
        (x1, y1), (x2, y2) = (circle_1[0], circle_1[1]), (circle_2[0], circle_2[1])
        dx, dy = (x1 - x2), (y1 - y2)

        # Need to find distance when circle_1 and circle_2 are within
        # (radius_1 + radius_2) distance of each other
        a = vx**2 + vy**2
        b = 2 * (dx * vx + dy * vy)
        c = dy**2 + dx**2 - (radius_1 + radius_2) ** 2
        disc = b**2 - 4 * a * c

        if disc <= 0:
            # no collision possible
            return None

        sqrtdisc = math.sqrt(disc)
        t1 = (-b - sqrtdisc) / (2 * a)
        t2 = (-b + sqrtdisc) / (2 * a)

        if (t1 < 0.0 and t2 < self.EPS) or (t1 < self.EPS and t2 < 0.0):
            # Case 1: (t1 < 0.0 and t2 < 0.0) - collision already, but moving away
            return None
        # elif t1 >= 0.0 and (t2 < 0.0 or t1 <= t2):
        elif t1 >= self.EPS and t2 < 0.0:
            # Case 2: (t1 >= 0.0 and t2 < 0.0) already in collision
            #         (allow to keep moving forward)
            if t1 <= 1.0:
                # will reach end of collision within step
                return math.sqrt((t1 * vx) ** 2 + (t1 * vy) ** 2)
            # will still be in collision within step so move small distance towards
            # closest edge
            t2 = abs(t2)
            if t2 <= t1:
                # move slightly backwards
                return -min(self.EPS, math.sqrt((t2 * vx) ** 2 + (t2 * vy) ** 2))
            return min(self.EPS, math.sqrt((t1 * vx) ** 2 + (t1 * vy) ** 2))
        # elif t2 >= 0.0 and (t1 < 0.0 or t2 <= t1):
        elif t2 >= self.EPS and t1 < 0.0:
            # Case 3: (t2 >= 0.0 and t1 < 0.0) already in collision
            #         (allow to keep moving forward)
            if t2 <= 1.0:
                # will reach end of collision within step
                return math.sqrt((t2 * vx) ** 2 + (t2 * vy) ** 2)
            # will still be in collision within step so move small distance towards
            # closest edge
            t1 = abs(t1)
            if t1 <= t2:
                # move slightly backwards
                return -min(self.EPS, math.sqrt((t1 * vx) ** 2 + (t1 * vy) ** 2))
            return min(self.EPS, math.sqrt((t2 * vx) ** 2 + (t2 * vy) ** 2))
        elif 0.0 <= t1 <= t2:
            # Case 4a:(0 <= t1 <= t2) collision in current direction (t1 closer)
            return math.sqrt((t1 * vx) ** 2 + (t1 * vy) ** 2)
        elif 0.0 <= t2 <= t1:
            # Case 4b:(0 <= t2 <= t1) collision in current direction (t2 closer)
            return math.sqrt((t2 * vx) ** 2 + (t2 * vy) ** 2)
        else:
            # Not sure
            d1 = math.sqrt((t1 * vx) ** 2 + (t1 * vy) ** 2)
            d2 = math.sqrt((t2 * vx) ** 2 + (t2 * vy) ** 2)
            raise AssertionError(
                "Something weird happened with collision between circles: "
                f"({x1=:.6f}, {y1=:.6f}), ({x2=:.6f}, {y2=:.6f}), with velocities: "
                f"({vx1=:.6f}, {vy1=:.6f}), ({vx2=:.6f}, {vy2=:.6f}), and radiis: "
                f"{radius_1=:.6f}, {radius_2=:.6f}. "
                f"({t1=:.10f} {d1=:.10f}) ({t2=:.10f} {d2=:.10f})"
            )

    def check_collision_ray(
        self,
        coord: Position,
        line_distance: float,
        line_angle: float,
        other_agents: Optional[Union[Tuple[Position, ...], np.ndarray]],
        skip_id: Optional[Union[int, List[int]]] = None,
        only_walls: bool = False,
        include_blocks: bool = True,
        use_relative_angle: bool = True,
    ) -> Tuple[Optional[int], float]:
        """Check for collision along ray.

        Returns entity index and distance if there is a collision, otherwise if there is
        no collision returns None and `line distance`.

        If collision is with a wall or block, return index of -1.

        If `use_relative_angle=True` then line angle is treated as relative to agents
        yaw angle. Otherwise line angle is treated as absolute (i.e. relative to angle
        of 0).

        """
        if skip_id is None:
            skip_id = []
        elif isinstance(skip_id, int):
            skip_id = [skip_id]

        closest_distance = line_distance
        closest_index = None

        x, y, rel_angle = coord
        if not use_relative_angle:
            rel_angle = 0.0
        end_x = x + line_distance * math.cos(line_angle + rel_angle)
        end_y = y + line_distance * math.sin(line_angle + rel_angle)

        if not only_walls and other_agents is not None:
            for index, pos in enumerate(other_agents):
                if index in skip_id:
                    continue
                dist = self.check_circle_line_intersection(
                    pos[0], pos[1], self.agent_size, x, y, end_x, end_y
                )
                if dist is not None and dist < closest_distance:
                    closest_distance = dist
                    closest_index = index

        if include_blocks:
            for index, (pos, size) in enumerate(self.block_coords):
                dist = self.check_circle_line_intersection(
                    pos[0], pos[1], size, x, y, end_x, end_y
                )
                if dist is not None and dist < closest_distance:
                    closest_distance = dist
                    closest_index = -1

        _, dist = self.check_collision_wall(coord, line_distance, line_angle)
        if dist is not None and dist < closest_distance:
            closest_distance = dist
            closest_index = -1

        return (closest_index, closest_distance)

    @abstractmethod
    def get_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        pass

    def get_cardinal_neighbours(
        self,
        coord: Position,
        ignore_blocks=False,
        include_out_of_bounds=False,
    ) -> List[Position]:
        """Get set of adjacent non-blocked coordinates."""
        (min_x, max_x), (min_y, max_y) = self.get_bounds()
        neighbours = []

        rounded_coord: Tuple[float, float] = (
            round(coord[0]),
            round(coord[1]),
        )

        if rounded_coord[1] > min_y or include_out_of_bounds:
            neighbours.append((rounded_coord[0], rounded_coord[1] - 1, 0.0))  # N
        if rounded_coord[0] < max_x - 1 or include_out_of_bounds:
            neighbours.append((rounded_coord[0] + 1, rounded_coord[1], 0.0))  # E
        if rounded_coord[1] < max_y or include_out_of_bounds:
            neighbours.append((rounded_coord[0], rounded_coord[1] + 1, 0.0))  # S
        if rounded_coord[0] > min_x or include_out_of_bounds:
            neighbours.append((rounded_coord[0] - 1, rounded_coord[1], 0.0))  # W

        if ignore_blocks:
            return neighbours

        for i in range(len(neighbours), 0, -1):
            if neighbours[i - 1] in self.block_coords:
                neighbours.pop(i - 1)

        return neighbours

    def generate_range(self, lower: float, upper: float, length: int) -> List[float]:
        # https://stackoverflow.com/questions/6683690/making-a-list-of-evenly-spaced-numbers-in-a-certain-range-in-python
        return [lower + x * (upper - lower) / length for x in range(length)]

    def get_possible_next_pos(
        self,
        coord: Position,
        num_samples: int = 20,
        distance: float = 1.0,
        ignore_blocks: bool = False,
        include_out_of_bounds: bool = False,
        force_non_colliding: bool = False,
        use_holonomic_model: bool = False,
    ):
        if use_holonomic_model:
            points = self.generate_range(0, 2 * math.pi, num_samples)
        else:
            # Restrict change in yaw -pi/10 -> pi/10
            points = self.generate_range(-self.yaw_limit, self.yaw_limit, num_samples)
        distances = self.generate_range(0, distance, num_samples)

        data = product(points, distances)

        output: List[Position] = []
        for yaw, d in data:
            if use_holonomic_model:
                dx = math.sin(yaw) * d
                dy = math.cos(yaw) * d
                action = [dx, dy]
            else:
                action = [yaw, distance]

            new_coords, success = self._get_next_coord(
                coord,
                action,
                ignore_blocks=ignore_blocks,
                use_holonomic_model=use_holonomic_model,
            )

            if success:
                if force_non_colliding:
                    if not self.check_agent_collisions(new_coords, output):
                        output.append(new_coords)
                else:
                    output.append(new_coords)

        return output

    def get_neighbours(
        self,
        coord: Position,
        num_samples: int = 20,
        distance: float = 1.0,
        ignore_blocks: bool = False,
        include_out_of_bounds: bool = False,
        force_non_colliding: bool = False,
    ):
        # This function should try to find all possible positions around a
        # coordinate where another agent could be.
        points = [i * (2 * math.pi) / (num_samples - 1) for i in range(num_samples)]
        output: List[Position] = []
        for yaw in points:
            dx = math.sin(yaw) * distance
            dy = math.cos(yaw) * distance
            new_coords, success = self._holonomic_model(
                coord,
                [dx, dy],
                ignore_blocks=ignore_blocks,
                include_out_of_bounds=include_out_of_bounds,
            )

            if success:
                if force_non_colliding:
                    if not self.check_agent_collisions(new_coords, output):
                        output.append(new_coords)
                else:
                    output.append(new_coords)

        return output

    def agents_collide(self, coord1: Position, coord2: Position, distance=None) -> bool:
        dist = ContinuousWorld.squared_euclidean_dist(coord1, coord2)
        return (
            abs(self.agent_size - self.agent_size)
            <= dist
            <= (self.agent_size + self.agent_size)
        )

    def check_collision(self, object: Object) -> bool:
        object_pos, agent_radius = object
        for pos, radius in self.block_coords:
            dist = ContinuousWorld.squared_euclidean_dist(object_pos, pos)
            if dist <= (radius + agent_radius):
                return True

        return False

    def check_agent_collisions(
        self, coord: Position, other_coords: Iterable[Position]
    ) -> bool:
        return any(self.agents_collide(coord, p) for p in other_coords)

    def sample_coords_within_dist(
        self,
        center: Position,
        min_dist_from_center: float,
        rng: Callable[[], float],
        ignore_blocks: bool = False,
        use_holonomic_model: bool = False,
        max_attempts: int = 100,
    ) -> Optional[Position]:
        """Attempts to sample collision free position within distance from center."""
        for _ in range(max_attempts):
            angle = 2 * math.pi * rng()
            distance = min_dist_from_center * math.sqrt(rng())

            x = center[0] + distance * math.cos(angle)
            y = center[1] + distance * math.sin(angle)
            yaw = 0.0 if use_holonomic_model else 2 * math.pi * rng()

            new_coord: Position = self.clamp_coords((x, y, yaw))
            if not self.check_collision((new_coord, self.agent_size)):
                return new_coord
        return None


class CircularContinuousWorld(ContinuousWorld):
    """A continuous world with a circular border."""

    def __init__(
        self,
        radius: float,
        block_coords: Optional[List[Object]] = None,
    ):
        super().__init__(world_type=ArenaTypes.Circle, block_coords=block_coords)
        self.radius = radius

    def get_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (-self.radius, self.radius), (-self.radius, self.radius)

    def coord_in_bounds(self, coord: Position) -> bool:
        self.radius = asserts_exist(self.radius)
        center = (0, 0, 0)
        dist_from_center = ContinuousWorld.euclidean_dist(coord, center)
        return dist_from_center < self.radius

    def check_collision_wall(
        self, coord: Position, line_distance: float, angle: float
    ) -> Tuple[bool, Optional[float]]:
        """Checks if a line starting at position 'coord' and traveling at angle 'angle'
        relative to the agent's angle collides with the wall.

        Args:
            coord (Position): The starting position of the line.
            line_distance (float): The maximum distance the line should travel.
            angle (float): The angle at which the line is traveling relative to the
                           agent's angle.

        Returns:
            A tuple containing a boolean value indicating whether there was a collision
            or not, and the distance to the wall if there was a collision, or None
            otherwise.
        """
        x, y, agent_angle = coord
        line_angle = agent_angle + angle

        end_x = x + line_distance * math.cos(line_angle)
        end_y = y + line_distance * math.sin(line_angle)

        # Check whether the line intersects with the circular boundary
        dx = end_x - x
        dy = end_y - y
        a = dx**2 + dy**2
        b = 2 * (dx * (x) + dy * (y))
        c = x**2 + y**2 - self.radius**2
        disc = b**2 - 4 * a * c

        if disc < 0:
            return False, None
        else:
            sqrt_disc = math.sqrt(disc)
            t1 = (-b - sqrt_disc) / (2 * a)
            t2 = (-b + sqrt_disc) / (2 * a)

            if 0 <= t1 <= 1:
                x1 = x + t1 * dx
                y1 = y + t1 * dy
                return True, math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
            elif 0 <= t2 <= 1:
                x1 = x + t2 * dx
                y1 = y + t2 * dy
                return True, math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
            else:
                return False, None

    def check_collision_circle_wall(
        self,
        origin: Position,
        max_distance: float,
        angle: float,
        r: float,
        use_relative_angle: bool = False,
    ) -> Optional[float]:
        # TODO Basically same as check_collision_wall, just need to include radius
        raise NotImplementedError

    def clamp_coords(self, coord: Position) -> Position:
        x, y, yaw = coord
        # Calculating polar coordinates
        center = (0, 0)
        r = self.euclidean_dist(center, coord)
        gamma = math.atan2(y, x)
        # Virtual barrier:
        if r > self.radius:
            x = self.radius * math.cos(gamma)
            y = self.radius * math.sin(gamma)
        return (x, y, yaw)


class RectangularContinuousWorld(ContinuousWorld):
    def __init__(
        self,
        width: float,
        height: float,
        block_coords: Optional[List[Object]] = None,
    ):
        super().__init__(world_type=ArenaTypes.Square, block_coords=block_coords)
        self.height = height
        self.width = width

    def coord_in_bounds(self, coord: Position) -> bool:
        return 0 <= coord[0] < self.width and 0 <= coord[1] < self.height

    def get_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (0, self.width), (0, self.height)

    def clamp_coords(self, coord: Position) -> Position:
        x, y, yaw = coord
        x = self.clamp(self.agent_size, self.width - self.agent_size, x)
        y = self.clamp(self.agent_size, self.height - self.agent_size, y)
        return (x, y, yaw)

    def check_collision_wall(
        self, coord: Position, line_distance: float, angle: float
    ) -> Tuple[bool, Optional[float]]:
        """Checks if a line collides with the arena's walls.

        Arguments
        ---------
        coord : The starting position of the line.
        line_distance : The maximum distance the line should travel.
        angle : The angle at which the line is traveling relative to the agent's
            angle.

        Returns
        -------
        collision : whether there was a collision or not
        distance : the distance to the wall if there was a collision, or None otherwise.

        """
        x, y, agent_angle = coord
        line_angle = angle + agent_angle
        end_x = x + line_distance * math.cos(line_angle)
        end_y = y + line_distance * math.sin(line_angle)

        # Check whether it intersects with L/R boundary
        if end_x < 0 or end_x > self.width:
            x1 = 0.0 if end_x < 0 else self.width

            # Compute the y-coordinate where the intersection occurs
            y1 = (end_y - y) / (end_x - x) * (x1 - x) + y

            if 0 <= y1 <= self.height:
                return True, math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)

        # Check whether it intersects with Top/Bottom boundary
        if end_y < 0 or end_y > self.height:
            y1 = 0.0 if end_y < 0 else self.height

            # Compute the x-coordinate where the intersection occurs
            x1 = (end_x - x) / (end_y - y) * (y1 - y) + x

            if 0 <= x1 <= self.width:
                return True, math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)

        return False, None


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


def position_to_array(coords: Iterable[Position], squeeze: bool = True) -> np.ndarray:
    """Converts list of positions into a 2D array of positions.

    New array has shape (len(coords), 3), where  Each row is the new array is a single
    position.

    If squeeze=True will output 1D array if the list contains a single position.
    """
    array = np.array([np.array(x, dtype=np.float32) for x in coords], dtype=np.float32)
    if squeeze:
        return array.squeeze()
    return array


def array_to_position(coords: np.ndarray) -> Tuple[Position, ...]:
    if coords.ndim == 2:
        assert coords.shape[1] == 3
        output = []
        for i in range(coords.shape[0]):
            output.append(tuple(coords[i, :]))
        return tuple(output)  # type: ignore

    elif coords.ndim == 1:
        assert coords.shape[0] == 3
        return tuple(coords)
    else:
        raise Exception("Cannot convert")


def single_item_to_position(coords: np.ndarray) -> Position:
    assert coords.shape[0] == 3
    return tuple(coords)  # type: ignore

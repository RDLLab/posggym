"""General grid world problem utility functions and classes."""
from queue import PriorityQueue
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar, Callable
import math
from enum import Enum
from abc import ABC, abstractmethod
import posggym.model as M
from gymnasium import spaces
import numpy as np

# (x, y) coord = (col, row) coord
Coord = Tuple[int, int]

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
    def __init__(
        self,
        world_type: ArenaTypes,
        agent_size: float = 0.5,
        use_holonomic_model: bool = True,
        block_coords: Optional[List[Object]] = None,
    ):
        self.agent_size = agent_size
        if block_coords is None:
            block_coords = []
        self.block_coords = block_coords

        self.use_holonomic_model = use_holonomic_model

    def set_holonomic_model(self, use_holonomic_model):
        self.use_holonomic_model = use_holonomic_model

    @staticmethod
    def manhattan_dist(coord1: Position, coord2: Position) -> float:
        """Get manhattan distance between two coordinates on the grid."""
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

    @staticmethod
    def euclidean_dist(coord1: Position, coord2: Position):
        return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    @staticmethod
    def squared_euclidean_dist(coord1: Position, coord2: Position):
        return (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2

    @abstractmethod
    def coord_in_bounds(self, coord: Position) -> bool:
        """Return whether a coordinate is inside the grid or not."""
        pass

    @abstractmethod
    def clamp_coords(self, coord: Position) -> Position:
        """Return whether a coordinate is inside the grid or not."""
        pass

    @abstractmethod
    def check_collision_wall(
        self, coord: Position, line_distance: float, angle: float
    ) -> Tuple[bool, Optional[float]]:
        pass

    def get_all_shortest_paths(
        self, origins: Iterable[Position]
    ) -> Dict[Position, Dict[Position, int]]:
        """Get shortest path distance from every origin to all other coords."""
        src_dists = {}
        for origin in origins:
            src_dists[origin] = self.dijkstra(origin)
        return src_dists

    def dijkstra(self, origin: Position) -> Dict[Position, int]:
        """Get shortest path distance between origin and all other coords."""
        dist = {origin: 0}
        pq = PriorityQueue()  # type: ignore
        pq.put((dist[origin], origin))

        visited = {origin}

        while not pq.empty():
            _, coord = pq.get()
            for adj_coord in self.get_cardinal_neighbours(coord, ignore_blocks=False):
                adj_coord = (int(adj_coord[0]), int(adj_coord[1]), int(adj_coord[2]))
                if dist[coord] + 1 < dist.get(adj_coord, float("inf")):
                    dist[adj_coord] = dist[coord] + 1

                    if adj_coord not in visited:
                        pq.put((dist[adj_coord], adj_coord))
                        visited.add(adj_coord)
        return dist

    def get_next_coord(
        self, coord: Position, action: List[float], ignore_blocks: bool = False
    ) -> Position:
        return self._get_next_coord(coord, action, ignore_blocks)[0]

    def _non_holonomic_model(
        self,
        coord: Position,
        action: List[float],
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
        action: List[float],
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
        self, coord: Position, action: List[float], ignore_blocks: bool = False
    ) -> Tuple[Position, bool]:
        """Get next position given loc and movement direction.

        If new position is outside of the grid boundary, or (ignore_blocks is
        False and new coordinate is a block) then returns the original
        coordinate.
        """
        if self.use_holonomic_model:
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
        self, cx: float, cy: float, r: float, ax: float, ay: float, bx: float, by: float
    ) -> Optional[float]:
        # https://math.stackexchange.com/questions/275529/check-if-line-intersects-with-circles-perimeter
        # // parameters: ax ay bx by cx cy r
        ax -= cx
        ay -= cy
        bx -= cx
        by -= cy

        a = (bx - ax) ** 2 + (by - ay) ** 2

        b = 2 * (ax * (bx - ax) + ay * (by - ay))
        c = ax**2 + ay**2 - r**2
        disc = b**2 - 4 * a * c

        if disc <= 0:
            return None

        sqrtdisc = math.sqrt(disc)
        t1 = (-b + sqrtdisc) / (2 * a)
        t2 = (-b - sqrtdisc) / (2 * a)

        if t1 > 0 and t1 < 1:
            intersection_x = ax + t2 * (bx - ax)
            intersection_y = ay + t2 * (by - ay)
            distance_to_intersection = math.sqrt(
                (intersection_x - ax) ** 2 + (intersection_y - ay) ** 2
            )
            return distance_to_intersection

        if t2 > 0 and t2 < 1:
            intersection_x = ax + t2 * (bx - ax)
            intersection_y = ay + t2 * (by - ay)
            distance_to_intersection = math.sqrt(
                (intersection_x - ax) ** 2 + (intersection_y - ay) ** 2
            )
            return distance_to_intersection

        return None

    def check_collision_ray(
        self,
        coord: Position,
        line_distance: float,
        angle: float,
        other_agents: Tuple[Position, ...],
        skip_id: Optional[int] = None,
    ) -> Tuple[Optional[int], float]:
        closest_agent_pos = None
        closest_agent_distance = float("inf")
        closest_agent_index = None

        x, y, agent_angle = coord

        for index, (agent_pos) in enumerate(other_agents):
            if skip_id is not None and skip_id == index:
                continue

            other_agent_x, other_agent_y, _ = agent_pos

            end_x = x + line_distance * math.cos(angle + agent_angle)
            end_y = y + line_distance * math.sin(angle + agent_angle)

            dist = self.check_circle_line_intersection(
                other_agent_x, other_agent_y, self.agent_size, x, y, end_x, end_y
            )

            if dist is not None and dist < closest_agent_distance:
                closest_agent_distance = dist
                closest_agent_pos = agent_pos
                closest_agent_index = index

        if closest_agent_pos is None:
            _, closest_agent_distance = self.check_collision_wall(
                coord, line_distance, angle
            )

            if closest_agent_distance is not None:
                closest_agent_index = -1

        if closest_agent_distance is None:
            closest_agent_distance = line_distance

        return (closest_agent_index, closest_agent_distance)

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

        rounded_coord: Tuple[float, float, float] = (
            round(coord[0]),
            round(coord[1]),
            coord[2],
        )

        if rounded_coord[1] > min_y or include_out_of_bounds:
            neighbours.append(
                (rounded_coord[0], rounded_coord[1] - 1, rounded_coord[2])
            )  # N
        if rounded_coord[0] < max_x - 1 or include_out_of_bounds:
            neighbours.append(
                (rounded_coord[0] + 1, rounded_coord[1], rounded_coord[2])
            )  # E
        if rounded_coord[1] < max_y or include_out_of_bounds:
            neighbours.append(
                (rounded_coord[0], rounded_coord[1] + 1, rounded_coord[2])
            )  # S
        if rounded_coord[0] > min_x or include_out_of_bounds:
            neighbours.append(
                (rounded_coord[0] - 1, rounded_coord[1], rounded_coord[2])
            )  # W

        if ignore_blocks:
            return neighbours

        for i in range(len(neighbours), 0, -1):
            if neighbours[i - 1] in self.block_coords:
                neighbours.pop(i - 1)

        return neighbours

    def get_neighbours(
        self,
        coord: Position,
        num_samples: int = 20,
        distance: float = 1.0,
        ignore_blocks: bool = False,
        include_out_of_bounds: bool = False,
        force_non_colliding: bool = False,
    ):
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
            if abs(radius - agent_radius) <= dist <= (radius + agent_radius):
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
        ignore_blocks=False,
        max_attempts=100,
    ) -> Position:
        for _ in range(max_attempts):
            angle = 2 * math.pi * rng()
            distance = min_dist_from_center * math.sqrt(rng())

            x = center[0] + distance * math.cos(angle)
            y = center[1] + distance * math.sin(angle)
            yaw = 0.0 if self.use_holonomic_model else 2 * math.pi * rng()

            new_coord: Position = (x, y, yaw)

            if not self.check_collision((new_coord, self.agent_size)):
                return new_coord

        raise Exception(
            (
                f"Cannot sample coordinate within distance {min_dist_from_center}",
                f"from {center} within {max_attempts}",
            )
        )


class CircularContinuousWorld(ContinuousWorld):
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
        relative to the agent's angle collides with the circle.

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

    def clamp_coords(self, coord: Position) -> Position:
        x, y, yaw = coord
        # Calculating polar coordinates
        center = (0, 0, 0)
        r = ContinuousWorld.euclidean_dist(center, (x, y, yaw))
        gamma = math.atan2(y, x)
        # Virtual barriere:
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
        self.eps = 10e-3

    def coord_in_bounds(self, coord: Position) -> bool:
        return 0 <= coord[0] < self.width and 0 <= coord[1] < self.height

    def clamp_coords(self, coord: Position) -> Position:
        x, y, yaw = coord
        x = self.clamp(0.5, self.width - 0.5, x)
        y = self.clamp(0.5, self.height - 0.5, y)
        return (x, y, yaw)

    def get_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (0, self.width), (0, self.height)

    def check_collision_wall(
        self, coord: Position, line_distance: float, angle: float
    ) -> Tuple[bool, Optional[float]]:
        """Checks if a line starting at position 'coord' and traveling at angle 'angle'
        relative to the agent's angle collides with the arena's walls.

        Args:
            coord (Position): The starting position of the line.
            line_distance (float): The maximum distance the line should travel.
            angle (float): The angle at which the line is traveling relative to the
            agent's angle.

        Returns:
            A tuple containing a boolean value indicating whether there was a collision
            pr not, and the distance to the wall if there was a collision, or None
            otherwise.
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
    actions: Dict[M.AgentID, List[float]], action_spaces: Dict[M.AgentID, spaces.Space]
) -> Dict[M.AgentID, List[float]]:
    assert all([isinstance(action_spaces[i], spaces.Box) for i in action_spaces])

    return {
        i: list(np.clip(actions[i], action_spaces[i].low, action_spaces[i].high))
        for i in actions
    }

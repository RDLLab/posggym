"""General grid world problem utility functions and classes."""
import enum
import itertools
import random
from queue import PriorityQueue, Queue
from typing import Dict, Iterable, List, Optional, Set, Tuple, TypeVar
import math
from enum import Enum
from abc import ABC, abstractmethod

# (x, y) coord = (col, row) coord
Coord = Tuple[int, int]

# (x, y, radius) in continous world
Position = Tuple[float, float, float]

# Position, radius
Object = Tuple[Position, float]


class ArenaTypes(Enum):
    Square = 0
    Circle = 1


T = TypeVar('T')      # Declare type variable


def asserts_exist(x: Optional[T]) -> T:
    assert x
    return x


class ContinousWorld(ABC):
    def __init__(
        self,
        world_type: ArenaTypes,
        agent_size: float = 0.5,
        block_coords: Optional[List[Object]] = None,
    ):
        self.agent_size = agent_size
        if block_coords is None:
            block_coords = []
        self.block_coords = block_coords

    @staticmethod
    def manhattan_dist(coord1: Position, coord2: Position) -> float:
        """Get manhattan distance between two coordinates on the grid."""
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

    @staticmethod
    def euclidean_dist(coord1: Position, coord2: Position):
        return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

    @abstractmethod
    def coord_in_bounds(self, coord: Position) -> bool:
        """Return whether a coordinate is inside the grid or not."""
        pass

    @abstractmethod
    def clamp_coords(self, coord: Position) -> Position:
        """Return whether a coordinate is inside the grid or not."""
        pass

    def get_next_coord(
        self, coord: Position, delta_yaw: float, ignore_blocks: bool = False
    ) -> Position:
        return self._get_next_coord(coord, delta_yaw, ignore_blocks)[0]

    def _get_next_coord(
        self, coord: Position, delta_yaw: float, velocity: float = 0.25, ignore_blocks: bool = False
    ) -> Tuple[Position, bool]:
        """Get next position given loc and movement direction.

        If new position is outside of the grid boundary, or (ignore_blocks is
        False and new coordinate is a block) then returns the original
        coordinate.
        """
        x, y, yaw = coord

        new_yaw = yaw + delta_yaw

        new_yaw = new_yaw % (2 * math.pi)  # To avoid bigger thatn 360 degrees
        if new_yaw > (math.pi):
            new_yaw = new_yaw - 2 * math.pi

        x += velocity * math.cos(new_yaw)
        y += velocity * math.sin(new_yaw)

        new_coord = self.clamp_coords((x, y, new_yaw))

        if not ignore_blocks and self.check_collision((new_coord, self.agent_size)):
            return (coord, False)

        return (new_coord, True)

    def clamp(self, min_bound, max_bound, value):
        return min(max_bound, max(value, min_bound))

    def get_neighbours(self, coord: Position, num_samples=20, ignore_blocks=False, include_out_of_bounds=False):
        points = [i * (2 * math.pi) / (num_samples - 1)
                  for i in range(num_samples)]
        output = []
        for yaw in points:
            new_coords, success = self._get_next_coord(
                coord, yaw, ignore_blocks=ignore_blocks)
            if success:
                output.append(new_coords)

        return output

    def agents_collide(self, coord1: Position, coord2: Position, distance=None) -> bool:
        return ContinousWorld.manhattan_dist(coord1, coord2) < (distance or (self.agent_size + self.agent_size))

    def check_collision(self, object: Object) -> bool:
        object_pos, agent_radius = object
        for pos, radius in self.block_coords:
            if ContinousWorld.manhattan_dist(object_pos, pos) < (radius + agent_radius):
                return True
        return False

    def sample_coords_within_dist(self, center: Position, min_dist_from_center: float, ignore_blocks=False, max_attempts=100) -> Position:
        import random
        for _ in range(max_attempts):
            angle = 2*math.pi*random.random()
            distance = min_dist_from_center*math.sqrt(random.random())

            x = center[0] + distance*math.cos(angle)
            y = center[1] + distance*math.sin(angle)
            yaw = 2*math.pi*random.random()
            new_coord: Position = (x, y, yaw)

            if not self.check_collision((new_coord, self.agent_size)):
                return new_coord

        raise Exception(
            f"Cannot sample coordinate within distance {min_dist_from_center} from {center} within {max_attempts}")


class CircularContinousWorld(ContinousWorld):
    def __init__(
        self,
        radius: float,
        block_coords: Optional[List[Object]] = None,
    ):
        super().__init__(world_type=ArenaTypes.Circle, block_coords=block_coords)
        self.radius = radius

    def coord_in_bounds(self, coord: Position) -> bool:
        self.radius = asserts_exist(self.radius)
        center = (0, 0, 0)
        dist_from_center = ContinousWorld.euclidean_dist(coord, center)
        return dist_from_center < self.radius

    def clamp_coords(self, coord: Position) -> Position:
        x, y, yaw = coord
        # Calculating polar coordinates
        center = (0, 0, 0)
        r = ContinousWorld.euclidean_dist(center, (x, y, yaw))
        gamma = math.atan2(y, x)
        # Virtual barriere:
        if r > self.radius:
            x = self.radius * math.cos(gamma)
            y = self.radius * math.sin(gamma)
        return (x, y, yaw)


class RectangularContinousWorld(ContinousWorld):
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

    def clamp_coords(self, coord: Position) -> Position:
        x, y, yaw = coord
        x = self.clamp(0, self.width, x)
        y = self.clamp(0, self.height, y)
        return (x, y, yaw)

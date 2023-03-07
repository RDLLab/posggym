"""General grid world problem utility functions and classes."""
import enum
import itertools
import random
from queue import PriorityQueue, Queue
from typing import Dict, Iterable, List, Optional, Set, Tuple
import math

# (x, y) coord = (col, row) coord
Coord = Tuple[int, int]

# (x, y, radius) in continous world
Position = Tuple[float, float, float]

# Position, radius
Object = Tuple[Position, float]

class ContinousWorld:
    def __init__(
        self,
        width: int,
        height: int,
        block_coords: Optional[List[Object]] = None,
    ):
        self.width = width
        self.height = height
        self.agent_size = 0.5

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

    def coord_in_bounds(self, coord: Coord) -> bool:
        """Return whether a coordinate is inside the grid or not."""
        return 0 <= coord[0] < self.width and 0 <= coord[1] < self.height

    def get_next_coord(
        self, coord: Position, delta_yaw: float, ignore_blocks: bool = False
    ) -> Position:
        return self._get_next_coord(coord, delta_yaw, ignore_blocks)[0]
    
    def _get_next_coord(
        self, coord: Position, delta_yaw: float, ignore_blocks: bool = False
    ) -> Tuple[Position, bool]:
        """Get next position given loc and movement direction.

        If new position is outside of the grid boundary, or (ignore_blocks is
        False and new coordinate is a block) then returns the original
        coordinate.
        """

        x,y,angle = coord

        new_yaw = angle + delta_yaw

        new_yaw = new_yaw % (2 * math.pi) # To avoid bigger thatn 360 degrees
        if new_yaw > (math.pi):
            new_yaw = new_yaw - 2 * math.pi

        velocity = 0.25
        x += velocity * math.cos(new_yaw)
        y += velocity * math.sin(new_yaw)
        x = self.clamp(0, self.height, x)
        y = self.clamp(0, self.width, y)

        new_coord = ( x, y, new_yaw)

        if not ignore_blocks and self.check_collision((new_coord, self.agent_size)):
            return (coord, False)

        return (new_coord, True)

    def clamp(self, min_bound, max_bound, value):
        return min(max_bound, max(value, min_bound))
    
    def get_neighbours(self, coord : Position, num_samples = 20, ignore_blocks=False, include_out_of_bounds=False):
        points = [i * (2 * math.pi) / (num_samples - 1) for i in range(num_samples)]
        output = []
        for yaw in points:
            new_coords, success = self._get_next_coord(coord, yaw, ignore_blocks=ignore_blocks)
            if success:
                output.append(new_coords)

        return output

    def agents_collide(self, coord1 : Position, coord2: Position) -> bool:
        return ContinousWorld.manhattan_dist(coord1, coord2) < (self.agent_size + self.agent_size)

    def check_collision(self, object: Object) -> bool:
        object_pos, agent_radius = object
        for pos, radius in self.block_coords:
            if ContinousWorld.manhattan_dist(object_pos, pos) < (radius + agent_radius):
                return True
        return False
        
    def sample_coords_within_dist(self, center : Position, min_dist_from_center : float, ignore_blocks=False, max_attempts=100) -> Position:
        import random
        for _ in range(max_attempts):
            angle = 2*math.pi*random.random()
            distance = min_dist_from_center*math.sqrt(random.random())

            x = center[0] + distance*math.cos(angle)
            y = center[1] + distance*math.sin(angle)
            yaw = 2*math.pi*random.random()
            new_coord : Position = (x,y,yaw)

            if not self.check_collision((new_coord, self.agent_size)):
                return new_coord
            
        raise Exception(f"Cannot sample coordinate within distance {min_dist_from_center} from {center} within {max_attempts}")

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
    
    @abstractmethod
    def check_collision_wall(self, coord: Position, line_distance: float, angle: float) -> Tuple[bool, Optional[float]]:
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
    
    def rotate_point_around_origin(self, point: Tuple[float, float], angle: float) -> Tuple[float, float]:
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


    def check_circle_line_intersection(self, cx : float, cy : float, r : float, ax : float, ay : float, bx: float, by : float) -> Optional[float]:
        # https://math.stackexchange.com/questions/275529/check-if-line-intersects-with-circles-perimeter
        og_start = ax, ay
        og_end = bx, by
        # // parameters: ax ay bx by cx cy r
        ax -= cx
        ay -= cy
        bx -= cx
        by -= cy

        a = (bx - ax)**2 + (by - ay)**2
        
        b = 2*(ax*(bx - ax) + ay*(by - ay))
        c = ax**2 + ay**2 - r**2
        disc = b**2 - 4*a*c

        
        if (disc <= 0):
            return None
        
        sqrtdisc = math.sqrt(disc)
        t1 = (-b + sqrtdisc)/(2*a)
        t2 = (-b - sqrtdisc)/(2*a)
  
        if (0 < t1 and t1 < 1):
            intersection_x = ax + t2 * (bx - ax)
            intersection_y = ay + t2 * (by - ay)
            distance_to_intersection = math.sqrt((intersection_x - ax)**2 + (intersection_y - ay)**2)
            return distance_to_intersection

        if (0 < t2 and t2 < 1):
            intersection_x = ax + t2 * (bx - ax)
            intersection_y = ay + t2 * (by - ay)
            distance_to_intersection = math.sqrt((intersection_x - ax)**2 + (intersection_y - ay)**2)
            return distance_to_intersection

        return None


    def check_collision_ray(self, coord: Position, line_distance : float, angle: float, other_agents : Tuple[Position, ...], skip_id : Optional[int] = None) -> Tuple[Optional[int], float]:
        
        closest_agent_pos = None
        closest_agent_distance = float('inf')
        closest_agent_index = None

        x, y, agent_angle = coord

        for index, (agent_pos) in enumerate(other_agents):

            if skip_id is not None and skip_id == index:
                continue

            other_agent_x, other_agent_y, _ = agent_pos

            end_x = x + line_distance * math.cos(angle + agent_angle)
            end_y = y + line_distance * math.sin(angle + agent_angle)

            dist = self.check_circle_line_intersection(other_agent_x, other_agent_y, self.agent_size, x,y,end_x, end_y)

            if dist is not None and dist < closest_agent_distance:
                closest_agent_distance = dist
                closest_agent_pos = agent_pos
                closest_agent_index = index



        if closest_agent_pos is None:
            _, closest_agent_distance = self.check_collision_wall(coord, line_distance, angle)
            if closest_agent_distance is not None:
                print("before return Wall", closest_agent_distance)


            if closest_agent_distance is not None:
                closest_agent_index = -1

        if closest_agent_distance is None:
            closest_agent_distance = line_distance

        return (closest_agent_index, closest_agent_distance)

        

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

    def check_collision_wall(self, coord: Position, line_distance: float, angle: float) -> Tuple[bool, Optional[float]]:
        """Checks if a line starting at position 'coord' and traveling at angle 'angle' relative to the agent's angle
        collides with the circle.
        
        Args:
            coord (Position): The starting position of the line.
            line_distance (float): The maximum distance the line should travel.
            angle (float): The angle at which the line is traveling relative to the agent's angle.
            
        Returns:
            A tuple containing a boolean value indicating whether there was a collision or not, and the distance to the
            wall if there was a collision, or None otherwise.
        """
        x, y, agent_angle = coord
        line_angle = agent_angle + angle

        end_x = x + line_distance * math.cos(line_angle)
        end_y = y + line_distance * math.sin(line_angle)

        # Check whether the line intersects with the circular boundary
        dx = end_x - x
        dy = end_y - y
        a = dx**2 + dy**2
        b = 2 * (dx*(x) + dy*(y))
        c = x**2 + y**2 - self.radius**2
        disc = b**2 - 4*a*c

        if disc < 0:
            return False, None
        else:
            sqrt_disc = math.sqrt(disc)
            t1 = (-b - sqrt_disc) / (2*a)
            t2 = (-b + sqrt_disc) / (2*a)

            if 0 <= t1 <= 1:
                x1 = x + t1*dx
                y1 = y + t1*dy
                return True, math.sqrt((x1 - x)**2 + (y1 - y)**2)
            elif 0 <= t2 <= 1:
                x1 = x + t2*dx
                y1 = y + t2*dy
                return True, math.sqrt((x1 - x)**2 + (y1 - y)**2)
            else:
                return False, None

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
        self.eps = 10e-3

    def coord_in_bounds(self, coord: Position) -> bool:
        return 0 <= coord[0] < self.width and 0 <= coord[1] < self.height

    def clamp_coords(self, coord: Position) -> Position:
        x, y, yaw = coord
        x = self.clamp(0 + self.eps, self.width - self.eps, x)
        y = self.clamp(0 + self.eps, self.height - self.eps, y)
        return (x, y, yaw)


    def check_collision_wall(self, coord: Position, line_distance: float, angle: float) -> Tuple[bool, Optional[float]]:
        """Checks if a line starting at position 'coord' and traveling at angle 'angle' relative to the agent's angle
        collides with the arena's walls.
        
        Args:
            coord (Position): The starting position of the line.
            line_distance (float): The maximum distance the line should travel.
            angle (float): The angle at which the line is traveling relative to the agent's angle.
            
        Returns:
            A tuple containing a boolean value indicating whether there was a collision or not, and the distance to the
            wall if there was a collision, or None otherwise.
        """

        x, y, agent_angle = coord
        line_angle = angle + agent_angle
        end_x = x + line_distance * math.cos(line_angle)
        end_y = y + line_distance * math.sin(line_angle)

        # Check whether it intersects with L/R boundary
        if end_x < 0 or end_x > self.width:
            if end_x < 0:
                x1 = 0
            else:
                x1 = self.width

            # Compute the y-coordinate where the intersection occurs
            y1 = (end_y - y) / (end_x - x) * (x1 - x) + y

            if 0 <= y1 <= self.height:
                # if math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2) > 0.9:
                    # import pdb
                    # pdb.set_trace()
                return True, math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)

        # Check whether it intersects with Top/Bottom boundary            
        if end_y < 0 or end_y > self.height:
            if end_y < 0:
                y1 = 0
            else:
                y1 = self.height

            # Compute the x-coordinate where the intersection occurs
            x1 = (end_x - x) / (end_y - y) * (y1 - y) + x

            if 0 <= x1 <= self.width:
                # if math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2) < 0.2:
                #     import pdb
                #     pdb.set_trace()
                return True, math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)

        return False, None
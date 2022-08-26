"""Functions and classes for rendering grid world environments.

This implemention is based on the gym-minigrid library:
- github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/minigrid.py
"""
import enum
from itertools import product
from typing import Tuple, Optional, List, Callable, Dict, Set, Union

import math
import numpy as np

from posggym.envs.grid_world.core import Direction, Coord, Grid


# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    'red': np.array([255, 0, 0]),
    'blue': np.array([0, 0, 255]),
    'green': np.array([0, 255, 0]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey': np.array([100, 100, 100]),
    'white': np.array([255, 255, 255]),
    'cyan': np.array([0, 195, 195]),
}

COLOR_TO_IDX = {
    'red': 0,
    'blue': 1,
    'green': 2,
    'purple': 3,
    'yellow': 4,
    'grey': 5,
    "white": 6,
    "cyan": 7
}

AGENT_COLORS = [
    'red', 'blue', 'green', 'purple', 'yellow', 'white', 'cyan'
]


def get_agent_color(agent_id: int) -> str:
    """Get color for agent."""
    return AGENT_COLORS[agent_id]


# Map of agent direction indices to vectors
# Ordering matches ordering in Direction enum
DIR_TO_VEC = [
    # NORTH (negative Y)
    np.array((0, -1)),
    # EAST (positive X)
    np.array((1, 0)),
    # SOUTH (positive Y)
    np.array((0, 1)),
    # WEST (negative X)
    np.array((-1, 0)),
]

DIR_TO_THETA = [
    # NORTH
    0.5*math.pi*2,
    # EAST
    0.5*math.pi*1,
    # SOUTH
    0.5*math.pi*0,
    # WEST
    0.5*math.pi*3
]


class Shape(enum.Enum):
    """An Object shape."""
    RECTANGLE = enum.auto()
    TRIANGLE = enum.auto()
    CIRCLE = enum.auto()
    LINE = enum.auto()


class GWObject:
    """An object in the grid world."""

    def __init__(self,
                 coord: Coord,
                 color: str,
                 shape: Shape,
                 direction: Optional[Direction] = None,
                 alpha: Optional[float] = None):
        if shape == Shape.TRIANGLE:
            assert direction is not None

        self.coord = coord
        self.color = color
        self.shape = shape
        self.direction = direction
        self.alpha = alpha

    def render(self, img: np.ndarray):
        """Draw object into image cell."""
        if self.shape == Shape.RECTANGLE:
            _fill_coords(img, _point_in_rect(0, 1, 0, 1), COLORS[self.color])
        elif self.shape == Shape.CIRCLE:
            _fill_coords(
                img, _point_in_circle(0.5, 0.5, 0.31), COLORS[self.color]
            )
        elif self.shape == Shape.TRIANGLE:
            tri_fn = _point_in_triangle(
                (0.12, 0.19), (0.87, 0.50), (0.12, 0.81)
            )
            theta = DIR_TO_THETA[self.direction]  # type: ignore
            tri_fn = _rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=theta)
            _fill_coords(img, tri_fn, COLORS[self.color])
        else:
            raise AttributeError("Unsupported Object shape: {self.shape}")

        if self.alpha is not None:
            _highlight_img(img, alpha=self.alpha)


class GWRenderer:
    """Handles generating grid world renders."""

    def __init__(self,
                 n_agents: int,
                 grid: Grid,
                 static_objs: List[GWObject],
                 render_blocks: bool = True):
        self._n_agents = n_agents
        self._grid = grid
        self._static_objs = static_objs
        self._width, self._height = grid.width, grid.height

        self._tiles: List[List[List[Optional[GWObject]]]] = []
        for x in range(self._width):
            self._tiles.append([])
            for y in range(self._height):
                self._tiles[x].append([])
                self._tiles[x][y].append(None)

        if render_blocks:
            for block_coord in grid.block_coords:
                block_obj = GWObject(block_coord, 'grey', Shape.RECTANGLE)
                self._tiles[block_coord[0]][block_coord[1]].append(block_obj)

        for obj in static_objs:
            self._tiles[obj.coord[0]][obj.coord[1]].append(obj)

    def _render_tile(self,
                     coords: Coord,
                     other_objs: List[GWObject],
                     observed: bool,
                     tile_size: int) -> np.ndarray:
        img = np.zeros(shape=(tile_size, tile_size, 3), dtype=np.uint8)

        # Draw the grid lines (tope and left edges)
        _fill_coords(img, _point_in_rect(0, 0.031, 0, 1), COLORS['grey'])
        _fill_coords(img, _point_in_rect(0, 1, 0, 0.031), COLORS['grey'])

        for obj in self._tiles[coords[0]][coords[1]]:
            if obj is not None:
                obj.render(img)

        for obj in other_objs:
            obj.render(img)

        if observed:
            _highlight_img(img)

        return img

    def _render_obj_tile(self,
                         obj: GWObject,
                         observed: bool,
                         tile_size: int) -> np.ndarray:
        img = np.zeros(shape=(tile_size, tile_size, 3), dtype=np.uint8)

        # Draw the grid lines (tope and left edges)
        _fill_coords(img, _point_in_rect(0, 0.031, 0, 1), COLORS['grey'])
        _fill_coords(img, _point_in_rect(0, 1, 0, 0.031), COLORS['grey'])

        obj.render(img)

        if observed:
            _highlight_img(img)

        return img

    def render(self,
               agent_coords: Tuple[Coord, ...],
               agent_obs_coords: Optional[Tuple[List[Coord], ...]],
               agent_dirs: Optional[Tuple[Direction, ...]],
               other_objs: Optional[List[GWObject]],
               agent_colors: Optional[Tuple[str, ...]] = None,
               tile_size: int = TILE_PIXELS) -> np.ndarray:
        """Generate Grid-World render.

        Returns a numpy array image representation of the grid world. This
        includes the agents inside the grid-world.
        """
        width_px = self._width * tile_size
        height_px = self._height * tile_size

        img = np.zeros(shape=(width_px, height_px, 3), dtype=np.uint8)

        observed_coords: Set[Coord] = set()
        if agent_obs_coords is not None:
            observed_coords.update(*agent_obs_coords)

        dynamic_objs_map: Dict[Coord, List[GWObject]] = {}
        if other_objs is None:
            other_objs = []

        for obj in other_objs:
            if obj.coord not in dynamic_objs_map:
                dynamic_objs_map[obj.coord] = []
            dynamic_objs_map[obj.coord].append(obj)

        if agent_dirs is None:
            agent_dirs = tuple(Direction.NORTH for _ in range(self._n_agents))

        if agent_colors is None:
            assert self._n_agents <= len(AGENT_COLORS), (
                "Agent colors must be specified when rendering envs with "
                f"more than {len(AGENT_COLORS)-1} agents."
            )
            agent_colors = tuple(
                AGENT_COLORS[i] for i in range(self._n_agents)
            )

        for coord, direction, color in zip(
                agent_coords, agent_dirs, agent_colors
        ):
            if coord not in dynamic_objs_map:
                dynamic_objs_map[coord] = []
            agent_obj = GWObject(coord, color, Shape.TRIANGLE, direction)
            dynamic_objs_map[coord].append(agent_obj)

        for x in range(self._width):
            for y in range(self._height):
                coord = (x, y)
                tile_img = self._render_tile(
                    coord,
                    dynamic_objs_map.get(coord, []),
                    coord in observed_coords,
                    tile_size
                )

                xmin, xmax = x * tile_size, (x+1) * tile_size
                ymin, ymax = y * tile_size, (y+1) * tile_size
                img[xmin:xmax, ymin:ymax, :] = tile_img

        img = img.transpose(1, 0, 2)
        return img

    def render_all_agent_obs(self,
                             env_img: np.ndarray,
                             agent_coords: Tuple[Coord, ...],
                             agent_dirs: Union[
                                 Direction, Tuple[Direction, ...]
                             ],
                             agent_obs_dims: Union[
                                 Tuple[int, int, int],
                                 Tuple[Tuple[int, int, int], ...]
                             ],
                             out_of_bounds_obj: GWObject,
                             agent_obs_coords: Optional[
                                 Tuple[List[Coord], ...]
                             ] = None,
                             tile_size: int = TILE_PIXELS
                             ) -> Tuple[np.ndarray, ...]:
        """Generate seperate render for a each grid-world agent.

        Returns a numpy array image representation of the agents observation
        of the grid world for each agent.

        Assumes env_img was render generated by the Renderer.render function.
        """
        agent_obs_imgs = []
        for i in range(self._n_agents):
            if isinstance(agent_dirs, tuple):
                agent_dir = agent_dirs[i]
            else:
                agent_dir = agent_dirs

            if isinstance(agent_obs_dims[0], tuple):
                obs_dims = agent_obs_dims[i]
            else:
                obs_dims = agent_obs_dims    # type: ignore

            if isinstance(agent_obs_coords, tuple):
                obs_coords = agent_obs_coords[i]
            else:
                obs_coords = None     # type: ignore

            agent_img = self.render_agent_obs(
                env_img,
                agent_coords[i],
                agent_dir,
                obs_dims,                   # type: ignore
                out_of_bounds_obj,
                obs_coords,
                tile_size
            )
            agent_obs_imgs.append(agent_img)
        return tuple(agent_obs_imgs)

    def render_agent_obs(self,
                         env_img: np.ndarray,
                         agent_coord: Coord,
                         agent_dir: Direction,
                         agent_obs_dims: Tuple[int, int, int],
                         out_of_bounds_obj: GWObject,
                         agent_obs_coords: Optional[List[Coord]] = None,
                         tile_size: int = TILE_PIXELS) -> np.ndarray:
        """Generate render for a single grid-world agent.

        Arguments
        ---------
        env_img : np.ndarray
            image render of entire environment assumed to be generated by the
            Renderer.render function.
        agent_coord: Coord
            Coordinates of agent within the environment grid
        agent_obs_dims: (int, int, int)
            Dimensions of agents observation in terms of number of cells
            in front, behind and to each side.
        agent_dir: Direction
            Direction the agent is facing within the environment grid.
        out_of_bounds_obj: GWObject
            Render object to use for cells in observation that are outside of
            environment grid (e.g. when agent is at the boundary of the
            environment)
        agent_obs_coords: Optional[List[Coord]]
            Optional list of environment coordinates for cells observed by
            agent. If provided, then treats all unobserved cells as out of
            bounds. If None, then assumes all coordinates in agents observation
            dims are observed.
        tile_size: int
            Defines the resolution of the image.

        Returns a numpy array image representation of the agents observation
        of the grid world.

        """
        # Reverse transpose to handle coord system properly
        env_img = env_img.transpose(1, 0, 2)
        env_width = env_img.shape[0] / tile_size
        env_height = env_img.shape[1] / tile_size

        obs_x_dim = (2 * agent_obs_dims[2]) + 1
        obs_y_dim = agent_obs_dims[0] + agent_obs_dims[1] + 1

        obs_width_px = obs_x_dim * tile_size
        obs_height_px = obs_y_dim * tile_size
        obs_img = np.zeros(
            shape=(obs_width_px, obs_height_px, 3), dtype=np.uint8
        )

        for (obs_x, obs_y) in product(range(obs_x_dim), range(obs_y_dim)):
            # Need to map to env (x, y)
            env_x, env_y = self._map_obs_to_env_coord(
                (obs_x, obs_y),
                agent_coord,
                agent_dir,
                agent_obs_forward_dim=agent_obs_dims[0],
                agent_obs_side_dim=agent_obs_dims[2]
            )

            cell_out_of_bounds = (
                not 0 <= env_x < env_width or not 0 <= env_y < env_height
            )
            cell_observed = (
                agent_obs_coords is None or (env_x, env_y) in agent_obs_coords
            )

            if cell_out_of_bounds or not cell_observed:
                tile_img = self._render_obj_tile(
                    out_of_bounds_obj, False, tile_size
                )
            else:
                env_xmin, env_xmax = env_x * tile_size, (env_x+1) * tile_size
                env_ymin, env_ymax = env_y * tile_size, (env_y+1) * tile_size
                tile_img = env_img[env_xmin:env_xmax, env_ymin:env_ymax, :]

                # rotate tile so its oriented to agent's viewpoint
                # function rotates counter-clockwise
                k = (len(Direction) - agent_dir) % len(Direction)
                tile_img = np.rot90(tile_img, k)

            obs_xmin, obs_xmax = obs_x * tile_size, (obs_x+1) * tile_size
            obs_ymin, obs_ymax = obs_y * tile_size, (obs_y+1) * tile_size
            obs_img[obs_xmin:obs_xmax, obs_ymin:obs_ymax, :] = tile_img

        # Needed to swap to formate expected by matplotlib.imshow function
        obs_img = obs_img.transpose(1, 0, 2)
        return obs_img

    def _map_obs_to_env_coord(self,
                              obs_coord: Coord,
                              agent_coord: Coord,
                              facing_dir: Direction,
                              agent_obs_forward_dim: int,
                              agent_obs_side_dim: int) -> Coord:
        grid_row = agent_coord[1]
        grid_col = agent_coord[0]

        if facing_dir == Direction.NORTH:
            grid_row += (obs_coord[1] - agent_obs_forward_dim)
            grid_col += (obs_coord[0] - agent_obs_side_dim)
        elif facing_dir == Direction.EAST:
            grid_row += (obs_coord[0] - agent_obs_side_dim)
            grid_col += (-obs_coord[1] + agent_obs_forward_dim)
        elif facing_dir == Direction.SOUTH:
            grid_row += (-obs_coord[1] + agent_obs_forward_dim)
            grid_col += (-obs_coord[0] + agent_obs_side_dim)
        else:
            grid_row += (-obs_coord[0] + agent_obs_side_dim)
            grid_col += (obs_coord[1] - agent_obs_forward_dim)
        return (grid_col, grid_row)


def _fill_coords(img: np.ndarray,
                 fn: Callable,
                 color: np.ndarray) -> np.ndarray:
    """Fill pixels of an image with coordinates matching a filter function."""
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color

    return img


def _rotate_fn(fin: Callable, cx: float, cy: float, theta: float) -> Callable:
    def _fout(x, y):
        x = x - cx
        y = y - cy

        x2 = cx + x * math.cos(-theta) - y * math.sin(-theta)
        y2 = cy + y * math.cos(-theta) + x * math.sin(-theta)

        return fin(x2, y2)

    return _fout


def _point_in_line(x0: float,
                   y0: float,
                   x1: float,
                   y1: float,
                   r: float) -> Callable:
    p0 = np.array([x0, y0])
    p1 = np.array([x1, y1])
    direction = p1 - p0
    dist = np.linalg.norm(direction)
    direction = direction / dist

    xmin = min(x0, x1) - r
    xmax = max(x0, x1) + r
    ymin = min(y0, y1) - r
    ymax = max(y0, y1) + r

    def _fn(x, y):
        # Fast, early escape test
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False

        q = np.array([x, y])
        pq = q - p0

        # Closest point on line
        a = np.dot(pq, direction)
        a = np.clip(a, 0, dist)
        p = p0 + a * direction

        dist_to_line = np.linalg.norm(q - p)
        return dist_to_line <= r

    return _fn


def _point_in_circle(cx, cy, r):
    def _fn(x, y):
        return (x-cx)*(x-cx) + (y-cy)*(y-cy) <= r * r
    return _fn


def _point_in_rect(xmin: float,
                   xmax: float,
                   ymin: float,
                   ymax: float) -> Callable:
    def _fn(x, y):
        return xmin <= x <= xmax and ymin <= y <= ymax
    return _fn


def _point_in_triangle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    def _fn(x, y):
        v0 = c - a
        v1 = b - a
        v2 = np.array((x, y)) - a

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v) < 1

    return _fn


def _highlight_img(img: np.ndarray, color=(255, 255, 255), alpha=0.2):
    """Add highlighting to an image."""
    blend_img = img + alpha * (np.array(color, dtype=np.uint8) - img)
    blend_img = blend_img.clip(0, 255).astype(np.uint8)
    img[:, :, :] = blend_img

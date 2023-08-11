"""Functions and classes for rendering grid world environments."""
import abc
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from posggym.envs.grid_world.core import Coord, Direction, Grid
from posggym.error import DependencyNotInstalled

ColorTuple = Union[Tuple[int, int, int], Tuple[int, int, int, int]]

try:
    import pygame
except ImportError as e:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install posggym[grid-world]`"
    ) from e


# (Main, Alternative) agent colors
AGENT_COLORS = [
    (pygame.colordict.THECOLORS["red"], pygame.colordict.THECOLORS["red3"]),
    (pygame.colordict.THECOLORS["blue"], pygame.colordict.THECOLORS["blue3"]),
    (pygame.colordict.THECOLORS["green"], pygame.colordict.THECOLORS["green3"]),
    (pygame.colordict.THECOLORS["purple"], pygame.colordict.THECOLORS["purple3"]),
    (pygame.colordict.THECOLORS["yellow"], pygame.colordict.THECOLORS["yellow3"]),
    (pygame.colordict.THECOLORS["cyan"], pygame.colordict.THECOLORS["cyan3"]),
    (pygame.colordict.THECOLORS["hotpink"], pygame.colordict.THECOLORS["hotpink3"]),
    (pygame.colordict.THECOLORS["orange"], pygame.colordict.THECOLORS["orange3"]),
]


def get_agent_color(agent_id: str) -> Tuple[ColorTuple, ColorTuple]:
    """Get color for agent."""
    return AGENT_COLORS[int(agent_id)]


def get_color(color_name: str) -> ColorTuple:
    """Get color from name."""
    return pygame.colordict.THECOLORS[color_name]


def load_img_file(img_path: str, cell_size: Tuple[int, int]):
    """Load an image from file and scale it to cell size."""
    return pygame.transform.scale(pygame.image.load(img_path), cell_size)


def get_default_font_size(cell_size: Tuple[int, int]) -> int:
    """Get the default font size based on cell size."""
    return cell_size[1] // 4


def load_font(font_name: str, font_size: int) -> pygame.font.Font:
    """Load font."""
    return pygame.font.SysFont(font_name, font_size)


class GWObject(abc.ABC):
    """An object in the grid world."""

    def __init__(
        self,
        coord: Coord,
        cell_size: Tuple[int, int],
    ):
        self.coord = coord
        self.cell_size = cell_size

    @property
    def pos(self) -> Tuple[int, int]:
        """The (x, y) position of the object on render surface."""
        return (self.coord[0] * self.cell_size[0], self.coord[1] * self.cell_size[1])

    @abc.abstractmethod
    def render(self, surface: pygame.Surface):
        """Render object on surface."""


class GWRectangle(GWObject):
    """A rectangle in the grid world."""

    def __init__(
        self,
        coord: Coord,
        cell_size: Tuple[int, int],
        color: ColorTuple,
    ):
        super().__init__(coord, cell_size)
        self.color = color

    def render(self, surface: pygame.Surface):
        rect = (*self.pos, *self.cell_size)
        surface.fill(self.color, rect=rect)


class GWTriangle(GWObject):
    """A equilateral triangle in the grid world."""

    def __init__(
        self,
        coord: Coord,
        cell_size: Tuple[int, int],
        color: ColorTuple,
        facing_dir: Direction,
    ):
        super().__init__(coord, cell_size)
        self.color = color
        self.facing_dir = facing_dir

    def _get_triangle_by_direction(self, facing_dir: Direction):
        # amount to shrink points away from cell edges, so they fit better
        delta = (int(0.05 * self.cell_size[0]), int(0.05 * self.cell_size[1]))

        max_pos = (
            self.pos[0] + self.cell_size[0] - delta[0],
            self.pos[1] + self.cell_size[1] - delta[1],
        )
        min_pos = (self.pos[0] + delta[0], self.pos[1] + delta[1])

        if facing_dir == Direction.NORTH:
            points = (
                (min_pos[0] + self.cell_size[0] // 2, min_pos[1]),
                (min_pos[0], max_pos[1]),
                (max_pos[0], max_pos[1]),
            )
        elif facing_dir == Direction.SOUTH:
            points = (
                (min_pos[0] + self.cell_size[0] // 2, max_pos[1]),
                (min_pos[0], min_pos[1]),
                (max_pos[0], min_pos[1]),
            )
        elif facing_dir == Direction.EAST:
            points = (
                (max_pos[0], min_pos[1] + self.cell_size[1] // 2),
                (min_pos[0], min_pos[1]),
                (min_pos[0], max_pos[1]),
            )
        elif facing_dir == Direction.WEST:
            points = (
                (min_pos[0], min_pos[1] + self.cell_size[1] // 2),
                (max_pos[0], min_pos[1]),
                (max_pos[0], max_pos[1]),
            )

        return points

    def render(self, surface: pygame.Surface):
        points = self._get_triangle_by_direction(self.facing_dir)
        pygame.draw.polygon(surface, self.color, points)


class GWCircle(GWObject):
    """A circle in the grid world."""

    def __init__(
        self,
        coord: Coord,
        cell_size: Tuple[int, int],
        color: ColorTuple,
    ):
        super().__init__(coord, cell_size)
        self.color = color

    def render(self, surface: pygame.Surface):
        radius = int(0.95 * self.cell_size[0]) // 2
        center = (
            self.pos[0] + self.cell_size[0] // 2,
            self.pos[1] + self.cell_size[1] // 2,
        )
        pygame.draw.circle(surface, self.color, center, radius)


class GWHighlight(GWObject):
    """A transparent rectangle for highlighting a cell in the grid world."""

    def __init__(self, coord: Coord, cell_size: Tuple[int, int], alpha: float = 0.25):
        super().__init__(coord, cell_size)
        self.alpha = alpha
        self.surface = pygame.Surface(cell_size, pygame.SRCALPHA)
        self.surface.fill((255, 255, 255, int(alpha * 255)))

    def render(self, surface: pygame.Surface):
        surface.blit(self.surface, self.pos)


class GWImage(GWObject):
    """An image object in the grid world."""

    def __init__(
        self,
        coord: Coord,
        cell_size: Tuple[int, int],
        img: pygame.Surface,
    ):
        super().__init__(coord, cell_size)
        self.img = img

    def render(self, surface: pygame.Surface):
        surface.blit(self.img, self.pos)


class GWText(GWObject):
    """An text object in the grid world."""

    def __init__(
        self,
        coord: Coord,
        cell_size: Tuple[int, int],
        text: str,
        font: pygame.font.Font,
    ):
        super().__init__(coord, cell_size)
        self.text = text
        self.font = font

    def render(self, surface: pygame.Surface):
        text_img = self.font.render(self.text, True, (255, 255, 255), (0, 0, 0))
        # +1 so text is in offset from corner of cell
        text_pos = (self.pos[0] + 1, self.pos[1] + 1)
        surface.blit(text_img, text_pos)


class GWImageAndText(GWObject):
    """A combined image and text grid-world object."""

    def __init__(
        self,
        coord: Coord,
        cell_size: Tuple[int, int],
        img: pygame.Surface,
        text: str,
        font: pygame.font.Font,
    ):
        super().__init__(coord, cell_size)
        self.img = img
        self.text = text
        self.font = font

    def render(self, surface: pygame.Surface):
        surface.blit(self.img, self.pos)

        text_img = self.font.render(self.text, True, (255, 255, 255), (0, 0, 0))
        # +1 so text is in offset from corner of cell
        text_pos = (self.pos[0] + 1, self.pos[1] + 1)
        surface.blit(text_img, text_pos)


class GWRenderer:
    """Handles generating grid world renders."""

    def __init__(
        self,
        render_mode: str,
        grid: Grid,
        render_fps: int = 30,
        env_name: str = "",
        bg_color: ColorTuple = (0, 0, 0),
        grid_line_color: ColorTuple = (255, 255, 255),
        block_color: ColorTuple = (131, 139, 139),
    ):
        self.render_mode = render_mode
        self.grid = grid
        self.render_fps = render_fps

        self.bg_color = bg_color
        self.grid_line_color = grid_line_color
        self.block_color = block_color

        self.window_size = (min(64 * grid.width, 512), min(64 * grid.height, 512))
        self.cell_size = (
            self.window_size[0] // grid.width,
            self.window_size[1] // grid.height,
        )

        self.blocks = [
            GWRectangle(coord, self.cell_size, self.block_color)
            for coord in grid.block_coords
        ]
        # list of static objects user can add to
        self.static_objects: List[GWObject] = []

        pygame.init()
        if render_mode == "human":
            pygame.display.init()
            pygame.display.set_caption(env_name)
            self.window_surface = pygame.display.set_mode(
                self.window_size, pygame.SRCALPHA
            )
        else:
            assert render_mode.startswith("rgb_array")
            pygame.font.init()
            self.window_surface = pygame.Surface(self.window_size, pygame.SRCALPHA)

        self.clock = pygame.time.Clock()

    def _reset_surface(self):
        self.window_surface.fill(self.bg_color)
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)
                pygame.draw.rect(self.window_surface, self.grid_line_color, rect, 1)

        for block in self.blocks:
            block.render(self.window_surface)

        for static_object in self.static_objects:
            static_object.render(self.window_surface)

    def reset_blocks(self):
        """Reset the blocks."""
        self.blocks = [
            GWRectangle(coord, self.cell_size, self.block_color)
            for coord in self.grid.block_coords
        ]

    def render(
        self, objects: List[GWObject], observed_coords: Optional[List[Coord]] = None
    ) -> Optional[np.ndarray]:
        """Generate Grid-World render."""
        self._reset_surface()
        for obj in objects:
            obj.render(self.window_surface)

        if observed_coords is None:
            observed_coords = []

        highlight_obj = GWHighlight((0, 0), self.cell_size)
        for coord in observed_coords:
            highlight_obj.coord = coord
            highlight_obj.render(self.window_surface)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.render_fps)
            return None

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
        )

    def render_agents(
        self,
        objects: List[GWObject],
        agent_coords_and_dirs: Dict[str, Tuple[Coord, Direction]],
        agent_obs_dims: Union[int, Tuple[int, int, int, int]],
        observed_coords: Optional[List[Coord]] = None,
        agent_obs_mask: Optional[List[Coord]] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate environment and agent-centric grid-world renders."""
        if agent_obs_mask is None:
            agent_obs_mask = []
        if observed_coords is None:
            observed_coords = []
        if isinstance(agent_obs_dims, int):
            d = agent_obs_dims
            agent_obs_dims = (d, d, d, d)

        self._reset_surface()
        for obj in objects:
            obj.render(self.window_surface)

        env_array = np.array(pygame.surfarray.pixels3d(self.window_surface))

        array_dict: Dict[str, np.ndarray] = {}
        for i, (coord, facing_dir) in agent_coords_and_dirs.items():
            # 1. get agent's view of env
            # (min_col, max_col, min_row, max_row) of coords in grid that agent observed
            view_coords_rect = self.grid.get_rectangular_bounds(
                coord, facing_dir, agent_obs_dims
            )
            view_pos_rect = (
                view_coords_rect[0] * self.cell_size[0],
                (view_coords_rect[1] + 1) * self.cell_size[0],
                view_coords_rect[2] * self.cell_size[1],
                (view_coords_rect[3] + 1) * self.cell_size[1],
            )
            agent_array = env_array[
                view_pos_rect[0] : view_pos_rect[1], view_pos_rect[2] : view_pos_rect[3]
            ]

            # 2. pad view with blocks so agent's view matches their obs_dims
            pad_coords = self.grid.get_rectangular_padding(
                coord, facing_dir, agent_obs_dims
            )
            pad_pos = [
                (
                    pad_coords[0][0] * self.cell_size[0],
                    pad_coords[0][1] * self.cell_size[0],
                ),
                (
                    pad_coords[1][0] * self.cell_size[1],
                    pad_coords[1][1] * self.cell_size[1],
                ),
                # padding for color axes (i.e no padding)
                (0, 0),
            ]
            agent_array = np.pad(
                agent_array,
                pad_width=pad_pos,
                mode="constant",
                constant_values=[
                    (self.block_color[0], self.block_color[0]),
                    (self.block_color[1], self.block_color[1]),
                    (self.block_color[2], self.block_color[2]),
                ],
            )

            # 3. apply masking if necessary
            for coord in agent_obs_mask:
                pos = (coord[0] * self.cell_size[0], coord[1] * self.cell_size[1])
                agent_array[
                    pos[0] : pos[0] + self.cell_size[0],
                    pos[1] : pos[1] + self.cell_size[1],
                ] = self.bg_color[:3]

            # 3. rotate agent's array so it is ego-centric
            agent_array = np.transpose(agent_array, axes=(1, 0, 2))
            # just so happens int value of direction lines up with # rotations needed
            agent_array = np.rot90(agent_array, k=int(facing_dir))

            array_dict[i] = agent_array

        # Add obs highlighting after generating agent-centric views
        highlight_obj = GWHighlight((0, 0), self.cell_size)
        for coord in observed_coords:
            highlight_obj.coord = coord
            highlight_obj.render(self.window_surface)

        # finally add env image, including highlighting
        array_dict["env"] = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
        )

        return array_dict

    def close(self):
        """Close renderer and perform any necessary cleanup."""
        pygame.display.quit()
        pygame.quit()

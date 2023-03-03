"""Functions and classes for rendering grid world environments."""
import abc
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import math
from posggym.envs.grid_world.core import Coord, Direction, Grid
from posggym.error import DependencyNotInstalled
from posggym.model import AgentID


ColorTuple = Union[Tuple[int, int, int], Tuple[int, int, int, int]]

try:
    import pygame
except ImportError as e:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install posggym[grid-world]`"
    ) from e


class GWContinousRender:
    def __init__(self,
        render_mode: str,
        grid: Grid,
        render_fps: int = 30,
        env_name: str = "",
        bg_color: ColorTuple = (0, 0, 0),
        grid_line_color: ColorTuple = (255, 255, 255),
        block_color: ColorTuple = (131, 139, 139),
        screen_width=640, screen_height=480, arena_size=400, agent_size=20, domain_size=1,
        num_colors: int = 10):
        # Initialize Pygame
        pygame.init()

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Triangles in Square Arena")

        # Set up the colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)

        # Set up the agent size
        self.agent_size = agent_size

        # Set up the arena position and size
        self.arena_size = arena_size
        self.arena_x = (screen_width - arena_size) // 2
        self.arena_y = (screen_height - arena_size) // 2

        self.domain_size = domain_size

        self.colors = self.generate_colors(num_colors)

    def generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        # Generate unique colors for each agent
        colors = []
        for i in range(num_colors):
            r = (i * 55) % 255
            g = (i * 155) % 255
            b = (i * 205) % 255
            colors.append((r, g, b))
        return colors



    def render(self, agents : Tuple[Tuple[float, float, float, int], ...]):

        scaled_agents = []
        for agent in agents:
            x, y, angle, color = agent
            scaled_x = int((x / self.domain_size) * self.arena_size) + self.arena_x
            scaled_y = int((y / self.domain_size) * self.arena_size) + self.arena_y
            scaled_agents.append((scaled_x, scaled_y, angle, color))

        self.screen.fill(self.WHITE)

        # Draw the arena
        arena_rect = pygame.Rect(self.arena_x, self.arena_y, self.arena_size, self.arena_size)
        pygame.draw.rect(self.screen, self.BLACK, arena_rect, width=1)

        # Draw the agents
        for agent in scaled_agents:
            x, y, angle, color = agent
            half_width = self.agent_size / 2
            height = self.agent_size * math.sqrt(3) / 2
            tri_points = [
                (x + half_width * math.cos(angle), y + half_width * math.sin(angle)),
                (x + half_width * math.cos(angle + 2*math.pi/3), y + half_width * math.sin(angle + 2*math.pi/3)),
                (x + half_width * math.cos(angle - 2*math.pi/3), y + half_width * math.sin(angle - 2*math.pi/3))
            ]
            pygame.draw.polygon(self.screen, self.colors[color % len(self.colors)], tri_points)

        # Update the screen
        pygame.display.flip()







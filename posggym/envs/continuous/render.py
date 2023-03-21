"""Functions and classes for rendering grid world environments."""
from typing import Dict, List, Optional, Tuple, Union
import math
from posggym.error import DependencyNotInstalled
from posggym.model import AgentID
from posggym.envs.continuous.core import ArenaTypes, Object

ColorTuple = Union[Tuple[int, int, int], Tuple[int, int, int, int]]

try:
    import pygame
except ImportError as e:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install posggym[grid-world]`"
    ) from e


class GWContinuousRender:
    def __init__(
        self,
        render_mode: str,
        env_name: str,
        arena_type: ArenaTypes = ArenaTypes.Square,
        render_fps: int = 15,
        screen_width=640,
        screen_height=480,
        arena_size=400,
        agent_size=20,
        domain_min=0,
        domain_max=1,
        num_colors: int = 10,
    ):
        # Initialize Pygame
        pygame.init()

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(env_name)

        # Set up the colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)

        self.arena_type = arena_type

        self.render_fps = render_fps

        # Set up the agent size
        self.agent_size = agent_size

        # Set up the arena position and size
        self.arena_size = arena_size
        self.arena_x = (screen_width - arena_size) // 2
        self.arena_y = (screen_height - arena_size) // 2

        self.min_domain_size = domain_min
        self.max_domain_size = domain_max
        self.clock = pygame.time.Clock()

        self.colors = self.generate_colors(num_colors)

    def generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        # Generate unique colors for each agent
        colors = []
        for i in range(num_colors):
            r = ((i + 1) * 55) % 255
            g = ((i + 1) * 155) % 255
            b = ((i + 1) * 205) % 255
            colors.append((r, g, b))
        return colors

    # https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
    def pairwise(self, iterable):
        "s -> (s0, s1), (s2, s3), (s4, s5), ..."
        a = iter(iterable)
        return zip(a, a)

    def render_lines(
        self,
        lines: Dict[AgentID, Tuple[Union[list, float]]],
        agents: Tuple[Tuple[float, float, float], ...],
    ):
        for index, agent_pos in enumerate(agents):
            x, y, agent_angle = agent_pos
            current_lines = lines[str(index)]

            for index, (type, distance) in enumerate(self.pairwise(current_lines)):
                angle = 2 * math.pi * index / (len(current_lines) // 2)
                angle += agent_angle

                end_pos = (
                    x + distance * math.cos(angle),
                    y + distance * math.sin(angle),
                )

                scaled_start = self.scale(agent_pos[0], agent_pos[1])
                scaled_end = self.scale(end_pos[0], end_pos[1])

                pygame.draw.line(
                    self.screen, self.colors[type], scaled_start, scaled_end
                )

    def scale(self, x: float, y: float) -> Tuple[float, float]:
        OldRange = self.max_domain_size - self.min_domain_size
        NewRange = self.arena_size * 2

        scaled_x = int(
            (((x - self.min_domain_size) * NewRange) / OldRange)
            + (self.arena_x - self.arena_size / 2)
        )
        scaled_y = int(
            (((y - self.min_domain_size) * NewRange) / OldRange)
            + (self.arena_y - self.arena_size / 2)
        )
        return (scaled_x, scaled_y)

    def scale_number(self, num: float) -> float:
        return num / (self.max_domain_size - self.min_domain_size) * self.arena_size * 2

    def draw_circle_alpha(self, surface, color, center, radius):
        target_rect = pygame.Rect(center, (0, 0)).inflate((radius * 2, radius * 2))
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(shape_surf, color, (radius, radius), radius)
        surface.blit(shape_surf, target_rect)

    def draw_blocks(self, blocks: List[Object]):
        for block in blocks:
            (x, y, _), radius = block

            scaled_x, scaled_y = self.scale(x, y)

            radius = self.scale_number(radius)

            pygame.draw.circle(self.screen, self.BLACK, (scaled_x, scaled_y), radius)

    def draw_agents(
        self,
        agents: Tuple[Tuple[float, float, float, int], ...],
        is_holonomic: Optional[List[bool]] = None,
        sizes: Optional[List[Optional[float]]] = None,
    ):
        scaled_agents = []
        if sizes is None:
            sizes = list([None] * len(agents))

        for i, agent in enumerate(agents):
            x, y, angle, color = agent

            scaled_x, scaled_y = self.scale(x, y)

            sizes[i] = (
                self.agent_size
                if sizes[i] is None
                else int(self.scale_number(sizes[i]))
            )

            scaled_agents.append((scaled_x, scaled_y, angle, color))

        # Draw the arena
        if self.arena_type == ArenaTypes.Square:
            arena_rect = pygame.Rect(
                self.arena_x - self.arena_size / 2,
                self.arena_y - self.arena_size / 2,
                self.arena_size * 2,
                self.arena_size * 2,
            )
            pygame.draw.rect(self.screen, self.BLACK, arena_rect, width=1)
        else:
            center_x = self.arena_x + self.arena_size / 2
            center_y = self.arena_y + self.arena_size / 2
            pygame.draw.circle(
                self.screen, self.BLACK, (center_x, center_y), self.arena_size, width=1
            )

        # Draw the agents
        for i, agent in enumerate(scaled_agents):
            x, y, angle, color = agent

            size = sizes[i]

            if is_holonomic is not None and is_holonomic[i]:
                pygame.draw.circle(
                    self.screen, self.colors[color % len(self.colors)], (x, y), size
                )
            else:
                half_width = size
                tri_points = [
                    (
                        x + half_width * math.cos(angle),
                        y + half_width * math.sin(angle),
                    ),
                    (
                        x + half_width * 0.5 * math.cos(angle + 2 * math.pi / 3),
                        y + half_width * 0.5 * math.sin(angle + 2 * math.pi / 3),
                    ),
                    (
                        x + half_width * 0.5 * math.cos(angle - 2 * math.pi / 3),
                        y + half_width * 0.5 * math.sin(angle - 2 * math.pi / 3),
                    ),
                ]
                pygame.draw.polygon(
                    self.screen, self.colors[color % len(self.colors)], tri_points
                )

                self.draw_circle_alpha(
                    self.screen,
                    self.colors[color % len(self.colors)] + (100,),
                    (x, y),
                    size,
                )

    def clear_render(self):
        self.screen.fill(self.WHITE)

    def render(self):
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.render_fps)
        return None
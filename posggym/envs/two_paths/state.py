"""State in the Two-Paths Problem """
import posggym.envs.two_paths.grid as grid_lib


class TPState:
    """A state in the Two-Paths Problem """

    def __init__(self,
                 runner_loc: grid_lib.Loc,
                 chaser_loc: grid_lib.Loc,
                 grid: grid_lib.TPGrid):
        self.runner_loc = runner_loc
        self.chaser_loc = chaser_loc
        self.grid = grid

    def get_ascii_repr(self):
        """Get ASCII representation of state """
        grid_repr = self.grid.get_ascii_repr()

        runner_coords = self.grid.loc_to_coord(self.runner_loc)
        chaser_coords = self.grid.loc_to_coord(self.chaser_loc)
        if self.runner_loc == self.chaser_loc:
            grid_repr[runner_coords[0]][runner_coords[1]] = "X"
        else:
            grid_repr[runner_coords[0]][runner_coords[1]] = "R"
            grid_repr[chaser_coords[0]][chaser_coords[1]] = "C"

        return (
            str(self)
            + "\n" + "\n".join(list(list((" ".join(r) for r in grid_repr))))
        )

    def __str__(self):
        return f"<{self.runner_loc}, {self.chaser_loc}>"

    def __hash__(self):
        return hash((self.runner_loc, self.chaser_loc))

    def __eq__(self, other):
        return (
            self.runner_loc == other.runner_loc
            and self.chaser_loc == other.chaser_loc
        )

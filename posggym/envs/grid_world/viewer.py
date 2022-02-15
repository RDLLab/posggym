"""Class for displaying renders of grid-world environments

The Viewer is seperated from the rendering functionality, in case users want
to use the renders without displaying them, and to seperate out dependencies
in that case.

This implemention is based on the gym-minigrid library:
- github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/minigrid.py
"""
import sys
from typing import Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('To display the environment in a window, please install matplotlib.')
    sys.exit(-1)


class GWViewer:
    """Handles displaying grid world renders """

    def __init__(self, title: str, figsize: Tuple[int, int] = (9, 9)):
        self._title = title
        self._figsize = figsize

        self._fig, self._ax = plt.subplots(
            1, 1, figsize=figsize, num=title
        )
        self._imshow_obj = None

        # Turn off x/y axis numbering/ticks
        self._ax.xaxis.set_ticks_position('none')
        self._ax.yaxis.set_ticks_position('none')
        self._ax.set_xticklabels([])
        self._ax.set_yticklabels([])

        # Flag indicating window was closed
        self._closed = False

        # pylint: disable=[unused-argument]
        def close_handler(event):
            self._closed = True

        self._fig.canvas.mpl_connect('close_event', close_handler)

    def display_img(self, img: np.ndarray):
        """Show an image or update the image being shown """

        # Show the first image of the environment
        if self._imshow_obj is None:
            self._imshow_obj = self._ax.imshow(
                img, interpolation='bilinear', origin='upper'
            )

        self._imshow_obj.set_data(img)  # type: ignore
        self._fig.canvas.draw()

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        plt.pause(0.01)

    @staticmethod
    def show(block: bool = True) -> None:
        """ Show the window, and start an event loop """
        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interative mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    def close(self) -> None:
        """ Close the window """
        plt.close(self._fig)
        self._closed = True

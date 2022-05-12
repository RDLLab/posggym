"""Class for displaying renders of grid-world environments.

The Viewer is seperated from the rendering functionality, in case users want
to use the renders without displaying them, and to seperate out dependencies
in that case.

This implemention is based on the gym-minigrid library:
- github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/minigrid.py
"""
import sys
from typing import Tuple, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    print('To display the environment in a window, please install matplotlib.')
    sys.exit(-1)


class GWViewer:
    """Handles displaying grid world renders."""

    def __init__(self,
                 title: str,
                 figsize: Tuple[int, int] = (9, 9),
                 num_agent_displays: Optional[int] = None):
        self._title = title
        self._figsize = figsize
        self._num_agent_displays = num_agent_displays

        self._fig = plt.figure(
            num=title, figsize=figsize, constrained_layout=True
        )

        if num_agent_displays is None:
            spec = gridspec.GridSpec(1, 1, figure=self._fig)
        else:
            spec = gridspec.GridSpec(num_agent_displays, 2, figure=self._fig)

        # Main world view is entire first column
        self._main_ax = self._fig.add_subplot(spec[:, 0])
        self._main_imshow_obj = None

        self._agent_axs = []
        self._agent_imshow_objs = []              # type: ignore
        if num_agent_displays is not None:
            for i in range(num_agent_displays):
                ax = self._fig.add_subplot(spec[i, 1])
                self._agent_axs.append(ax)
                self._agent_imshow_objs.append(None)

        for ax in [self._main_ax] + self._agent_axs:
            # Turn off x/y axis numbering/ticks
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        # Flag indicating window was closed
        self._closed = False

        # pylint: disable=[unused-argument]
        def close_handler(event):
            self._closed = True

        self._fig.canvas.mpl_connect('close_event', close_handler)

    def display_img(self, img: np.ndarray, agent_idx: Optional[int] = None):
        """Show an image or update the image being shown."""
        if agent_idx is None:
            ax = self._main_ax
            imshow_obj = self._main_imshow_obj
        else:
            ax = self._agent_axs[agent_idx]
            imshow_obj = self._agent_imshow_objs[agent_idx]

        # Show the first image of the environment
        if imshow_obj is None:
            imshow_obj = ax.imshow(
                img, interpolation='bilinear', origin='upper'
            )

            if agent_idx is None:
                self._main_imshow_obj = imshow_obj
            else:
                self._agent_imshow_objs[agent_idx] = imshow_obj

        imshow_obj.set_data(img)  # type: ignore
        self._fig.canvas.draw()

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        plt.pause(0.01)

    @staticmethod
    def show(block: bool = True) -> None:
        """Show the window, and start an event loop."""
        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interative mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    def close(self) -> None:
        """Close the window."""
        plt.close(self._fig)
        self._closed = True

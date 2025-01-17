import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm, colormaps

from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(self, decaps, probe, action_mask, ax=None, legend=True):
    """
    Plot a grid of 1x1 squares representing the environment.
    The keepout regions are the action_mask - decaps - probe
    """
    import matplotlib.pyplot as plt

    settings = {
        0: {"color": "white", "label": "available"},
        1: {"color": "grey", "label": "keepout"},
        2: {"color": "tab:red", "label": "probe"},
        3: {"color": "tab:blue", "label": "decap"},
    }

    nonzero_indices = torch.nonzero(~action_mask, as_tuple=True)[0]
    keepout = torch.cat([nonzero_indices, probe, decaps.squeeze(-1)])
    unique_elements, counts = torch.unique(keepout, return_counts=True)
    keepout = unique_elements[counts == 1]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    grid = np.meshgrid(np.arange(0, self.size), np.arange(0, self.size))
    grid = np.stack(grid, axis=-1)

    # Add new dimension to grid filled up with 0s
    grid = np.concatenate([grid, np.zeros((self.size, self.size, 1))], axis=-1)

    # Add keepout = 1
    grid[keepout // self.size, keepout % self.size, 2] = 1
    # Add probe = 2
    grid[probe // self.size, probe % self.size, 2] = 2
    # Add decaps = 3
    grid[decaps // self.size, decaps % self.size, 2] = 3

    xdim, ydim = grid.shape[0], grid.shape[1]
    ax.imshow(np.zeros((xdim, ydim)), cmap="gray")

    ax.set_xlim(0, xdim)
    ax.set_ylim(0, ydim)

    for i in range(xdim):
        for j in range(ydim):
            color = settings[grid[i, j, 2]]["color"]
            x, y = grid[i, j, 0], grid[i, j, 1]
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color, linestyle="-"))

    # Add grid with 1x1 squares
    ax.grid(
        which="major", axis="both", linestyle="-", color="k", linewidth=1, alpha=0.5
    )
    # set 10 ticks
    ax.set_xticks(np.arange(0, xdim, 1))
    ax.set_yticks(np.arange(0, ydim, 1))

    # Invert y axis
    ax.invert_yaxis()

    # Add legend
    if legend:
        num_unique = 4
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=settings[i]["color"])
            for i in range(num_unique)
        ]
        ax.legend(
            handles,
            [settings[i]["label"] for i in range(num_unique)],
            ncol=num_unique,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
        )

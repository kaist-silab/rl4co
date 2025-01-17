import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm, colormaps

from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(self, td, actions=None, ax=None, legend=True, settings=None):
    """Plot a grid of squares representing the environment.
    The keepout regions are the action_mask - decaps - probe
    """

    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D
    from matplotlib.patches import Annulus, Rectangle, RegularPolygon

    if settings is None:
        settings = {
            "available": {"color": "white", "label": "available"},
            "keepout": {"color": "grey", "label": "keepout"},
            "probe": {"color": "tab:red", "label": "probe"},
            "decap": {"color": "tab:blue", "label": "decap"},
        }

    def draw_capacitor(ax, x, y, color="black"):
        # Backgrund rectangle: same as color but with alpha=0.5
        ax.add_patch(Rectangle((x, y), 1, 1, color=color, alpha=0.5))

        # Create the plates of the capacitor
        plate_width, plate_height = (
            0.3,
            0.1,
        )  # Width and height switched to make vertical
        plate_gap = 0.2
        plate1 = Rectangle(
            (x + 0.5 - plate_width / 2, y + 0.5 - plate_height - plate_gap / 2),
            plate_width,
            plate_height,
            color=color,
        )
        plate2 = Rectangle(
            (x + 0.5 - plate_width / 2, y + 0.5 + plate_gap / 2),
            plate_width,
            plate_height,
            color=color,
        )

        # Add the plates to the axes
        ax.add_patch(plate1)
        ax.add_patch(plate2)

        # Add connection lines (wires)
        line_length = 0.2
        line1 = Line2D(
            [x + 0.5, x + 0.5],
            [
                y + 0.5 - plate_height - plate_gap / 2 - line_length,
                y + 0.5 - plate_height - plate_gap / 2,
            ],
            color=color,
        )
        line2 = Line2D(
            [x + 0.5, x + 0.5],
            [
                y + 0.5 + plate_height + plate_gap / 2,
                y + 0.5 + plate_height + plate_gap / 2 + line_length,
            ],
            color=color,
        )

        # Add the lines to the axes
        ax.add_line(line1)
        ax.add_line(line2)

    def draw_probe(ax, x, y, color="black"):
        # Backgrund rectangle: same as color but with alpha=0.5
        ax.add_patch(Rectangle((x, y), 1, 1, color=color, alpha=0.5))
        ax.add_patch(Annulus((x + 0.5, y + 0.5), (0.2, 0.2), 0.1, color=color))

    def draw_keepout(ax, x, y, color="black"):
        # Backgrund rectangle: same as color but with alpha=0.5
        ax.add_patch(Rectangle((x, y), 1, 1, color=color, alpha=0.5))
        ax.add_patch(
            RegularPolygon(
                (x + 0.5, y + 0.5), numVertices=6, radius=0.45, color=color
            )
        )

    size = self.size
    td = td.detach().cpu()
    # if batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]

    if actions is None:
        actions = td.get("action", None)

    # Transform actions from idx to one-hot
    decaps = torch.zeros(size**2)
    decaps.scatter_(0, actions, 1)
    decaps = decaps.reshape(size, size)

    keepout = ~td["action_mask"].reshape(size, size)
    probes = td["probe"].reshape(size, size)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    grid = np.meshgrid(np.arange(0, size), np.arange(0, size))
    grid = np.stack(grid, axis=-1)

    xdim, ydim = grid.shape[0], grid.shape[1]
    # ax.imshow(np.zeros((xdim, ydim)), cmap="gray")

    ax.set_xlim(0, xdim)
    ax.set_ylim(0, ydim)

    for i in range(xdim):
        for j in range(ydim):
            x, y = grid[i, j, 0], grid[i, j, 1]

            if decaps[i, j] == 1:
                draw_capacitor(ax, x, y, color=settings["decap"]["color"])
            elif probes[i, j] == 1:
                draw_probe(ax, x, y, color=settings["probe"]["color"])
            elif keepout[i, j] == 1:
                draw_keepout(ax, x, y, color=settings["keepout"]["color"])

    ax.grid(
        which="major", axis="both", linestyle="-", color="k", linewidth=1, alpha=0.5
    )
    # set 10 ticks
    ax.set_xticks(np.arange(0, xdim, 1))
    ax.set_yticks(np.arange(0, ydim, 1))

    # Invert y axis
    ax.invert_yaxis()

    # # Add legend
    if legend:
        colors = [settings[k]["color"] for k in settings.keys()]
        labels = [settings[k]["label"] for k in settings.keys()]
        handles = [
            plt.Rectangle(
                (0, 0), 1, 1, color=c, edgecolor="k", linestyle="-", linewidth=1
            )
            for c in colors
        ]
        ax.legend(
            handles,
            [label for label in labels],
            ncol=len(colors),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
        )

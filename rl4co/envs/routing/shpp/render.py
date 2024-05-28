import torch
import numpy as np
import matplotlib.pyplot as plt

from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td, actions=None, ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        # Create a plot of the nodes
        _, ax = plt.subplots(figsize=(3, 3))

    td = td.detach().cpu()

    if actions is None:
        actions = td.get("action", None)
    # if batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        actions = actions[0]

    locs = td["locs"]

    # gather locs in order of action if available
    if actions is None:
        log.warning("No action in TensorDict, rendering unsorted locs")
    else:
        actions = actions.detach().cpu()
        locs = gather_by_index(locs, actions, dim=0)

    start_x, start_y = locs[0, 0], locs[0, 1]
    end_x, end_y = locs[-1, 0], locs[-1, 1]
    city_x, city_y = locs[1:-1, 0], locs[1:-1, 1]
    x, y = locs[:, 0], locs[:, 1]

    # Plot the start and end nodes
    ax.scatter(start_x, start_y, color="tab:green", marker="s")
    ax.scatter(end_x, end_y, color="tab:red", marker="x")

    # Plot the visited nodes
    ax.scatter(city_x, city_y, color="tab:blue")

    # Add arrows between visited nodes as a quiver plot
    dx, dy = np.diff(x), np.diff(y)
    ax.quiver(
        x[:-1],
        y[:-1],
        dx,
        dy,
        scale_units="xy",
        angles="xy",
        scale=1,
        color="gray",
        width=0.003,
        headwidth=8,
    )

    # Setup limits and show
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

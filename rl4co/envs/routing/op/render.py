import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm, colormaps

from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td, actions=None, ax=None):
    # Create a plot of the nodes
    if ax is None:
        _, ax = plt.subplots()

    td = td.detach().cpu()

    # Actions
    if actions is None:
        actions = td.get("action", None)
    actions = actions.detach().cpu() if actions is not None else None

    # if batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        actions = actions[0] if actions is not None else None

    # Variables
    depot = td["locs"][0, :]
    customers = td["locs"][1:, :]
    prizes = td["prize"][1:]
    normalized_prizes = (
        200 * (prizes - torch.min(prizes)) / (torch.max(prizes) - torch.min(prizes))
        + 10
    )

    # Plot depot and customers with prize
    ax.scatter(
        depot[0],
        depot[1],
        marker="s",
        c="tab:green",
        edgecolors="black",
        zorder=5,
        s=100,
    )  # Plot depot as square
    ax.scatter(
        customers[:, 0],
        customers[:, 1],
        s=normalized_prizes,
        c=normalized_prizes,
        cmap="autumn_r",
        alpha=0.6,
        edgecolors="black",
    )  # Plot all customers with size and color indicating the prize

    # Gather locs in order of action if available
    if actions is None:
        log.warning("No action in TensorDict, rendering unsorted locs")
    else:
        # Reorder the customers and their corresponding prizes based on actions
        tour = customers[actions - 1]  # subtract 1 to match Python's 0-indexing

        # Append the depot at the beginning and the end of the tour
        tour = np.vstack((depot, tour, depot))

        # Use quiver to plot the tour
        dx, dy = np.diff(tour[:, 0]), np.diff(tour[:, 1])
        ax.quiver(
            tour[:-1, 0],
            tour[:-1, 1],
            dx,
            dy,
            scale_units="xy",
            angles="xy",
            scale=1,
            zorder=2,
            color="black",
            width=0.0035,
        )

    # Setup limits and show
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

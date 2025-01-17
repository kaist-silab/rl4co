import matplotlib.pyplot as plt
import numpy as np
import torch

from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td, actions=None, ax=None):
    if ax is None:
        # Create a plot of the nodes
        _, ax = plt.subplots()

    td = td.detach().cpu()

    if actions is None:
        actions = td.get("action", None)

    # If batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        actions = actions[0]

    locs = td["locs"]

    # Gather locs in order of action if available
    if actions is None:
        log.warning("No action in TensorDict, rendering unsorted locs")
    else:
        actions = actions.detach().cpu()
        locs = gather_by_index(locs, actions, dim=0)

    # Cat the first node to the end to complete the tour
    locs = torch.cat((locs, locs[0:1]))
    x, y = locs[:, 0], locs[:, 1]

    # Plot the visited nodes
    ax.scatter(x, y, color="tab:blue")

    # Add arrows between visited nodes as a quiver plot
    dx, dy = np.diff(x), np.diff(y)
    ax.quiver(x[:-1], y[:-1], dx, dy, scale_units="xy", angles="xy", scale=1, color="k")

    # Setup limits and show
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)


def render_improvement(td, current_soltuion, best_soltuion):
    coordinates = td["locs"][0]
    real_seq = current_soltuion[:1]
    real_best = best_soltuion[:1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # Create two side-by-side axes

    for ax in [ax1, ax2]:  # Plot on both axes
        if ax == ax1:
            ax.axis([-0.05, 1.05] * 2)
            # plot the nodes
            ax.scatter(
                coordinates[:, 0], coordinates[:, 1], marker="H", s=55, c="blue", zorder=2
            )
            # plot the tour
            real_seq_coordinates = coordinates.gather(
                0, real_seq[0].unsqueeze(1).repeat(1, 2)
            )
            real_seq_coordinates = torch.cat(
                (real_seq_coordinates, real_seq_coordinates[:1]), 0
            )
            ax.plot(
                real_seq_coordinates[:, 0],
                real_seq_coordinates[:, 1],
                color="black",
                zorder=1,
            )
            # mark node
            for i, txt in enumerate(range(real_seq.size(1))):
                ax.annotate(
                    txt,
                    (coordinates[i, 0] + 0.01, coordinates[i, 1] + 0.01),
                )
            ax.set_title("Current Solution")
        else:
            ax.axis([-0.05, 1.05] * 2)
            # plot the nodes
            ax.scatter(
                coordinates[:, 0], coordinates[:, 1], marker="H", s=55, c="blue", zorder=2
            )
            # plot the tour
            real_best_coordinates = coordinates.gather(
                0, real_best[0].unsqueeze(1).repeat(1, 2)
            )
            real_best_coordinates = torch.cat(
                (real_best_coordinates, real_best_coordinates[:1]), 0
            )
            ax.plot(
                real_best_coordinates[:, 0],
                real_best_coordinates[:, 1],
                color="black",
                zorder=1,
            )
            # mark node
            for i, txt in enumerate(range(real_seq.size(1))):
                ax.annotate(
                    txt,
                    (coordinates[i, 0] + 0.01, coordinates[i, 1] + 0.01),
                )
            ax.set_title("Best Solution")
    plt.tight_layout()

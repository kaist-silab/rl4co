import torch


def render(td, actions=None, ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    from matplotlib import colormaps

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
    prizes = td["real_prize"][1:]
    penalties = td["penalty"][1:]
    normalized_prizes = (
        200 * (prizes - torch.min(prizes)) / (torch.max(prizes) - torch.min(prizes))
        + 10
    )
    normalized_penalties = (
        3
        * (penalties - torch.min(penalties))
        / (torch.max(penalties) - torch.min(penalties))
    )

    # Represent penalty with colormap and size of edges
    penalty_cmap = colormaps.get_cmap("BuPu")
    penalty_colors = penalty_cmap(normalized_penalties)

    # Plot depot and customers with prize (size of nodes) and penalties (size of borders)
    ax.scatter(
        depot[0],
        depot[1],
        marker="s",
        c="tab:green",
        edgecolors="black",
        zorder=1,
        s=100,
    )  # Plot depot as square
    ax.scatter(
        customers[:, 0],
        customers[:, 1],
        s=normalized_prizes,
        c=normalized_prizes,
        cmap="autumn_r",
        alpha=1,
        edgecolors=penalty_colors,
        linewidths=normalized_penalties,
    )  # Plot all customers with size and color indicating the prize

    # Gather locs in order of action if available
    if actions is None:
        print("No action in TensorDict, rendering unsorted locs")
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

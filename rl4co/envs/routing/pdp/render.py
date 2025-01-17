import matplotlib.pyplot as plt
import torch

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


# Only render the first instance
def render(td, actions=None, ax=None):
    markersize = 8

    td = td.detach().cpu()
    # if batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        if actions is not None:
            actions = actions[0]

    # Variables
    init_deliveries = td["to_deliver"][1:]
    delivery_locs = td["locs"][1:][~init_deliveries.bool()]
    pickup_locs = td["locs"][1:][init_deliveries.bool()]
    depot_loc = td["locs"][0]
    actions = actions if actions is not None else td["action"]

    fig, ax = plt.subplots()

    # Plot the actions in order
    for i in range(len(actions)):
        from_node = actions[i]
        to_node = (
            actions[i + 1] if i < len(actions) - 1 else actions[0]
        )  # last goes back to depot
        from_loc = td["locs"][from_node]
        to_loc = td["locs"][to_node]
        ax.plot([from_loc[0], to_loc[0]], [from_loc[1], to_loc[1]], "k-")
        ax.annotate(
            "",
            xy=(to_loc[0], to_loc[1]),
            xytext=(from_loc[0], from_loc[1]),
            arrowprops=dict(arrowstyle="->", color="black"),
            annotation_clip=False,
        )

    # Plot the depot location
    ax.plot(
        depot_loc[0],
        depot_loc[1],
        "g",
        marker="s",
        markersize=markersize,
        label="Depot",
    )

    # Plot the pickup locations
    for i, pickup_loc in enumerate(pickup_locs):
        ax.plot(
            pickup_loc[0],
            pickup_loc[1],
            "r",
            marker="^",
            markersize=markersize,
            label="Pickup" if i == 0 else None,
        )

    # Plot the delivery locations
    for i, delivery_loc in enumerate(delivery_locs):
        ax.plot(
            delivery_loc[0],
            delivery_loc[1],
            "b",
            marker="v",
            markersize=markersize,
            label="Delivery" if i == 0 else None,
        )

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

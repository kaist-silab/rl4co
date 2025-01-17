from tensordict.tensordict import TensorDict


def render(td: TensorDict, actions=None, ax=None):
    import matplotlib.pyplot as plt
    markersize = 8

    td = td.detach().cpu()

    # If batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        if actions is not None:
            actions = actions[0]

    n_depots = td["capacity"].size(-1)
    n_pickups = (td["locs"].size(-2) - n_depots) // 2

    # Variables
    init_deliveries = td["to_deliver"][n_depots:]
    delivery_locs = td["locs"][n_depots:][~init_deliveries.bool()]
    pickup_locs = td["locs"][n_depots:][init_deliveries.bool()]
    depot_locs = td["locs"][:n_depots]
    actions = actions if actions is not None else td["action"]

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    # Plot the actions in order
    last_depot = 0
    for i in range(len(actions)-1):
        if actions[i+1] < n_depots:
            last_depot = actions[i+1]
        if actions[i] < n_depots and actions[i+1] < n_depots:
            continue
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

    # Plot last back to the depot
    from_node = actions[-1]
    to_node = last_depot
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

    # Annotate node location
    for i, loc in enumerate(td["locs"]):
        ax.annotate(
            str(i),
            (loc[0], loc[1]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
        )

    for i, depot_loc in enumerate(depot_locs):
        ax.plot(
            depot_loc[0],
            depot_loc[1],
            "tab:green",
            marker="s",
            markersize=markersize,
            label="Depot" if i == 0 else None,
        )

    # Plot the pickup locations
    for i, pickup_loc in enumerate(pickup_locs):
        ax.plot(
            pickup_loc[0],
            pickup_loc[1],
            "tab:red",
            marker="^",
            markersize=markersize,
            label="Pickup" if i == 0 else None,
        )

    # Plot the delivery locations
    for i, delivery_loc in enumerate(delivery_locs):
        ax.plot(
            delivery_loc[0],
            delivery_loc[1],
            "tab:blue",
            marker="x",
            markersize=markersize,
            label="Delivery" if i == 0 else None,
        )

    # Plot pickup and delivery pair: from loc[n_depot + i ] to loc[n_depot + n_pickups + i]
    for i in range(n_pickups):
        pickup_loc = td["locs"][n_depots + i]
        delivery_loc = td["locs"][n_depots + n_pickups + i]
        ax.plot(
            [pickup_loc[0], delivery_loc[0]],
            [pickup_loc[1], delivery_loc[1]],
            "k--",
            alpha=0.5,
        )

    # Setup limits and show
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm, colormaps
from tensordict.tensordict import TensorDict

from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td: TensorDict, idx: int):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    # TODO: fix this render function parameters
    num_machine_total = td["num_machine_total"][idx].item()
    num_job = td["num_job"][idx].item()

    job_durations = td["job_duration"][idx, :, :]
    # shape: (job, machine)
    schedule = td["schedule"][idx, :, :]
    # shape: (machine, job)

    total_machine_cnt = num_machine_total
    makespan = -td["reward"][idx].item()

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(makespan / 3, 5))
    cmap = _get_cmap(num_job)

    plt.xlim(0, makespan)
    plt.ylim(0, total_machine_cnt)
    ax.invert_yaxis()

    plt.plot([0, makespan], [4, 4], "black")
    plt.plot([0, makespan], [8, 8], "black")

    for machine_idx in range(total_machine_cnt):
        duration = job_durations[:, machine_idx]
        # shape: (job)
        machine_schedule = schedule[machine_idx, :]
        # shape: (job)

        for job_idx in range(num_job):
            job_length = duration[job_idx].item()
            job_start_time = machine_schedule[job_idx].item()
            if job_start_time >= 0:
                # Create a Rectangle patch
                rect = patches.Rectangle(
                    (job_start_time, machine_idx),
                    job_length,
                    1,
                    facecolor=cmap(job_idx),
                )
                ax.add_patch(rect)

    ax.grid()
    ax.set_axisbelow(True)
    plt.show()

def _get_cmap(color_cnt):
    from random import shuffle

    from matplotlib.colors import CSS4_COLORS, ListedColormap

    color_list = list(CSS4_COLORS.keys())
    shuffle(color_list)
    cmap = ListedColormap(color_list, N=color_cnt)
    return cmap

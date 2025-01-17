import logging

from typing import Tuple

import torch

from tensordict import TensorDict
from torch import Size, Tensor

from rl4co.envs.scheduling.fjsp import INIT_FINISH

logger = logging.getLogger(__name__)


def get_op_features(td: TensorDict):
    return torch.stack((td["lbs"], td["is_ready"], td["num_eligible"]), dim=-1)


def cat_and_norm_features(
    td: TensorDict, feats: list[str], time_feats: list[str], norm_const: int
):
    # logger.info(f"will scale the features {','.join(time_feats)} with a constant ({norm_const})")
    feature_list = []
    for feat in feats:
        if feat in time_feats:
            feature_list.append(td[feat] / norm_const)
        else:
            feature_list.append(td[feat])

    return torch.stack(feature_list, dim=-1).to(torch.float32)


def view(
    tensor: Tensor,
    idx: Tuple[Tensor],
    pad_mask: Tensor,
    new_shape: Size | list[int],
    pad_value: float | int,
):
    # convert mask specifying which entries are padded into mask specifying which entries to keep
    mask = ~pad_mask
    new_view = tensor.new_full(size=new_shape, fill_value=pad_value)
    new_view[idx] = tensor[mask]
    return new_view


def _get_idx_for_job_op_view(td: TensorDict) -> tuple:
    bs, _, n_total_ops = td["job_ops_adj"].shape
    # (bs, ops)
    batch_idx = torch.arange(bs, device=td.device).repeat_interleave(n_total_ops)
    batch_idx = batch_idx.reshape(bs, -1)
    # (bs, ops)
    ops_job_map = td["ops_job_map"]
    # (bs, ops)
    ops_sequence_order = td["ops_sequence_order"]
    # (bs*n_ops_max, 3)
    idx = (
        torch.stack((batch_idx, ops_job_map, ops_sequence_order), dim=-1)
        .to(torch.long)
        .flatten(0, 1)
    )
    # (bs, n_ops_max)
    mask = ~td["pad_mask"]
    # (total_ops_in_batch, 3)
    idx = idx[mask.flatten(0, 1)]
    b, j, o = map(lambda x: x.squeeze(1), idx.chunk(3, dim=-1))
    return b, j, o


def get_job_op_view(td: TensorDict, keys: list[str] = [], pad_value: float | int = 0):
    """This function reshapes all tensors of the tensordict from a flat operations-only view
    to a nested job-operation view and creates a new tensordict from it.
    :param _type_ td: tensordict
    :return _type_: dict
    """
    # ============= Prepare the new index =============
    bs, num_jobs, _ = td["job_ops_adj"].shape
    max_ops_per_job = int(td["job_ops_adj"].sum(-1).max())
    idx = _get_idx_for_job_op_view(td)
    new_shape = Size((bs, num_jobs, max_ops_per_job))
    pad_mask = td["pad_mask"]
    # ==============================================

    # due to special structure, processing times are treated seperately
    if "proc_times" in keys:
        keys.remove("proc_times")
    # reshape processing times; (bs, ma, ops) -> (bs, ma, jobs, ops_per_job)
    new_proc_times_view = view(
        td["proc_times"].permute(0, 2, 1), idx, pad_mask, new_shape, pad_value
    ).permute(0, 3, 1, 2)

    # add padding mask if not in keys
    if "pad_mask" not in keys:
        keys.append("pad_mask")

    new_views = dict(
        map(lambda key: (key, view(td[key], idx, pad_mask, new_shape)), keys)
    )

    # update tensordict clone with reshaped tensors
    return {"proc_times": new_proc_times_view, **new_views}


def blockify(td, tensor: Tensor, pad_value: float | int = 0):
    assert len(tensor.shape) in [
        2,
        3,
    ], "blockify only supports tensors of shape (bs, seq, (d)), where the feature dim d is optional"
    # get the size of the blockified tensor
    bs, _, *d = tensor.shape
    num_jobs = td["job_ops_adj"].size(1)
    max_ops_per_job = int(td["job_ops_adj"].sum(-1).max())
    new_shape = Size((bs, num_jobs, max_ops_per_job, *d))
    # get indices of valid entries of blockified tensor
    idx = _get_idx_for_job_op_view(td)
    pad_mask = td["pad_mask"]
    # create the blockified view
    new_view_tensor = view(tensor, idx, pad_mask, new_shape, pad_value)
    return new_view_tensor


def unblockify(
    td: TensorDict, tensor: Tensor, mask: Tensor = None, pad_value: float | int = 0
):
    assert len(tensor.shape) in [
        3,
        4,
    ], "blockify only supports tensors of shape (bs, nb, s, (d)), where the feature dim d is optional"
    # get the size of the blockified tensor
    bs, _, _, *d = tensor.shape
    n_ops_per_batch = td["job_ops_adj"].sum((1, 2)).unsqueeze(1)  # (bs)
    seq_len = int(n_ops_per_batch.max())
    new_shape = Size((bs, seq_len, *d))

    # create the mask to gather then entries of the blockified tensor. NOTE that only by
    # blockifying the original pad_mask
    pad_mask = td["pad_mask"]
    pad_mask = blockify(td, pad_mask, True)

    # get indices of valid entrie in flat matrix
    b = torch.arange(bs, device=td.device).repeat_interleave(seq_len).reshape(bs, seq_len)
    i = torch.arange(seq_len, device=td.device)[None].repeat(bs, 1)
    idx = tuple(map(lambda x: x[i < n_ops_per_batch], (b, i)))
    # create view
    new_tensor = view(tensor, idx, pad_mask, new_shape, pad_value=pad_value)
    return new_tensor


def first_diff(x: Tensor, dim: int):
    shape = x.shape
    shape = (*shape[:dim], 1, *shape[dim + 1 :])
    seq_cutoff = x.index_select(dim, torch.arange(x.size(dim) - 1, device=x.device))
    first_diff_seq = x - torch.cat((seq_cutoff.new_zeros(*shape), seq_cutoff), dim=dim)
    return first_diff_seq


def spatial_encoding(td: TensorDict):
    """We use a spatial encoing as proposed in GraphFormer (https://arxiv.org/abs/2106.05234)
    The spatial encoding in GraphFormer determines the distance of the shortest path between and
    nodes i and j and uses a special value for node pairs that cannot be connected at all.
    For any two operations i<j of the same job, we determine the number of operations to be completet
    when starting at i before j can be started (e.g. i=3 and j=5 -> e=2) and for i>j the negative number of
    operations that starting from j, have been completet before arriving at i (e.g. i=5 j=3 -> e=-2).
    For i=j we set e=0 as well as for operations of different jobs.

    :param torch.Tensor[bs, n_ops] ops_job_map: tensor specifying the index of its corresponding job
    :return torch.Tensor[bs, n_ops, n_ops]: length of shortest path between any two operations
    """
    bs, _, n_total_ops = td["job_ops_adj"].shape
    max_ops_per_job = int(td["job_ops_adj"].sum(-1).max())
    ops_job_map = td["ops_job_map"]
    pad_mask = td["pad_mask"]

    same_job = (ops_job_map[:, None] == ops_job_map[..., None]).to(torch.int32)
    # mask padded
    same_job[pad_mask.unsqueeze(2).expand_as(same_job)] = 0
    same_job[pad_mask.unsqueeze(1).expand_as(same_job)] = 0
    # take upper triangular of same_job and set diagonal to zero for counting purposes
    upper_tri = torch.triu(same_job) - torch.diag(
        torch.ones(n_total_ops, device=td.device)
    )[None].expand_as(same_job)
    # cumsum and masking of operations that do not belong to the same job
    num_jumps = upper_tri.cumsum(2) * upper_tri
    # mirror the matrix
    num_jumps = num_jumps + num_jumps.transpose(1, 2)
    # NOTE: shifted this logic into the spatial encoding module
    # num_jumps = num_jumps + (-num_jumps.transpose(1,2))
    assert not torch.any(num_jumps >= max_ops_per_job)
    # special value for ops of different jobs and self-loops
    num_jumps = torch.where(num_jumps == 0, -1, num_jumps)
    self_mask = torch.eye(n_total_ops).repeat(bs, 1, 1).bool()
    num_jumps[self_mask] = 0
    return num_jumps


def calc_lower_bound(td: TensorDict):
    """Here we calculate the lower bound of the operations finish times. In the FJSP case, multiple things need to
    be taken into account due to the usability of the different machines for multiple ops of different jobs:

    1.) Operations may only start once their direct predecessor is finished. We calculate its lower bound by
    adding the minimum possible operation time to this detected start time. However, we cannot use the proc_times
    directly, but need to account for the fact, that machines might still be busy, once an operation can be processed.
    We detect this offset by detecting ops-machine pairs, where the first possible start point of the operation is before
    the machine becomes idle again - Therefore, we add this discrepancy to the proc_time of the respective ops-ma combination

    2.) If an operation has been scheduled, we use its actual finishing time as lower bound. In this case, using the cumulative sum
    of all peedecessors of a job does not make sense, since it is likely to differ from the real finishing time of its direct
    predecessor (its only a lower bound). Therefore, we add the finish time to the cumulative sum of processing time of all
    UNSCHEDULED operations, to obtain the lower bound.
    Making this work is a bit hacky: We compute the first differences of finishing times of those operations scheduled and
    add them to the matrix of processing times, where already processed operations are masked (with zero)


    """

    proc_times = td["proc_times"].clone()  # (bs, ma, ops)
    busy_until = td["busy_until"]  # (bs, ma)
    ops_adj = td["ops_adj"]  # (bs, ops, ops, 2)
    finish_times = td["finish_times"]  # (bs, ops)
    job_ops_adj = td["job_ops_adj"]  # (bs, jobs, ops)
    op_scheduled = td["op_scheduled"].to(torch.float32)  # (bs, ops)

    ############## REGARDING POINT 1 OF DOCSTRING ##############
    # for operations whose immidiate predecessor is scheduled, we can determine its earliest
    # start time by the end time of the predecessor.
    # (bs, num_ops, 1)
    maybe_start_at = torch.bmm(ops_adj[..., 0], finish_times[..., None]).squeeze(2)
    # using the start_time, we can determine if and how long an op needs to wait for a machine to finish
    wait_for_ma_offset = torch.clip(busy_until[..., None] - maybe_start_at[:, None], 0)
    # we add this required waiting time to the respective processing time
    proc_time_plus_wait = torch.where(
        proc_times == 0, proc_times, proc_times + wait_for_ma_offset
    )
    # NOTE get the mean processing time over all eligible machines for lb calulation
    # ops_proc_times = torch.where(proc_times == 0, torch.inf, proc_time_plus_wait).min(1).values)
    ops_proc_times = proc_time_plus_wait.sum(1) / (proc_times.gt(0).sum(1) + 1e-9)
    # mask proc times for already scheduled ops
    ops_proc_times[op_scheduled.to(torch.bool)] = 0

    ############### REGARDING POINT 2 OF DOCSTRING ###################
    # Now we determine all operations that are not scheduled yet (and thus have no finish_time). We will compute the cumulative
    # sum over the processing time to determine the lower bound of unscheduled operations...
    proc_matrix = job_ops_adj
    ops_assigned = proc_matrix * op_scheduled[:, None]
    proc_matrix_not_scheduled = proc_matrix * (
        torch.ones_like(proc_matrix) - op_scheduled[:, None]
    )

    # ...and add the finish_time of the last scheduled operation of the respective job to that. To make this work, using the cumsum logic,
    # we calc the first differences of the finish times and seperate by job.
    # We use the first differences, so that the finish times do not add up during cumulative sum below
    # (bs, num_jobs, num_ops)
    finish_times_1st_diff = ops_assigned * first_diff(
        ops_assigned * finish_times[:, None], 2
    )

    # masking the processing time of scheduled operations and add their finish times instead (first diff thereof)
    lb_end_expand = (
        proc_matrix_not_scheduled * ops_proc_times.unsqueeze(1).expand_as(job_ops_adj)
        + finish_times_1st_diff
    )
    # (bs, max_ops); lower bound finish time per operation using the cumsum logic
    LBs = torch.sum(job_ops_adj * lb_end_expand.cumsum(-1), dim=1)
    # remove nans
    LBs = torch.nan_to_num(LBs, nan=0.0)

    # test
    assert torch.where(
        finish_times != INIT_FINISH, torch.isclose(LBs, finish_times), True
    ).all()

    return LBs


def op_is_ready(td: TensorDict):
    # compare finish times of predecessors with current time step; shape=(b, n_ops_max)
    is_ready = (
        torch.bmm(td["ops_adj"][..., 0], td["finish_times"][..., None]).squeeze(2)
        <= td["time"][:, None]
    )
    # shape=(b, n_ops_max)
    is_scheduled = td["ma_assignment"].sum(1).bool()
    # op is ready for scheduling if it has not been scheduled and its predecessor is finished
    return torch.logical_and(is_ready, ~is_scheduled)


def get_job_ops_mapping(
    start_op_per_job: torch.Tensor, end_op_per_job: torch.Tensor, n_ops_max: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Implements a mapping function from operations to jobs

    :param torch.Tensor start_op_per_job: index of first operation of each job
    :param torch.Tensor end_op_per_job: index of last operation of each job
    :return Tuple[torch.Tensor, torch.Tensor]:
        1st.) index mapping (bs, num_ops): [0,0,1,1,1] means that first two operations belong to job 0
        2st.) binary mapping (bs, num_jobs, num_ops): [[1,1,0], [0,0,1]] means that first two operations belong to job 0
    """
    device = end_op_per_job.device
    end_op_per_job = end_op_per_job.clone()

    bs, num_jobs = end_op_per_job.shape

    # in order to avoid shape conflicts, set the end operation id to the id of max_ops (all batches have same #ops)
    end_op_per_job[:, -1] = n_ops_max - 1

    # here we will generate the operations-job mapping:
    # Therefore we first generate a sequence of operation ids and expand it the the size of the mapping matrix:
    # (bs, jobs, max_ops)
    ops_seq_exp = torch.arange(n_ops_max, device=device)[None, None].expand(
        bs, num_jobs, -1
    )
    # (bs, jobs, max_ops)  # expanding start and end operation ids
    end_op_per_job_exp = end_op_per_job[..., None].expand_as(ops_seq_exp)
    start_op_per_job_exp = start_op_per_job[..., None].expand_as(ops_seq_exp)
    # given ids of start and end operations per job, this generates the mapping of ops to jobs
    # (bs, jobs, max_ops)
    ops_job_map = torch.nonzero(
        (ops_seq_exp <= end_op_per_job_exp) & (ops_seq_exp >= start_op_per_job_exp)
    )
    # (bs, max_ops)
    ops_job_map = torch.stack(ops_job_map[:, 1].split(n_ops_max), dim=0)

    # we might also want a binary mapping / adjacency matrix connecting jobs to operations
    # (bs, num_jobs, num_ops)
    ops_job_bin_map = torch.scatter_add(
        input=ops_job_map.new_zeros((bs, num_jobs, n_ops_max)),
        dim=1,
        index=ops_job_map.unsqueeze(1),
        src=ops_job_map.new_ones((bs, num_jobs, n_ops_max)),
    )

    return ops_job_map, ops_job_bin_map

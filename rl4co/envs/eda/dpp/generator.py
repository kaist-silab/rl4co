import os
import zipfile
from typing import Union, Callable

import torch
import numpy as np

from robust_downloader import download
from torch.distributions import Uniform
from tensordict.tensordict import TensorDict

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.utils.pylogger import get_pylogger
from rl4co.envs.common.utils import get_sampler, Generator

log = get_pylogger(__name__)



class DPPGenerator(Generator):
    """Data generator for the Decap Placement Problem (DPP).

    Args:
        min_loc: Minimum location value. Defaults to 0.
        max_loc: Maximum location value. Defaults to 1.
        num_keepout_min: Minimum number of keepout regions. Defaults to 1.
        num_keepout_max: Maximum number of keepout regions. Defaults to 50.
        max_decaps: Maximum number of decaps. Defaults to 20.
        data_dir: Directory to store data. Defaults to "data/dpp/".
            This can be downloaded from this [url](https://drive.google.com/uc?id=1IEuR2v8Le-mtHWHxwTAbTOPIkkQszI95).
        chip_file: Name of the chip file. Defaults to "10x10_pkg_chip.npy".
        decap_file: Name of the decap file. Defaults to "01nF_decap.npy".
        freq_file: Name of the frequency file. Defaults to "freq_201.npy".
        url: URL to download data from. Defaults to None.
    
    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each customer
            depot [batch_size, 2]: location of the depot
            demand [batch_size, num_loc]: demand of each customer
            capacity [batch_size]: capacity of the vehicle
    """
    def __init__(
        self,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        num_keepout_min: int = 1,
        num_keepout_max: int = 50,
        max_decaps: int = 20,
        data_dir: str = "data/dpp/",
        chip_file: str = "10x10_pkg_chip.npy",
        decap_file: str = "01nF_decap.npy",
        freq_file: str = "freq_201.npy",
        url: str = None,
        **unused_kwargs
    ):
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.num_keepout_min = num_keepout_min
        self.num_keepout_max = num_keepout_max
        self.max_decaps = max_decaps
        self.data_dir = data_dir

        # DPP environment doen't have any other kwargs
        if len(unused_kwargs) > 0:
            log.error(f"Found {len(unused_kwargs)} unused kwargs: {unused_kwargs}")


        # Download and load the data from online dataset
        self.url = (
            "https://github.com/kaist-silab/devformer/raw/main/data/data.zip"
            if url is None
            else url
        )
        self.backup_url = (
            "https://drive.google.com/uc?id=1IEuR2v8Le-mtHWHxwTAbTOPIkkQszI95"
        )
        self._load_dpp_data(chip_file, decap_file, freq_file)

        # Check the validity of the keepout parameters
        assert (
            num_keepout_min <= num_keepout_max
        ), "num_keepout_min must be <= num_keepout_max"
        assert (
            num_keepout_max <= self.size**2
        ), "num_keepout_max must be <= size * size (total number of locations)"

    def _generate(self, batch_size) -> TensorDict:
        """
        Generate initial observations for the environment with locations, probe, and action mask
        Action_mask eliminates the keepout regions and the probe location, and is updated to eliminate placed decaps
        """
        m = n = self.size
        # if int, convert to list and make it a batch for easier generation
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        batched = len(batch_size) > 0
        bs = [1] if not batched else batch_size

        # Create a list of locs on a grid
        locs = torch.meshgrid(
            torch.arange(m), torch.arange(n)
        )
        locs = torch.stack(locs, dim=-1).reshape(-1, 2)
        # normalize the locations by the number of rows and columns
        locs = locs / torch.tensor([m, n], dtype=torch.float)
        locs = locs[None].expand(*bs, -1, -1)

        # Create available mask
        available = torch.ones((*bs, m * n), dtype=torch.bool)

        # Sample probe location from m*n
        probe = torch.randint(m * n, size=(*bs, 1))
        available.scatter_(1, probe, False)

        # Sample keepout locations from m*n except probe
        num_keepout = torch.randint(
            self.num_keepout_min,
            self.num_keepout_max,
            size=(*bs, 1),
        )
        keepouts = [torch.randperm(m * n)[:k] for k in num_keepout]
        for i, (a, k) in enumerate(zip(available, keepouts)):
            available[i] = a.scatter(0, k, False)

        return TensorDict(
            {
                "locs": locs if batched else locs.squeeze(0),
                "probe": probe if batched else probe.squeeze(0),
                "action_mask": available if batched else available.squeeze(0),
            },
            batch_size=batch_size,
        )

    def _load_dpp_data(self, chip_file, decap_file, freq_file):
        def _load_file(fpath):
            f = os.path.join(self.data_dir, fpath)
            if not os.path.isfile(f):
                self._download_data()
            with open(f, "rb") as f_:
                return torch.from_numpy(np.load(f_))

        self.raw_pdn = _load_file(chip_file)  # [num_freq, size^2, size^2]
        self.decap = _load_file(decap_file).to(torch.complex64)  # [num_freq, 1, 1]
        self.freq = _load_file(freq_file)  # [num_freq]
        self.size = int(np.sqrt(self.raw_pdn.shape[-1]))
        self.num_freq = self.freq.shape[0]

    def _download_data(self):
        log.info("Downloading data...")
        try:
            download(self.url, self.data_dir, "data.zip")
        except Exception:
            log.error(
                f"Download from main url {self.url} failed. Trying backup url {self.backup_url}..."
            )
            download(self.backup_url, self.data_dir, "data.zip")
        log.info("Download complete. Unzipping...")
        zipfile.ZipFile(os.path.join(self.data_dir, "data.zip"), "r").extractall(
            self.data_dir
        )
        log.info("Unzip complete. Removing zip file")
        os.remove(os.path.join(self.data_dir, "data.zip"))

    def load_data(self, fpath, batch_size=[]):
        data = load_npz_to_tensordict(fpath)
        # rename key if necessary (old dpp version)
        if "observation" in data.keys():
            data["locs"] = data.pop("observation")
        return data

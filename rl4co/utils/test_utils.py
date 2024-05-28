from torch.utils.data import DataLoader

from rl4co.envs import (
    CVRPEnv,
    CVRPTWEnv,
    DPPEnv,
    MDPPEnv,
    MTSPEnv,
    OPEnv,
    PCTSPEnv,
    PDPEnv,
    SDVRPEnv,
    SMTWTPEnv,
    SPCTSPEnv,
    TSPEnv,
    SHPPEnv,
)


def get_env(name, size):
    if name == "tsp":
        env = TSPEnv(generator_params=dict(num_loc=size))
    elif name == "cvrp":
        env = CVRPEnv(generator_params=dict(num_loc=size))
    elif name == "cvrptw":
        env = CVRPTWEnv(generator_params=dict(num_loc=size))
    elif name == "sdvrp":
        env = SDVRPEnv(generator_params=dict(num_loc=size))
    elif name == "pdp":
        env = PDPEnv(generator_params=dict(num_loc=size))
    elif name == "op":
        env = OPEnv(generator_params=dict(num_loc=size))
    elif name == "mtsp":
        env = MTSPEnv(generator_params=dict(num_loc=size))
    elif name == "pctsp":
        env = PCTSPEnv(generator_params=dict(num_loc=size))
    elif name == "spctsp":
        env = SPCTSPEnv(generator_params=dict(num_loc=size))
    elif name == "dpp":
        env = DPPEnv()
    elif name == "mdpp":
        env = MDPPEnv()
    elif name == "smtwtp":
        env = SMTWTPEnv()
    elif name == "shpp":
        env = SHPPEnv()
    else:
        raise ValueError(f"Unknown env_name: {name}")

    return env.transform()


def generate_env_data(env, size, batch_size):
    env = get_env(env, size)
    dataset = env.dataset([batch_size])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    return env, next(iter(dataloader))

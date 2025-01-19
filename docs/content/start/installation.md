# Installation

<a href="https://colab.research.google.com/github/ai4co/rl4co/blob/main/examples/1-quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

RL4CO is now available for installation on `pip`!
```bash
pip install rl4co
```

## Local install and development
If you want to develop RL4CO or access the latest builds, you may install it locally after downloading the repo:

```bash
git clone https://github.com/ai4co/rl4co && cd rl4co
```

The simplest way is via `pip` in editable mode with
```bash
pip install -e .
```

To install optional dependencies, you may specify them as follows `pip install -e ".[dev,graph,routing,docs]"`.

We recommend installing in virtual environments with a package manager such as the blazing-fast [`uv`](https://docs.astral.sh/uv/), [`poetry`](https://python-poetry.org/), or [`conda`](https://docs.conda.io/en/latest/), with quickstart commands below:

<details>
    <summary>Install with `uv`</summary>

You first need to install `uv`, i.e., with `pip`:
```bash
pip install uv
```

Then, you can create a virtual environment locally and activate it:
```bash
uv sync
source .venv/bin/activate
```

Note that `uv` directly generates the `.venv` folder in the current directory.


To install (all) extras, you may use `uv sync --all-extras` or specify them individually with `uv sync --extra dev --extra graph --extra routing --extra docs`.

</details>


<details>
    <summary>Install with `poetry`</summary>

Make sure that you have `poetry` installed from the [official website](https://python-poetry.org/docs/).

Then, you can create a virtual environment locally:
```bash
poetry install
poetry env activate # poetry shell removed in poetry 2.0.0
```

Note: you need to upgrade `poetry` to the latest version with `poetry self update` to versions >=2.0.0 (see [blog post](https://python-poetry.org/blog/announcing-poetry-2.0.0/)). This is also the reason why we don't need a special `pyproject.toml` anymore.

</details>


<details>
    <summary>Install with `conda`</summary>

After [installing `conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), you can create a virtual environment locally with:
```bash
conda create -n rl4co python=3.12
conda activate rl4co
```
</details>


## Minimalistic Example

Here is a minimalistic example training the Attention Model with greedy rollout baseline on TSP in less than 30 lines of code:

```python
from rl4co.envs.routing import TSPEnv, TSPGenerator
from rl4co.models import AttentionModelPolicy, POMO
from rl4co.utils import RL4COTrainer

# Instantiate generator and environment
generator = TSPGenerator(num_loc=50, loc_distribution="uniform")
env = TSPEnv(generator)

# Create policy and RL model
policy = AttentionModelPolicy(env_name=env.name, num_encoder_layers=6)
model = POMO(env, policy, batch_size=64, optimizer_kwargs={"lr": 1e-4})

# Instantiate Trainer and fit
trainer = RL4COTrainer(max_epochs=10, accelerator="gpu", precision="16-mixed")
trainer.fit(model)
```


!!! tip
    We recommend checking out our [quickstart notebook](../../../examples/1-quickstart.ipynb)!
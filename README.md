</br>
<div align="center">

<img src="https://raw.githubusercontent.com/ai4co/assets/main/svg/rl4co_animated_full.svg" alt="AI4CO Logo" style="width: 40%; height: auto;">

</br></br>


<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://github.com/pytorch/rl"><img alt="base: TorchRL" src="https://img.shields.io/badge/base-TorchRL-red"></a>
<a href="https://hydra.cc/"><img alt="config: Hydra" src="https://img.shields.io/badge/config-Hydra-89b8cd"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://join.slack.com/t/rl4co/shared_invite/zt-1ytz2c1v4-0IkQ8NQH4TRXIX8PrRmDhQ"><img alt="Slack" src="https://img.shields.io/badge/slack-chat-611f69.svg?logo=slack"></a>
<a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-red.svg"></a>
<a href="https://colab.research.google.com/github/ai4co/rl4co/blob/main/examples/1-quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
<a href="https://pypi.org/project/rl4co"><img alt="PyPI" src="https://img.shields.io/pypi/v/rl4co?logo=pypi"></a>
<a href="https://app.codecov.io/github/ai4co/rl4co/tree/main/rl4co"><img alt="Codecov" src="https://codecov.io/github/ai4co/rl4co/tree/main/badge.svg"></a>
<a href="https://github.com/ai4co/rl4co/actions/workflows/tests.yml"><img alt="Test" src="https://github.com/ai4co/rl4co/actions/workflows/tests.yml/badge.svg"></a>

<p>
  <a href="https://rl4.co/"><strong>Documentation</strong></a> |
  <a href="#getting-started"><strong>Getting Started</strong></a> |
  <a href="#usage"><strong>Usage</strong></a> |
  <a href="#contributing"><strong>Contributing</strong></a> |
  <a href="https://arxiv.org/abs/2306.17100"><strong>Paper</strong></a> |
  <a href="#join-us"><strong>Join Us</strong></a>
</p>



</div>



An extensive Reinforcement Learning (RL) for Combinatorial Optimization (CO) benchmark. Our goal is to provide a unified framework for RL-based CO algorithms, and to facilitate reproducible research in this field, decoupling the science from the engineering.


RL4CO is built upon:
- [TorchRL](https://github.com/pytorch/rl): official PyTorch framework for RL algorithms and vectorized environments on GPUs
- [TensorDict](https://github.com/pytorch-labs/tensordict): a library to easily handle heterogeneous data such as states, actions and rewards
- [PyTorch Lightning](https://github.com/Lightning-AI/lightning): a lightweight PyTorch wrapper for high-performance AI research
- [Hydra](https://github.com/facebookresearch/hydra): a framework for elegantly configuring complex applications

<div align="center">
  <img src="https://github.com/ai4co/rl4co/assets/48984123/0e409784-05a9-4799-b7aa-6c0f76ecf27f" alt="RL4CO-Overview" style="max-width: 90%;">
</div>

We offer flexible and efficient implementations of the following policies:
- **Constructive**: learn to construct a solution from scratch
  - _Autoregressive (AR)_: construct solutions one step at a time via a decoder
  - _NonAutoregressive (NAR)_: learn to predict a heuristic, such as a heatmap, to then construct a solution
- **Improvement**: learn to improve a pre-existing solution

<div align="center">
  <img src="https://github.com/ai4co/rl4co/assets/48984123/9e1f32f9-9884-49b9-b6cd-364861cc8fe7" alt="RL4CO-Policy-Overview" style="max-width: 90%;">
</div>

We provide several utilities and modularization. For example, we modularize reusable components such as _environment embeddings_ that can easily be swapped to [solve new problems](https://github.com/ai4co/rl4co/blob/main/examples/3-creating-new-env-model.ipynb).


<div align="center">
  <img src="https://github.com/ai4co/rl4co/assets/48984123/c47a9301-4c9f-43fd-b21f-761abeae9717" alt="RL4CO-Env-Embedding" style="max-width: 90%;">
</div>


## Getting started
<a href="https://colab.research.google.com/github/ai4co/rl4co/blob/main/examples/1-quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

RL4CO is now available for installation on `pip`!
```bash
pip install rl4co
```

To get started, we recommend checking out our [quickstart notebook](examples/1-quickstart.ipynb) or the [minimalistic example](#minimalistic-example) below.

### Install from source
This command installs the bleeding edge `main` version, useful for staying up-to-date with the latest developments - for instance, if a bug has been fixed since the last official release but a new release hasn’t been rolled out yet:

```bash
pip install -U git+https://github.com/ai4co/rl4co.git
```

### Local install and development
If you want to develop RL4CO we recommend you to install it locally with `pip` in editable mode:

```bash
git clone https://github.com/ai4co/rl4co && cd rl4co
pip install -e .
```

We recommend installing in virtual environments with a package manager such as the blazing-fast [`uv`](https://docs.astral.sh/uv/), [`poetry`](https://python-poetry.org/), or [`conda`](https://docs.conda.io/en/latest/); instruction are available in the [documentation](https://rl4.co/docs/content/start/installation/).



## Usage


Train model with default configuration (AM on TSP environment):
```bash
python run.py
```

> [!TIP]
> You may check out [this notebook](examples/advanced/1-hydra-config.ipynb) to get started with Hydra!

<details>
    <summary>Change experiment settings</summary>

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)
```bash
python run.py experiment=routing/am env=tsp env.num_loc=50 model.optimizer_kwargs.lr=2e-4
```
Here you may change the environment, e.g. with `env=cvrp` by command line or by modifying the corresponding experiment e.g. [configs/experiment/routing/am.yaml](configs/experiment/routing/am.yaml).

</details>




<details>
    <summary>Disable logging</summary>

```bash
python run.py experiment=routing/am logger=none '~callbacks.learning_rate_monitor'
```
Note that `~` is used to disable a callback that would need a logger.

</details>


<details>
    <summary>Create a sweep over hyperparameters (-m for multirun)</summary>

```bash
python run.py -m experiment=routing/am  model.optimizer.lr=1e-3,1e-4,1e-5
```
</details>



### Minimalistic Example

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

Other examples can be found on our [documentation](https://rl4.co/examples/1-quickstart/)!


### Testing

Run tests with `pytest` from the root directory:

```bash
pytest tests
```

### Known Bugs

You may check out the [issues](https://github.com/ai4co/rl4co/issues) and [discussions](https://github.com/ai4co/rl4co/discussions). We will also periodically post updates on the [FAQ section](https://rl4.co/docs/content/general/faq/).

## Contributing

Have a suggestion, request, or found a bug? Feel free to [open an issue](https://github.com/ai4co/rl4co/issues) or [submit a pull request](https://github.com/ai4co/rl4co/pulls).
If you would like to contribute, please check out our contribution guidelines   [here](.github/CONTRIBUTING.md). We welcome and look forward to all contributions to RL4CO!

We are also on [Slack](https://join.slack.com/t/rl4co/shared_invite/zt-1ytz2c1v4-0IkQ8NQH4TRXIX8PrRmDhQ) if you have any questions or would like to discuss RL4CO with us. We are open to collaborations and would love to hear from you 🚀

### Contributors
<a href="https://github.com/ai4co/rl4co/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ai4co/rl4co" />
</a>

## Citation
If you find RL4CO valuable for your research or applied projects:

```bibtex
@article{berto2024rl4co,
    title={{RL4CO: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark}},
    author={Federico Berto and Chuanbo Hua and Junyoung Park and Laurin Luttmann and Yining Ma and Fanchen Bu and Jiarui Wang and Haoran Ye and Minsu Kim and Sanghyeok Choi and Nayeli Gast Zepeda and Andr\'e Hottung and Jianan Zhou and Jieyi Bi and Yu Hu and Fei Liu and Hyeonah Kim and Jiwoo Son and Haeyeon Kim and Davide Angioni and Wouter Kool and Zhiguang Cao and Jie Zhang and Kijung Shin and Cathy Wu and Sungsoo Ahn and Guojie Song and Changhyun Kwon and Lin Xie and Jinkyoo Park},
    year={2024},
    journal={arXiv preprint arXiv:2306.17100},
    note={\url{https://github.com/ai4co/rl4co}}
}
```

Note that a [previous version of RL4CO](https://openreview.net/forum?id=YXSJxi8dOV) has been accepted as an oral presentation at the [NeurIPS 2023 GLFrontiers Workshop](https://glfrontiers.github.io/). Since then, the library has greatly evolved and improved!

---


## Join us
[![Slack](https://img.shields.io/badge/slack-chat-611f69.svg?logo=slack)](https://join.slack.com/t/rl4co/shared_invite/zt-1ytz2c1v4-0IkQ8NQH4TRXIX8PrRmDhQ)

We invite you to join our AI4CO community, an open research group in Artificial Intelligence (AI) for Combinatorial Optimization (CO)!


<div align="center">
    <img src="https://raw.githubusercontent.com/ai4co/assets/main/svg/ai4co_animated_full.svg" alt="AI4CO Logo" style="width: 30%; height: auto;">
</div>

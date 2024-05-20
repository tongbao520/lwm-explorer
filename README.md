
# Latent World Models for Intrinsically Motivated Exploration
This is the official repository for the NeurIPS 2020 Spotlight paper | [arXiv:2010.02302](https://arxiv.org/abs/2010.02302) |.

[10m video presentation from NeurIPS](https://slideslive.com/38937965/latent-world-models-for-intrinsically-motivated-exploration)

![montezuma's revenge t-sne](https://raw.githubusercontent.com/tongbao520/lwm-explorer/master/montezuma.png)

## Installation
The implementation is based on PyTorch. Logging works on [wandb.ai](https://wandb.ai/). For Docker, please refer to `docker/Dockerfile`.

## Usage
After training, the models are saved as `models/dqn.pt`, `models/predictor.pt` etc. For evaluation, models will be loaded from the same filenames.

#### Atari
Follow the steps below to reproduce LWM results as given in [Table 2](https://arxiv.org/abs/2010.02302):
```sh
cd atari
python -m train --env MontezumaRevenge --seed 0
python -m eval --env MontezumaRevenge --seed 0
```
Refer to `default.yaml` for detailed configuration.

#### Partially Observable Labyrinth
Follow the steps below to reproduce results as given in Table 1:
```sh
cd pol
# DQN agent
python -m train --size 3
python -m eval --size 3

# DQN + WM agent
python -m train --size 3 --add_ri
python -m eval --size 3 --add_ri

# random agent
python -m eval --size 3 --random
```

Environment code is in [pol/pol_env.py](https://github.com/tongbao520/lwm-explorer/blob/master/pol/pol_env.py). It extends `gym.Env` and can be used as usual:

```python
from pol_env import PolEnv
env = PolEnv(size=3)
obs = env.reset()
action = env.observation_space.sample()
obs, reward, done, infos = env.step(action)
env.render()
```

## Reference
If you find this work helpful, please cite: [Bibtex](https://arxiv.org/abs/2010.02302v2)

```@inproceedings{LWM,
 author = {Ermolov, Aleksandr and Sebe, Nicu},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {5565--5575},
 publisher = {Curran Associates, Inc.},
 title = {Latent World Models For Intrinsically Motivated Exploration},
 volume = {33},
 year = {2020}
}
```
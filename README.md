# Optimistic Sequential Multi-Agent Reinforcement Learning With Motivational Communication

This repository is the implementation of the paper "Optimistic Sequential Multi-Agent Reinforcement Learning With Motivational Communication"[[PDF](https://doi.org/10.1016/j.neunet.2024.106547)]. 

We conduct the experiments using version SC2.4.6.2.69232, which is same as the SMAC run data release (https://github.com/oxwhirl/smac/releases/tag/v1).


## Installation
Set up StarCraft II and SMAC with the following command:

```bash
bash install_sc2.sh
```
It will download SC2.4.6.2.69232 into the 3rdparty folder and copy the maps necessary to run over. You also need to set the global environment variable:

```bash
export SC2PATH=[Your SC2 Path/StarCraftII]
```

Install Python environment with command:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
pip install numpy scipy pyyaml pygame pytest probscale imageio snakeviz 
```

## Training
To train OSSMC in smac , run this command:

```bash
python train.py --exp_name=3s5z
```

You can increase parameters with this form:

```bash
python train.py --exp_name=3s5z --seed=777
```

All configuration files are placed in configs, where `algos_cfgs/ossmc.yaml` contains some common parameters and algorithm hyperparameters. Smac related parameters are placed in `envs_cfgs/smac.yaml`.

## Acknowledgements
Thanks to the excellent open source framework Heterogeneous-Agent Reinforcement Learning, our work is based on https://sites.google.com/view/meharl.
## Ciation

If you found OSSMC useful, please consider citing with:
```
@article{HUANG2024106547,
title = {Optimistic sequential multi-agent reinforcement learning with motivational communication},
journal = {Neural Networks},
volume = {179},
pages = {106547},
year = {2024},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2024.106547},
url = {https://www.sciencedirect.com/science/article/pii/S0893608024004714},
author = {Anqi Huang and Yongli Wang and Xiaoliang Zhou and Haochen Zou and Xu Dong and Xun Che},
}
```


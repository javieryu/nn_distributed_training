## Introduction
This implementation of PPO is from https://github.com/ericyangyu/PPO-for-Beginners.
Please see that repo's README for documentation.
Slight modifications were made to train a MARL policy using PettingZoo.

## Usage

To train from scratch:
```
python main.py
```

To test model:
```
python main.py --mode test --actor_model trained/ppo_actor_tag.pth
```

To train with existing actor/critic models:
```
python main.py --actor_model trained/ppo_actor_tag.pth --critic_model trained/ppo_critic_tag.pth
```

NOTE: to change hyperparameters, environments, etc. do it in [main.py](main.py).

To plot the currently saved training statistics:
```
python plot_reward.py
```

## Misc.
I brought the pettingzoo module into this repo instead of pip installing it.
This is because pip install pettingzoo gets a version that is too old and specifying a more recent version in pip takes too long to install for whatever reason.
Also I had to edit the mpe environment for our experiment, so it was just easier to include all of our code then to tell users to edit some installed version of mpe with the same edits I have.
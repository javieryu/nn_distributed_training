import torch
import model
import cadmmPPO
from dist_ppo import DistPPOProblem
import gym
import sys
import networkx as nx

sys.path.insert(0, "../")
from pettingzoo.mpe import simple_tag_v2


def main():
    steps = 200
    env = simple_tag_v2.env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=8,
        max_cycles=steps,
        continuous_actions=True,
    )
    hyperparameters = {
        "timesteps_per_batch": 2000,
        "max_timesteps_per_episode": steps,
        "gamma": 0.99,
        "n_updates_per_iteration": 10,  # epochs
        "lr": 3e-4,
        "clip": 0.2,
        "render": False,
        "render_every_i": 1,
    }
    env.reset()
    obs_dim = env.observation_spaces["adversary_0"].shape[0]
    act_dim = env.action_spaces["adversary_0"].shape[0]

    base_actor = model.FFReLUNet([obs_dim, 64, 64, 64, act_dim])
    base_critic = model.FFReLUNet([obs_dim, 64, 64, 64, 1])
    graph = nx.wheel_graph(3)
    dppo = DistPPOProblem(
        base_actor, base_critic, graph, env, **hyperparameters
    )

    opt_confs = {
        "rho_init": 1.0,
        "rho_scaling": 1.0,
        "primal_lr_start": 1e-4,
        "primal_lr_finish": 0.001,
        "lr_decay_type": "constant",
        "persistant_primal_opt": False,
        "primal_iterations": 10,
        "outer_iterations": 500,
    }

    device = torch.device("cpu")

    dopt = cadmmPPO.CADMMPPO(dppo, device, opt_confs)
    dopt.train()


if __name__ == "__main__":
    main()

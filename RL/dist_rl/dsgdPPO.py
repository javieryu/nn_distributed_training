import torch
from utils import graph_generation
import math
import numpy as np


class DSGDPPO:
    def __init__(self, ddl_problem, device, conf):
        self.pr = ddl_problem
        self.conf = conf
        self.device = device

        # Compute consensus weight matrix
        self.W = graph_generation.get_metropolis(ddl_problem.graph)
        self.W = self.W.to(self.device)

        # Get list of all model parameter pointers
        self.plists_actor = {
            i: list(self.pr.actors[i].parameters()) for i in range(self.pr.N)
        }
        self.plists_critic = {
            i: list(self.pr.actors[i].parameters()) for i in range(self.pr.N)
        }

        # Useful numbers
        self.num_params_actor = len(self.plists_actor[0])
        self.num_params_critic = len(self.plists_critic[0])

        self.alph0 = conf["alpha0"]
        self.mu = conf["mu"]

    def train(self, profiler=None):
        # eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
        max_rl_timesteps = self.conf["max_rl_timesteps"]

        # Comm weights
        W = graph_generation.get_metropolis(self.pr.graph)
        W = W.to(self.device)

        # Optimization loop
        alph = self.alph0
        k = 0
        avg_ep_rews = []
        timesteps = []
        while self.pr.logger["t_so_far"] < max_rl_timesteps:
            self.pr.split_rollout_marl()
            self.pr.update_advantage()
            alph = alph * (1 - self.mu * alph)
            for _ in range(self.conf["n_updates_per_iteration"]):
                # Iterate over the agents for communication step
                for i in range(self.pr.N):
                    neighs = list(self.pr.graph.neighbors(i))
                    with torch.no_grad():
                        # Update each parameter individually across all neighbors
                        for p in range(self.num_params_actor):
                            # Ego update
                            self.plists_actor[i][p].multiply_(W[i, i])
                            # Neighbor updates
                            for j in neighs:
                                self.plists_actor[i][p].add_(
                                    W[i, j] * self.plists_actor[j][p]
                                )
                        for p in range(self.num_params_critic):
                            # Ego update
                            self.plists_critic[i][p].multiply_(W[i, i])
                            # Neighbor updates
                            for j in neighs:
                                self.plists_actor[i][p].add_(
                                    W[i, j] * self.plists_critic[j][p]
                                )

                # Compute the batch loss and update using the gradients
                for i in range(self.pr.N):
                    # Batch loss
                    actor_loss, critic_loss = self.pr.ev_ppo_loss(i)

                    actor_loss.backward(retain_graph=True)
                    # Locally update model with gradient
                    with torch.no_grad():
                        for p in range(self.num_params_actor):
                            self.plists_actor[i][p].add_(
                                -alph * self.plists_actor[i][p].grad
                            )
                            self.plists_actor[i][p].grad.zero_()

                    critic_loss.backward()
                    # Locally update model with gradient
                    with torch.no_grad():
                        for p in range(self.num_params_critic):
                            self.plists_critic[i][p].add_(
                                -alph * self.plists_critic[i][p].grad
                            )
                            self.plists_critic[i][p].grad.zero_()

            avg_ep_rews.append(
                np.mean(
                    [
                        np.sum(ep_rews)
                        for ep_rews in self.pr.logger["batch_rews"]
                    ]
                )
            )
            timesteps.append(self.pr.logger["t_so_far"])
            self.pr._log_summary()

            if profiler is not None:
                profiler.step()

            # Save our model if it's time
            if k % self.pr.save_freq == 0:
                # marl
                # predator-prey
                torch.save(
                    {
                        "actor0": self.pr.actors[0].state_dict(),
                        "actor1": self.pr.actors[1].state_dict(),
                        "actor2": self.pr.actors[2].state_dict(),
                    },
                    f'./trained/ppo_actors_tag_dsgd_{self.conf["ID"]}.pth',
                )
                torch.save(
                    {
                        "critic0": self.pr.critics[0].state_dict(),
                        "critic1": self.pr.critics[1].state_dict(),
                        "critic2": self.pr.critics[2].state_dict(),
                    },
                    f'./trained/ppo_critics_tag_dsgd_{self.conf["ID"]}.pth',
                )

                # save plotting data
                np.save(
                    f'./trained/avg_ep_rews_dsgd_{self.conf["ID"]}.npy',
                    np.asarray(avg_ep_rews),
                )
                np.save(
                    f'./trained/timesteps_dsgd_{self.conf["ID"]}.npy',
                    np.asarray(timesteps),
                )

            k += 1
        return

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
        agree_0 = np.array([])
        agree_1 = np.array([])
        agree_2 = np.array([])
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
            # Compute and save agreements
            with torch.no_grad():
                # The average distance from a single node to all of the other nodes in the problem
                actor_params = [torch.nn.utils.parameters_to_vector(self.pr.actors[i].parameters()) for i in range(self.pr.N)]
                critic_params = [torch.nn.utils.parameters_to_vector(self.pr.critics[i].parameters()) for i in range(self.pr.N)]

                # Stack all of the parameters into rows
                th_stack_a = torch.stack(actor_params)
                th_stack_c = torch.stack(critic_params)
                th_stack = torch.hstack((th_stack_a, th_stack_c))

                # Normalize the stack
                th_stack = torch.nn.functional.normalize(th_stack, dim=1)
                
                # Compute row-wise distances
                th_mean = torch.mean(th_stack, dim=0).reshape(1, -1)
                distances_mean = torch.cdist(th_stack, th_mean)
                agree_0 = np.append(agree_0, distances_mean[0].item())
                agree_1 = np.append(agree_1, distances_mean[1].item())
                agree_2 = np.append(agree_2, distances_mean[2].item())
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
                np.savez(f'./trained/agreements_dsgd_{self.conf["ID"]}', agree_0=agree_0, agree_1=agree_1, agree_2=agree_2)

            k += 1
        return

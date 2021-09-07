import torch
from utils import graph_generation
import copy
import numpy as np


class DSGTPPO:
    def __init__(self, ddl_problem, device, conf):
        self.pr = ddl_problem
        self.conf = conf
        self.device = device

        # Get list of all actor parameter pointers
        self.plists_actor = {
            i: list(self.pr.actors[i].parameters()) for i in range(self.pr.N)
        }
        self.num_params_actor = len(self.plists_actor[0])
        base_zeros_actor = [
            torch.zeros_like(p, requires_grad=False, device=self.device)
            for p in self.plists_actor[0]
        ]
        self.glists_actor = {
            i: copy.deepcopy(base_zeros_actor) for i in range(self.pr.N)
        }
        self.ylists_actor = {
            i: copy.deepcopy(base_zeros_actor) for i in range(self.pr.N)
        }

        # Get list of all critic parameter pointers
        self.plists_critic = {
            i: list(self.pr.critics[i].parameters()) for i in range(self.pr.N)
        }
        # Useful numbers
        self.num_params_critic = len(self.plists_critic[0])
        base_zeros_critic = [
            torch.zeros_like(p, requires_grad=False, device=self.device)
            for p in self.plists_critic[0]
        ]
        self.glists_critic = {
            i: copy.deepcopy(base_zeros_critic) for i in range(self.pr.N)
        }
        self.ylists_critic = {
            i: copy.deepcopy(base_zeros_critic) for i in range(self.pr.N)
        }

        # Training hyper params
        self.alpha_actor = conf["alpha_actor"]
        self.alpha_critic = conf["alpha_critic"]

    def train(self, profiler=None):
        # eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
        max_rl_timesteps = self.conf["max_rl_timesteps"]

        # Comm weights
        W = graph_generation.get_metropolis(self.pr.graph)
        W = W.to(self.device)

        # Initialize Ylists and Glists
        self.pr.split_rollout_marl()
        self.pr.update_advantage()
        for i in range(self.pr.N):
            # This requires one rollout but no steps are taken yet
            actor_loss, critic_loss = self.pr.ev_ppo_loss(i)

            actor_loss.backward(retain_graph=True)
            with torch.no_grad():
                for p in range(self.num_params_actor):
                    self.ylists_actor[i][p] = (
                        self.plists_actor[i][p].grad.detach().clone()
                    )
                    self.glists_actor[i][p] = (
                        self.plists_actor[i][p].grad.detach().clone()
                    )
                    self.plists_actor[i][p].grad.zero_()

            critic_loss.backward()
            with torch.no_grad():
                for p in range(self.num_params_critic):
                    self.ylists_critic[i][p] = (
                        self.plists_critic[i][p].grad.detach().clone()
                    )
                    self.glists_critic[i][p] = (
                        self.plists_critic[i][p].grad.detach().clone()
                    )
                    self.plists_critic[i][p].grad.zero_()

        # Optimization loop
        k = 0
        avg_ep_rews = []
        timesteps = []
        while self.pr.logger["t_so_far"] < max_rl_timesteps:
            # Compute graph weights
            # Iterate over the agents for communication step
            self.pr.split_rollout_marl()
            self.pr.update_advantage()
            for _ in range(self.conf["n_updates_per_iteration"]):
                for i in range(self.pr.N):
                    neighs = list(self.pr.graph.neighbors(i))
                    with torch.no_grad():
                        # Update each parameter individually across all neighbors
                        for p in range(self.num_params_actor):
                            # Ego update
                            self.plists_actor[i][p].multiply_(W[i, i])
                            self.plists_actor[i][p].add_(
                                -self.alpha_actor * self.ylists_actor[i][p]
                            )
                            # Neighbor updates
                            for j in neighs:
                                self.plists_actor[i][p].add_(
                                    W[i, j] * self.plists_actor[j][p]
                                )
                        for p in range(self.num_params_critic):
                            # Ego update
                            self.plists_critic[i][p].multiply_(W[i, i])
                            self.plists_critic[i][p].add_(
                                -self.alpha_critic * self.ylists_critic[i][p]
                            )
                            # Neighbor updates
                            for j in neighs:
                                self.plists_critic[i][p].add_(
                                    W[i, j] * self.plists_critic[j][p]
                                )

                # Compute the batch loss and update using the gradients
                for i in range(self.pr.N):
                    neighs = list(self.pr.graph.neighbors(i))
                    # print(i, " neighs: ", neighs)

                    # Compute PPO losses
                    actor_loss, critic_loss = self.pr.ev_ppo_loss(i)

                    actor_loss.backward(retain_graph=True)
                    # Locally update model with gradient
                    with torch.no_grad():
                        ysum = 0.0
                        gsum = 0.0
                        for p in range(self.num_params_actor):
                            self.ylists_actor[i][p].multiply_(W[i, i])
                            for j in neighs:
                                self.ylists_actor[i][p].add_(
                                    W[i, j] * self.ylists_actor[j][p]
                                )

                            self.ylists_actor[i][p].add_(
                                self.plists_actor[i][p].grad
                            )
                            self.ylists_actor[i][p].add_(
                                -1.0 * self.glists_actor[i][p]
                            )

                            self.glists_actor[i][p] = (
                                self.plists_actor[i][p].grad.detach().clone()
                            )
                            self.plists_actor[i][p].grad.zero_()

                    critic_loss.backward()
                    #                    torch.nn.utils.clip_grad_norm_(
                    #                        self.pr.critics[i].parameters(), 1000.0
                    #                    )
                    # Locally update model with gradient
                    with torch.no_grad():
                        ysum = 0.0
                        gsum = 0.0
                        for p in range(self.num_params_critic):
                            self.ylists_critic[i][p].multiply_(W[i, i])
                            for j in neighs:
                                self.ylists_critic[i][p].add_(
                                    W[i, j] * self.ylists_critic[j][p]
                                )

                            self.ylists_critic[i][p].add_(
                                self.plists_critic[i][p].grad
                            )
                            self.ylists_critic[i][p].add_(
                                -1.0 * self.glists_critic[i][p]
                            )

                            self.glists_critic[i][p] = (
                                self.plists_critic[i][p].grad.detach().clone()
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
                    f'./trained/ppo_actors_tag_dsgt_{self.conf["ID"]}.pth',
                )
                torch.save(
                    {
                        "critic0": self.pr.critics[0].state_dict(),
                        "critic1": self.pr.critics[1].state_dict(),
                        "critic2": self.pr.critics[2].state_dict(),
                    },
                    f'./trained/ppo_critics_tag_dsgt_{self.conf["ID"]}.pth',
                )

                # save plotting data
                np.save(
                    f'./trained/avg_ep_rews_dsgt_{self.conf["ID"]}.npy',
                    np.asarray(avg_ep_rews),
                )
                np.save(
                    f'./trained/timesteps_dsgt_{self.conf["ID"]}.npy',
                    np.asarray(timesteps),
                )

            k += 1

        return

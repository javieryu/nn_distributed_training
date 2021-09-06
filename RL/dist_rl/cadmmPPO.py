import torch
import math
import numpy as np


class CADMMPPO:
    def __init__(self, ddl_problem, device, conf):
        self.pr = ddl_problem
        self.conf = conf

        self.duals_actor = {
            i: torch.zeros((self.pr.n_actor), device=device)
            for i in range(self.pr.N)
        }

        self.duals_critic = {
            i: torch.zeros((self.pr.n_critic), device=device)
            for i in range(self.pr.N)
        }

        self.rho = self.conf["rho_init"]
        self.rho_scaling = self.conf["rho_scaling"]
        if self.conf["lr_decay_type"] == "constant":
            self.primal_lr = self.conf["primal_lr_start"] * torch.ones(
                self.conf["outer_iterations"]
            )
        elif self.conf["lr_decay_type"] == "linear":
            self.primal_lr = torch.linspace(
                self.conf["primal_lr_start"],
                self.conf["primal_lr_finish"],
                self.conf["outer_iterations"],
            )
        elif self.conf["lr_decay_type"] == "log":
            self.primal_lr = torch.logspace(
                math.log(self.conf["primal_lr_start"], 10),
                math.log(self.conf["primal_lr_finish"], 10),
                self.conf["outer_iterations"],
            )
        else:
            raise NameError("Unknow primal learning rate decay type.")
        self.pits = self.conf["primal_iterations"]

        if self.conf["persistant_primal_opt"]:
            self.opts = {}
            for i in range(self.pr.N):
                if self.conf["primal_optimizer"] == "adam":
                    self.opts[i] = torch.optim.Adam(
                        self.pr.models[i].parameters(), self.primal_lr[0]
                    )
                elif self.conf["primal_optimizer"] == "sgd":
                    self.opts[i] = torch.optim.SGD(
                        self.pr.models[i].parameters(), self.primal_lr[0]
                    )
                elif self.conf["primal_optimizer"] == "adamw":
                    self.opts[i] = torch.optim.AdamW(
                        self.pr.models[i].parameters(), self.primal_lr[0]
                    )
                else:
                    raise NameError("CADMM primal optimizer is unknown.")

    def primal_update(self, i, th_reg_actor, th_reg_critic, k):
        # if self.conf["persistant_primal_opt"]:
        #    opt = self.opts[i]
        # else:
        #    if self.conf["primal_optimizer"] == "adam":
        #        opt = torch.optim.Adam(
        #            self.pr.models[i].parameters(), self.primal_lr[k]
        #        )
        #    elif self.conf["primal_optimizer"] == "sgd":
        #        opt = torch.optim.SGD(
        #            self.pr.models[i].parameters(), self.primal_lr[k]
        #        )
        #    elif self.conf["primal_optimizer"] == "adamw":
        #        opt = torch.optim.AdamW(
        #            self.pr.models[i].parameters(), self.primal_lr[k]
        #        )
        #    else:
        #        raise NameError("CADMM primal optimizer is unknown.")

        opt_actor = torch.optim.Adam(
            self.pr.actors[i].parameters(), lr=self.primal_lr[0]
        )
        opt_critic = torch.optim.Adam(
            self.pr.critics[i].parameters(), lr=self.primal_lr[0]
        )

        for _ in range(self.pits):

            # Model pass on the batch
            # pred_loss = self.pr.local_batch_loss(i)
            actor_loss, critic_loss = self.pr.ev_ppo_loss(i)

            # Get the primal variable WITH the autodiff graph attached.
            th_actor = torch.nn.utils.parameters_to_vector(
                self.pr.actors[i].parameters()
            )

            th_critic = torch.nn.utils.parameters_to_vector(
                self.pr.critics[i].parameters()
            )

            reg_actor = torch.sum(
                torch.square(
                    torch.cdist(th_actor.reshape(1, -1), th_reg_actor)
                )
            )
            reg_critic = torch.sum(
                torch.square(
                    torch.cdist(th_critic.reshape(1, -1), th_reg_critic)
                )
            )

            aloss = (
                actor_loss
                + torch.dot(th_actor, self.duals_actor[i])
                + self.rho * reg_actor
            )
            closs = (
                critic_loss
                + torch.dot(th_critic, self.duals_critic[i])
                + self.rho * reg_critic
            )

            opt_actor.zero_grad()
            aloss.backward(retain_graph=True)
            opt_actor.step()

            opt_critic.zero_grad()
            closs.backward()
            opt_critic.step()

        return

    def train(self, profiler=None):
        # eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
        k = 0
        avg_ep_rews = []
        timesteps = []
        while self.pr.logger["t_so_far"] < self.conf["max_rl_timesteps"]:
            self.pr.split_rollout_marl()
            self.pr.update_advantage()

            # Get the current primal variables
            ths_actor = {
                i: torch.nn.utils.parameters_to_vector(
                    self.pr.actors[i].parameters()
                )
                .clone()
                .detach()
                for i in range(self.pr.N)
            }

            ths_critic = {
                i: torch.nn.utils.parameters_to_vector(
                    self.pr.critics[i].parameters()
                )
                .clone()
                .detach()
                for i in range(self.pr.N)
            }

            # Update the penalty parameter
            self.rho *= self.rho_scaling

            # Per node updates
            for i in range(self.pr.N):
                neighs = list(self.pr.graph.neighbors(i))
                thj_actor = torch.stack([ths_actor[j] for j in neighs])
                thj_critic = torch.stack([ths_critic[j] for j in neighs])

                self.duals_actor[i] += self.rho * torch.sum(
                    ths_actor[i] - thj_actor, dim=0
                )
                self.duals_critic[i] += self.rho * torch.sum(
                    ths_critic[i] - thj_critic, dim=0
                )
                th_reg_actor = (thj_actor + ths_actor[i]) / 2.0
                th_reg_critic = (thj_critic + ths_critic[i]) / 2.0

                self.primal_update(i, th_reg_actor, th_reg_critic, k)

            avg_ep_rews.append(np.mean([np.sum(ep_rews) for ep_rews in self.pr.logger['batch_rews']]))
            timesteps.append(self.pr.logger["t_so_far"])
            self.pr._log_summary()

            if profiler is not None:
                profiler.step()

            # Save our model if it's time
            if k % self.pr.save_freq == 0:
                # marl
                # predator-prey
                torch.save({'actor0': self.pr.actors[0].state_dict(), 
                            'actor1': self.pr.actors[1].state_dict(),
                            'actor2': self.pr.actors[2].state_dict()},f'./trained/ppo_actors_tag_cadmm_{self.conf["ID"]}.pth')
                torch.save({'critic0': self.pr.critics[0].state_dict(), 
                            'critic1': self.pr.critics[1].state_dict(),
                            'critic2': self.pr.critics[2].state_dict()},f'./trained/ppo_critics_tag_cadmm_{self.conf["ID"]}.pth')

                # save plotting data
                np.save(f'./trained/avg_ep_rews_cadmm_{self.conf["ID"]}.npy', np.asarray(avg_ep_rews))
                np.save(f'./trained/timesteps_cadmm_{self.conf["ID"]}.npy', np.asarray(timesteps))

            k += 1

        return

"""
    The file contains the PPO class to train with.
    NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
            It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import gym
import time
import copy

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal


class DistPPOProblem:
    """
    This is the PPO class we will use as our model in main.py
    """

    def __init__(self, base_actor, base_critic, graph, env, **hyperparameters):
        """
        Initializes the PPO model, including hyperparameters.

        Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

        Returns:
                None
        """
        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env

        # marl
        self.obs_dim = env.observation_spaces["adversary_0"].shape[0]
        self.act_dim = env.action_spaces["adversary_0"].shape[0]

        # Graph and n
        self.graph = graph
        self.N = graph.number_of_nodes()
        self.n_actor = torch.nn.utils.parameters_to_vector(
            base_actor.parameters()
        ).shape[0]
        self.n_critic = torch.nn.utils.parameters_to_vector(
            base_critic.parameters()
        ).shape[0]

        # Initialize actor and critic networks
        # self.actors = policy_class(self.obs_dim, self.act_dim)  # ALG STEP 1
        self.actors = {i: copy.deepcopy(base_actor) for i in range(self.N)}
        self.critics = {i: copy.deepcopy(base_critic) for i in range(self.N)}

        # Initialize optimizers for actor and critic
        # self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        # self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,  # timesteps so far
            "i_so_far": 0,  # iterations so far
            "batch_lens": [],  # episodic lengths in batch
            "batch_rews": [],  # episodic returns in batch
            "actor_losses": [],  # losses of actor network in current iteration
        }

    def learn(self, total_timesteps):
        """
        Train the actor and critic networks. Here is where the main PPO algorithm resides.

        Parameters:
                total_timesteps - the total number of timesteps to train for

        Return:
                None
        """
        print(
            f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ",
            end="",
        )
        print(
            f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps"
        )
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        avg_ep_rews = []
        timesteps = []
        while t_so_far < total_timesteps:  # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            # batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()                     # ALG STEP 3
            (
                batch_obs,
                batch_acts,
                batch_log_probs,
                batch_rtgs,
                batch_lens,
            ) = self.rollout_marl()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger["t_so_far"] = t_so_far
            self.logger["i_so_far"] = i_so_far

            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()  # ALG STEP 5

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                # For why we use log probabilities instead of actual probabilities,
                # here's a great explanation:
                # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                # TL;DR makes gradient ascent easier behind the scenes.
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger["actor_losses"].append(actor_loss.detach())

            # collect data for plotting
            avg_ep_rews.append(
                np.mean(
                    [np.sum(ep_rews) for ep_rews in self.logger["batch_rews"]]
                )
            )
            timesteps.append(t_so_far)

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                # pendulum
                # torch.save(self.actor.state_dict(), './trained/ppo_actor_pend.pth')
                # torch.save(self.critic.state_dict(), './trained/ppo_critic_pend.pth')

                # marl
                # predator-prey
                torch.save(
                    self.actor.state_dict(), "./trained/ppo_actor_tag.pth"
                )
                torch.save(
                    self.critic.state_dict(), "./trained/ppo_critic_tag.pth"
                )

                # save plotting data
                np.save("./trained/avg_ep_rews.npy", np.asarray(avg_ep_rews))
                np.save("./trained/timesteps.npy", np.asarray(timesteps))

    def heuristic(self, obs):
        """
        Get action such that evader moves opposite closest adversary and doesn't move past boundary.

        Parameters:
                obs - the evader observation

        Return:
                action
        """
        # Get adversary relative distances
        dists = np.zeros((self.env.num_agents - 1, 2))
        for i, j in enumerate(range(4, obs.size, 2)):
            dists[i, 0] = obs[j]
            dists[i, 1] = obs[j + 1]

        # find closest adversary
        min_d = float("inf")
        near_ad = 10000
        for i in range(dists.shape[0]):
            d = np.sqrt(np.sum(np.square(dists[i, :])))
            if d < min_d:
                min_d = d
                near_ad = i

        # Set action as opposite closest adversary, normalized to fit action_space of [0, 1]
        # action = [None, Right, Left, Up, Down]
        force = -dists[near_ad, :] / max(abs(-dists[near_ad, :]))
        action = np.zeros(5)
        if force[0] > 0:
            action[1] = force[0]
        else:
            action[2] = -force[0]
        if force[1] > 0:
            action[3] = force[1]
        else:
            action[4] = -force[1]

        # edit action to prevent leaving box
        if obs[2] <= -1.2:
            action[2] = 0.0
        elif obs[2] >= 1.2:
            action[1] = 0.0
        if obs[3] <= -1.2:
            action[4] = 0.0
        elif obs[3] >= 1.2:
            action[3] = 0.0
        return action

    def ev_ppo_loss(self, i):
        # Calculate V_phi and pi_theta(a_t | s_t)
        V, ev_log_probs = self.evaluate(i)

        # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
        # NOTE: we just subtract the logs, which is the same as
        # dividing the values and then canceling the log with e^log.
        # For why we use log probabilities instead of actual probabilities,
        # here's a great explanation:
        # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
        # TL;DR makes gradient ascent easier behind the scenes.
        ratios = torch.exp(ev_log_probs - self.curr_log_probs[i])

        # Calculate surrogate losses.
        surr1 = ratios * self.A_k[i]
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * self.A_k[i]

        # Calculate actor and critic losses.
        # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
        # the performance function, but Adam minimizes the loss. So minimizing the negative
        # performance function maximizes it.
        actor_loss = (-torch.min(surr1, surr2)).mean()
        critic_loss = nn.MSELoss()(V, self.curr_rtgs[i])

        self.logger["actor_losses"].append(actor_loss.detach())

        return actor_loss, critic_loss

    def update_advantage(self):
        self.A_k = {}
        for i in range(self.N):
            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(i)
            self.A_k[i] = self.curr_rtgs[i] - V.detach()  # ALG STEP 5
            # Normalize advantage
            self.A_k[i] = (self.A_k[i] - self.A_k[i].mean()) / (
                self.A_k[i].std() + 1e-10
            )

        return

    def split_rollout_marl(self):
        """
        Too many transformers references, I'm sorry. This is where we collect the batch of data
        from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
        of data each time we iterate the actor/critic networks.

        Parameters:
                None

        Return:
                batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
                batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """
        # Batch data. For more details, check function header.

        batch_obs = {i: [] for i in range(self.N)}
        batch_acts = {i: [] for i in range(self.N)}
        batch_log_probs = {i: [] for i in range(self.N)}
        batch_rews = {i: [] for i in range(self.N)}
        batch_rtgs = {i: [] for i in range(self.N)}
        batch_lens = []
        joint_batch_rews = []
        joint_ep_rews = []

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = {
                i: [] for i in range(self.N)
            }  # rewards collected per episode

            # Reset the environment. Note that obs is short for observation.
            self.env.reset()
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            i = 0
            for ep_t, agent in enumerate(self.env.agent_iter()):
                obs, rew, done, _ = self.env.last()  # get agent observation

                if agent == "agent_0":  # the prey
                    # action = np.random.rand(5) if not done else None
                    action = self.heuristic(obs)
                    self.env.step(action)
                    i = 0

                else:  # predators
                    t += 1  # Increment timesteps ran this batch so far
                    # Track observations in this batch
                    # Don't track observations for which we won't act
                    if (
                        ep_t
                        <= self.max_timesteps_per_episode - self.env.num_agents
                    ):
                        batch_obs[i].append(obs)

                    # Calculate action and make a step in the env.
                    if not done:
                        action, log_prob = self.get_action(i, obs)
                        self.env.step(action)
                    else:
                        action, log_prob = (
                            np.array(
                                [1.0, 0.0, 0.0, 0.0, 0.0], dtype="float32"
                            ),
                            0.0,
                        )
                        self.env.step(None)

                    if self.render:
                        self.env.render()

                    # Track recent reward, action, and action log probability
                    if (
                        ep_t >= self.env.num_agents
                    ):  # first reward from env.last() isn't associated with first action. ep_t starts at 0
                        ep_rews[i].append(rew)
                        joint_ep_rews.append(rew)

                    # Don't track actions for which we won't observe an associated reward
                    if (
                        ep_t
                        <= self.max_timesteps_per_episode - self.env.num_agents
                    ):
                        batch_acts[i].append(action)
                        batch_log_probs[i].append(log_prob)

                    i += 1

                # If the environment tells us the episode is terminated, break
                if ep_t == self.max_timesteps_per_episode:
                    break

            # Track episodic lengths and rewards
            batch_lens.append(ep_t)
            joint_batch_rews.append(joint_ep_rews)
            for i in range(self.N):
                batch_rews[i].append(ep_rews[i])

        # Reshape data as tensors in the shape specified in function description, before returning
        for i in range(self.N):
            batch_obs[i] = torch.tensor(batch_obs[i], dtype=torch.float)
            batch_acts[i] = torch.tensor(batch_acts[i], dtype=torch.float)
            batch_log_probs[i] = torch.tensor(
                batch_log_probs[i], dtype=torch.float
            )
            batch_rtgs[i] = self.compute_rtgs(batch_rews[i])  # ALG STEP 4

        # Log the episodic returns and episodic lengths in this batch.
        self.logger["batch_rews"] = joint_batch_rews
        self.logger["batch_lens"] = batch_lens
        self.logger["t_so_far"] += np.sum(batch_lens)
        self.logger["i_so_far"] += 1

        self.curr_obs = batch_obs
        self.curr_acts = batch_acts
        self.curr_log_probs = batch_log_probs
        self.curr_rtgs = batch_rtgs

        return  # batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        """
        Compute the Reward-To-Go of each timestep in a batch given the rewards.

        Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

        Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_action(self, i, obs):
        """
        Queries an action from the actor network, should be called from rollout.

        Parameters:
                obs - the observation at the current timestep

        Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        mean = self.actors[i](obs)

        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, i):
        """
        Estimate the values of each observation, and the log probs of
        each action in the most recent batch with the most recent
        iteration of the actor network. Should be called from learn.

        Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                                        Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                                        Shape: (number of timesteps in batch, dimension of action)

        Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critics[i](self.curr_obs[i]).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actors[i](self.curr_obs[i])
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(self.curr_acts[i])

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def _init_hyperparameters(self, hyperparameters):
        """
        Initialize default and custom values for hyperparameters

        Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                                        hyperparameters defined below with custom values.

        Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = (
            1600  # Max number of timesteps per episode
        )
        self.n_updates_per_iteration = (
            5  # Number of times to update actor/critic per iteration (epochs)
        )
        self.lr = 0.005  # Learning rate of actor optimizer
        self.gamma = 0.95  # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA

        # Miscellaneous parameters
        self.render = True  # If we should render during rollout
        self.render_every_i = 10  # Only render every n iterations
        self.save_freq = (
            10  # How often we save the NNs in number of iterations
        )
        self.seed = None  # Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec("self." + param + " = " + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert type(self.seed) == int

            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
        Print to stdout what we've logged so far in the most recent batch.

        Parameters:
                None

        Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger["delta_t"]
        self.logger["delta_t"] = time.time_ns()
        delta_t = (self.logger["delta_t"] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger["t_so_far"]
        i_so_far = self.logger["i_so_far"]
        avg_ep_lens = np.mean(self.logger["batch_lens"])
        avg_ep_rews = np.mean(
            [np.sum(ep_rews) for ep_rews in self.logger["batch_rews"]]
        )
        avg_actor_loss = np.mean(
            [losses.float().mean() for losses in self.logger["actor_losses"]]
        )

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(
            f"-------------------- Iteration #{i_so_far} --------------------",
            flush=True,
        )
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(
            f"------------------------------------------------------",
            flush=True,
        )
        print(flush=True)

        # Reset batch-specific logging data
        self.logger["batch_lens"] = []
        self.logger["batch_rews"] = []
        self.logger["actor_losses"] = []

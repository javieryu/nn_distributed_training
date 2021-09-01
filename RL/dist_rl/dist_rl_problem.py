import torch
import copy


class DistPPOProblem:
    def __init__(self, base_actor, base_critic, rl_env, graph):
        self.graph = graph
        self.N = graph.number_of_nodes()

        self.actors = {i: copy.deepcopy(base_actor) for i in range(self.N)}
        self.critics = {i: copy.deepcopy(base_critic) for i in range(self.N)}
    
    def rollout_marl(self):
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
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_rtgs = []
		batch_lens = []

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		ep_rews = []

		t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			ep_rews = [] # rewards collected per episode

			# Reset the environment. Note that obs is short for observation. 
			self.env.reset()
			done = False
			
			# Run an episode for a maximum of max_timesteps_per_episode timesteps			
			for ep_t, agent in enumerate(self.env.agent_iter()):
				obs, rew, done, _ = self.env.last() # get agent observation 

				if agent == "agent_0": # the prey
					# action = np.random.rand(5) if not done else None
					action = self.heuristic(obs)
					self.env.step(action)
					
				else: # predators
					t += 1 # Increment timesteps ran this batch so far
					# Track observations in this batch
					# Don't track observations for which we won't act
					if ep_t <= self.max_timesteps_per_episode - self.env.num_agents:
						batch_obs.append(obs)
					
					# Calculate action and make a step in the env.
					if not done:
						action, log_prob = self.get_action(obs)
						self.env.step(action)
					else:
						action, log_prob = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype='float32'), 0.0
						self.env.step(None)
					
					if self.render:
						self.env.render()

					# Track recent reward, action, and action log probability
					if ep_t >= self.env.num_agents: # first reward from env.last() isn't associated with first action. ep_t starts at 0
						ep_rews.append(rew)
					
					# Don't track actions for which we won't observe an associated reward
					if ep_t <= self.max_timesteps_per_episode - self.env.num_agents:
						batch_acts.append(action)
						batch_log_probs.append(log_prob)
				
				# If the environment tells us the episode is terminated, break
				if ep_t == self.max_timesteps_per_episode:
					break

			# Track episodic lengths and rewards
			batch_lens.append(ep_t)
			batch_rews.append(ep_rews)

		# Reshape data as tensors in the shape specified in function description, before returning
		batch_obs = torch.tensor(batch_obs, dtype=torch.float)
		batch_acts = torch.tensor(batch_acts, dtype=torch.float)
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
		batch_rtgs = self.compute_rtgs(batch_rews) # ALG STEP 4

		# Log the episodic returns and episodic lengths in this batch.
		#self.logger['batch_rews'] = batch_rews
		#self.logger['batch_lens'] = batch_lens

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens       
    
    def compute_ep_loss(self):

        

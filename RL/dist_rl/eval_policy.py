import sys
sys.path.insert(0, "../")

import torch
import model
import gym
import networkx as nx
import numpy as np

from pettingzoo.mpe import simple_tag_v2


def _log_summary(ep_len, ep_ret, ep_num):
		"""
			Print to stdout what we've logged so far in the most recent episode.
			Parameters:
				None
			Return:
				None
		"""
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

def heuristic(env, obs):
	# Get adversary relative distances
	dists = np.zeros((env.num_agents-1, 2))
	for i,j in enumerate(range(4, obs.size, 2)):
		dists[i,0] = obs[j]
		dists[i,1] = obs[j+1]

	# find closest adversary
	min_d = float('inf')
	near_ad = 10000
	for i in range(dists.shape[0]):
		d = np.sqrt(np.sum(np.square(dists[i,:])))
		if d < min_d:
			min_d = d
			near_ad = i

	# Set action as opposite closest adversary, normalized to fit action_space of [0, 1]
	# action = [None, Right, Left, Up, Down]
	force = -dists[near_ad,:] / max(abs(-dists[near_ad,:]))
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

def rollout_marl(actors, env, render, state_log=False):
	"""
		Returns a generator to roll out each episode given a trained policy and
		environment to test on. 
		Parameters:
			policy - The trained policy to test
			env - The environment to evaluate the policy on
			render - Specifies whether to render or not
		
		Return:
			A generator object rollout, or iterable, which will return the latest
			episodic length and return on each iteration of the generator.
	"""
	# Rollout until user kills process
	while True:
		obs = env.reset()
		done = False

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return
		positions = {'agent_0': np.zeros((200,2)),
					 'adversary_0': np.zeros((200,2)),
					 'adversary_1': np.zeros((200,2)),
					 'adversary_2': np.zeros((200,2)),}

		i = 0 # agent index
		# Run an episode for a maximum of max_timesteps_per_episode timesteps			
		for ep_t, agent in enumerate(env.agent_iter()):
			obs, rew, done, _ = env.last() # get agent observation 

			if i == 0:
				positions['agent_0']

			if agent == "agent_0": # the prey
				# action = np.random.rand(5) if not done else None
				action = heuristic(env, obs)
				env.step(action)
				i = 0
				
			else: # predators
				t += 1 # Increment timesteps ran this batch so far

				# Calculate action and make a step in the env.
				if not done:
					action = actors[i](obs).detach().numpy()
					env.step(action)
				else:
					env.step(None)
				
				if render:
					env.render()

				# Track recent reward, action, and action log probability
				ep_ret += rew
				i += 1

				# If the environment tells us the episode is terminated, break
				if ep_t == 200:
					break

		# Track episodic length
		ep_len = t

		if state_log:
			return positions

		# returns episodic length and return in this iteration
		yield ep_len, ep_ret







### Scripting ###

env = simple_tag_v2.env(
    num_good=1,
    num_adversaries=3,
    num_obstacles=8,
    max_cycles=200,
    continuous_actions=True,
)

env.reset()
obs_dim = env.observation_spaces["adversary_0"].shape[0]
act_dim = env.action_spaces["adversary_0"].shape[0]


# load policy networks
actor0 = model.FFReLUNet([obs_dim, 64, 64, 64, act_dim])
actor1 = model.FFReLUNet([obs_dim, 64, 64, 64, act_dim])
actor2 = model.FFReLUNet([obs_dim, 64, 64, 64, act_dim])

actor_models = torch.load('./trained/ppo_actors_tag.pth')
actor0.load_state_dict(actor_models['actor0'])
actor1.load_state_dict(actor_models['actor1'])
actor2.load_state_dict(actor_models['actor2'])

actors = {0: actor0, 1: actor1, 2: actor2}

render = True

# Rollout with the policy and environment, and log each episode's data
for ep_num, (ep_len, ep_ret) in enumerate(rollout_marl(actors, env, render)):
	_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)


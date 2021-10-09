import sys
sys.path.insert(0, "../")

import torch
import model
import gym
import networkx as nx
import numpy as np
from pettingzoo.mpe import simple_tag_v2
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib.animation as animation 


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

def rollout_marl(actors, env, render, state_log=False, seed=None):
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
		env.seed(seed=seed)
		obs = env.reset()
		done = False

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return
		positions = {'agent_0': np.zeros((1,2)),
					 'adversary_0': np.zeros((1,2)),
					 'adversary_1': np.zeros((1,2)),
					 'adversary_2': np.zeros((1,2)),}

		i = 0 # agent index
		j = 0 # position index
		# Run an episode for a maximum of max_timesteps_per_episode timesteps			
		for ep_t, agent in enumerate(env.agent_iter()):
			obs, rew, done, _ = env.last() # get agent observation 

			if agent == "agent_0": # the prey
				positions['agent_0'] = np.vstack((positions['agent_0'], [obs[2:4]]))
				# action = np.random.rand(5) if not done else None
				action = heuristic(env, obs)
				env.step(action)
				i = 0
				
			else: # predators
				t += 1 # Increment timesteps ran this batch so far

				if agent == "adversary_0":
					positions['adversary_0'] = np.vstack((positions['adversary_0'], [obs[2:4]]))
				elif agent == "adversary_1":
					positions['adversary_1'] = np.vstack((positions['adversary_1'], [obs[2:4]]))
				elif agent == "adversary_2":
					positions['adversary_2'] = np.vstack((positions['adversary_2'], [obs[2:4]]))


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
			positions['agent_0'] = positions['agent_0'][1:,:]
			positions['adversary_0'] = positions['adversary_0'][1:,:]
			positions['adversary_1'] = positions['adversary_1'][1:,:]
			positions['adversary_2'] = positions['adversary_2'][1:,:]
			return positions

		# returns episodic length and return in this iteration
		# yield ep_len, ep_ret


def rollout_render(env):
	env.reset()
	obs_dim = env.observation_spaces["adversary_0"].shape[0]
	act_dim = env.action_spaces["adversary_0"].shape[0]
	render = True
	# seed = 1 # 1!, 5, 21, 38, 50
	act_model = './trained/ppo_actors_tag_cadmm_40.pth'

	# Rollout single learned policy
	actor0 = model.FFReLUNet([obs_dim, 64, 64, 64, act_dim])
	actor1 = model.FFReLUNet([obs_dim, 64, 64, 64, act_dim])
	actor2 = model.FFReLUNet([obs_dim, 64, 64, 64, act_dim])
	
	actor_models = torch.load(act_model)
	actor0.load_state_dict(actor_models['actor0'])
	actor1.load_state_dict(actor_models['actor1'])
	actor2.load_state_dict(actor_models['actor2'])
	actors = {0: actor0, 1: actor1, 2: actor2}
	positions = rollout_marl(actors, env, render, state_log=True)
	return


def interpolate(x, y, res):
	x, y = x, y  ##self.poly.xy[:].T
	i = np.arange(len(x))
	interp_i = np.linspace(0, i.max(), res * i.max())
	xi = interp.interp1d(i, x, kind="cubic")(interp_i)
	yi = interp.interp1d(i, y, kind="cubic")(interp_i)
	return xi, yi


def create_gif(env, policy, seed):
	env.reset()
	obs_dim = env.observation_spaces["adversary_0"].shape[0]
	act_dim = env.action_spaces["adversary_0"].shape[0]
	render = False

	# Rollout with the policy and environment, and log each episode's data
	actor_models = torch.load(f'./trained/ppo_actors_tag_cadmm_50_{policy}.pth')
	actor0 = model.FFReLUNet([obs_dim, 64, 64, 64, act_dim])
	actor1 = model.FFReLUNet([obs_dim, 64, 64, 64, act_dim])
	actor2 = model.FFReLUNet([obs_dim, 64, 64, 64, act_dim])
	actor0.load_state_dict(actor_models['actor0'])
	actor1.load_state_dict(actor_models['actor1'])
	actor2.load_state_dict(actor_models['actor2'])
	actors = {0: actor0, 1: actor1, 2: actor2}
	positions = rollout_marl(actors, env, render, state_log=True, seed=seed)
	positions['obstacles'] = np.array([[-1.2, -0.6], [0.1, -1.1], [-0.3, 0.4], [0.9, 0.75],
						   [-0.9, 1.2], [-0.1, 1.3], [-1.2, 0.0], [1.3, 0.0]])

	# Plotting #
	plt.rcParams.update({'font.size': 16})
	# (fig, ax0) = plt.subplots(figsize=(8, 8), tight_layout=True)
	# plt.xlim([-2.5, 2.5])
	# plt.ylim([-2.5, 2.5])

	fig = plt.figure(figsize=(8, 8), tight_layout=True)
	# plt.xlim([-2.5, 2.5])
	# plt.ylim([-2.5, 2.5])
	ax0 = plt.gca()
	ax0.set_aspect(1)
	ax0.set_xlim(-2.5, 2.5)

	t_steps = min(positions['agent_0'].shape[0], positions['adversary_0'].shape[0], positions['adversary_1'].shape[0], positions['adversary_2'].shape[0])

	

	def animate(i):
		ax0.clear()
		ax0.set_aspect(1)
		ax0.set_xlim(-2.5, 2.5)
		ax0.set_ylim(-2.5, 2.5)
		patches = [plt.gca().add_patch(plt.Circle((positions['agent_0'][i,:]), 0.05, fc=[0.35, 0.85, 0.35]) ),
		ax0.add_patch(plt.Circle((positions['adversary_0'][i,:]), 0.075, fc=[0.85, 0.35, 0.35]) ),
		ax0.add_patch(plt.Circle((positions['adversary_1'][i,:]), 0.075, fc=[0.85, 0.35, 0.35]) ),
		ax0.add_patch(plt.Circle((positions['adversary_2'][i,:]), 0.075, fc=[0.85, 0.35, 0.35]) )]

		# add landmarks
		for i in range(positions['obstacles'].shape[0]):
			patches.append(ax0.add_patch(plt.Circle((positions['obstacles'][i,:]), 0.2, fc=[0.25, 0.25, 0.25])))

		return patches

	anim = animation.FuncAnimation(fig, animate, frames=t_steps, interval=47, blit=True, repeat=False)
	# plt.show()
	# f = r"'./vids.mp4" 
	# writervideo = animation.FFMpegWriter(fps=60) 
	FFwriter = animation.FFMpegWriter(fps=30)
	anim.save(f'vids/cadmm_50_{policy}.mp4', writer=FFwriter)
	return
	



### Scripting ###


env = simple_tag_v2.env(
    num_good=1,
    num_adversaries=3,
    num_obstacles=8,
    max_cycles=200,
    continuous_actions=True)

seed = 12
for policy in [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3570]:
	create_gif(env, policy, seed)


# rollout_render(env)

# env.reset()
# obs_dim = env.observation_spaces["adversary_0"].shape[0]
# act_dim = env.action_spaces["adversary_0"].shape[0]
# render = False
# seed = 1 # 1!, 5, 21, 38, 50


# # Rollout with the policy and environment, and log each episode's data
# # load policy networks
# actor0 = model.FFReLUNet([obs_dim, 64, 64, 64, act_dim])
# actor1 = model.FFReLUNet([obs_dim, 64, 64, 64, act_dim])
# actor2 = model.FFReLUNet([obs_dim, 64, 64, 64, act_dim])
# # Change specific models here:
# actor_models = torch.load('./trained/ppo_actors_tag_cadmm_40.pth')
# actor0.load_state_dict(actor_models['actor0'])
# actor1.load_state_dict(actor_models['actor1'])
# actor2.load_state_dict(actor_models['actor2'])
# actors = {0: actor0, 1: actor1, 2: actor2}
# positions = rollout_marl(actors, env, render, state_log=True, seed=seed)
# positions['obstacles'] = np.array([[-1.2, -0.6], [0.1, -1.1], [-0.3, 0.4], [0.9, 0.75],
# 					   [-0.9, 1.2], [-0.1, 1.3], [-1.2, 0.0], [1.3, 0.0]])



# ## For 1 Trajectory ###
# plt.rcParams.update({'font.size': 16})
# (fig, ax0) = plt.subplots(figsize=(10, 8), tight_layout=True)
# t_steps = min(positions['agent_0'].shape[0], positions['adversary_0'].shape[0], positions['adversary_1'].shape[0], positions['adversary_2'].shape[0])

# # add landmarks
# for i in range(positions['obstacles'].shape[0]):
# 	plt.gca().add_patch(plt.Circle((positions['obstacles'][i,:]), 0.2, fc=[0.25, 0.25, 0.25]))

# mod = 4
# # add smooth trajectories
# xspline, yspline = interpolate(positions['agent_0'][0:t_steps-mod + 2,0], positions['agent_0'][0:t_steps-mod + 2,1], 120)
# plt.plot(xspline, yspline, color=[0.35, 0.85, 0.35], alpha=0.5)
# xspline, yspline = interpolate(positions['adversary_0'][0:t_steps-mod + 2,0], positions['adversary_0'][0:t_steps-mod + 2,1], 120)
# plt.plot(xspline, yspline, color=[0.85, 0.35, 0.35], alpha=0.5)
# xspline, yspline = interpolate(positions['adversary_1'][0:t_steps-mod + 2,0], positions['adversary_1'][0:t_steps-mod + 2,1], 120)
# plt.plot(xspline, yspline, color=[0.85, 0.35, 0.35], alpha=0.5)
# xspline, yspline = interpolate(positions['adversary_2'][0:t_steps-mod + 2,0], positions['adversary_2'][0:t_steps-mod + 2,1], 120)
# plt.plot(xspline, yspline, color=[0.85, 0.35, 0.35], alpha=0.5)

# opacities = np.linspace(0, 1, num=t_steps)
# # add agent and adversaries
# for i in range(t_steps):
# 	if i % mod == 0:
# 		plt.gca().add_patch(plt.Circle((positions['agent_0'][i,:]),      0.05, fc=[0.35, 0.85, 0.35], alpha=opacities[i]) )
# 		plt.gca().add_patch(plt.Circle((positions['adversary_0'][i,:]), 0.075, fc=[0.85, 0.35, 0.35], alpha=opacities[i]) )
# 		plt.gca().add_patch(plt.Circle((positions['adversary_1'][i,:]), 0.075, fc=[0.85, 0.35, 0.35], alpha=opacities[i]) )
# 		plt.gca().add_patch(plt.Circle((positions['adversary_2'][i,:]), 0.075, fc=[0.85, 0.35, 0.35], alpha=opacities[i]) )


# plt.axis('scaled')
# plt.show()
# fig.savefig("ghost_traj_actx.png")



#  ### For 3 Trajectory ###
# plt.rcParams.update({'font.size': 16})
# (fig, ax0) = plt.subplots(figsize=(10, 8), tight_layout=True)
# t_steps = min(positions0['agent_0'].shape[0], positions0['adversary_0'].shape[0], positions0['adversary_1'].shape[0], positions0['adversary_2'].shape[0])

# # add landmarks
# for i in range(positions0['obstacles'].shape[0]):
# 	plt.gca().add_patch(plt.Circle((positions0['obstacles'][i,:]), 0.2, fc=[0.25, 0.25, 0.25]))

# mod = 4
# # add smooth trajectories
# # actor0 policy
# xspline, yspline = interpolate(positions0['agent_0'][0:t_steps-mod + 2,0], positions0['agent_0'][0:t_steps-mod + 2,1], 120)
# plt.plot(xspline, yspline, color=[0.35, 0.85, 0.35], alpha=0.5)
# xspline, yspline = interpolate(positions0['adversary_0'][0:t_steps-mod + 2,0], positions0['adversary_0'][0:t_steps-mod + 2,1], 120)
# plt.plot(xspline, yspline, color=[0.85, 0.35, 0.35], alpha=0.5)
# xspline, yspline = interpolate(positions0['adversary_1'][0:t_steps-mod + 2,0], positions0['adversary_1'][0:t_steps-mod + 2,1], 120)
# plt.plot(xspline, yspline, color=[0.85, 0.35, 0.35], alpha=0.5)
# xspline, yspline = interpolate(positions0['adversary_2'][0:t_steps-mod + 2,0], positions0['adversary_2'][0:t_steps-mod + 2,1], 120)
# plt.plot(xspline, yspline, color=[0.85, 0.35, 0.35], alpha=0.5)

# # actor1 policy
# xspline, yspline = interpolate(positions1['agent_0'][0:t_steps-mod + 2,0], positions1['agent_0'][0:t_steps-mod + 2,1], 120)
# plt.plot(xspline, yspline, color='yellowgreen', alpha=0.5)
# xspline, yspline = interpolate(positions1['adversary_0'][0:t_steps-mod + 2,0], positions1['adversary_0'][0:t_steps-mod + 2,1], 120)
# plt.plot(xspline, yspline, color='indianred', alpha=0.5)
# xspline, yspline = interpolate(positions1['adversary_1'][0:t_steps-mod + 2,0], positions1['adversary_1'][0:t_steps-mod + 2,1], 120)
# plt.plot(xspline, yspline, color='indianred', alpha=0.5)
# xspline, yspline = interpolate(positions1['adversary_2'][0:t_steps-mod + 2,0], positions1['adversary_2'][0:t_steps-mod + 2,1], 120)
# plt.plot(xspline, yspline, color='indianred', alpha=0.5)

# # actor2 policy
# xspline, yspline = interpolate(positions2['agent_0'][0:t_steps-mod + 2,0], positions2['agent_0'][0:t_steps-mod + 2,1], 120)
# plt.plot(xspline, yspline, color='darkgreen', alpha=0.5)
# xspline, yspline = interpolate(positions2['adversary_0'][0:t_steps-mod + 2,0], positions2['adversary_0'][0:t_steps-mod + 2,1], 120)
# plt.plot(xspline, yspline, color='firebrick', alpha=0.5)
# xspline, yspline = interpolate(positions2['adversary_1'][0:t_steps-mod + 2,0], positions2['adversary_1'][0:t_steps-mod + 2,1], 120)
# plt.plot(xspline, yspline, color='firebrick', alpha=0.5)
# xspline, yspline = interpolate(positions2['adversary_2'][0:t_steps-mod + 2,0], positions2['adversary_2'][0:t_steps-mod + 2,1], 120)
# plt.plot(xspline, yspline, color='firebrick', alpha=0.5)


# plt.axis('scaled')
# plt.show()
# # fig.savefig("ghost_traj_multi.svg")
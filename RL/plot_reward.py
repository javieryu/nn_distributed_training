import numpy as np
import matplotlib.pyplot as plt

# Load Centralized
times_cent = np.load('trained/timesteps_2.npy')
rewards1 = np.load('trained/avg_ep_rews_1.npy')
rewards2 = np.load('trained/avg_ep_rews_2.npy')
rewards3 = np.load('trained/avg_ep_rews_3.npy')
rewards4 = np.load('trained/avg_ep_rews_4.npy')
rewards_arr_cent = np.vstack((rewards1, rewards2, rewards3, rewards4))

# Load CADMM
times_cadmm = np.load('dist_rl/trained/timesteps_1.npy')
rewards1 = np.load('dist_rl/trained/avg_ep_rews_1.npy')
rewards2 = np.load('dist_rl/trained/avg_ep_rews_2.npy')
rewards3 = np.load('dist_rl/trained/avg_ep_rews_3.npy')
rewards4 = np.load('dist_rl/trained/avg_ep_rews_4.npy')

rewards_arr_cadmm = np.vstack((rewards1, rewards2, rewards3, rewards4))


fig, ax0 = plt.subplots(figsize=(20, 8), tight_layout = True)

# Centralized
ax0.plot(times_cent, np.mean(rewards_arr_cent, axis=0), c="indianred", label="Centralized")
ax0.fill_between(times_cent, np.amax(rewards_arr_cent, axis=0), np.amin(rewards_arr_cent, axis=0), color="indianred", alpha=0.5)

# CADMM
ax0.plot(times_cadmm, np.mean(rewards_arr_cadmm, axis=0), c="blue", label="CADMM")
ax0.fill_between(times_cadmm, np.amax(rewards_arr_cadmm, axis=0), np.amin(rewards_arr_cadmm, axis=0), color="blue", alpha=0.5)

ax0.set_title("RL Training Over Time")
ax0.set_xlabel("Timestep")
ax0.set_ylabel("Average Episode Reward")
ax0.legend()
plt.show()



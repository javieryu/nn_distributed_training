import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

# Load Centralized
times_cent = np.load('trained/timesteps_3.npy')
rewards3 = np.load('trained/avg_ep_rews_3.npy')
rewards4 = np.load('trained/avg_ep_rews_4.npy')

rewards_arr_cent = np.vstack((rewards3, rewards4))


# Load CADMM
# times_cadmm = np.load('dist_rl/trained/timesteps_1.npy')
# rewards1 = np.load('dist_rl/trained/avg_ep_rews_1.npy')
# rewards2 = np.load('dist_rl/trained/avg_ep_rews_2.npy')
# rewards3 = np.load('dist_rl/trained/avg_ep_rews_3.npy')
# rewards4 = np.load('dist_rl/trained/avg_ep_rews_4.npy')
# rewards_arr_cadmm = np.vstack((rewards1, rewards2, rewards3, rewards4))

times_long_cadmm = np.load('dist_rl/trained/timesteps_long.npy')
rewards_long_cadmm = np.load('dist_rl/trained/avg_ep_rews_long.npy')

# Load DSGT


# Load DSGD


# Construct Plot
(fig, ax0) = plt.subplots(figsize=(10, 8), tight_layout=True)

cadmm_color="darkorange"
dsgt_color="limegreen"
dsgd_color="purple"
cent_color="indigo"
solo_color="cornflowerblue"

# Centralized
ax0.plot(times_cent, np.mean(rewards_arr_cent, axis=0), c=cent_color, label="Centralized")
ax0.fill_between(times_cent, np.amax(rewards_arr_cent, axis=0), np.amin(rewards_arr_cent, axis=0), color=cent_color, alpha=0.5)

# CADMM
# ax0.plot(times_cadmm, np.mean(rewards_arr_cadmm, axis=0), c=cadmm_color, label="CADMM")
# ax0.fill_between(times_cadmm, np.amax(rewards_arr_cadmm, axis=0), np.amin(rewards_arr_cadmm, axis=0), color=cadmm_color, alpha=0.5)

# Long CADMM
ax0.plot(times_long_cadmm, rewards_long_cadmm, c=cadmm_color, label="Long CADMM")

# DSGT

# DSGD


ax0.legend()
ax0.set_xlabel("Timestep")
ax0.set_ylabel("Average Episode Reward")
ax0.grid(zorder=0)
plt.show()

fig.savefig("RL_reward.svg")


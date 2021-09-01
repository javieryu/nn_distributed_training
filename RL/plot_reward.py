import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")


times1 = np.load('trained/timesteps_1.npy')
rewards1 = np.load('trained/avg_ep_rews_1.npy')

times2 = np.load('trained/timesteps_2.npy')
rewards2 = np.load('trained/avg_ep_rews_2.npy')

times3 = np.load('trained/timesteps_3.npy')
rewards3 = np.load('trained/avg_ep_rews_3.npy')

times4 = np.load('trained/timesteps_4.npy')
rewards4 = np.load('trained/avg_ep_rews_4.npy')

rewards_arr = np.vstack((rewards1, rewards2, rewards3, rewards4))




fig, ax0 = plt.subplots(figsize=(20, 8))
ax0.plot(times4, np.mean(rewards_arr, axis=0), c="indianred")
ax0.fill_between(times4, np.amax(rewards_arr, axis=0), np.amin(rewards_arr, axis=0), color="indianred", alpha=0.5)
ax0.set_title("RL Training Over Time")
ax0.set_xlabel("Timestep")
ax0.set_ylabel("Average Episode Reward")

plt.show()



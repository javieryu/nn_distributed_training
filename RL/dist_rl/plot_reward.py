import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")


times = np.load('trained/timesteps.npy')
rewards = np.load('trained/avg_ep_rews.npy')


fig, ax0 = plt.subplots(figsize=(20, 8))
ax0.plot(times, rewards, c="indianred")
ax0.set_title("RL Training Over Time")
ax0.set_xlabel("Timestep")
ax0.set_ylabel("Average Episode Reward")

plt.show()



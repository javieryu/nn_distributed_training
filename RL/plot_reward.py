import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")

times = np.load('trained/timesteps_1.npy')
rewards = np.load('trained/avg_ep_rews_1.npy')

fig, ax0 = plt.subplots(figsize=(20, 8))
ax0.plot(times, rewards, c="indianred")
# ax0.fill_between(np.arange(vlb.shape[0]), torch.amax(vlb, dim=1), torch.amin(vlb, dim=1), color="indianred", alpha=0.5)
ax0.set_title("RL Training Over Time")
ax0.set_xlabel("Timestep")
ax0.set_ylabel("Average Episode Reward")

plt.show()



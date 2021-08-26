import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")

avg_ep_rews = np.load('avg_ep_rews.npy')
timesteps = np.load('timesteps.npy')

fig, ax0 = plt.subplots(figsize=(20, 8))

ax0.plot(timesteps, avg_ep_rews, c="indianred")
# ax0.fill_between(np.arange(vlb.shape[0]), torch.amax(vlb, dim=1), torch.amin(vlb, dim=1), color="indianred", alpha=0.5)
ax0.set_title("RL Training Over Time")
ax0.set_xlabel("Timestep")
ax0.set_ylabel("Average Episode Reward")

plt.show()
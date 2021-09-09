import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

# Load CADMM
times_cadmm = np.load('dist_rl/trained/timesteps_cadmm_40.npy')
agree_cadmm = np.load('dist_rl/trained/agreements_cadmm_40.npz')
l_cadmm = int(3.*times_cadmm.shape[0] / 4.0)

# Load DSGT
times_dsgt = np.load('dist_rl/trained/timesteps_dsgt_7.npy')
agree_dsgt = np.load('dist_rl/trained/agreements_dsgt_7.npz')

# Load DSGD
times_dsgd = np.load('dist_rl/trained/timesteps_dsgd_7.npy')
agree_dsgd = np.load('dist_rl/trained/agreements_dsgd_7.npz')

# Construct Plot
(fig, ax0) = plt.subplots(figsize=(10, 8), tight_layout=True)

cadmm_color="darkorange"
dsgt_color="limegreen"
dsgd_color="purple"
cent_color="indigo"
solo_color="cornflowerblue"

# CADMM
ax0.plot(times_cadmm[0:l_cadmm], agree_cadmm['agree_0'][0:l_cadmm], c=cadmm_color, label="DiNNO")
ax0.plot(times_cadmm[0:l_cadmm], agree_cadmm['agree_1'][0:l_cadmm], c=cadmm_color)
ax0.plot(times_cadmm[0:l_cadmm], agree_cadmm['agree_2'][0:l_cadmm], c=cadmm_color)

# DSGT
ax0.plot(times_dsgt, agree_dsgt['agree_0'], c=dsgt_color, label="DSGT")
ax0.plot(times_dsgt, agree_dsgt['agree_1'], c=dsgt_color)
ax0.plot(times_dsgt, agree_dsgt['agree_2'], c=dsgt_color)

# DSGD
ax0.plot(times_dsgd, agree_dsgd['agree_0'], c=dsgd_color, label="DSGD")
ax0.plot(times_dsgd, agree_dsgd['agree_1'], c=dsgd_color)
ax0.plot(times_dsgd, agree_dsgd['agree_2'], c=dsgd_color)

ax0.legend()
ax0.set_xlabel("Timestep")
ax0.set_ylabel("Distance to Mean Parameter Value")
ax0.grid(zorder=0)
plt.show()

fig.savefig("RL_agreement.svg")


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

# Load CADMM
times_cadmm = np.load('dist_rl/trained/timesteps_cadmm_7.npy')
agree_cadmm = np.load('dist_rl/trained/agreements_cadmm_7.npz')

# Load DSGT
times_dsgt = np.load('dist_rl/trained/timesteps_dsgt_7.npy')
agree_dsgt = np.load('dist_rl/trained/agreements_dsgt_7.npz')

# Load DSGD
times_dsgd = np.load('dist_rl/trained/timesteps_dsgd_7.npy')
agree_dsgd = np.load('dist_rl/trained/agreements_dsgd_7.npz')

# Load Joe
times_joe = np.load('dist_rl/trained/timesteps_cadmm_103.npy')
agree_103 = np.load('dist_rl/trained/agreements_cadmm_103.npz')
agree_104 = np.load('dist_rl/trained/agreements_cadmm_104.npz')


# Construct Plot
(fig, ax0) = plt.subplots(figsize=(10, 8), tight_layout=True)

cadmm_color="darkorange"
dsgt_color="limegreen"
dsgd_color="purple"
cent_color="indigo"
solo_color="cornflowerblue"

# CADMM
ax0.plot(times_cadmm, agree_cadmm['agree_0'], c=cadmm_color, label="DDL-ADMM")
ax0.plot(times_cadmm, agree_cadmm['agree_1'], c=cadmm_color)
ax0.plot(times_cadmm, agree_cadmm['agree_2'], c=cadmm_color)

# DSGT
ax0.plot(times_dsgt, agree_dsgt['agree_0'], c=dsgt_color, label="DSGT")
ax0.plot(times_dsgt, agree_dsgt['agree_1'], c=dsgt_color)
ax0.plot(times_dsgt, agree_dsgt['agree_2'], c=dsgt_color)

# DSGD
ax0.plot(times_dsgd, agree_dsgd['agree_0'], c=dsgd_color, label="DSGD")
ax0.plot(times_dsgd, agree_dsgd['agree_1'], c=dsgd_color)
ax0.plot(times_dsgd, agree_dsgd['agree_2'], c=dsgd_color)

# Joe
ax0.plot(times_joe, agree_103['agree_0'], c='blue', label="3 updates")
ax0.plot(times_joe, agree_103['agree_1'], c='blue')
ax0.plot(times_joe, agree_103['agree_2'], c='blue')

ax0.plot(times_joe, agree_104['agree_0'], c='green', label="4 updates")
ax0.plot(times_joe, agree_104['agree_1'], c='green')
ax0.plot(times_joe, agree_104['agree_2'], c='green')

ax0.legend()
ax0.set_xlabel("Timestep")
ax0.set_ylabel("Distance to Mean Parameter Value")
ax0.grid(zorder=0)
plt.show()

fig.savefig("RL_agreement.svg")


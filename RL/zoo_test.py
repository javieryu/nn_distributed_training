import numpy as np
from pettingzoo.mpe import simple_tag_v2

env = simple_tag_v2.env(num_good=1, num_adversaries=3, num_obstacles=8, \
                        max_cycles=200, continuous_actions=True)

print(env.observation_spaces["adversary_0"].shape[0])
print(env.action_spaces["adversary_0"].shape[0])


# actions = [None, Right, Left, Up, Down]
# observation = [self_vel, self_pos, landmark_rel_pos, other_adversary/agent_rel_pos, agent_rel_vel]

env.reset()
for agent in env.agent_iter():
    print("\nAgent: ")
    print(agent)
    
    obs, reward, done, info = env.last()
    if agent == "agent_0": # the prey
        print(obs)
        print(obs.size)
        act = np.random.rand(5) if not done else None
    else: # predators
        act = np.array([0.0, 0.0, 0.0, 0.2, 0.1]) if not done else None
    
    env.step(act)
    env.render()
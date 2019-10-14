#%%
from models.armed_bandit import ArmedBandit
import matplotlib.pyplot as plt
import numpy as np

true_means = np.random.uniform(-1, 1, 10)
bandit = ArmedBandit(10, 0, true_means)
accumulated_rewards = bandit.iterate(1000)
plt.plot(accumulated_rewards, label='e = 0')
bandit = ArmedBandit(10, 0.01, true_means)
accumulated_rewards = bandit.iterate(1000)
plt.plot(accumulated_rewards, label='e=0.01')

plt.show()


#%%

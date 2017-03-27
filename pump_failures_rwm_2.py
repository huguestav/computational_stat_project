from time import time
import numpy as np
from matplotlib import pyplot as plt
import pdb

import metropolis_hastings

print("Test RWM_2 script", end="\n")
initial_time = time()
# np.random.seed(1)

# Target density
from pump_failures_parameters import pi, D_mala


# Initial values for the adaptative parameters
gamma_0 = np.identity(11)
sigma_2_0 = 0.1
mu_0 = 0.1 * np.ones(11)

# Adaptation parameters
gamma = lambda x : 10 / (x + 1)
epsilon_1 = 1e-7
epsilon_2 = 1e-6
A_1 = 1e7

# Test rwm_2
n_runs = 1
n_steps = int(1.6 * 1e3)
# n_steps = int(50 * 1e3)
initial_value = 0.01 * np.ones(11)

mean_jumps = np.zeros(n_runs)
for i in range(n_runs):
    print("\n\tStart rwm_2 algorithm")
    values = metropolis_hastings.rwm_2(
        initial_value=initial_value,
        pi=pi,
        gamma=gamma,
        mu_0=mu_0,
        gamma_0=gamma_0,
        sigma_2_0=sigma_2_0,
        n_steps=n_steps,
        epsilon_1=epsilon_1,
        epsilon_2=epsilon_2,
        A_1=A_1,
    )
    acceptance_ratio = (values.shape[0] - 1) / n_steps
    print("\tAcceptance ratio :", round(acceptance_ratio, 5))

    # Handle jumps
    mean_jumps[i] = metropolis_hastings.mean_square_jump(values, n_steps)

print("means square jump : ", round(np.mean(mean_jumps), 5))

print("\nScript completed in %0.2f seconds" % (time() - initial_time))


# Plot
# plt.scatter(values[:,0], values[:,1], c='b', s=20)
# plt.show()

timeLimit = 100

plt.acorr(values[:,9], usevlines=True, normed=True, maxlags=timeLimit, lw=2, color = 'blue')
plt.axis([0, timeLimit, 0, 1])
plt.ylabel("Autocorrelation Function")
plt.xlabel("Time (s)")
plt.show()

from time import time
import numpy as np
from matplotlib import pyplot as plt
import pdb

import metropolis_hastings

print("Test MALA_2 script", end="\n")
initial_time = time()
np.random.seed(1)


# Target density
SIGMA = np.loadtxt("tmalaexcov.txt")
def pi(x):
    # mu = np.zeros(20)
    vec = x.reshape((len(x), 1))
    result = np.exp(- 0.5 * vec.T.dot(np.linalg.inv(SIGMA)).dot(vec))
    return float(result)

# Mala parameters
delta = 1000
def D_mala(x):
    sigma_inv = np.linalg.inv(SIGMA)
    grad_log_pi = - sigma_inv.dot(x)
    return delta * grad_log_pi / max(delta, np.linalg.norm(grad_log_pi, ord=2))

# Initial values for the adaptative parameters
gamma_0 = np.identity(20)
sigma_2_0 = 1
mu_0 = np.ones(20)

# Adaptation parameters
gamma = lambda x : 10 / (x + 1)
epsilon_1 = 1e-7
epsilon_2 = 1e-6
A_1 = 1e7

# Test rwm_1
n_runs = 1
n_steps = int(50 * 1e3)
initial_value = 5 * np.ones(20)

mean_values = np.zeros(n_runs)
mean_jumps = np.zeros(n_runs)
for i in range(n_runs):
    print("\n\tStart mala_2 algorithm")
    values, sigma_2_res = metropolis_hastings.mala_2(
        initial_value=initial_value,
        pi=pi,
        gamma=gamma,
        D_mala=D_mala,
        mu_0=mu_0,
        gamma_0=gamma_0,
        sigma_2_0=sigma_2_0,
        n_steps=n_steps,
        epsilon_1=epsilon_1,
        epsilon_2=epsilon_2,
        A_1=A_1,
    )
    print("\n\tsigma_2 :", round(sigma_2_res, 5))
    acceptance_ratio = (values.shape[0] - 1) / n_steps
    print("\tAcceptance ratio :", round(acceptance_ratio, 5))
    mean_values[i] = np.mean(values, axis=0)[0]
    print("\tvalue :", round(mean_values[i], 5))

    # Handle jumps
    mean_jumps[i] = metropolis_hastings.mean_square_jump(values, n_steps)

print("\nValues :", mean_values)
print("mean :", round(np.mean(mean_values), 5))
print("std :", round(np.std(mean_values), 5))
print("means square jump : ", round(np.mean(mean_jumps), 5))

print("\nScript completed in %0.2f seconds" % (time() - initial_time))


# Plot
plt.scatter(values[:,0], values[:,1], c='b', s=20)
plt.show()

timeLimit = 100

plt.acorr(values[:,0],usevlines=True, normed=True, maxlags=timeLimit, lw=2, color = 'blue')
plt.axis([0, timeLimit, 0, 1])
plt.ylabel("Autocorrelation Function")
plt.xlabel("Time (s)")
plt.show()

from time import time
import numpy as np
from matplotlib import pyplot as plt
import pdb

import metropolis_hastings

print("Test RWM_1 script", end="\n")
initial_time = time()

# Target density
mu = np.zeros(20)
SIGMA = np.loadtxt("tmalaexcov.txt")
def pi(x):
    vec = (x - mu).reshape((len(x), 1))
    result = np.exp(- 0.5 * vec.T.dot(np.linalg.inv(SIGMA)).dot(vec))
    return float(result)

# Initial values for the adaptative parameters
gamma_0 = SIGMA
sigma_2 = 0.59**2
# sigma_2 = 0.59

# Test rwm_opt
n_runs = 1
n_steps = int(50 * 1e3)
initial_value = 5 * np.ones(20)

mean_values = np.zeros(n_runs)
for i in range(n_runs):
    print("\n\tStart rwm_opt algorithm")
    values, mean_square_jump = metropolis_hastings.rwm_no_adapt(
        initial_value=initial_value,
        pi=pi,
        sigma_2=sigma_2,
        lbda=gamma_0,
        n_steps=n_steps,
    )
    acceptance_ratio = (values.shape[0] - 1) / n_steps
    print("\tAcceptance ratio :", acceptance_ratio)
    mean_values[i] = np.mean(values, axis=0)[0]
    print("\tvalue :", round(mean_values[i], 5))

print("\nValues :", mean_values)
print("mean :", np.mean(mean_values))
print("std :", np.std(mean_values))
print("means square jump : ", mean_square_jump)

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

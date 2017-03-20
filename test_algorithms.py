import numpy as np
from matplotlib import pyplot as plt

import metropolis_hastings


## Test rwm1
# Random walk metropolis parameters
initial_value = np.array([3, -5])
sigma_2 = 1
lbda = [[1,0], [0,1]]

# Target density
mu = np.array([7, 7])
sigma = np.array([[2,0], [0,2]])
pi = lambda x : np.exp(- 0.5 * (x - mu).T.dot(np.linalg.inv(sigma)).dot(x - mu))

n_steps = 2000
values = metropolis_hastings.rwm_1(initial_value, pi, sigma_2, lbda, n_steps)
acceptance_ratio = (values.shape[0] - 1) / n_steps
print("Acceptance ratio :", acceptance_ratio)
# print(values)


plt.scatter(values[:,0], values[:,1], c='b', s=20)
plt.show()



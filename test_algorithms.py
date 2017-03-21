import numpy as np
from matplotlib import pyplot as plt
import pdb

import metropolis_hastings

# Load tmalaexcov.txt
SIGMA = np.loadtxt("tmalaexcov.txt")


# Random walk metropolis parameters
initial_value = np.array([3, -5])
sigma_2 = 1
lbda = [[1,0], [0,1]]

# Target density
mu = np.array([7, 7])
sigma = np.array([[2,0], [0,2]])
def pi(x):
    vec = (x - mu).reshape((len(x), 1))
    result = np.exp(- 0.5 * vec.T.dot(np.linalg.inv(sigma)).dot(vec))
    return float(result)

gamma = lambda x : 10 / (x + 1)
delta = 1000
epsilon_1 = 1e-7
epsilon_2 = 1e-6
A_1 = 1e7

mu_0 = np.array([1, 1])
gamma_0 = np.array([[1,0], [0,1]])
sigma_2_0 = 1


def D_mala(x):
    sigma_inv = np.linalg.inv(sigma)
    grad_log_pi = - sigma_inv.dot(x - mu)
    return delta * grad_log_pi / max(delta, np.linalg.norm(grad_log_pi, ord=2))

n_steps = int(20 * 1e3)


# # Test rwm_1
# values = metropolis_hastings.rwm_1(initial_value, pi, sigma_2, lbda, n_steps)
# acceptance_ratio = (values.shape[0] - 1) / n_steps
# print("Acceptance ratio :", acceptance_ratio)
# print("mean :", np.mean(values, axis=0))
# # print(values)
# plt.scatter(values[:,0], values[:,1], c='b', s=20)
# plt.show()


# Test rwm_2
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
print("\nAcceptance ratio :", acceptance_ratio)
print("mean :", np.mean(values, axis=0))
plt.scatter(values[:,0], values[:,1], c='b', s=20)
plt.show()



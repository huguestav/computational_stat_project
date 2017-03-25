import numpy as np


p_values = np.array([5, 1, 5, 14, 3, 19, 1, 1, 4, 22])
t_values = np.array(
    [94.32, 15.72, 62.88, 125.76, 5.24, 31.44, 1.05, 1.05, 2.10, 10.48]
)


def pi(x):
    if np.sum(x < 0) > 0:
        return 0

    beta = x[-1]
    res_1 = beta**(17.01) * np.exp(- beta)
    res_2 = (
        np.power(x[:-1], p_values + 0.8)
        * np.exp(- x[:-1] * (t_values + beta))
    )

    res = res_1 * np.prod(res_2)

    return res


delta = 1000
def D_mala(x):
    if np.sum(x <= 0) > 0:
        return np.zeros(len(x))

    grad_log_pi = np.zeros(len(x))
    # Handle the last variable
    beta = x[-1]
    grad_log_pi[-1] = 17.01 / beta - 1
    # Handle the rest of the variables
    grad_log_pi[:-1] = (p_values + 0.8) / x[:-1] - t_values - beta

    return delta * grad_log_pi / max(delta, np.linalg.norm(grad_log_pi, ord=2))

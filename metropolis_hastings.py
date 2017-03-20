import numpy as np
import scipy.stats


def proposal_density(x, mean, covariance):
    return scipy.stats.multivariate_normal.pdf(x, mean=mean, cov=covariance)


def rwm_1(initial_value, pi, sigma_2, lbda, n_steps):
    values = np.zeros((n_steps +1, len(initial_value)))
    values[0] = initial_value
    values_idx = 0

    for i in range(n_steps):
        current_value = values[values_idx]
        mean = current_value
        covariance = sigma_2 * lbda

        # Compute the candidate value
        candidate_value = np.random.multivariate_normal(mean, covariance)

        # Compute the acceptation probability
        alpha = (
            pi(candidate_value)
            / pi(current_value)
            * proposal_density(
                  x=current_value,
                  mean=candidate_value,
                  covariance=covariance)
            / proposal_density(
                  x=candidate_value,
                  mean=mean,
                  covariance=covariance)
        )
        alpha = min(1, alpha)

        # Accept or reject the candidate value
        if np.random.rand() < alpha:
            values_idx += 1
            values[values_idx] = candidate_value

    return values[:values_idx+1]


def projection_1(x, epsilon_1, A_1):
    if epsilon_1 <= x <= A_1:
        return x
    if x < epsilon_1:
        return epsilon_1
    if x > A_1:
        return A_1


def is_pos_semi_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)


def projection_2(x, A_1):
    frobenius_norm = np.linalg.norm(x, ord="fro")

    assert is_pos_semi_def(x)

    if frobenius_norm <= A_1:
        return x
    else:
        return x * A_1 / frobenius_norm


def projection_3(x, A_1):
    euclidian_norm = np.linalg.norm(x, ord=2)

    if euclidian_norm <= A_1:
        return x
    else:
        return x * A_1 / euclidian_norm

## TODO : rmw_2

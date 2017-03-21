import numpy as np
import scipy.stats
import pdb


def proposal_density(x, mean, covariance):
    return scipy.stats.multivariate_normal.pdf(x, mean=mean, cov=covariance)


def rwm_1(initial_value, pi, sigma_2, lbda, n_steps):
    values = np.zeros((n_steps + 1, len(initial_value)))
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


def rwm_2(initial_value, pi, gamma, mu_0, gamma_0, sigma_2_0, n_steps,
          epsilon_1, epsilon_2, A_1):
    """
    pi and gamma are functions
    """
    values = np.zeros((n_steps + 1, len(initial_value)))
    values[0] = initial_value
    values_idx = 0

    # Initialize the adaptative parameters
    mu_n = mu_0
    gamma_n = gamma_0
    sigma_2_n = sigma_2_0

    for i in range(n_steps):
        current_value = values[values_idx]

        # Update lambda_n
        if i > 5000:
            lambda_n = gamma_n + epsilon_2 * np.identity(len(initial_value))
        else:
            # lambda_n = gamma_0
            lambda_n = gamma_0 + epsilon_2 * np.identity(len(initial_value))

        # Step 1 : sample a candidate value
        mean = current_value
        covariance = sigma_2_n * lambda_n
        candidate_value = np.random.multivariate_normal(mean=mean, cov=covariance, )

        # Step 2 : accept or reject
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
        if np.random.rand() < alpha:
            values_idx += 1
            values[values_idx] = candidate_value
            current_value = values[values_idx]

        # Step 3 : update the adaptative parameters (mu_n, gamma_n, sigma_2_n)
        mu_n_new = projection_3(
            x=mu_n + gamma(i) * (current_value - mu_n),
            A_1=A_1,
        )
        vec = (current_value - mu_n).reshape((len(current_value), 1))
        if i > 1000:
        # if i > 10:
            gamma_n_new = projection_2(
                x=gamma_n + gamma(i) * (vec.dot(vec.T) - gamma_n),
                A_1=A_1,
            )
        else:
            gamma_n_new = gamma_n
        sigma_2_n_new = projection_1(
            x=sigma_2_n + gamma(i) * (alpha - 0.2),
            epsilon_1=epsilon_1,
            A_1=A_1,
        )
        mu_n = mu_n_new
        sigma_2_n = sigma_2_n_new
        gamma_n = gamma_n_new

    print("Final mu :", mu_n)
    print("Final sigma_2 :", sigma_2_n)
    print("Final gamma :", gamma_n)
    return values[:values_idx+1]

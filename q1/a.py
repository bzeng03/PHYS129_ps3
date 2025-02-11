import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import det, inv

def gaussian_integral_mc_fixed(A, w, num_samples=10**5):
    """
    Monte Carlo approximation of the Gaussian integral:

    I = ∫ exp(-1/2 * v^T A v + v^T w) dv

    Parameters:
    A (ndarray): N x N symmetric positive-definite matrix.
    w (ndarray): N-dimensional vector.
    num_samples (int): Number of Monte Carlo samples.

    Returns:
    Tuple: (Numerical integral approximation, Analytical result)
    """
    N = A.shape[0]
    A_inv = inv(A)

    # Importance sampling distribution: Normal with mean A_inv @ w and covariance A_inv
    mean_proposal = A_inv @ w
    cov_proposal = A_inv  # Covariance is A^-1
    proposal_dist = multivariate_normal(mean=mean_proposal, cov=cov_proposal)

    # Sample from the proposal distribution
    samples = proposal_dist.rvs(size=num_samples)

    # Compute the function values at sampled points
    exponent_values = -0.5 * np.einsum('ij,ij->i', samples @ A, samples) + np.dot(samples, w)
    true_integrand_values = np.exp(exponent_values)

    # Compute proposal distribution density at sampled points
    proposal_pdf_values = proposal_dist.pdf(samples)

    # Monte Carlo estimate with importance sampling correction
    integral_estimate = np.mean(true_integrand_values / proposal_pdf_values)

    # Compute the analytical result
    analytical_result = np.sqrt((2 * np.pi) ** N * det(A_inv)) * np.exp(0.5 * np.dot(w.T, np.dot(A_inv, w)))

    return integral_estimate, analytical_result

# Example test case
A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]])
w = np.array([1, 2, 3])

num_result, analytic_result = gaussian_integral_mc_fixed(A, w)
print(f"Numerical Result (Monte Carlo Approximation): {num_result}")
print(f"Analytical Result: {analytic_result}")
print(f"Relative Error: {abs(num_result - analytic_result) / abs(analytic_result):.6f}")
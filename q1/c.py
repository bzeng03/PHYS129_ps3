import numpy as np
from a import gaussian_integral_mc_fixed  # Import function
from scipy.linalg import inv

def compute_moments(A, w, num_samples=10**5):
    """
    Computes the first and second moments numerically and analytically.
    """
    N = A.shape[0]
    A_inv = inv(A)
    mean_vec = A_inv @ w  # Mean of the Gaussian
    cov_matrix = A_inv  # Covariance matrix

    # Generate Monte Carlo samples
    samples = np.random.multivariate_normal(mean_vec, cov_matrix, size=num_samples)

    # Compute numerical moments
    v1, v2, v3 = samples[:, 0], samples[:, 1], samples[:, 2]
    numerical_moments = {
        "E[v1]": np.mean(v1),
        "E[v2]": np.mean(v2),
        "E[v3]": np.mean(v3),
        "E[v1 v2]": np.mean(v1 * v2),
        "E[v2 v3]": np.mean(v2 * v3),
        "E[v1 v3]": np.mean(v1 * v3),
        "E[v1^2 v2]": np.mean(v1**2 * v2),
        "E[v2 v3^2]": np.mean(v2 * v3**2),
        "E[v1^2 v2^2]": np.mean(v1**2 * v2**2),
        "E[v2^2 v3^2]": np.mean(v2**2 * v3**2),
    }

    # Compute analytical moments
    analytical_moments = {
        "E[v1]": mean_vec[0],
        "E[v2]": mean_vec[1],
        "E[v3]": mean_vec[2],
        "E[v1 v2]": A_inv[0, 1] + mean_vec[0] * mean_vec[1],
        "E[v2 v3]": A_inv[1, 2] + mean_vec[1] * mean_vec[2],
        "E[v1 v3]": A_inv[0, 2] + mean_vec[0] * mean_vec[2],
        "E[v1^2 v2]": None,  # Higher-order moments require additional formulas
        "E[v2 v3^2]": None,
        "E[v1^2 v2^2]": None,
        "E[v2^2 v3^2]": None,
    }

    return numerical_moments, analytical_moments

# Define the matrix and vector
A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]])
w = np.array([1, 2, 3])

# Compute moments
numerical_results, analytical_results = compute_moments(A, w)

# Write results to a text file
output_file = "results_c.txt"
with open(output_file, "w") as f:
    f.write("Moments of the Multivariate Normal Distribution\n\n")
    
    f.write("Numerical Moments:\n")
    for key, value in numerical_results.items():
        f.write(f"{key}: {value:.6f}\n")

    f.write("\nAnalytical Moments:\n")
    for key, value in analytical_results.items():
        if value is not None:
            f.write(f"{key}: {value:.6f}\n")
        else:
            f.write(f"{key}: (Not computed analytically)\n")

print(f"Results have been written to {output_file}")
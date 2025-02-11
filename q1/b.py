import numpy as np
from a import gaussian_integral_mc_fixed  # Import function from a.py

# Define the matrices and vector for part (b)
A1 = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]])
A2 = np.array([[4, 2, 1], [2, 1, 3], [1, 3, 6]])
w = np.array([1, 2, 3])

# Compute results using the imported function
num_result_A1, analytic_result_A1 = gaussian_integral_mc_fixed(A1, w)
# num_result_A2, analytic_result_A2 = gaussian_integral_mc_fixed(A2, w)

# Write the results to a text file
output_file = "results_b.txt"

with open(output_file, "w") as f:
    f.write("Results for A1:\n")
    f.write(f"Numerical Result: {num_result_A1}\n")
    f.write(f"Analytical Result: {analytic_result_A1}\n")
    f.write(f"Relative Error: {abs(num_result_A1 - analytic_result_A1) / abs(analytic_result_A1):.6f}\n\n")
    '''
    f.write("Results for A2:\n")
    f.write(f"Numerical Result: {num_result_A2}\n")
    f.write(f"Analytical Result: {analytic_result_A2}\n")
    f.write(f"Relative Error: {abs(num_result_A2 - analytic_result_A2) / abs(analytic_result_A2):.6f}\n")
    '''

print(f"Results have been written to {output_file}")
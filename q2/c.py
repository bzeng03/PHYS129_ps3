import os
import numpy as np
import matplotlib.pyplot as plt

# Define the base directory containing LDOS data files
base_directory = "Local_density_of_states_near_band_edge"

# Get all text files in the directory
data_files = sorted([f for f in os.listdir(base_directory) if f.endswith(".txt")])

# Define output paths
plot_output = "average_ldos_vs_index.png"
log_output = "results_2c.txt"

# Randomly select a sub-region
np.random.seed(42)  # For reproducibility
subregion_start_x, subregion_start_y = np.random.randint(0, 10, size=2)  # Random start position
subregion_width, subregion_height = 5, 5  # Fixed small region

# Store LDOS averages per file
ldos_averages = []

# Process each file
for index, file_name in enumerate(data_files):
    file_path = os.path.join(base_directory, file_name)

    try:
        # Load data
        data = np.loadtxt(file_path, delimiter=",")

        # Ensure sub-region does not exceed data bounds
        max_x, max_y = data.shape[1], data.shape[0]
        end_x = min(subregion_start_x + subregion_width, max_x)
        end_y = min(subregion_start_y + subregion_height, max_y)

        # Extract sub-region and compute average LDOS
        subregion = data[subregion_start_y:end_y, subregion_start_x:end_x]
        avg_ldos = np.mean(subregion)
        ldos_averages.append(avg_ldos)

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Generate the plot of average LDOS vs file index
plt.figure(figsize=(8, 6))
plt.plot(range(len(ldos_averages)), ldos_averages, marker='o', linestyle='-')
plt.xlabel("File Index")
plt.ylabel("Average Local Density of States")
plt.title("Average LDOS in Selected Subregion vs File Index")
plt.grid(True)

# Save the plot
plt.savefig(plot_output)
plt.close()

# Save the results in a text file
with open(log_output, "w") as log_file:
    log_file.write(f"Selected Subregion: Start=({subregion_start_x}, {subregion_start_y}), "
                   f"Width={subregion_width}, Height={subregion_height}\n\n")
    log_file.write("File Index | Average LDOS\n")
    log_file.write("----------------------------\n")
    for i, avg in enumerate(ldos_averages):
        log_file.write(f"{i} | {avg:.6f}\n")

print(f"Plot saved as: {plot_output}")
print(f"Log saved as: {log_output}")
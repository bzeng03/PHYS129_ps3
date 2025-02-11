import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the base directory and output directory
base_directory = "Local_density_of_states_near_band_edge"
output_directory = os.path.join(base_directory, "local_density_of_states_height")

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Initialize a log file to record processed files
output_log = "results_2b.txt"

# Process all text files in the directory
processed_files = []
for file_name in os.listdir(base_directory):
    if file_name.endswith(".txt"):
        file_path = os.path.join(base_directory, file_name)
        output_path = os.path.join(output_directory, f"{file_name}.png")

        try:
            # Load data
            data = np.loadtxt(file_path, delimiter=",")
            X = np.arange(data.shape[1])
            Y = np.arange(data.shape[0])
            X, Y = np.meshgrid(X, Y)

            # Generate 3D surface plot
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, Y, data, cmap="viridis")

            # Labels
            ax.set_title(f"LDOS Surface Plot - {file_name}")
            ax.set_xlabel("X Index")
            ax.set_ylabel("Y Index")
            ax.set_zlabel("Local Density of States")

            # Save the plot
            plt.savefig(output_path)
            plt.close()

            processed_files.append(file_name)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")


print(f"All surface plots saved in: {output_directory}")
print(f"Processing log saved in: {output_log}")
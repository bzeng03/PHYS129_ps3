import os
import numpy as np
import matplotlib.pyplot as plt

# Define the base directory and output directory
base_directory = "Local_density_of_states_near_band_edge"
output_directory = os.path.join(base_directory, "local_density_of_states_heatmap")

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Initialize a log file to record processed files
output_log = "results_2a.txt"

# Process all text files in the directory
processed_files = []
for file_name in os.listdir(base_directory):
    if file_name.endswith(".txt"):
        file_path = os.path.join(base_directory, file_name)
        output_path = os.path.join(output_directory, f"{file_name}.png")

        try:
            # Load data
            data = np.loadtxt(file_path, delimiter=",")

            # Generate heatmap
            plt.figure(figsize=(8, 6))
            plt.imshow(data, cmap="hot", aspect="auto", origin="lower")
            plt.colorbar(label="Local Density of States")
            plt.title(f"LDOS Heatmap - {file_name}")
            plt.xlabel("X Index")
            plt.ylabel("Y Index")

            # Save the heatmap
            plt.savefig(output_path)
            plt.close()

            processed_files.append(file_name)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")



print(f"All heatmaps saved in: {output_directory}")
print(f"Processing log saved in: {output_log}")
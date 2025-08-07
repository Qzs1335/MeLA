import numpy as np
import os

num_sn = 200

print("Generating SN positions in a smaller 40x40 area to ensure solvability...")
sn_positions = np.random.rand(num_sn, 2) * 40

sn_positions = sn_positions + 5


root_dir = os.getcwd()


file_path_npy = os.path.join(root_dir, 'sn_pos.npy')


np.save(file_path_npy, sn_positions)

print(f"SN position data has been successfully saved to: {file_path_npy}")
print(f"SN coordinate range: X from {sn_positions[:, 0].min():.2f} to {sn_positions[:, 0].max():.2f}, Y from {sn_positions[:, 1].min():.2f} to {sn_positions[:, 1].max():.2f}")
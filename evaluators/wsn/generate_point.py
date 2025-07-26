import numpy as np
import os

# 定义SN的数量
num_sn = 200

# --- 修改点 ---
# 将SN的生成区域从50x50缩小到40x40
# 这使得SNs更加集中，从而大大降低了覆盖和连通的难度。
# CNs仍然可以在50x50的范围内自由移动，为寻找最优位置提供了更大的灵活性。
print("Generating SN positions in a smaller 40x40 area to ensure solvability...")
sn_positions = np.random.rand(num_sn, 2) * 40

# 将SN的位置平移，使其在(5,5)到(45,45)的区域内，位于整个50x50区域的中心
sn_positions = sn_positions + 5

# 获取当前工作目录
root_dir = os.getcwd()

# 准备保存的文件路径
file_path_npy = os.path.join(root_dir, 'sn_pos.npy')

# 保存为 .npy 文件
np.save(file_path_npy, sn_positions)

print(f"SN position data has been successfully saved to: {file_path_npy}")
print(f"SN coordinate range: X from {sn_positions[:, 0].min():.2f} to {sn_positions[:, 0].max():.2f}, Y from {sn_positions[:, 1].min():.2f} to {sn_positions[:, 1].max():.2f}")

# 如何加载 .npy 文件 (示例)
# loaded_sn = np.load(file_path_npy)
# print("Sample of loaded data:")
# print(loaded_sn[:5])
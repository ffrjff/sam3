import numpy as np
from PIL import Image

def save_depth_as_png(npy_path, png_path, normalize=True):
    """
    将 depth.npy 可视化并保存为 depth.png
    
    参数:
        npy_path: str, 输入的 depth.npy 路径
        png_path: str, 输出的 PNG 文件路径
        normalize: bool, 是否对深度值归一化到 0~255
    """
    # 读取深度数据
    depth = np.load(npy_path)  # shape: [H, W] 或 [H, W, 1]
    print(depth.shape)
    if len(depth.shape) == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    
    if normalize:
        # 将 depth 映射到 0~255
        min_val = np.min(depth)
        max_val = np.max(depth)
        if max_val - min_val > 1e-6:
            depth_norm = (depth - min_val) / (max_val - min_val)
        else:
            depth_norm = depth * 0  # 全零
        depth_img = (depth_norm * 255).astype(np.uint8)
    else:
        # 如果不归一化，直接转换为 16-bit PNG（保持原始数值）
        depth_img = depth.astype(np.uint16)
    
    # 保存为 PNG
    im = Image.fromarray(depth_img)
    im.save(png_path)
    print(f"Depth image saved to {png_path}")

# 示例使用
save_depth_as_png("assets/record/env0_0000_depth.npy", "depth.png")
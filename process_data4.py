import os
import numpy as np
from PIL import Image

# 掩码目录
mask_dirs = ["./ISIC_DATA_seg/train/masks", "./ISIC_DATA_seg/test/masks"]

def count_pixel_values(mask_dir):
    total_zeros = 0
    total_ones = 0
    total_pixels = 0

    print(f"统计目录: {mask_dir}")
    for mask_file in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = np.array(Image.open(mask_path).convert("L"))  # 转为灰度图

        # 统计 0 和 255 的像素数量
        zeros = np.sum(mask == 0)
        ones = np.sum(mask == 255)
        total_zeros += zeros
        total_ones += ones
        total_pixels += mask.size

        # print(f"{mask_file}: 0 的数量={zeros}, 255 的数量={ones}")

    print(f"目录 {mask_dir} 的统计结果:")
    print(f"总像素数: {total_pixels}")
    print(f"0 的总数量: {total_zeros}")
    print(f"255 的总数量: {total_ones}")
    print(f"前景比例 (255): {total_ones / total_pixels:.4f}")
    print(f"背景比例 (0): {total_zeros / total_pixels:.4f}")
    print("-" * 50)

# 对每个掩码目录进行统计
for mask_dir in mask_dirs:
    count_pixel_values(mask_dir)
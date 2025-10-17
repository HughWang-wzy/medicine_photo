import os
import shutil

# 定义路径
train_image_dir = "./ISIC_DATA_seg/train/images"
train_mask_dir = "./ISIC_DATA_seg/train/masks"
test_image_dir = "./ISIC_DATA_seg/test/images"
test_mask_dir = "./ISIC_DATA_seg/test/masks"

def remove_duplicates(train_dir, test_dir):
    # 获取文件名（去掉扩展名）
    train_files = set(os.path.splitext(f)[0] for f in os.listdir(train_dir))
    test_files = set(os.path.splitext(f)[0] for f in os.listdir(test_dir))

    # 找到重复的文件
    duplicates = train_files.intersection(test_files)
    print(f"发现 {len(duplicates)} 个重复文件：")

    # 删除 train 中的重复文件
    for duplicate in duplicates:
        train_file_path = os.path.join(train_dir, duplicate + ".jpg")  # 假设扩展名为 .jpg
        if os.path.exists(train_file_path):
            os.remove(train_file_path)
            print(f"已删除 {train_file_path}")

# 去除 train/images 和 test/images 中的重复文件
remove_duplicates(train_image_dir, test_image_dir)

# 去除 train/masks 和 test/masks 中的重复文件
remove_duplicates(train_mask_dir, test_mask_dir)
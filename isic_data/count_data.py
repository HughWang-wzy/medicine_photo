import os

# 定义输出目录
output_dir = "./ISIC_DATA"

# 检查每个类别的图像数量
def count_images_in_yolo_format(output_dir):
    for split in ["train", "test"]:
        split_dir = os.path.join(output_dir, split)
        print(f"\nSplit: {split}")
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                num_images = len([f for f in os.listdir(class_dir) if f.endswith(".jpg")])
                print(f"Class '{class_name}': {num_images} images")

# 调用函数
count_images_in_yolo_format(output_dir)
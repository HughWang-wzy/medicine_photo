from datasets import load_dataset,load_from_disk
from PIL import Image

# # 加载数据集
# ds = load_dataset("ahishamm/isic_binary_sharpened")

# # 保存到本地
# ds.save_to_disk("./isic_binary_sharpened")


# 加载本地数据集
ds = load_from_disk("./isic_binary_sharpened")

# 查看数据集的分割
print(ds)

import os

# 定义输出目录
output_dir = "./ISIC_DATA"
os.makedirs(output_dir, exist_ok=True)

# 获取类别名称
class_names = ds["train"].features["label"].names
for class_name in class_names:
    os.makedirs(os.path.join(output_dir, "train", class_name), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", class_name), exist_ok=True)


# 保存图像到 YOLO 格式
def save_images_to_yolo(split, dataset):
    for i, example in enumerate(dataset):
        # 获取图像和标签
        image = example["image"]
        label = example["label"]
        class_name = class_names[label]

        # 定义保存路径
        save_dir = os.path.join(output_dir, split, class_name)
        save_path = os.path.join(save_dir, f"{i}.jpg")

        # 保存图像
        image.save(save_path)

# 处理训练集和测试集
save_images_to_yolo("train", ds["train"])
save_images_to_yolo("test", ds["test"])
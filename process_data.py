import os
import shutil
import pandas as pd

# 定义输入路径和输出路径
csv_path = '/home/hugh/newdisk/medicine_photo/医学图像处理/2025《医学图像处理》课程实验数据集/黑色素瘤分类数据集/test/label.csv'
image_dir = '/home/hugh/newdisk/medicine_photo/医学图像处理/2025《医学图像处理》课程实验数据集/黑色素瘤分类数据集/test/image'
output_dir = '/home/hugh/newdisk/medicine_photo/医学图像处理/2025《医学图像处理》课程实验数据集/黑色素瘤分类数据集/test/sorted_images'

# 读取原始 CSV 文件
df = pd.read_csv(csv_path, header=None, names=['image_id', 'label'])

# 创建标签映射
label_map = {'benign': 'benign', 'malignant': 'malignant'}

# 遍历数据并将图像移动到对应的文件夹
for _, row in df.iterrows():
    image_id = row['image_id']
    label = row['label']
    label_folder = label_map[label]  # 获取对应的标签文件夹名

    # 定义源图像路径和目标文件夹路径
    src_image_path = os.path.join(image_dir, f"{image_id}.png")
    dest_folder = os.path.join(output_dir, label_folder)

    # 创建目标文件夹（如果不存在）
    os.makedirs(dest_folder, exist_ok=True)

    # 移动图像到目标文件夹
    dest_image_path = os.path.join(dest_folder, f"{image_id}.png")
    shutil.move(src_image_path, dest_image_path)

print(f"图像已分类并移动到 {output_dir}")
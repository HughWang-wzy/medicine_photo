import os
import zipfile
import shutil

# 定义数据集文件路径
data_dir = "/home/hugh/newdisk/medicine_photo"
zips = [
    "ISIC-2017_Training_Data.zip",
    "ISIC-2017_Training_Part1_GroundTruth.zip",
    "ISIC-2017_Test_v2_Data.zip",
    "ISIC-2017_Test_v2_Part1_GroundTruth.zip"
]

# 定义解压后的目标目录
output_dir = os.path.join(data_dir, "ISIC_DATA_seg")
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

# 创建目标目录
os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "masks"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "masks"), exist_ok=True)

# 解压所有 ZIP 文件
for zip_file in zips:
    zip_path = os.path.join(data_dir, zip_file)
    if os.path.exists(zip_path):
        print(f"解压 {zip_file}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    else:
        print(f"未找到文件: {zip_file}")

# 整合训练数据
train_data_dir = os.path.join(output_dir, "ISIC-2017_Training_Data")
train_gt_dir = os.path.join(output_dir, "ISIC-2017_Training_Part1_GroundTruth")
if os.path.exists(train_data_dir) and os.path.exists(train_gt_dir):
    print("整合训练数据...")
    for file in os.listdir(train_data_dir):
        src_path = os.path.join(train_data_dir, file)
        dst_path = os.path.join(train_dir, "images", file)
        shutil.move(src_path, dst_path)
    for file in os.listdir(train_gt_dir):
        src_path = os.path.join(train_gt_dir, file)
        dst_path = os.path.join(train_dir, "masks", file)
        shutil.move(src_path, dst_path)
else:
    print("训练数据或 GroundTruth 未找到！")

# 整合测试数据
test_data_dir = os.path.join(output_dir, "ISIC-2017_Test_v2_Data")
test_gt_dir = os.path.join(output_dir, "ISIC-2017_Test_v2_Part1_GroundTruth")
if os.path.exists(test_data_dir) and os.path.exists(test_gt_dir):
    print("整合测试数据...")
    for file in os.listdir(test_data_dir):
        src_path = os.path.join(test_data_dir, file)
        dst_path = os.path.join(test_dir, "images", file)
        shutil.move(src_path, dst_path)
    for file in os.listdir(test_gt_dir):
        src_path = os.path.join(test_gt_dir, file)
        dst_path = os.path.join(test_dir, "masks", file)
        shutil.move(src_path, dst_path)
else:
    print("测试数据或 GroundTruth 未找到！")

# 清理多余的文件夹
for folder in [train_data_dir, train_gt_dir, test_data_dir, test_gt_dir]:
    if os.path.exists(folder):
        shutil.rmtree(folder)

print("数据集整合完成！")
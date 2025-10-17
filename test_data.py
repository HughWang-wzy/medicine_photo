from PIL import Image
import os

data_dir = '/home/hugh/newdisk/medicine_photo/ISIC_DATA/test'  # 替换为你的数据集路径

for root, _, files in os.walk(data_dir):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            img = Image.open(file_path)
            img.verify()  # 验证图像是否有效
        except Exception as e:
            print(f"Invalid image file: {file_path} - {e}")
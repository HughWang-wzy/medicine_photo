from ultralytics import YOLO

# 加载最好的模型权重
model = YOLO('runs/classify/train6/weights/best.pt')  # 确保路径正确

# 定义测试集路径（分类文件夹的根目录）
test_data = '/home/hugh/newdisk/medicine_photo/医学图像处理/2025《医学图像处理》课程实验数据集/黑色素瘤分类数据集/test'

# 在测试集上进行评估
results = model.val(data=test_data, split='val')  # split='val' 表示验证集或测试集

# 打印结果
print("测试集评估结果：")
print(results)

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from tqdm import tqdm
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import time
from PIL import Image

# --- 1. 环境和数据加载 (与您原代码相同) ---

# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义图像尺寸和其他超参数
IMG_SIZE = 224
BATCH_SIZE = 64 # 可以适当调大，因为没有反向传播，显存占用较小

# 注意：为了特征提取，我们不再需要数据增强，所以训练集和测试集用相同的transform
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 替换为您的数据集路径
data_dir = '/home/hugh/newdisk/medicine_photo/ISIC_DATA'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms)
                  for x in ['train', 'test']}
# 注意：shuffle在特征提取时可以为False，以便标签和数据对应
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
               for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
print("Dataset Sizes:", dataset_sizes)
print("Class Names:", class_names)


# --- 2. 构建特征提取器 ---
model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
model.classifier = nn.Identity()

# 将模型移动到GPU并设置为评估模式
model = model.to(device)
model.eval()

print("Feature extractor model loaded successfully.")

# --- 3. 定义特征提取函数 ---

def extract_features(dataloader):
    all_features = []
    all_labels = []

    with torch.no_grad(): # 关闭梯度计算，节省内存和计算资源
        for inputs, labels in tqdm(dataloader, desc="Extracting Features"):
            inputs = inputs.to(device)
            
            # 获取模型输出的特征
            features = model(inputs)
            
            # 将特征和标签移动到CPU并转换为Numpy数组
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 将列表中的所有批次数据合并成一个大的Numpy数组
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_features, all_labels

# --- 4. 执行特征提取 ---

print("\n--- Starting Feature Extraction ---")
# 提取训练集特征
train_features, train_labels = extract_features(dataloaders['train'])
# 提取测试集特征
test_features, test_labels = extract_features(dataloaders['test'])

print("\nFeature extraction complete.")
print("Shape of training features:", train_features.shape)
print("Shape of training labels:", train_labels.shape)
print("Shape of testing features:", test_features.shape)
print("Shape of testing labels:", test_labels.shape)


# --- 5. 训练和评估传统机器学习模型 ---

# --- 模型A: 支持向量机 (Support Vector Machine) ---
print("\n--- Training Support Vector Machine (SVC) ---")
start_time = time.time()

# 初始化SVC分类器。C是正则化参数，kernel是核函数
# 'rbf' (径向基函数) 是一个常用的默认选项
svc_classifier = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)

# 训练模型
svc_classifier.fit(train_features, train_labels)

end_time = time.time()
print(f"SVC training finished in {end_time - start_time:.2f} seconds.")

# 在测试集上进行预测
print("\n--- Evaluating SVC ---")
test_predictions_svc = svc_classifier.predict(test_features)

# 打印评估报告
accuracy_svc = accuracy_score(test_labels, test_predictions_svc)
print(f"SVC Accuracy: {accuracy_svc * 100:.2f}%")
print("SVC Classification Report:")
print(classification_report(test_labels, test_predictions_svc, target_names=class_names))


# --- 模型B: 随机森林 (Random Forest) ---
print("\n--- Training Random Forest Classifier ---")
start_time = time.time()

# 初始化随机森林分类器
# n_estimators 是森林中树的数量
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 使用所有可用的CPU核心

# 训练模型
rf_classifier.fit(train_features, train_labels)

end_time = time.time()
print(f"Random Forest training finished in {end_time - start_time:.2f} seconds.")

# 在测试集上进行预测
print("\n--- Evaluating Random Forest ---")
test_predictions_rf = rf_classifier.predict(test_features)

# 打印评估报告
accuracy_rf = accuracy_score(test_labels, test_predictions_rf)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")
print("Random Forest Classification Report:")
print(classification_report(test_labels, test_predictions_rf, target_names=class_names))
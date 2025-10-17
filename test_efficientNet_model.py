import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision import models
import os
import matplotlib.pyplot as plt

# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义图像尺寸和数据增强
IMG_SIZE = 224
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载测试数据
data_dir = '/home/hugh/newdisk/medicine_photo/ISIC_DATA/test'
test_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# 加载 EfficientNet-B2 模型
model = models.efficientnet_b2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, 2)  # 替换分类头
model.load_state_dict(torch.load('./efficientnet_run/train5/best_model_joint_training.pth'))
model = model.to(device)
model.eval()

# 初始化变量
all_labels = []
all_preds = []
all_probs = []

# 测试模型
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # 获取正类的概率
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# 转换为 NumPy 数组
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# 计算混淆矩阵
tp = np.sum((all_preds == 1) & (all_labels == 1))
tn = np.sum((all_preds == 0) & (all_labels == 0))
fp = np.sum((all_preds == 1) & (all_labels == 0))
fn = np.sum((all_preds == 0) & (all_labels == 1))

# 计算指标
accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# 计算 ROC 和 AUC
fpr, tpr = [], []
thresholds = np.linspace(0, 1, 101)
for thresh in thresholds:
    preds_thresh = (all_probs >= thresh).astype(int)
    tp_thresh = np.sum((preds_thresh == 1) & (all_labels == 1))
    tn_thresh = np.sum((preds_thresh == 0) & (all_labels == 0))
    fp_thresh = np.sum((preds_thresh == 1) & (all_labels == 0))
    fn_thresh = np.sum((preds_thresh == 0) & (all_labels == 1))
    fpr.append(fp_thresh / (fp_thresh + tn_thresh) if (fp_thresh + tn_thresh) > 0 else 0)
    tpr.append(tp_thresh / (tp_thresh + fn_thresh) if (tp_thresh + fn_thresh) > 0 else 0)

fpr = np.array(fpr)
tpr = np.array(tpr)
# 确保 fpr 和 tpr 按 fpr 升序排列
sorted_indices = np.argsort(fpr)
fpr = fpr[sorted_indices]
tpr = tpr[sorted_indices]

# 计算 AUC
auc = np.trapz(tpr, fpr)

# 打印结果
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"AUC: {auc:.4f}")

# 可视化 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# 保存图片到指定路径
plt.savefig('./roc_curve.png')  # 保存为 PNG 格式
plt.close()  # 关闭图形以释放内存

'''
Accuracy: 0.9364
Sensitivity (Recall): 0.9344
Specificity: 0.9382
AUC: 0.9746
'''
import torch
import numpy as np
from u2net2 import U2NETP
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 映射掩码值到 [0, 1]
        mask = np.array(mask)
        # print(f"Unique mask values: {np.unique(mask)}")  # 调试信息
        # exit()
        mask = Image.fromarray(mask.astype(np.uint8))

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
test_dataset = SegmentationDataset(
    image_dir="./ISIC_DATA_seg/test/images",
    mask_dir="./ISIC_DATA_seg/test/masks",
    transform=image_transform,
    mask_transform=mask_transform
)
BATCH_SIZE = 4
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

def calculate_metrics(pred, target):
    """
    计算分割模型的 Dice、Accuracy、Sensitivity、Specificity。
    Args:
        pred (torch.Tensor): 模型预测的二值化分割结果 (H, W)。
        target (torch.Tensor): 真实标签 (H, W)。
    Returns:
        dict: 包含 Dice、Accuracy、Sensitivity、Specificity 的字典。
    """
    pred = pred.view(-1)  # 展平
    target = target.view(-1)  # 展平

    tp = torch.sum((pred == 1) & (target == 1)).item()  # True Positive
    tn = torch.sum((pred == 0) & (target == 0)).item()  # True Negative
    fp = torch.sum((pred == 1) & (target == 0)).item()  # False Positive
    fn = torch.sum((pred == 0) & (target == 1)).item()  # False Negative

    # 计算指标
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "Dice": dice,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }

def evaluate_model(model, test_loader):
    """
    在测试集上评估模型性能。
    Args:
        model (torch.nn.Module): 已训练的模型。
        test_loader (DataLoader): 测试集数据加载器。
    """
    model.eval()
    metrics_list = []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)

            # 模型推理
            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # 使用 Sigmoid 将输出映射到 [0, 1]
            predictions = (outputs > 0.5).float()  # 二值化
            # print(f"Outputs min: {outputs.min().item()}, max: {outputs.max().item()}")
            # print(f"Predictions unique values: {torch.unique(predictions)}")

            # 逐样本计算指标
            for i in range(images.size(0)):
                metrics = calculate_metrics(predictions[i, 0], masks[i, 0])
                metrics_list.append(metrics)

    # 计算平均指标
    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    print("Evaluation Metrics:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")

# 加载最佳模型权重
model = U2NETP(in_ch=3, out_ch=1).to(device)
best_model_path = "./u2net_runs/train19/best_model.pth"  # 修改为实际路径
model.load_state_dict(torch.load(best_model_path, map_location=device))

# 在测试集上评估模型
evaluate_model(model, test_loader)



'''17
Dice: 0.7734
Accuracy: 0.8858
Sensitivity: 0.9550
Specificity: 0.8634
'''

'''18
Dice: 0.7734
Accuracy: 0.8858
Sensitivity: 0.9550
Specificity: 0.8634
'''

'''19
Dice: 0.9118
Accuracy: 0.9606
Sensitivity: 0.9681
Specificity: 0.9367
'''
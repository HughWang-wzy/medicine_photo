import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from u2net2 import U2NETP

# --- 数据集定义 ---
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
        # print(f"Mask unique values: {np.unique(mask)}")  # 打印掩码的唯一值
        # print(f"Foreground ratio: {np.sum(mask) / mask.size:.4f}")  # 打印前景比例

        mask = Image.fromarray(mask.astype(np.uint8))

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

# --- 数据增强 ---
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# --- 数据加载 ---
train_dataset = SegmentationDataset(
    image_dir="./ISIC_DATA_seg/train/images",
    mask_dir="./ISIC_DATA_seg/train/masks",
    transform=image_transform,
    mask_transform=mask_transform
)

test_dataset = SegmentationDataset(
    image_dir="./ISIC_DATA_seg/test/images",
    mask_dir="./ISIC_DATA_seg/test/masks",
    transform=image_transform,
    mask_transform=mask_transform
)
BATCH_SIZE = 8
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# --- 模型定义 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = U2NETP(in_ch=3, out_ch=1).to(device)

# --- 加载预训练模型权重 ---
pretrained_model_path = "./u2net_runs/train17/best_model.pth"  # 修改为实际路径
if os.path.exists(pretrained_model_path):
    print(f"Loading pretrained model from {pretrained_model_path}")
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
else:
    print(f"Pretrained model not found at {pretrained_model_path}. Starting training from scratch.")
    
# --- 损失函数和优化器 ---
# 计算前景权重
foreground_weight = 1 / 0.1714  # 根据训练集前景比例调整
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([foreground_weight]).to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)

# --- 创建训练结果保存目录 ---
run_dir = "./u2net_runs"
os.makedirs(run_dir, exist_ok=True)

# 自动生成子目录
train_id = 0
while os.path.exists(os.path.join(run_dir, f"train{train_id}")):
    train_id += 1
train_dir = os.path.join(run_dir, f"train{train_id}")
os.makedirs(train_dir)

# 创建 CSV 文件
csv_path = os.path.join(train_dir, "training_log.csv")
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Phase", "Loss"])

# --- 训练函数 ---
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=20, patience=5):
    since = time.time()
    best_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = test_loader

            running_loss = 0.0

            with tqdm(total=len(dataloader), desc=f"{phase} Epoch {epoch+1}/{num_epochs}") as pbar:
                for images, masks in dataloader:
                    images, masks = images.to(device), masks.to(device)
                    unique_values = torch.unique(masks)
                    # print(f"Unique mask values: {unique_values.cpu().numpy()}")
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(images)
                        outputs = outputs[:, 0, :, :]
                        outputs = outputs.unsqueeze(1)  # 添加通道维度
                        loss = criterion(outputs, masks)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    pbar.update(1)

            epoch_loss = running_loss / len(dataloader.dataset)
            print(f"{phase} Loss: {epoch_loss:.4f}")

            # 保存训练日志到 CSV
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, phase, epoch_loss])

            # 保存最佳模型
            if phase == "test" and epoch_loss < best_loss:
                best_loss = epoch_loss
                no_improve_epochs = 0
                torch.save(model.state_dict(), os.path.join(train_dir, "best_model.pth"))
                print(f"Saved best model with Loss: {best_loss:.4f}")
            elif phase == "test":
                no_improve_epochs += 1

        # 检查早停条件
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best val Loss: {best_loss:.4f}")
            break

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Loss: {best_loss:.8f}")

# --- 开始训练 ---
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=300, patience=25)
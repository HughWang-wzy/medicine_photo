import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import csv
import time
from DermoExpert import DermoExpert

# 定义设备
device = torch.device("cuda:0")
print(f"Using device: {device}")

# 定义图像尺寸和其他超参数
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 500
PATIENCE = 30
LEARNING_RATE = 3.3e-6

# --- 数据准备与增强 ---
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.RandomGrayscale(p=0.1),  # 增加随机灰度转换
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 增加随机平移
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/home/hugh/newdisk/medicine_photo/ISIC_DATA'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
               for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
print("Dataset Sizes:", dataset_sizes)
print("Class Names:", class_names)

# --- 构建 DermoExpert 模型 ---
print("\n--- 构建 DermoExpert 模型 ---")
model = DermoExpert(num_classes=NUM_CLASSES, pretrained=True, dropout_rate=0.5)
model = model.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# --- 从指定模型开始训练 ---
print("\n--- 加载模型并开始训练 ---")

best_model_path = "dermoexpert_run/train14/best_model_dermoexpert_training.pth"
if os.path.exists(best_model_path):
    print(f"Loading model weights from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
else:
    raise FileNotFoundError(f"Model weights not found at: {best_model_path}")


for param in model.parameters():
    param.requires_grad = True

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 创建训练结果保存目录 ---
run_dir = "./dermoexpert_run"
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
    writer.writerow(["Epoch", "Phase", "Loss", "Accuracy"])

# --- 训练函数 ---
def train_model(model, criterion, optimizer, num_epochs=10, stage_name="", patience=10):
    since = time.time()
    best_acc = 0.0
    best_epoch = 0
    no_improve_epochs = 0  # 记录连续未提升的 epoch 数

    # 用于绘图的历史记录
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 使用 tqdm 显示进度条
            with tqdm(total=len(dataloaders[phase]), desc=f"{phase} Epoch {epoch+1}/{num_epochs}") as pbar:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 梯度清零
                    optimizer.zero_grad()

                    # 只在训练阶段进行前向传播和梯度计算
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # 统计损失和准确率
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # 更新进度条
                    pbar.update(1)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # 记录历史
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存训练日志到 CSV
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, phase, epoch_loss, epoch_acc.item()])

            # 保存最佳模型
            if phase == 'test':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    no_improve_epochs = 0  # 重置未提升计数
                    torch.save(model.state_dict(), os.path.join(train_dir, f"best_model_{stage_name}.pth"))
                    print(f"Saved best model for {stage_name} with Acc: {best_acc:.4f}")
                else:
                    no_improve_epochs += 1  # 未提升计数加 1

        # 保存最近一次的模型
        torch.save(model.state_dict(), os.path.join(train_dir, f"last_model_{stage_name}.pth"))

        # 检查早停条件
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best val Acc: {best_acc:.4f} at epoch {best_epoch+1}")
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    return model, history

# --- 开始训练 ---
model, history = train_model(model, criterion, optimizer, num_epochs=EPOCHS, stage_name="dermoexpert_training", patience=PATIENCE)
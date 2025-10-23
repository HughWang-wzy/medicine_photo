import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from dataset import get_cifar10_datasets
from models.vanilla_vit import ViT
from models.lightweight_vit import DynamicViT, HierarchicalViT
from trainer import train_model, evaluate_model
from utils import calculate_class_weights, analyze_model_complexity

from torchvision import transforms, models

from timm.data import Mixup
import random

class RandomChoice:
    def __init__(self, transforms, probabilities):
        self.transforms = transforms
        self.probabilities = probabilities

    def __call__(self, inputs, labels):
        transform = random.choices(self.transforms, weights=self.probabilities, k=1)[0]
        return transform(inputs, labels)
    
def main(args):
        
    """
    主函数，用于执行整个实验流程。
    """
    # 1. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备是: {device}")

    # 2. 加载数据集
    print("开始加载CIFAR-10数据集...")
    train_dataset, test_dataset = get_cifar10_datasets()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print("数据集加载完成。")

    # 3. 初始化模型
    print(f"正在初始化模型: {args.model_type}")
    # model = ViT(
    #     image_size=224,
    #     patch_size=16,
    #     num_classes=args.num_classes,
    #     dim=512,
    #     depth=12,
    #     heads=16,
    #     mlp_dim=2048,
    #     dropout=0.2,
    #     emb_dropout=0.2
    # ).to(device)

    # 2a. 加载预训练模型
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1).to(device)

    # 2b. 替换分类头
    #    args.num_classes 应该被设置为 2 (你的数据集是2类)
    original_in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(original_in_features, args.num_classes)
    model = model.to(device)

    # 4. 模型复杂度分析
    # print("\n模型复杂度分析:")
    # analyze_model_complexity(model, (3, 224, 224))
    # print("-" * 50)

    # 5. 定义损失函数和优化器
    # --- 类不平衡问题解决方案 ---
    print("正在计算类别权重以解决数据不平衡问题...")
    # 注意：此处需要访问dataset的内部targets，实际应用中请确保dataset类支持此操作

    # class_weights = calculate_class_weights(train_dataset.targets).to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    
     # 我们用 RandomChoice 来实现“随机选一个”
    mixup_cutmix = RandomChoice(
        transforms=[
            Mixup(num_classes=args.num_classes, mixup_alpha=0.8, cutmix_alpha=0.0),
            Mixup(num_classes=args.num_classes, mixup_alpha=0.0, cutmix_alpha=1.0)
        ],
        probabilities=[0.5, 0.5]
    )
    
    print("已启用 MixUp 和 CutMix。")

    # --- 关键步骤 2: 修改损失函数 ---
    # Mixup/CutMix 
    criterion = nn.CrossEntropyLoss() 

    print("类别权重计算完成，已应用到损失函数中。")
    # --------------------------

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # 6. 训练和评估模型
    print("开始训练模型...")
    train_model(
        model=model,
        dataloaders={'train': train_loader, 'test': test_loader},
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=50,
        stage_name="vit_stage1",
        patience=10,
        train_dir="./models",
        mixup_cutmix=mixup_cutmix
    )
    # train_model(model, train_loader, criterion, optimizer, scheduler, device, epochs=args.epochs)
    print("训练完成。")

    print("\n开始在平衡测试集上评估模型...")
    evaluate_model(model, test_loader, criterion, device)
    print("评估完成。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViT 轻量化实验')
    parser.add_argument('--model_type', type=str, default='vanilla_vit',
                        choices=['vanilla_vit', 'dynamic_vit', 'hierarchical_vit'],
                        help='选择要运行的模型类型')
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--num_classes', type=int, default=2, help='类别数量')

    args = parser.parse_args()
    main(args)

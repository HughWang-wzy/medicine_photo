import torch
from tqdm import tqdm

def train_model(model, train_loader, criterion, optimizer, scheduler, device, epochs):
    """
    训练模型的函数。
    """
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # 使用tqdm显示进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计损失和准确率
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # 更新进度条信息
            progress_bar.set_postfix(loss=running_loss/total_samples, accuracy=100. * correct_predictions/total_samples)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions / len(train_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        
        scheduler.step()

def evaluate_model(model, test_loader, criterion, device):
    """
    评估模型的函数。
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            progress_bar.set_postfix(loss=running_loss/total_samples, accuracy=100. * correct_predictions/total_samples)

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = correct_predictions / len(test_loader.dataset)
    
    print(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

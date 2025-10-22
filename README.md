# 医学图像处理项目

本项目旨在处理和分析医学图像，主要包括图像分割、分类和模型训练。以下是项目中几个主要文件的功能说明：

---

## 文件说明

### 1. `test_efficientNet_model.py`
- **功能**：用于测试 EfficientNet 模型的性能。
- **主要内容**：
  - 加载预训练的 EfficientNet 模型。
  - 对测试数据集进行推理，计算模型的准确率、召回率、F1 分数等性能指标。
  - 可视化测试结果（如 ROC 曲线）。

---

### 2. `test_DermoExpert.py`
- **功能**：测试 DermoExpert 模型的性能。
- **主要内容**：
  - 加载 DermoExpert 模型。
  - 对测试数据集进行推理，评估模型在医学图像分类任务中的表现。
  - 输出分类报告和混淆矩阵。

---

### 3. `efficientNet_model.py`
- **功能**：定义 EfficientNet 模型的结构和训练流程。
- **主要内容**：
  - 使用 PyTorch 实现 EfficientNet 模型。
  - 包含模型的训练、验证和保存逻辑。
  - 支持迁移学习，加载预训练权重。

---

### 4. `DermoExpert.py`
- **功能**：定义 DermoExpert 模型的结构。
- **主要内容**：
  - 实现多个特征提取模块（ConvNeXt、Xception、EfficientNet）。
  - 提供增强的分类头，用于医学图像分类任务。
  - 支持模块化设计，便于扩展和修改。

---

### 5. `Ai_Competition/u2net2.py`
- **功能**：实现 U²-Net 模型，用于医学图像分割。
- **主要内容**：
  - 定义 U²-Net 模型的网络结构。
  - 支持多尺度特征提取，适用于复杂的医学图像分割任务。
  - 提供模型的前向传播逻辑。

---

### 6. `svc_model.py`
- **功能**：实现支持向量机（SVM）分类器。
- **主要内容**：
  - 使用 SVM 对特征进行分类。
  - 使用 EfficientNet 特征提取模型结合，完成分类任务。
  - 输出分类结果和性能评估指标。

---

## 数据说明
- **数据目录**：
  - `ISIC_DATA`：原始医学图像数据集。
  - `ISIC_DATA_seg`：分割任务的标注数据。

- **模型文件**：
  - `best_model_stage1.pth` 和 `best_model_stage2.pth`：训练好的模型权重文件。
  - `yolo11m-cls.pt`、`yolo11n.pt`、`yolo11x-cls.pt`：YOLO 模型的权重文件。

---

## 运行说明
1. **测试 EfficientNet 模型**：
   ```bash
   python test_efficientNet_model.py

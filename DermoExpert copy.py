
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm

class FMG_ResNet50(nn.Module):
    """ Feature Map Generator 1 (FMG-1) based on ResNet-50 """
    def __init__(self, pretrained=True):
        super().__init__()
        # Load pre-trained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        # Remove the final average pooling and fully connected layers to get feature maps
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.features(x)

class FMG_Xception(nn.Module):
    """ Feature Map Generator 2 (FMG-2) based on Xception """
    def __init__(self, pretrained=True):
        super().__init__()
        # Load pre-trained Xception model using timm library
        # num_classes=0 removes the final classification layer
        self.features = timm.create_model('xception', pretrained=pretrained, num_classes=0, global_pool='')

    def forward(self, x):
        out = self.features(x)
        # print(f"FMG_Xception output shape: {out.shape}")  # 添加调试信息
        return out

class FMG_DenseNet121(nn.Module):
    """ Feature Map Generator 3 (FMG-3) based on DenseNet-121 """
    def __init__(self, pretrained=True):
        super().__init__()
        # Load pre-trained DenseNet-121 and use its feature extractor part
        densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = densenet.features

    def forward(self, x):
        # DenseNet's feature extractor output needs a ReLU activation at the end
        out = self.features(x)
        out = F.relu(out, inplace=True)
        return out

class ClassificationHead(nn.Module):
    """ A standard classification head: GAP -> Dropout -> Linear """
    def __init__(self, in_channels, num_classes, dropout_rate=0.5):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.gap(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class EnhancedClassificationHead(nn.Module):
    """增强的分类头"""
    def __init__(self, in_channels, num_classes, dropout_rate=0.3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)  # 增加最大池化
        
        # 更深的MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // 2),  # 结合GAP和GMP
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(in_channels // 2, in_channels // 4),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(in_channels // 4, num_classes)
        )
        
    def forward(self, x):
        gap = self.gap(x)
        gmp = self.gmp(x)
        x_combined = torch.cat([gap, gmp], dim=1)
        x_combined = x_combined.view(x_combined.size(0), -1)
        return self.mlp(x_combined)
    
class DermoExpert(nn.Module):
    """
    The complete DermoExpert model implementing the 'Method-5' hybrid architecture.
    
    Args:
        num_classes (int): The number of output classes for the classification task.
        pretrained (bool): Whether to load ImageNet pre-trained weights for the FMGs.
        dropout_rate (float): The dropout probability to use in the classification heads.
    """
    def __init__(self, num_classes, pretrained=True, dropout_rate=0.5):
        super().__init__()
        
        # --- 1. Instantiate Feature Map Generators (FMGs) ---
        self.fmg1 = FMG_ResNet50(pretrained=pretrained)
        self.fmg2 = FMG_Xception(pretrained=pretrained)
        self.fmg3 = FMG_DenseNet121(pretrained=pretrained)

        # --- 2. Define output channels for each FMG ---
        # These values are standard for these architectures
        fmg1_out_channels = 2048  # ResNet-50
        fmg2_out_channels = 2048  # Xception
        fmg3_out_channels = 1024  # DenseNet-121
        fused_out_channels = fmg1_out_channels + fmg2_out_channels + fmg3_out_channels

        # --- 3. Instantiate the four parallel classification heads ---
        self.head1 = EnhancedClassificationHead(fmg1_out_channels, num_classes, dropout_rate)
        self.head2 = EnhancedClassificationHead(fmg2_out_channels, num_classes, dropout_rate)
        self.head3 = EnhancedClassificationHead(fmg3_out_channels, num_classes, dropout_rate)
        self.head_fused = EnhancedClassificationHead(fused_out_channels, num_classes, dropout_rate)

    def forward(self, x):
        # --- Level 1 Ensemble (Part 1): Feature Extraction ---
        # Get feature maps from each parallel FMG
        x1 = self.fmg1(x)  # Output shape: (batch, 2048, H/32, W/32)
        x2 = self.fmg2(x)  # Output shape: (batch, 2048, H/32, W/32)
        x3 = self.fmg3(x)  # Output shape: (batch, 1024, H/32, W/32)

        # --- Level 1 Ensemble (Part 2): Feature Fusion ---
        target_size = x1.shape[2:]
        # print(f"x2 shape: {x2.shape}")  # 添加调试信息
        x2_resized = F.adaptive_avg_pool2d(x2, target_size)
        x3_resized = F.adaptive_avg_pool2d(x3, target_size)
        
        # Concatenate feature maps along the channel dimension
        x_fused = torch.cat([x1, x2_resized, x3_resized], dim=1)

        # --- Level 2 Ensemble: Prediction Aggregation ---
        # Get logits from each of the four heads
        logits1 = self.head1(x1)
        logits2 = self.head2(x2)
        logits3 = self.head3(x3)
        logits_fused = self.head_fused(x_fused)

        # Convert logits to probabilities using softmax
        probs1 = F.softmax(logits1, dim=1)
        probs2 = F.softmax(logits2, dim=1)
        probs3 = F.softmax(logits3, dim=1)
        probs_fused = F.softmax(logits_fused, dim=1)

        # Average the probabilities from the four heads for the final prediction
        final_probs = (probs1 + probs2 + probs3 + probs_fused) / 4.0
        
        return final_probs

# --- Example Usage ---
if __name__ == '__main__':
    # Define the number of classes based on the target dataset
    # ISIC-2016: 2 classes
    # ISIC-2017: 3 classes
    # ISIC-2018: 7 classes
    NUM_CLASSES = 7 
    
    # Input image size as specified in the paper for the classification model
    INPUT_SIZE = 192

    # Instantiate the DermoExpert model
    print("Instantiating DermoExpert model...")
    model = DermoExpert(num_classes=NUM_CLASSES, pretrained=True)
    print("Model instantiated successfully.")

    # Create a dummy input tensor to test the forward pass
    # Shape: (batch_size, channels, height, width)
    dummy_input = torch.randn(4, 3, INPUT_SIZE, INPUT_SIZE)

    # Perform a forward pass
    print(f"\nPerforming a forward pass with a dummy input of shape: {dummy_input.shape}")
    with torch.no_grad():
        model.eval()
        output_probs = model(dummy_input)
    
    print(f"Forward pass successful.")
    print(f"Output probability tensor shape: {output_probs.shape}")
    print(f"Example output probabilities for the first image in batch:\n{output_probs}")
    
    # Verify that the probabilities sum to 1
    print(f"Sum of probabilities for the first image: {torch.sum(output_probs):.4f}")

    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal model parameters: {total_params:,}")
    print(f"Trainable model parameters: {trainable_params:,}")
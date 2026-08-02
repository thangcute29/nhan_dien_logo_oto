import torch
import torch.nn as nn
from torchvision import models

def setup_model(num_classes, device, freeze_layers=True):
    """
    Tải mô hình ResNet-50 đã được huấn luyện trước, sửa lớp FC và đóng băng các lớp.
    """
    # Tải mô hình ResNet-50 đã được huấn luyện trước
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Thay thế lớp phân loại cuối cùng (Fully Connected layer)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Đóng băng tất cả các lớp trong mô hình (trừ lớp fc mới)
    if freeze_layers:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    model = model.to(device)
    return model

def load_best_model(num_classes, model_path, device):
    """
    Khởi tạo mô hình và tải trọng số đã lưu.
    """
    model = models.resnet50(weights=None) # Không cần weights mặc định
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        print(f"Đã tải trọng số mô hình từ {model_path}")
        return model
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file trọng số tại {model_path}. Hãy huấn luyện trước.")
        return None
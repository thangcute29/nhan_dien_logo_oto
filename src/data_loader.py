# src/data_loader.py (Đã sửa lỗi TypeError)

import kagglehub
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, ConcatDataset 
import torch
import os
import numpy as np
import math
from tqdm import tqdm
# Import cụ thể các hằng số từ config
from src.config import (
    DATASET_IDENTIFIER, ROBOFLOW_DATA_PATH, IMAGE_SIZE, MEAN_VALUES, 
    STD_VALUES, BATCH_SIZE, NUM_WORKERS
)


def _find_imagefolder_root(base_path, max_depth=6):
    """
    Đi sâu vào các thư mục con để tìm thư mục chứa các thư mục lớp (class folders).
    """
    cur = base_path
    for _ in range(max_depth):
        try:
            entries = os.listdir(cur)
        except Exception:
            break
        subdirs = [d for d in entries if os.path.isdir(os.path.join(cur, d))]
        if not subdirs:
            break
        
        # Kiểm tra xem có ảnh trong các thư mục con không (dấu hiệu của thư mục lớp)
        any_sub_contains_images = False
        for sd in subdirs:
            sd_path = os.path.join(cur, sd)
            try:
                sd_files = os.listdir(sd_path)
            except Exception:
                sd_files = []
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')) for f in sd_files):
                any_sub_contains_images = True
                break
        
        if any_sub_contains_images:
            return cur # Tìm thấy thư mục gốc chứa các lớp
        
        # Nếu chưa phải thư mục gốc, đi sâu vào thư mục con đầu tiên
        cur = os.path.join(cur, subdirs[0])
        
    return base_path # Fallback nếu không tìm thấy cấu trúc chuẩn


# Lớp bọc (wrapper) để áp dụng transforms khác nhau cho các Subset
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        # *** QUAN TRỌNG: x lúc này là PIL Image vì ImageFolder được tạo với transform=None ***
        x, y = self.subset[index] 
        if self.transform:
            x = self.transform(x) # Áp dụng transform (bao gồm ToTensor) cho PIL Image, chỉ 1 lần
        return x, y
    def __len__(self):
        return len(self.subset)


def get_dataloaders(mean_values, std_values, image_size, batch_size, train_ratio=0.7, val_ratio=0.15):
    """Tạo transforms, tải dataset từ Kaggle/Roboflow, chia và trả về các DataLoader."""

    # 1. ĐỊNH NGHĨA TRANSFORMS (Đầy đủ, bao gồm ToTensor/Normalize)
    
    # Transforms Augmentation cho tập Train
    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean_values, std=std_values)
    ])

    # Transforms cho tập Val/Test (chỉ resize và normalize)
    transform_val_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean_values, std=std_values)
    ])


    # 2. TẢI VÀ KẾT HỢP DATASET (Không áp dụng transform ngay)

    datasets_to_combine = []
    classes = None
    
    # --- Tải Dataset từ Kaggle Hub ---
    print(f"\n📦 Đang tải dataset '{DATASET_IDENTIFIER}' từ Kaggle Hub...")
    try:
        dataset_path = kagglehub.dataset_download(DATASET_IDENTIFIER)
        kaggle_root_path = _find_imagefolder_root(dataset_path) 
        print(f"ℹ️ Sử dụng thư mục gốc ImageFolder (Kaggle): {kaggle_root_path}")
        
        # LOAD DATASET KAGGLE - QUAN TRỌNG: transform=None để trả về ảnh PIL thô
        kaggle_dataset = ImageFolder(kaggle_root_path, transform=None) 
        datasets_to_combine.append(kaggle_dataset)
        classes = kaggle_dataset.classes
        print(f"✅ Đã thêm Kaggle Dataset. Số mẫu: {len(kaggle_dataset)}")
    except Exception as e:
        print(f"❌ Lỗi khi tải Kaggle Dataset: {e}. Bỏ qua Kaggle dataset.")


    # --- Tải Dataset từ Roboflow (Local) ---
    if os.path.isdir(ROBOFLOW_DATA_PATH):
        print(f"\nℹ️ Đang tải Dataset từ Roboflow (Local): {ROBOFLOW_DATA_PATH}")
        try:
            roboflow_root = _find_imagefolder_root(ROBOFLOW_DATA_PATH)
            # LOAD DATASET ROBOFLOW - QUAN TRỌNG: transform=None để trả về ảnh PIL thô
            roboflow_dataset = ImageFolder(roboflow_root, transform=None) 
            
            if classes and set(classes) != set(roboflow_dataset.classes):
                print("❌ CẢNH BÁO: Tập lớp của Roboflow Dataset không khớp với Kaggle Dataset. Đang sử dụng lớp từ Kaggle.")
            else:
                datasets_to_combine.append(roboflow_dataset)
                if not classes: classes = roboflow_dataset.classes
                print(f"✅ Đã thêm Roboflow Dataset. Số mẫu: {len(roboflow_dataset)}")

        except Exception as e:
            print(f"⚠️ Lỗi khi tải Roboflow Dataset: {e}. Bỏ qua Roboflow dataset.")
    else:
        print(f"⚠️ Không tìm thấy thư mục Roboflow tại: {ROBOFLOW_DATA_PATH}. Bỏ qua.")


    if not datasets_to_combine:
        raise RuntimeError("Không tìm thấy dataset nào để tải. Vui lòng kiểm tra đường dẫn và kết nối internet.")
        
    full_dataset = ConcatDataset(datasets_to_combine) 
    print(f"\n🎉 Tổng số lượng mẫu sau khi kết hợp: {len(full_dataset)}")
    print(f"Số lượng lớp (Classes): {len(classes)}")


    # 3. PHÂN CHIA DATASET (70/15/15)
    train_size = math.floor(train_ratio * len(full_dataset))
    val_size = math.floor(val_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_subset, val_subset, test_subset = random_split(full_dataset, [train_size, val_size, test_size])

    # 4. ÁP DỤNG TRANSFORMS VÀ TẠO DATALOADER
    # Sử dụng TransformSubset để áp dụng transform khác nhau cho các tập đã chia
    train_dataset = TransformSubset(train_subset, transform=transform_train)
    val_dataset = TransformSubset(val_subset, transform=transform_val_test)
    test_dataset = TransformSubset(test_subset, transform=transform_val_test)


    # Tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    
    return train_loader, val_loader, test_loader, classes
# main.py
# PHIÊN BẢN HOÀN HẢO: DATA FIX SIÊU MẠNH + GIAO DIỆN ĐẸP + MENU SAU KHI TRAIN

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import os
import sys
import shutil # Thư viện thao tác file

# Cần cài đặt thư viện ultralytics: pip install ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    pass

# Import các hằng số từ config
from src.config import (
    DEVICE, MODEL_SAVE_PATH, MEAN_VALUES, STD_VALUES, IMAGE_SIZE, BATCH_SIZE, 
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, STEP_SIZE, GAMMA, ROBOFLOW_DATA_PATH
)
from src.data_loader import get_dataloaders 
from src.model import setup_model, load_best_model
from src.train import train_model
from src.predict import predict_image , predict_from_image, predict_with_camera

# ======================================================================================
# 🛠️ HÀM TỰ ĐỘNG SỬA DỮ LIỆU (PHIÊN BẢN SIÊU MẠNH - TÌM TRONG IMAGES/LABELS)
# ======================================================================================
def auto_fix_roboflow_data(raw_data_path, target_path, classes):
    print(f"\n🔍 Đang kiểm tra dữ liệu tại: {raw_data_path}")

    # 1. Kiểm tra folder gốc
    if not os.path.exists(raw_data_path):
        print(f"⚠️ Không tìm thấy folder gốc '{raw_data_path}'. Code sẽ chỉ dùng Kaggle.")
        return

    # 2. Kiểm tra nếu dữ liệu đích đã có
    test_check_path = os.path.join(target_path, classes[0])
    if os.path.exists(test_check_path):
        if len(os.listdir(test_check_path)) > 0:
            print(f"✅ Dữ liệu tại '{target_path}' đã sẵn sàng. Bỏ qua bước chuyển đổi.")
            return

    print(f"⚙️  Đang cấu trúc lại dữ liệu (Gom train/test/valid về chung 1 folder)...")
    
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.makedirs(target_path)

    total_count = 0
    # Các đuôi ảnh có thể gặp
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG']

    # Duyệt qua 3 tập gốc
    for split in ['train', 'valid', 'test']:
        old_split_path = os.path.join(raw_data_path, split)
        if not os.path.exists(old_split_path) and split == 'valid':
             old_split_path = os.path.join(raw_data_path, 'val')
        
        if not os.path.exists(old_split_path): continue

        # Tìm folder con labels/images hoặc dùng luôn folder cha
        labels_path = os.path.join(old_split_path, 'labels')
        images_path = os.path.join(old_split_path, 'images')
        if not os.path.exists(labels_path): labels_path = old_split_path
        if not os.path.exists(images_path): images_path = old_split_path

        # Lấy danh sách file nhãn
        try:
            label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt') and f != 'classes.txt']
        except Exception:
            continue
        
        for label_file in tqdm(label_files, desc=f"Gom dữ liệu {split}"):
            try:
                with open(os.path.join(labels_path, label_file), 'r') as f:
                    line = f.readline()
                    if not line: continue
                    class_id = int(line.split()[0])
                
                if class_id >= len(classes): continue
                
                class_name = classes[class_id]
                base_name = os.path.splitext(label_file)[0]
                
                # Tìm ảnh tương ứng
                image_file = None
                for ext in valid_extensions:
                    img_name = base_name + ext
                    if os.path.exists(os.path.join(images_path, img_name)):
                        image_file = img_name
                        break
                
                if image_file:
                    dest_folder = os.path.join(target_path, class_name)
                    os.makedirs(dest_folder, exist_ok=True)
                    new_filename = f"{split}_{image_file}"
                    shutil.copy(
                        os.path.join(images_path, image_file),
                        os.path.join(dest_folder, new_filename)
                    )
                    total_count += 1
            except Exception:
                continue
                
    print(f"✅ Đã xử lý xong! Tổng cộng {total_count} ảnh Roboflow đã sẵn sàng.")

# ======================================================================================
# CÁC HÀM CŨ GIỮ NGUYÊN
# ======================================================================================

def get_image_path_from_terminal():
    print("\n🖼️ Vui lòng dán đường dẫn đầy đủ tới file ảnh của bạn vào đây.")
    print("(Mẹo: Bạn có thể kéo và thả file ảnh vào cửa sổ này để lấy đường dẫn)")
    image_path = input("👉 Đường dẫn file ảnh: ")
    return image_path.strip().replace('"', '')

def evaluate_test_set(model, test_loader, classes, device):
    print("\nBắt đầu đánh giá trên tập Test...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Thu thập kết quả..."):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n✅ Độ chính xác trên tập Test: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\n📊 Báo cáo phân loại trên tập Test:")
    print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))
    
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    ax.set_title("Ma trận nhầm lẫn (Test Set)")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("✅ Đã lưu Ma trận nhầm lẫn vào file: confusion_matrix.png")
    plt.close(fig)

def run_menu(best_model, classes, device):
    """Hàm chạy menu lựa chọn (Tách riêng để dùng được ở cả 2 kịch bản)"""
    # Tải YOLOv8
    try:
        yolo_model = YOLO("yolov8n.pt") 
        yolo_model.to(device)
        print("✅ Tải YOLOv8 thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi tải YOLOv8: {e}")
        return

    while True:
        print("\n" + "="*45)
        print(" LUA CHON CHUC NANG NHAN DIEN LOGO O TO ")
        print("-"*(45))
        print("1. Nhận diện qua Camera trực tiếp 📸")
        print("2. Nhận diện từ một hình ảnh 🖼️")
        print("3. Thoát chương trình 🚪")
        print("="*(45))
        choice = input("👉 Vui lòng nhập lựa chọn của bạn (1, 2, hoặc 3): ")
        
        if choice == '1':
            predict_with_camera(best_model, yolo_model, classes, device) 
        elif choice == '2':
            path = get_image_path_from_terminal()
            if path and os.path.exists(path):
                predict_from_image(best_model, yolo_model, classes, device, path) 
            else:
                print("⚠️ Đường dẫn file không tồn tại hoặc bạn chưa nhập.")
        elif choice == '3':
            print("👋 Tạm biệt...")
            break
        else:
            print("❌ Lựa chọn không hợp lệ.")

# ======================================================================================
# CHƯƠNG TRÌNH CHÍNH
# ======================================================================================
def main():
    
    # --- BƯỚC 0: TỰ ĐỘNG CHUẨN BỊ DỮ LIỆU ---
    RAW_ROBOFLOW_FOLDER = "nhan biet logo o to.v4i.yolov8" 
    ROBOFLOW_DATA_PATH = "./roboflow_car_logos" 
    CLASS_NAMES = ['hyundai', 'lexus', 'mazda', 'mercedes', 'opel', 'skoda', 'toyota', 'volkswagen']

    # Gọi hàm sửa lỗi siêu mạnh
    auto_fix_roboflow_data(RAW_ROBOFLOW_FOLDER, ROBOFLOW_DATA_PATH, CLASS_NAMES)

    # --- BƯỚC 1: Tải dữ liệu ---
    try:
        train_loader, val_loader, test_loader, classes = get_dataloaders(
            MEAN_VALUES, STD_VALUES, IMAGE_SIZE, BATCH_SIZE 
        )
        print(f"🔍 Tìm thấy {len(classes)} lớp: {classes}")
    except RuntimeError as e:
        print(f"❌ Lỗi nghiêm trọng: {e}")
        return
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return

    # --- BƯỚC 2: Model ---
    model = setup_model(len(classes), DEVICE, freeze_layers=True)
    
    # KỊCH BẢN 1: Đã có mô hình -> Vào thẳng Menu
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"✅ Tìm thấy mô hình đã huấn luyện: '{MODEL_SAVE_PATH}'.")
        best_model = load_best_model(len(classes), MODEL_SAVE_PATH, DEVICE)
        evaluate_test_set(best_model, test_loader, classes, DEVICE)
        
        # Gọi menu
        run_menu(best_model, classes, DEVICE)
    
    # KỊCH BẢN 2: Chưa có mô hình -> Huấn luyện -> Rồi mới vào Menu
    else: 
        print("❌ Không tìm thấy mô hình đã huấn luyện. Bắt đầu huấn luyện.")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
        
        trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, DEVICE, MODEL_SAVE_PATH)
        
        if trained_model:
            evaluate_test_set(trained_model, test_loader, classes, DEVICE)
            # Sau khi train xong, chuyển sang chế độ dùng luôn!
            print("\n🎉 Huấn luyện hoàn tất! Chuyển sang chế độ nhận diện...")
            run_menu(trained_model, classes, DEVICE)

if __name__ == "__main__":
    main()
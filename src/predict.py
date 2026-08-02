import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 # Cần thiết cho camera và xử lý ảnh YOLOv8
# Cần cài đặt thư viện ultralytics: pip install ultralytics
from src.model import load_best_model # Import từ module model đã định nghĩa
from ultralytics import YOLO
from src.config import IMAGE_SIZE, MEAN_VALUES, STD_VALUES, DEVICE


def get_inference_transform():
    """Tạo transform chuẩn hóa cho mô hình ResNet-50 khi inference."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_VALUES, std=STD_VALUES)
    ])


# Hàm xử lý Camera (ĐÃ CHUYỂN TỪ MAIN)
def predict_with_camera(classification_model, yolo_model, classes, device):
    """Phát hiện bằng YOLOv8 và Phân loại bằng ResNet-50 qua Camera."""
    print("\n--- BẮT ĐẦU NHẬN DIỆN CAMERA (Ấn 'q' để thoát) ---")
    cap = cv2.VideoCapture(0)
    inference_transform = get_inference_transform() # LẤY TRANSFORM

    if not cap.isOpened():
        print("❌ Lỗi: Không thể mở camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Phát hiện vật thể (YOLOv8)
        results = yolo_model(frame, verbose=False) # Thêm verbose=False để code sạch hơn
        
        found_logo = False # Biến cờ để kiểm tra có phát hiện được gì không

        # Lặp qua các vật thể được phát hiện
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                # cls không cần thiết nếu ta chỉ crop

                # Giả sử ta chỉ quan tâm đến các vật thể có độ tin cậy cao
                if conf > 0.5:
                    found_logo = True
                    
                    # 2. Cắt và Tiền xử lý (Cho mô hình ResNet-50)
                    cropped_img_bgr = frame[y1:y2, x1:x2]
                    
                    if cropped_img_bgr.size == 0: continue # Xử lý lỗi cắt ảnh rỗng

                    cropped_img_rgb = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(cropped_img_rgb)
                    
                    # Áp dụng transform đã lấy
                    input_tensor = inference_transform(pil_image).unsqueeze(0).to(device)
                    
                    # 3. Phân loại (ResNet-50)
                    classification_model.eval()
                    with torch.no_grad():
                        output = classification_model(input_tensor)
                        probabilities = F.softmax(output, dim=1)
                        _, predicted_idx = torch.max(output, 1)
                    
                    predicted_class = classes[predicted_idx.item()]
                    confidence = probabilities[0, predicted_idx.item()].item()
                    
                    # 4. Hiển thị kết quả
                    label = f'{predicted_class} ({confidence:.2f})'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if not found_logo:
             cv2.putText(frame, 'No object detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        cv2.imshow('YOLOv8 + ResNet Logo Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("--- NHẬN DIỆN CAMERA KẾT THÚC ---")


# Hàm xử lý Ảnh tĩnh (ĐÃ CHUYỂN TỪ MAIN)
def predict_from_image(classification_model, yolo_model, classes, device, image_path):
    """Phát hiện bằng YOLOv8 và Phân loại bằng ResNet-50 từ Ảnh tĩnh."""
    print(f"\n--- BẮT ĐẦU NHẬN DIỆN TỪ ẢNH: {os.path.basename(image_path)} ---")
    
    inference_transform = get_inference_transform() # LẤY TRANSFORM
    
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            print("❌ Lỗi: Không thể đọc file ảnh.")
            return

        # 1. Phát hiện vật thể (YOLOv8)
        results = yolo_model(frame, verbose=False)
        display_frame = frame.copy()
        found_logo = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()

                if conf > 0.5:
                    found_logo = True
                    
                    # 2. Cắt và Tiền xử lý
                    cropped_img_bgr = frame[y1:y2, x1:x2]
                    
                    if cropped_img_bgr.size == 0: continue

                    cropped_img_rgb = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(cropped_img_rgb)
                    
                    input_tensor = inference_transform(pil_image).unsqueeze(0).to(device)
                    
                    # 3. Phân loại (ResNet-50)
                    classification_model.eval()
                    with torch.no_grad():
                        output = classification_model(input_tensor)
                        probabilities = F.softmax(output, dim=1)
                        _, predicted_idx = torch.max(output, 1)
                    
                    predicted_class = classes[predicted_idx.item()]
                    confidence = probabilities[0, predicted_idx.item()].item()
                    
                    # 4. Hiển thị kết quả lên ảnh
                    label = f'{predicted_class} ({confidence:.2f})'
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        if not found_logo:
             cv2.putText(display_frame, 'No object detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Hiển thị ảnh kết quả
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        plt.title('YOLOv8 Detection + ResNet-50 Classification')
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"❌ Xảy ra lỗi trong quá trình nhận diện ảnh: {e}")


def predict_image(image_path, model, classes, mean_values, std_values, image_size, device):
    """
    Dự đoán nhãn cho một ảnh riêng lẻ và hiển thị kết quả.
    """
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy file ảnh tại {image_path}")
        return

    try:
        # Load ảnh
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Lỗi khi tải ảnh: {e}")
        return

    # Transform cho Inference (Không có Augmentation)
    inference_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values, std=std_values)
    ])

    input_tensor = inference_transform(image).unsqueeze(0).to(device)

    # Dự đoán
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        # Chuyển output logit sang xác suất
        probabilities = F.softmax(output, dim=1)
        _, predicted_idx = torch.max(probabilities, 1)

    predicted_class = classes[predicted_idx.item()]
    confidence = probabilities[0, predicted_idx.item()].item()
    
    print(f"\nẢnh: {os.path.basename(image_path)}")
    print(f"Dự đoán: {predicted_class} (Độ tin cậy: {confidence:.2%})")

    # --- Hiển thị ảnh (Optional) ---
    # Transform ảnh để hiển thị (chỉ resize, toTensor)
    img_display = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])(image).numpy().transpose((1, 2, 0)) # C, H, W -> H, W, C

    plt.imshow(img_display)
    plt.title(f'Predicted: {predicted_class} ({confidence:.2%})')
    plt.axis('off')
    plt.show()

    return predicted_class
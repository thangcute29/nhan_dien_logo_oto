import torch

# Thiết lập thiết bị
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đường dẫn dữ liệu
DATASET_IDENTIFIER = "volkandl/car-brand-logos"
ROBOFLOW_DATA_PATH = "./roboflow_car_logos"
MODEL_SAVE_PATH = "best_model.pth"

# Cấu hình tiền xử lý (Mean/Std đã tính toán)
# Mean: [0.5400, 0.5392, 0.5472], Std: [0.2506, 0.2489, 0.2503]
MEAN_VALUES = [0.5400, 0.5392, 0.5472]
STD_VALUES = [0.2506, 0.2489, 0.2503]
IMAGE_SIZE = (224, 224)

NUM_WORKERS = 0

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Cấu hình Scheduler
STEP_SIZE = 7
GAMMA = 0.1

print(f"Sử dụng thiết bị: {DEVICE}")
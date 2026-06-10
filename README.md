# Nhận Diện Món Ăn Việt Nam & Ước Tính Calories

Hệ thống nhận diện **29 món ăn Việt Nam** từ ảnh chụp, sử dụng mô hình **EfficientNet-B0** kết hợp Transfer Learning. Ứng dụng tích hợp tra cứu thông tin dinh dưỡng, giải thích trực quan bằng Grad-CAM, và giao diện demo Streamlit.

> **Đồ án môn học:** Trí Tuệ Nhân Tạo — Nhóm 5

---

## Tính năng chính

- **Nhận diện món ăn** — Phân loại 29 món ăn Việt Nam phổ biến với độ chính xác ~84%
- **Grad-CAM** — Trực quan hóa vùng ảnh mô hình tập trung để đưa ra dự đoán
- **Tra cứu dinh dưỡng** — Calories, Protein, Carbs, Fat, vi chất (nguồn: Viện Dinh dưỡng Quốc gia VN)
- **Giới thiệu món ăn** — Câu chuyện, nguyên liệu chính của từng món
- **So sánh mô hình** — EfficientNet-B0 vs ResNet-18 với đánh giá chi tiết

## Danh sách 29 món ăn

| | | | |
|---|---|---|---|
| Bánh bèo | Bánh bột lọc | Bánh căn | Bánh canh |
| Bánh chưng | Bánh cuốn | Bánh giò | Bánh khọt |
| Bánh mì | Bánh pía | Bánh tét | Bánh tráng nướng |
| Bánh xèo | Bún bò Huế | Bún đậu mắm tôm | Bún mắm |
| Bún riêu | Bún thịt nướng | Cá kho tộ | Canh chua |
| Cao lầu | Cháo lòng | Cơm tấm | Gỏi cuốn |
| Hủ tiếu | Mì Quảng | Nem chua | Phở |
| Xôi xéo | | | |

---

## Kiến trúc hệ thống

```
Ảnh đầu vào (bất kỳ kích thước)
    │
    ▼
┌─────────────────────────────┐
│  Tầng Tiền xử lý            │
│  FixedAspectRatioPadding     │  ← Giữ nguyên tỷ lệ, padding đen
│  Resize 224×224 (LANCZOS)    │
│  Normalize (ImageNet)        │
│  Augmentation (Train only)   │  ← RandomFlip, Rotation, ColorJitter
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Tầng Mô hình               │
│  EfficientNet-B0 Backbone    │  ← Pretrained ImageNet (4.0M params)
│  Custom Classifier Head      │  ← Dropout→Linear(1280,256)→ReLU→Dropout→Linear(256,29)
│  WeightedRandomSampler       │  ← Xử lý mất cân bằng lớp
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Tầng Đầu ra                │
│  Top-K Predictions           │
│  Grad-CAM Heatmap            │
│  Thông tin dinh dưỡng        │
│  Câu chuyện món ăn           │
└─────────────────────────────┘
```

---

## Cấu trúc dự án

```
AI-Final-test-main/
├── app.py                      # Ứng dụng Streamlit demo
├── requirements.txt            # Dependencies
├── README.md
│
├── src/                        # Source code chính
│   ├── preprocess.py           # Tiền xử lý, Augmentation, WeightedRandomSampler
│   ├── model.py                # Kiến trúc mô hình, Grad-CAM
│   ├── train.py                # Huấn luyện EfficientNet-B0 (2-phase)
│   ├── train_resnet18.py       # Huấn luyện ResNet-18 (baseline)
│   ├── evaluate.py             # Đánh giá: Confusion Matrix, Classification Report
│   ├── predict.py              # Suy luận & tra cứu dinh dưỡng
│   ├── food_metadata.py        # Metadata 29 món ăn (tên, câu chuyện, nguyên liệu)
│   ├── load_calories.py        # Load dữ liệu calories từ CSV
│   ├── ingredients.py          # Thông tin nguyên liệu
│   ├── recipes.py              # Công thức nấu ăn
│   └── recipe_calculator.py    # Tính toán dinh dưỡng theo khẩu phần
│
├── data/raw/Images/            # Bộ dữ liệu 30VNFoods
│   ├── Train/                  # 17,118 ảnh (29 class)
│   ├── Validate/               # 2,478 ảnh
│   └── Test/                   # 5,040 ảnh
│
├── checkpoints/                # Model checkpoints
│   ├── best_model.pth          # EfficientNet-B0 (best val_acc=83.87%)
│   └── best_model_resnet18.pth # ResNet-18 (best val_acc=74.81%)
│
├── figures/                    # Biểu đồ kết quả
│   ├── training_curves_efficientnet_b0.png
│   ├── confusion_matrix_*.png
│   └── per_class_accuracy_*.png
│
├── notebooks/                  # Jupyter notebooks EDA
│   └── figures/
│       ├── eda_distribution.png
│       ├── eda_image_sizes.png
│       └── eda_sample_grid.png
│
└── docs/                       # Tài liệu báo cáo
    ├── ly_thuyet_chi_tiet.md
    └── pipeline_architecture.png
```

---

## Cài đặt

### Yêu cầu

- Python >= 3.10
- NVIDIA GPU với CUDA (khuyến nghị, có thể chạy CPU)
- ~4GB VRAM (RTX 3050 trở lên)

### Bước 1: Clone repository

```bash
git clone https://github.com/<your-username>/AI-Final-test-main.git
cd AI-Final-test-main
```

### Bước 2: Cài đặt dependencies

```bash
pip install -r requirements.txt
pip install streamlit
```

### Bước 3: Tải dữ liệu

Đặt bộ dữ liệu 30VNFoods vào `data/raw/Images/` với cấu trúc:
```
data/raw/Images/
├── Train/
│   ├── Banh beo/
│   ├── Banh bot loc/
│   └── ...
├── Validate/
└── Test/
```

---

## Sử dụng

### Chạy ứng dụng demo

```bash
streamlit run app.py
```

Truy cập `http://localhost:8501` → tải ảnh lên hoặc chọn ảnh mẫu → nhận kết quả.

### Huấn luyện mô hình

```bash
# EfficientNet-B0 (mô hình chính)
py -m src.train --model efficientnet_b0 --epochs 30 --batch_size 32

# ResNet-18 (baseline so sánh)
py -m src.train_resnet18
```

### Đánh giá mô hình

```bash
# Đánh giá 1 mô hình
py -m src.evaluate --checkpoint checkpoints/best_model.pth

# So sánh 2 mô hình
py -m src.evaluate --checkpoint checkpoints/best_model.pth --checkpoint2 checkpoints/best_model_resnet18.pth
```

Kết quả lưu trong `figures/` (Confusion Matrix, Classification Report, Per-class Accuracy).

---

## Demo ứng dụng Streamlit

Ứng dụng demo được xây dựng bằng **Streamlit**, cho phép người dùng tải ảnh lên và nhận kết quả nhận diện trong vài giây.

### Khởi chạy

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`

### Giao diện tổng quan

Ứng dụng gồm 2 phần:

- **Sidebar (bên trái):** Tùy chỉnh cài đặt
  - Khối lượng phần ăn: 100–600g (mặc định 350g)
  - Số lượng dự đoán hiển thị: 3–10 (mặc định 5)
  - Bật/tắt Grad-CAM
  - Thông tin mô hình & nhóm thực hiện

- **Khu vực chính:** Upload ảnh hoặc chọn ảnh mẫu → hiển thị kết quả

### Chức năng 1 — Nhận diện món ăn

Sau khi tải ảnh lên, hệ thống tự động:

1. Tiền xử lý ảnh (resize 224×224, normalize ImageNet)
2. Đưa qua mô hình EfficientNet-B0
3. Hiển thị kết quả:

```
┌────────────────────┬──────────────────────────────┐
│                    │  Kết quả nhận diện           │
│   Ảnh đầu vào     │  ─────────────               │
│                    │  Phở bò                      │
│                    │  Confidence: 96.5%            │
│                    │                              │
│                    │  Top dự đoán:                │
│                    │  Phở       ████████████ 96.5% │
│                    │  Bún riêu  ██           3.1% │
│                    │  Bún bò    █            0.4% │
└────────────────────┴──────────────────────────────┘
```

- Tên món ăn hiển thị tiếng Việt có dấu
- Thanh progress bar màu: xanh (>70%), vàng (40-70%), đỏ (<40%)

### Chức năng 2 — Grad-CAM (Giải thích trực quan)

Hiển thị 3 ảnh song song:

```
┌──────────────┬──────────────┬──────────────┐
│  Ảnh gốc     │  Heatmap     │   Overlay    │
│  (224×224)   │  (Jet color) │  (Gốc+Heat) │
└──────────────┴──────────────┴──────────────┘
```

- **Ảnh gốc** — resize về 224×224
- **Heatmap** — bản đồ nhiệt Grad-CAM (đỏ = tập trung cao)
- **Overlay** — phủ heatmap lên ảnh gốc

Vùng đỏ/vàng cho thấy mô hình đang "nhìn" vào đâu để quyết định. Ví dụ: với Phở, mô hình tập trung vào sợi phở + nước dùng thay vì viền bát.

### Chức năng 3 — Tra cứu dinh dưỡng

Thông tin tính theo khẩu phần tùy chỉnh (100–600g):

```
┌────────────┬────────────┬────────────┬────────────┐
│  Calories  │  Protein   │   Carbs    │    Fat     │
│  525 kcal  │   24.5g    │   62.3g    │   18.7g   │
│  150/100g  │            │            │           │
└────────────┴────────────┴────────────┴────────────┘

▸ Vi chất dinh dưỡng chi tiết (mở rộng)
  ┌──────────────────┬───────────────────┐
  │ Chất xơ    2.1g  │ Natri     850mg  │
  │ Canxi      45mg  │ Kali      320mg  │
  │ Sắt       2.8mg  │ Vitamin A  25mcg │
  │ Phospho   180mg  │ Vitamin C  12mg  │
  └──────────────────┴───────────────────┘
```

Nguồn dữ liệu: Viện Dinh dưỡng Quốc gia Việt Nam.

### Chức năng 4 — Câu chuyện & Nguyên liệu

```
┌──────────────────────┬──────────────────────┐
│  Câu chuyện món ăn   │  Nguyên liệu chính  │
│                      │                      │
│  Phở có nguồn gốc   │  - Bánh phở          │
│  từ Nam Định, đầu   │  - Xương bò          │
│  thế kỷ 20...       │  - Hành, gừng        │
│                      │  - Quế, hồi, thảo   │
│                      │    quả               │
└──────────────────────┴──────────────────────┘
```

### Chức năng 5 — Danh sách 29 món ăn

Khi chưa upload ảnh, trang chính hiển thị danh sách 29 món ăn được hỗ trợ dạng grid 4 cột.

### Luồng hoạt động

```
Người dùng upload ảnh
        │
        ▼
┌─── Tiền xử lý ────┐
│ Padding → Resize   │
│ Normalize ImageNet │
└────────┬───────────┘
         │
         ▼
┌─── Mô hình ───────┐
│ EfficientNet-B0    │──→ Top-K predictions
│ (best_model.pth)   │──→ Grad-CAM heatmap
└────────┬───────────┘
         │
         ▼
┌─── Tra cứu ───────┐
│ food_metadata.py   │──→ Tên VN, câu chuyện, nguyên liệu
│ load_calories.py   │──→ Calories, macro, vi chất
└────────┬───────────┘
         │
         ▼
┌─── Hiển thị ──────┐
│ Kết quả Top-K     │
│ Grad-CAM 3 ảnh    │
│ Bảng dinh dưỡng   │
│ Story + Nguyên    │
│   liệu            │
└────────────────────┘
```

---

## Chiến lược huấn luyện

### Two-Phase Training

| Pha | Epochs | Backbone | Learning Rate | Mục đích |
|---|---|---|---|---|
| **Phase 1** | 1–5 | Đóng băng (Freeze) | 0.001 (head only) | Classifier head hội tụ nhanh |
| **Phase 2** | 6–30 | Mở khóa (Unfreeze) | 1e-5 (backbone) / 1e-4 (head) | Fine-tune toàn bộ |

### Các kỹ thuật nâng cao

| Kỹ thuật | Mô tả | File |
|---|---|---|
| Custom Classifier Head | Dropout(0.3)→Linear(1280,256)→ReLU→Dropout(0.2)→Linear(256,29) | `src/model.py` |
| Discriminative LR | Backbone LR thấp hơn Head LR 10 lần | `src/train.py` |
| CosineAnnealingLR | Giảm LR mượt từ max → 1e-7 theo hàm cosine | `src/train.py` |
| Early Stopping | patience=7 epochs, chỉ kích hoạt ở Phase 2 | `src/train.py` |
| Mixed Precision (AMP) | FP16 forward + FP32 backward, tiết kiệm VRAM | `src/train.py` |
| WeightedRandomSampler | Cân bằng lớp thiểu số bằng trọng số mẫu | `src/preprocess.py` |
| Grad-CAM | Trực quan hóa vùng ảnh mô hình tập trung | `src/model.py` |

### Siêu tham số

| Tham số | Giá trị |
|---|---|
| Batch size | 32 |
| Tổng epoch | 30 (5+25) |
| Optimizer | Adam (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=25, eta_min=1e-7) |
| Input size | 224 × 224 |
| Normalization | ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) |
| Random seed | 42 |

---

## Kết quả

### So sánh mô hình trên tập Test (5,040 ảnh)

| Mô hình | Val Accuracy | Params | Kích thước |
|---|---|---|---|
| **EfficientNet-B0** | **83.87%** | ~5.3M | 50.2 MB |
| ResNet-18 | 74.81% | ~11.2M | 129.6 MB |

EfficientNet-B0 được chọn làm mô hình chính nhờ: accuracy cao hơn, kích thước nhỏ hơn 2.5 lần, và inference nhanh hơn.

---

## Phần cứng huấn luyện

| Thành phần | Chi tiết |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 (4GB VRAM) |
| Framework | PyTorch >= 2.0 |
| Python | 3.14 |
| Mixed Precision | AMP (FP16/FP32) |
| Hệ điều hành | Windows |

---

## Nhóm thực hiện — Nhóm 5

| Thành viên | Nhiệm vụ chính |
|---|---|
| Nguyễn Hoàng Nam | Xây dựng mô hình, huấn luyện, pipeline, phương pháp mới |
| Trần Thị Nhi | Đánh giá mô hình, độ đo, demo kết quả |
| Đỗ Huy Lương | Trích chọn đặc trưng, Transfer Learning |
| Đỗ Minh Tường | Thu thập & tiền xử lý dữ liệu |
| Lê Thị Kim Yến | Lý thuyết nền tảng CNN, tổng hợp báo cáo |

---

## Tài liệu tham khảo

- Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- He, K., et al. (2015). *Deep Residual Learning for Image Recognition.* CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks.* ICCV 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- Howard, J., & Ruder, S. (2018). *Universal Language Model Fine-tuning for Text Classification.* ACL 2018. [arXiv:1801.06146](https://arxiv.org/abs/1801.06146)
- Bộ dữ liệu: [30VNFoods](https://www.kaggle.com/datasets) — 30 món ăn Việt Nam phổ biến

---

## License

Dự án phục vụ mục đích học tập và nghiên cứu.
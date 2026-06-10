# PHÂN CÔNG CÔNG VIỆC — NHÓM 5

> Đề tài: Ứng dụng AI nhận diện món ăn Việt Nam và ước tính dinh dưỡng

| TV | Họ tên | Vai trò chính |
|---|---|---|
| 1 | Đỗ Minh Tường | Thu thập & tiền xử lý dữ liệu |
| 2 | Lê Thị Kim Yến | Lý thuyết nền tảng CNN |
| 3 | Đỗ Huy Lương | ML Engineer chính — Train EfficientNet-B0, xây dựng pipeline |
| 4 | Nguyễn Hoàng Nam | Train ResNet-18 (baseline so sánh) |
| 5 | Trần Thị Nhi | Đánh giá mô hình, chạy evaluate |

---

## Chi tiết

### TV1 — Tường: Dữ liệu
- Thu thập bộ dữ liệu 30VNFoods (29 lớp, ~24,636 ảnh)
- Chia tập Train/Validate/Test (70/10/20)
- Xây dựng cơ sở dữ liệu dinh dưỡng (CSV) từ Viện Dinh dưỡng Quốc gia
- Viết **Chương 2**: Cơ sở lý thuyết + Chuẩn bị dữ liệu

### TV2 — Yến: Lý thuyết
- Nghiên cứu lý thuyết CNN, Transfer Learning, EfficientNet
- Viết **Chương 1**: Giới thiệu bài toán
- Viết **Mục 2.1–2.3**: Cơ sở lý thuyết mạng CNN
- Tổng hợp báo cáo Word, format, mục lục

### TV3 — Lương: ML Engineer chính
- Viết toàn bộ source code: `model.py`, `train.py`, `preprocess.py`, `predict.py`, `evaluate.py`, `app.py`
- Thiết kế kiến trúc Custom Classifier Head
- Huấn luyện EfficientNet-B0 (2-phase fine-tuning, 30 epochs)
- Xây dựng pipeline: Augmentation, WeightedRandomSampler, CosineAnnealingLR, Early Stopping, Mixed Precision, Grad-CAM
- Viết ứng dụng demo Streamlit
- Viết **Mục 3.2**: Các phương pháp xây dựng mô hình
- Viết **Mục 3.4**: Cấu hình huấn luyện

### TV4 — Nam: Train ResNet-18
- Huấn luyện mô hình ResNet-18 làm baseline so sánh
- Viết `train_resnet18.py`
- Viết **Mục 3.3**: Mô hình ResNet-18 (kiến trúc + kết quả)
- Lập bảng so sánh EfficientNet-B0 vs ResNet-18

### TV5 — Nhi: Đánh giá
- Chạy `evaluate.py` với 2 checkpoint (EfficientNet-B0 + ResNet-18)
- Xuất Confusion Matrix, Classification Report, biểu đồ per-class accuracy
- Viết **Mục 4.1**: Độ đo đánh giá (Accuracy, Precision, Recall, F1)
- Viết **Mục 4.2**: Kết quả thực nghiệm + So sánh 2 mô hình
- Viết **Mục 4.3**: Demo sản phẩm (chụp screenshot Streamlit)

---

## Phân công báo cáo

| Mục | Nội dung | Người viết |
|---|---|---|
| Chương 1 | Giới thiệu bài toán | Yến |
| Chương 2 | Cơ sở lý thuyết + Dữ liệu | Tường + Yến |
| 3.1 | Trích chọn đặc trưng (Transfer Learning) | Lương |
| 3.2 | Phương pháp xây dựng mô hình | Lương |
| 3.3 | Mô hình ResNet-18 | Nam |
| 3.4 | Cấu hình huấn luyện | Lương |
| 4.1 | Độ đo đánh giá | Nhi |
| 4.2 | Kết quả + So sánh | Nhi |
| 4.3 | Demo sản phẩm | Nhi |
| Kết luận | Kết luận + Hướng phát triển | Yến |

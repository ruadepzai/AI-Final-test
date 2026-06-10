# 4.3. Demo sản phẩm

## 4.3.1. Công nghệ triển khai

Để triển khai mô hình nhận diện món ăn Việt Nam thành một ứng dụng thực tế, nhóm lựa chọn framework Streamlit — một thư viện Python mã nguồn mở cho phép xây dựng nhanh chóng các ứng dụng web tương tác mà không cần viết mã HTML, CSS hoặc JavaScript. Người dùng chỉ cần viết một file `.py` và chạy lệnh `streamlit run app.py` là có thể khởi chạy ứng dụng trên trình duyệt tại địa chỉ `http://localhost:8501`.

| Ưu điểm | Mô tả |
|:---|:---|
| Dễ triển khai | Chỉ cần viết 1 file `.py` đã có thể triển khai trên localhost |
| Hỗ trợ nhiều thư viện AI | Tích hợp tốt với PyTorch, TorchVision, Matplotlib, Seaborn |
| Giao diện tương tác | Hỗ trợ upload file, slider, checkbox, hiển thị ảnh và biểu đồ |
| Triển khai linh hoạt | Có thể chạy cục bộ hoặc deploy trên cloud |

Bảng: Tổng quan về Streamlit

Nguồn: Nhóm tác giả

Trong giai đoạn triển khai, phiên bản mô hình đã được huấn luyện và lưu trữ tại thời điểm có hiệu suất dự đoán tốt nhất (best validation accuracy = 83.87% tại epoch 28) sẽ được tải lại từ file checkpoint `best_model.pth` để phục vụ suy luận trong ứng dụng.

## 4.3.2. Giao diện demo

Sau khi triển khai mô hình trên Streamlit, nhóm thu được kết quả như sau:

*Hình: Giao diện tổng quan ứng dụng nhận diện món ăn Việt Nam trên Streamlit*

Nguồn: Nhóm tác giả

Giao diện ứng dụng được chia thành hai khu vực chính:

**Thanh bên trái (Sidebar)** chứa các tùy chọn cài đặt cho phép người dùng tùy chỉnh trải nghiệm:
- Khối lượng phần ăn: thanh kéo từ 100g đến 600g (mặc định 350g), dùng để tính toán thông tin dinh dưỡng theo khẩu phần thực tế.
- Số lượng dự đoán hiển thị (Top-K): từ 3 đến 10 (mặc định 5).
- Bật/tắt Grad-CAM: checkbox cho phép hiển thị hoặc ẩn phần giải thích trực quan.
- Bảng thông tin mô hình: kiến trúc EfficientNet-B0, pretrained ImageNet, 29 lớp, val accuracy 83.87%.
- Danh sách nhóm thực hiện.

**Khu vực chính (Content area)** gồm vùng tải ảnh lên (hỗ trợ định dạng JPG, JPEG, PNG, WEBP) hoặc chọn ảnh mẫu từ tập Test. Khi chưa tải ảnh, trang hiển thị danh sách 29 món ăn được hỗ trợ dạng lưới 4 cột.

## 4.3.3. Chức năng nhận diện món ăn

Sau khi người dùng tải ảnh lên hoặc chọn ảnh mẫu, hệ thống tự động thực hiện các bước: tiền xử lý ảnh (giữ tỷ lệ khung hình, resize 224×224, chuẩn hóa ImageNet), đưa qua mô hình EfficientNet-B0, và hiển thị kết quả.

Kết quả được trình bày dạng hai cột: cột trái hiển thị ảnh đầu vào gốc, cột phải hiển thị thẻ kết quả chính gồm tên món ăn bằng tiếng Việt có dấu và độ tin cậy (confidence) với mã màu trực quan (xanh lá khi >70%, vàng cam khi 40–70%, đỏ khi <40%). Bên dưới là danh sách Top-K dự đoán, mỗi dự đoán có thanh tiến trình (progress bar) với chiều dài tỷ lệ thuận với xác suất.

*Hình: Kết quả nhận diện món ăn — Ảnh đầu vào bên trái, thẻ kết quả và Top-5 dự đoán với thanh tiến trình bên phải*

Nguồn: Nhóm tác giả

## 4.3.4. Chức năng Grad-CAM — Giải thích trực quan

Khi checkbox "Hiển thị Grad-CAM" được bật, ứng dụng hiển thị phần giải thích trực quan bằng kỹ thuật Gradient-weighted Class Activation Mapping (Grad-CAM). Phần này trình bày ba hình ảnh song song trên cùng một hàng:

*Hình: Grad-CAM visualization — Ảnh gốc (trái), Heatmap (giữa) và Overlay (phải)*

Nguồn: Nhóm tác giả

Trong đó:
- **Ảnh gốc (224×224):** Ảnh đầu vào đã được resize về kích thước chuẩn.
- **Heatmap:** Bản đồ nhiệt sử dụng bảng màu Jet, vùng màu đỏ/vàng thể hiện các khu vực mà mô hình tập trung chú ý nhiều nhất khi đưa ra dự đoán, vùng xanh dương thể hiện các khu vực ít ảnh hưởng.
- **Overlay:** Kết quả phủ heatmap lên ảnh gốc, cho phép người dùng trực quan thấy mối liên hệ giữa vùng ảnh thực tế và mức độ chú ý của mô hình.

Ví dụ, với món Phở, mô hình tập trung vào sợi phở và bề mặt nước dùng thay vì viền bát hoặc đũa — chứng minh mô hình đã học đúng các đặc trưng phân biệt. Grad-CAM giúp tăng tính minh bạch và khả năng giải thích (explainability) của mô hình.

## 4.3.5. Chức năng tra cứu thông tin dinh dưỡng

Sau khi nhận diện thành công, ứng dụng tự động tra cứu và hiển thị thông tin dinh dưỡng chi tiết của món ăn, tính theo khẩu phần mà người dùng đã cài đặt trên Sidebar.

*Hình: Thông tin dinh dưỡng — 4 thẻ macro (Calories, Protein, Carbs, Fat) và bảng vi chất chi tiết*

Nguồn: Nhóm tác giả

Thông tin được trình bày theo hai cấp độ:

**Thông tin dinh dưỡng vĩ mô (Macro nutrients):** Hiển thị dạng 4 thẻ metric nổi bật gồm Calories (kcal) kèm giá trị trên 100g, Protein (g), Carbohydrate (g) và Fat (g).

**Thông tin vi chất dinh dưỡng (Micro nutrients):** Ẩn trong phần mở rộng, hiển thị dạng hai bảng song song gồm 10 vi chất: Chất xơ, Cholesterol, Canxi, Sắt, Phospho, Natri, Kali, Vitamin A, Vitamin B1 và Vitamin C.

Dữ liệu dinh dưỡng được xây dựng dựa trên nguồn Viện Dinh dưỡng Quốc gia Việt Nam, giá trị được tính toán tỷ lệ tuyến tính theo khẩu phần người dùng chọn.

## 4.3.6. Chức năng giới thiệu món ăn

Phần cuối cùng hiển thị hai nội dung bổ sung được trình bày song song dạng hai cột:

*Hình: Câu chuyện món ăn (trái) và nguyên liệu chính (phải)*

Nguồn: Nhóm tác giả

**Cột trái — Câu chuyện món ăn:** Giới thiệu nguồn gốc, lịch sử hình thành, vùng miền đặc trưng và ý nghĩa văn hóa ẩm thực. Ví dụ: Phở có nguồn gốc từ Nam Định đầu thế kỷ 20, Cao lầu là đặc sản riêng có của phố cổ Hội An.

**Cột phải — Nguyên liệu chính:** Liệt kê danh sách các nguyên liệu cần thiết để chế biến món ăn, giúp người dùng có thể tham khảo.

Toàn bộ dữ liệu metadata cho 29 món ăn được quản lý tập trung trong file `food_metadata.py`. Tính năng này biến ứng dụng không chỉ là một công cụ nhận diện hình ảnh, mà còn mang giá trị giáo dục và quảng bá văn hóa ẩm thực Việt Nam.

## 4.3.7. Tổng hợp chức năng sản phẩm

| STT | Chức năng | Mô tả |
|:---:|:---|:---|
| 1 | Nhận diện món ăn | Phân loại ảnh thành 1 trong 29 lớp, hiển thị Top-K dự đoán với thanh tiến trình |
| 2 | Grad-CAM | Trực quan hóa vùng ảnh mô hình tập trung bằng 3 ảnh song song (Gốc, Heatmap, Overlay) |
| 3 | Tra cứu dinh dưỡng | Calories, Protein, Carbs, Fat và 10 vi chất, tính theo khẩu phần tùy chỉnh |
| 4 | Giới thiệu món ăn | Câu chuyện nguồn gốc, lịch sử văn hóa và nguyên liệu chế biến |
| 5 | Danh sách 29 món | Hiển thị tên tiếng Việt có dấu dạng lưới 4 cột |

Bảng: Tổng hợp các chức năng chính của ứng dụng demo

Nguồn: Nhóm tác giả

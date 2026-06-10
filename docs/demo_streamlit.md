# Demo ứng dụng Streamlit — Nhận diện món ăn Việt Nam

## Giới thiệu

Sản phẩm cuối cùng của dự án là một ứng dụng web demo được xây dựng bằng framework Streamlit (file `app.py`), cho phép người dùng tải lên ảnh món ăn Việt Nam và nhận kết quả nhận diện cùng các thông tin bổ sung trong vài giây. Ứng dụng được thiết kế với giao diện đơn giản, thân thiện, phù hợp với người dùng phổ thông không cần kiến thức kỹ thuật.

Để khởi chạy ứng dụng, người dùng thực hiện lệnh sau trong terminal:

```
streamlit run app.py
```

Sau đó truy cập địa chỉ `http://localhost:8501` trên trình duyệt.

---

## Giao diện tổng quan

Giao diện ứng dụng được chia thành hai khu vực chính:

**Thanh bên trái (Sidebar):** Chứa các tùy chọn cài đặt cho phép người dùng tùy chỉnh trải nghiệm:
- Khối lượng phần ăn (gram): thanh kéo từ 100g đến 600g, mặc định 350g — dùng để tính toán thông tin dinh dưỡng theo khẩu phần thực tế.
- Số lượng dự đoán hiển thị (Top-K): từ 3 đến 10, mặc định 5 — số lượng kết quả phân loại được hiển thị theo thứ tự xác suất giảm dần.
- Bật/tắt Grad-CAM: checkbox cho phép hiển thị hoặc ẩn phần giải thích trực quan.
- Thông tin mô hình: bảng tóm tắt kiến trúc EfficientNet-B0, pretrained ImageNet, 29 lớp, val accuracy 83.87%, chiến lược 2-phase fine-tune.
- Nhóm thực hiện: danh sách 5 thành viên nhóm và nhiệm vụ chính.

**Khu vực chính (Content area):** Gồm vùng tải ảnh lên (Upload) hoặc chọn ảnh mẫu có sẵn từ tập Test, và toàn bộ kết quả phân tích hiển thị bên dưới.

> *Hình: Giao diện tổng quan ứng dụng khi mới mở — Sidebar bên trái chứa cài đặt, khu vực chính hiển thị vùng upload và danh sách 29 món ăn được hỗ trợ*

---

## Chức năng 1: Nhận diện món ăn

Đây là chức năng cốt lõi của ứng dụng. Sau khi người dùng tải ảnh lên (hỗ trợ định dạng JPG, JPEG, PNG, WEBP) hoặc chọn một ảnh mẫu từ tập Test, hệ thống tự động thực hiện:

1. Tiền xử lý ảnh đầu vào: giữ nguyên tỷ lệ khung hình bằng kỹ thuật FixedAspectRatioPadding, resize về kích thước 224×224 pixel bằng nội suy LANCZOS, chuẩn hóa theo giá trị ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
2. Đưa tensor ảnh qua mô hình EfficientNet-B0 đã huấn luyện (checkpoint `best_model.pth`) để tính xác suất cho 29 lớp.
3. Hiển thị kết quả gồm hai phần:
   - Thẻ kết quả chính: tên món ăn bằng tiếng Việt có dấu và độ tin cậy (confidence) dạng phần trăm, với mã màu trực quan: xanh lá (>70%), vàng cam (40–70%), đỏ (<40%).
   - Danh sách Top-K dự đoán: mỗi dự đoán hiển thị tên món, thanh tiến trình (progress bar) với chiều dài tỷ lệ thuận với xác suất, và giá trị phần trăm cụ thể.

Bố cục hiển thị dạng hai cột: cột trái là ảnh đầu vào gốc, cột phải là thẻ kết quả và Top-K.

> *Hình: Kết quả nhận diện món Phở — Ảnh đầu vào bên trái, thẻ kết quả và Top-5 dự đoán với thanh tiến trình bên phải*

---

## Chức năng 2: Grad-CAM — Giải thích trực quan quyết định của mô hình

Ngay bên dưới kết quả nhận diện (khi checkbox Grad-CAM được bật), ứng dụng hiển thị phần giải thích trực quan bằng kỹ thuật Gradient-weighted Class Activation Mapping (Grad-CAM). Phần này trình bày ba hình ảnh song song trên cùng một hàng:

- **Ảnh gốc (224×224):** Ảnh đầu vào đã được resize về kích thước chuẩn mà mô hình sử dụng.
- **Heatmap:** Bản đồ nhiệt Grad-CAM sử dụng bảng màu Jet (xanh dương → xanh lá → vàng → đỏ), trong đó vùng màu đỏ/vàng thể hiện các khu vực mà mô hình tập trung chú ý nhiều nhất khi đưa ra dự đoán, còn vùng xanh dương thể hiện các khu vực ít ảnh hưởng đến quyết định.
- **Overlay:** Kết quả phủ heatmap lên ảnh gốc với độ trong suốt (alpha blending), cho phép người dùng trực quan thấy mối liên hệ giữa vùng ảnh thực tế và mức độ chú ý của mô hình.

Ý nghĩa thực tiễn: Ví dụ với món Phở, mô hình tập trung vào sợi phở và bề mặt nước dùng thay vì viền bát hoặc đũa, chứng minh mô hình đã học được đúng các đặc trưng phân biệt của món ăn. Grad-CAM giúp tăng tính minh bạch (explainability) của mô hình, cho phép người dùng hiểu và tin tưởng kết quả dự đoán.

> *Hình: Grad-CAM visualization cho món Phở bò — Ảnh gốc (trái), Heatmap (giữa) và Overlay (phải). Vùng đỏ tập trung vào sợi phở và nước dùng*

---

## Chức năng 3: Tra cứu thông tin dinh dưỡng

Sau khi nhận diện thành công, ứng dụng tự động tra cứu và hiển thị thông tin dinh dưỡng chi tiết của món ăn, tính theo khẩu phần mà người dùng đã cài đặt trên Sidebar. Thông tin được trình bày theo hai cấp độ:

**Thông tin dinh dưỡng vĩ mô (Macro nutrients):** Hiển thị dạng 4 thẻ metric nổi bật trên cùng một hàng:
- Calories (kcal) — kèm giá trị trên 100g để tham chiếu
- Protein (g)
- Carbohydrate (g)
- Fat (g)

**Thông tin vi chất dinh dưỡng (Micro nutrients):** Ẩn trong phần mở rộng (expander) "Vi chất dinh dưỡng chi tiết", hiển thị dạng hai bảng song song gồm 10 vi chất:
- Bảng trái: Chất xơ (g), Cholesterol (mg), Canxi (mg), Sắt (mg), Phospho (mg)
- Bảng phải: Natri (mg), Kali (mg), Vitamin A (mcg), Vitamin B1 (mg), Vitamin C (mg)

Dữ liệu dinh dưỡng được tra cứu từ cơ sở dữ liệu xây dựng dựa trên nguồn Viện Dinh dưỡng Quốc gia Việt Nam, lưu trong các file CSV của dự án. Giá trị được tính toán tỷ lệ tuyến tính theo khẩu phần: nếu dữ liệu gốc cho 100g thì khẩu phần 350g sẽ nhân với 3.5.

> *Hình: Thông tin dinh dưỡng của món Bún bò Huế — 4 thẻ macro (Calories, Protein, Carbs, Fat) và bảng vi chất chi tiết*

---

## Chức năng 4: Giới thiệu câu chuyện và nguyên liệu món ăn

Phần cuối cùng của kết quả hiển thị hai nội dung bổ sung được trình bày song song dạng hai cột:

**Cột trái — Câu chuyện món ăn:** Giới thiệu nguồn gốc, lịch sử hình thành, vùng miền đặc trưng và ý nghĩa văn hóa ẩm thực của món ăn. Ví dụ: Phở có nguồn gốc từ Nam Định đầu thế kỷ 20, Cao lầu là đặc sản riêng có của phố cổ Hội An với nguồn nước giếng Bá Lễ đặc trưng, Bánh xèo miền Nam khác biệt với bánh khoái miền Trung về kích thước và nguyên liệu.

**Cột phải — Nguyên liệu chính:** Liệt kê danh sách các nguyên liệu cần thiết để chế biến món ăn, giúp người dùng có thể tham khảo hoặc thử nấu tại nhà.

Toàn bộ dữ liệu metadata (tên tiếng Việt có dấu, câu chuyện, nguyên liệu) cho 29 món ăn được quản lý tập trung trong file `src/food_metadata.py`.

Tính năng này biến ứng dụng không chỉ là một công cụ nhận diện hình ảnh đơn thuần, mà còn mang giá trị giáo dục và quảng bá văn hóa ẩm thực Việt Nam.

> *Hình: Câu chuyện (trái) và nguyên liệu chính (phải) của món Bánh mì*

---

## Chức năng 5: Hiển thị danh sách món ăn được hỗ trợ

Khi người dùng chưa tải ảnh lên, trang chính hiển thị thông báo hướng dẫn sử dụng kèm danh sách đầy đủ 29 món ăn Việt Nam mà hệ thống hỗ trợ nhận diện. Danh sách được trình bày dạng lưới 4 cột với tên tiếng Việt có dấu, giúp người dùng biết trước phạm vi hoạt động của ứng dụng.

> *Hình: Trang chủ khi chưa upload ảnh — hiển thị hướng dẫn và danh sách 29 món ăn*

---

## Tổng kết các chức năng

| STT | Chức năng | Mô tả | Dữ liệu nguồn |
|:---:|:---|:---|:---|
| 1 | Nhận diện món ăn | Phân loại ảnh thành 1 trong 29 lớp, hiển thị Top-K với thanh tiến trình | Mô hình EfficientNet-B0 (`best_model.pth`) |
| 2 | Grad-CAM | Trực quan hóa vùng ảnh mô hình tập trung, 3 ảnh song song | Hook vào layer cuối backbone |
| 3 | Tra cứu dinh dưỡng | Calories, macro (Protein/Carbs/Fat), 10 vi chất, theo khẩu phần tùy chỉnh | Viện Dinh dưỡng Quốc gia VN (CSV) |
| 4 | Câu chuyện + Nguyên liệu | Nguồn gốc, lịch sử, văn hóa và nguyên liệu chế biến | `food_metadata.py` |
| 5 | Danh sách 29 món | Grid 4 cột hiển thị tên tiếng Việt có dấu | Thư mục `data/raw/Images/Train` |

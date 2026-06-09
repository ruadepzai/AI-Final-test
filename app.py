"""
Demo nhan dien mon an Viet Nam bang Streamlit.

Giao dien cho phep:
- Upload anh mon an
- Nhan dien mon an + do tin cay
- Hien thi thong tin dinh duong chi tiet
- Grad-CAM: AI nhin vao dau de nhan dien
- Top-5 du doan

Chay:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Them project root vao sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

from src.predict import FoodPredictor
from src.model import GradCAM, get_target_layer


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="VN Food AI - Nhận diện món ăn Việt Nam",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Header */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #ff6b35, #f7c948);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p { margin: 0.3rem 0 0 0; font-size: 1rem; opacity: 0.9; }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .result-card h2 { margin: 0; font-size: 1.8rem; }
    .result-card .confidence { font-size: 2.5rem; font-weight: bold; }

    /* Nutrition card */
    .nutrition-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #2ecc71;
    }

    /* Metric card */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .metric-card .value { font-size: 1.5rem; font-weight: bold; color: #2c3e50; }
    .metric-card .label { font-size: 0.8rem; color: #7f8c8d; }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #95a5a6;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# LOAD MODEL (CACHED)
# ============================================================================

@st.cache_resource
def load_predictor():
    """Load model 1 lan, cache cho cac lan sau."""
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "best_model.pth"
    if not checkpoint_path.exists():
        st.error(f"❌ Khong tim thay checkpoint: {checkpoint_path}")
        st.stop()
    return FoodPredictor(str(checkpoint_path), device="auto")


# ============================================================================
# GRAD-CAM
# ============================================================================

def generate_gradcam_overlay(predictor, img_pil):
    """Tao Grad-CAM overlay cho 1 PIL Image."""
    model = predictor.model
    model_name = predictor.model_name
    transform = predictor.transform

    # Prepare input
    input_tensor = transform(img_pil.convert("RGB")).unsqueeze(0)
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Grad-CAM
    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)

    try:
        with torch.enable_grad():
            heatmap = gradcam.generate(input_tensor)
    finally:
        gradcam.remove_hooks()

    # Overlay
    img_resized = img_pil.resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    heatmap_colored = cm.jet(heatmap)[:, :, :3]
    overlay = 0.5 * img_array + 0.5 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)

    return heatmap, overlay


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # --- Header ---
    st.markdown("""
    <div class="main-header">
        <h1>🍜 Nhận diện Món ăn Việt Nam</h1>
        <p>Ứng dụng AI sử dụng EfficientNetB0 — Nhận diện 29 món ăn + Thông tin dinh dưỡng</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        serving_grams = st.slider(
            "🍽️ Khối lượng phần ăn (gram)",
            min_value=100, max_value=600, value=350, step=50,
        )
        top_k = st.slider("🔝 Số lượng dự đoán hiển thị", 3, 10, 5)
        show_gradcam = st.checkbox("🔥 Hiển thị Grad-CAM", value=True)

        st.markdown("---")
        st.markdown("### 📊 Thông tin mô hình")
        st.markdown("""
        | | |
        |---|---|
        | **Kiến trúc** | EfficientNetB0 |
        | **Pretrained** | ImageNet |
        | **Số class** | 29 món ăn |
        | **Val Accuracy** | 83.87% |
        | **Training** | 2-phase fine-tune |
        """)

        st.markdown("---")
        st.markdown("### 👥 Nhóm thực hiện")
        st.markdown("""
        - **TV1** Hương — Demo + Word
        - **TV2** Huyền — Data Engineer
        - **TV3** Lương — ML Engineer
        - **TV4** Nam — Train ResNet-18
        - **TV5** Nhi — Evaluator
        """)

    # --- Load model ---
    predictor = load_predictor()

    # --- Upload ---
    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "📸 Tải ảnh món ăn lên",
            type=["jpg", "jpeg", "png", "webp"],
            help="Chụp hoặc chọn ảnh món ăn Việt Nam",
        )

    with col_sample:
        st.markdown("**🖼️ Hoặc dùng ảnh mẫu:**")
        sample_dir = PROJECT_ROOT / "data" / "raw" / "Images" / "Test"
        if sample_dir.exists():
            sample_classes = sorted([d.name for d in sample_dir.iterdir() if d.is_dir()])[:5]
            sample_choice = st.selectbox("Chọn món:", sample_classes)
            if st.button("Dùng ảnh mẫu"):
                sample_folder = sample_dir / sample_choice
                sample_images = list(sample_folder.glob("*.jpg")) + list(sample_folder.glob("*.png"))
                if sample_images:
                    uploaded_file = sample_images[0]

    # --- Process ---
    if uploaded_file is not None:
        # Load image
        if isinstance(uploaded_file, Path):
            img = Image.open(uploaded_file).convert("RGB")
        else:
            img = Image.open(uploaded_file).convert("RGB")

        st.markdown("---")

        # Predict
        with st.spinner("🔍 AI đang nhận diện..."):
            result = predictor.predict(img, top_k=top_k, serving_grams=serving_grams)

        # ===== RESULTS =====
        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.image(img, caption="📸 Ảnh đầu vào", use_container_width=True)

        with col_result:
            # Main result
            confidence = result["confidence"]
            color = "#2ecc71" if confidence > 0.7 else "#f39c12" if confidence > 0.4 else "#e74c3c"

            st.markdown(f"""
            <div class="result-card">
                <p>🍜 Kết quả nhận diện</p>
                <h2>{result['display_name']}</h2>
                <p class="confidence" style="color: {color}">{confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

            # Top-K predictions
            st.markdown("**🔝 Top dự đoán:**")
            for _, display, prob in result["top_k"]:
                pct = prob * 100
                st.markdown(f"""
                <div style="margin: 4px 0;">
                    <span style="display:inline-block; width:120px; font-size:0.9rem;">{display}</span>
                    <span style="display:inline-block; width:200px; background:#ecf0f1; border-radius:4px; height:18px;">
                        <span style="display:inline-block; width:{pct}%; background:{color}; border-radius:4px; height:18px;"></span>
                    </span>
                    <span style="font-size:0.85rem; margin-left:8px;">{prob:.1%}</span>
                </div>
                """, unsafe_allow_html=True)

        # ===== GRAD-CAM =====
        if show_gradcam:
            st.markdown("---")
            st.markdown("### 🔥 Grad-CAM — AI nhìn vào đâu?")
            st.caption("Vùng màu đỏ/vàng = nơi AI tập trung để nhận diện món ăn")

            with st.spinner("Đang tạo Grad-CAM..."):
                heatmap, overlay = generate_gradcam_overlay(predictor, img)

            cam_col1, cam_col2, cam_col3 = st.columns(3)
            with cam_col1:
                st.image(img.resize((224, 224)), caption="Ảnh gốc", use_container_width=True)
            with cam_col2:
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(heatmap, cmap="jet")
                ax.axis("off")
                ax.set_title("Heatmap", fontsize=10)
                st.pyplot(fig, use_container_width=True)
                plt.close()
            with cam_col3:
                st.image(overlay, caption="Overlay", use_container_width=True, clamp=True)

        # ===== NUTRITION =====
        st.markdown("---")
        st.markdown(f"### 🥗 Thông tin dinh dưỡng — {result['display_name']}")
        st.caption(f"Tính cho khẩu phần {serving_grams}g | Nguồn: Viện Dinh dưỡng Quốc gia Việt Nam")

        nutrition = result["nutrition"]

        # Macro nutrients
        n_col1, n_col2, n_col3, n_col4 = st.columns(4)
        with n_col1:
            st.metric("🔥 Calories", f"{nutrition['serving_kcal']} kcal",
                      f"{nutrition['kcal_per_100g']} /100g")
        with n_col2:
            st.metric("🥩 Protein", f"{nutrition['protein_g']}g")
        with n_col3:
            st.metric("🍚 Carbs", f"{nutrition['carb_g']}g")
        with n_col4:
            st.metric("🧈 Fat", f"{nutrition['fat_g']}g")

        # Micro nutrients
        with st.expander("📋 Vi chất dinh dưỡng chi tiết"):
            micro_col1, micro_col2 = st.columns(2)
            with micro_col1:
                st.markdown(f"""
                | Vi chất | Hàm lượng |
                |---|---|
                | 🥬 Chất xơ | {nutrition.get('fiber_g', 0)}g |
                | 💊 Cholesterol | {nutrition.get('cholesterol_mg', 0)}mg |
                | 🦴 Canxi | {nutrition.get('calcium_mg', 0)}mg |
                | 🔴 Sắt | {nutrition.get('iron_mg', 0)}mg |
                | 🧪 Phospho | {nutrition.get('phosphorus_mg', 0)}mg |
                """)
            with micro_col2:
                st.markdown(f"""
                | Vi chất | Hàm lượng |
                |---|---|
                | 🧂 Natri | {nutrition.get('sodium_mg', 0)}mg |
                | 🍌 Kali | {nutrition.get('potassium_mg', 0)}mg |
                | 🥕 Vitamin A | {nutrition.get('vitamin_a_mcg', 0)}μg |
                | 🌾 Vitamin B1 | {nutrition.get('vitamin_b1_mg', 0)}mg |
                | 🍊 Vitamin C | {nutrition.get('vitamin_c_mg', 0)}mg |
                """)

        # ===== STORY & INGREDIENTS =====
        if result.get("story"):
            st.markdown("---")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown(f"### 📖 Câu chuyện món ăn")
                st.markdown(result["story"])
            with col_s2:
                if result.get("ingredients"):
                    st.markdown(f"### 🧑‍🍳 Nguyên liệu chính")
                    st.markdown(result["ingredients"])

    else:
        # Khi chua upload
        st.markdown("---")
        st.info("👆 Tải ảnh món ăn lên hoặc chọn ảnh mẫu để bắt đầu nhận diện!")

        st.markdown("### 🍜 29 món ăn Việt Nam được hỗ trợ:")
        from src.food_metadata import get_display_name
        data_dir = PROJECT_ROOT / "data" / "raw" / "Images" / "Train"
        if data_dir.exists():
            class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
            display_names = [get_display_name(name) for name in class_names]

            # Hien thi 4 cot
            cols = st.columns(4)
            for i, name in enumerate(display_names):
                with cols[i % 4]:
                    st.markdown(f"- {name}")

    # --- Footer ---
    st.markdown("""
    <div class="footer">
        🍜 VN Food AI — Bài tập lớn Trí tuệ Nhân tạo<br>
        Mô hình: EfficientNetB0 | Dataset: 29 món ăn Việt Nam | Nguồn dinh dưỡng: Viện Dinh dưỡng Quốc gia
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

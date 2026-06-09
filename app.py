"""
Demo nhan dien mon an Viet Nam bang Streamlit.

Giao dien cho phep:
- Upload anh mon an
- Nhan dien mon an + do tin cay
- Hien thi thong tin dinh duong chi tiet
- Grad-CAM: mo hinh tap trung vao vung nao de nhan dien
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
    page_title="Nhan dien mon an Viet Nam",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.2rem 0;
        background: linear-gradient(135deg, #d35400, #e67e22);
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p { margin: 0.3rem 0 0 0; font-size: 0.95rem; opacity: 0.9; }

    .result-card {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        border-radius: 10px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .result-card h2 { margin: 0; font-size: 1.6rem; }
    .result-card .confidence { font-size: 2.2rem; font-weight: bold; }

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
        st.error(f"Khong tim thay checkpoint: {checkpoint_path}")
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

    input_tensor = transform(img_pil.convert("RGB")).unsqueeze(0)
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)

    try:
        with torch.enable_grad():
            heatmap = gradcam.generate(input_tensor)
    finally:
        gradcam.remove_hooks()

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
        <h1>Nhan dien Mon an Viet Nam</h1>
        <p>Ung dung su dung EfficientNetB0 — Nhan dien 29 mon an va tra cuu dinh duong</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.header("Cai dat")
        serving_grams = st.slider(
            "Khoi luong phan an (gram)",
            min_value=100, max_value=600, value=350, step=50,
        )
        top_k = st.slider("So luong du doan hien thi", 3, 10, 5)
        show_gradcam = st.checkbox("Hien thi Grad-CAM", value=True)

        st.markdown("---")
        st.markdown("### Thong tin mo hinh")
        st.markdown("""
        | | |
        |---|---|
        | **Kien truc** | EfficientNetB0 |
        | **Pretrained** | ImageNet |
        | **So class** | 29 mon an |
        | **Val Accuracy** | 83.87% |
        | **Training** | 2-phase fine-tune |
        """)

        st.markdown("---")
        st.markdown("### Nhom thuc hien")
        st.markdown("""
        - **TV1** Huong — Demo + Word
        - **TV2** Huyen — Data Engineer
        - **TV3** Luong — ML Engineer
        - **TV4** Nam — Train ResNet-18
        - **TV5** Nhi — Evaluator
        """)

    # --- Load model ---
    predictor = load_predictor()

    # --- Upload ---
    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Tai anh mon an len",
            type=["jpg", "jpeg", "png", "webp"],
            help="Chup hoac chon anh mon an Viet Nam",
        )

    with col_sample:
        st.markdown("**Hoac dung anh mau:**")
        sample_dir = PROJECT_ROOT / "data" / "raw" / "Images" / "Test"
        if sample_dir.exists():
            sample_classes = sorted([d.name for d in sample_dir.iterdir() if d.is_dir()])[:5]
            sample_choice = st.selectbox("Chon mon:", sample_classes)
            if st.button("Dung anh mau"):
                sample_folder = sample_dir / sample_choice
                sample_images = list(sample_folder.glob("*.jpg")) + list(sample_folder.glob("*.png"))
                if sample_images:
                    uploaded_file = sample_images[0]

    # --- Process ---
    if uploaded_file is not None:
        if isinstance(uploaded_file, Path):
            img = Image.open(uploaded_file).convert("RGB")
        else:
            img = Image.open(uploaded_file).convert("RGB")

        st.markdown("---")

        with st.spinner("Dang nhan dien..."):
            result = predictor.predict(img, top_k=top_k, serving_grams=serving_grams)

        # ===== RESULTS =====
        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.image(img, caption="Anh dau vao", use_container_width=True)

        with col_result:
            confidence = result["confidence"]
            color = "#27ae60" if confidence > 0.7 else "#f39c12" if confidence > 0.4 else "#e74c3c"

            st.markdown(f"""
            <div class="result-card">
                <p>Ket qua nhan dien</p>
                <h2>{result['display_name']}</h2>
                <p class="confidence" style="color: {color}">{confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Top du doan:**")
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
            st.markdown("### Grad-CAM — Vung anh mo hinh tap trung")
            st.caption("Vung mau do/vang cho thay noi mo hinh tap trung de nhan dien mon an")

            with st.spinner("Dang tao Grad-CAM..."):
                heatmap, overlay = generate_gradcam_overlay(predictor, img)

            cam_col1, cam_col2, cam_col3 = st.columns(3)
            with cam_col1:
                st.image(img.resize((224, 224)), caption="Anh goc", use_container_width=True)
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
        st.markdown(f"### Thong tin dinh duong — {result['display_name']}")
        st.caption(f"Tinh cho khau phan {serving_grams}g | Nguon: Vien Dinh duong Quoc gia Viet Nam")

        nutrition = result["nutrition"]

        n_col1, n_col2, n_col3, n_col4 = st.columns(4)
        with n_col1:
            st.metric("Calories", f"{nutrition['serving_kcal']} kcal",
                      f"{nutrition['kcal_per_100g']} /100g")
        with n_col2:
            st.metric("Protein", f"{nutrition['protein_g']}g")
        with n_col3:
            st.metric("Carbs", f"{nutrition['carb_g']}g")
        with n_col4:
            st.metric("Fat", f"{nutrition['fat_g']}g")

        with st.expander("Vi chat dinh duong chi tiet"):
            micro_col1, micro_col2 = st.columns(2)
            with micro_col1:
                st.markdown(f"""
                | Vi chat | Ham luong |
                |---|---|
                | Chat xo | {nutrition.get('fiber_g', 0)}g |
                | Cholesterol | {nutrition.get('cholesterol_mg', 0)}mg |
                | Canxi | {nutrition.get('calcium_mg', 0)}mg |
                | Sat | {nutrition.get('iron_mg', 0)}mg |
                | Phospho | {nutrition.get('phosphorus_mg', 0)}mg |
                """)
            with micro_col2:
                st.markdown(f"""
                | Vi chat | Ham luong |
                |---|---|
                | Natri | {nutrition.get('sodium_mg', 0)}mg |
                | Kali | {nutrition.get('potassium_mg', 0)}mg |
                | Vitamin A | {nutrition.get('vitamin_a_mcg', 0)}mcg |
                | Vitamin B1 | {nutrition.get('vitamin_b1_mg', 0)}mg |
                | Vitamin C | {nutrition.get('vitamin_c_mg', 0)}mg |
                """)

        # ===== STORY & INGREDIENTS =====
        if result.get("story"):
            st.markdown("---")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("### Cau chuyen mon an")
                st.markdown(result["story"])
            with col_s2:
                if result.get("ingredients"):
                    st.markdown("### Nguyen lieu chinh")
                    st.markdown(result["ingredients"])

    else:
        st.markdown("---")
        st.info("Tai anh mon an len hoac chon anh mau de bat dau nhan dien.")

        st.markdown("### 29 mon an Viet Nam duoc ho tro:")
        from src.food_metadata import get_display_name
        data_dir = PROJECT_ROOT / "data" / "raw" / "Images" / "Train"
        if data_dir.exists():
            class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
            display_names = [get_display_name(name) for name in class_names]

            cols = st.columns(4)
            for i, name in enumerate(display_names):
                with cols[i % 4]:
                    st.markdown(f"- {name}")

    # --- Footer ---
    st.markdown("""
    <div class="footer">
        Bai tap lon Tri tue Nhan tao<br>
        Mo hinh: EfficientNetB0 | Dataset: 29 mon an Viet Nam | Nguon dinh duong: Vien Dinh duong Quoc gia
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

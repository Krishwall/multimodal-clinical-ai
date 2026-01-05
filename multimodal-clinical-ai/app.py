import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from demo.utils.load_model import load_fusion_model
from demo.utils.grad_cam import GradCAM, overlay_cam
from demo.utils.saliency import (
    compute_text_saliency,
    merge_wordpieces,
    filter_tokens,
    highlight_text,
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Multimodal Clinical AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <h2 style="margin-bottom:0">Multimodal Clinical Decision Support</h2>
    <p style="color:gray; margin-top:4px">
    Chest X-ray + Radiology Text → Ranked Diagnoses with Explainability
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# Load model (cached)
# --------------------------------------------------
@st.cache_resource
def load_all():
    return load_fusion_model(
        "checkpoints/fusion_model/fusion_layer4_tuned.pt"
    )

model, tokenizer, image_transform, LABELS, device = load_all()

# --------------------------------------------------
# Input Section
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Chest X-ray")
    uploaded_image = st.file_uploader(
        "Upload Chest X-ray",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

with col2:
    st.subheader("Radiology Findings")
    findings = st.text_area(
        "Enter findings",
        height=180,
        placeholder="e.g. Enlarged cardiac silhouette with pulmonary congestion...",
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)
analyze = st.button("Analyze Case", use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# Inference + Explainability
# --------------------------------------------------
if analyze and uploaded_image and findings:

    # ---- Preprocess inputs ----
    image = Image.open(uploaded_image).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    enc = tokenizer(
        findings,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # ---- Forward pass ----
    with torch.no_grad():
        logits = model(image_tensor, input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)

    top2_prob, top2_idx = torch.topk(probs, k=2, dim=1)
    primary_idx = top2_idx[0, 0].item()
    secondary_idx = top2_idx[0, 1].item()

    # --------------------------------------------------
    # Diagnosis Output
    # --------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🩺 Primary Diagnosis")
        st.success(
            f"{LABELS[primary_idx]}  \nConfidence: {top2_prob[0,0]:.2f}"
        )

    with col2:
        st.markdown("### 🔍 Secondary Diagnosis")
        st.info(
            f"{LABELS[secondary_idx]}  \nConfidence: {top2_prob[0,1]:.2f}"
        )

    # --------------------------------------------------
    # Explainability
    # --------------------------------------------------
    st.divider()
    st.markdown("## Explainability")

    col1, col2 = st.columns(2)

    # ---- Grad-CAM ----
    with col1:
        st.markdown("#### Image Evidence (Grad-CAM)")

        gradcam = GradCAM(model, model.image_encoder.layer4)
        cam = gradcam.generate(
            image_tensor,
            input_ids,
            attention_mask,
            class_idx=primary_idx
        )

        overlay = overlay_cam(image_tensor, cam)
        st.image(
            overlay,
            use_column_width=True,
            caption="Regions influencing the primary diagnosis"
        )

    # ---- Text Saliency ----
    with col2:
        st.markdown("#### Text Evidence (Important Terms)")

        saliency, attn_mask = compute_text_saliency(
            model,
            input_ids,
            attention_mask,
            target_class=primary_idx
        )

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        # Clean tokens
        tokens, scores = filter_tokens(tokens, saliency, attn_mask)

        # Merge wordpieces
        tokens, scores = merge_wordpieces(tokens, scores)

        # Highlight text
        html_text = highlight_text(tokens, scores)
        st.markdown(html_text, unsafe_allow_html=True)

# --------------------------------------------------
# Footer / Disclaimer
# --------------------------------------------------
st.divider()
st.caption(
    "⚠️ For educational and research purposes only. "
    "Not intended for clinical use."
)

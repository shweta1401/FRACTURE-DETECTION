"""
Streamlit Application for Fracture Detection AI
- Predicts bone type (EfficientNet-B3)
- Predicts fracture presence (EfficientNet-B0 per bone type)
- Provides Grad-CAM++ visual explanation
- Generates GPT-powered diagnostic summaries
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, efficientnet_b3
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from io import BytesIO
import base64
from fpdf import FPDF  
import re

# Import helper modules
from gradcam_explainability import generate_gradcam_plus, auto_suggest_cam_region
from gpt_summary import generate_gpt_summary

# Page Configuration
st.set_page_config(
    page_title="Fracture Detection AI",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.write("Secrets available:", list(st.secrets.keys()))


# ✅ Secrets check
api_key_available = bool(st.secrets.get("openrouter_api_key", ""))
if api_key_available:
    st.success("🔑 OpenRouter API key loaded successfully.")
else:
    st.warning(
        "⚠️ No OpenRouter API key found. GPT summaries will be disabled until you add it in "
        "Streamlit Cloud → App → Settings → Secrets."
    )

# Class Labels
class_names = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']
fracture_classes = ['No Fracture', 'Fracture']
cam_region_options = ["humeral head", "joint interface", "implant screw zone", "medial epicondyle", "distal radius", "not clearly localized"]

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Bone Classification Model
@st.cache_resource
def load_bone_model():
    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(torch.load("models/best_bone_classifier_effb3.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Load Fracture Detection Model
@st.cache_resource
def load_fracture_model(bone_type):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model_path = f"models/efficientnet_best1_{bone_type}.pt"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Download Heatmap Button
def get_image_download_link(img_np, filename="gradcam_output.png"):
    buffered = BytesIO()
    Image.fromarray(img_np).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/png;base64,{img_str}" download="{filename}">Download Grad-CAM Heatmap</a>'

# --- PDF Helpers ---
def _clean_text(text: str) -> str:
    replacements = {
        "\u00A0": " ", "\u202F": " ", "\u2009": " ", "\u200A": " ", "\u200B": " ",
        "\u200C": " ", "\u200D": " ", "\u2060": " ", "\ufeff": " ", "\u00AD": "-",
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-",
        "“": '"', "”": '"', "’": "'", "‘": "'", "\t": " ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = re.sub(r" {2,}", " ", text)
    text = "".join(ch if (ch == "\n" or ch.isprintable()) else " " for ch in text)
    return text

def _write_wrapped_paragraph(pdf: FPDF, text: str, epw: float, line_h: float) -> None:
    for raw_line in text.split("\n"):
        line = _clean_text(raw_line)
        if line.strip() == "":
            pdf.ln(line_h); pdf.set_x(pdf.l_margin)
            continue

        pdf.set_x(pdf.l_margin)
        xlimit = pdf.l_margin + epw

        for word in line.split(" "):
            chunk = word + " "
            chunk_w = pdf.get_string_width(chunk)
            if pdf.get_x() + chunk_w <= xlimit:
                pdf.cell(chunk_w, line_h, chunk, ln=0)
            else:
                if pdf.get_string_width(word) > epw:
                    for ch in word:
                        ch_w = pdf.get_string_width(ch)
                        if pdf.get_x() + ch_w > xlimit:
                            pdf.ln(line_h); pdf.set_x(pdf.l_margin)
                        pdf.cell(ch_w, line_h, ch, ln=0)
                    pdf.cell(pdf.get_string_width(" "), line_h, " ", ln=0)
                else:
                    pdf.ln(line_h); pdf.set_x(pdf.l_margin)
                    pdf.cell(chunk_w, line_h, chunk, ln=0)
        pdf.ln(line_h)
        pdf.set_x(pdf.l_margin)

def create_pdf(summary_text: str) -> BytesIO:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    try:
        pdf.add_font("DejaVu", "", "fonts/DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=12)
    except Exception:
        pdf.set_font("Helvetica", size=12)
        summary_text = _clean_text(summary_text)
    epw = pdf.w - pdf.l_margin - pdf.r_margin
    line_h = pdf.font_size * 1.6
    _write_wrapped_paragraph(pdf, summary_text, epw, line_h)
    return BytesIO(pdf.output(dest="S"))

# -------------------------
# Streamlit Interface
# -------------------------
tab1, tab2 = st.tabs(["Run", "Info"])

with tab1:
    st.title("Fracture Detection with AI")
    uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])
    enhance = st.checkbox("Enhance contrast (recommended for low-quality scans)")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        if enhance:
            image = ImageEnhance.Contrast(image).enhance(1.5)

        img_tensor = transform(image).unsqueeze(0)
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        # Step 1: Bone Type Prediction
        st.subheader("1️⃣ Bone Type Prediction")
        bone_model = load_bone_model()
        with torch.no_grad():
            bone_output = bone_model(img_tensor)
            _, bone_pred = torch.max(bone_output, 1)
            bone_type = class_names[bone_pred.item()]
        st.success(f"Predicted Bone Type: `{bone_type}`")

        # Step 2: Fracture Detection
        st.subheader("2️⃣ Fracture Detection")
        fracture_model = load_fracture_model(bone_type)
        with torch.no_grad():
            fracture_output = fracture_model(img_tensor)
            _, fracture_pred = torch.max(fracture_output, 1)
            fracture_status = fracture_classes[fracture_pred.item()]
            fracture_prob = torch.softmax(fracture_output, dim=1)[0][1].item()

        risk_level = "High" if fracture_prob > 0.75 else "Moderate" if fracture_prob > 0.5 else "Low"
        st.info(f"Fracture Status: `{fracture_status}`")
        st.metric("Fracture Probability", f"{fracture_prob:.2f}", delta=risk_level)

        # Step 3: Grad-CAM++ Visual Explanation
        st.subheader("3️⃣ Grad-CAM++ Visual Explanation")
        with st.spinner("Generating Grad-CAM++ heatmap..."):
            target_layer = dict(fracture_model.named_children())['features'][4]
            heatmap = generate_gradcam_plus(fracture_model, img_tensor.clone(), fracture_pred.item(), target_layer)
            img_np = np.array(image.resize((224, 224)))
            heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
            col1, col2 = st.columns(2)
            col1.image(image.resize((224, 224)), caption="Original X-ray")
            col2.image(superimposed_img.astype(np.uint8), caption="Grad-CAM++ Heatmap")
            st.markdown(get_image_download_link(superimposed_img.astype(np.uint8)), unsafe_allow_html=True)
            auto_region = auto_suggest_cam_region(heatmap)

        # Step 4: GPT-Powered Diagnostic Summary
        st.subheader("4️⃣ AI-Powered Diagnostic Summary (GPT)")
        with st.expander("Generate AI Summary"):
            cam_region = st.selectbox("Where is the Grad-CAM focused?", cam_region_options, index=cam_region_options.index(auto_region))
            if st.button("Generate GPT Summary"):
                api_key = st.secrets.get("openrouter_api_key", "")
                if not api_key:
                    st.error("⚠️ OpenRouter API key not found. Please set it in Streamlit Cloud → App → Settings → Secrets.")
                else:
                    with st.spinner("Querying GPT via OpenRouter..."):
                        gpt_result = generate_gpt_summary(
                            bone_type, fracture_status, fracture_prob,
                            cam_region, api_key=api_key
                        )
                        st.success("AI-Captioned Summary:")
                        st.markdown(gpt_result)
                        pdf_data = create_pdf(gpt_result)
                        st.download_button("Download PDF Report", data=pdf_data, file_name="fracture_summary.pdf", mime="application/pdf")

with tab2:
    st.title("Model & System Info")
    st.markdown("""
    **Bone Classifier:** EfficientNet-B3  
    **Fracture Detector:** EfficientNet-B0  
    **Explainability:** Grad-CAM++  
    **AI Summary:** GPT-4 via OpenRouter  
    **Input:** 224x224 RGB X-rays  
    **Dataset:** MURA  
    **Frameworks:** PyTorch, OpenCV, Streamlit  
    """)

st.markdown("---")
st.caption("Built by Shweta S. and Aathil Ibrahim | Fracture Detection AI © 2025")

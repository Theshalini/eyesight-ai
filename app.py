import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import Paragraph, Frame, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
import datetime
import re
import matplotlib.pyplot as plt

from recommendation import cnv, dme, drusen, normal

# ================= PAGE =================
st.set_page_config(page_title="EyeSight AI", page_icon="👁️", layout="wide")

# ================= MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Trained_Model.keras")

def model_prediction(path):
    model = load_model()
    img = tf.keras.utils.load_img(path, target_size=(224,224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x,0)
    x = preprocess_input(x)
    p = model.predict(x)
    return np.argmax(p), float(np.max(p))

# ================= RISK LEVEL =================
def risk_level(conf):
    if conf >= 0.90:
        return "HIGH", "#8B1E1E"
    elif conf >= 0.75:
        return "MODERATE", "#C96A00"
    else:
        return "LOW", "#1E7F43"

# ================= FORMAT RECOMMENDATION =================
def format_recommendation(txt):
    txt = txt.replace("**","")
    lines = txt.split("\n")
    clean = []
    for line in lines:
        line=line.strip()
        if not line:
            continue
        if line.endswith(":"):
            clean.append(f"<b>{line}</b>")
        else:
            clean.append(line)
    return clean

# ================= PDF =================
def make_pdf(disease, conf, img_path):

    file_path = "EyeSight_AI_Report.pdf"

    doc = SimpleDocTemplate(
      file_path,
      pagesize=letter,
      rightMargin=40,
      leftMargin=40,
      topMargin=40,
      bottomMargin=40
    )

    story = []
    styles = getSampleStyleSheet()

    # -------- Header --------
    title_style = ParagraphStyle(
       name="title",
       fontSize=20,
       leading=24,
       textColor=colors.HexColor("#0B7285"),
       spaceAfter=12
    )

    story.append(Paragraph("EyeSight AI Diagnostic Report", title_style))
    story.append(Paragraph(
      datetime.datetime.now().strftime("%d %b %Y  %H:%M"),
      styles["Normal"]
    ))
    story.append(Spacer(1, 15))

    # -------- Image --------
    try:
       story.append(Image(img_path, width=240, height=200))
       story.append(Spacer(1, 12))
    except:
       pass

    # -------- Result Text --------
    risk, rc = risk_level(conf)

    story.append(Paragraph(
       f"<b>Prediction:</b> {disease}",
       styles["Heading3"]
    ))
    story.append(Paragraph(
       f"<b>Confidence:</b> {conf*100:.2f}%",
       styles["Normal"]
    ))
    story.append(Paragraph(
       f"<b>Risk Level:</b> {risk}",
       styles["Normal"]
    ))

    story.append(Spacer(1, 12))

    # -------- Small Line Graph --------
    fig, ax = plt.subplots(figsize=(3,1.5))
    ax.plot([0,1], [0, conf*100])
    ax.scatter(1, conf*100)
    ax.set_ylim(0,100)
    ax.set_xticks([])
    ax.set_ylabel("%")
    plt.tight_layout()

    graph_path = "confidence_graph.png"
    plt.savefig(graph_path, dpi=200)
    plt.close()

    story.append(Image(graph_path, width=220, height=90))
    story.append(Spacer(1, 16))

    # -------- Recommendations --------
    desc_map = {"CNV":cnv,"DME":dme,"DRUSEN":drusen,"NORMAL":normal}
    lines = format_recommendation(desc_map[disease])

    story.append(Paragraph(
       "<b>Clinical Notes & Recommendations</b>",
       styles["Heading2"]
    ))
    story.append(Spacer(1, 10))

    body_style = ParagraphStyle(
      name="body",
      fontSize=11,
      leading=15
    )

    for line in lines:
       story.append(Paragraph(line, body_style))
       story.append(Spacer(1, 6))

    story.append(Spacer(1, 20))
    story.append(Paragraph(
       "AI-generated output — Not a medical diagnosis",
       styles["Italic"]
    ))

    # -------- Build PDF --------
    doc.build(story)

    return file_path
# ================= THEME CSS =================
st.markdown("""
<style>
.resultcard{
background:white;
padding:22px;
border-radius:18px;
box-shadow:0 8px 24px rgba(0,0,0,.15);
}
.riskbadge{
padding:6px 14px;
border-radius:12px;
color:white;
font-weight:700;
display:inline-block;
}
.main-title{
font-size:42px;
font-weight:900;
margin-top:-20px;
margin-bottom:5px;
}

.subtitle{
font-size:18px;
font-weight:600;
color:#555;
margin-bottom:18px;
}

.home-card{
background:white;
padding:22px;
border-radius:16px;
box-shadow:0 6px 18px rgba(0,0,0,.12);
}
 
            
.stTabs [data-baseweb="tab-list"] {
    margin-bottom: 50px !important;
    gap: 48px !important;
}


.stTabs [data-baseweb="tab"] {
    font-size: 25px !important;
    font-weight: 900 !important;
    padding: 14px 85px !important;
}

</style>
""",unsafe_allow_html=True)

# ================= TABS =================
tabs = st.tabs(["🏠 Home","🔍 Prediction","📚 Learn","🧠 Model","ℹ️ About"])


# ================= HOME =================
with tabs[0]:

    col_left, col_right = st.columns([2,1], gap="large")

    with col_left:
        st.markdown("""<h1 style='color:#1F4E79;'>👁️EyeSight AI</h1>""", unsafe_allow_html=True)
        st.markdown("### AI-Powered OCT Retinal Disease Detection")

        st.markdown("""
        #### System Features

        • OCT retinal scan classification  
        • CNN deep learning based detection  
        • Identifies **CNV, DME, DRUSEN, NORMAL**  
        • Instant AI prediction  
        • Downloadable PDF report  
        """)

        st.success("👉 Go to Prediction tab to analyze an OCT scan")

    with col_right:
        st.markdown("##")
        st.image(
            "assets/hero_retina.jpg", 
             use_container_width=True
        )
      

# ================= PREDICTION — CLEAN + BETTER GRAPH =================
with tabs[1]:
    
    st.markdown("""<h1 style='color:#1F4E79;'>📂Upload OCT image</h1>""", unsafe_allow_html=True)

    up = st.file_uploader("", type=["jpg","png","jpeg"])

    if up:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as t:
            t.write(up.read())
            path = t.name

        col1, col2 = st.columns([1,1])

        with col1:
            st.image(up, use_container_width=True)

        with col2:

            run = st.button("🔎Run Analysis", use_container_width=True)

            if run:
                idx, conf = model_prediction(path)

                classes = ["CNV","DME","DRUSEN","NORMAL"]
                dis = classes[idx]
                risk, rc = risk_level(conf)

                # ----------- RESULT TEXT (NO EMPTY WHITE CARD) ----------
                st.subheader(f"Prediction: {dis}")
                st.write(f"Confidence: {conf*100:.2f}%")

                st.markdown(
                    f"""
                    <div style="
                        display:inline-block;
                        padding:8px 16px;
                        border-radius:8px;
                        background:{rc};
                        color:white;
                        font-weight:bold;">
                        {risk} RISK
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ----------- BETTER GRAPH  ----------

                fig, ax = plt.subplots(figsize=(4,2))

                ax.plot([0,1],[0,conf*100])
                ax.scatter(1,conf*100)

                ax.set_xlim(0,1)
                ax.set_ylim(0,100)
                ax.set_xticks([])
                ax.set_ylabel("Confidence %")
                ax.set_title("Confidence Trend")

                plt.tight_layout()
                st.pyplot(fig)

                # ----------- RECOMMENDATIONS ----------
                st.markdown("### Clinical Notes & Recommendations")

                with st.expander("View Recommendations"):
                    for line in format_recommendation(
                        {"CNV": cnv, "DME": dme, "DRUSEN": drusen, "NORMAL": normal}[dis]
                    ):
                        st.markdown(line, unsafe_allow_html=True)

                # ----------- PDF ----------
                pdf = make_pdf(dis, conf, path)

                with open(pdf, "rb") as f:
                    st.download_button(
                        "Download Report PDF",
                        f,
                        "EyeSight_AI_Report.pdf",
                        use_container_width=True
                    )

# ================= LEARN TAB — ADVANCED UI =================
with tabs[2]:
    
    st.markdown("""<h1 style='color:#1F4E79;'>📚 Learn — OCT Retinal Disease Guide</h1>""", unsafe_allow_html=True)
    st.markdown(
        "Interactive medical overview of retinal conditions detected by your AI model."
    )

    st.markdown("---")

    # ========= CNV =========
    with st.expander("🔴 CNV — Choroidal Neovascularization", expanded=False):

        c1, c2 = st.columns([2,1])

        with c1:
            st.markdown("""
**Definition**  
Abnormal new blood vessels grow beneath the retina and leak fluid or blood.

**OCT Indicators**
- Sub-retinal fluid pockets  
- Elevated irregular layers  
- Hyper-reflective lesions

**Symptoms**
- Distorted vision  
- Central blind spots  
- Rapid vision change
""")

        with c2:
            st.image("assets/cnv.jpeg", width=250,caption="CNV OCT image reference ")
            st.info("""
**Clinical Notes**
- Often linked to AMD  
- Needs urgent care  
- Vision loss risk high
""")

        st.warning("💉 Common Treatment: Anti-VEGF injections")

    # ========= DME =========
    with st.expander("🟠 DME — Diabetic Macular Edema"):

        c1, c2 = st.columns([2,1])

        with c1:
            st.markdown("""
**Definition**  
Fluid accumulation in macula due to diabetic vessel leakage.

**OCT Indicators**
- Retinal thickening  
- Intraretinal cysts  
- Fluid zones

**Symptoms**
- Blurry center vision  
- Reading difficulty  
- Color dullness
""")

        with c2:
            st.image("assets/dme.jpeg", width=250,caption="DME OCT image reference ")
            st.info("""
**Risk Factors**
- Diabetes duration  
- Poor sugar control  
- Hypertension
""")

        st.warning("🩺 Management: Anti-VEGF + glucose control")

    # ========= DRUSEN =========
    with st.expander("🟡 DRUSEN — Retinal Deposits"):

        c1, c2 = st.columns([2,1])

        with c1:
            st.markdown("""
**Definition**  
Yellow lipid deposits beneath retina linked to degeneration risk.

**OCT Indicators**
- Dome elevations  
- RPE irregularities  
- Sub-retinal deposits

**Symptoms**
- Often silent early  
- Mild blur later
""")

        with c2:
            st.image("assets/drunsen.jpeg", width=250,caption="Drunsen OCT image reference ")
            st.info("""
**Types**
- Hard drusen → lower risk  
- Soft drusen → higher risk
""")

        st.warning("🥦 Management: Monitoring + nutrition")

    # ========= NORMAL =========
    with st.expander("🟢 NORMAL Retina"):

        c1, c2 = st.columns([2,1])

        with c1:
            st.markdown("""
**Definition**  
Healthy retinal layer structure without leakage or deposits.

**OCT Indicators**
- Smooth layered pattern  
- No cysts  
- No fluid pockets

**Vision**
- Stable  
- Clear central focus
""")

        with c2:
            st.image("assets/normal.jpeg", width=250,caption="Normal OCT image reference ")
            st.success("""
**Status**
Healthy scan  
Routine screening only
""")

    # ========= COMPARISON TABLE =========
    st.markdown("---")
    st.markdown("## 📊 Quick Comparison")

    st.table({
        "Condition": ["CNV","DME","DRUSEN","NORMAL"],
        "Fluid": ["Yes","Yes","No","No"],
        "Deposits": ["No","No","Yes","No"],
        "Swelling": ["Sometimes","Yes","No","No"],
        "Urgency": ["High","Moderate","Monitor","None"]
    })

    st.success("👉 Upload an OCT scan in Prediction tab to see AI classification.")

# =========================
# MODEL INFO TAB
# =========================

with tabs[3]:

    st.markdown("""
    <h1 style='color:#1F4E79;'>📊 Model Information & Performance</h1>
    <p style='font-size:15px'>
    This section explains the deep learning model, preprocessing pipeline, and evaluation metrics
    used in the EyeSight AI retinal disease detection system.
    </p>
    <hr>
    """, unsafe_allow_html=True)

    # -----------------------
    # Model Overview
    # -----------------------
    st.markdown("### 🧠 Model Overview")

    st.markdown("""
    The EyeSight AI diagnostic system is built using a **Transfer Learning-based Convolutional Neural Network (CNN)** 
    for multi-class retinal OCT image classification.

    **Key Characteristics:**
    - Multi-class classification (4 classes)
    - Detects: CNV, DME, DRUSEN, NORMAL
    - Lightweight and optimized architecture
    - Probability-based prediction output
    - Designed for medical image analysis
    """)

    st.markdown("---")

    # -----------------------
    # Algorithms & Techniques
    # -----------------------
    st.markdown("### ⚙️ Algorithms & Techniques Used")

    st.markdown("""
    - MobileNetV3 (Pretrained CNN Architecture)
    - Transfer Learning
    - Fine-tuning of final layers
    - Image resizing (224 x 224)
    - Pixel normalization (0–1 scaling)
    - Data augmentation:
        - Rotation
        - Horizontal flip
        - Zoom
    - Softmax activation (multi-class output)
    - Categorical Crossentropy loss
    - Adam optimizer
    """)

    st.markdown("---")

    # -----------------------
    # Feature Engineering Pipeline
    # -----------------------
    st.markdown("### 🔄 Feature Engineering Pipeline")

    st.markdown("""
    1. OCT retinal images are collected and labeled.
    2. Images are resized to fixed dimensions.
    3. Pixel values are normalized.
    4. Data augmentation improves generalization.
    5. Images are passed through MobileNetV3 feature extractor.
    6. Deep features are extracted.
    7. Fully connected layers perform classification.
    8. Softmax layer outputs final probabilities.
    """)

    st.markdown("---")

    # -----------------------
    # Model Training Details
    # -----------------------
    st.markdown("### 🏋️ Model Training Details")

    st.markdown("""
    - Model Type: Convolutional Neural Network (CNN)
    - Base Model: MobileNetV3
    - Training Method: Transfer Learning + Fine-Tuning
    - Loss Function: Categorical Crossentropy
    - Optimizer: Adam
    - Evaluation Metric: Accuracy
    """)

    st.markdown("---")

    # -----------------------
    # Performance Metrics
    # -----------------------
    st.markdown("### 📈 Performance Metrics")

    st.markdown("""
    The model performance is evaluated using:

    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - Confusion Matrix
    - Validation Accuracy
    - Validation Loss
    """)

    st.markdown("---")

    # -----------------------
    # Dataset Preview Section
    # -----------------------
    st.markdown("### 🗂️ Sample Training Dataset (Preview)")

    st.markdown("""
    The dataset consists of labeled OCT retinal images categorized into:

    - CNV
    - DME
    - DRUSEN
    - NORMAL

    Images are organized into class-specific folders and used for supervised training.
    """)

    # Optional: show small preview images if available
    # st.image("assets/dataset_preview.jpg", use_container_width=True)
# ==================================================
# ABOUT
# ==================================================
with tabs[4]:
    st.markdown("""<h1 style='color:#1F4E79;'>ℹ️ About EyeSight AI</h1>""", unsafe_allow_html=True)

    st.markdown("""
    EyeSight AI is an academic deep learning project for
    OCT retinal disease detection.

    🎯 Built for research and student demonstration.

    ⚠️ Not a medical diagnostic system.
    """)

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<center>EyeSight AI — OCT Retinal Deep Learning System</center>",
    unsafe_allow_html=True
)

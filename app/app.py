import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import qrcode

# ---------------- Models Load ----------------
# Note: Ensure these paths are correct in your local environment
@st.cache_resource
def load_all_models():
    # Using try-except to prevent crash if files are missing during testing
    try:
        disease_model = load_model("../models/disease_model.h5")
        class_names = joblib.load("../models/disease_classes.pkl")
        yield_model = joblib.load("../models/yield_model.pkl")
        yield_features = joblib.load("../models/yield_features.pkl")
        return disease_model, class_names, yield_model, yield_features
    except:
        return None, None, None, None

disease_model, class_names, yield_model, yield_features = load_all_models()

# ---------------- Functions ----------------
def predict_disease(image):
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 3)
    preds = disease_model.predict(img)[0]
    top_indices = np.argsort(preds)[::-1][:3]
    top_predictions = [(class_names[i], preds[i]*100) for i in top_indices]
    return top_predictions

def predict_yield(input_dict):
    df = pd.DataFrame([input_dict])
    for col in yield_features:
        if col not in df:
            df[col] = 0
    df = df[yield_features]
    prediction = yield_model.predict(df)[0]
    return round(prediction, 2)

def generate_qr(url):
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img

# ---------------- Professional Glassy UI ----------------
st.set_page_config(page_title="AI Crop Intelligence", layout="wide", page_icon="🌿")

# Custom CSS for Glassmorphism and Shiny Professional Look
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }
    
    /* Glassmorphism Effect for Cards */
    div[data-testid="stVerticalBlock"] > div:has(div.stButton) {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Shiny Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #00b09b, #96c93d);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 12px;
        font-weight: bold;
        transition: 0.3s;
        box-shadow: 0 4px 15px rgba(0, 176, 155, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 176, 155, 0.5);
        background: linear-gradient(45deg, #00c9b1, #a8e04d);
    }

    /* Input Fields */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0px 0px;
        color: white;
        padding: 0px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 176, 155, 0.2) !important;
        border-bottom: 3px solid #00b09b !important;
    }

    /* QR Code & Sidebar Shiny */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 32, 39, 0.8);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.title("🌿 Crop Intelligence")
st.sidebar.markdown("Advanced AI-driven agricultural insights.")
st.sidebar.markdown("---")

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["🔍 Disease Detection", "📈 Yield Prediction", "💡 Smart Tips"])

# ============================
# 🌿 DISEASE DETECTION TAB
# ============================
with tab1:
    st.markdown("## 🌿 Leaf Diagnostic Center")
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        crop_type = st.selectbox("Select Crop", ["Tomato", "Potato", "Wheat", "Other"])
        uploaded_file = st.file_uploader("Upload Leaf Scan", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        with col_b:
            st.image(image, caption="Uploaded Scan", use_container_width=True)
            if st.button("🚀 Analyze Plant Health"):
                if disease_model:
                    top_preds = predict_disease(image)
                    st.subheader("Analysis Results")
                    
                    for disease, conf in top_preds:
                        st.write(f"**{disease}**")
                        st.progress(conf/100)
                        st.caption(f"Confidence: {conf:.2f}%")

                    # CSV Report
                    report = pd.DataFrame({"Crop": [crop_type], "Disease": [top_preds[0][0]], "Conf": [f"{top_preds[0][1]:.2f}%"]})
                    st.download_button("📥 Download Report", report.to_csv(index=False), "report.csv")
                else:
                    st.error("Model files not found. Please check paths.")

# ============================
# 🌾 YIELD PREDICTION TAB
# ============================
with tab2:
    st.markdown("## 🌾 Yield Forecasting")
    
    with st.container():
        c1, c2, c3 = st.columns(3)
        rainfall = c1.number_input("Rainfall (mm)", 0.0, 5000.0, 1000.0)
        pesticides = c2.number_input("Pesticides (tonnes)", 0.0, 100000.0, 100.0)
        temp = c3.number_input("Temp (°C)", -10.0, 60.0, 25.0)

    if st.button("📊 Calculate Expected Yield"):
        if yield_model:
            data = {"average_rain_fall_mm_per_year": rainfall, "pesticides_tonnes": pesticides, "avg_temp": temp}
            prediction = predict_yield(data)
            
            # Big Shiny Metric
            st.markdown(f"""
                <div style="background: rgba(0, 176, 155, 0.1); padding: 20px; border-radius: 15px; border: 1px solid #00b09b; text-align: center;">
                    <h2 style="color: #00b09b; margin: 0;">{prediction} Tonnes</h2>
                    <p style="margin: 0;">Predicted Yield per Hectare</p>
                </div>
            """, unsafe_allow_html=True)

            # Chart
            history = pd.DataFrame({"Year": [2022, 2023, 2024, 2025], "Yield": [2.5, 2.8, 3.0, prediction]})
            fig = px.line(history, x="Year", y="Yield", markers=True, template="plotly_dark")
            fig.update_traces(line_color='#00b09b', marker=dict(size=10, color="#96c93d"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Yield model not loaded.")

# ============================
# 💡 TIPS & FORECAST TAB
# ============================
with tab3:
    st.markdown("## 💡 Agricultural Intelligence")
    tips = [
        "💧 **Irrigation:** Use drip irrigation to save 40% water.",
        "🧪 **Soil:** Test Nitrogen levels every 6 months.",
        "🌡️ **Climate:** High heat detected. Increase mulch layer.",
        "🛡️ **Protection:** Early blight signs usually appear on lower leaves first."
    ]
    for tip in tips:
        st.info(tip)

# ---------------- Sidebar QR ----------------
st.sidebar.markdown("---")
st.sidebar.write("📱 **Mobile Access**")
app_url = "http://10.233.254.78:8501"
qr_img = generate_qr(app_url)
buf = BytesIO()
qr_img.save(buf, format="PNG")
st.sidebar.image(buf.getvalue(), width=150)
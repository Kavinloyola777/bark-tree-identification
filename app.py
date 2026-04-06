import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import plotly.express as px
import pandas as pd
import os
import logging
import gdown
import requests
import time

# ================== SETTINGS ==================
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = 'best_model_resnet50v2.keras'
json_path = 'tree_co2_data.json'
class_indices_path = 'class_indices.json'

# ✅ FIXED DRIVE LINK
model_url = "https://drive.google.com/uc?id=16oK8L6Bd3_cHgt0jU2BfqX3IZtfQZN1_"

# ================== DOWNLOAD MODEL ==================
def download_model():
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... ⏳"):
            try:
                gdown.download(model_url, model_path, quiet=False)
                logger.info("Model downloaded successfully")
            except Exception as e:
                st.error(f"Download failed: {e}")
                st.stop()

download_model()

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model_safe():
    try:
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False
        )
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

model = load_model_safe()

# ================== LOAD DATA ==================
try:
    with open(json_path, 'r') as f:
        tree_data = json.load(f)

    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)

    class_labels = list(class_indices.keys())
    tree_data_lower = {k.lower(): v for k, v in tree_data.items()}

except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# ================== UI ==================
st.title("🌳 Bark-Based Tree Species Identification & CO₂ Estimator")

# ================== PREDICTION ==================
def predict_species(image):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array, verbose=0)
        confidence = float(np.max(preds))
        return class_labels[np.argmax(preds)], confidence

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# ================== WIKIPEDIA ==================
def get_wikipedia_summary(species):
    api_url = "https://en.wikipedia.org/w/api.php"

    try:
        params = {
            "action": "query",
            "format": "json",
            "titles": species,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "exsentences": 3
        }

        response = requests.get(api_url, params=params)
        data = response.json()

        pages = data['query']['pages']
        page_id = next(iter(pages))

        if page_id != "-1":
            return pages[page_id]['extract']

    except:
        pass

    return "Description not available."

# ================== UPLOAD ==================
uploaded_file = st.file_uploader("Upload a bark image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    species, confidence = predict_species(image)

    if species:
        info = tree_data_lower.get(species.lower(), {
            "thinai": "Unknown",
            "co2_daily_kg": 0,
            "co2_monthly_kg": 0,
            "co2_yearly_kg": 0
        })

        st.success(f"🌿 Predicted Species: {species}")
        st.info(f"📊 Confidence: {confidence*100:.2f}%")

        st.write(f"🌍 Thinai: {info['thinai']}")
        st.write(f"🌱 Daily CO₂: {info['co2_daily_kg']} kg")
        st.write(f"📅 Monthly CO₂: {info['co2_monthly_kg']} kg")
        st.write(f"📆 Yearly CO₂: {info['co2_yearly_kg']} kg")

        st.subheader("📖 About Tree")
        st.write(get_wikipedia_summary(species))

        # ================== VISUALIZATION ==================
        df = pd.DataFrame.from_dict(tree_data, orient='index').reset_index()
        df.columns = ['Species', 'Thinai', 'Yearly_CO2', 'Monthly_CO2', 'Daily_CO2']

        fig_bar = px.bar(df, x='Species', y='Yearly_CO2',
                         title='Yearly CO₂ Absorption',
                         color='Thinai')
        st.plotly_chart(fig_bar)

        thinai_counts = df['Thinai'].value_counts().reset_index()
        thinai_counts.columns = ['Thinai', 'Count']

        fig_pie = px.pie(thinai_counts, values='Count', names='Thinai',
                         title='Thinai Distribution')
        st.plotly_chart(fig_pie)

# ================== FOOTER ==================
st.markdown("---")
st.caption("🚀 BarkID Project | AI Tree Classification + CO₂ Estimation")

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

# -------------------- SETTINGS --------------------
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- MODEL DOWNLOAD --------------------
model_path = "final_model.h5"
model_url = "https://drive.google.com/uc?id=1aUvN-ZDtDmRvdR7sJHI0ep2aCjBg1D0W"

if not os.path.exists(model_path):
    with st.spinner("Downloading model... (1-2 mins first time)"):
        gdown.download(model_url, model_path, quiet=False)

# -------------------- LOAD MODEL (FIXED) --------------------
try:
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False   # 🔥 THIS FIXES YOUR ERROR
    )
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# -------------------- LOAD DATA --------------------
try:
    with open("tree_co2_data.json", "r") as f:
        tree_data = json.load(f)

    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)

    class_labels = list(class_indices.keys())
    tree_data_lower = {k.lower(): v for k, v in tree_data.items()}

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# -------------------- UI --------------------
st.title("🌳 Bark-Based Tree Species Identification & CO₂ Estimator")

# -------------------- PREDICTION --------------------
def predict_species(image):
    try:
        image = image.convert("RGB")
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array, verbose=0)
        return class_labels[np.argmax(preds)]

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# -------------------- UPLOAD --------------------
uploaded_file = st.file_uploader("Upload a bark image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    species = predict_species(image)

    if species:
        species_lower = species.lower()

        st.success(f"Predicted: {species}")

        info = tree_data_lower.get(
            species_lower,
            {"thinai": "Unknown", "co2_daily_kg": 0, "co2_monthly_kg": 0, "co2_yearly_kg": 0}
        )

        st.info(f"Thinai Region: {info['thinai']}")
        st.write(f"Daily CO₂: {info['co2_daily_kg']:.3f} kg")
        st.write(f"Monthly CO₂: {info['co2_monthly_kg']:.2f} kg")
        st.write(f"Yearly CO₂: {info['co2_yearly_kg']:.2f} kg")

        # -------------------- VISUALS --------------------
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

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("Final Year Project | AI + Ecology | 2026")

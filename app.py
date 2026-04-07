import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import plotly.express as px
import pandas as pd
import requests
import time
import os
import logging

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
model_path = 'best_model_resnet50v2.keras'
json_path = 'tree_co2_data.json'
class_indices_path = 'class_indices.json'
model_url = "https://drive.google.com/uc?export=download&id=16oK8L6Bd3_cHgt0jU2BfqX3IZtfQZN1_"

# Download model if not exists
def download_model(url, dest_path):
    if not os.path.exists(dest_path):
        with st.spinner("Downloading model file..."):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(dest_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Model downloaded to {dest_path}")
            except requests.exceptions.RequestException as e:
                st.error(f"Model download failed: {e}")
                st.stop()

download_model(model_url, model_path)

st.title("Bark-Based Tree Species Identification & CO₂ Absorption Estimator")

try:
    model = tf.keras.models.load_model(model_path)
    with open(json_path, 'r') as f:
        tree_data = json.load(f)
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    class_labels = list(class_indices.keys())
    tree_data_lower = {k.lower(): v for k, v in tree_data.items()}
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

def predict_species(image):
    try:
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array, verbose=0)
        return class_labels[np.argmax(preds)]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def get_wikipedia_summary(species):
    api_url = "https://en.wikipedia.org/w/api.php"
    scientific_names = {
        "teak": "Tectona grandis",
        "amla": "Phyllanthus emblica",
        "banyan": "Ficus benghalensis",
        "chir_pine": "Pinus roxburghii",
        "coconut": "Cocos nucifera",
        "desert_date": "Balanites aegyptiaca",
        "guava": "Psidium guajava",
        "indian_beech": "Millettia pinnata",
        "indian_trumpet": "Oroxylum indicum",
        "jackfruit": "Artocarpus heterophyllus",
        "jamun": "Syzygium cumini",
        "jand": "Prosopis cineraria",
        "karungali": "Acacia catechu",
        "mango": "Mangifera indica",
        "muringa_tree": "Moringa oleifera",
        "neem_tree": "Azadirachta indica",
        "palmyra_palm": "Borassus flabellifer",
        "peepal": "Ficus religiosa",
        "punnai": "Calophyllum inophyllum",
        "sandalwood": "Santalum album",
        "turmeric_tree": "Berberis_aristata",
        "vagai": "Albizia lebbeck",
        "vathakkani": "Acacia chundra",
        "wild_date_palm": "Phoenix sylvestris",
        "anjan": "Hardwickia binata"
    }
    fallback_descriptions = {
        "teak": "Teak (Tectona grandis) is a tropical hardwood tree in the Lamiaceae family, known for its durable and water-resistant wood, commonly used in furniture and boat building.",
        "amla": "Amla (Phyllanthus emblica) is a deciduous tree native to India, valued for its edible fruit rich in vitamin C and used in traditional medicine.",
        "banyan": "Banyan (Ficus benghalensis) is a large, evergreen tree known for its extensive canopy and aerial roots, often considered sacred in India.",
        "chir_pine": "Chir Pine (Pinus roxburghii) is a coniferous tree native to the Himalayas, used for timber and resin production.",
        "coconut": "Coconut (Cocos nucifera) is a palm tree grown in coastal regions, known for its versatile fruit used in food and industry.",
        "desert_date": "Desert Date (Balanites aegyptiaca) is a drought-resistant tree found in arid regions, valued for its fruit and oil.",
        "guava": "Guava (Psidium guajava) is a tropical tree known for its sweet, edible fruit rich in vitamins.",
        "indian_beech": "Indian Beech (Millettia pinnata) is a fast-growing tree used for timber and biofuel production.",
        "indian_trumpet": "Indian Trumpet (Oroxylum indicum) is a medicinal tree native to India, known for its use in Ayurvedic remedies.",
        "jackfruit": "Jackfruit (Artocarpus heterophyllus) is a large tropical tree producing the world's largest tree-borne fruit.",
        "jamun": "Jamun (Syzygium cumini) is a tropical tree known for its dark purple fruit and medicinal properties.",
        "jand": "Jand (Prosopis cineraria) is a desert tree vital to arid ecosystems, providing shade and fodder.",
        "karungali": "Karungali (Acacia catechu) is a hardwood tree used for timber and traditional medicine.",
        "mango": "Mango (Mangifera indica) is a widely cultivated tropical tree known for its delicious fruit.",
        "muringa_tree": "Moringa (Moringa oleifera) is a fast-growing tree with nutritious leaves and pods, used in food and medicine.",
        "neem_tree": "Neem (Azadirachta indica) is a versatile tree known for its medicinal properties and pest-repellent qualities.",
        "palmyra_palm": "Palmyra Palm (Borassus flabellifer) is a coastal palm tree used for its fruit, sap, and leaves.",
        "peepal": "Peepal (Ficus religiosa) is a sacred fig tree in India, known for its heart-shaped leaves and cultural significance.",
        "punnai": "Punnai (Calophyllum inophyllum) is a coastal tree valued for its oil and ornamental qualities.",
        "sandalwood": "Sandalwood (Santalum album) is a fragrant tree prized for its aromatic wood used in perfumes and rituals.",
        "turmeric_tree": "Turmeric Tree (Berberis_aristata) is a shrub used in traditional medicine for its root properties.",
        "vagai": "Vagai (Albizia lebbeck) is a fast-growing tree used for timber and shade in tropical regions.",
        "vathakkani": "Vathakkani (Acacia chundra) is a hardwood tree used in furniture and traditional applications.",
        "wild_date_palm": "Wild Date Palm (Phoenix sylvestris) is a palm tree found in dry regions, known for its sweet sap.",
        "anjan": "Anjan (Hardwickia binata) is a hardwood tree native to India, valued for its strong timber used in construction."
    }
    species_lower = species.lower()
    queries = [species_lower + " (tree)", scientific_names.get(species_lower, species_lower), species_lower]
    for query in queries:
        try:
            logger.info(f"Trying Wikipedia query: {query}")
            params = {
                "action": "query",
                "format": "json",
                "titles": query,
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "exsentences": 3,
                "redirects": 1
            }
            headers = {"User-Agent": "Capstone Project/1.0 (kavinloyola777@gmail.com)"}
            time.sleep(1)
            response = requests.get(api_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            pages = data['query']['pages']
            page_id = next(iter(pages))
            if page_id != "-1" and 'extract' in pages[page_id]:
                logger.info(f"Success for query: {query}")
                return pages[page_id]['extract']
            logger.warning(f"No page found for query: {query}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {query}: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error for {query}: {e}")
            continue
    return fallback_descriptions.get(species_lower, f"No Wikipedia page found for '{species}'.")

uploaded_file = st.file_uploader("Upload a bark image (JPG/PNG)", type=["jpg", "png", "jpeg"])
if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Bark Image", width='stretch')  # Fixed deprecation
        species = predict_species(image)
        if species:
            species_lower = species.lower()
            logger.info(f"Predicted species: {species_lower}")
            if species_lower not in [label.lower() for label in class_labels]:
                st.warning(f"Species '{species}' is not in the trained dataset. Predictions may be inaccurate. Consider retraining the model.")
                info = {"thinai": "Unknown", "co2_daily_kg": 0, "co2_monthly_kg": 0, "co2_yearly_kg": 0}
            else:
                info = tree_data_lower.get(species_lower, {"thinai": "Unknown", "co2_daily_kg": 0, "co2_monthly_kg": 0, "co2_yearly_kg": 0})
            st.success(f"Predicted Species: {species}")
            st.info(f"Thinai Region: {info['thinai']}")
            st.write(f"Daily CO₂ Absorption: {info['co2_daily_kg']:.3f} kg")
            st.write(f"Monthly CO₂ Absorption: {info['co2_monthly_kg']:.2f} kg")
            st.write(f"Yearly CO₂ Absorption: {info['co2_yearly_kg']:.2f} kg")

            # Wikipedia Summary
            st.subheader("About the Species")
            with st.spinner("Fetching Wikipedia summary..."):
                wiki_summary = get_wikipedia_summary(species_lower)  # Use species_lower here
                logger.info(f"Wikipedia summary: {wiki_summary[:100]}...")
            if wiki_summary and "No Wikipedia page found" not in wiki_summary:
                st.write(wiki_summary)
            else:
                st.write("Wikipedia summary not available. Using fallback description.")
                st.write(fallback_descriptions.get(species_lower, "No description available."))

            # Visualizations
            df = pd.DataFrame.from_dict(tree_data, orient='index').reset_index()
            df.columns = ['Species', 'Thinai', 'Yearly_CO2', 'Monthly_CO2', 'Daily_CO2']
            fig_bar = px.bar(df, x='Species', y='Yearly_CO2', title='Yearly CO₂ Absorption by Species (kg)', color='Thinai')
            st.plotly_chart(fig_bar)
            thinai_counts = df['Thinai'].value_counts().reset_index()
            fig_pie = px.pie(thinai_counts, values='count', names='Thinai', title='Tree Distribution by Thinai Region')
            st.plotly_chart(fig_pie)
    except Exception as e:
        st.error(f"Image processing error: {e}")

st.markdown("---")
st.caption("Capstone Project | Wikipedia API for Species Info | April , 2026")

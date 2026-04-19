import streamlit as st # type: ignore
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------ CONFIG ------------------
st.set_page_config(page_title="🌿 AI Plant Doctor", layout="wide")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_detection_model.keras")

model = load_model()

# ------------------ CLASS NAMES (ORDER MUST MATCH TRAINING) ------------------
class_names = {
    0: "Pepper__bell___Bacterial_spot",
    1: "Pepper__bell___healthy",
    2: "Potato___Early_blight",
    3: "Potato___Late_blight",
    4: "Potato___healthy",
    5: "Tomato_Bacterial_spot",
    6: "Tomato_Early_blight",
    7: "Tomato_Late_blight",
    8: "Tomato_Leaf_Mold",
    9: "Tomato_Septoria_leaf_spot",
    10: "Tomato_Spider_mites_Two_spotted_spider_mite",
    11: "Tomato_Target_Spot",
    12: "Tomato_Tomato_YellowLeaf_Curl_Virus",
    13: "Tomato_Tomato_mosaic_virus",
    14: "Tomato_healthy"
}

# ------------------ DISEASE INFO ------------------
disease_info = {

    "Pepper__bell___Bacterial_spot": {
        "name": "Bell Pepper - Bacterial Spot",
        "status": "Diseased",
        "description": "Small dark lesions on leaves and fruits caused by bacteria.",
        "treatment": "Remove infected leaves immediately. Spray copper-based bactericides every 7–10 days. Avoid overhead watering and ensure proper spacing for airflow."
    },

    "Pepper__bell___healthy": {
        "name": "Bell Pepper - Healthy",
        "status": "Healthy",
        "description": "Plant is healthy with no visible disease.",
        "treatment": "Maintain balanced fertilization, proper watering, and regular inspection to prevent early disease development."
    },

    "Potato___Early_blight": {
        "name": "Potato - Early Blight",
        "status": "Diseased",
        "description": "Brown spots with concentric rings on leaves.",
        "treatment": "Apply fungicides like Mancozeb or Chlorothalonil every 7–10 days. Remove infected leaves and practice crop rotation."
    },

    "Potato___Late_blight": {
        "name": "Potato - Late Blight",
        "status": "Diseased",
        "description": "Serious disease causing rapid plant decay.",
        "treatment": "Immediately remove infected plants. Apply systemic fungicides like Metalaxyl. Avoid excess moisture and ensure proper drainage."
    },

    "Potato___healthy": {
        "name": "Potato - Healthy",
        "status": "Healthy",
        "description": "Plant is disease-free.",
        "treatment": "Ensure proper irrigation, balanced nutrients, and periodic monitoring to maintain plant health."
    },

    "Tomato_Bacterial_spot": {
        "name": "Tomato - Bacterial Spot",
        "status": "Diseased",
        "description": "Dark water-soaked lesions on leaves.",
        "treatment": "Use copper-based sprays weekly. Avoid working with wet plants and remove infected debris from soil."
    },

    "Tomato_Early_blight": {
        "name": "Tomato - Early Blight",
        "status": "Diseased",
        "description": "Brown spots with rings on older leaves.",
        "treatment": "Apply fungicides like Mancozeb regularly. Remove affected leaves and maintain crop rotation."
    },

    "Tomato_Late_blight": {
        "name": "Tomato - Late Blight",
        "status": "Diseased",
        "description": "Dark lesions causing rapid destruction.",
        "treatment": "Remove infected plants immediately. Spray fungicides like Metalaxyl. Avoid high humidity and water accumulation."
    },

    "Tomato_Leaf_Mold": {
        "name": "Tomato - Leaf Mold",
        "status": "Diseased",
        "description": "Yellow spots with mold growth underneath leaves.",
        "treatment": "Improve ventilation in greenhouse. Reduce humidity. Apply fungicides like Chlorothalonil."
    },

    "Tomato_Septoria_leaf_spot": {
        "name": "Tomato - Septoria Leaf Spot",
        "status": "Diseased",
        "description": "Small circular spots with gray centers.",
        "treatment": "Remove infected leaves. Apply fungicides regularly. Avoid splashing water on leaves."
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Tomato - Spider Mites",
        "status": "Diseased",
        "description": "Tiny mites causing yellow speckles on leaves.",
        "treatment": "Spray strong water jets to remove mites. Use neem oil or insecticidal soap every few days."
    },

    "Tomato_Target_Spot": {
        "name": "Tomato - Target Spot",
        "status": "Diseased",
        "description": "Dark concentric ring spots.",
        "treatment": "Apply fungicides like Azoxystrobin. Remove infected leaves and avoid overcrowding."
    },

    "Tomato_Tomato_YellowLeaf_Curl_Virus": {
        "name": "Tomato - Yellow Leaf Curl Virus",
        "status": "Diseased",
        "description": "Leaves curl and turn yellow due to viral infection.",
        "treatment": "Control whiteflies using neem oil or insecticides. Remove infected plants immediately. Use resistant varieties."
    },

    "Tomato_Tomato_mosaic_virus": {
        "name": "Tomato - Mosaic Virus",
        "status": "Diseased",
        "description": "Mosaic patterns on leaves.",
        "treatment": "Remove infected plants. Disinfect tools regularly. Avoid handling plants after tobacco use."
    },

    "Tomato_healthy": {
        "name": "Tomato - Healthy",
        "status": "Healthy",
        "description": "Plant is healthy.",
        "treatment": "Maintain good irrigation, sunlight exposure, and nutrient balance. Monitor regularly for early symptoms."
    }
}
# ------------------ UI STYLE ------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.title {
    text-align: center;
    font-size: 42px;
    color: white;
    font-weight: bold;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌿 AI Plant Doctor</div>', unsafe_allow_html=True)

# ------------------ UPLOAD ------------------
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

col1, col2 = st.columns(2)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # LEFT
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # PREPROCESS
    img = image.resize((224,224))
    from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore

    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # PREDICT
    prediction = model.predict(img_array)
    pred_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    result_key = class_names.get(pred_index, "Unknown")

    info = disease_info.get(result_key, {
        "name": result_key,
        "status": "Unknown",
        "description": "No data available",
        "treatment": "Consult expert"
    })

    # RIGHT SIDE RESULT
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🧠 Prediction")
        if info["status"] == "Healthy":
            st.success(info["name"])
        else:
            st.error(info["name"])

        st.subheader("📊 Confidence")
        st.progress(int(confidence*100))
        st.write(f"{confidence*100:.2f}%")

        st.subheader("📌 Status")
        st.write(info["status"])

        st.subheader("📖 Description")
        st.write(info["description"])

        st.subheader("💊 Treatment")
        st.write(info["treatment"])

        st.markdown('</div>', unsafe_allow_html=True)
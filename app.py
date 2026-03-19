import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import platform

# --- CONFIGURACIÓN ---
st.set_page_config(
    page_title="Reconocimiento de Imágenes",
    page_icon="🤖",
    layout="centered"
)

# --- CACHE DEL MODELO (IMPORTANTE) ---
@st.cache_resource
def cargar_modelo():
    return load_model('keras_model.h5')

model = cargar_modelo()

# --- UI ---
st.title("🤖 Reconocimiento de Imágenes")
st.caption(f"Python {platform.python_version()}")

with st.sidebar:
    st.subheader("📸 Instrucciones")
    st.write("Toma una foto y el modelo intentará clasificarla")

# Imagen de referencia (opcional)
try:
    image = Image.open('OIG5.jpg')
    st.image(image, width=300, caption="Ejemplo")
except:
    st.warning("No se encontró imagen de ejemplo")

# --- CÁMARA ---
img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    # Convertir a RGB (IMPORTANTE)
    img = Image.open(img_file_buffer).convert("RGB")

    st.image(img, caption="Imagen capturada", width=300)

    # --- PREPROCESAMIENTO ---
    img = img.resize((224, 224))
    img_array = np.array(img)

    normalized = (img_array.astype(np.float32) / 127.0) - 1
    data = np.expand_dims(normalized, axis=0)

    # --- PREDICCIÓN ---
    with st.spinner("Analizando imagen..."):
        prediction = model.predict(data)

    st.write("Predicción cruda:", prediction)

    # --- INTERPRETACIÓN ---
    clases = ["Izquierda", "Arriba"]  # AJUSTA si tienes más clases
    index = np.argmax(prediction)
    prob = prediction[0][index]

    st.success(f"Resultado: {clases[index]}")
    st.metric("Confianza", f"{prob:.2f}")

import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# --- CONFIG ---
st.set_page_config(page_title="Reconocimiento IA", page_icon="🤖", layout="centered")

# --- CACHE MODELO ---
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

# Imagen de referencia
try:
    image = Image.open('OIG5.jpg')
    st.image(image, width=300, caption="Ejemplo")
except:
    st.warning("No se encontró imagen de ejemplo")

# Cámara
img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    img = Image.open(img_file_buffer).convert("RGB")

    st.image(img, caption="Imagen capturada", width=300)

    # Preprocesamiento
    img = img.resize((224, 224))
    img_array = np.array(img)

    normalized = (img_array.astype(np.float32) / 127.0) - 1
    data = np.expand_dims(normalized, axis=0)

    # Predicción
    with st.spinner("Analizando imagen..."):
        prediction = model.predict(data)

    st.write("Predicción:", prediction)

    # Interpretación
    clases = ["Izquierda", "Arriba"]  # ajusta según tu modelo
    index = np.argmax(prediction)
    prob = prediction[0][index]

    st.success(f"Resultado: {clases[index]}")
    st.metric("Confianza", f"{prob:.2f}")

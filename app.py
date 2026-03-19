
app.py
requirements.txt
keras_model.h5   ✅
OIG5.jpg (opcional)

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

# --- ESTILOS BONITOS ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
h1, h2, h3 {
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# --- CARGA SEGURA DEL MODELO ---
@st.cache_resource
def cargar_modelo():
    try:
        model = load_model('keras_model.h5')
        return model
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {e}")
        return None

model = cargar_modelo()

# Si falla el modelo, detener app
if model is None:
    st.stop()

# --- UI ---
st.title("🤖 Reconocimiento de Imágenes")
st.caption(f"Python {platform.python_version()}")

with st.sidebar:
    st.subheader("📸 Instrucciones")
    st.write("Toma una foto y el modelo intentará clasificarla")

# Imagen de ejemplo (opcional)
try:
    image = Image.open('OIG5.jpg')
    st.image(image, width=300, caption="Ejemplo")
except:
    st.warning("No se encontró imagen de ejemplo")

# --- CÁMARA ---
img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    try:
        # Convertir imagen correctamente
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

        st.write("🔍 Predicción cruda:", prediction)

        # --- INTERPRETACIÓN ---
        clases = ["Izquierda", "Arriba"]  # Ajusta según tu modelo
        index = np.argmax(prediction)
        prob = prediction[0][index]

        st.success(f"Resultado: {clases[index]}")
        st.metric("Confianza", f"{prob:.2f}")

    except Exception as e:
        st.error(f"❌ Error procesando la imagen: {e}")

# --- FOOTER ---
st.markdown("""
<div style='text-align:center; color:#94a3b8; margin-top:30px;'>
Hecho con ❤️ usando Streamlit + TensorFlow
</div>
""", unsafe_allow_html=True)

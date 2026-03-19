import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import platform

# --- CONFIG ---
st.set_page_config(
    page_title="Reconocimiento IA",
    page_icon="🦈",
    layout="wide"
)

# --- ESTILOS CELESTE ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e0f2fe, #bae6fd);
    color: #0c4a6e;
}

h1 {
    text-align: center;
    color: #0284c7;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 0 20px rgba(2,132,199,0.2);
    margin-top: 20px;
}

section[data-testid="stSidebar"] {
    background: #e0f2fe;
}

.stButton>button {
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
    color: white;
    border-radius: 12px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<h1>🦈 Reconocimiento Inteligente</h1>
<p style='text-align:center; color:#0369a1;'>
Tu modelo detectando magia en tiempo real 💙✨
</p>
""", unsafe_allow_html=True)

st.caption(f"Python {platform.python_version()}")

# --- MODELO ---
@st.cache_resource
def cargar_modelo():
    try:
        return load_model('keras_model.h5')
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {e}")
        return None

model = cargar_modelo()

if model is None:
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.subheader("⚙️ Info")
    st.write("Modelo entrenado con Teachable Machine 🧠")

# --- IMAGEN TIBURONCIN 🦈 ---
try:
    image = Image.open('TIBURONCIN.jpg')
    st.image(image, width=280, caption="Nuestro tiburoncín 🦈💙")
except:
    st.warning("No se encontró la imagen TIBURONCIN.jpg")

# --- CÁMARA ---
img_file_buffer = st.camera_input("📸 Toma una Foto")

if img_file_buffer is not None:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # --- IMAGEN ---
    with col1:
        img = Image.open(img_file_buffer).convert("RGB")
        st.image(img, caption="Imagen capturada", use_container_width=True)

    # --- RESULTADOS ---
    with col2:
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)

        normalized = (img_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized

        with st.spinner("Analizando imagen... 🧠"):
            prediction = model.predict(data)

        st.markdown("### 📊 Resultados")

        if prediction[0][0] > 0.5:
            st.success(f"📍 Izquierda ({prediction[0][0]:.2f})")

        if prediction[0][1] > 0.5:
            st.success(f"💃 MACARENA 1 ({prediction[0][1]:.2f})")

        # Barra visual
        st.progress(float(np.max(prediction)))

    st.markdown("</div>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div style='text-align:center; color:#0284c7; margin-top:30px;'>
Hecho con 💙 + IA + tiburoncín 🦈
</div>
""", unsafe_allow_html=True)

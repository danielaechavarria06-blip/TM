import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import platform

# --- CONFIG ---
st.set_page_config(
    page_title="Reconocimiento IA",
    page_icon="🤖",
    layout="wide"
)

# --- ESTILOS ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

h1 {
    text-align: center;
    color: #38bdf8;
}

.card {
    background: #020617;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 0 25px rgba(56,189,248,0.2);
    margin-top: 20px;
}

section[data-testid="stSidebar"] {
    background: #020617;
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<h1>🤖 Reconocimiento Inteligente de Imágenes</h1>
<p style='text-align:center; color:#94a3b8;'>
Toma una foto y deja que la IA identifique lo que ve 👀✨
</p>
""", unsafe_allow_html=True)

# --- INFO PYTHON ---
st.caption(f"Python {platform.python_version()}")

# --- CARGA MODELO ---
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
    st.markdown("## ⚙️ Información")
    st.write("Este modelo fue entrenado con Teachable Machine.")
    st.write("Apunta la cámara y prueba detecciones 😏")

# --- IMAGEN EJEMPLO ---
try:
    image = Image.open('OIG5.jpg')
    st.image(image, width=300, caption="Ejemplo")
except:
    st.warning("No se encontró imagen de ejemplo")

# --- CÁMARA ---
img_file_buffer = st.camera_input("📸 Toma una Foto")

if img_file_buffer is not None:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        img = Image.open(img_file_buffer).convert("RGB")
        st.image(img, caption="Imagen capturada", use_container_width=True)

    with col2:
        # --- PROCESAMIENTO ---
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)

        normalized = (img_array.astype(np.float32) / 127.0) - 1
        data = np.expand_dims(normalized, axis=0)

        # --- PREDICCIÓN ---
        with st.spinner("Analizando imagen..."):
            try:
                prediction = model.predict(data)
            except Exception as e:
                st.error(f"❌ Error en predicción: {e}")
                st.stop()

        st.markdown("### 📊 Resultados")

        # --- RESULTADOS ---
        if prediction[0][0] > 0.5:
            st.success(f"📍 Izquierda ({prediction[0][0]:.2f})")

        if prediction[0][1] > 0.5:
            st.success(f"⬆️ Arriba ({prediction[0][1]:.2f})")

        # EXTRA VISUAL 🔥
        st.progress(float(np.max(prediction)))

    st.markdown("</div>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div style='text-align:center; color:#64748b; margin-top:30px;'>
Hecho con ❤️ usando Streamlit + TensorFlow
</div>
""", unsafe_allow_html=True)

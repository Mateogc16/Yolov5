import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
from PIL import Image

# ----------------------------------------
# Configuración de página
# ----------------------------------------
st.set_page_config(
    page_title="Oráculo del Castillo – Visión Mística de Objetos",
    page_icon="🧙‍♂️",
    layout="wide"
)

# ----------------------------------------
# Estilos medievales
# ----------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=UnifrakturCook:wght@700&display=swap');

    html, body, [class*="css"] {
        background-color: #1a1a1a;
        color: #e0d8c3;
        font-family: 'UnifrakturCook', cursive;
    }

    .stButton>button {
        background-color: #5b3e1d;
        color: white;
        border-radius: 12px;
        border: 2px solid #a67c52;
        font-size: 18px;
        padding: 0.5em 1em;
    }

    .stSidebar {
        background-color: #2f2f2f;
    }

    h1, h2, h3 {
        color: #f8f3dc !important;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------
# Cargar modelo YOLOv5 vía torch.hub
# ----------------------------------------
@st.cache_resource
def load_model():
    # Esto descargará automáticamente yolov5s desde el repositorio oficial
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

# ----------------------------------------
# Título y bienvenida
# ----------------------------------------
st.title("🔮 Oráculo del Castillo – Visión Mística de Objetos")
st.markdown("""
¡Bienvenido al gran salón real!  
Convoca el hechizo de visión profunda para revelar las criaturas y artefactos ocultos en tus imágenes.
""")

# ----------------------------------------
# Parámetros del hechizo
# ----------------------------------------
st.sidebar.title("⚙️ Parámetros del Hechizo")
conf = st.sidebar.slider('🧠 Confianza mínima', 0.0, 1.0, 0.25, 0.01)
iou  = st.sidebar.slider('🎯 Umbral IoU',     0.0, 1.0, 0.45, 0.01)
agnostic    = st.sidebar.checkbox('🔄 NMS sin clase', False)
multi_label = st.sidebar.checkbox('🏷️ Etiquetas múltiples', False)
max_det     = st.sidebar.number_input('🧮 Máximas detecciones', 10, 2000, 1000, 10)

# ----------------------------------------
# Captura de imagen
# ----------------------------------------
picture = st.camera_input("📸 Captura tu visión mágica")

if picture:
    # Leer imagen en OpenCV
    file_bytes = np.asarray(bytearray(picture.getvalue()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Configurar el modelo
    model.conf = conf
    model.iou  = iou
    model.agnostic_nms = agnostic
    model.multi_label  = multi_label
    model.max_det      = max_det

    # Invocar detección
    with st.spinner("🔍 Consultando el oráculo..."):
        results = model(img)

    # Dibujar resultados directamente
    results.render()  # modifica results.imgs in-place

    # Mostrar visión revelada
    st.subheader("🖼️ Visión Revelada")
    # results.imgs[0] es una lista de arrays
    st.image(results.imgs[0], channels='BGR', use_column_width=True)

    # Mostrar tabla de detección
    st.subheader("📜 Registro de criaturas y artefactos")
    df = results.pandas().xyxy[0][['name','confidence','xmin','ymin','xmax','ymax']]
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df['name'].value_counts())

# ----------------------------------------
# Pie de página
# ----------------------------------------
st.markdown("---")
st.caption("🏰 Desarrollado por el gremio de alquimistas digitales con Streamlit y PyTorch")


import cv2
import streamlit as st
import numpy as np
import torch
from PIL import Image

# Configurar la página de Streamlit
st.set_page_config(page_title="Detección de Objetos - YOLOv5", page_icon="🔍", layout="wide")

# Función para cargar el modelo YOLOv5
@st.cache_resource
def load_yolov5_model():
    import yolov5
    model = yolov5.load('yolov5s.pt')  # Asegúrate de que el archivo esté disponible o usa el modelo preentrenado
    return model

# Cargar el modelo
model = load_yolov5_model()

# Título de la aplicación
st.title("🔍 Detección de Objetos con YOLOv5")
st.markdown("""
Esta aplicación utiliza el modelo **YOLOv5** para realizar detección de objetos en imágenes cargadas por el usuario.
Sube una imagen y detecta objetos en ella.
""")

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convertir la imagen a formato que OpenCV pueda procesar
    img = Image.open(uploaded_file)
    img_cv = np.array(img)

    # Realizar la detección
    with st.spinner("Detectando objetos..."):
        results = model(img_cv)
        results.render()  # Dibujar las cajas de los objetos detectados

    # Mostrar la imagen con las predicciones
    st.image(results.imgs[0], caption="Imagen con detección", use_column_width=True)

    # Mostrar información de los objetos detectados
    st.subheader("Objetos Detectados")
    results_df = results.pandas().xywh[0]  # Obtener los resultados en formato pandas DataFrame
    st.dataframe(results_df[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])

# Información adicional sobre el proyecto
st.markdown("---")
st.caption("""
**Acerca de la aplicación**: Esta aplicación utiliza YOLOv5 para detección de objetos en tiempo real.
Desarrollada con Streamlit y PyTorch.
""")


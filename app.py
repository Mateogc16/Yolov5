import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página Streamlit
st.set_page_config(
    page_title="Oráculo del Castillo – Visión Mística de Objetos",
    page_icon="🧙‍♂️",
    layout="wide"
)

# Estilo medieval
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

    .st-bf, .st-cf {
        background-color: #2f2f2f !important;
    }

    h1, h2, h3 {
        color: #f8f3dc !important;
    }

    .css-1v0mbdj p {
        font-family: 'UnifrakturCook', cursive;
    }
    </style>
""", unsafe_allow_html=True)

# Función para cargar el modelo YOLOv5
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            try:
                model = yolov5.load(model_path)
                return model
            except:
                st.warning("Intentando método alternativo de carga...")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

# Título y bienvenida
st.title("🔮 Oráculo del Castillo – Visión Mística de Objetos")
st.markdown("""
¡Bienvenido, viajero del reino! 📜<br>
Convoca el hechizo de visión profunda para revelar las entidades ocultas en tus imágenes.  
Ajusta los pergaminos del costado para mejorar la precisión de tus visiones.
""", unsafe_allow_html=True)

# Cargar modelo
with st.spinner("🧙‍♂️ Invocando el poder de YOLOv5..."):
    model = load_yolov5_model()

if model:
    st.sidebar.title("🧪 Parámetros del hechizo")

    with st.sidebar:
        model.conf = st.slider('🧠 Confianza mínima', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('🎯 Umbral IoU', 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")
        try:
            model.agnostic = st.checkbox('🔄 NMS sin clase', False)
            model.multi_label = st.checkbox('🏷️ Etiquetas múltiples', False)
            model.max_det = st.number_input('🧮 Máximas detecciones', 10, 2000, 1000, 10)
        except:
            st.warning("Algunas opciones avanzadas no están disponibles.")

    main_container = st.container()
    with main_container:
        picture = st.camera_input("📸 Captura tu visión mágica")

        if picture:
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            with st.spinner("🔍 Consultando el oráculo..."):
                try:
                    results = model(cv2_img)
                except Exception as e:
                    st.error(f"Error durante la detección: {str(e)}")
                    st.stop()

            try:
                predictions = results.pred[0]
                boxes = predictions[:, :4]
                scores = predictions[:, 4]
                categories = predictions[:, 5]

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🖼️ Visión revelada")
                    results.render()
                    st.image(cv2_img, channels='BGR', use_column_width=True)

                with col2:
                    st.subheader("📜 Registro de criaturas y artefactos")
                    label_names = model.names
                    category_count = {}

                    for category in categories:
                        category_idx = int(category.item()) if hasattr(category, 'item') else int(category)
                        category_count[category_idx] = category_count.get(category_idx, 0) + 1

                    data = []
                    for category, count in category_count.items():
                        label = label_names[category]
                        confidence = scores[categories == category].mean().item() if len(scores) > 0 else 0
                        data.append({
                            "Categoría": label,
                            "Cantidad": count,
                            "Confianza promedio": f"{confidence:.2f}"
                        })

                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                        st.bar_chart(df.set_index('Categoría')['Cantidad'])
                    else:
                        st.info("⚠️ No se detectaron objetos. Ajusta los valores mágicos en la barra lateral.")
            except Exception as e:
                st.error(f"Error al procesar resultados: {str(e)}")
                st.stop()
else:
    st.error("⚠️ No se pudo cargar el modelo. Asegúrate de que todos los hechizos (dependencias) estén instalados.")

# Pie de página medieval
st.markdown("---")
st.caption("🏰 Desarrollado por el gremio de alquimistas digitales. YOLOv5 y Streamlit como grimorios principales.")


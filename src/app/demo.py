"""Demo funcional - Predicción de Precios de Viviendas en Boston.

Tarea 8 del Proyecto 1 (Dataset Estático) del curso de MLOps.
Ejecutar con: streamlit run src/app/demo.py
"""

import joblib
import pandas as pd
import streamlit as st

# =============================================================================
# Configuración
# =============================================================================
MODEL_PATH = "models/best_model.joblib"

st.set_page_config(
    page_title="Predicción Precio Viviendas Boston",
    page_icon="🏠",
    layout="centered",
)

# =============================================================================
# Rangos de validación basados en el diccionario de datos (Tarea 2 - EDA)
# Ningún campo acepta valores nulos
# =============================================================================
FEATURE_CONFIG = {
    "crim": {
        "label": "CRIM - Tasa de criminalidad",
        "min": 0.01,
        "max": 89.0,
        "default": 3.61,
        "step": 0.1,
        "help": "Tasa de criminalidad per cápita por zona (0.01 - 89.0)",
    },
    "zn": {
        "label": "ZN - Terreno residencial (%)",
        "min": 0.0,
        "max": 100.0,
        "default": 11.36,
        "step": 1.0,
        "help": "Proporción de terreno residencial para lotes > 25,000 sq.ft (0 - 100)",
    },
    "indus": {
        "label": "INDUS - Zona industrial (%)",
        "min": 0.74,
        "max": 27.74,
        "default": 11.14,
        "step": 0.5,
        "help": "Proporción de acres de negocios no minoristas (0.74 - 27.74)",
    },
    "nox": {
        "label": "NOX - Contaminación (ppm/10M)",
        "min": 0.38,
        "max": 0.87,
        "default": 0.55,
        "step": 0.01,
        "help": "Concentración de óxidos de nitrógeno (0.38 - 0.87)",
    },
    "rm": {
        "label": "RM - Habitaciones promedio",
        "min": 3.5,
        "max": 8.8,
        "default": 6.28,
        "step": 0.1,
        "help": "Número promedio de habitaciones por vivienda (3.5 - 8.8)",
    },
    "age": {
        "label": "AGE - Antigüedad (%)",
        "min": 0.0,
        "max": 100.0,
        "default": 68.57,
        "step": 1.0,
        "help": "Proporción de unidades construidas antes de 1940 (0 - 100)",
    },
    "dis": {
        "label": "DIS - Distancia a centros de empleo",
        "min": 1.1,
        "max": 12.1,
        "default": 3.80,
        "step": 0.1,
        "help": "Distancia ponderada a centros de empleo de Boston (1.1 - 12.1)",
    },
    "tax": {
        "label": "TAX - Impuesto ($10,000)",
        "min": 187.0,
        "max": 711.0,
        "default": 408.0,
        "step": 10.0,
        "help": "Tasa de impuesto a la propiedad por $10,000 (187 - 711)",
    },
    "ptratio": {
        "label": "PTRATIO - Ratio alumnos/profesor",
        "min": 12.6,
        "max": 22.0,
        "default": 18.46,
        "step": 0.1,
        "help": "Ratio alumnos-profesor por zona (12.6 - 22.0)",
    },
    "black": {
        "label": "BLACK - Índice demográfico",
        "min": 0.32,
        "max": 396.9,
        "default": 356.67,
        "step": 5.0,
        "help": "1000(Bk - 0.63)² donde Bk es proporción demográfica (0.32 - 396.9)",
    },
    "lstat": {
        "label": "LSTAT - Estatus socioeconómico bajo (%)",
        "min": 1.73,
        "max": 37.97,
        "default": 12.65,
        "step": 0.5,
        "help": "Porcentaje de población de estatus socioeconómico bajo (1.73 - 37.97)",
    },
}

# Valores válidos para variables categóricas/ordinales
CHAS_OPTIONS = {0: "No", 1: "Sí"}
RAD_VALID_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 24]

# Segmentos de precio para clasificar la predicción
PRICE_SEGMENTS = [
    (15, "Bajo", "info", "< $15,000"),
    (25, "Medio", "success", "$15,000 - $25,000"),
    (35, "Alto", "warning", "$25,000 - $35,000"),
    (float("inf"), "Premium", "error", "> $35,000"),
]

# Promedios del dataset para comparación
FEATURE_AVERAGES = {"rm": 6.28, "lstat": 12.65, "crim": 3.61}


# =============================================================================
# Funciones
# =============================================================================
@st.cache_resource
@st.cache_resource
def load_model() -> object:
    """Carga el modelo entrenado desde disco."""
    return joblib.load(MODEL_PATH)


def get_price_segment(prediction: float) -> tuple[str, str, str]:
    """Clasifica la predicción en un segmento de precio."""
    for threshold, name, style, label in PRICE_SEGMENTS:
        if prediction < threshold:
            return name, style, label
    return "Premium", "error", "> $35k"


def build_input_dataframe(features: dict) -> pd.DataFrame:
    """Construye el DataFrame de entrada para el modelo con tipos correctos."""
    return pd.DataFrame(
        [
            {
                "crim": features["crim"],
                "zn": features["zn"],
                "indus": features["indus"],
                "chas": str(features["chas"]),
                "nox": features["nox"],
                "rm": features["rm"],
                "age": features["age"],
                "dis": features["dis"],
                "rad": float(features["rad"]),
                "tax": features["tax"],
                "ptratio": features["ptratio"],
                "black": features["black"],
                "lstat": features["lstat"],
            }
        ]
    )


# =============================================================================
# UI
# =============================================================================
st.title("🏠 Predicción de Precio de Viviendas en Boston")
st.markdown(
    "Ingrese las características de la zona y la vivienda "
    "para obtener una estimación del precio mediano."
)
st.divider()

# Cargar modelo
try:
    model = load_model()
    st.sidebar.success("Modelo cargado correctamente")
except FileNotFoundError:
    st.error(
        "No se encontró el modelo en la ruta esperada. "
        "Asegúrese de ejecutar la app desde la raíz del proyecto."
    )
    st.stop()

# Sidebar
st.sidebar.title("Información")
st.sidebar.markdown(
    """
    **Proyecto:** MLOps - Dataset Estático

    **Dataset:** Boston Housing

    **Modelo:** Pipeline (preprocessor + modelo)

    **Validaciones:**
    - Rangos basados en el diccionario de datos
    - No se permiten valores nulos
    - Valores por defecto = promedios del dataset
    """
)

# Formulario
st.subheader("📋 Características de la zona")

col1, col2 = st.columns(2)
features = {}

# Columna izquierda
left_features = ["crim", "zn", "indus", "nox", "rm"]
with col1:
    for feat_name in left_features:
        cfg = FEATURE_CONFIG[feat_name]
        features[feat_name] = st.number_input(
            cfg["label"],
            min_value=cfg["min"],
            max_value=cfg["max"],
            value=cfg["default"],
            step=cfg["step"],
            help=cfg["help"],
        )

    features["chas"] = st.selectbox(
        "CHAS - Colinda con río Charles",
        options=list(CHAS_OPTIONS.keys()),
        format_func=lambda x: CHAS_OPTIONS[x],
        help="1 si la zona colinda con el río Charles, 0 en caso contrario",
    )

# Columna derecha
right_features = ["age", "dis", "tax", "ptratio", "black"]
with col2:
    for feat_name in right_features:
        cfg = FEATURE_CONFIG[feat_name]
        features[feat_name] = st.number_input(
            cfg["label"],
            min_value=cfg["min"],
            max_value=cfg["max"],
            value=cfg["default"],
            step=cfg["step"],
            help=cfg["help"],
        )

    features["rad"] = st.selectbox(
        "RAD - Accesibilidad autopistas",
        options=RAD_VALID_VALUES,
        index=3,
        help="Índice de accesibilidad a autopistas radiales (valores válidos: 1-8, 24)",
    )

# lstat ocupa todo el ancho
cfg_lstat = FEATURE_CONFIG["lstat"]
features["lstat"] = st.number_input(
    cfg_lstat["label"],
    min_value=cfg_lstat["min"],
    max_value=cfg_lstat["max"],
    value=cfg_lstat["default"],
    step=cfg_lstat["step"],
    help=cfg_lstat["help"],
)

st.divider()

# Predicción
if st.button("🔮 Predecir Precio", type="primary", use_container_width=True):
    input_data = build_input_dataframe(features)
    prediction = model.predict(input_data)[0]

    # Resultado
    st.subheader("💰 Resultado")
    col_result, col_info = st.columns([1, 1])

    with col_result:
        st.metric(
            label="Precio estimado",
            value=f"${prediction * 1000:,.0f} USD",
            help="Valor mediano estimado en miles de USD",
        )

    with col_info:
        seg_name, seg_style, seg_label = get_price_segment(prediction)
        getattr(st, seg_style)(f"Segmento: **{seg_name}** ({seg_label})")

    # Factores principales
    st.subheader("📊 Factores principales")
    for feat, avg in FEATURE_AVERAGES.items():
        val = features[feat]
        direction = "por encima" if val > avg else "por debajo"
        st.markdown(f"- **{feat.upper()}:** {val:.2f} — {direction} del promedio ({avg})")

    st.caption(
        "Nota: esta es una estimación basada en datos históricos de la década de 1970. "
        "Los precios reales actuales pueden diferir significativamente."
    )

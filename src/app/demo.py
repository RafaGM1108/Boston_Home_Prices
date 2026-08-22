"""Demo funcional - Predicción de Precios de Viviendas en Boston.

Tarea 8 del Proyecto 1 (Dataset Estático) del curso de MLOps.
Ejecutar con: streamlit run src/app/demo.py
"""

import joblib
import pandas as pd
import streamlit as st

# =============================================================================
# Configuración general
# =============================================================================
MODEL_PATH = "models/best_model.joblib"
PRICE_SCALE = 1000  # medv está en miles de USD

st.set_page_config(
    page_title="Predicción de Precio de Viviendas · Boston",
    page_icon="🏠",
    layout="wide",
)

# =============================================================================
# Rangos de validación basados en el diccionario de datos (Tarea 2 - EDA)
# Los sliders impiden por diseño valores fuera de rango y valores nulos
# =============================================================================
FEATURE_CONFIG = {
    "rm": {
        "label": "Habitaciones promedio de la zona",
        "min": 3.5,
        "max": 8.8,
        "default": 6.3,
        "step": 0.1,
        "unit": "hab.",
        "help": (
            "Número promedio de habitaciones por vivienda **en toda la zona**. "
            "Es un promedio del barrio, por eso puede tener decimales "
            "(ej. 6.3 significa que las casas de la zona tienen en promedio "
            "entre 6 y 7 habitaciones). A más habitaciones, mayor precio."
        ),
    },
    "age": {
        "label": "Antigüedad de las viviendas",
        "min": 0.0,
        "max": 100.0,
        "default": 68.0,
        "step": 1.0,
        "unit": "%",
        "help": (
            "Porcentaje de viviendas de la zona construidas antes de 1940. "
            "Un valor alto significa un barrio con construcciones más antiguas."
        ),
    },
    "lstat": {
        "label": "Población de estatus socioeconómico bajo",
        "min": 1.7,
        "max": 38.0,
        "default": 12.6,
        "step": 0.1,
        "unit": "%",
        "help": (
            "Porcentaje de la población de la zona clasificada como de estatus "
            "socioeconómico bajo. **Es el factor que más influye en el precio**: "
            "a mayor porcentaje, menor precio de la vivienda."
        ),
    },
    "crim": {
        "label": "Tasa de criminalidad",
        "min": 0.0,
        "max": 89.0,
        "default": 3.6,
        "step": 0.1,
        "unit": "",
        "help": (
            "Tasa de criminalidad per cápita de la zona. "
            "Valores bajos (cercanos a 0) indican barrios seguros; "
            "valores altos reducen el precio de la vivienda."
        ),
    },
    "ptratio": {
        "label": "Alumnos por profesor",
        "min": 12.6,
        "max": 22.0,
        "default": 18.5,
        "step": 0.1,
        "unit": "",
        "help": (
            "Número de alumnos por cada profesor en las escuelas de la zona. "
            "Es un indicador de calidad educativa: menos alumnos por profesor "
            "suele asociarse con mejores zonas y mayor precio."
        ),
    },
    "tax": {
        "label": "Impuesto a la propiedad",
        "min": 187.0,
        "max": 711.0,
        "default": 408.0,
        "step": 1.0,
        "unit": "por $10k",
        "help": (
            "Tasa de impuesto a la propiedad por cada $10,000 de valor. "
            "Un impuesto más alto tiende a asociarse con precios menores."
        ),
    },
    "nox": {
        "label": "Contaminación del aire",
        "min": 0.38,
        "max": 0.87,
        "default": 0.55,
        "step": 0.01,
        "unit": "ppm",
        "help": (
            "Concentración de óxidos de nitrógeno en el aire (partes por 10 "
            "millones). Mide la contaminación de la zona: más contaminación, "
            "menor precio."
        ),
    },
    "dis": {
        "label": "Distancia a centros de empleo",
        "min": 1.1,
        "max": 12.1,
        "default": 3.8,
        "step": 0.1,
        "unit": "km aprox.",
        "help": (
            "Distancia ponderada a los cinco principales centros de empleo de "
            "Boston. Zonas mejor conectadas con el empleo suelen valer más."
        ),
    },
    "indus": {
        "label": "Suelo industrial",
        "min": 0.7,
        "max": 27.7,
        "default": 11.1,
        "step": 0.1,
        "unit": "%",
        "help": (
            "Porcentaje de la zona dedicado a negocios no minoristas "
            "(industria). Más zona industrial suele reducir el precio "
            "residencial."
        ),
    },
    "zn": {
        "label": "Suelo residencial amplio",
        "min": 0.0,
        "max": 100.0,
        "default": 11.4,
        "step": 1.0,
        "unit": "%",
        "help": (
            "Porcentaje de terreno residencial destinado a lotes grandes "
            "(más de 25,000 pies cuadrados). Indica presencia de viviendas "
            "amplias en la zona."
        ),
    },
    "black": {
        "label": "Índice demográfico (variable histórica)",
        "min": 0.3,
        "max": 396.9,
        "default": 356.7,
        "step": 1.0,
        "unit": "",
        "help": (
            "Variable histórica del dataset original de 1978 que codifica la "
            "composición demográfica de la zona. **Su uso es éticamente "
            "cuestionable** y en un modelo real debería evaluarse su exclusión. "
            "Se mantiene aquí solo por fidelidad al dataset del curso."
        ),
    },
}

# Variables categóricas / ordinales
CHAS_OPTIONS = {0: "No colinda con el río", 1: "Colinda con el río Charles"}
RAD_OPTIONS = {
    1: "1 - Muy baja",
    2: "2 - Baja",
    3: "3 - Baja",
    4: "4 - Media",
    5: "5 - Media",
    6: "6 - Media-alta",
    7: "7 - Alta",
    8: "8 - Alta",
    24: "24 - Máxima (centro/autopistas)",
}

# Segmentos de precio (en miles de USD): umbral, nombre, color, etiqueta
PRICE_SEGMENTS = [
    (15, "Económico", "#3b82f6", "Menos de $15,000"),
    (25, "Medio", "#22c55e", "$15,000 - $25,000"),
    (35, "Alto", "#f59e0b", "$25,000 - $35,000"),
    (float("inf"), "Premium", "#ef4444", "Más de $35,000"),
]

# Promedios del dataset para dar contexto al usuario
FEATURE_AVERAGES = {
    "rm": 6.3,
    "lstat": 12.6,
    "crim": 3.6,
    "ptratio": 18.5,
}


# =============================================================================
# Funciones auxiliares
# =============================================================================
@st.cache_resource
def load_model() -> object:
    """Carga el modelo entrenado desde disco (una sola vez)."""
    return joblib.load(MODEL_PATH)


def get_price_segment(prediction: float) -> tuple[str, str, str]:
    """Clasifica la predicción (en miles USD) en un segmento de precio."""
    for threshold, name, color, label in PRICE_SEGMENTS:
        if prediction < threshold:
            return name, color, label
    return "Premium", "#ef4444", "Más de $35,000"


def slider_for(feature_name: str) -> float:
    """Renderiza un slider para una feature usando su configuración."""
    cfg = FEATURE_CONFIG[feature_name]
    unit = f" {cfg['unit']}" if cfg["unit"] else ""
    return float(
        st.slider(
            f"{cfg['label']}{unit}",
            min_value=cfg["min"],
            max_value=cfg["max"],
            value=cfg["default"],
            step=cfg["step"],
            help=cfg["help"],
        )
    )


def build_input_dataframe(features: dict) -> pd.DataFrame:
    """Construye el DataFrame de entrada para el modelo con los tipos correctos."""
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
# Encabezado
# =============================================================================
st.title("🏠 Estimador de Precio de Viviendas en Boston")
st.markdown(
    "Ajusta las características del barrio y la vivienda con los controles "
    "de abajo. El modelo estimará el **precio mediano** de una vivienda con "
    "esas características. Pasa el cursor sobre el ícono **ⓘ** de cada control "
    "para ver qué significa."
)

# Cargar modelo
try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "No se encontró el modelo entrenado. "
        "Ejecuta la app desde la raíz del proyecto para que encuentre "
        "`models/best_model.joblib`."
    )
    st.stop()

# =============================================================================
# Barra lateral informativa
# =============================================================================
with st.sidebar:
    st.header("Cómo usar la demo")
    st.markdown(
        """
        1. Ajusta los **controles deslizantes** de cada característica.
        2. Cada grupo está organizado por tema (vivienda, entorno, vecindario).
        3. Presiona **Estimar precio** al final.
        4. Verás el precio estimado y en qué segmento cae.
        """
    )
    st.divider()
    st.caption(
        "Proyecto MLOps · Dataset Boston Housing · "
        "El modelo es un pipeline (preprocesamiento + modelo de ML) "
        "entrenado con datos históricos del dataset del curso."
    )

st.divider()

# =============================================================================
# Formulario organizado en tres grupos temáticos
# =============================================================================
features: dict = {}

tab_vivienda, tab_entorno, tab_vecindario = st.tabs(
    ["🏡 La vivienda", "🌳 El entorno", "🏘️ El vecindario"]
)

with tab_vivienda:
    st.caption("Características físicas de las viviendas de la zona.")
    col1, col2 = st.columns(2)
    with col1:
        features["rm"] = slider_for("rm")
    with col2:
        features["age"] = slider_for("age")

with tab_entorno:
    st.caption("Ubicación, conexión y calidad ambiental de la zona.")
    col1, col2 = st.columns(2)
    with col1:
        features["nox"] = slider_for("nox")
        features["dis"] = slider_for("dis")
        features["chas"] = st.selectbox(
            "Cercanía al río Charles",
            options=list(CHAS_OPTIONS.keys()),
            format_func=lambda x: CHAS_OPTIONS[x],
            help=(
                "Indica si la zona colinda con el río Charles. "
                "Las zonas junto al río suelen tener un precio superior."
            ),
        )
    with col2:
        features["indus"] = slider_for("indus")
        features["zn"] = slider_for("zn")
        features["rad"] = st.selectbox(
            "Accesibilidad a autopistas",
            options=list(RAD_OPTIONS.keys()),
            index=3,
            format_func=lambda x: RAD_OPTIONS[x],
            help=(
                "Índice de acceso a las autopistas radiales de Boston. "
                "Un valor más alto indica mejor conexión vial."
            ),
        )

with tab_vecindario:
    st.caption("Indicadores socioeconómicos y de servicios de la zona.")
    col1, col2 = st.columns(2)
    with col1:
        features["lstat"] = slider_for("lstat")
        features["crim"] = slider_for("crim")
        features["tax"] = slider_for("tax")
    with col2:
        features["ptratio"] = slider_for("ptratio")
        features["black"] = slider_for("black")

st.divider()

# =============================================================================
# Predicción
# =============================================================================
_, col_btn, _ = st.columns([1, 2, 1])
with col_btn:
    predict = st.button("🔮 Estimar precio", type="primary", use_container_width=True)

if predict:
    input_data = build_input_dataframe(features)
    prediction = float(model.predict(input_data)[0])
    price_usd = prediction * PRICE_SCALE
    seg_name, seg_color, seg_label = get_price_segment(prediction)

    st.markdown("### Resultado de la estimación")

    col_price, col_seg = st.columns([1, 1])
    with col_price:
        st.metric(
            label="Precio estimado de la vivienda",
            value=f"${price_usd:,.0f}",
            help="Valor mediano estimado para una vivienda con estas características.",
        )
    with col_seg:
        st.markdown(
            f"""
            <div style="
                background-color:{seg_color}22;
                border-left: 6px solid {seg_color};
                padding: 16px 20px;
                border-radius: 8px;">
                <div style="font-size:0.85rem; color:#666;">Segmento de precio</div>
                <div style="font-size:1.4rem; font-weight:700; color:{seg_color};">
                    {seg_name}
                </div>
                <div style="font-size:0.9rem; color:#444;">{seg_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### ¿Por qué este precio?")
    st.caption("Comparación de los factores más influyentes frente al promedio del dataset.")

    for feat, avg in FEATURE_AVERAGES.items():
        val = features[feat]
        cfg = FEATURE_CONFIG[feat]
        diff = val - avg
        arrow = "🔼" if diff > 0 else "🔽" if diff < 0 else "▶️"
        direction = "por encima" if diff > 0 else "por debajo" if diff < 0 else "igual"
        st.markdown(f"{arrow} **{cfg['label']}:** {val:g} ({direction} del promedio de {avg:g})")

    st.info(
        "Esta estimación se basa en el dataset histórico de Boston Housing "
        "usado en el curso. No representa precios de mercado actuales.",
        icon="💡",
    )

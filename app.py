from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="TechGadgets | Retención",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Diseño compacto
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
}
h1 {
    font-size: 1.85rem !important;
    padding-bottom: 0.3rem !important;
}
h3 {
    font-size: 1.05rem !important;
}
[data-testid="stMetric"] {
    border: 1px solid #dce3eb;
    border-radius: 9px;
    padding: 12px;
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem;
}
[data-testid="stMetricLabel"] {
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

DATA_PATH = (
    Path(__file__).parent
    / "data"
    / "base_churn_techgadgets.xlsx"
)
UMBRAL = 0.70


@st.cache_data
def cargar_datos():
    datos = pd.read_excel(
        DATA_PATH,
        sheet_name="Clientes_Churn",
    )
    datos["Score_Churn"] = pd.to_numeric(
        datos["Score_Churn"], errors="raise"
    )
    datos["Churn_90d"] = pd.to_numeric(
        datos["Churn_90d"], errors="raise"
    ).astype(int)
    datos["Alto_Riesgo"] = datos["Score_Churn"] >= UMBRAL
    return datos


def calcular_metricas(datos):
    real = datos["Churn_90d"].eq(1)
    predicho = datos["Alto_Riesgo"]

    vp = int((predicho & real).sum())
    fp = int((predicho & ~real).sum())
    fn = int((~predicho & real).sum())

    precision = vp / (vp + fp) if vp + fp else None
    recall = vp / (vp + fn) if vp + fn else None
    denominador = 2 * vp + fp + fn
    f1 = 2 * vp / denominador if denominador else None

    return precision, recall, f1


def porcentaje(valor):
    return f"{valor:.1%}" if valor is not None else "N/D"


df = cargar_datos()

# Filtros
with st.sidebar:
    st.header("TechGadgets")
    st.caption("Retención de clientes · S09.2")

    canales = sorted(df["Canal"].unique().tolist())
    segmentos = sorted(df["Segmento_RFM"].unique().tolist())

    canal = st.multiselect(
        "Canal",
        options=canales,
        default=canales,
    )
    segmento = st.multiselect(
        "Segmento RFM",
        options=segmentos,
        default=segmentos,
    )

    st.divider()
    st.caption("Datos y scores simulados · Uso académico")

    with st.expander("Reglas y metodología"):
        st.write("Alto riesgo: score ≥ 70 %.")
        st.write("Churn observado: 90 días o más sin comprar.")
        st.write(
            "Precision, Recall y F1 comparan el score simulado "
            "con Churn_90d dentro de los filtros seleccionados. "
            "No son una validación predictiva de un modelo entrenado."
        )
        st.write(
            "Actualización propuesta: semanal, "
            "reemplazando la base de clientes."
        )

datos = df[
    df["Canal"].isin(canal)
    & df["Segmento_RFM"].isin(segmento)
].copy()

st.title("TechGadgets · Priorizar clientes en riesgo")

if datos.empty:
    st.info("Selecciona un canal y un segmento con clientes.")
    st.stop()

precision, recall, f1 = calcular_metricas(datos)
cantidad_riesgo = int(datos["Alto_Riesgo"].sum())

st.caption(
    f"Acción semanal: revisar {cantidad_riesgo} clientes "
    "con score ≥ 70 % para la campaña de reactivación."
)

# Indicadores en una sola fila
indicadores = [
    (
        "Clientes",
        str(len(datos)),
        "Clientes incluidos en los filtros seleccionados.",
    ),
    (
        "Alto riesgo",
        str(cantidad_riesgo),
        "Clientes con score igual o superior al 70 %.",
    ),
    (
        "Churn observado",
        porcentaje(datos["Churn_90d"].mean()),
        "Porcentaje con 90 días o más sin compras.",
    ),
    (
        "Precision",
        porcentaje(precision),
        "De los clasificados como alto riesgo, cuántos tienen churn observado.",
    ),
    (
        "Recall",
        porcentaje(recall),
        "De los clientes con churn observado, cuántos fueron clasificados como alto riesgo.",
    ),
    (
        "F1-Score",
        porcentaje(f1),
        "Equilibrio entre Precision y Recall.",
    ),
]

for columna, (nombre, valor, ayuda) in zip(
    st.columns(6), indicadores
):
    columna.metric(nombre, valor, help=ayuda)

# Gráficos antes del ranking
izquierda, derecha = st.columns(2)

with izquierda:
    st.subheader("Alto riesgo por segmento RFM")

    por_segmento = (
        datos.groupby("Segmento_RFM")["Alto_Riesgo"]
        .sum()
        .astype(int)
        .sort_values(ascending=False)
        .rename("Clientes")
    )

    st.bar_chart(
        por_segmento,
        horizontal=True,
        color="#E07A36",
        height=260,
        x_label="Clientes de alto riesgo",
        y_label="Segmento RFM",
        sort=False,
    )

with derecha:
    st.subheader("Riesgo y abandono por canal")

    por_canal = (
        datos.groupby("Canal")
        .agg(
            riesgo=("Alto_Riesgo", "sum"),
            abandono=("Churn_90d", "sum"),
        )
        .astype(int)
        .rename(columns={
            "riesgo": "Alto riesgo",
            "abandono": "Churn observado",
        })
    )

    st.bar_chart(
        por_canal,
        horizontal=True,
        stack=False,
        color=["#E07A36", "#2874A6"],
        height=260,
        x_label="Clientes",
        y_label="Canal",
    )

# Ranking: este filtro solo afecta la tabla
titulo, control = st.columns([3, 1])

with titulo:
    st.subheader("Ranking y acción recomendada")

with control:
    solo_alto = st.checkbox("Solo alto riesgo", value=True)

ranking = datos.sort_values(
    ["Score_Churn", "Cliente_ID"],
    ascending=[False, True],
).copy()

if solo_alto:
    ranking = ranking[ranking["Alto_Riesgo"]].copy()

ranking["Score (%)"] = (ranking["Score_Churn"] * 100).round(1)

columnas = [
    "Cliente_ID",
    "Score (%)",
    "Nivel_Riesgo",
    "Segmento_RFM",
    "Canal",
    "Accion_Recomendada",
]

tabla = ranking[columnas].rename(columns={
    "Cliente_ID": "Cliente",
    "Nivel_Riesgo": "Riesgo",
    "Segmento_RFM": "Segmento RFM",
    "Accion_Recomendada": "Acción recomendada",
})

if tabla.empty:
    st.info("No hay clientes de alto riesgo en esta selección.")
else:
    st.dataframe(
        tabla,
        hide_index=True,
        use_container_width=True,
        height=250,
        column_config={
            "Score (%)": st.column_config.ProgressColumn(
                "Score (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%",
                width="medium",
            ),
            "Acción recomendada": st.column_config.TextColumn(
                "Acción recomendada",
                width="large",
            ),
        },
    )

    st.download_button(
        "Descargar clientes seleccionados",
        data=tabla.to_csv(index=False).encode("utf-8-sig"),
        file_name="clientes_campana_techgadgets.csv",
        mime="text/csv",
    )

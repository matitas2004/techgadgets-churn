from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Riesgo de abandono | TechGadgets",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "data" / "base_churn_techgadgets.xlsx"
THRESHOLD = 0.70

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH, sheet_name="Clientes_Churn")
    df["Fecha_Ultima_Compra"] = pd.to_datetime(df["Fecha_Ultima_Compra"])
    df["Score_Churn"] = pd.to_numeric(df["Score_Churn"], errors="coerce")
    df["Churn_90d"] = pd.to_numeric(df["Churn_90d"], errors="coerce").fillna(0).astype(int)
    df["Alto_Riesgo"] = df["Score_Churn"] >= THRESHOLD
    return df

def score_metrics(data):
    actual = data["Churn_90d"].astype(int)
    predicted = data["Alto_Riesgo"].astype(int)
    tp = int(((predicted == 1) & (actual == 1)).sum())
    fp = int(((predicted == 1) & (actual == 0)).sum())
    fn = int(((predicted == 0) & (actual == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return precision, recall, f1

def show_pct(value):
    return f"{value:.1%}" if value is not None else "N/D"

df = load_data()

st.title("Priorizar esta semana a los clientes con mayor riesgo de abandono")
st.caption("TechGadgets | Tablero académico de churn y acciones de retención")
st.info(
    "Los clientes y los scores de esta base son simulados. El tablero muestra cómo se priorizaría una campaña; "
    "no representa un modelo entrenado para producción."
)

with st.sidebar:
    st.header("Filtros")
    channels = sorted(df["Canal"].dropna().unique().tolist())
    segments = sorted(df["Segmento_RFM"].dropna().unique().tolist())
    selected_channels = st.multiselect("Canal", channels, default=channels)
    selected_segments = st.multiselect("Segmento RFM", segments, default=segments)
    only_high = st.checkbox("Filtrar la tabla a alto riesgo (score >= 70 %)", value=False)

filtered = df[
    df["Canal"].isin(selected_channels)
    & df["Segmento_RFM"].isin(selected_segments)
].copy()
precision, recall, f1 = score_metrics(filtered)
risk_count = int(filtered["Alto_Riesgo"].sum())
churn_rate = filtered["Churn_90d"].mean() if len(filtered) else None

st.subheader("Indicadores del grupo filtrado")
k1, k2, k3 = st.columns(3)
k1.metric("Clientes analizados", f"{len(filtered)}")
k2.metric("Clientes de alto riesgo", f"{risk_count}")
k3.metric("Churn observado", show_pct(churn_rate))
k4, k5, k6 = st.columns(3)
k4.metric("Precision", show_pct(precision), help="De los marcados como alto riesgo, cuántos tienen Churn_90d = 1.")
k5.metric("Recall", show_pct(recall), help="De los clientes con Churn_90d = 1, cuántos detecta el umbral.")
k6.metric("F1-Score", show_pct(f1), help="Equilibrio entre Precision y Recall.")

st.subheader("Clientes priorizados")
st.write("Ordenados por Score_Churn, de mayor a menor. La acción aparece junto a cada cliente.")
ranking = filtered.sort_values("Score_Churn", ascending=False).copy()
if only_high:
    ranking = ranking[ranking["Alto_Riesgo"]]
ranking["Score_Churn"] = (ranking["Score_Churn"] * 100).round(1).astype(str) + " %"
show_columns = [
    "Cliente_ID", "Score_Churn", "Nivel_Riesgo", "Segmento_RFM", "Canal",
    "Fecha_Ultima_Compra", "Accion_Recomendada",
]
st.dataframe(
    ranking[show_columns],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Cliente_ID": "Cliente",
        "Score_Churn": "Score de churn",
        "Nivel_Riesgo": "Nivel de riesgo",
        "Segmento_RFM": "Segmento RFM",
        "Fecha_Ultima_Compra": st.column_config.DateColumn("Última compra", format="DD/MM/YYYY"),
        "Accion_Recomendada": "Acción recomendada",
    },
)

csv_bytes = ranking[show_columns].to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Descargar lista filtrada",
    data=csv_bytes,
    file_name="clientes_priorizados_techgadgets.csv",
    mime="text/csv",
)

left, right = st.columns(2)
with left:
    st.subheader("Clientes de alto riesgo por segmento RFM")
    by_segment = (
        filtered.assign(Grupo_Riesgo=filtered["Alto_Riesgo"].map({True: "Alto riesgo", False: "Bajo/medio"}))
        .groupby(["Segmento_RFM", "Grupo_Riesgo"])
        .size()
        .unstack(fill_value=0)
    )
    st.bar_chart(by_segment, height=300)
with right:
    st.subheader("Alto riesgo y churn observado por canal")
    by_channel = (
        filtered.assign(Alto_Riesgo_n=filtered["Alto_Riesgo"].astype(int))
        .groupby("Canal")
        .agg(
            Clientes_alto_riesgo=("Alto_Riesgo_n", "sum"),
            Churn_observado=("Churn_90d", "sum"),
        )
    )
    st.bar_chart(by_channel, height=300)

st.caption(
    "Regla del tablero: Score_Churn >= 0,70 = alto riesgo. "
    "Las métricas se calculan con la etiqueta simulada Churn_90d y el grupo filtrado."
)

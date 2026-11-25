"""
Dashboard Interactivo en Streamlit
Proyecto Integrador: Segmentación de Perfiles de Obesidad mediante K-Means

Cómo ejecutarlo:
    streamlit run dashboard/app.py
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# Configuración básica de la página
st.set_page_config(
    page_title="Obesidad K-Means | Panel Biométrico",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Rutas de los archivos generados por codigo/proyecto_obesidad_kmeans.py
carpeta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ruta_css = os.path.join(carpeta_base, "dashboard", "estilos.css")
ruta_datos = os.path.join(carpeta_base, "datos", "procesados", "obesidad_con_clusters.csv")
ruta_centroides = os.path.join(carpeta_base, "datos", "procesados", "centroides.csv")
ruta_metricas = os.path.join(carpeta_base, "datos", "procesados", "metricas_k.csv")
ruta_comparacion = os.path.join(carpeta_base, "datos", "procesados", "comparacion_modelos.csv")

# Cargar hoja de estilos personalizada y animaciones
if os.path.exists(ruta_css):
    with open(ruta_css, encoding="utf-8") as archivo_css:
        st.markdown(f"<style>{archivo_css.read()}</style>", unsafe_allow_html=True)

# Cargar datos procesados
if not os.path.exists(ruta_datos):
    st.error("Primero ejecuta: python codigo/proyecto_obesidad_kmeans.py")
    st.stop()

df = pd.read_csv(ruta_datos)
df_centroides = pd.read_csv(ruta_centroides)
df_metricas = pd.read_csv(ruta_metricas)
df_comparacion = pd.read_csv(ruta_comparacion)

# Colores luminosos de alto contraste para fondo oscuro
orden_clusters = [
    "Jóvenes Activos",
    "Jóvenes Sedentarios",
    "Adultos con Sobrepeso",
    "Obesidad Severa Temprana",
]
colores = {
    "Jóvenes Activos": "#34D399",
    "Jóvenes Sedentarios": "#38BDF8",
    "Adultos con Sobrepeso": "#FBBF24",
    "Obesidad Severa Temprana": "#FB7185",
}


# Funciones simples de apoyo para mantener el mismo estilo visual en todo el panel
def estilo_grafica(fig, titulo):
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=titulo,
            x=0,
            y=0.97,
            font=dict(family="Space Grotesk, sans-serif", size=15, color="#F1F5F9"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.55)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#CBD5E1", size=12),
        margin=dict(l=24, r=36, t=82, b=32),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            title_text="",
            font=dict(size=11, color="#CBD5E1"),
        ),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="#2DD4BF", font_color="#F1F5F9"),
    )
    fig.update_xaxes(
        automargin=True,
        gridcolor="rgba(148, 163, 184, 0.14)",
        zerolinecolor="rgba(148, 163, 184, 0.2)",
    )
    fig.update_yaxes(
        automargin=True,
        gridcolor="rgba(148, 163, 184, 0.14)",
        zerolinecolor="rgba(148, 163, 184, 0.2)",
    )
    return fig


def tarjeta_kpi(etiqueta, valor, subtitulo, icono, color_borde, retraso_ms=0):
    return f"""
    <div class="kpi-card" style="--kpi-accent: {color_borde}; --delay: {retraso_ms}ms;">
      <div class="kpi-header">
        <span>{etiqueta}</span>
        <span class="kpi-icon" aria-hidden="true">{icono}</span>
      </div>
      <div class="kpi-value">{valor}</div>
      <div class="kpi-foot">{subtitulo}</div>
    </div>
    """


def titulo_seccion(texto):
    st.markdown(
        f'<div class="section-heading"><span class="section-heading-bar"></span><h2>{texto}</h2></div>',
        unsafe_allow_html=True,
    )


# ====================================================================
# BARRA LATERAL: NAVEGACIÓN Y 3 FILTROS INTERACTIVOS
# ====================================================================
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
      <div class="sidebar-brand-eyebrow">Panel Biométrico · K-Means</div>
      <p class="sidebar-brand-title">Obesidad &amp; Hábitos</p>
    </div>
    """,
    unsafe_allow_html=True,
)

pagina_inicial = 1 if st.query_params.get("pagina") == "2" else 0
pagina = st.sidebar.radio(
    "Vista activa:",
    ["Página 1: Análisis general", "Página 2: Segmentación mediante K-Means"],
    index=pagina_inicial,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros interactivos")

# Filtro 1: Clúster / Clasificación
clusters_sel = st.sidebar.multiselect(
    "1. Clúster (perfil):",
    options=orden_clusters,
    default=orden_clusters,
)

# Filtro 2: Género
generos_disponibles = sorted(df["Genero"].unique().tolist())
generos_sel = st.sidebar.multiselect(
    "2. Género:",
    options=generos_disponibles,
    default=generos_disponibles,
)

# Filtro 3: Rango de edad
edad_min = int(df["Age"].min())
edad_max = int(df["Age"].max())
rango_edad = st.sidebar.slider(
    "3. Rango de edad (años):",
    min_value=edad_min,
    max_value=edad_max,
    value=(edad_min, edad_max),
)

# Aplicar los 3 filtros al dataframe
df_filtrado = df[
    (df["Clasificacion"].isin(clusters_sel))
    & (df["Genero"].isin(generos_sel))
    & (df["Age"] >= rango_edad[0])
    & (df["Age"] <= rango_edad[1])
]

# Indicador visual de cobertura filtrada en la barra lateral
pct_muestra = len(df_filtrado) / len(df) * 100
st.sidebar.markdown(
    f"""
    <div class="sidebar-meter">
      <div class="sidebar-meter-label">
        <span>Cobertura actual</span>
        <strong>{pct_muestra:.1f}% ({len(df_filtrado):,})</strong>
      </div>
      <div class="sidebar-meter-track">
        <div class="sidebar-meter-fill" style="width: {pct_muestra:.1f}%;"></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if df_filtrado.empty:
    st.warning("No hay registros que coincidan con los filtros seleccionados. Ajusta los filtros en la barra lateral.")
    st.stop()


# ====================================================================
# ENCABEZADO HERO PRINCIPAL
# ====================================================================
etiqueta_vista = "VISTA 01 · ANÁLISIS GENERAL" if pagina.startswith("Página 1") else "VISTA 02 · SEGMENTACIÓN K-MEANS"
st.markdown(
    f"""
    <div class="hero-banner">
      <div class="hero-badge">● {etiqueta_vista} · K = 4 CLÚSTERES · SILHOUETTE 0.3552</div>
      <h1 class="hero-title">Segmentación de Perfiles de Obesidad y Hábitos</h1>
      <p class="hero-subtitle">
        Exploración interactiva de <strong>{len(df_filtrado):,}</strong> registros filtrados (de {len(df):,} limpios)
        usando <strong>IMC</strong>, <strong>Edad</strong> y <strong>Actividad Física (FAF)</strong>.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ====================================================================
# PÁGINA 1: ANÁLISIS GENERAL DEL DATASET
# ====================================================================
if pagina == "Página 1: Análisis general":
    # 4 indicadores KPI generales con entrada escalonada
    k1, k2, k3, k4 = st.columns(4)
    pct_ant = df_filtrado["Antecedentes_Num"].mean() * 100

    with k1:
        st.markdown(
            tarjeta_kpi("Total de registros", f"{len(df_filtrado):,}", f"{pct_muestra:.1f}% del dataset limpio", "◆", "#2DD4BF", 0),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            tarjeta_kpi("IMC promedio", f"{df_filtrado['IMC'].mean():.2f}", "kg/m² (límite obesidad: 30)", "▲", "#38BDF8", 80),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            tarjeta_kpi("Edad promedio", f"{df_filtrado['Age'].mean():.1f} años", f"Rango: {df_filtrado['Age'].min()} a {df_filtrado['Age'].max()} años", "●", "#A78BFA", 160),
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            tarjeta_kpi("Antecedentes familiares", f"{pct_ant:.1f}%", "Con historial de sobrepeso", "■", "#FB7185", 240),
            unsafe_allow_html=True,
        )

    titulo_seccion("Distribución corporal y niveles de obesidad")
    col1, col2 = st.columns(2)

    with col1:
        fig_imc = px.histogram(
            df_filtrado,
            x="IMC",
            nbins=26,
            color_discrete_sequence=["#2DD4BF"],
            labels={"IMC": "IMC (kg/m²)", "count": "Personas"},
        )
        fig_imc.update_traces(marker_line_width=0.6, marker_line_color="#0B1120", opacity=0.9)
        fig_imc.add_vline(
            x=25, line_dash="dash", line_color="#FBBF24",
            annotation_text="Sobrepeso (25)", annotation_position="top left",
            annotation_font_color="#FBBF24",
        )
        fig_imc.add_vline(
            x=30, line_dash="dash", line_color="#FB7185",
            annotation_text="Obesidad (30)", annotation_position="top right",
            annotation_font_color="#FB7185",
        )
        st.plotly_chart(estilo_grafica(fig_imc, "Distribución del Índice de Masa Corporal (IMC)"), use_container_width=True)

    with col2:
        orden_niveles = [
            "Peso Insuficiente",
            "Peso Normal",
            "Sobrepeso Nivel I",
            "Sobrepeso Nivel II",
            "Obesidad Tipo I",
            "Obesidad Tipo II",
            "Obesidad Tipo III",
        ]
        conteo_niv = (
            df_filtrado["Nivel_Obesidad"]
            .value_counts()
            .reindex(orden_niveles, fill_value=0)
            .reset_index()
        )
        conteo_niv.columns = ["Nivel_Obesidad", "Cantidad"]
        fig_niv = px.bar(
            conteo_niv,
            x="Cantidad",
            y="Nivel_Obesidad",
            orientation="h",
            text="Cantidad",
            color="Cantidad",
            color_continuous_scale=["#1E3A5F", "#2DD4BF", "#A78BFA"],
            labels={"Nivel_Obesidad": "Nivel de obesidad", "Cantidad": "Personas"},
        )
        fig_niv.update_traces(textposition="outside", marker_line_width=0)
        fig_niv.update_layout(coloraxis_showscale=False, xaxis_range=[0, conteo_niv["Cantidad"].max() * 1.15])
        st.plotly_chart(estilo_grafica(fig_niv, "Cantidad de Personas por Nivel de Obesidad"), use_container_width=True)

    titulo_seccion("Relaciones físicas y actividad por género")
    col3, col4 = st.columns(2)

    with col3:
        fig_peso = px.scatter(
            df_filtrado,
            x="Height",
            y="Weight",
            color="Genero",
            opacity=0.72,
            color_discrete_map={"Femenino": "#C084FC", "Masculino": "#38BDF8"},
            labels={"Height": "Estatura (m)", "Weight": "Peso (kg)", "Genero": "Género"},
            hover_data=["Age", "IMC", "Nivel_Obesidad"],
        )
        fig_peso.update_traces(marker=dict(size=7, line=dict(width=0.4, color="#0B1120")))
        st.plotly_chart(estilo_grafica(fig_peso, "Relación entre Estatura y Peso por Género"), use_container_width=True)

    with col4:
        fig_faf = px.histogram(
            df_filtrado,
            x="FAF",
            color="Genero",
            barmode="overlay",
            nbins=15,
            opacity=0.72,
            color_discrete_map={"Femenino": "#C084FC", "Masculino": "#38BDF8"},
            labels={"FAF": "Actividad física (0 = Nada, 3 = Alta)", "count": "Personas", "Genero": "Género"},
        )
        st.plotly_chart(estilo_grafica(fig_faf, "Frecuencia de Actividad Física (FAF) por Género"), use_container_width=True)

    titulo_seccion("Estadísticas descriptivas generales")
    cols_est = ["Age", "Height", "Weight", "IMC", "FAF", "CH2O", "TUE", "FCVC"]
    st.dataframe(df_filtrado[cols_est].describe().round(2), use_container_width=True)


# ====================================================================
# PÁGINA 2: SEGMENTACIÓN MEDIANTE K-MEANS
# ====================================================================
else:
    # 4 indicadores KPI relevantes de la segmentación
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            tarjeta_kpi("Personas segmentadas", f"{len(df_filtrado):,}", "Asignadas con fit_predict()", "◆", "#2DD4BF", 0),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            tarjeta_kpi("Clústeres activos", f"{df_filtrado['Clasificacion'].nunique()} de 4", "K = 4 óptimo (9 iteraciones)", "●", "#A78BFA", 80),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            tarjeta_kpi("IMC promedio", f"{df_filtrado['IMC'].mean():.2f}", "kg/m² en la selección actual", "▲", "#FBBF24", 160),
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            tarjeta_kpi("Actividad física (FAF)", f"{df_filtrado['FAF'].mean():.2f} / 3.0", "Promedio semanal filtrado", "■", "#34D399", 240),
            unsafe_allow_html=True,
        )

    titulo_seccion("Distribución y mapa interactivo de agrupamientos")
    col1, col2 = st.columns([1, 1.45])

    with col1:
        conteo_cl = (
            df_filtrado["Clasificacion"]
            .value_counts()
            .reindex([c for c in orden_clusters if c in clusters_sel], fill_value=0)
            .reset_index()
        )
        conteo_cl.columns = ["Clasificacion", "Cantidad"]
        conteo_cl["Porcentaje"] = (conteo_cl["Cantidad"] / len(df_filtrado) * 100).round(1)

        fig_dist = px.bar(
            conteo_cl,
            x="Clasificacion",
            y="Cantidad",
            color="Clasificacion",
            text=conteo_cl.apply(lambda r: f"{r['Cantidad']}<br>({r['Porcentaje']}%)", axis=1),
            color_discrete_map=colores,
            labels={"Clasificacion": "Clúster", "Cantidad": "Personas"},
        )
        fig_dist.update_traces(textposition="outside", marker_line_width=0)
        fig_dist.update_layout(showlegend=False, yaxis_range=[0, conteo_cl["Cantidad"].max() * 1.24])
        fig_dist.update_xaxes(
            tickangle=0,
            tickfont=dict(size=10),
            tickvals=orden_clusters,
            ticktext=["Jóvenes<br>Activos", "Jóvenes<br>Sedentarios", "Adultos<br>Sobrepeso", "Obesidad<br>Severa"],
        )
        st.plotly_chart(estilo_grafica(fig_dist, "Distribución de Registros por Clúster"), use_container_width=True)

    with col2:
        vista_disp = st.selectbox(
            "Ejes en la gráfica de dispersión:",
            ["IMC vs Actividad Física (FAF)", "Edad vs IMC", "Proyección PCA 2D"],
        )

        if vista_disp == "IMC vs Actividad Física (FAF)":
            eje_x, eje_y = "IMC", "FAF"
            lbl_x, lbl_y = "IMC (kg/m²)", "Actividad física (FAF)"
            cx, cy = "IMC", "FAF"
        elif vista_disp == "Edad vs IMC":
            eje_x, eje_y = "Age", "IMC"
            lbl_x, lbl_y = "Edad (años)", "IMC (kg/m²)"
            cx, cy = "Age", "IMC"
        else:
            eje_x, eje_y = "PCA1", "PCA2"
            lbl_x, lbl_y = "Componente Principal 1", "Componente Principal 2"
            cx, cy = None, None

        fig_disp = px.scatter(
            df_filtrado,
            x=eje_x,
            y=eje_y,
            color="Clasificacion",
            category_orders={"Clasificacion": orden_clusters},
            color_discrete_map=colores,
            opacity=0.76,
            labels={eje_x: lbl_x, eje_y: lbl_y, "Clasificacion": "Clúster"},
            hover_data=["Age", "Weight", "IMC", "FAF", "Nivel_Obesidad"],
        )
        fig_disp.update_traces(marker=dict(size=7, line=dict(width=0.4, color="#0B1120")))

        # Agregar centroides luminosos si la vista es en variables originales
        if cx is not None:
            cent_vis = df_centroides[df_centroides["Clasificacion"].isin(clusters_sel)]
            fig_disp.add_trace(
                go.Scatter(
                    x=cent_vis[cx],
                    y=cent_vis[cy],
                    mode="markers",
                    marker=dict(symbol="x", size=15, color="#FFFFFF", line=dict(width=2, color="#0B1120")),
                    name="Centroides",
                )
            )
        st.plotly_chart(estilo_grafica(fig_disp, f"Agrupamientos K-Means: {vista_disp}"), use_container_width=True)

    # Fila 2: Dos gráficas adicionales para comparar características de los grupos
    titulo_seccion("Comparación de características entre clústeres")
    col3, col4 = st.columns(2)

    with col3:
        fig_box = px.box(
            df_filtrado,
            x="Clasificacion",
            y="IMC",
            color="Clasificacion",
            category_orders={"Clasificacion": orden_clusters},
            color_discrete_map=colores,
            labels={"Clasificacion": "Clúster", "IMC": "IMC (kg/m²)"},
        )
        fig_box.update_layout(showlegend=False)
        fig_box.update_xaxes(
            tickangle=0,
            tickvals=orden_clusters,
            ticktext=["Jóvenes<br>Activos", "Jóvenes<br>Sedentarios", "Adultos con<br>Sobrepeso", "Obesidad Severa<br>Temprana"],
        )
        st.plotly_chart(estilo_grafica(fig_box, "Dispersión del IMC por Clúster (Boxplot)"), use_container_width=True)

    with col4:
        habitos_cl = (
            df_filtrado.groupby("Clasificacion")[["FAF", "CH2O", "TUE"]]
            .mean()
            .reindex([c for c in orden_clusters if c in clusters_sel])
            .dropna()
            .reset_index()
        )
        habitos_largo = habitos_cl.melt(
            id_vars="Clasificacion",
            var_name="Hábito",
            value_name="Promedio",
        )
        habitos_largo["Hábito"] = habitos_largo["Hábito"].map({
            "FAF": "Actividad (FAF)",
            "CH2O": "Agua (CH2O)",
            "TUE": "Tecnología (TUE)",
        })
        fig_hab = px.bar(
            habitos_largo,
            x="Clasificacion",
            y="Promedio",
            color="Hábito",
            barmode="group",
            color_discrete_map={
                "Actividad (FAF)": "#2DD4BF",
                "Agua (CH2O)": "#38BDF8",
                "Tecnología (TUE)": "#A78BFA",
            },
            labels={"Clasificacion": "Clúster", "Promedio": "Valor promedio"},
        )
        fig_hab.update_xaxes(
            tickangle=0,
            tickvals=orden_clusters,
            ticktext=["Jóvenes<br>Activos", "Jóvenes<br>Sedentarios", "Adultos con<br>Sobrepeso", "Obesidad Severa<br>Temprana"],
        )
        st.plotly_chart(estilo_grafica(fig_hab, "Comparación de Hábitos Promedio por Clúster"), use_container_width=True)

    # Tabla con estadísticas descriptivas de los clústeres
    titulo_seccion("Estadísticas descriptivas por clúster")
    tabla_cl = (
        df_filtrado.groupby("Clasificacion")
        .agg(
            Registros=("IMC", "count"),
            IMC_Prom=("IMC", "mean"),
            IMC_Min=("IMC", "min"),
            IMC_Max=("IMC", "max"),
            Peso_kg=("Weight", "mean"),
            Edad=("Age", "mean"),
            FAF=("FAF", "mean"),
            Agua=("CH2O", "mean"),
            Antec_Pct=("Antecedentes_Num", lambda s: s.mean() * 100),
            Calorica_Pct=("ComidaCalorica_Num", lambda s: s.mean() * 100),
        )
        .reindex([c for c in orden_clusters if c in clusters_sel])
        .dropna(how="all")
        .round(2)
    )
    st.dataframe(tabla_cl, use_container_width=True)

    # Evaluación de K y comparación de configuraciones
    with st.expander("Ver métricas de evaluación de K (2 a 7) y comparación de configuraciones"):
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**Evaluación de valores de K (Inercia y Silhouette)**")
            st.dataframe(df_metricas, use_container_width=True, hide_index=True)
        with m2:
            st.markdown("**Comparación de configuraciones del modelo**")
            st.dataframe(df_comparacion, use_container_width=True, hide_index=True)

    # Sección con la descripción de los grupos identificados (tarjetas con icono + texto + mini métricas)
    titulo_seccion("Descripción de los 4 perfiles identificados")
    d1, d2 = st.columns(2)

    with d1:
        st.markdown(
            """
            <div class="profile-card" style="--card-color: #34D399; --delay: 0ms;">
              <div class="profile-top">
                <h3 class="profile-title">1. Jóvenes Activos</h3>
                <span class="profile-pill">● Riesgo Bajo · 20.7%</span>
              </div>
              <div class="profile-metrics">
                <span class="mini-stat">IMC 23.28</span>
                <span class="mini-stat">20.9 años</span>
                <span class="mini-stat">FAF 2.18</span>
                <span class="mini-stat">431 personas</span>
              </div>
              <p class="profile-desc">
                Peso corporal saludable y la frecuencia de actividad física más alta del estudio.
                Destacan por buena hidratación (2.14) y el menor consumo de alimentos altos en calorías (77.5%).
              </p>
            </div>

            <div class="profile-card" style="--card-color: #38BDF8; --delay: 100ms;">
              <div class="profile-top">
                <h3 class="profile-title">2. Jóvenes Sedentarios</h3>
                <span class="profile-pill">▲ Prevención · 32.9%</span>
              </div>
              <div class="profile-metrics">
                <span class="mini-stat">IMC 24.80</span>
                <span class="mini-stat">20.9 años</span>
                <span class="mini-stat">FAF 0.54</span>
                <span class="mini-stat">686 personas</span>
              </div>
              <p class="profile-desc">
                Misma edad y peso casi idéntico (~70 kg) que el grupo activo, pero con cuatro veces menos
                actividad física e IMC en el umbral del sobrepeso. Es el grupo más grande y clave para prevención temprana.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with d2:
        st.markdown(
            """
            <div class="profile-card" style="--card-color: #FBBF24; --delay: 180ms;">
              <div class="profile-top">
                <h3 class="profile-title">3. Adultos con Sobrepeso</h3>
                <span class="profile-pill">◆ Riesgo Moderado · 14.5%</span>
              </div>
              <div class="profile-metrics">
                <span class="mini-stat">IMC 30.13</span>
                <span class="mini-stat">36.7 años</span>
                <span class="mini-stat">FAF 0.95</span>
                <span class="mini-stat">303 personas</span>
              </div>
              <p class="profile-desc">
                Clúster de mayor edad (29 a 61 años), con sobrepeso alto u obesidad grado I, actividad física
                moderada-baja y el menor uso de dispositivos recreativos (TUE = 0.29).
              </p>
            </div>

            <div class="profile-card" style="--card-color: #FB7185; --delay: 260ms;">
              <div class="profile-top">
                <h3 class="profile-title">4. Obesidad Severa Temprana</h3>
                <span class="profile-pill">■ Riesgo Alto · 32.0%</span>
              </div>
              <div class="profile-metrics">
                <span class="mini-stat">IMC 38.91</span>
                <span class="mini-stat">24.6 años</span>
                <span class="mini-stat">FAF 0.78</span>
                <span class="mini-stat">667 personas</span>
              </div>
              <p class="profile-desc">
                Adultos jóvenes con el peso promedio más alto (115.7 kg) y baja actividad física.
                El 99.6% cuenta con antecedentes familiares de sobrepeso y el 98.5% consume comida calórica frecuentemente.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Sección con los principales hallazgos y recomendaciones
    titulo_seccion("Principales hallazgos y recomendaciones")
    h1, h2 = st.columns(2)

    with h1:
        st.markdown(
            """
            <div class="insight-panel">
              <h3><span aria-hidden="true">◆</span> Hallazgos relevantes</h3>
              <div class="insight-item">
                <span class="insight-num" style="--item-color: #FB7185;">01</span>
                <div class="insight-text">
                  <strong>Obesidad severa en población joven:</strong> El clúster de mayor riesgo tiene apenas 24.6 años promedio, combinando 99.6% de antecedentes familiares y 98.5% de dieta altamente calórica.
                </div>
              </div>
              <div class="insight-item">
                <span class="insight-num" style="--item-color: #38BDF8;">02</span>
                <div class="insight-text">
                  <strong>El sedentarismo divide a los jóvenes:</strong> Activos y sedentarios comparten edad (~21 años) y peso (~69–70 kg), pero cuatro veces menos ejercicio desplaza al grupo hacia el sobrepeso.
                </div>
              </div>
              <div class="insight-item">
                <span class="insight-num" style="--item-color: #FBBF24;">03</span>
                <div class="insight-text">
                  <strong>Menor tiempo en pantallas no evita el sobrepeso adulto:</strong> El grupo de ~37 años usa menos tecnología (TUE = 0.29), pero mantiene un IMC de 30.13 por poca actividad y dieta calórica (90.1%).
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h2:
        st.markdown(
            """
            <div class="insight-panel">
              <h3><span aria-hidden="true">▲</span> Recomendaciones basadas en los datos</h3>
              <div class="insight-item">
                <span class="insight-num" style="--item-color: #34D399;">01</span>
                <div class="insight-text">
                  <strong>Prevención en Jóvenes Sedentarios:</strong> Impulsar rutinas accesibles de 30 minutos diarios de ejercicio en universidades y trabajos antes de que progresen a obesidad.
                </div>
              </div>
              <div class="insight-item">
                <span class="insight-num" style="--item-color: #2DD4BF;">02</span>
                <div class="insight-text">
                  <strong>Intervención integral en Obesidad Severa Temprana:</strong> Priorizar acompañamiento médico-nutricional enfocado en reducir comida calórica, iniciar ejercicio de bajo impacto e incluir a la familia.
                </div>
              </div>
              <div class="insight-item">
                <span class="insight-num" style="--item-color: #A78BFA;">03</span>
                <div class="insight-text">
                  <strong>Seguimiento en Adultos con Sobrepeso:</strong> Fomentar caminatas diarias, chequeos periódicos de IMC y planes alimenticios balanceados adaptados a la jornada laboral.
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

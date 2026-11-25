import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA


# Estilo sencillo para todas las gráficas
sns.set_style("whitegrid")

# Rutas principales del proyecto (funciona desde la raíz o desde codigo/)
carpeta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ruta_original = os.path.join(
    carpeta_base, "datos", "original", "ObesityDataSet_raw_and_data_sinthetic.csv"
)
carpeta_procesados = os.path.join(carpeta_base, "datos", "procesados")
carpeta_graficas = os.path.join(carpeta_base, "graficas")

os.makedirs(carpeta_procesados, exist_ok=True)
os.makedirs(carpeta_graficas, exist_ok=True)


# ====================================================================
# PARTE 2. OBTENCIÓN Y EXPLORACIÓN DEL DATASET
# ====================================================================
print("============================================================")
print("PARTE 2. OBTENCIÓN Y EXPLORACIÓN DEL DATASET")
print("============================================================")
print("Fuente: UCI Machine Learning Repository / Kaggle")
print("Dataset: Estimation of obesity levels based on eating habits and physical condition")
print("Origen: Encuestas en México, Perú y Colombia (Palechor y Manotas, 2019)\n")

# Importar el dataset con read_csv()
df = pd.read_csv(ruta_original)

print("Primeros 10 registros:")
print(df.head(10))

print("\nDimensiones del dataset (filas, columnas):")
print(df.shape)

print("\nColumnas y tipos de datos:")
print(df.dtypes)

print("\nEstadísticas descriptivas:")
print(df.describe().round(2))

print("\nValores nulos por columna:")
print(df.isnull().sum())

print("\nCantidad de registros duplicados:")
print(df.duplicated().sum())

# Calculamos el IMC de manera preliminar para las gráficas exploratorias
df["IMC"] = df["Weight"] / (df["Height"] ** 2)

# --- Gráfica exploratoria 1: Distribución del IMC y del Peso ---
fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))

sns.histplot(df["IMC"], bins=25, kde=True, color="#2E86AB", ax=ejes[0])
ejes[0].axvline(25, color="orange", linestyle="--", label="Límite Sobrepeso (25)")
ejes[0].axvline(30, color="red", linestyle="--", label="Límite Obesidad (30)")
ejes[0].set_title("Distribución del Índice de Masa Corporal (IMC)")
ejes[0].set_xlabel("IMC (kg/m²)")
ejes[0].set_ylabel("Cantidad de personas")
ejes[0].legend()

sns.histplot(df["Weight"], bins=25, kde=True, color="#4CAF50", ax=ejes[1])
ejes[1].set_title("Distribución del Peso Corporal")
ejes[1].set_xlabel("Peso (kg)")
ejes[1].set_ylabel("Cantidad de personas")

plt.tight_layout()
plt.savefig(os.path.join(carpeta_graficas, "01_distribucion_imc_peso.png"), dpi=150)
plt.close()

# --- Gráfica exploratoria 2: Niveles de obesidad y Actividad física ---
fig, ejes = plt.subplots(1, 2, figsize=(12, 4.8))

orden_obesidad = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]
etiquetas_obesidad = [
    "Peso Bajo",
    "Peso Normal",
    "Sobrepeso I",
    "Sobrepeso II",
    "Obesidad I",
    "Obesidad II",
    "Obesidad III",
]
conteo_ob = df["NObeyesdad"].value_counts().reindex(orden_obesidad)

ejes[0].barh(etiquetas_obesidad, conteo_ob.values, color="#5C7AEA")
ejes[0].set_title("Registros por Nivel de Obesidad")
ejes[0].set_xlabel("Cantidad de personas")

sns.histplot(df["FAF"], bins=15, color="#F39C12", ax=ejes[1])
ejes[1].set_title("Distribución de Actividad Física Semanal (FAF)")
ejes[1].set_xlabel("Nivel de actividad física (0 = Nada, 3 = Alta)")
ejes[1].set_ylabel("Cantidad de personas")

plt.tight_layout()
plt.savefig(os.path.join(carpeta_graficas, "02_niveles_obesidad_actividad.png"), dpi=150)
plt.close()

# --- Gráfica exploratoria 3: Correlación y diagrama de caja (outliers) ---
fig, ejes = plt.subplots(1, 2, figsize=(12, 4.8))

cols_corr = ["Age", "Height", "Weight", "IMC", "FAF", "CH2O", "TUE", "FCVC"]
sns.heatmap(
    df[cols_corr].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    ax=ejes[0],
)
ejes[0].set_title("Mapa de Correlación de Variables Numéricas")

sns.boxplot(data=df[["Age", "IMC", "Weight"]], palette="Set2", ax=ejes[1])
ejes[1].set_title("Detección Visual de Valores Atípicos (Boxplot)")
ejes[1].set_ylabel("Valor")

plt.tight_layout()
plt.savefig(os.path.join(carpeta_graficas, "03_correlacion_y_outliers.png"), dpi=150)
plt.close()

print("\nVariables seleccionadas para el agrupamiento:")
print("1. IMC (Weight / Height^2): resume el estado corporal mejor que peso o estatura solos.")
print("2. Age (Edad): permite diferenciar jóvenes de adultos con distintos patrones metabólicos.")
print("3. FAF (Actividad física): mide el hábito modificable clave frente al sedentarismo.")


# ====================================================================
# PARTE 3. LIMPIEZA Y PREPARACIÓN DE DATOS
# ====================================================================
print("\n============================================================")
print("PARTE 3. LIMPIEZA Y PREPARACIÓN DE DATOS")
print("============================================================")

filas_antes = len(df)

# 1. Eliminar espacios extra en columnas de texto
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# 2. Eliminar registros duplicados (no hay valores nulos)
duplicados = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)

# 3. Corregir inconsistencias decimales generadas por SMOTE y tipos de datos
# La edad debe ser entera y las mediciones/escalas se redondean a 2 decimales
df["Age"] = df["Age"].round(0).astype(int)
df["Height"] = df["Height"].round(2)
df["Weight"] = df["Weight"].round(2)
for col in ["FCVC", "NCP", "CH2O", "FAF", "TUE"]:
    df[col] = df[col].round(2)

# 4. Recalcular IMC limpio y aplicar codificaciones numéricas simples
df["IMC"] = (df["Weight"] / (df["Height"] ** 2)).round(2)
df["Antecedentes_Num"] = df["family_history_with_overweight"].map({"yes": 1, "no": 0})
df["ComidaCalorica_Num"] = df["FAVC"].map({"yes": 1, "no": 0})

# Etiquetas legibles en español para gráficas y dashboard
mapa_obesidad = {
    "Insufficient_Weight": "Peso Insuficiente",
    "Normal_Weight": "Peso Normal",
    "Overweight_Level_I": "Sobrepeso Nivel I",
    "Overweight_Level_II": "Sobrepeso Nivel II",
    "Obesity_Type_I": "Obesidad Tipo I",
    "Obesity_Type_II": "Obesidad Tipo II",
    "Obesity_Type_III": "Obesidad Tipo III",
}
mapa_genero = {"Female": "Femenino", "Male": "Masculino"}

df["Nivel_Obesidad"] = df["NObeyesdad"].map(mapa_obesidad)
df["Genero"] = df["Gender"].map(mapa_genero)

# 5. Identificación de valores atípicos mediante método IQR
print("Análisis de valores atípicos (método IQR):")
for col in ["Age", "Height", "Weight", "IMC", "FAF"]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr
    atipicos = ((df[col] < limite_inf) | (df[col] > limite_sup)).sum()
    print(
        f"  - {col:6s}: {atipicos:3d} atípicos "
        f"(rango real: {df[col].min()} a {df[col].max()})"
    )

print("\nJustificación del tratamiento de atípicos:")
print("Los valores altos de Edad (hasta 61 años) y Peso (hasta 173 kg) son casos")
print("clínicamente reales y no errores de captura; se conservan para no perder")
print("información valiosa sobre adultos mayores y obesidad severa.")

print(f"\nResumen de limpieza: {filas_antes} -> {len(df)} registros ({duplicados} duplicados eliminados, 0 nulos).")
print("\nMuestra de datos limpios y transformados:")
print(df[["Genero", "Age", "Height", "Weight", "IMC", "FAF", "Antecedentes_Num"]].head(5))

# 6. Selección de variables y escalamiento con StandardScaler
variables_modelo = ["IMC", "Age", "FAF"]
X = df[variables_modelo]

scaler = StandardScaler()
X_escalado = scaler.fit_transform(X)

print("\nPrimeros 5 registros escalados (media = 0, desviación estándar = 1):")
print(pd.DataFrame(X_escalado, columns=variables_modelo).head(5).round(3))

print("\nImportancia del escalamiento:")
print("K-Means agrupa usando distancia euclidiana. Sin StandardScaler, variables con")
print("rangos grandes como IMC (13-51) y Edad (14-61) dominarían sobre FAF (0-3).")


# ====================================================================
# PARTE 4. IMPLEMENTACIÓN DEL ALGORITMO K-MEANS
# ====================================================================
print("\n============================================================")
print("PARTE 4. IMPLEMENTACIÓN DEL ALGORITMO K-MEANS")
print("============================================================")

# Evaluar 6 valores diferentes de K (K = 2 a 7)
valores_k = [2, 3, 4, 5, 6, 7]
resultados_k = []

for k in valores_k:
    modelo_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    grupos_k = modelo_k.fit_predict(X_escalado)
    inercia = modelo_k.inertia_
    silueta = silhouette_score(X_escalado, grupos_k)
    db_index = davies_bouldin_score(X_escalado, grupos_k)
    resultados_k.append({
        "K": k,
        "Inercia": round(inercia, 2),
        "Silhouette": round(silueta, 4),
        "Davies_Bouldin": round(db_index, 4),
        "Iteraciones": modelo_k.n_iter_,
    })

df_metricas = pd.DataFrame(resultados_k)
print("\nEvaluación de distintos valores de K:")
print(df_metricas.to_string(index=False))

# Gráfica del Método del Codo y Coeficiente de Silueta
fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))

ejes[0].plot(df_metricas["K"], df_metricas["Inercia"], marker="o", color="#2E86AB", linewidth=2)
ejes[0].axvline(4, color="red", linestyle="--", label="K elegido = 4")
ejes[0].set_title("Método del Codo (Inercia vs K)")
ejes[0].set_xlabel("Número de clústeres (K)")
ejes[0].set_ylabel("Inercia")
ejes[0].legend()
ejes[0].grid(True)

ejes[1].plot(df_metricas["K"], df_metricas["Silhouette"], marker="s", color="#27AE60", linewidth=2)
ejes[1].axvline(4, color="red", linestyle="--", label="K elegido = 4")
ejes[1].set_title("Coeficiente de Silueta vs K")
ejes[1].set_xlabel("Número de clústeres (K)")
ejes[1].set_ylabel("Silhouette Score")
ejes[1].legend()
ejes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(carpeta_graficas, "04_metodo_codo_silhouette.png"), dpi=150)
plt.close()

print("\nJustificación de K = 4:")
print("En K = 4 se observa el codo claro de la inercia (baja fuerte hasta 2303.41)")
print("y un salto importante en Silhouette (0.3552), generando 4 perfiles claros.")

# --- Entrenamiento del modelo final (K = 4) ---
print("\nEJECUCIÓN K-MEANS FINAL (K = 4)")
modelo = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Grupo"] = modelo.fit_predict(X_escalado)

# Ordenar los grupos según su IMC promedio (de menor a mayor), igual que en p3.py
orden_grupos = df.groupby("Grupo")["IMC"].mean().sort_values().index

etiquetas = {
    orden_grupos[0]: "Jóvenes Activos",
    orden_grupos[1]: "Jóvenes Sedentarios",
    orden_grupos[2]: "Adultos con Sobrepeso",
    orden_grupos[3]: "Obesidad Severa Temprana",
}

df["Clasificacion"] = df["Grupo"].map(etiquetas)

# Obtener centroides en escala escalada y regresarlos a escala original
centroides_escalados = modelo.cluster_centers_
centroides_originales = scaler.inverse_transform(centroides_escalados)

df_centroides = pd.DataFrame(centroides_originales, columns=variables_modelo)
df_centroides["Grupo"] = range(4)
df_centroides["Clasificacion"] = df_centroides["Grupo"].map(etiquetas)
df_centroides["Cantidad"] = df["Grupo"].value_counts().sort_index().values
df_centroides["Porcentaje"] = (df_centroides["Cantidad"] / len(df) * 100).round(1)
df_centroides = df_centroides.sort_values("IMC").reset_index(drop=True)

print("\nCentroides (escala original):")
print(df_centroides[["Grupo", "Clasificacion", "IMC", "Age", "FAF", "Cantidad", "Porcentaje"]].round(2).to_string(index=False))

print("\nCentroides (escala estandarizada):")
print(pd.DataFrame(centroides_escalados, columns=variables_modelo).round(3))

print("\nDistancias a los centroides (primeros 5 registros):")
print(pd.DataFrame(modelo.transform(X_escalado)[:5], columns=["Dist_G0", "Dist_G1", "Dist_G2", "Dist_G3"]).round(3))

print("\nNúmero de iteraciones (K = 4):")
print(modelo.n_iter_)

print("\nRegistros por grupo:")
print(df["Grupo"].value_counts().sort_index())

print("\nRegistros por clasificación:")
print(df["Clasificacion"].value_counts())

# --- Gráfica de agrupamientos (estilo p3.py con centroides en X roja) ---
colores = {
    "Jóvenes Activos": "#27AE60",
    "Jóvenes Sedentarios": "#2E86AB",
    "Adultos con Sobrepeso": "#F39C12",
    "Obesidad Severa Temprana": "#E74C3C",
}

fig, ejes = plt.subplots(1, 2, figsize=(13, 5.2))

# Panel 1: IMC vs Actividad Física (FAF)
for clasificacion, color in colores.items():
    sub = df[df["Clasificacion"] == clasificacion]
    ejes[0].scatter(
        sub["IMC"],
        sub["FAF"],
        c=color,
        s=35,
        alpha=0.7,
        label=clasificacion,
    )

ejes[0].scatter(
    centroides_originales[:, 0],
    centroides_originales[:, 2],
    c="black",
    s=220,
    marker="X",
    edgecolors="white",
    linewidths=1.2,
    label="Centroides",
)
ejes[0].set_title("Agrupamiento K-Means: IMC vs Actividad Física")
ejes[0].set_xlabel("IMC (kg/m²)")
ejes[0].set_ylabel("Actividad física (FAF)")
ejes[0].legend(fontsize=8)
ejes[0].grid(True)

# Panel 2: Edad vs IMC
for clasificacion, color in colores.items():
    sub = df[df["Clasificacion"] == clasificacion]
    ejes[1].scatter(
        sub["Age"],
        sub["IMC"],
        c=color,
        s=35,
        alpha=0.7,
        label=clasificacion,
    )

ejes[1].scatter(
    centroides_originales[:, 1],
    centroides_originales[:, 0],
    c="black",
    s=220,
    marker="X",
    edgecolors="white",
    linewidths=1.2,
    label="Centroides",
)
ejes[1].set_title("Agrupamiento K-Means: Edad vs IMC")
ejes[1].set_xlabel("Edad (años)")
ejes[1].set_ylabel("IMC (kg/m²)")
ejes[1].legend(fontsize=8)
ejes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(carpeta_graficas, "05_clusters_imc_actividad.png"), dpi=150)
plt.close()

# --- Gráfica PCA 2D: Clústeres vs Nivel de obesidad original ---
pca = PCA(n_components=2, random_state=42)
componentes = pca.fit_transform(X_escalado)
df["PCA1"] = componentes[:, 0].round(4)
df["PCA2"] = componentes[:, 1].round(4)
centroides_pca = pca.transform(centroides_escalados)

fig, ejes = plt.subplots(1, 2, figsize=(13, 5.2))

for clasificacion, color in colores.items():
    sub = df[df["Clasificacion"] == clasificacion]
    ejes[0].scatter(sub["PCA1"], sub["PCA2"], c=color, s=30, alpha=0.7, label=clasificacion)

ejes[0].scatter(
    centroides_pca[:, 0],
    centroides_pca[:, 1],
    c="black",
    s=200,
    marker="X",
    edgecolors="white",
    linewidths=1.2,
    label="Centroides",
)
ejes[0].set_title("Proyección PCA 2D por Clúster K-Means")
ejes[0].set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
ejes[0].set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
ejes[0].legend(fontsize=8)
ejes[0].grid(True)

orden_niveles_es = [
    "Peso Insuficiente",
    "Peso Normal",
    "Sobrepeso Nivel I",
    "Sobrepeso Nivel II",
    "Obesidad Tipo I",
    "Obesidad Tipo II",
    "Obesidad Tipo III",
]
sns.scatterplot(
    data=df,
    x="PCA1",
    y="PCA2",
    hue="Nivel_Obesidad",
    hue_order=orden_niveles_es,
    palette="Spectral_r",
    s=30,
    alpha=0.75,
    ax=ejes[1],
)
ejes[1].set_title("Proyección PCA 2D por Nivel de Obesidad Original")
ejes[1].set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
ejes[1].set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
ejes[1].legend(fontsize=7, title="Nivel de obesidad", loc="upper right")
ejes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(carpeta_graficas, "06_clusters_pca_2d.png"), dpi=150)
plt.close()

# --- Comparación de dos configuraciones del modelo (estilo p2.py) ---
print("\n============================================================")
print("COMPARACIÓN DE DOS CONFIGURACIONES DEL MODELO")
print("============================================================")

# Configuración 1: K = 4 con 3 variables (IMC, Age, FAF) -> Modelo Final
sil_config1 = silhouette_score(X_escalado, df["Grupo"])
db_config1 = davies_bouldin_score(X_escalado, df["Grupo"])

# Configuración 2: K = 4 con 5 variables (IMC, Age, FAF, CH2O, TUE)
variables_config2 = ["IMC", "Age", "FAF", "CH2O", "TUE"]
scaler_config2 = StandardScaler()
X_config2 = scaler_config2.fit_transform(df[variables_config2])

modelo_config2 = KMeans(n_clusters=4, random_state=42, n_init=10)
grupos_config2 = modelo_config2.fit_predict(X_config2)
sil_config2 = silhouette_score(X_config2, grupos_config2)
db_config2 = davies_bouldin_score(X_config2, grupos_config2)

# También comparamos contra K = 3 con las mismas 3 variables (como en p2.py)
modelo_k3 = KMeans(n_clusters=3, random_state=42, n_init=10)
grupos_k3 = modelo_k3.fit_predict(X_escalado)
sil_k3 = silhouette_score(X_escalado, grupos_k3)
db_k3 = davies_bouldin_score(X_escalado, grupos_k3)

df_comparacion_modelos = pd.DataFrame([
    {
        "Configuracion": "Config 1: K=4 (3 vars: IMC, Edad, FAF)",
        "K": 4,
        "Variables": 3,
        "Inercia": round(modelo.inertia_, 2),
        "Silhouette": round(sil_config1, 4),
        "Davies_Bouldin": round(db_config1, 4),
        "Iteraciones": modelo.n_iter_,
    },
    {
        "Configuracion": "Config 2: K=4 (5 vars: +Agua, +Tecnología)",
        "K": 4,
        "Variables": 5,
        "Inercia": round(modelo_config2.inertia_, 2),
        "Silhouette": round(sil_config2, 4),
        "Davies_Bouldin": round(db_config2, 4),
        "Iteraciones": modelo_config2.n_iter_,
    },
    {
        "Configuracion": "Referencia: K=3 (3 vars: IMC, Edad, FAF)",
        "K": 3,
        "Variables": 3,
        "Inercia": round(modelo_k3.inertia_, 2),
        "Silhouette": round(sil_k3, 4),
        "Davies_Bouldin": round(db_k3, 4),
        "Iteraciones": modelo_k3.n_iter_,
    },
])

print(df_comparacion_modelos.to_string(index=False))
print("\nDiferencias encontradas:")
print("1. Al pasar de 3 a 5 variables (Config 1 vs Config 2), el Silhouette baja de")
print(f"   {sil_config1:.4f} a {sil_config2:.4f} y Davies-Bouldin empeora de {db_config1:.4f} a {db_config2:.4f}.")
print("   Agregar CH2O y TUE introduce dispersión porque no separan tan claramente los perfiles.")
print(f"2. Frente a K=3 (Silhouette {sil_k3:.4f}), K=4 mejora la cohesión y permite separar")
print("   a los jóvenes activos de los jóvenes sedentarios, evitando mezclarlos en un solo grupo.")

# Gráfica comparativa de configuraciones
fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))
nombres_cortos = ["Config 1\n(K=4, 3 vars)", "Config 2\n(K=4, 5 vars)", "Referencia\n(K=3, 3 vars)"]
colores_bar = ["#27AE60", "#E74C3C", "#F39C12"]

ejes[0].bar(nombres_cortos, df_comparacion_modelos["Silhouette"], color=colores_bar)
ejes[0].set_title("Comparación de Silhouette (mayor es mejor)")
ejes[0].set_ylabel("Silhouette Score")
ejes[0].set_ylim(0, 0.45)
for i, v in enumerate(df_comparacion_modelos["Silhouette"]):
    ejes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

ejes[1].bar(nombres_cortos, df_comparacion_modelos["Davies_Bouldin"], color=colores_bar)
ejes[1].set_title("Comparación de Davies-Bouldin (menor es mejor)")
ejes[1].set_ylabel("Índice Davies-Bouldin")
ejes[1].set_ylim(0, 1.75)
for i, v in enumerate(df_comparacion_modelos["Davies_Bouldin"]):
    ejes[1].text(i, v + 0.04, f"{v:.3f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(carpeta_graficas, "07_comparacion_modelos.png"), dpi=150)
plt.close()


# ====================================================================
# PARTE 5. ANÁLISIS E INTERPRETACIÓN DE RESULTADOS
# ====================================================================
print("\n============================================================")
print("PARTE 5. ANÁLISIS E INTERPRETACIÓN DE RESULTADOS")
print("============================================================")

orden_nombres = [
    "Jóvenes Activos",
    "Jóvenes Sedentarios",
    "Adultos con Sobrepeso",
    "Obesidad Severa Temprana",
]

# Tabla comparativa de promedios por grupo
tabla_grupos = (
    df.groupby("Clasificacion")
    .agg(
        Registros=("IMC", "count"),
        IMC_Prom=("IMC", "mean"),
        IMC_DE=("IMC", "std"),
        Peso_Prom=("Weight", "mean"),
        Edad_Prom=("Age", "mean"),
        Actividad_FAF=("FAF", "mean"),
        Agua_CH2O=("CH2O", "mean"),
        Tecnologia_TUE=("TUE", "mean"),
        Pct_Antecedentes=("Antecedentes_Num", lambda s: s.mean() * 100),
        Pct_ComidaCalorica=("ComidaCalorica_Num", lambda s: s.mean() * 100),
    )
    .reindex(orden_nombres)
    .round(2)
)
tabla_grupos["Porcentaje"] = (tabla_grupos["Registros"] / len(df) * 100).round(1)

print("\nTabla comparativa de los grupos identificados:")
print(tabla_grupos.to_string())

print("\nEstadísticas descriptivas detalladas (IMC, Edad, FAF) por clasificación:")
print(
    df.groupby("Clasificacion")[["IMC", "Age", "FAF"]]
    .agg(["mean", "std", "min", "max"])
    .reindex(orden_nombres)
    .round(2)
    .to_string()
)

# Guardar archivos procesados para el dashboard y el repositorio
df.to_csv(os.path.join(carpeta_procesados, "obesidad_con_clusters.csv"), index=False)
df_centroides.to_csv(os.path.join(carpeta_procesados, "centroides.csv"), index=False)
df_metricas.to_csv(os.path.join(carpeta_procesados, "metricas_k.csv"), index=False)
tabla_grupos.reset_index().to_csv(
    os.path.join(carpeta_procesados, "comparacion_grupos.csv"), index=False
)
df_comparacion_modelos.to_csv(
    os.path.join(carpeta_procesados, "comparacion_modelos.csv"), index=False
)

print("\nArchivos guardados correctamente en datos/procesados/ y graficas/.")
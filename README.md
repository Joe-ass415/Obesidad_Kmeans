# Proyecto Integrador: Segmentación de Perfiles de Obesidad mediante K-Means y Dashboard Interactivo

## Descripción del proyecto
Este proyecto de minería de datos aplica el algoritmo de aprendizaje no supervisado **K-Means** sobre el dataset público *Estimation of obesity levels based on eating habits and physical condition* (2,111 registros originales; 2,087 registros limpios tras eliminar duplicados).

A partir de tres variables clave —**Índice de Masa Corporal (IMC)**, **Edad (`Age`)** y **Frecuencia de Actividad Física (`FAF`)**— se identifican **4 perfiles poblacionales** claramente diferenciados:
1. **Jóvenes Activos** (20.7%)
2. **Jóvenes Sedentarios** (32.9%)
3. **Adultos con Sobrepeso** (14.5%)
4. **Obesidad Severa Temprana** (32.0%)

Los resultados del análisis se presentan además en un **dashboard interactivo en Streamlit** con dos páginas y tres filtros funcionales para facilitar la exploración y la toma de decisiones preventivas.

---

## Herramientas utilizadas
- **Python 3.10+**
- **Pandas**: carga, limpieza, transformación y análisis descriptivo de datos.
- **Scikit-learn**: estandarización (`StandardScaler`), agrupamiento (`KMeans`), reducción de dimensionalidad (`PCA`) y métricas (`silhouette_score`, `davies_bouldin_score`).
- **Matplotlib y Seaborn**: gráficas exploratorias, método del codo, dispersión de clústeres y comparación de modelos.
- **Streamlit y Plotly**: desarrollo del dashboard interactivo.

---

## Estructura del repositorio

```text
ProyectoObesidad/
├── codigo/
│   └── proyecto_obesidad_kmeans.py       # Script principal (exploración, limpieza, K-Means y análisis)
├── dashboard/
│   ├── app.py                            # Dashboard interactivo en Streamlit (2 páginas y 3 filtros)
│   └── estilos.css                       # Tema oscuro personalizado (Obsidian & Mint) y animaciones CSS
├── datos/
│   ├── original/
│   │   └── ObesityDataSet_raw_and_data_sinthetic.csv
│   └── procesados/
│       ├── obesidad_con_clusters.csv     # Dataset limpio con clúster y clasificación asignada
│       ├── centroides.csv                # Centroides del modelo final en escala original
│       ├── comparacion_grupos.csv        # Tabla comparativa de estadísticas por clúster
│       ├── comparacion_modelos.csv       # Comparación de configuraciones K-Means
│       └── metricas_k.csv                # Evaluación de K = 2 a 7 (Inercia y Silhouette)
├── documentacion/
│   └── Reporte_Proyecto_Obesidad_KMeans.pdf  # Documentación oficial del proyecto (8 apartados)
├── graficas/                             # Gráficas generadas y capturas del dashboard
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Cómo ejecutar el proyecto paso a paso

### 1. Clonar el repositorio y entrar a la carpeta
```bash
git clone <URL_DEL_REPOSITORIO>
cd ProyectoObesidad
```

### 2. Crear y activar un entorno virtual (recomendado)
En Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

En macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar las dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar el script de análisis y entrenamiento K-Means
Este comando limpia los datos, evalúa `K = 2..7`, entrena el modelo final (`K = 4`), compara configuraciones y genera los archivos en `datos/procesados/` y las gráficas en `graficas/`:
```bash
python codigo/proyecto_obesidad_kmeans.py
```

### 5. Ejecutar el dashboard interactivo en Streamlit
```bash
streamlit run dashboard/app.py
```
Al ejecutar el comando se abrirá automáticamente en el navegador (generalmente en `http://localhost:8501`).

---

## Fuente del dataset
- **Nombre:** *Estimation of obesity levels based on eating habits and physical condition*
- **Autores:** Fabio Mendoza Palechor y Alexis de la Hoz Manotas (2019), UCI Machine Learning Repository / Kaggle.
- **Contenido:** 2,111 registros y 17 atributos sobre hábitos alimenticios, condición física y nivel de obesidad en población de México, Perú y Colombia.

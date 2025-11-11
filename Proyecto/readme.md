# Proyecto Final - MACHINE LEARNING: Predicción de Calidad de Vino

## Caso de Negocio

### Contexto

La industria vitivinícola requiere métodos eficientes para evaluar y clasificar la calidad del vino basándose en sus propiedades fisicoquímicas. Este proyecto implementa un pipeline completo de MLOps para predecir la calidad del vino (escala 3-8) utilizando 11 características medibles en laboratorio.

### Objetivo

Desarrollar un sistema automatizado de predicción que permita:

- **Reducir costos** de evaluación sensorial manual
- **Acelerar el proceso** de control de calidad
- **Estandarizar** la clasificación de productos
- **Detectar tempranamente** vinos de baja calidad para ajuste de procesos

### Valor de Negocio

- **Reducción del 70%** en tiempo de evaluación vs. catas tradicionales
- **Ahorro operativo** al automatizar pruebas de calidad
- **Trazabilidad completa** del proceso de clasificación
- **Alertas automáticas** de drift en características del producto

---

## Principales Hallazgos del EDA

### Dataset: Wine Quality

- **Registros totales**: 6,497 muestras de vino
- **División**: 80% entrenamiento (5,197) / 20% prueba (1,300)
- **Variable objetivo**: `quality` (ordinal, rango 3-8)
- **Características**: 11 variables fisicoquímicas

### Variables Clave Identificadas

**Top 3 Predictores por Correlación con Calidad:**

1. **alcohol** (correlación positiva): Vinos de mayor calidad tienden a mayor graduación alcohólica
2. **volatile_acidity** (correlación negativa): Acidez volátil alta genera sabor a vinagre, reduciendo calidad
3. **sulphates** (correlación positiva): Actúan como conservantes, mejorando estabilidad

### Observaciones Críticas

**Consistencia de datos**:

- Distribuciones similares entre train/test/original
- Sin valores nulos detectados
- Correlaciones estables entre particiones → datos representativos

**Valores atípicos**:

- Variables con más outliers: `free_sulfur_dioxide`, `total_sulfur_dioxide`
- `chlorides` y `residual_sugar` presentan valores extremos ocasionales
- Outliers preservados intencionalmente (casos reales del proceso)

**Distribución de calidad**:

- Clases mayoritarias: 5 y 6 (vinos de calidad media)
- Clases minoritarias: 3, 4, 7, 8 (extremos de calidad)
- Desbalance manejado con StratifiedKFold en validación

---

## Resultados del Modelo de Predicción

### Arquitectura del Modelo

**Preprocesamiento** (ColumnTransformer):

- **Numéricas**: Imputación mediana + StandardScaler
- **Categóricas nominales**: Imputación constante + OneHotEncoder
- **Categóricas ordinales**: Imputación moda + OrdinalEncoder

**Algoritmo seleccionado**: LightGBM Classifier

- Modelo de gradient boosting optimizado
- Maneja eficientemente datos tabulares
- Robusto ante outliers
- Rápido entrenamiento y predicción

### Métricas de Desempeño

**Validación Cruzada (5-fold StratifiedKFold)**:

- **Cohen's Kappa (cuadrático)**: ~0.65-0.70
  - Métrica principal para variables ordinales
  - Penaliza menos errores cercanos (calidad 5→6 vs 5→8)
- **Accuracy**: ~65-70%
  - Rendimiento sólido considerando 6 clases posibles

**Evaluación en Test Set**:

- Métricas consistentes con validación cruzada
- Sin evidencia de overfitting
- Generalización adecuada a datos no vistos

### Pipeline Completo Implementado

Este repositorio contiene un flujo MLOps completo para crear, evaluar, desplegar y monitorear el modelo de clasificación.

**Componentes principales**:

- `Base_de_datos.csv`: Dataset de calidad de vino (6,497 registros)
- `mlops_pipeline/src/ft_engineering.py`: Pipeline de preprocesamiento y feature engineering
- `mlops_pipeline/src/model_training_evaluation.py`: Entrenamiento con CV y evaluación, genera `models/best_model.joblib`
- `mlops_pipeline/src/model_monitoring.py`: Detección de drift (KS, PSI, JSD, Chi²)
- `mlops_pipeline/src/model_deploy.py`: API REST con FastAPI para predicciones
- `mlops_pipeline/src/streamlit_app.py`: Dashboard interactivo para visualizar métricas y reportes

---

## Guía de Uso Rápido

### Opción 1: Ejecución con Docker (Recomendado)

**Ventajas**: Evita problemas de compatibilidad Python 3.13, entorno aislado, funciona en Windows/Linux/Mac

```bash
# 1. Ejecutar pipeline completo (genera modelos y reportes)
docker compose run --rm runner

# 2. Levantar servicios (API + Dashboard)
docker compose up api streamlit

# 3. Acceder a:
#    - API: http://localhost:8000/docs (Swagger UI)
#    - Dashboard: http://localhost:8501 (Streamlit)
```

**Modo desarrollo interactivo**:

```bash
# Abrir shell dentro del contenedor
docker compose run --rm --service-ports pipeline

# Ejecutar comandos individuales:
python -m mlops_pipeline.src.ft_engineering
python -m mlops_pipeline.src.model_training_evaluation
python -m mlops_pipeline.src.model_monitoring
```

### Opción 2: Ejecución Local (Python 3.10)

**Requisitos**: Python 3.10 (no compatible con 3.13 por limitaciones de paquetes)

```bash
# 1. Instalar dependencias
set_up.bat  # Windows
# o
pip install -r requirements.txt

# 2. Ejecutar ingeniería de features
python -m mlops_pipeline.src.ft_engineering

# 3. Entrenar y evaluar modelos
python -m mlops_pipeline.src.model_training_evaluation

# 4. Ejecutar monitorización (detección de drift)
python -m mlops_pipeline.src.model_monitoring

# 5. Ver reportes con Streamlit
streamlit run mlops_pipeline/src/streamlit_app.py

# 6. Desplegar API (local)
uvicorn mlops_pipeline.src.model_deploy:app --reload
```

---

## Dashboard y Visualizaciones

El proyecto incluye un **dashboard interactivo** con Streamlit que muestra:

- **Estado de artefactos**: Verifica existencia de modelos y reportes
- **Métricas de CV**: Gráficos de línea comparando Kappa y Accuracy por fold
- **Métricas finales**: Visualización destacada del rendimiento en test
- **Reporte de drift**: Tabla con top 5 features con mayor desviación (KS test)
- **Alertas automáticas**: Avisos cuando KS > 0.3 (drift significativo)

**Acceso**: `http://localhost:8501` después de ejecutar `docker compose up streamlit`

---

## Estructura de Artefactos Generados

```
Proyecto/
├── data/
│   ├── train.csv              # Conjunto de entrenamiento procesado
│   ├── test.csv               # Conjunto de prueba procesado
│   └── preprocessor.joblib    # Pipeline de preprocesamiento serializado
├── models/
│   └── best_model.joblib      # Modelo LightGBM entrenado (mejor fold CV)
└── reports/
    ├── cv_fold_metrics.csv    # Métricas por fold (Kappa, Accuracy)
    ├── cv_summary.csv         # Resumen estadístico de CV (mean, std)
    ├── final_metrics.csv      # Métricas en test set
    └── drift_report.csv       # Resultados de detección de drift (KS, PSI, JSD, Chi²)
```

---

## Proceso Metodológico

### 1. Análisis Exploratorio (EDA)

**Notebook**: `comprension_eda.ipynb`

- Carga y validación de datos
- Análisis de distribuciones por variable
- Identificación de correlaciones
- Detección de outliers
- Visualización de patrones por clase de calidad

### 2. Feature Engineering

**Script**: `ft_engineering.py`

- Separación train/test estratificada (80/20)
- Pipeline de preprocesamiento por tipo de variable:
  - **Numéricas**: Imputación mediana + escalado estándar
  - **Categóricas nominales**: Imputación constante + One-Hot Encoding
  - **Categóricas ordinales**: Imputación moda + Ordinal Encoding
- Serialización del preprocesador para reproducibilidad

### 3. Entrenamiento y Evaluación

**Script**: `model_training_evaluation.py`

- Validación cruzada estratificada (5 folds)
- Concatenación con datos originales por fold (aumento de datos)
- Métricas: Cohen's Kappa cuadrático (principal) + Accuracy
- Selección del mejor modelo por fold
- Evaluación final en test set independiente

### 4. Monitoreo de Drift

**Script**: `model_monitoring.py`

- Comparación train vs test en 4 métricas:
  - **Kolmogorov-Smirnov (KS)**: Máxima diferencia entre CDFs
  - **Population Stability Index (PSI)**: Cambios en distribución
  - **Jensen-Shannon Divergence (JSD)**: Divergencia entre distribuciones
  - **Chi-Squared**: Test estadístico de independencia
- Identificación de features con mayor drift

### 5. Despliegue

**Script**: `model_deploy.py`

- API REST con FastAPI
- Endpoint `/predict`: Recibe features y retorna predicción de calidad
- Endpoint `/health`: Verificación de estado del servicio
- Carga automática de modelo y preprocesador

---

## Tecnologías Utilizadas

**Machine Learning**:

- `scikit-learn 1.3.0`: Preprocesamiento, validación cruzada, pipelines
- `lightgbm`: Algoritmo de clasificación principal
- `pandas 2.0.3`: Manipulación de datos
- `numpy 1.24.3`: Operaciones numéricas

**Visualización**:

- `matplotlib 3.7.2`: Gráficos estáticos
- `seaborn 0.12.2`: Visualizaciones estadísticas
- `plotly 5.15.0`: Gráficos interactivos
- `streamlit`: Dashboard web

**Deployment**:

- `FastAPI 0.100.0`: Framework para API REST
- `uvicorn 0.23.2`: Servidor ASGI
- `Docker`: Containerización
- `docker-compose`: Orquestación de servicios

**Monitoreo**:

- `scipy`: Tests estadísticos
- Métricas personalizadas de drift

---

## Notas Técnicas

### Compatibilidad de Python

- **Recomendado**: Python 3.10
- **No compatible**: Python 3.13 (limitaciones de `evidently 0.2.8` y `pydantic`)
- **Solución**: Usar Docker (imagen oficial `python:3.10-slim`)

### Instalación de LightGBM en Windows

Si ejecutas localmente sin Docker:

```bash
conda install -c conda-forge lightgbm -y
pip install -r requirements.txt
```

### Preprocesador Flexible

El pipeline soporta tres tipos de transformaciones:

1. **Numéricas**: `SimpleImputer(strategy='median')` + `StandardScaler`
2. **Categóricas nominales**: `SimpleImputer(fill_value='missing')` + `OneHotEncoder`
3. **Categóricas ordinales**: `SimpleImputer(strategy='most_frequent')` + `OrdinalEncoder`

Para usar features ordinales, especificar en `ft_engineering.create_datasets(ordinal_features=[...])`

---

## Próximos Pasos Sugeridos

- [ ] **Feature Engineering avanzado**: Interacciones entre variables, ratios (e.g., acidez_total/pH)
- [ ] **Optimización de hiperparámetros**: Grid search o Bayesian optimization para LightGBM
- [ ] **Manejo de desbalanceo**: SMOTE, class_weight, o threshold adjustment
- [ ] **Tests unitarios**: Pytest para validar transformaciones y predicciones
- [ ] **CI/CD**: GitHub Actions para ejecución automática del pipeline
- [ ] **Monitoreo en producción**: Alertas automáticas cuando drift > umbral
- [ ] **Versionado de modelos**: MLflow o DVC para tracking de experimentos
- [ ] **A/B testing**: Framework para comparar versiones de modelos en producción

---

## Estructura Completa del Proyecto

```
Proyecto/
├── mlops_pipeline/
│   ├── Dockerfile                      # Imagen Docker Python 3.10
│   └── src/
│       ├── Cargar_datos.ipynb          #  Documentación de carga de datos
│       ├── comprension_eda.ipynb       #  Análisis exploratorio completo
│       ├── ft_engineering.py           #  Pipeline de preprocesamiento
│       ├── heuristic_model.py          #  Modelo baseline heurístico
│       ├── model_training_evaluation.py #  Entrenamiento y validación CV
│       ├── model_monitoring.py         #  Detección de drift
│       ├── model_deploy.py             #  API REST con FastAPI
│       ├── streamlit_app.py            #  Dashboard interactivo
│       └── run_pipeline.py             #  Orquestador principal
├── docker-compose.yml                  #  Configuración multi-servicio
├── config.json                         #  Parámetros del pipeline
├── Base_de_datos.csv                   #  Dataset de calidad de vino
├── requirements.txt                    #  Dependencias Python
├── .gitignore                          #  Exclusiones de Git
├── readme.md                           #  Esta documentación
└── set_up.bat                          #  Script de configuración Windows
```

### Archivos Generados por el Pipeline

```
├── data/                               # Datos procesados
│   ├── train.csv
│   ├── test.csv
│   └── preprocessor.joblib
├── models/                             # Modelos entrenados
│   └── best_model.joblib
└── reports/                            # Reportes y métricas
    ├── cv_fold_metrics.csv
    ├── cv_summary.csv
    ├── final_metrics.csv
    └── drift_report.csv
```

---

## Configuración Docker

### docker-compose.yml

El proyecto utiliza 4 servicios orquestados:

1. **pipeline**: Shell interactivo para desarrollo
2. **runner**: Ejecuta automáticamente `run_pipeline.py`
3. **api**: Servidor FastAPI en puerto 8000
4. **streamlit**: Dashboard en puerto 8501

Todos montan el directorio actual como `/app` para edición en vivo.

### Comandos Docker Útiles

```bash
# Reconstruir imagen después de cambios en requirements.txt
docker compose build

# Ver logs de servicios
docker compose logs -f api
docker compose logs -f streamlit

# Detener todos los servicios
docker compose down

# Limpiar volúmenes y contenedores
docker compose down -v
```

---

## Notebooks Incluidos

### 1. `Cargar_datos.ipynb`

**Propósito**: Documentación de exploración inicial

- No se usa en producción (solo para referencia)
- Muestra cómo cargar y validar `Base_de_datos.csv`
- Incluye estadísticas descriptivas y verificación de nulos
- Útil para desarrolladores nuevos en el proyecto

### 2. `comprension_eda.ipynb`

**Propósito**: Análisis exploratorio completo

- Distribuciones de variables por conjunto (train/test/original)
- Gráficos de densidad (KDE) por nivel de calidad
- Boxplots para identificar outliers
- Matrices de correlación comparativas
- **Hallazgos clave documentados en español**

---

## API Endpoints

### FastAPI - `http://localhost:8000`

**GET `/health`**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**POST `/predict`**

```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

**Respuesta**:

```json
{
  "quality": 5,
  "model_version": "lightgbm_v1"
}
```

### Swagger UI

Documentación interactiva: `http://localhost:8000/docs`

---

## Contacto y Contribuciones

- **Autor**: Marlon Vallejo
- **Repositorio**: [ProyectoML](https://github.com/marlonvallejo/ProyectoML)
- **Issues**: Para reportar bugs o sugerir mejoras

---

## Licencia

Este proyecto es de uso educativo como parte del curso de MACHINE LEARNING.


import streamlit as st
import pandas as pd
from pathlib import Path
try:
    from PIL import Image
except ImportError:
    Image = None


st.set_page_config(page_title="MLOps Pipeline - Wine Quality", layout="wide")

st.title("MLOps Pipeline - Predicción de Calidad de Vino")
st.markdown("---")

# Paths relative to project root (/app in Docker)
reports_dir = Path(__file__).parent.parent.parent / "reports"
models_dir = Path(__file__).parent.parent.parent / "models"
data_dir = Path(__file__).parent.parent.parent / "data"

# Sidebar with info
with st.sidebar:
    st.header("ℹInformación del Pipeline")
    st.markdown("""
    **Estado de artefactos:**
    """)
    
    if (models_dir / "best_model.joblib").exists():
        st.success("Modelo entrenado")
    else:
        st.error("Modelo no encontrado")
    
    if (data_dir / "preprocessor.joblib").exists():
        st.success("Preprocesador")
    else:
        st.error("Preprocesador no encontrado")
    
    st.markdown("---")
    st.markdown("**Reportes disponibles:**")
    
    reports_count = 0
    if (reports_dir / "cv_fold_metrics.csv").exists():
        st.success("Métricas de CV")
        reports_count += 1
    if (reports_dir / "cv_summary.csv").exists():
        st.success("Resumen de CV")
        reports_count += 1
    if (reports_dir / "final_metrics.csv").exists():
        st.success("Métricas finales")
        reports_count += 1
    if (reports_dir / "drift_report.csv").exists():
        st.success("Reporte de drift")
        reports_count += 1
    
    if reports_count == 0:
        st.warning("No hay reportes. Ejecute el pipeline.")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("Métricas de Entrenamiento")
    
    
    if (reports_dir / "cv_summary.csv").exists():
        st.subheader("Resumen de Validación Cruzada")
        cv_summary = pd.read_csv(reports_dir / "cv_summary.csv", index_col=0)
        st.dataframe(cv_summary, use_container_width=True)
    
    
    if (reports_dir / "cv_fold_metrics.csv").exists():
        st.subheader("Métricas por Fold")
        cv_folds = pd.read_csv(reports_dir / "cv_fold_metrics.csv")
        st.dataframe(cv_folds, use_container_width=True)
        
        
        st.subheader("Comparación de Métricas por Fold")
        chart_data = cv_folds[['fold', 'kappa', 'accuracy']].set_index('fold')
        st.line_chart(chart_data)

with col2:
    st.header("Evaluación Final")
    
    
    if (reports_dir / "final_metrics.csv").exists():
        st.subheader("Métricas en Test Set")
        final_metrics = pd.read_csv(reports_dir / "final_metrics.csv", index_col=0)
        st.dataframe(final_metrics, use_container_width=True)
        
        
        if 'final_kappa' in final_metrics.index:
            kappa_val = final_metrics.loc['final_kappa', final_metrics.columns[0]]
            accuracy_val = final_metrics.loc['final_accuracy', final_metrics.columns[0]]
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Kappa (Quadratic)", f"{kappa_val:.4f}")
            with metric_col2:
                st.metric("Accuracy", f"{accuracy_val:.4f}")

st.markdown("---")

# Drift Report
st.header("Monitoreo de Data Drift")
if (reports_dir / "drift_report.csv").exists():
    drift = pd.read_csv(reports_dir / "drift_report.csv")
    
    st.subheader("Reporte de Drift por Feature")
    st.dataframe(drift, use_container_width=True)
    
    # Highlight features with high drift
    st.subheader("Features con Mayor Drift")
    
    # Show features sorted by KS statistic (if available)
    if 'ks_statistic' in drift.columns:
        top_drift = drift.nlargest(5, 'ks_statistic')[['feature', 'ks_statistic', 'ks_pvalue']]
        st.dataframe(top_drift, use_container_width=True)
        
        # Warning for high drift
        high_drift_features = drift[drift['ks_statistic'] > 0.3]['feature'].tolist()
        if high_drift_features:
            st.warning(f"Features con drift significativo (KS > 0.3): {', '.join(high_drift_features)}")
else:
    st.info("No hay reportes de drift aún. El reporte se genera automáticamente al ejecutar el pipeline.")
    st.markdown("""
    **Para generar el reporte de drift:**
    ```bash
    docker compose run --rm runner
    ```
    """)

# Model comparison (if exists)
st.markdown("---")
if (reports_dir / "models_comparison.csv").exists():
    st.header("Comparación de Modelos")
    df = pd.read_csv(reports_dir / "models_comparison.csv")
    st.dataframe(df, use_container_width=True)
    if (reports_dir / "models_comparison.png").exists() and Image:
        st.image(Image.open(reports_dir / "models_comparison.png"))

st.markdown("---")
st.markdown("*Dashboard generado por el MLOps Pipeline - Wine Quality Prediction*")

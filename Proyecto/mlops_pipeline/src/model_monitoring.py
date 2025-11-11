



from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
from scipy.stats import ks_2samp, entropy, chi2_contingency


def psi(expected, actual, buckets=10):
    
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    # create bins based on expected
    try:
        quantiles = np.linspace(0, 1, buckets + 1)
        bins = np.unique(np.quantile(expected, quantiles))
        expected_perc, _ = np.histogram(expected, bins=bins)
        actual_perc, _ = np.histogram(actual, bins=bins)
        expected_perc = expected_perc / expected_perc.sum()
        actual_perc = actual_perc / actual_perc.sum()
        # avoid zeros
        actual_perc = np.where(actual_perc == 0, 1e-8, actual_perc)
        expected_perc = np.where(expected_perc == 0, 1e-8, expected_perc)
        psi_val = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
        return float(psi_val)
    except Exception:
        return np.nan


def jensen_shannon(p, q):
    p = np.asarray(p) / np.sum(p)
    q = np.asarray(q) / np.sum(q)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def drift_for_numeric(ref, curr):
    ks_stat, ks_p = ks_2samp(ref, curr)
    psi_val = psi(ref, curr)
    return {"ks_stat": float(ks_stat), "ks_p": float(ks_p), "psi": float(psi_val)}


def drift_for_categorical(ref, curr):
    # build value counts aligned
    ref_counts = ref.value_counts()
    curr_counts = curr.value_counts()
    all_idx = ref_counts.index.union(curr_counts.index)
    ref_vec = ref_counts.reindex(all_idx).fillna(0).values
    curr_vec = curr_counts.reindex(all_idx).fillna(0).values
    # chi-square
    try:
        chi2, p, dof, _ = chi2_contingency([ref_vec, curr_vec])
    except Exception:
        chi2, p = np.nan, np.nan
    # jensen-shannon
    try:
        jsd = jensen_shannon(ref_vec, curr_vec)
    except Exception:
        jsd = np.nan
    return {"chi2": float(chi2) if not np.isnan(chi2) else np.nan, "chi2_p": float(p) if not np.isnan(p) else np.nan, "jsd": float(jsd)}


def generate_drift_alerts(report_df: pd.DataFrame, thresholds: Dict[str, float] = None) -> List[str]:
    
    if thresholds is None:
        thresholds = {
            "ks_stat": 0.3,    # Drift significativo si KS > 0.3
            "psi": 0.2,        # Drift significativo si PSI > 0.2
            "jsd": 0.15,       # Drift significativo si JSD > 0.15
            "chi2_p": 0.05     # Drift significativo si p-value < 0.05
        }
    
    alerts = []
    
    # Verificar KS (variables numéricas)
    if "ks_stat" in report_df.columns:
        high_ks = report_df[report_df["ks_stat"] > thresholds["ks_stat"]]
        if len(high_ks) > 0:
            alerts.append(f" ALERTA CRÍTICA: {len(high_ks)} variable(s) con KS > {thresholds['ks_stat']}")
            for _, row in high_ks.iterrows():
                alerts.append(f"   • {row['feature']}: KS = {row['ks_stat']:.3f}")
    
    # Verificar PSI
    if "psi" in report_df.columns:
        high_psi = report_df[report_df["psi"] > thresholds["psi"]]
        if len(high_psi) > 0:
            alerts.append(f" ADVERTENCIA: {len(high_psi)} variable(s) con PSI > {thresholds['psi']}")
            for _, row in high_psi.iterrows():
                alerts.append(f"   • {row['feature']}: PSI = {row['psi']:.3f}")
    
    # Verificar JSD (variables categóricas)
    if "jsd" in report_df.columns:
        high_jsd = report_df[report_df["jsd"] > thresholds["jsd"]]
        if len(high_jsd) > 0:
            alerts.append(f" ADVERTENCIA: {len(high_jsd)} variable(s) con JSD > {thresholds['jsd']}")
            for _, row in high_jsd.iterrows():
                alerts.append(f"   • {row['feature']}: JSD = {row['jsd']:.3f}")
    
    # Verificar Chi² (variables categóricas)
    if "chi2_p" in report_df.columns:
        significant_chi2 = report_df[report_df["chi2_p"] < thresholds["chi2_p"]]
        if len(significant_chi2) > 0:
            alerts.append(f" ADVERTENCIA: {len(significant_chi2)} variable(s) con Chi² p-value < {thresholds['chi2_p']}")
    
    if len(alerts) == 0:
        alerts.append(" No se detectó drift significativo en ninguna variable")
    else:
        alerts.insert(0, "\n" + "="*80)
        alerts.insert(1, "ALERTAS DE DATA DRIFT")
        alerts.insert(2, "="*80)
        alerts.append("="*80)
        alerts.append("\n RECOMENDACIONES:")
        alerts.append("   1. Revisar el pipeline de datos de entrada")
        alerts.append("   2. Considerar reentrenamiento del modelo")
        alerts.append("   3. Validar calidad de datos en producción")
        alerts.append("   4. Investigar cambios en la población objetivo")
    
    return alerts


def plot_distribution_comparison(
    ref_data: pd.Series,
    current_data: pd.Series,
    feature_name: str,
    output_path: Path
):
    
    plt.figure(figsize=(12, 5))
    
    if pd.api.types.is_numeric_dtype(ref_data):
        # KDE plot para variables numéricas
        plt.subplot(1, 2, 1)
        ref_data.plot(kind='hist', bins=30, alpha=0.6, label='Histórico (Train)', density=True, color='#568f8b')
        current_data.plot(kind='hist', bins=30, alpha=0.6, label='Actual (Production)', density=True, color='#cd7e59')
        plt.xlabel(feature_name, fontsize=12)
        plt.ylabel('Densidad', fontsize=12)
        plt.title(f'Distribución: {feature_name}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        ref_data.plot(kind='kde', label='Histórico (Train)', color='#568f8b', linewidth=2)
        current_data.plot(kind='kde', label='Actual (Production)', color='#cd7e59', linewidth=2)
        plt.xlabel(feature_name, fontsize=12)
        plt.ylabel('Densidad (KDE)', fontsize=12)
        plt.title(f'Densidad de Kernel: {feature_name}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    else:
        # Bar plot para variables categóricas
        plt.subplot(1, 2, 1)
        ref_counts = ref_data.value_counts().sort_index()
        ref_counts.plot(kind='bar', alpha=0.7, color='#568f8b', label='Histórico')
        plt.xlabel('Categoría', fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        plt.title(f'Histórico: {feature_name}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        curr_counts = current_data.value_counts().sort_index()
        curr_counts.plot(kind='bar', alpha=0.7, color='#cd7e59', label='Actual')
        plt.xlabel('Categoría', fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        plt.title(f'Actual: {feature_name}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def monitor(
    reference_csv: str,
    current_csv: str,
    output_report: str = "../../reports/drift_report.csv",
    generate_plots: bool = True,
    top_n_plots: int = 5
) -> pd.DataFrame:
    
    print("\n" + "="*80)
    print("MONITOREO DE DATA DRIFT")
    print("="*80 + "\n")
    
    # Load data
    ref = pd.read_csv(reference_csv)
    cur = pd.read_csv(current_csv)
    
    print(f" Datos cargados:")
    print(f"   Referencia (Train): {ref.shape}")
    print(f"   Actual (Production): {cur.shape}\n")
    
    # Calculate drift metrics
    report_rows = []
    for col in ref.columns:
        if col not in cur.columns:
            continue
        if pd.api.types.is_numeric_dtype(ref[col]):
            stats = drift_for_numeric(ref[col].dropna(), cur[col].dropna())
        else:
            stats = drift_for_categorical(ref[col].dropna().astype(str), cur[col].dropna().astype(str))
        row = {"feature": col}
        row.update(stats)
        report_rows.append(row)

    report_df = pd.DataFrame(report_rows)
    
    # Add timestamp
    report_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save report
    out = Path(output_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out, index=False)
    
    print(f" Reporte guardado en: {out}\n")
    
    # Generate alerts
    alerts = generate_drift_alerts(report_df)
    for alert in alerts:
        print(alert)
    
    # Generate comparison plots for top N features with highest drift
    if generate_plots and "ks_stat" in report_df.columns:
        print("\n" + "="*80)
        print(f"GENERANDO GRÁFICOS COMPARATIVOS (Top {top_n_plots})")
        print("="*80 + "\n")
        
        plots_dir = out.parent / "drift_plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Sort by KS statistic (highest drift first)
        top_drift = report_df.nlargest(top_n_plots, "ks_stat")
        
        for _, row in top_drift.iterrows():
            feature = row["feature"]
            print(f"    Graficando: {feature} (KS = {row['ks_stat']:.3f})")
            plot_distribution_comparison(
                ref[feature].dropna(),
                cur[feature].dropna(),
                feature,
                plots_dir / f"drift_{feature}.png"
            )
        
        print(f"\n Gráficos guardados en: {plots_dir}")
    
    print("\n" + "="*80 + "\n")
    
    return report_df


if __name__ == "__main__":
    """
    Ejemplo de uso: comparar conjunto de entrenamiento vs datos actuales.
    
    En producción, este script se ejecutaría periódicamente (diario/semanal)
    para monitorear drift en datos de entrada.
    """
    # Paths relativos al script
    script_dir = Path(__file__).parent
    ref_csv = script_dir.parent.parent / "data" / "train.csv"
    cur_csv = script_dir.parent.parent / "data" / "test.csv"
    output_report = script_dir.parent.parent / "reports" / "drift_report.csv"
    
    try:
        # Ejecutar monitoreo con generación de gráficos
        df = monitor(
            reference_csv=str(ref_csv),
            current_csv=str(cur_csv),
            output_report=str(output_report),
            generate_plots=True,
            top_n_plots=5
        )
        
        print("\n RESUMEN DE DRIFT POR VARIABLE:")
        print("="*80)
        if "ks_stat" in df.columns:
            print("\nTop 5 variables con mayor KS statistic:")
            print(df.nlargest(5, "ks_stat")[["feature", "ks_stat", "ks_p", "psi"]])
        
        if "jsd" in df.columns and df["jsd"].notna().any():
            print("\nVariables categóricas con mayor JSD:")
            print(df[df["jsd"].notna()][["feature", "jsd", "chi2", "chi2_p"]])
        
        print("\n" + "="*80)
        print(" Monitoreo completado exitosamente")
        print("="*80 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n Error: Archivos no encontrados.")
        print(f"   Asegúrate de ejecutar primero ft_engineering.py")
        print(f"   Buscando: {ref_csv} y {cur_csv}\n")
    except Exception as e:
        print(f"\n Error en monitoreo: {e}\n")

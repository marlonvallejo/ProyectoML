


from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    cohen_kappa_score, 
    accuracy_score, 
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False


def FE(df: pd.DataFrame) -> pd.DataFrame:
   
    preproc_path = Path(__file__).parent.parent.parent / "data" / "preprocessor.joblib"
    if preproc_path.exists():
        preprocessor = joblib.load(preproc_path)
        arr = preprocessor.transform(df)
        return pd.DataFrame(arr)

    # fallback: simple imputation for numeric columns
    df_out = df.copy()
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df_out[c] = df_out[c].fillna(df_out[c].median())
    return df_out


def summarize_classification(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
    
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa_quadratic": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "kappa_linear": float(cohen_kappa_score(y_true, y_pred, weights="linear")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    }


def build_model(model_type: str, **kwargs) -> Any:
   
    if model_type == "lightgbm" and _HAS_LGBM:
        return LGBMClassifier(
            random_state=kwargs.get('random_state', 42),
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', -1),
            learning_rate=kwargs.get('learning_rate', 0.1),
            verbose=-1
        )
    elif model_type == "randomforest":
        return RandomForestClassifier(
            random_state=kwargs.get('random_state', 42),
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 2)
        )
    elif model_type == "svm":
        return SVC(
            random_state=kwargs.get('random_state', 42),
            kernel=kwargs.get('kernel', 'rbf'),
            C=kwargs.get('C', 1.0),
            gamma=kwargs.get('gamma', 'scale')
        )
    else:
        # Default to RandomForest
        return RandomForestClassifier(random_state=42, n_estimators=200)


def train_cv_single_model(
    train_df: pd.DataFrame, 
    features: List[str], 
    model_type: str = "lightgbm",
    target_col: str = "quality", 
    n_splits: int = 5,
    **model_kwargs
) -> Tuple[pd.DataFrame, Dict, Any]:
    
    X = train_df[features]
    y = train_df[target_col]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    fold_metrics = []

    for fold, (tr_ix, vl_ix) in enumerate(cv.split(train_df, train_df[target_col])):
        X_tr, y_tr = X.iloc[tr_ix].copy(), y.iloc[tr_ix].copy()
        X_vl, y_vl = X.iloc[vl_ix].copy(), y.iloc[vl_ix].copy()

        # Apply FE
        X_tr_fe = FE(X_tr)
        X_vl_fe = FE(X_vl)

        # Build model
        model = build_model(model_type, **model_kwargs)
        model.fit(X_tr_fe, y_tr)
        y_pred = model.predict(X_vl_fe)
        
        # Calculate metrics
        metrics = summarize_classification(y_vl, y_pred, model_name=f"{model_type}_fold{fold}")
        metrics["fold"] = fold
        fold_metrics.append(metrics)
        models.append(model)

    # Aggregate results
    metrics_df = pd.DataFrame(fold_metrics)
    summary = {
        "model_type": model_type,
        "mean_kappa_quadratic": float(metrics_df["kappa_quadratic"].mean()),
        "std_kappa_quadratic": float(metrics_df["kappa_quadratic"].std()),
        "mean_accuracy": float(metrics_df["accuracy"].mean()),
        "std_accuracy": float(metrics_df["accuracy"].std()),
        "mean_f1_macro": float(metrics_df["f1_macro"].mean()),
        "std_f1_macro": float(metrics_df["f1_macro"].std())
    }

    # Select best model (highest kappa in a fold)
    best_idx = int(metrics_df["kappa_quadratic"].idxmax())
    best_model = models[best_idx]

    return metrics_df, summary, best_model


def compare_models(
    train_df: pd.DataFrame,
    features: List[str],
    model_types: List[str] = None,
    target_col: str = "quality",
    n_splits: int = 5,
    output_dir: Path = None,
    reports_dir: Path = None
) -> Dict[str, Any]:
    
    # Setup paths
    script_dir = Path(__file__).parent
    if output_dir is None:
        output_dir = script_dir.parent.parent / "models"
    if reports_dir is None:
        reports_dir = script_dir.parent.parent / "reports"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Default models to compare
    if model_types is None:
        model_types = ["lightgbm", "randomforest"] if _HAS_LGBM else ["randomforest"]

    results = {}
    all_fold_metrics = []
    comparison_summary = []

    print("="*80)
    print("COMPARACIÓN DE MODELOS")
    print("="*80)

    for model_type in model_types:
        print(f"\n Entrenando {model_type.upper()}...")
        
        fold_metrics, summary, best_model = train_cv_single_model(
            train_df, features, model_type, target_col, n_splits
        )
        
        results[model_type] = {
            "fold_metrics": fold_metrics,
            "summary": summary,
            "best_model": best_model
        }
        
        # Collect for comparison
        fold_metrics["model_type"] = model_type
        all_fold_metrics.append(fold_metrics)
        comparison_summary.append(summary)
        
        print(f"    Kappa medio: {summary['mean_kappa_quadratic']:.4f} (±{summary['std_kappa_quadratic']:.4f})")
        print(f"    Accuracy medio: {summary['mean_accuracy']:.4f} (±{summary['std_accuracy']:.4f})")

    # Combine all metrics
    all_metrics_df = pd.concat(all_fold_metrics, ignore_index=True)
    comparison_df = pd.DataFrame(comparison_summary)

    # Select best model (highest mean kappa)
    best_model_type = comparison_df.loc[comparison_df["mean_kappa_quadratic"].idxmax(), "model_type"]
    best_model_obj = results[best_model_type]["best_model"]
    
    print(f"\n{'='*80}")
    print(f" MEJOR MODELO: {best_model_type.upper()}")
    print(f"   Kappa: {comparison_df.loc[comparison_df['model_type']==best_model_type, 'mean_kappa_quadratic'].values[0]:.4f}")
    print(f"   Accuracy: {comparison_df.loc[comparison_df['model_type']==best_model_type, 'mean_accuracy'].values[0]:.4f}")
    print(f"{'='*80}\n")

    # Save best model
    joblib.dump(best_model_obj, output_dir / "best_model.joblib")
    
    # Save reports
    all_metrics_df.to_csv(reports_dir / "cv_fold_metrics.csv", index=False)
    comparison_df.to_csv(reports_dir / "cv_summary.csv", index=False)

    # Generate comparison plots
    plot_model_comparison(all_metrics_df, comparison_df, reports_dir)

    return {
        "best_model_type": best_model_type,
        "best_model": best_model_obj,
        "all_results": results,
        "comparison_summary": comparison_df,
        "all_fold_metrics": all_metrics_df
    }


def plot_model_comparison(
    all_metrics_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    reports_dir: Path
):
   
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Figure 1: Comparison of mean metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Kappa comparison
    axes[0].bar(comparison_df["model_type"], comparison_df["mean_kappa_quadratic"], 
                color=['#568f8b', '#cd7e59', '#ddb247'][:len(comparison_df)], alpha=0.7)
    axes[0].errorbar(comparison_df["model_type"], comparison_df["mean_kappa_quadratic"],
                     yerr=comparison_df["std_kappa_quadratic"], fmt='none', color='black', capsize=5)
    axes[0].set_title("Cohen's Kappa (Quadratic)", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Kappa Score", fontsize=12)
    axes[0].set_ylim([0, 1])
    
    # Accuracy comparison
    axes[1].bar(comparison_df["model_type"], comparison_df["mean_accuracy"],
                color=['#568f8b', '#cd7e59', '#ddb247'][:len(comparison_df)], alpha=0.7)
    axes[1].errorbar(comparison_df["model_type"], comparison_df["mean_accuracy"],
                     yerr=comparison_df["std_accuracy"], fmt='none', color='black', capsize=5)
    axes[1].set_title("Accuracy", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Accuracy Score", fontsize=12)
    axes[1].set_ylim([0, 1])
    
    # F1 comparison
    axes[2].bar(comparison_df["model_type"], comparison_df["mean_f1_macro"],
                color=['#568f8b', '#cd7e59', '#ddb247'][:len(comparison_df)], alpha=0.7)
    axes[2].errorbar(comparison_df["model_type"], comparison_df["mean_f1_macro"],
                     yerr=comparison_df["std_f1_macro"], fmt='none', color='black', capsize=5)
    axes[2].set_title("F1-Score (Macro)", fontsize=14, fontweight='bold')
    axes[2].set_ylabel("F1 Score", fontsize=12)
    axes[2].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(reports_dir / "model_comparison_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Boxplots by fold
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.boxplot(data=all_metrics_df, x="model_type", y="kappa_quadratic", ax=axes[0], palette="Set2")
    axes[0].set_title("Kappa Distribution by Model (5-Fold CV)", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Cohen's Kappa (Quadratic)", fontsize=12)
    axes[0].set_xlabel("Model Type", fontsize=12)
    
    sns.boxplot(data=all_metrics_df, x="model_type", y="accuracy", ax=axes[1], palette="Set2")
    axes[1].set_title("Accuracy Distribution by Model (5-Fold CV)", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_xlabel("Model Type", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(reports_dir / "model_comparison_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Gráficos guardados en {reports_dir}")


def main(csv_path: str = "../../Base_de_datos.csv"):
    
    print("\n" + "="*80)
    print("ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
    print("="*80 + "\n")
    
    # Load dataset
    df = pd.read_csv(csv_path)
    if "quality" not in df.columns:
        raise ValueError("Expected target column 'quality' in dataset")

    # Train/test split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, 
        stratify=df.quality if df.quality.nunique() > 1 else None
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    features = [c for c in train_df.columns if c != "quality"]

    print(f" Dataset cargado:")
    print(f"   Train: {train_df.shape}")
    print(f"   Test: {test_df.shape}")
    print(f"   Features: {len(features)}\n")

    # Compare models
    model_types = ["lightgbm", "randomforest"] if _HAS_LGBM else ["randomforest"]
    comparison_results = compare_models(train_df, features, model_types=model_types)
    
    best_model = comparison_results["best_model"]
    best_model_type = comparison_results["best_model_type"]

    # Final evaluation on holdout test
    print("\n" + "="*80)
    print("EVALUACIÓN FINAL EN TEST SET")
    print("="*80 + "\n")
    
    X_test_fe = FE(test_df[features])
    y_test = test_df.quality
    y_pred = best_model.predict(X_test_fe)
    
    final_metrics = summarize_classification(y_test, y_pred, model_name=best_model_type)
    
    print(f" Modelo: {best_model_type.upper()}")
    print(f"   Kappa (Quadratic): {final_metrics['kappa_quadratic']:.4f}")
    print(f"   Kappa (Linear): {final_metrics['kappa_linear']:.4f}")
    print(f"   Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"   F1-Score (Macro): {final_metrics['f1_macro']:.4f}")
    print(f"   Precision (Macro): {final_metrics['precision_macro']:.4f}")
    print(f"   Recall (Macro): {final_metrics['recall_macro']:.4f}")

    # Save final metrics
    reports_dir = Path(__file__).parent.parent.parent / "reports"
    pd.Series(final_metrics).to_csv(reports_dir / "final_metrics.csv")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {best_model_type.upper()}', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(class_report).T.to_csv(reports_dir / "classification_report.csv")
    
    print(f"\n Todos los reportes guardados en {reports_dir}")
    print("="*80 + "\n")

    return {
        "comparison": comparison_results,
        "final_metrics": final_metrics,
        "best_model_type": best_model_type
    }


if __name__ == "__main__":
    res = main()
    print(res)

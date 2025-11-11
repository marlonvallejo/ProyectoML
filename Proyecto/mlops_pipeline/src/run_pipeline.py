
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).parent.parent

def main():
    logging.info("Starting full pipeline run")

    # 1) Feature engineering
    try:
        logging.info("Running feature engineering")
        from mlops_pipeline.src.ft_engineering import create_datasets
        # Use repo-root Base_de_datos.csv
        data_path = ROOT.parent / "Base_de_datos.csv"
        create_datasets(csv_path=str(data_path), output_dir=str(ROOT.parent / "data"))
    except Exception as e:
        logging.exception("Feature engineering failed: %s", e)

    # 2) Training and evaluation
    logging.info("Running training and evaluation")
    from mlops_pipeline.src.model_training_evaluation import main as train_main
    train_main(csv_path=str(data_path))

    # 3) Monitoring (compare train vs test processed files)
    try:
        logging.info("Running monitoring checks")
        from mlops_pipeline.src.model_monitoring import monitor
        processed = ROOT.parent / "data"
        ref = processed / "X_train_processed.csv"
        cur = processed / "X_test_processed.csv"
        if ref.exists() and cur.exists():
            monitor(str(ref), str(cur), output_report=str(ROOT.parent / "reports" / "drift_report.csv"))
        else:
            logging.warning("Processed datasets not found for monitoring: %s %s", ref, cur)
    except Exception as e:
        logging.exception("Monitoring failed: %s", e)

    logging.info("Pipeline finished")


if __name__ == "__main__":
    main()

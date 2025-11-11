

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import joblib


def build_feature_pipeline(numeric_features, categorical_features):

	numeric_transformer = Pipeline([
		("imputer", SimpleImputer(strategy="median")),
		("scaler", StandardScaler()),
	])

	categorical_transformer = Pipeline([
		("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
		("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
	])

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numeric_features),
			("cat", categorical_transformer, categorical_features),
		],
		remainder="drop",
	)

	return preprocessor


def create_datasets(csv_path: str = "../../Base_de_datos.csv", target_col: str = "quality", test_size: float = 0.2, random_state: int = 42, output_dir: str = "../../data", ordinal_features: list = None):
	
	csv_path = Path(csv_path)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(csv_path)
	if target_col not in df.columns:
		raise ValueError(f"target_col '{target_col}' not found in CSV columns: {list(df.columns)}")

	
	X = df.drop(columns=[target_col])
	y = df[target_col]

	numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
	categorical_features = X.select_dtypes(include=[object, "category"]).columns.tolist()

	# If ordinal features provided, remove them from categorical_features and handle separately
	ordinal_features = ordinal_features or []
	categorical_features = [c for c in categorical_features if c not in ordinal_features]

	
	numeric_transformer = Pipeline([
		("imputer", SimpleImputer(strategy="median")),
		("scaler", StandardScaler()),
	])

	# Categorical (nominal)
	categorical_transformer = Pipeline([
		("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
		("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
	])

	# Ordinal
	ordinal_transformer = Pipeline([
		("imputer", SimpleImputer(strategy="most_frequent")),
		("ordinal", OrdinalEncoder()),
	])

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numeric_features),
			("cat", categorical_transformer, categorical_features),
			("ord", ordinal_transformer, ordinal_features),
		],
		remainder="drop",
	)

	

	X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique()>1 else None)

	
	preprocessor.fit(X_train_raw)
	X_train = pd.DataFrame(preprocessor.transform(X_train_raw))
	X_test = pd.DataFrame(preprocessor.transform(X_test_raw))

	
	joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
	X_train.to_csv(output_dir / "X_train_processed.csv", index=False)
	X_test.to_csv(output_dir / "X_test_processed.csv", index=False)
	y_train.to_csv(output_dir / "y_train.csv", index=False)
	y_test.to_csv(output_dir / "y_test.csv", index=False)

	return X_train, X_test, y_train, y_test


if __name__ == "__main__":
	# Quick smoke-run when executed directly
	try:
		X_train, X_test, y_train, y_test = create_datasets()
		print("Datasets created:", X_train.shape, X_test.shape)
	except Exception as e:
		print("Error creating datasets:", e)

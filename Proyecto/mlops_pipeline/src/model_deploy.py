
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import io


class WineFeatures(BaseModel):
    """Modelo de features individuales del vino."""
    fixed_acidity: float = Field(..., ge=0, le=20, description="Acidez fija (g/dm³)")
    volatile_acidity: float = Field(..., ge=0, le=2, description="Acidez volátil (g/dm³)")
    citric_acid: float = Field(..., ge=0, le=2, description="Ácido cítrico (g/dm³)")
    residual_sugar: float = Field(..., ge=0, le=100, description="Azúcar residual (g/dm³)")
    chlorides: float = Field(..., ge=0, le=1, description="Cloruros (g/dm³)")
    free_sulfur_dioxide: float = Field(..., ge=0, le=200, description="SO₂ libre (mg/dm³)")
    total_sulfur_dioxide: float = Field(..., ge=0, le=400, description="SO₂ total (mg/dm³)")
    density: float = Field(..., ge=0.98, le=1.01, description="Densidad (g/cm³)")
    pH: float = Field(..., ge=2.5, le=4.5, description="pH")
    sulphates: float = Field(..., ge=0, le=2, description="Sulfatos (g/dm³)")
    alcohol: float = Field(..., ge=8, le=15, description="Alcohol (% vol)")


class PredictRequest(BaseModel):
    """Solicitud de predicción para uno o más registros."""
    data: List[Dict[str, Any]]
    
    @validator('data')
    def validate_data(cls, v):
        if len(v) == 0:
            raise ValueError("El campo 'data' no puede estar vacío")
        return v


class PredictResponse(BaseModel):
    """Respuesta de predicción."""
    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None
    model_version: str
    timestamp: str
    n_samples: int


app = FastAPI(
    title="Wine Quality Prediction API",
    description="API REST para predicción de calidad de vino usando LightGBM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Paths are relative to project root (/app in Docker)
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "best_model.joblib"
PREPROCESSOR_PATH = Path(__file__).parent.parent.parent / "data" / "preprocessor.joblib"


def load_artifacts():
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError("Model or preprocessor artifact not found. Run training pipeline first.")
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


@app.on_event("startup")
def startup_event():
    global MODEL, PREPROCESSOR
    MODEL, PREPROCESSOR = load_artifacts()


@app.get("/")
def root():
    """Endpoint raíz con información de la API."""
    return {
        "message": "Wine Quality Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Predicción (JSON)",
            "/predict/batch": "Predicción por lotes (CSV)",
            "/docs": "Documentación interactiva (Swagger)",
            "/redoc": "Documentación (ReDoc)"
        }
    }


@app.get("/health")
def health():
    """
    Health check endpoint para monitoreo.
    
    Verifica que el modelo y preprocesador estén cargados.
    """
    try:
        model_exists = MODEL is not None
        preprocessor_exists = PREPROCESSOR is not None
        return {
            "status": "healthy" if (model_exists and preprocessor_exists) else "degraded",
            "model_loaded": model_exists,
            "preprocessor_loaded": preprocessor_exists,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
   
    try:
        # Convert to DataFrame
        df = pd.DataFrame(req.data)
        
        # Validate required columns
        required_cols = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
            'density', 'pH', 'sulphates', 'alcohol'
        ]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Faltan columnas requeridas: {missing_cols}"
            )
        
        # Preprocess and predict
        X = PREPROCESSOR.transform(df[required_cols])
        preds = MODEL.predict(X)
        
        # Get probabilities if available
        proba = None
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X).tolist()
        
        return PredictResponse(
            predictions=preds.tolist(),
            probabilities=proba,
            model_version="lightgbm_v1",
            timestamp=datetime.now().isoformat(),
            n_samples=len(df)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en predicción: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="El archivo debe ser formato CSV"
            )
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validate required columns
        required_cols = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
            'density', 'pH', 'sulphates', 'alcohol'
        ]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Faltan columnas requeridas en CSV: {missing_cols}"
            )
        
        # Preprocess and predict
        X = PREPROCESSOR.transform(df[required_cols])
        preds = MODEL.predict(X)
        
        # Add predictions to dataframe
        df['predicted_quality'] = preds
        
        # Add probabilities if available
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X)
            for i in range(proba.shape[1]):
                df[f'prob_class_{i+3}'] = proba[:, i]  # Classes 3-8
        
        # Convert to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        
        return JSONResponse(
            content={
                "message": f"Predicciones generadas para {len(df)} registros",
                "n_samples": len(df),
                "csv_preview": output.getvalue().split('\n')[:5],
                "download_url": "/predict/batch (use Content-Type: text/csv para descargar)"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en predicción batch: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

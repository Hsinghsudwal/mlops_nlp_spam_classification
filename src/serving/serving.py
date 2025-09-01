import os
import pickle
import mlflow.pyfunc
import pandas as pd


def load_model():
    """
    Try to load MLflow model first, else local pickle.
    Returns wrapper dict: model, pipeline, and label_encoder.
    """
    try:
        # Example: adjust to your MLflow model URI
        model_uri = "models:/spam_classifier/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        return {"type": "mlflow", "model": model}
    except Exception:
        # Fallback: local pickle
        local_path = "outputs/models/trained_models.pkl"
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"No MLflow model or local model found at {local_path}")

        with open(local_path, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict):
            return {
                "type": "local",
                "model": obj.get("best_model"),
                "pipeline": obj.get("pipeline"),
                "label_encoder": obj.get("label_encoder"),
            }
        else:
            return {"type": "local", "model": obj, "pipeline": None, "label_encoder": None}


def predict_single(text_input: str, wrapper: dict):
    """Run single text prediction depending on source (MLflow or local pickle)."""
    if wrapper["type"] == "mlflow":
        return wrapper["model"].predict([text_input])[0]
    else:
        pipeline = wrapper.get("pipeline")
        model = wrapper["model"]
        le = wrapper.get("label_encoder")

        X = pipeline.transform([text_input]) if pipeline else [text_input]
        pred = model.predict(X)

        if le:
            pred = le.inverse_transform(pred)
        return pred[0]


def predict_batch(df: pd.DataFrame, wrapper: dict, text_column: str = None) -> pd.DataFrame:
    """Run batch predictions on a DataFrame."""
    if not text_column:
        text_column = df.columns[0]
    
    texts = df[text_column].astype(str)

    if wrapper["type"] == "mlflow":
        preds = wrapper["model"].predict(texts.tolist())
    else:
        pipeline = wrapper.get("pipeline")
        model = wrapper["model"]
        le = wrapper.get("label_encoder")

        X = pipeline.transform(texts) if pipeline else texts
        preds = model.predict(X)

        if le:
            preds = le.inverse_transform(preds)

    df["prediction"] = preds
    return df

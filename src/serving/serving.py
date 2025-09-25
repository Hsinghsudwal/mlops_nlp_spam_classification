import os
import json
import pickle
import random
import mlflow
import pandas as pd
from datetime import datetime
from mlflow import MlflowClient
from utils.config_manager import ConfigManager

# ----------------- Paths -----------------
config = ConfigManager.load_file("config/config.yaml")
DEPLOYFILE = "outputs/registry/latest_deployment.json"
LOCALMODEL = "outputs/evaluate/full_pipeline.pkl"
HISTORY_FILE = "user_app/history/predictions.csv"
os.makedirs("user_app/history", exist_ok=True)
os.makedirs("user_app/uploads", exist_ok=True)


MODEL_CACHE = {}

# def get_mlflow_client():
#     """Initialize MLflow client from config.yaml."""
    
#     tracking_uri = config.get("mlflow_config.remote_server_uri") or config.get("mlflow_config.mlflow_tracking_uri")
#     if tracking_uri:
#         mlflow.set_tracking_uri(tracking_uri)
#         return MlflowClient(tracking_uri=tracking_uri)
#     return MlflowClient()

# def get_experiment():
#     """Fetch experiment object from config.yaml."""
    
#     tracking_uri = config.get("mlflow_config.remote_server_uri") or config.get("mlflow_config.mlflow_tracking_uri")
#     if tracking_uri:
#         mlflow.set_tracking_uri(tracking_uri)
#     experiment_name = config.get("mlflow_config.experiment_name", "default_experiment")
#     return mlflow.get_experiment_by_name(experiment_name)
    

# def get_latest_production_model():
#     if "latest_production_model" in MODEL_CACHE:
#         print("✅ Loaded model from cache.")
#         return MODEL_CACHE["latest_production_model"]
    
#     try:
        
#         client = get_mlflow_client()
#         exp = get_experiment()
#         if not exp:
#             print("Experiment not found in config.")
#             return None, None, None
        
#         exp_id = exp.experiment_id
        
#         all_versions = client.search_model_versions("") 

#         latest_model = None
#         latest_ts = 0

#         for mv in all_versions:
#             # print(mv)
#             #check only match
#             run = client.get_run(mv.run_id)
#             if run.info.experiment_id != exp_id:
#                 continue  # skip models
            
#             print(f"Checking {mv.name} v{mv.version} (aliases={mv.aliases})")
#             aliases = mv.aliases or []
#             deployment_status = mv.tags.get("deployment_status", "inactive") if mv.tags else "inactive"

#             if "production" in aliases and deployment_status == "active":
#                 ts = int(mv.creation_timestamp)
#                 if ts > latest_ts:
#                     latest_ts = ts
#                     latest_model = mv

#         if not latest_model:
#             print("No active production model found.")
#             return None, None, None

#         run_id = latest_model.run_id
#         model_uri = f"runs:/{run_id}/model"
#         pipeline = mlflow.sklearn.load_model(model_uri)

#         prev_model = json.loads(latest_model.tags.get("previous_model", "null")) if latest_model.tags else None
#         deploy_info = {
#             "model_name": latest_model.name,
#             "version": latest_model.version,
#             "run_id": run_id,
#             "aliases": latest_model.aliases,
#             "tags": latest_model.tags or {},
#             "routing_config": json.loads(latest_model.tags.get("routing_target", "{}")) if latest_model.tags else {},
#             "deployment_status": deployment_status,
#             "previous_model": prev_model,
#         }

#         print(f"✅ Loaded production model: {latest_model.name} v{latest_model.version}")
#         MODEL_CACHE["latest_production_model"] = (pipeline, deploy_info, prev_model)
#         return pipeline, deploy_info, prev_model

#     except Exception as e:
#         print(f"⚠️ Error loading production model: {e}")
#         return None, None, None


# local
def load_from_local():
    if os.path.exists(LOCALMODEL):
        with open(LOCALMODEL, "rb") as f:
            pipeline = pickle.load(f)

        if os.path.exists(DEPLOYFILE):
            with open(DEPLOYFILE, "r") as f:
                deploy_info = json.load(f)
        else:
            deploy_info = {
                "status": "local",
                "data": {
                    "model_name": "local_pipeline",
                    "version": "N/A",
                    "deployment_status": "active",
                    "routing_config": {"strategy": "direct", "target": "local"},
                    "previous_model": None,
                },
                "message": "Local deployment",
            }
        return pipeline, deploy_info, None
    raise FileNotFoundError(f"No local model at {LOCALMODEL}")


def load_deploy_pipeline():
    # pipeline, deploy_info, prev_model = get_latest_production_model()
    # if not pipeline:
    #     print("⚠️ Falling back to local model...")
    pipeline, deploy_info, prev_model = load_from_local()
    return pipeline, deploy_info, prev_model


# pred
def predict_text(pipeline, text: str):
    pred = pipeline.predict([text])[0]
    encoder = pipeline.named_steps["preprocessor"].label_encoder
    return encoder.classes_[pred]


def log_history(records, mode="single"):
    """Append new predictions to CSV with mode."""
    new_df = pd.DataFrame(records)
    new_df["mode"] = mode

    if os.path.exists(HISTORY_FILE):
        new_df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        new_df.to_csv(HISTORY_FILE, mode="w", header=True, index=False)

def route_prediction(text: str):
    pipeline, deploy_info, prev_model_info = load_deploy_pipeline()
    strategy = deploy_info.get("routing_config", {}).get("strategy", "direct")
    pred = predict_text(pipeline, text)

    if strategy == "shadow" and prev_model_info:
        try:
            prev_pipeline_uri = f"runs:/{prev_model_info['run_id']}/model"
            if prev_pipeline_uri not in MODEL_CACHE:
                MODEL_CACHE[prev_pipeline_uri] = mlflow.sklearn.load_model(prev_pipeline_uri)
            prev_pipeline = MODEL_CACHE[prev_pipeline_uri]
            prev_pred = predict_text(prev_pipeline, text)
        except Exception:
            pass
    elif strategy == "canary" and prev_model_info:
        canary_ratio = deploy_info.get("routing_config", {}).get("canary_ratio", 0.1)
        if random.random() < canary_ratio:
            try:
                prev_pipeline_uri = f"runs:/{prev_model_info['run_id']}/model"
                if prev_pipeline_uri not in MODEL_CACHE:
                    MODEL_CACHE[prev_pipeline_uri] = mlflow.sklearn.load_model(prev_pipeline_uri)
                prev_pipeline = MODEL_CACHE[prev_pipeline_uri]
                pred = predict_text(prev_pipeline, text)
            except Exception:
                pass
    elif strategy == "blue_green" and prev_model_info:
        if random.random() < 0.5:
            try:
                prev_pipeline_uri = f"runs:/{prev_model_info['run_id']}/model"
                if prev_pipeline_uri not in MODEL_CACHE:
                    MODEL_CACHE[prev_pipeline_uri] = mlflow.sklearn.load_model(prev_pipeline_uri)
                prev_pipeline = MODEL_CACHE[prev_pipeline_uri]
                pred = predict_text(prev_pipeline, text)
            except Exception:
                pass

    log_history([{"input": text, "prediction": pred}], mode="single")
    return pred

# def route_prediction(text: str):
#     pipeline, deploy_info, prev_model_info = load_deploy_pipeline()
#     strategy = deploy_info.get("routing_config", {}).get("strategy", "direct")
#     pred = predict_text(pipeline, text) 

#     if strategy == "shadow" and prev_model_info:
#         try:
#             prev_pipeline_uri = f"runs:/{prev_model_info['run_id']}/model"
#             prev_pipeline = mlflow.sklearn.load_model(prev_pipeline_uri)
#             prev_pred = predict_text(prev_pipeline, text)
#         except Exception:
#             pass
#     elif strategy == "canary" and prev_model_info:
#         canary_ratio = deploy_info.get("routing_config", {}).get("canary_ratio", 0.1)
#         # if prev_model_info and random.random() >= canary_ratio:
#         if random.random() < canary_ratio:
#             try:
#                 prev_pipeline_uri = f"runs:/{prev_model_info['run_id']}/model"
#                 prev_pipeline = mlflow.sklearn.load_model(prev_pipeline_uri)
#                 pred = predict_text(prev_pipeline, text)
#             except Exception:
#                 pass
            
#     elif strategy == "blue_green" and prev_model_info:
#         if random.random() < 0.5:
#             try:
#                 prev_pipeline_uri = f"runs:/{prev_model_info['run_id']}/model"
#                 prev_pipeline = mlflow.sklearn.load_model(prev_pipeline_uri)
#                 pred = predict_text(prev_pipeline, text)
#             except Exception as e:
#                 pass

#     log_history([{"input": text, "prediction": pred}], mode="single")
#     return pred


def batch_prediction(file_path: str):
    pipeline, _, _ = load_deploy_pipeline()
    df = pd.read_csv(file_path)

    if "text" not in df.columns:
        raise ValueError("CSV must have a 'text' column")

    preds = pipeline.predict(df["text"].tolist())
    encoder = pipeline.named_steps["preprocessor"].label_encoder
    df["prediction"] = [encoder.classes_[p] for p in preds]

    log_history(
        df[["text", "prediction"]].rename(columns={"text": "input"}).to_dict(orient="records"),
        mode="batch"
    )
    return df


def get_history(mode=None):
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        if mode in ["single", "batch"]:
            df = df[df["mode"] == mode]
        return df
    return pd.DataFrame(columns=["input", "prediction", "mode"])

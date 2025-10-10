import os
import json
import pickle
import random
import mlflow
import pandas as pd
from datetime import datetime
from mlflow import MlflowClient
from utils.config_manager import ConfigManager
from utils.logger import logger


class DeployServe:
    def __init__(self):
        self.DEPLOYFILE = "outputs/registry/latest_deployment.json"
        self.LOCALMODEL = "outputs/evaluate/full_pipeline.pkl"
        self.HISTORY_FILE = "user_app/history/predictions.csv"
        os.makedirs("user_app/history", exist_ok=True)
        os.makedirs("user_app/uploads", exist_ok=True)

        self.deployment_info = {}
        self.pipeline = None  # Main model (in-memory)
        self.prev_pipeline = None  # Previous model for A/B testing
        self.prev_model_info = None

        # Track what's loaded to avoid redundant MLflow calls
        self.current_run_id = None

        self._load_deployment_info()
        self._load_or_fetch_model()

    def _load_deployment_info(self):
        """Load deployment metadata from JSON file."""
        if not os.path.exists(self.DEPLOYFILE):
            logger.warning("Deployment file not found, will use local model only.")
            return

        try:
            with open(self.DEPLOYFILE, "r") as f:
                data = json.load(f)

            if "metrics" not in data:
                raise ValueError("Invalid deployment file: missing 'metrics' section")

            self.deployment_info = data["metrics"]
            self.prev_model_info = self.deployment_info.get("previous_model")

            logger.info(
                f"Deployment config loaded: model={self.deployment_info.get('model_name')} "
                f"version={self.deployment_info.get('version')} "
                f"strategy={self.deployment_info.get('routing_config', {}).get('strategy', 'direct')}"
            )
        except Exception as e:
            logger.error(f"Failed to load deployment info: {e}")
            self.deployment_info = {}

    def _load_or_fetch_model(self):
        """Load model with priority: local pickle → MLflow (if deployment info exists)."""

        # Priority 1: Try local pickle (fastest)
        if os.path.exists(self.LOCALMODEL):
            try:
                with open(self.LOCALMODEL, "rb") as f:
                    self.pipeline = pickle.load(f)
                logger.info("Pipeline loaded from local cache")

                # Validate it matches deployment info if available
                if self.deployment_info:
                    self.current_run_id = self.deployment_info.get("run_id")
                return
            except Exception as e:
                logger.warning(f"Failed to load local pickle: {e}")

        # Priority 2: Load from MLflow if deployment info exists
        if self.deployment_info:
            self._load_from_mlflow()
        else:
            raise RuntimeError(
                "No model available: local pickle missing and no deployment info found"
            )

    def _load_from_mlflow(self):
        """Fetch pipeline from MLflow registry and cache locally."""
        run_id = self.deployment_info.get("run_id")
        model_name = self.deployment_info.get("model_name")
        version = self.deployment_info.get("version")

        if not (run_id and model_name and version):
            raise ValueError(
                "Deployment info incomplete: missing run_id, model_name, or version"
            )

        # if already loaded
        if self.current_run_id == run_id and self.pipeline is not None:
            logger.info("Model already loaded in memory, skipping MLflow fetch")
            return

        model_uri = f"models:/{model_name}/{version}"
        logger.info(f"Fetching model from MLflow: {model_uri} (this may take time...)")

        try:
            self.pipeline = mlflow.sklearn.load_model(model_uri)
            self.current_run_id = run_id

            # Cache load
            with open(self.LOCALMODEL, "wb") as f:
                pickle.dump(self.pipeline, f)
            logger.info("✓ Pipeline fetched and cached locally")
        except Exception as e:
            logger.error(f"Failed to load from MLflow: {e}")
            raise

    def _load_previous_model(self):
        """Load previous model from MLflow (lazy loading, only when needed)."""
        if not self.prev_model_info:
            return None

        # Already loaded
        if self.prev_pipeline is not None:
            return self.prev_pipeline

        prev_run_id = self.prev_model_info.get("run_id")
        prev_uri = f"runs:/{prev_run_id}/model"

        try:
            logger.info(
                f"Loading previous model for {self.deployment_info.get('routing_config', {}).get('strategy')} routing..."
            )
            self.prev_pipeline = mlflow.sklearn.load_model(prev_uri)
            logger.info(f"✓ Previous model loaded: {prev_uri}")
        except Exception as e:
            logger.error(f"Failed to load previous model: {e}")
            return None

        return self.prev_pipeline

    def _predict_text(self, pipeline, text: str):
        """Run prediction on single text input."""
        pred = pipeline.predict([text])[0]
        encoder = pipeline.named_steps["preprocessor"].label_encoder
        return encoder.classes_[pred]

    
    def single_predict(self, text: str):
        """Route prediction according to deployment strategy."""
        if self.pipeline is None:
            raise RuntimeError("No model loaded. Check deployment setup.")

        strategy = self.deployment_info.get("routing_config", {}).get(
            "strategy", "direct"
        )
        routing_config = self.deployment_info.get("routing_config", {})

        # Default: use current model
        pred = self._predict_text(self.pipeline, text)
        model_used = "current"

        # --- Strategy Routing ---
        if strategy in ["shadow", "canary", "blue_green"] and self.prev_model_info:
            prev_pipeline = self._load_previous_model()

            if prev_pipeline:
                if strategy == "shadow":
                    # Shadow mode: always use new, log old prediction silently
                    shadow_pred = self._predict_text(prev_pipeline, text)
                    logger.debug(f"Shadow prediction (not used): {shadow_pred}")

                elif strategy == "canary":
                    # Canary: route X% to old model
                    canary_ratio = routing_config.get("canary_ratio", 0.1)
                    if random.random() < canary_ratio:
                        pred = self._predict_text(prev_pipeline, text)
                        model_used = "previous"
                        logger.debug(f"Canary routing: using previous model")


        self.log_history(
            [{"input": text, "prediction": pred, "model_used": model_used}],
            mode="single",
        )

        return pred

    def batch_predict(self, file_path: str):
        """Run batch predictions from CSV file."""
        if self.pipeline is None:
            raise RuntimeError("No model loaded. Check deployment setup.")

        df = pd.read_csv(file_path)
        if "text" not in df.columns:
            raise ValueError("CSV must have a 'text' column")

        logger.info(f"Running batch prediction on {len(df)} samples...")

        # Batch prediction
        preds = self.pipeline.predict(df["text"].tolist())
        encoder = self.pipeline.named_steps["preprocessor"].label_encoder
        df["prediction"] = [encoder.classes_[p] for p in preds]

   
        records = []
        for idx, row in df.iterrows():
            records.append({
                "input": row["text"],
                "prediction": row["prediction"],
                "model_used": "current"
            })


        self.log_history(records, mode="batch")

        logger.info(f"Batch prediction completed")
        return df
    def log_history(self, records: list, mode: str = "single"):
        """Append predictions to history CSV with fixed 7 columns."""
        if not records:
            return

        columns = ["input", "prediction", "model_used", "mode", "timestamp", "model_version", "strategy"]

        for record in records:
            record.setdefault("input", "")
            record.setdefault("prediction", "")
            record.setdefault("model_used", "current")
            record.setdefault("mode", mode)
            record["timestamp"] = datetime.now().isoformat()
            record["model_version"] = self.deployment_info.get("version", "local")
            record["strategy"] = self.deployment_info.get("routing_config", {}).get("strategy", "direct")

        new_df = pd.DataFrame(records)[columns]  
        if os.path.exists(self.HISTORY_FILE):
            try:
                old_df = pd.read_csv(self.HISTORY_FILE)
                if list(old_df.columns) != columns:
                    logger.warning("Old history CSV has wrong columns. Overwriting with correct columns.")
                    new_df.to_csv(self.HISTORY_FILE, mode="w", header=True, index=False)
                else:
                    new_df.to_csv(self.HISTORY_FILE, mode="a", header=False, index=False)
            except Exception as e:
                logger.error(f"Failed to append history, overwriting file: {e}")
                new_df.to_csv(self.HISTORY_FILE, mode="w", header=True, index=False)
        else:
            new_df.to_csv(self.HISTORY_FILE, mode="w", header=True, index=False)

        logger.info(f"✓ Logged {len(records)} records to history ({mode} mode)")


    def get_history(self, mode: str = None, limit: int = 100):
      
        columns = ["input", "prediction", "model_used", "mode", "timestamp", "model_version", "strategy"]

        if not os.path.exists(self.HISTORY_FILE):
            return pd.DataFrame(columns=columns)

        try:
            df = pd.read_csv(self.HISTORY_FILE)

            for col in columns:
                if col not in df.columns:
                    df[col] = None

            if mode in ["single", "batch"]:
                df = df[df["mode"] == mode]

            df = df.drop_duplicates(subset=["input", "timestamp"], keep="last")
            df = df.sort_values("timestamp", ascending=False).head(limit)

            return df[columns]

        except Exception as e:
            logger.error(f"Failed to read history, returning empty DataFrame: {e}")
            return pd.DataFrame(columns=columns)

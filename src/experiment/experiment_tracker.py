import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

from utils.logger import logger
from utils.config_manager import ConfigManager
from utils.results import Result, Status
from integration.storage import ArtifactStoreFactory


class MLflowTracker:
    """MLflow experiment tracking,registration, and stage/alias."""

    def __init__(self, config: str):
        self.config  = config
        # self.config = ConfigManager.load_file(config_file)
        self.artifact_store = ArtifactStoreFactory.create_store(self.config)
        self.client: Optional[MlflowClient] = None
        self.experiment_id: Optional[str] = None
        self.tracking_uri: Optional[str] = None
        self.auto_register: bool = self.config.get("mlflow_config.auto_register", False)
        self.run_id: Optional[str] = None
        self.run_name: Optional[str] = None

    def setup_mlflow(self) -> None:
        uri = (
            self.config.get("mlflow_config.remote_server_uri")
            or self.config.get("mlflow_config.mlflow_tracking_uri")
        )
        if uri:
            mlflow.set_tracking_uri(uri)
            self.tracking_uri = uri

        self.client = MlflowClient()
        mlflow.sklearn.autolog(log_models=False)
        logger.info(f"MLflow setup complete: {self.tracking_uri}")

    def setup_experiment(self) -> None:
        experiment_name = self.config.get(
            "mlflow_config.experiment_name", "default_experiment"
        )
        description = self.config.get("mlflow_config.description", "")

        exp = self.client.get_experiment_by_name(experiment_name)
        if exp:
            self.experiment_id = exp.experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")
        else:
            self.experiment_id = self.client.create_experiment(
                experiment_name,
                tags={
                    "description": description,
                    "created_at": datetime.now().isoformat(),
                },
            )
            logger.info(f"Created new experiment: {experiment_name}")

    def load_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        trainer = results.get("trainer")
        evaluator = results.get("evaluate")

        if not trainer or trainer.status != Status.SUCCESS:
            raise ValueError("Trainer results missing or failed")
        if not evaluator or evaluator.status != Status.SUCCESS:
            raise ValueError("Evaluator results missing or failed")

        trainer_data, eval_data = trainer.data, evaluator.data

        return {
            "best_model": trainer_data["best_model"],
            "best_model_name": trainer_data["best_model_name"],
            "cv_score": trainer_data.get("best_score", 0.0),
            "test_score": trainer_data.get("test_score", 0.0),
            "X_test": trainer_data["X_test"],
            "y_test": trainer_data["y_test"],
            "wrapper_pipeline": trainer_data["wrapper_pipeline"],
            "metrics": eval_data["metrics"],
            "full_pipeline": eval_data["full_pipeline"],
        }

    def log_parameters(self, extracted_data: Dict[str, Any]) -> None:
        params = {
            "problem_type": self.config.get("base.problem_type", "classification"),
            "target_column": self.config.get("base.target_column", "target"),
            "cv_folds": self.config.get("base.cv_folds", 5),
            "test_size": self.config.get("base.test_size", 0.2),
            "random_state": self.config.get("base.random_state", 42),
            "selected_model": extracted_data["best_model_name"],
            "data_shape": str(extracted_data["X_test"].shape),
        }

        model = extracted_data["best_model"]
        if hasattr(model, "get_params"):
            prefix = f"{extracted_data['best_model_name']}_"
            for key, value in model.get_params().items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    params[f"{prefix}{key}"] = value

        mlflow.log_params(params)
        logger.info("Parameters logged to MLflow")

    def log_metrics(self, extracted_data: Dict[str, Any]) -> None:
        metrics = extracted_data["metrics"]
        main_metrics = {
            "cv_score": extracted_data.get("cv_score", 0.0),
            "test_score": extracted_data.get("test_score", 0.0),
            "accuracy": metrics.get("accuracy", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1_score": metrics.get("f1_score", 0.0),
            "roc_auc": metrics.get("roc_auc", 0.0),
        }

        for key, value in main_metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                mlflow.log_metric(key, value)

        logger.info("Metrics logged to MLflow")

    def log_model(self, extracted_data: Dict[str, Any]) -> None:
        """Log trained model to MLflow."""
        # model = extracted_data["best_model"]
        full_pipeline = extracted_data["full_pipeline"]
        model_name = extracted_data["best_model_name"]
        X_test = extracted_data["X_test"]

        if full_pipeline is None:
            logger.error("No full_pipeline found in extracted_data")
            return None

        try:
            # signature = None
            X_sig = X_test.toarray() if hasattr(X_test, "toarray") else X_test
            signature = infer_signature(X_sig, full_pipeline.predict(X_test))
            # signature = infer_signature(X_test, full_pipeline.predict(X_test))
        except Exception as e:
            logger.warning(f"Could not infer signature: {e}")
            signature = None

        try:
            mlflow.sklearn.log_model(
                sk_model=full_pipeline,
                artifact_path="model",
                signature=signature,
                input_example=X_test[:5] if hasattr(X_test, "__getitem__") else None,
                # input_example=(
                #     X_test.iloc[:5].toarray()
                #     if hasattr(X_test, "iloc") and hasattr(X_test.iloc[:5], "toarray")
                #     else (
                #         X_test[:5].toarray()
                #         if hasattr(X_test, "toarray")
                #         else X_test[:5]
                #     )
                ),
                # registered_model_name=f"{model_name}_model", # if self.auto_register else None,
                # input_example=(
                #     X_test.iloc[:5] if hasattr(X_test, "iloc") else X_test[:5]
                # ),
            # )
            logger.info(f"Model {model_name} logged to MLflow")

            # Register
            registered_name = f"{model_name}_pipeline"
            model_uri = f"runs:/{self.run_id}/model"

            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=registered_name,
                tags={
                    "accuracy": str(
                        extracted_data.get("metrics", {}).get("accuracy", 0.0)
                    ),
                    "f1_score": str(
                        extracted_data.get("metrics", {}).get("f1_score", 0.0)
                    ),
                },
            )
            logger.info(
                f"Registered model: {registered_name}, version: {model_version.version}, model_uri: {model_uri},"
            )

            # mlflow.log_artifact(__file__)

            return {
                "model_name": registered_name,
                "model_version": model_version.version,
                "model_uri": model_uri,
                "run_id": self.run_id,
            }
        except Exception as e:
            logger.error(f"Failed to register model {registered_name}: {e}")

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start MLflow run."""
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=self.run_name,
            tags={"created_at": datetime.now().isoformat(), "version": "1.0.0"},
        )
        self.run_id = run.info.run_id
        logger.info(f"Started MLflow run: {self.run_id}")

    def end_run(self) -> None:
        """End MLflow run."""
        if mlflow.active_run():
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.run_id}")

    def track_experiment(
        self, results: Dict[str, Any], pipeline_id: str, run_name: Optional[str] = None
    ) -> Result:
        """Track full experiment lifecycle in MLflow."""
        try:
            self.setup_mlflow()
            self.setup_experiment()
            extracted_data = self.load_results(results)

            self.start_run(run_name)
            try:
                self.log_parameters(extracted_data)
                self.log_metrics(extracted_data)
                model_info = self.log_model(extracted_data)

                # registered_model_info = self.register_model(extracted_data)
                output = {
                    "run_id": self.run_id,
                    "run_name": self.run_name,
                    "experiment_id": self.experiment_id,
                    # "model_name": extracted_data["best_model_name"],
                    # "accuracy": extracted_data["metrics"].get("accuracy", 0.0),
                    # "f1_score": extracted_data["metrics"].get("f1_score", 0.0),
                    "model_info": model_info,
                    "tracking_uri": self.tracking_uri,
                }

                return Result(
                    status=Status.SUCCESS,
                    data=output,
                    message=f"MLflow tracking completed. Run ID: {self.run_id}",
                )
            finally:
                self.end_run()

        except Exception as e:
            logger.error(f"MLflow tracking failed: {e}")
            return Result(
                status=Status.FAILED, message=f"MLflow tracking failed: {str(e)}"
            )

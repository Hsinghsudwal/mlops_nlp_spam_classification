import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from utils.logger import logger
from utils.audit_logger import audit_log
from utils.config_manager import ConfigManager
from utils.artifact_manager import ArtifactManager
from utils.results import Result, Status


class ModelTrainer:
    """Model training with full sklearn Pipeline."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
        self.models = {
            "naive_bayes": MultinomialNB(),
            "logistic_regression": LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
            ),
            # "random_forest": RandomForestClassifier(
            #     random_state=self.random_state, n_estimators=100, n_jobs=-1
            # ),
            # "svm": SVC(
            #     kernel="linear", probability=True, random_state=self.random_state, C=1.0
            # ),
        }

    def load_transformation_data(self, results):
        """Load transformed data."""
        try:
            transformation = results.get("transformer")
            if not transformation or transformation.status != Status.SUCCESS:
                raise ValueError("Transformation results missing or failed")

            transformed_data = transformation.data
            return transformed_data
        except Exception as e:
            logger.error(f"Error loading trained models: {e}")
            raise

    def evaluate_model(self, model, X_train, y_train, cv_folds: int = 5):
        """Evaluate model with cross-validation (fallback to train score)."""
        try:
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=cv_folds, scoring="accuracy", n_jobs=-1
            )
            return cv_scores.mean()
        except Exception as e:
            logger.warning(f"Cross-validation failed, using train score: {e}")
            return model.score(X_train, y_train)

    def model_train(self, model_name: str, X_train, y_train, cv_folds: int):
        """Train and evaluate a single model with unified pipeline."""
        if model_name not in self.models:
            return {
                "model": None,
                "model_name": model_name,
                "train_score": 0.0,
                "cv_score": 0.0,
                "error": f"Unknown model: {model_name}",
                "status": "failed",
            }

        try:
            model = self.models[model_name]

            logger.info(f"Training {model_name} on transformer data...")

            # Fit / evaluate
            model.fit(X_train, y_train)

            cv_score = self.evaluate_model(model, X_train, y_train, cv_folds)
            train_score = model.score(X_train, y_train)

            logger.info(
                f"{model_name} - Train Score: {train_score:.4f}, CV Score: {cv_score:.4f}"
            )

            return {
                "model": model,
                "model_name": model_name,
                "train_score": train_score,
                "cv_score": cv_score,
                "status": "success",
            }
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            return {
                "pipeline": None,
                "model_name": model_name,
                "train_score": 0.0,
                "cv_score": 0.0,
                "error": str(e),
                "status": "failed",
            }

    def model_trainer(
        self,
        results: Dict[str, Any],
        config: ConfigManager,
        artifact_store: ArtifactManager,
        pipeline_id: str,
    ) -> Result:
        try:
            # return_decision = False
            model_dir = config.get("dir_path.models", "models")
            algorithms = config.get(
                "algorithms", ["naive_bayes", "logistic_regression", "random_forest"]
            )
            cv_folds = config.get("base.cv_folds", 5)

            transformer_output = self.load_transformation_data(results)
            X_train = transformer_output["X_train"]
            X_test = transformer_output["X_test"]
            y_train = transformer_output["y_train"]
            y_test = transformer_output["y_test"]
            wrapper_pipeline = transformer_output["transformer_pipeline"]

            logger.info(f"Training on data shape: {X_train.shape}")
            logger.info(f"Training {len(algorithms)} algorithms: {algorithms}")

            # Training loop
            trained_models = {}
            best_model = None
            best_model_name = None
            best_score = -1.0

            for model_name in algorithms:
                result = self.model_train(model_name, X_train, y_train, cv_folds)
                trained_models[model_name] = result

                if (
                    result["status"] == "success"
                    and result["model"]
                    and result["cv_score"] > best_score
                ):
                    best_model = result["model"]
                    best_model_name = model_name
                    best_score = result["cv_score"]

            if not best_model:
                raise ValueError("No models were successfully trained")

            # Test evaluation
            test_score = best_model.score(X_test, y_test)

            logger.info(
                f"Best model: {best_model_name} (CV Score: {best_score:.4f}, Test Score: {test_score:.4f})"
            )

            output = {
                "trained_models_metadata": trained_models,
                "best_model": best_model,
                "best_model_name": best_model_name,
                "best_score": best_score,
                "test_score": test_score,
                "X_test": X_test,
                "y_test": y_test,
                "wrapper_pipeline": wrapper_pipeline,
            }

            audit_log("Model_trainer output", details=output)

            artifact_store.save(output, model_dir, "trained_output.pkl", pipeline_id)

            return Result(
                status=Status.SUCCESS,
                data=output,  # "decision": decision if return_decision else None}
                message=(
                    f"Trained {len([m for m in trained_models.values() if m['model']])} models. "
                    f"Best: {best_model_name} (CV: {best_score:.4f}, Test: {test_score:.4f})"
                ),
            )

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return Result(
                status=Status.FAILED, message=f"Model training failed: {str(e)}"
            )

    def run(self, **kwargs) -> Result:
        return self.model_trainer(
            results=kwargs.get("results", {}),
            config=kwargs.get("config"),
            artifact_store=kwargs.get("artifact_store"),
            pipeline_id=kwargs.get("pipeline_id", "default"),
        )

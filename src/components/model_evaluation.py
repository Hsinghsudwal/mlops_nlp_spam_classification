import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.pipeline import Pipeline

from utils.logger import logger
from utils.config_manager import ConfigManager
from utils.artifact_manager import ArtifactManager
from utils.results import Result, Status
from src.mlops.governance import Governance
from utils.audit_logger import audit_log


class ModelEvaluation:
    """Evaluate trained models on test data."""

    def __init__(self, governance: Governance = None):
        self.governance = governance or Governance()

    def load_trained_models(self, results: Dict[str, Any]):
        """Extract trained model output from previous pipeline step."""
        try:
            trainer = results.get("trainer")
            if not trainer or trainer.status != Status.SUCCESS:
                raise ValueError("Trainer results missing or failed")

            trained_data = trainer.data

            return trained_data
        except Exception as e:
            logger.error(f"Error loading trained models: {e}")
            raise

    def compute_metrics(self, model, X_test, y_test, label_encoder) -> Dict[str, Any]:
        """Compute evaluation metrics."""

        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=label_encoder.classes_,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        # ROC-AUC
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
                if len(label_encoder.classes_) == 2:
                    metrics["roc_auc"] = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    metrics["roc_auc"] = roc_auc_score(
                        y_test, y_prob, multi_class="ovr"
                    )
            except Exception as e:
                logger.warning(f"ROC-AUC calculation failed: {e}")

        return metrics

    def model_evaluation(
        self,
        results: Dict[str, Any],
        config: ConfigManager,
        artifact_store: ArtifactManager,
        pipeline_id: str,
    ) -> Result:
        try:
            eval_dir = config.get("dir_path.evaluation", "evaluation")

            # Load trainer
            trained_output = self.load_trained_models(results)
            best_model = trained_output["best_model"]
            best_model_name = trained_output["best_model_name"]
            best_score = trained_output["best_score"]
            test_score = trained_output["test_score"]
            X_test = trained_output["X_test"]
            y_test = trained_output["y_test"]
            wrapper_pipeline = trained_output["wrapper_pipeline"]

            label_encoder = wrapper_pipeline.label_encoder

            logger.info(f"Evaluating best model: {best_model_name}")
            logger.info(f"Label classes: {label_encoder.classes_}")

            # Build ready pipeline
            full_pipeline = Pipeline(
                [
                    ("preprocessor", wrapper_pipeline),
                    ("classifier", best_model),
                ]
            )
            artifact_store.save(
                full_pipeline, eval_dir, "full_pipeline.pkl", pipeline_id
            )

            metrics = self.compute_metrics(best_model, X_test, y_test, label_encoder)

            # Save metrics
            artifact_store.save(
                metrics, eval_dir, "evaluation_metrics.json", pipeline_id
            )

            # Plot confusion matrix
            cm = np.array(metrics["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix - {best_model_name}")

            # Save
            artifact_store.save(fig, eval_dir, "confusion_matrix.png", pipeline_id)
            plt.close(fig)

            # cr= classification_report(
            #     y_test,
            #     y_pred,
            #     target_names=label_encoder.classes_,
            #     # output_dict=True,
            #     zero_division=0,
            # )

            # artifact_store.save(cr, eval_dir, "classification_report", pipeline_id)

            # complete_pipeline = {

            #     # "full_pipeline": full_pipeline, # inference raw
            # }

            output = {
                "metrics": metrics,
                "wrapper_pipeline": wrapper_pipeline,
                "model": best_model,  # Trained model
                "model_name": best_model_name,
                "label_encoder": label_encoder,
                "classes": label_encoder.classes_.tolist(),
                "full_pipeline": full_pipeline,
            }

            metadata = {
                "pipeline_id": pipeline_id,
                "best_model": best_model_name,
                "cv_score": best_score,
                "test_score": test_score,
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics.get("roc_auc", "N/A"),
                "artifacts": {
                    "data": "raw/train.csv raw/test.csv",
                    "transformer": "transformed/transformed.pkl",
                    "trained_models": "models/trained_output.pkl",
                    "metrics_json": f"{eval_dir}/evaluation_metrics.json",
                    "complete_pipeline": f"{eval_dir}/complete_pipeline.pkl",
                    "confusion_matrix": f"{eval_dir}/confusion_matrix.png",
                },
            }

            # govern = Governance(config)
            decision = self.governance.policy_evaluate(metrics)
            if not decision["approved"]:
                audit_log("Governance blocked", details=decision)
                logger.warning(f"Governance blocked: {decision}")

            artifact_store.save(metadata, eval_dir, "eval_metadata.json", pipeline_id)

            artifact_store.save(output, eval_dir, "output.pkl", pipeline_id)

            return Result(
                status=Status.SUCCESS,
                data=output,
                message=f"Evaluation completed for {best_model_name}. Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics.get('roc_auc', 'N/A')}",
                # message=f"Evaluation completed for {best_model_name}. Accuracy: {metrics['accuracy']:.4f}",
            )

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return Result(
                status=Status.FAILED, message=f"Model evaluation failed: {str(e)}"
            )

    def run(self, **kwargs) -> Result:
        return self.model_evaluation(
            results=kwargs.get("results", {}),
            config=kwargs.get("config"),
            artifact_store=kwargs.get("artifact_store"),
            pipeline_id=kwargs.get("pipeline_id", "default"),
        )

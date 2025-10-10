import os
from datetime import datetime
from typing import Dict, Any, Optional

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from utils.logger import logger
from utils.audit_logger import audit_log
from utils.results import Result, Status


class ExperimentManager:
    """MLflow model staging, validation, and production management."""

    def __init__(self, config):
        self.config = config
        self.client = MlflowClient()
        self.author = self.config.get("project.author", "unknown")
        self.staging = self.config.get("mlflow_config.staging", "staging")
        self.production = self.config.get("mlflow_config.production", "production")

    def get_latest_registered_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get the latest version of a registered model."""
        try:
            model_versions = self.client.search_model_versions(f"name='{model_name}'")
            if not model_versions:
                logger.warning(f"No model versions found for {model_name}")
                return None
            latest_version = max(int(mv.version) for mv in model_versions)
            latest_model = self.client.get_model_version(
                model_name, str(latest_version)
            )
            return {
                "name": latest_model.name,
                "version": latest_model.version,
                "status": latest_model.status,
                "run_id": latest_model.run_id,
                "creation_timestamp": latest_model.creation_timestamp,
                "last_updated_timestamp": latest_model.last_updated_timestamp,
                "description": latest_model.description,
            }
        except Exception as e:
            logger.error(f"Error fetching latest model: {e}")
            return None

    def get_last_production_version(self, model_name: str) -> Optional[int]:
        """Find the most recent production version, even if archived."""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            prod_versions = [
                int(v.version)
                for v in versions
                if "aliases" in v and self.production in v.aliases
            ]
            if prod_versions:
                return max(prod_versions)
            return None
        except Exception as e:
            logger.error(f"Error fetching last production version: {e}")
            return None

    def stage_model(self, model_name: str, version: str) -> Result:
        """Stage a model by setting the staging alias."""
        try:
            self.client.set_registered_model_alias(
                name=model_name, alias=self.staging, version=version
            )
            now = datetime.utcnow().isoformat()
            self.client.set_model_version_tag(
                model_name, version, "staged_by", self.author
            )
            self.client.set_model_version_tag(model_name, version, "staged_at", now)
            audit_log(
                "StageModel",
                {
                    "model": model_name,
                    "version": version,
                    "alias": self.staging,
                    "staged_by": self.author,
                    "staged_at": now,
                },
            )
            return Result(
                status=Status.SUCCESS,
                data={
                    "model_name": model_name,
                    "version": version,
                    "alias": self.staging,
                },
                message=f"Model {model_name} v{version} staged successfully",
            )
        except Exception as e:
            logger.error(f"Error staging model: {e}")
            return Result(
                status=Status.FAILED, message=f"Error staging model: {str(e)}"
            )

    def validate_model_quality(self, model_name: str, version: str) -> Result:
        """Validate model against configured quality thresholds."""
        try:
            quality_thresholds = self.config.get(
                "mlflow_config.production_quality_thresholds", {}
            )
            model_version = self.client.get_model_version(model_name, version)
            if not model_version:
                return Result(
                    status=Status.FAILED, message=f"Model version {version} not found"
                )
            run = self.client.get_run(model_version.run_id)
            metrics = run.data.metrics

            validation_results = {}
            passed = True
            for metric, threshold in quality_thresholds.items():
                val = metrics.get(metric, 0.0)
                ok = val >= threshold
                validation_results[metric] = {
                    "value": val,
                    "threshold": threshold,
                    "passed": ok,
                }
                if not ok:
                    passed = False

            audit_log(
                "ValidateModel",
                {
                    "model": model_name,
                    "version": version,
                    "results": validation_results,
                    "passed": passed,
                },
            )

            return Result(
                status=Status.SUCCESS if passed else Status.FAILED,
                data={
                    "model_name": model_name,
                    "version": version,
                    "validation_results": validation_results,
                    "passed": passed,
                    "thresholds_used": quality_thresholds,
                },
                message=f"Quality validation {'passed' if passed else 'failed'}",
            )
        except Exception as e:
            logger.error(f"Error validating model quality: {e}")
            return Result(
                status=Status.FAILED, message=f"Error validating model: {str(e)}"
            )

    def safe_promote_to_production(self, model_name: str) -> Result:
        """Promote staged model to production with validation and audit logs."""
        prev_prod_version = None
        try:
            staged = self.client.get_model_version_by_alias(model_name, self.staging)
            if not staged:
                return Result.failure(f"No staged model found for {model_name}")

            staged_version = int(staged.version)

            try:
                prod = self.client.get_model_version_by_alias(
                    model_name, self.production
                )
                prev_prod_version = int(prod.version)
            except Exception:
                prev_prod_version = self.get_last_production_version(model_name)

            validation = self.validate_model_quality(model_name, staged_version)
            if validation.status != Status.SUCCESS:
                audit_log(
                    "PromotionValidationFailed",
                    {
                        "model": model_name,
                        "version": staged_version,
                        "results": validation.data,
                    },
                )
                return Result.failure(
                    f"Validation failed for staged model v{staged_version}. "
                    f"Production remains at v{prev_prod_version or 'none'}"
                )

            self.client.set_registered_model_alias(
                model_name, self.production, staged_version
            )
            promoted_at = datetime.utcnow().isoformat()
            self.client.set_model_version_tag(
                model_name, staged_version, "production_timestamp", promoted_at
            )
            self.client.set_model_version_tag(
                model_name, staged_version, "promoted_from", self.staging
            )
            if prev_prod_version:
                self.client.set_model_version_tag(
                    model_name,
                    staged_version,
                    "previous_prod_version",
                    prev_prod_version,
                )

            audit_log(
                "PromotionSuccess",
                {
                    "model": model_name,
                    "new_version": staged_version,
                    "previous_version": prev_prod_version,
                    "validated": True,
                    "staged_by": self.author,
                    "promoted_at": promoted_at,
                },
            )

            return Result(
                status=Status.SUCCESS,
                message=f"Promoted {model_name} v{staged_version} to Production",
                data={
                    "model_name": model_name,
                    "version": staged_version,
                    "alias": self.production,
                    "previous_prod_version": prev_prod_version,
                    "validation_results": validation.data,
                },
            )

        except Exception as e:
            logger.error(f"Promotion failed: {e}")
            return Result(status=Status.FAILED, message=f"Promotion failed: {str(e)}")
